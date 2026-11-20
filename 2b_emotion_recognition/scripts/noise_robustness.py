"""How do the speech-only and fusion models degrade when the speech is noisy?

Fusion is expected to help most when one modality becomes unreliable and the
other still carries information. This script makes the speech channel
unreliable on purpose: white Gaussian noise is added to the test clips
(word-disjoint protocol) at several signal-to-noise ratios, WavLM features are
re-extracted from the noisy audio, and the trained checkpoints (3 seeds each)
are evaluated on them. The text input is unchanged.

TESS text carries no emotion information, so any advantage of a fusion
model under noise cannot come from the text itself. The control model
fusion_gmu_wavlm_notext (same architecture, text branch fed zeros) separates
"text helps" from "the fusion architecture is more robust".

Writes Results/noise_robustness.csv|md and Results/plots/noise_robustness.png.
Checkpoints for seeds 3-9 come from scripts/run_noise_study.py.

    python scripts/noise_robustness.py [--snrs 20 10 5 0] [--protocol word] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from extract_features import SPEECH_MODEL, SR, preprocess  # noqa: E402
from models.common.data import RESULTS, TESS_DIR, load_metadata, split_rows  # noqa: E402
from models.common.features import FeatureStore, clip_id, make_loader  # noqa: E402
from models.common.runner import DEVICE, ckpt_path, evaluate  # noqa: E402

MODELS = {  # variant -> module with build_model
    "speech_wavlm": "models.speech_pipeline.model",
    "fusion_gmu_wavlm": "models.fusion_pipeline.model",
    "fusion_gmu_wavlm_notext": "models.fusion_pipeline.model",   # control: text branch fed zeros
    "fusion_gmu_wavlm_shuftext": "models.fusion_pipeline.model", # control: transcript of another word
    "fusion_concat_wavlm": "models.fusion_pipeline.model",
}
LABELS = {"speech_wavlm": "Speech only (WavLM)", "fusion_gmu_wavlm": "Fusion: GMU",
          "fusion_gmu_wavlm_notext": "GMU, text replaced by zeros",
          "fusion_gmu_wavlm_shuftext": "GMU, transcript of a different word", "fusion_concat_wavlm": "Fusion: concat"}
COLOURS = {"speech_wavlm": "#2a78d6", "fusion_gmu_wavlm": "#1baf7a", "fusion_gmu_wavlm_notext": "#eb6834",
           "fusion_gmu_wavlm_shuftext": "#e87ba4",
           "fusion_concat_wavlm": "#6250d6"}


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_signal = np.mean(x ** 2)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise *= np.sqrt(p_signal / (10 ** (snr_db / 10)) / np.mean(noise ** 2))
    return np.clip(x + noise, -1, 1)


@torch.no_grad()
def wavlm_features(rows, snr, fe, wavlm, seed=0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    out = {}
    for r in rows:
        audio, _ = preprocess(TESS_DIR / r["path"])
        if snr is not None:
            audio = add_noise(audio, snr, rng)
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt")
        hs = wavlm(inputs["input_values"].to(DEVICE), output_hidden_states=True).hidden_states
        out[clip_id(r)] = torch.stack(hs, 0)[:, 0].to(torch.float16).cpu()
    return out


def main() -> None:
    import importlib

    from transformers import AutoFeatureExtractor, WavLMModel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snrs", nargs="+", type=float, default=[20, 10, 5, 0])
    ap.add_argument("--protocol", default="word")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()

    rows = split_rows(load_metadata(), args.protocol, "test")
    fe = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL)
    wavlm = WavLMModel.from_pretrained(SPEECH_MODEL).to(DEVICE).eval()
    # one store per text condition; both get the same (noisy) speech features below
    stores = {t: FeatureStore("wavlm", text_input=t) for t in ("transcript", "zeros", "shuffled")}

    models = {}
    for v, mod in MODELS.items():
        build = importlib.import_module(mod).build_model
        for s in args.seeds:
            path = ckpt_path(v, args.protocol, s)
            if not path.exists():
                print(f"  (no checkpoint {path.name}, skipped)")
                continue
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            m = build(ck["config"]).to(DEVICE)
            m.load_state_dict(ck["state_dict"])
            models[(v, s)] = (m, ck["config"].get("text_input", "transcript"))

    results = []
    for snr in [None, *args.snrs]:
        speech = wavlm_features(rows, snr, fe, wavlm)               # (noisy) speech features
        loaders = {}
        for name, store in stores.items():
            store._speech = speech
            loaders[name] = make_loader(rows, store, 64, shuffle=False)
        for (v, s), (m, text_input) in models.items():
            res = evaluate(m, loaders[text_input])
            acc = float(np.mean(np.array(res["y_true"]) == np.array(res["y_pred"])))
            gate = float(res["gate"].mean()) if res["gate"] is not None else np.nan
            results.append({"snr_db": "clean" if snr is None else int(snr), "variant": v, "seed": s,
                            "acc": acc, "speech_gate": gate})
        snr_txt = "clean" if snr is None else f"{snr:g} dB"
        cur = pd.DataFrame([r for r in results if r["snr_db"] == ("clean" if snr is None else int(snr))])
        print(f"{snr_txt:>6}: " + "  ".join(f"{v} {100 * g.acc.mean():.1f}%" for v, g in cur.groupby("variant")),
              flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "noise_robustness.csv", index=False)
    order = ["clean"] + [int(s) for s in args.snrs]
    tab = (df.groupby(["variant", "snr_db"]).agg(acc=("acc", "mean"), sd=("acc", "std"),
                                                gate=("speech_gate", "mean")).reset_index())
    lines = ["# Accuracy (%) on noisy test speech (word-disjoint test set, white noise, mean ± std over seeds)\n",
             "| Model | " + " | ".join(f"{o} dB" if o != "clean" else "clean" for o in order) + " |",
             "|---|" + "---|" * len(order)]
    for v in MODELS:
        cells = []
        for o in order:
            r = tab[(tab.variant == v) & (tab.snr_db.astype(str) == str(o))]
            cells.append(f"{100 * r.acc.iloc[0]:.1f} ± {100 * r.sd.iloc[0]:.1f}" if len(r) else "-")
        n = df[df.variant == v].seed.nunique()
        lines.append(f"| {LABELS[v]} ({n} seeds) | " + " | ".join(cells) + " |")

    # Welch's t-test of every model against speech-only at each SNR (seeds as samples)
    from scipy.stats import ttest_ind
    lines += ["", "Difference to speech-only in points (Welch's t-test p-value):", "",
              "| Model | " + " | ".join(f"{o} dB" if o != "clean" else "clean" for o in order) + " |",
              "|---|" + "---|" * len(order)]
    for v in MODELS:
        if v == "speech_wavlm":
            continue
        cells = []
        for o in order:
            a = df[(df.variant == v) & (df.snr_db.astype(str) == str(o))].acc.values
            b = df[(df.variant == "speech_wavlm") & (df.snr_db.astype(str) == str(o))].acc.values
            if len(a) > 1 and len(b) > 1:
                cells.append(f"{100 * (a.mean() - b.mean()):+.1f} (p={ttest_ind(a, b, equal_var=False).pvalue:.3f})")
            else:
                cells.append("-")
        lines.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
    g = tab[tab.variant == "fusion_gmu_wavlm"]
    lines += ["", "GMU speech gate (mean z): " + ", ".join(
        f"{o}: {g[g.snr_db.astype(str) == str(o)].gate.iloc[0]:.3f}" for o in order)]
    (RESULTS / "noise_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    xs = np.arange(len(order))
    for v in MODELS:
        sel = [tab[(tab.variant == v) & (tab.snr_db.astype(str) == str(o))] for o in order]
        if any(len(s) == 0 for s in sel):
            continue
        means = np.array([s.acc.iloc[0] * 100 for s in sel])
        sds = np.array([s.sd.iloc[0] * 100 for s in sel])
        ax.fill_between(xs, means - sds, means + sds, color=COLOURS[v], alpha=0.12, lw=0)
        ax.plot(xs, means, marker="o", lw=2, ms=6, color=COLOURS[v], label=LABELS[v])
        ax.text(xs[-1] + 0.08, means[-1], f"{means[-1]:.0f}", va="center", fontsize=7.5, color="#52514e")
    ax.axhline(100 / 7, color="#898781", ls="--", lw=0.8)
    ax.text(0, 100 / 7 + 1.5, "chance", fontsize=7, color="#898781")
    ax.set_xticks(xs, [o if o == "clean" else f"{o} dB" for o in order])
    ax.set_xlabel("signal-to-noise ratio of the test speech")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Robustness to white noise (word-disjoint test set; line = mean, band = ± std over "
                 f"{df.seed.nunique()} seeds)", fontsize=9)
    fig.savefig(RESULTS / "plots" / "noise_robustness.png", dpi=160, bbox_inches="tight")
    print("  plots/noise_robustness.png")


if __name__ == "__main__":
    main()
