"""Build every table and figure in Results/ from the test outputs.

Reads   Results/metrics.csv, Results/predictions/*.csv, Results/training_logs/*.json,
        <work>/embeddings/*.npz
Writes  Results/accuracy_table.md|csv        all variants x protocols, mean +- std over 3 seeds
        Results/per_class_accuracy.md        which emotions are easy / hard
        Results/fusion_vs_speech.md          where fusion fixes or breaks speech-only decisions
        Results/separability.md              silhouette scores of the learned representations
        Results/error_analysis.md            failure cases
        Results/plots/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models.common.data import EMOTIONS, RESULTS, TESS_DIR, WORK_DIR, load_metadata  # noqa: E402

PLOTS = RESULTS / "plots"
EMB = WORK_DIR / "embeddings"
FEATS = WORK_DIR / "features"

MAIN = {"Speech only": "speech_wavlm", "Text only": "text_bert", "Speech + text (GMU fusion)": "fusion_gmu_wavlm"}
VARIANT_NAMES = {
    "speech_wavlm": "Speech: WavLM + BiGRU",
    "speech_mel": "Speech: log-mel + BiGRU",
    "text_bert": "Text: BERT + BiGRU",
    "fusion_gmu_wavlm": "Fusion: GMU (WavLM + BERT)",
    "fusion_concat_wavlm": "Fusion: concat (WavLM + BERT)",
    "fusion_gmu_mel": "Fusion: GMU (log-mel + BERT)",
    "speech_wavlm_spknorm": "Speech: WavLM + speaker norm.",
    "fusion_gmu_wavlm_spknorm": "Fusion: GMU + speaker norm.",
    "fusion_gmu_wavlm_notext": "Control: GMU, text = zeros",
    "fusion_gmu_wavlm_shuftext": "Control: GMU, wrong word's text",
}
# short two-line names for chart axes
CHART_NAMES = {
    "speech_wavlm": "Speech\nWavLM", "speech_mel": "Speech\nlog-mel", "text_bert": "Text\nBERT",
    "fusion_gmu_wavlm": "Fusion GMU\nWavLM+BERT", "fusion_concat_wavlm": "Fusion concat\nWavLM+BERT",
    "fusion_gmu_mel": "Fusion GMU\nlog-mel+BERT", "speech_wavlm_spknorm": "Speech WavLM\n+ spk. norm",
    "fusion_gmu_wavlm_spknorm": "Fusion GMU\n+ spk. norm", "fusion_gmu_wavlm_notext": "Control GMU\ntext = zeros",
    "fusion_gmu_wavlm_shuftext": "Control GMU\nwrong word",
}
PROTOCOLS = ["word", "oaf2yaf", "yaf2oaf"]
PROTO_NAMES = {"word": "Word-disjoint (main)", "oaf2yaf": "Cross-speaker OAF->YAF",
               "yaf2oaf": "Cross-speaker YAF->OAF"}
SHORT = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
         "neutral": "neutral", "pleasant_surprise": "pl. surprise", "sad": "sad"}

# Validated categorical palette (fixed order = EMOTIONS order) and neutral inks
EMO_COLOURS = dict(zip(EMOTIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
PROTO_COLOURS = dict(zip(PROTOCOLS, ["#2a78d6", "#eb6834", "#1baf7a"]))
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#f9f9f7", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def pm(mean: float, std: float) -> str:
    return f"{100 * mean:.1f} ± {100 * std:.1f}"


MAIN_SEEDS = (0, 1, 2)       # main tables; the seed study (seed_study.md) uses every seed available


def load_preds(variant: str, protocol: str, seeds=MAIN_SEEDS) -> pd.DataFrame:
    frames = []
    for f in sorted((RESULTS / "predictions").glob(f"{variant}_{protocol}_seed*.csv")):
        seed = int(f.stem.rsplit("seed", 1)[1])
        if seeds is not None and seed not in seeds:
            continue
        df = pd.read_csv(f)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  plots/{name}")


# ------------------------------------------------------------------ accuracy tables
def accuracy_tables(m: pd.DataFrame) -> str:
    g = m.groupby(["variant", "protocol"]).agg(acc=("test_acc", "mean"), acc_sd=("test_acc", "std"),
                                               f1=("test_macro_f1", "mean"), f1_sd=("test_macro_f1", "std"),
                                               seeds=("seed", "count")).reset_index()
    g.to_csv(RESULTS / "accuracy_table.csv", index=False)
    rows = []
    for v in VARIANT_NAMES:
        if v not in set(g["variant"]):
            continue
        row = {"Model": VARIANT_NAMES[v]}
        for p in PROTOCOLS:
            s = g[(g.variant == v) & (g.protocol == p)]
            row[PROTO_NAMES[p]] = pm(s.acc.iloc[0], s.acc_sd.iloc[0]) if len(s) else "-"
        cross = m[(m.variant == v) & m.protocol.isin(["oaf2yaf", "yaf2oaf"])].test_acc
        row["Cross-speaker mean"] = f"{100 * cross.mean():.1f}" if len(cross) else "-"
        rows.append(row)
    acc = pd.DataFrame(rows)
    rows_f1 = []
    for v in VARIANT_NAMES:
        if v not in set(g["variant"]):
            continue
        row = {"Model": VARIANT_NAMES[v]}
        for p in PROTOCOLS:
            s = g[(g.variant == v) & (g.protocol == p)]
            row[PROTO_NAMES[p]] = pm(s.f1.iloc[0], s.f1_sd.iloc[0]) if len(s) else "-"
        rows_f1.append(row)
    text = ("# Test accuracy (%) - mean ± std over 3 seeds\n\nChance = 14.3 % (7 balanced classes). "
            "Test sets: 420 clips (word-disjoint) / 1400 clips of the unseen speaker (cross-speaker).\n\n"
            + md_table(acc) + "\n# Macro-F1 (%)\n\n" + md_table(pd.DataFrame(rows_f1)))
    (RESULTS / "accuracy_table.md").write_text(text, encoding="utf-8")

    # bar chart: main three + ablations, grouped by protocol
    variants = [v for v in VARIANT_NAMES if v in set(g["variant"])]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    width = 0.26
    x = np.arange(len(variants))
    for k, p in enumerate(PROTOCOLS):
        vals = [g[(g.variant == v) & (g.protocol == p)].acc.mean() * 100 for v in variants]
        sds = [g[(g.variant == v) & (g.protocol == p)].acc_sd.mean() * 100 for v in variants]
        bars = ax.bar(x + (k - 1) * (width + 0.02), vals, width, yerr=sds, color=PROTO_COLOURS[p],
                      label=PROTO_NAMES[p], error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2})
        for b, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK2)
    ax.axhline(100 / 7, color=MUTED, lw=0.8, ls="--", label="chance (14.3 %)")
    ax.set_xticks(x, [CHART_NAMES[v] for v in variants], fontsize=7.5)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12), fontsize=8)
    save(fig, "accuracy_by_variant.png")
    return text


# ------------------------------------------------------------------ per-class accuracy
def per_class(m: pd.DataFrame) -> str:
    rows = []
    for label, v in MAIN.items():
        for pname, prot in (("word-disjoint", ["word"]), ("cross-speaker", ["oaf2yaf", "yaf2oaf"])):
            s = m[(m.variant == v) & m.protocol.isin(prot)]
            if s.empty:
                continue
            row = {"Model": label, "Protocol": pname}
            row.update({SHORT[e]: f"{100 * s[f'acc_{e}'].mean():.1f}" for e in EMOTIONS})
            rows.append(row)
    df = pd.DataFrame(rows)
    text = ("# Per-emotion test accuracy (recall, %), mean over seeds (and both directions for cross-speaker)\n\n"
            + md_table(df))
    (RESULTS / "per_class_accuracy.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ confusion matrices
def confusions() -> None:
    groups = [("word-disjoint", ["word"]), ("cross-speaker (both directions)", ["oaf2yaf", "yaf2oaf"])]
    fig, axes = plt.subplots(len(groups), len(MAIN), figsize=(13, 8.4))
    for r, (gname, prots) in enumerate(groups):
        for c, (label, v) in enumerate(MAIN.items()):
            ax = axes[r, c]
            df = pd.concat([load_preds(v, p) for p in prots], ignore_index=True)
            if df.empty:
                ax.axis("off")
                continue
            cm = confusion_matrix(df["true"], df["pred"], labels=EMOTIONS, normalize="true") * 100
            ax.imshow(cm, cmap=SEQ, vmin=0, vmax=100)
            for i in range(len(EMOTIONS)):
                for j in range(len(EMOTIONS)):
                    if cm[i, j] >= 0.5:
                        ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                                color="#ffffff" if cm[i, j] > 55 else INK)
            ticks = [SHORT[e] for e in EMOTIONS]
            ax.set_xticks(range(7), ticks, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(7), ticks if c == 0 else [""] * 7, fontsize=7)
            acc = (df["true"] == df["pred"]).mean() * 100
            ax.set_title(f"{label}\n{gname}: {acc:.1f} %", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("true emotion")
            ax.set_xlabel("predicted")
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("Confusion matrices (% of each true emotion), pooled over 3 seeds", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "confusion_matrices.png")


# ------------------------------------------------------------------ representation separability
def separability() -> str:
    """t-SNE of the three learned representations of the fusion model, plus silhouette scores."""
    meta = {Path(r["path"]).stem: r for r in load_metadata()}
    blocks = [("temporal", "Temporal modelling (speech branch)"),
              ("contextual", "Contextual modelling (text branch)"),
              ("fusion", "Fusion block (GMU output)")]
    rows = []
    for v, block_list in (("fusion_gmu_wavlm", blocks), ("speech_wavlm", [blocks[0]]),
                          ("text_bert", [blocks[1]])):
        for p in PROTOCOLS:
            for f in sorted(EMB.glob(f"{v}_{p}_seed*.npz")):
                if int(f.stem.rsplit("seed", 1)[1]) not in MAIN_SEEDS:
                    continue
                z = np.load(f)
                for key, _ in block_list:
                    if f"emb_{key}" not in z:
                        continue
                    x = z[f"emb_{key}"]
                    y = z["y_true"]
                    # constant vectors (text branch) have no defined cosine distance
                    sil = silhouette_score(x, y, metric="cosine") if np.ptp(x) > 1e-6 else np.nan
                    rows.append({"model": v, "block": key, "protocol": p, "seed": f.stem.rsplit("seed", 1)[1],
                                 "silhouette": sil})
    sil = pd.DataFrame(rows)
    table = (sil.groupby(["model", "block", "protocol"]).silhouette.agg(["mean", "std"]).reset_index()
             .assign(silhouette=lambda d: d["mean"].map(lambda x: f"{x:.3f}") + " ± " + d["std"].map(lambda x: f"{x:.3f}"))
             [["model", "block", "protocol", "silhouette"]])
    text = ("# Separability of the learned representations\n\nSilhouette score of the test-set embeddings "
            "with the true emotion as cluster label (cosine distance; 1 = tight, well-separated clusters, "
            "0 = overlapping, < 0 = mixed). Mean ± std over 3 seeds.\n\n" + md_table(table))
    (RESULTS / "separability.md").write_text(text, encoding="utf-8")

    for p in ("word", "oaf2yaf"):
        f = EMB / f"fusion_gmu_wavlm_{p}_seed0.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ids = z["ids"]
        spk = np.array([meta[i]["speaker"] for i in ids])
        y = z["y_true"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (key, title) in zip(axes, blocks):
            x = z[f"emb_{key}"]
            if np.ptp(x) < 1e-6:
                xy = np.zeros((len(x), 2))
            else:
                xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(x)
            s = silhouette_score(x, y, metric="cosine") if np.ptp(x) > 1e-6 else float("nan")
            for k, e in enumerate(EMOTIONS):
                for s_name, marker in (("OAF", "o"), ("YAF", "^")):
                    m = (y == k) & (spk == s_name)
                    if m.any():
                        ax.scatter(xy[m, 0], xy[m, 1], s=16, marker=marker, color=EMO_COLOURS[e],
                                   edgecolors=SURFACE, linewidths=0.4, alpha=0.9)
                if s > 0.05:            # cluster labels only where there are clusters
                    cm = xy[y == k].mean(0)
                    ax.text(cm[0], cm[1], SHORT[e], fontsize=8, fontweight="bold", color=INK, ha="center",
                            va="center", bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.75})
            if not s > 0.05:
                ax.text(0.5, 0.02, "no emotion structure: every clip of a word has the same text,\n"
                        "so its 7 emotions (x 2 speakers) sit on one point", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7.5, color=INK2,
                        bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": GRID})
            ax.set_title(f"{title}\nsilhouette {s:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
        handles = [plt.Line2D([], [], marker="o", ls="", color=EMO_COLOURS[e], label=SHORT[e]) for e in EMOTIONS]
        handles += [plt.Line2D([], [], marker=mk, ls="", color=MUTED, label=f"speaker {s}")
                    for s, mk in (("OAF", "o"), ("YAF", "^"))]
        fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"t-SNE of test-set representations from the GMU fusion model (seed 0) - "
                     f"{PROTO_NAMES[p]} test set", fontsize=10, x=0.01, ha="left")
        fig.tight_layout()
        save(fig, f"tsne_blocks_{p}.png")
    return text


# ------------------------------------------------------------------ learned layer weights
def layer_weights() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, (v, key, title) in zip(axes, (("speech_wavlm", "speech", "WavLM-Base+ layer weights (speech model)"),
                                          ("text_bert", "text", "BERT-base layer weights (text model)"))):
        ws = []
        for f in sorted(EMB.glob(f"{v}_word_seed*.npz")):
            if int(f.stem.rsplit("seed", 1)[1]) in MAIN_SEEDS:
                ws.append(json.loads(str(np.load(f)["layer_weights"]))[key])
        if not ws:
            ax.axis("off")
            continue
        w = np.array(ws)
        ax.bar(range(w.shape[1]), w.mean(0), yerr=w.std(0), color="#2a78d6", width=0.7,
               error_kw={"elinewidth": 0.8, "ecolor": INK2})
        ax.axhline(1 / w.shape[1], color=MUTED, ls="--", lw=0.8)
        ax.text(w.shape[1] - 0.5, 1 / w.shape[1], "uniform", fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.set_xticks(range(w.shape[1]), ["emb"] + [str(i) for i in range(1, w.shape[1])], fontsize=7)
        ax.set_xlabel("layer (emb = input to the first Transformer layer)")
        ax.set_ylabel("softmax weight")
        ax.set_title(title + "  (word-disjoint, mean ± std, 3 seeds)", fontsize=9)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "layer_weights.png")


# ------------------------------------------------------------------ fusion vs speech-only
def fusion_vs_speech() -> str:
    pairs = [("fusion_gmu_wavlm", "speech_wavlm"), ("fusion_concat_wavlm", "speech_wavlm"),
             ("fusion_gmu_mel", "speech_mel"), ("fusion_gmu_wavlm_spknorm", "speech_wavlm_spknorm")]
    rows, emo_rows = [], []
    for fv, sv in pairs:
        for p in PROTOCOLS:
            f, s = load_preds(fv, p), load_preds(sv, p)
            if f.empty or s.empty:
                continue
            j = f.merge(s, on=["id", "seed"], suffixes=("_f", "_s"))
            fix = ((j.pred_s != j.true_s) & (j.pred_f == j.true_f))
            brk = ((j.pred_s == j.true_s) & (j.pred_f != j.true_f))
            n_seeds = j.seed.nunique()
            rows.append({"fusion": VARIANT_NAMES[fv], "vs speech-only": VARIANT_NAMES[sv], "protocol": PROTO_NAMES[p],
                         "speech acc": f"{100 * (j.pred_s == j.true_s).mean():.1f}",
                         "fusion acc": f"{100 * (j.pred_f == j.true_f).mean():.1f}",
                         "fixed / seed": f"{fix.sum() / n_seeds:.1f}", "broken / seed": f"{brk.sum() / n_seeds:.1f}"})
            if fv == "fusion_gmu_wavlm" and p != "word":
                for e in EMOTIONS:
                    m = j.true_f == e
                    emo_rows.append({"protocol": PROTO_NAMES[p], "emotion": SHORT[e],
                                     "fixed": int(fix[m].sum()), "broken": int(brk[m].sum())})
    text = ("# Where does fusion change the speech-only decision?\n\nPaired comparison on the same test clips and "
            "seeds. *fixed* = speech-only wrong, fusion right; *broken* = speech-only right, fusion wrong.\n\n"
            + md_table(pd.DataFrame(rows)))
    if emo_rows:
        e = pd.DataFrame(emo_rows).groupby("emotion")[["fixed", "broken"]].sum().reset_index()
        text += "\n## GMU fusion vs WavLM speech-only, cross-speaker, by emotion (summed over seeds and directions)\n\n" + md_table(e)

    # gate statistics
    g = []
    for p in PROTOCOLS:
        d = load_preds("fusion_gmu_wavlm", p)
        if not d.empty and "speech_gate" in d:
            g.append(d.assign(protocol=p))
    if g:
        gd = pd.concat(g)
        gt = gd.groupby("protocol").speech_gate.agg(["mean", "std", "min", "max"]).round(3).reset_index()
        text += ("\n## GMU gate: mean of z over the 256 dimensions (1 = all speech, 0 = all text)\n\n" + md_table(gt))
        fig, ax = plt.subplots(figsize=(7, 3))
        for k, p in enumerate(PROTOCOLS):
            vals = gd[gd.protocol == p].speech_gate.values
            if len(vals):
                ax.hist(vals, bins=40, alpha=0.75, color=PROTO_COLOURS[p], label=PROTO_NAMES[p])
        ax.set_xlabel("speech gate z (mean over dimensions), per test clip")
        ax.set_ylabel("clips")
        ax.legend(fontsize=8)
        ax.set_title("How much the GMU relies on speech vs text", fontsize=9)
        save(fig, "gmu_gate.png")
    (RESULTS / "fusion_vs_speech.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ training curves
def training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    colours = dict(zip(MAIN.values(), ["#2a78d6", "#eb6834", "#1baf7a"]))
    for label, v in MAIN.items():
        f = RESULTS / "training_logs" / f"{v}_oaf2yaf_seed0.json"
        if not f.exists():
            continue
        h = pd.DataFrame(json.loads(f.read_text())["history"])
        axes[0].plot(h.epoch, h.train_loss, color=colours[v], lw=2, label=label)
        axes[1].plot(h.epoch, 100 * h.val_acc, color=colours[v], lw=2, label=label)
    axes[0].set_title("Training loss (cross-speaker OAF->YAF, seed 0)", fontsize=9)
    axes[1].set_title("Validation accuracy (held-out words of the training speaker)", fontsize=9)
    axes[0].set_ylabel("cross-entropy")
    axes[1].set_ylabel("%")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "training_curves.png")


# ------------------------------------------------------------------ error analysis
def error_analysis() -> str:
    import librosa
    import soundfile as sf

    meta = {Path(r["path"]).stem: r for r in load_metadata()}
    v = "fusion_gmu_wavlm"
    cases = []
    for p in PROTOCOLS:
        d = load_preds(v, p)
        s = load_preds("speech_wavlm", p)
        if d.empty:
            continue
        err = d[d.true != d.pred]
        # how many seeds make the same mistake on a clip -> consistent errors are the interesting ones
        agg = (err.groupby(["id", "true", "pred"]).agg(seeds=("seed", "count"), confidence=("confidence", "mean"))
               .reset_index().sort_values(["seeds", "confidence"], ascending=False))
        for _, r in agg.head(3 if p == "word" else 2).iterrows():
            sp = s[(s.id == r.id)].pred.mode()
            cases.append({"protocol": PROTO_NAMES[p], "clip": r.id, "speaker": meta[r.id]["speaker"],
                          "word": meta[r.id]["word"], "true": r.true, "predicted": r.pred,
                          "seeds wrong": f"{r.seeds}/3", "mean confidence": f"{r.confidence:.2f}",
                          "speech-only says": sp.iloc[0] if len(sp) else "-"})
    cases = cases[:5]
    if not cases:
        return ""
    df = pd.DataFrame(cases)
    text = ("# Error analysis: failure cases of the GMU fusion model\n\nMost consistent errors (same mistake "
            "across seeds, then highest confidence), from each protocol.\n\n" + md_table(df))

    # Figure: each failure clip next to the same speaker saying the same word in the predicted emotion.
    # TESS makes this a controlled comparison: only the emotion differs.
    fig, axes = plt.subplots(len(cases), 2, figsize=(10, 2.1 * len(cases)), squeeze=False)
    for i, c in enumerate(cases):
        for j, emo in enumerate((c["true"], c["predicted"])):
            ax = axes[i, j]
            cand = [k for k, r in meta.items() if r["speaker"] == c["speaker"] and r["word"] == c["word"]
                    and r["emotion"] == emo]
            if not cand:
                ax.axis("off")
                continue
            path = TESS_DIR / meta[cand[0]]["path"]
            y, sr = sf.read(path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            y, _ = librosa.effects.trim(y, top_db=30)
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=16000, n_mels=80, hop_length=160), ref=np.max)
            ax.imshow(mel, origin="lower", aspect="auto", cmap="Greys", vmin=-70, vmax=0,
                      extent=[0, len(y) / 16000, 0, 80])
            f0, _, _ = librosa.pyin(y, fmin=80, fmax=500, sr=16000, hop_length=160)
            t = np.arange(len(f0)) * 0.01
            ax2 = ax.twinx()
            ax2.plot(t, f0, color="#eb6834", lw=2)
            ax2.set_ylim(80, 500)
            ax2.tick_params(labelsize=6, colors=MUTED)
            ax2.spines["right"].set_visible(True)
            if j == 1:
                ax2.set_ylabel("F0 (Hz)", fontsize=7, color=INK2)
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
            tag = "misclassified clip" if j == 0 else "same speaker & word, predicted emotion"
            ax.set_title(f"{meta[cand[0]]['path'].split('/')[-1]}  ({tag})", fontsize=7.5, loc="left")
        axes[i, 0].set_ylabel(f"case {i + 1}\nmel bands", fontsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Failure cases: log-mel spectrogram (grey) and pitch contour (orange). Left: the clip the model got "
                 "wrong (true emotion). Right: the same speaker saying the same word in the emotion it predicted.",
                 fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "failure_cases.png")
    (RESULTS / "error_analysis.md").write_text(text, encoding="utf-8")
    return text


# ------------------------------------------------------------------ seed study
SEED_STUDY = [("speech_wavlm", None), ("fusion_gmu_wavlm", "speech_wavlm"), ("fusion_concat_wavlm", "speech_wavlm"),
              ("speech_wavlm_spknorm", "speech_wavlm"), ("fusion_gmu_wavlm_spknorm", "speech_wavlm_spknorm")]


def seed_study(m_all: pd.DataFrame) -> str:
    """Cross-speaker results vary a lot between seeds, because model selection can only use the
    training speaker (whose validation accuracy is ~100 % almost at once). With every seed
    available, test whether a variant really differs from its speech-only counterpart
    (Welch's t-test over seeds)."""
    from scipy.stats import ttest_ind

    rows, pts = [], []
    for v, ref in SEED_STUDY:
        for p in ("oaf2yaf", "yaf2oaf"):
            a = m_all[(m_all.variant == v) & (m_all.protocol == p)].test_acc.values
            if len(a) == 0:
                continue
            row = {"model": VARIANT_NAMES[v], "protocol": PROTO_NAMES[p], "seeds": len(a),
                   "accuracy": f"{100 * a.mean():.1f} ± {100 * a.std(ddof=1):.1f}",
                   "min-max": f"{100 * a.min():.0f}-{100 * a.max():.0f}"}
            if ref:
                b = m_all[(m_all.variant == ref) & (m_all.protocol == p)].test_acc.values
                t = ttest_ind(a, b, equal_var=False)
                row["vs"] = VARIANT_NAMES[ref]
                row["difference"] = f"{100 * (a.mean() - b.mean()):+.1f}"
                row["p (Welch)"] = f"{t.pvalue:.3f}"
            else:
                row["vs"], row["difference"], row["p (Welch)"] = "-", "-", "-"
            rows.append(row)
            pts += [{"variant": v, "protocol": p, "acc": x} for x in a]
    if not rows:
        return ""
    text = ("# Seed study: cross-speaker accuracy over all seeds\n\nMean ± sample std over seeds; "
            "*difference* and Welch's t-test against the speech-only model with the same features.\n\n"
            + md_table(pd.DataFrame(rows)))
    (RESULTS / "seed_study.md").write_text(text, encoding="utf-8")

    d = pd.DataFrame(pts)
    variants = [v for v, _ in SEED_STUDY if v in set(d.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, p in zip(axes, ("oaf2yaf", "yaf2oaf")):
        for i, v in enumerate(variants):
            vals = d[(d.variant == v) & (d.protocol == p)].acc.values * 100
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), vals, s=22, color=PROTO_COLOURS[p], edgecolors=SURFACE, linewidths=0.5)
            ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=INK, lw=2)
            ax.text(i + 0.28, vals.mean(), f"{vals.mean():.1f}", va="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(variants)), [CHART_NAMES[v] for v in variants], fontsize=7)
        ax.set_title(PROTO_NAMES[p] + " (dots = seeds, bar = mean)", fontsize=9)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test accuracy (%)")
    fig.tight_layout()
    save(fig, "seed_study.png")
    return text


def main() -> None:
    m_all = pd.read_csv(RESULTS / "metrics.csv")
    m = m_all[m_all.seed.isin(MAIN_SEEDS)]
    parts = [accuracy_tables(m), per_class(m), seed_study(m_all)]
    confusions()
    parts.append(separability())
    layer_weights()
    parts.append(fusion_vs_speech())
    training_curves()
    parts.append(error_analysis())
    (RESULTS / "summary.md").write_text("\n\n".join(parts), encoding="utf-8")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
