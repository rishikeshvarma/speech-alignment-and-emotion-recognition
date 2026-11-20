"""Praat-style pictures of the alignments: waveform, broadband spectrogram,
then one word tier and one phone tier per run, with boundaries drawn
through the spectrogram so they can be checked against the acoustics.

Figures (in <results>/plots/):
  <utt>_<run>.png        one run, whole utterance (ISLE) or first 3.5 s (BU news)
  compare_<name>.png     several runs stacked over the same stretch of audio
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from textgrid_utils import UNKNOWN, read_textgrid, textgrids

MAIN = "arpa_g2p"
COMPARE_RUNS = ["arpa_base", "arpa_g2p", "mfa_g2p", "arpa_g2p_adapted", "isle_variants"]
# (figure name, utterance, word the window is centred on, words either side)
COMPARE_WINDOWS = [
    ("oov_margo_melnicove", "F2BJRLP1", "melnicove", 4),
    ("oov_dukakis", "F2BJRLP2", "dukakis", 3),
    ("oov_maffy", "F2BJRLP2", "maffy", 3),
    ("isle_white_bait", "ISLE_SESS0131_BLOCKD02_01_sprt1", None, 0),
    ("isle_bad_bed", "ISLE_SESS0131_BLOCKD02_03_sprt1", None, 0),
]
RUN_COLOURS = {"arpa_base": "#b23a48", "arpa_g2p": "#1d4e89", "arpa_g2p_finetune": "#2a7f62",
               "mfa_g2p": "#8a5a00", "arpa_g2p_adapted": "#5b3f8c", "isle_variants": "#00707a"}


def spectro(ax, audio, sr, t0, t1):
    seg = audio[int(t0 * sr):int(t1 * sr)]
    # ~6 ms window: broadband, like Praat's default 5 ms Gaussian window, so
    # formant transitions and closures (where boundaries sit) are visible.
    f, t, S = spectrogram(seg, fs=sr, window="hann", nperseg=96, noverlap=88, nfft=512)
    S_db = 10 * np.log10(S + 1e-12)
    vmax = S_db.max()
    ax.pcolormesh(t + t0, f, S_db, cmap="Greys", vmin=vmax - 50, vmax=vmax, shading="auto")
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Hz", fontsize=7)
    ax.tick_params(labelsize=6)


def tier(ax, intervals, t0, t1, colour, name, boundary_axes):
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7, color=colour)
    for iv in intervals:
        if iv.end < t0 or iv.start > t1:
            continue
        for x in (iv.start, iv.end):
            if t0 <= x <= t1:
                ax.axvline(x, color=colour, lw=0.8)
                for bax in boundary_axes:
                    bax.axvline(x, color=colour, lw=0.5, alpha=0.45, ls="--")
        width = (min(iv.end, t1) - max(iv.start, t0)) / (t1 - t0)      # share of the window
        size = 7 if width > 0.025 else 6 if width > 0.015 else 5
        bad = iv.label in UNKNOWN
        ax.text((max(iv.start, t0) + min(iv.end, t1)) / 2, 0.5, iv.label, ha="center", va="center",
                fontsize=size, color="#c00000" if bad else "black", fontweight="bold" if bad else None,
                clip_on=True)


def figure(audio, sr, t0, t1, runs: dict[str, dict], title: str, out: Path) -> None:
    n = len(runs)
    heights = [1.0, 2.0] + [0.45, 0.45] * n
    fig, axes = plt.subplots(len(heights), 1, figsize=(11, 2.6 + 0.75 * n * 2 * 0.45 + 1.4),
                             sharex=True, gridspec_kw={"height_ratios": heights, "hspace": 0.06})
    t = np.arange(len(audio)) / sr
    m = (t >= t0) & (t <= t1)
    axes[0].plot(t[m], audio[m], color="black", lw=0.4)
    axes[0].set_yticks([])
    axes[0].set_title(title, fontsize=9, loc="left")
    spectro(axes[1], audio, sr, t0, t1)
    for k, (run, tg) in enumerate(runs.items()):
        c = RUN_COLOURS.get(run, "black")
        boundary_axes = [axes[0], axes[1]] if k == 0 else []
        label = run if n > 1 else ""
        tier(axes[2 + 2 * k], tg["words"], t0, t1, c, f"{label}\nwords", boundary_axes)
        tier(axes[3 + 2 * k], tg["phones"], t0, t1, c, "phones", [])
    axes[-1].set_xlabel("time (s)", fontsize=8)
    axes[-1].set_xlim(t0, t1)
    axes[-1].tick_params(labelsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def load_audio(corpus: Path, utt: str):
    wav = next(corpus.rglob(f"{utt}.wav"))
    audio, sr = sf.read(wav, dtype="float32")
    return audio, sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    plots = args.results / "plots"

    runs = textgrids(args.outputs)
    tg = {r: {u: read_textgrid(p) for u, p in utts.items()} for r, utts in runs.items()}

    # 1. one picture per utterance for the main run
    for utt in sorted(tg.get(MAIN, {})):
        audio, sr = load_audio(args.corpus, utt)
        t1 = len(audio) / sr if utt.startswith("ISLE") else 3.5
        figure(audio, sr, 0.0, t1, {MAIN: tg[MAIN][utt]},
               f"{utt}  ({MAIN}: english_us_arpa acoustic model + dictionary with G2P words)"
               + ("" if utt.startswith("ISLE") else "  - first 3.5 s"),
               plots / f"{utt}_{MAIN}.png")

    # 2. runs stacked over the same stretch, to show where they disagree
    for name, utt, centre, ctx in COMPARE_WINDOWS:
        have = {r: tg[r][utt] for r in COMPARE_RUNS if r in tg and utt in tg[r]}
        if not have:
            continue
        audio, sr = load_audio(args.corpus, utt)
        if centre is None:
            t0, t1 = 0.0, len(audio) / sr
        else:
            words = tg[MAIN][utt]["words"]
            i = next(k for k, w in enumerate(words) if w.label == centre)
            t0 = max(0.0, words[max(0, i - ctx)].start - 0.15)
            t1 = min(len(audio) / sr, words[min(len(words) - 1, i + ctx)].end + 0.15)
        figure(audio, sr, t0, t1, have, f"{utt}  {t0:.2f}-{t1:.2f} s  (runs compared)",
               plots / f"compare_{name}.png")


if __name__ == "__main__":
    main()
