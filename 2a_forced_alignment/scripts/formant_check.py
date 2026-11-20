"""Acoustic cross-check of the ISLE pronunciation check.

The isle_variants run suggests the learner said "bad" with the vowel of
"bed". If so, the two vowels should have about the same first formant
(F1, tongue height): /ae/ in "bad" normally has an F1 well above /E/ in
"bed". This script measures F1/F2 in the middle of every stressed vowel the
aligner found in the ISLE files, Praat-style: resample to 2 x max formant,
pre-emphasis, 25 ms LPC frames, formants = LPC roots with bandwidth < 400 Hz.
It also measures the voicing of the final stop closure in bad/bed/said
(share of frames with F0), since final devoicing is the other suspected error.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from textgrid_utils import phones_in, read_textgrid

VOWELS = {"AE1", "EH1", "EY1", "AY1", "AA1", "OW1", "UW1"}


def formants(seg: np.ndarray, sr: int, max_formant: float) -> tuple[float, float]:
    target = int(2 * max_formant)
    y = librosa.resample(seg, orig_sr=sr, target_sr=target)
    y = librosa.effects.preemphasis(y, coef=0.97)
    win, hop = int(0.025 * target), int(0.005 * target)
    f1s, f2s = [], []
    for start in range(0, max(1, len(y) - win), hop):
        frame = y[start:start + win] * np.hamming(win)
        if np.sum(frame ** 2) < 1e-8:
            continue
        a = librosa.lpc(frame, order=10)
        roots = [r for r in np.roots(a) if np.imag(r) > 0]
        freqs = np.angle(roots) * target / (2 * np.pi)
        bws = -np.log(np.abs(roots)) * target / np.pi
        cand = sorted(f for f, b in zip(freqs, bws) if f > 90 and b < 400)
        if len(cand) >= 2:
            f1s.append(cand[0])
            f2s.append(cand[1])
    return (float(np.median(f1s)) if f1s else np.nan, float(np.median(f2s)) if f2s else np.nan)


def voiced_share(seg: np.ndarray, sr: int) -> float:
    if len(seg) < int(0.04 * sr):
        return np.nan
    f0, voiced, _ = librosa.pyin(seg, fmin=70, fmax=400, sr=sr, frame_length=640, hop_length=80)
    return float(np.mean(voiced))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--run", default="arpa_g2p", help="run whose canonical phone labels give the vowel intervals")
    args = ap.parse_args()

    rows = []
    for tg_path in sorted((args.outputs / args.run / "isle_sess0131").glob("*.TextGrid")):
        utt = tg_path.stem
        audio, sr = sf.read(next(args.corpus.rglob(f"{utt}.wav")), dtype="float32")
        f0, voiced, _ = librosa.pyin(audio, fmin=70, fmax=400, sr=sr)
        median_f0 = float(np.nanmedian(f0))
        max_formant = 5500.0 if median_f0 > 160 else 5000.0       # Praat's usual female/male setting
        tg = read_textgrid(tg_path)
        for w in tg["words"]:
            ph = phones_in(w, tg["phones"])
            for k, p in enumerate(ph):
                if p.label not in VOWELS:
                    continue
                mid = (p.start + p.end) / 2
                half = max(0.02, p.dur / 4)                             # middle 50 % of the vowel
                seg = audio[int((mid - half) * sr):int((mid + half) * sr)]
                f1, f2 = formants(seg, sr, max_formant)
                row = {"utterance": utt[-12:], "word": w.label, "vowel": p.label,
                       "vowel_ms": round(p.dur * 1000), "F1_Hz": round(f1), "F2_Hz": round(f2),
                       "final_stop": "", "final_stop_voiced_share": ""}
                if k + 1 < len(ph) and ph[k + 1].label in {"D", "T"}:
                    stop = ph[k + 1]
                    s = audio[int(stop.start * sr):int(stop.end * sr)]
                    row["final_stop"] = f"{stop.label} ({stop.dur*1000:.0f} ms)"
                    row["final_stop_voiced_share"] = round(voiced_share(s, sr), 2)
                rows.append(row)
        print(f"{utt}: median F0 {median_f0:.0f} Hz -> max formant {max_formant:.0f} Hz")

    out = args.results / "isle_formants.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
