"""Turn the provided Assignment 2a data into a corpus MFA can read.

MFA wants one folder per speaker, and inside it a .wav and a transcript
(.lab) with the same base name for every utterance:

    corpus/
      f2b/            F2BJRLP1.wav  F2BJRLP1.lab  ...
      isle_sess0131/  ISLE_SESS0131_BLOCKD02_01_sprt1.wav  ....lab  ...

Two things in the provided data have to be fixed first.

1. Audio. The three BU Radio News files (F2BJRLP*.wav) are NIST SPHERE files
   that only carry a .wav name; they have no RIFF header, so MFA cannot read
   them. libsndfile reads SPHERE, so they are re-written as 16-bit PCM WAV.
   The ISLE files are already 16 kHz RIFF WAV and are copied as they are.

2. Text. The BU transcripts are line-wrapped news copy with punctuation,
   dotted abbreviations (S.J.C.'s), spelled-out acronyms (WBUR's) and numbers
   broken across a line ("seventy- six"). They become one line of lower-case
   words that a pronunciation dictionary can look up. The ISLE transcripts are
   already plain upper-case words and only get lower-cased.

Usage:
    python scripts/prepare_corpus.py --src "<folder with wav/ and transcripts/>" --out corpus
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf

# Speaker id for each file prefix. MFA adapts to each speaker separately
# (fMLLR), so the speaker folders matter.
SPEAKERS = {
    "F2B": "f2b",                      # BU Radio News, female announcer F2B
    "ISLE_SESS0131": "isle_sess0131",  # ISLE corpus, non-native learner
}


def speaker_of(stem: str) -> str:
    for prefix, spk in SPEAKERS.items():
        if stem.upper().startswith(prefix):
            return spk
    raise ValueError(f"no speaker rule for {stem}")


def normalise(text: str) -> str:
    """Raw transcript -> one line of lower-case dictionary words."""
    text = " ".join(text.split())                       # join wrapped lines

    # "seventy- six" (a number split across a line) -> "seventy six"
    text = re.sub(r"(\w)-\s+(\w)", r"\1 \2", text)

    # Acronyms are read letter by letter, so split them into letters. This
    # only makes sense for mixed-case text; the ISLE prompts are all
    # upper-case words and must be left alone.
    if text != text.upper():
        # dotted: S.J.C.'s -> s j c's
        text = re.sub(
            r"\b((?:[A-Z]\.){2,})('s)?",
            lambda m: " ".join(m.group(1).replace(".", "")) + (m.group(2) or ""),
            text,
        )
        # undotted capitals: WBUR's -> w b u r's (only 2-5 letter all-caps tokens)
        text = re.sub(
            r"\b([A-Z]{2,5})('s)?\b",
            lambda m: " ".join(m.group(1)) + (m.group(2) or ""),
            text,
        )

    # Hyphenated words that are not in the dictionaries are split so each
    # part can be looked up ("de-politicize" -> "de politicize").
    text = text.replace("-", " ")

    # Drop punctuation but keep word-internal apostrophes (c's, hennessy's).
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return " ".join(text.lower().split())


def convert_audio(src: Path, dst: Path) -> dict:
    """Read WAV or NIST SPHERE, write 16-bit PCM RIFF WAV. Returns basic info."""
    with open(src, "rb") as fh:
        header = fh.read(8)
    fmt = "NIST SPHERE" if header.startswith(b"NIST_1A") else "RIFF WAV"
    audio, sr = sf.read(src, dtype="int16", always_2d=True)
    audio = audio[:, 0]                                 # mono (all inputs are mono)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, audio, sr, subtype="PCM_16")
    peak = np.abs(audio.astype(np.int32)).max() / 32768
    return {"source_format": fmt, "sample_rate": sr,
            "duration_s": round(len(audio) / sr, 2), "peak": round(float(peak), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder holding wav/ and transcripts/")
    ap.add_argument("--out", default=Path("corpus"), type=Path, help="MFA corpus folder to create")
    ap.add_argument("--report", default=Path("results/corpus_summary.csv"), type=Path)
    args = ap.parse_args()

    transcripts = {p.stem.upper(): p for p in (args.src / "transcripts").iterdir()}
    rows = []
    for wav in sorted((args.src / "wav").glob("*.wav")):
        txt = transcripts.get(wav.stem.upper())
        if txt is None:
            print(f"skip {wav.name}: no transcript")
            continue
        spk = speaker_of(wav.stem)
        info = convert_audio(wav, args.out / spk / wav.name)
        raw = txt.read_text(encoding="utf-8", errors="replace")
        norm = normalise(raw)
        # newline="\n": on Windows text mode would add "\r" to the last word
        with open(args.out / spk / f"{wav.stem}.lab", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(norm + "\n")
        rows.append({"utterance": wav.stem, "speaker": spk, **info,
                     "n_words": len(norm.split()), "raw_text": " ".join(raw.split()),
                     "normalised_text": norm})
        print(f"{spk:14s} {wav.name:38s} {info['source_format']:11s} "
              f"{info['duration_s']:6.2f}s  {len(norm.split()):3d} words")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} utterances -> {args.out}  (summary: {args.report})")


if __name__ == "__main__":
    main()
