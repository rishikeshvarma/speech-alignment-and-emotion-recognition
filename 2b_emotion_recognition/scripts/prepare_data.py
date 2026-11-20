"""Build data/metadata.csv for TESS and assign the word-disjoint split.

TESS: two actresses (OAF, 64 years; YAF, 26 years) each say the carrier
phrase "Say the word ___" with 200 target words in 7 emotions
(2 x 200 x 7 = 2800 clips). Every word is spoken in every emotion by both
speakers, so:

* the text of an utterance says nothing about its emotion, and
* a random clip-level split puts the same speaker saying the same word (in a
  different emotion) in train and test, which inflates speech accuracy.

The main protocol therefore splits by *word*: 140 words for training, 30 for
validation, 30 for testing. Every split keeps both speakers and all 7
emotions in equal numbers (each word brings 14 clips).

Two small problems in the Kaggle copy are fixed here:
* the speaker is taken from the folder name, because one file is called
  OA_bite_neutral.wav instead of OAF_bite_neutral.wav;
* emotion spellings (ps, pleasant_surprised, Fear, ...) are unified.
One clip is stored at 96 kHz instead of 24.4 kHz; resampling in feature
extraction takes care of it.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common.data import EMOTION_ALIASES, EMOTIONS, METADATA, TESS_DIR  # noqa: E402

CARRIER = "Say the word {}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tess", type=Path, default=TESS_DIR)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n_val", type=int, default=30)
    ap.add_argument("--n_test", type=int, default=30)
    args = ap.parse_args()

    rows = []
    for wav in sorted(args.tess.rglob("*.wav")):
        speaker = wav.parent.name.split("_")[0].upper()          # OAF / YAF from the folder
        _, word, emo = wav.stem.split("_", 2)
        emotion = EMOTION_ALIASES.get(emo.lower(), emo.lower())
        assert emotion in EMOTIONS, wav
        info = sf.info(wav)
        rows.append({"path": wav.relative_to(args.tess).as_posix(), "speaker": speaker,
                     "word": word.lower(), "emotion": emotion, "text": CARRIER.format(word.lower()),
                     "sample_rate": info.samplerate, "duration": round(info.duration, 3)})

    words = sorted({r["word"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(words)
    test = set(words[:args.n_test])
    val = set(words[args.n_test:args.n_test + args.n_val])
    for r in rows:
        r["word_split"] = "test" if r["word"] in test else "val" if r["word"] in val else "train"

    METADATA.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} clips, {len(words)} words -> {METADATA}")
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["word_split"] == split]
        per_emo = {e: sum(r["emotion"] == e for r in sub) for e in EMOTIONS}
        per_spk = {s: sum(r["speaker"] == s for r in sub) for s in ("OAF", "YAF")}
        print(f"  {split:5s} {len(sub):5d} clips  {len({r['word'] for r in sub}):3d} words  {per_spk}  {per_emo}")
    odd_sr = [r["path"] for r in rows if r["sample_rate"] != 24414]
    print(f"  clips not at 24414 Hz: {odd_sr}")


if __name__ == "__main__":
    main()
