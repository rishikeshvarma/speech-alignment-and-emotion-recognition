"""Labels, paths and dataset helpers shared by all three pipelines."""
from __future__ import annotations

import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Data and cached features live outside the repository (TESS is CC BY-NC-ND,
# so the audio must not be redistributed). Override with environment variables.
TESS_DIR = Path(os.environ.get("TESS_DIR", REPO.parent / "data" / "TESS"))
WORK_DIR = Path(os.environ.get("EMO_WORK_DIR", REPO.parent / "work"))
METADATA = REPO / "data" / "metadata.csv"
RESULTS = REPO / "Results"
CHECKPOINTS = WORK_DIR / "checkpoints"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}

# TESS file names use these spellings for the emotion
EMOTION_ALIASES = {"ps": "pleasant_surprise", "pleasant_surprised": "pleasant_surprise",
                   "pleasant_surprise": "pleasant_surprise"}

# Split protocols (see scripts/prepare_data.py):
#   word        - test words never heard in training, both speakers everywhere (main results)
#   oaf2yaf     - train on the older speaker, test on the younger one (and unseen words)
#   yaf2oaf     - the reverse
PROTOCOLS = ["word", "oaf2yaf", "yaf2oaf"]


def load_metadata() -> list[dict]:
    with open(METADATA, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split_rows(rows: list[dict], protocol: str, split: str) -> list[dict]:
    """Rows of one split ("train" / "val" / "test") under a split protocol."""
    if protocol == "word":
        return [r for r in rows if r["word_split"] == split]
    if protocol in ("oaf2yaf", "yaf2oaf"):
        src, tgt = ("OAF", "YAF") if protocol == "oaf2yaf" else ("YAF", "OAF")
        spk = tgt if split == "test" else src
        return [r for r in rows if r["speaker"] == spk and r["word_split"] == split]
    raise ValueError(f"unknown protocol {protocol}")
