"""Train the text-only model (one checkpoint per split protocol and seed).

    python models/text_pipeline/train.py [--protocols word] [--seeds 0 1 2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.runner import train_main  # noqa: E402
from models.text_pipeline.model import build_model  # noqa: E402


def variant_of(args):
    return "text_bert"


def config_of(args):
    return {"speech_features": None}          # text only: do not load speech features


if __name__ == "__main__":
    train_main(variant_of, build_model, config_of, None, __doc__)
