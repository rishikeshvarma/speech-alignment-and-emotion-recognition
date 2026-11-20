"""Train the speech-only model (one checkpoint per split protocol and seed).

    python models/speech_pipeline/train.py                     # WavLM features, all protocols, seeds 0-2
    python models/speech_pipeline/train.py --features mel      # log-mel baseline
    python models/speech_pipeline/train.py --protocols word --seeds 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.runner import train_main  # noqa: E402
from models.speech_pipeline.model import build_model  # noqa: E402


def add_args(ap):
    ap.add_argument("--features", default="wavlm", choices=["wavlm", "mel"])
    ap.add_argument("--speaker_norm", action="store_true",
                    help="standardise features per speaker (speaker-level CMVN)")


def variant_of(args):
    return f"speech_{args.features}" + ("_spknorm" if args.speaker_norm else "")


def config_of(args):
    return {"speech_features": args.features, "speaker_norm": args.speaker_norm}


if __name__ == "__main__":
    train_main(variant_of, build_model, config_of, add_args, __doc__)
