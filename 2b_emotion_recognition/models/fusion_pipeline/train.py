"""Train the speech + text fusion model (one checkpoint per split protocol and seed).

    python models/fusion_pipeline/train.py                    # gated fusion, WavLM, all protocols
    python models/fusion_pipeline/train.py --fusion concat    # concatenation baseline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.runner import train_main  # noqa: E402
from models.fusion_pipeline.model import build_model  # noqa: E402


def add_args(ap):
    ap.add_argument("--fusion", default="gmu", choices=["gmu", "concat"])
    ap.add_argument("--features", default="wavlm", choices=["wavlm", "mel"], help="speech features")
    ap.add_argument("--speaker_norm", action="store_true",
                    help="standardise speech features per speaker (speaker-level CMVN)")
    ap.add_argument("--text_input", default="transcript", choices=["transcript", "zeros", "shuffled"],
                    help="controls: zeros = text branch gets all-zero input; "
                         "shuffled = transcript of a different word")


def variant_of(args):
    return (f"fusion_{args.fusion}_{args.features}" + ("_spknorm" if args.speaker_norm else "")
            + {"transcript": "", "zeros": "_notext", "shuffled": "_shuftext"}[args.text_input])


def config_of(args):
    return {"speech_features": args.features, "fusion": args.fusion, "speaker_norm": args.speaker_norm,
            "text_input": args.text_input}


if __name__ == "__main__":
    train_main(variant_of, build_model, config_of, add_args, __doc__)
