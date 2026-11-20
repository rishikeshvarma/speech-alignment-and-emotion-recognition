"""Evaluate the speech-only checkpoints on the test split of each protocol.

Writes Results/metrics.csv rows, per-clip predictions in Results/predictions/
and the temporal-block embeddings (for t-SNE) in <work>/embeddings/.

    python models/speech_pipeline/test.py [--features mel] [--protocols word] [--seeds 0 1 2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.runner import test_main  # noqa: E402
from models.speech_pipeline.model import build_model  # noqa: E402
from models.speech_pipeline.train import add_args, variant_of  # noqa: E402

if __name__ == "__main__":
    test_main(variant_of, build_model, add_args, __doc__)
