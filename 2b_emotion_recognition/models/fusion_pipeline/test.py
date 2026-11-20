"""Evaluate the fusion checkpoints on the test split of each protocol.

Besides the metrics and predictions, this saves the temporal, contextual and
fusion embeddings of every test clip and the per-clip speech gate value.

    python models/fusion_pipeline/test.py [--fusion concat] [--protocols word] [--seeds 0 1 2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.runner import test_main  # noqa: E402
from models.fusion_pipeline.model import build_model  # noqa: E402
from models.fusion_pipeline.train import add_args, variant_of  # noqa: E402

if __name__ == "__main__":
    test_main(variant_of, build_model, add_args, __doc__)
