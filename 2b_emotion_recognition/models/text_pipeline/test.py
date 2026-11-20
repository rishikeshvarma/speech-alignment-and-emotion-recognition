"""Evaluate the text-only checkpoints on the test split of each protocol.

    python models/text_pipeline/test.py [--protocols word] [--seeds 0 1 2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.runner import test_main  # noqa: E402
from models.text_pipeline.model import build_model  # noqa: E402
from models.text_pipeline.train import variant_of  # noqa: E402

if __name__ == "__main__":
    test_main(variant_of, build_model, None, __doc__)
