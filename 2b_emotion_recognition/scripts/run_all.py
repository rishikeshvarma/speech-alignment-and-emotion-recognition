"""Train and test every model variant, then build the result tables and plots.

    python scripts/run_all.py                 # everything (about an hour on one RTX 5060 Ti)
    python scripts/run_all.py --only fusion   # only the variants whose name contains "fusion"

Assumes scripts/prepare_data.py and scripts/extract_features.py have run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

VARIANTS = [
    # (name, pipeline, extra arguments)
    ("speech_wavlm", "speech_pipeline", []),
    ("speech_mel", "speech_pipeline", ["--features", "mel"]),
    ("text_bert", "text_pipeline", []),
    ("fusion_gmu_wavlm", "fusion_pipeline", []),
    ("fusion_concat_wavlm", "fusion_pipeline", ["--fusion", "concat"]),
    ("fusion_gmu_mel", "fusion_pipeline", ["--features", "mel"]),
    # extra: speaker-level feature normalisation, aimed at the cross-speaker gap
    ("speech_wavlm_spknorm", "speech_pipeline", ["--speaker_norm"]),
    ("fusion_gmu_wavlm_spknorm", "fusion_pipeline", ["--speaker_norm"]),
    # control for the fusion analysis: same GMU model, text branch fed zeros (word-disjoint only)
    ("fusion_gmu_wavlm_notext", "fusion_pipeline", ["--text_input", "zeros", "--protocols", "word"]),
    ("fusion_gmu_wavlm_shuftext", "fusion_pipeline", ["--text_input", "shuffled", "--protocols", "word"]),
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run only variants whose name contains this text")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for name, pipeline, extra in VARIANTS:
        if args.only and args.only not in name:
            continue
        if not args.skip_train:
            run([sys.executable, f"models/{pipeline}/train.py", *extra])
        run([sys.executable, f"models/{pipeline}/test.py", *extra])
    run([sys.executable, "scripts/analyze_results.py"])
    print(f"\nall done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
