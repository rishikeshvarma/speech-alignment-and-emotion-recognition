"""Seven extra seeds (3-9) for the cross-speaker protocols of the WavLM models.

Cross-speaker results move by 5-10 points between seeds: model selection can
only use held-out words of the training speaker, which reach ~100 % almost at
once, so the chosen epoch is close to arbitrary for the unseen speaker. Ten
seeds per model make it possible to tell real differences (e.g. fusion vs.
speech-only) from that noise; scripts/analyze_results.py then writes
Results/seed_study.md with Welch's t-tests. Run after scripts/run_all.py.

    python scripts/run_seed_study.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = [str(s) for s in range(3, 10)]
PROTOCOLS = ["oaf2yaf", "yaf2oaf"]
VARIANTS = [
    ("speech_pipeline", []),
    ("fusion_pipeline", []),
    ("fusion_pipeline", ["--fusion", "concat"]),
    ("speech_pipeline", ["--speaker_norm"]),
    ("fusion_pipeline", ["--speaker_norm"]),
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> None:
    for pipeline, extra in VARIANTS:
        args = ["--protocols", *PROTOCOLS, "--seeds", *SEEDS, *extra]
        run([sys.executable, f"models/{pipeline}/train.py", *args])
        run([sys.executable, f"models/{pipeline}/test.py", *args])
    run([sys.executable, "scripts/analyze_results.py"])


if __name__ == "__main__":
    main()
