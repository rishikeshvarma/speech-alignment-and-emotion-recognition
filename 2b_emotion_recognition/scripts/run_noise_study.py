"""Train what the noise-robustness comparison needs, then run it.

* seeds 3-9 (word-disjoint protocol) for speech-only, GMU fusion and concat
  fusion, so the comparison uses 10 seeds per model;
* two controls (seeds 0-9): fusion_gmu_wavlm_notext, GMU fusion whose text
  branch only ever sees zeros, and fusion_gmu_wavlm_shuftext, which always gets
  the transcript of a different word.

Then scripts/noise_robustness.py evaluates all of them on noisy test speech.

    python scripts/run_noise_study.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORD = ["--protocols", "word"]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> None:
    extra_seeds = ["--seeds", *[str(s) for s in range(3, 10)]]
    all_seeds = ["--seeds", *[str(s) for s in range(10)]]
    run([sys.executable, "models/fusion_pipeline/train.py", *WORD, *all_seeds, "--text_input", "zeros"])
    run([sys.executable, "models/fusion_pipeline/train.py", *WORD, *all_seeds, "--text_input", "shuffled"])
    run([sys.executable, "models/speech_pipeline/train.py", *WORD, *extra_seeds])
    run([sys.executable, "models/fusion_pipeline/train.py", *WORD, *extra_seeds])
    run([sys.executable, "models/fusion_pipeline/train.py", *WORD, *extra_seeds, "--fusion", "concat"])
    run([sys.executable, "scripts/noise_robustness.py"])


if __name__ == "__main__":
    main()
