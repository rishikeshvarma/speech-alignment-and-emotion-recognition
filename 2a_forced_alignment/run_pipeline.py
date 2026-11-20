"""Full Assignment 2a pipeline in one command.

    conda activate mfa
    python run_pipeline.py --src "<folder with wav/ and transcripts/>"

Steps
  1. prepare   SPHERE -> WAV, transcript normalisation, MFA corpus layout
  2. g2p       pronunciations for out-of-vocabulary words with MFA's G2P
               models, added to copies of the stock dictionaries
  3. align     one MFA run per experiment in EXPERIMENTS (TextGrids in outputs/)
  4. analyse   tables in results/ and figures in results/plots/

Every step can be re-run on its own with --steps, e.g. --steps align analyse.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
DICTS = ROOT / "dictionaries"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

# Pretrained models live under MFA_ROOT_DIR (default: ~/Documents/MFA).
MFA_ROOT = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))
PRETRAINED = MFA_ROOT / "pretrained_models"

# Stock dictionary -> G2P model trained on the same phone set.
DICTIONARIES = {
    "english_us_arpa": "english_us_arpa",   # ARPAbet, CMUdict-style (39 phones + stress)
    "english_us_mfa": "english_us_mfa",     # MFA's IPA-based phone set
}

# Long news paragraphs (25-30 s) need wider beams than MFA's defaults (10/40).
BEAM = ["--beam", "100", "--retry_beam", "400"]

EXPERIMENTS = [
    # name,              acoustic model,     dictionary,              extra flags,      adapt first?, speaker subset
    ("arpa_base",        "english_us_arpa",  "english_us_arpa",       [],               False, None),
    ("arpa_g2p",         "english_us_arpa",  "english_us_arpa_g2p",   [],               False, None),
    ("arpa_g2p_finetune","english_us_arpa",  "english_us_arpa_g2p",   ["--fine_tune"],  False, None),
    ("mfa_g2p",          "english_mfa",      "english_us_mfa_g2p",    [],               False, None),
    ("arpa_g2p_adapted", "english_us_arpa",  "english_us_arpa_g2p",   [],               True,  None),
    # Pronunciation check for the non-native ISLE speaker: dictionaries/isle_variants.dict
    # lists the canonical pronunciation of each prompt word next to typical
    # learner variants (final devoicing, bad/bed vowel merger, w->v), all
    # equally likely. The aligner keeps whichever variant fits the audio best.
    ("isle_variants",    "english_us_arpa",  "isle_variants",         [],               False, "isle_sess0131"),
]


def mfa(*args: str) -> None:
    exe = shutil.which("mfa")
    if exe is None:
        sys.exit("`mfa` not found - activate the MFA conda environment first")
    cmd = [exe, *args]
    print("\n$ mfa " + " ".join(args), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUTF8": "1"})
    print(f"  ({time.time() - t0:.0f} s)")


def dictionary_path(name: str) -> str:
    """Stock dictionaries are referred to by name, custom ones by file path."""
    custom = DICTS / f"{name}.dict"
    return str(custom) if custom.exists() else name


def step_prepare(src: Path) -> None:
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_corpus.py"),
                    "--src", str(src), "--out", str(CORPUS),
                    "--report", str(RESULTS / "corpus_summary.csv")], check=True)


def step_g2p(jobs: int) -> None:
    """For each stock dictionary: G2P the corpus words it lacks, then add them
    to a copy of the dictionary (english_us_arpa -> english_us_arpa_g2p)."""
    DICTS.mkdir(exist_ok=True)
    for dict_name, g2p_name in DICTIONARIES.items():
        oov_dict = DICTS / f"{dict_name}_oov_g2p.dict"
        # With --dictionary_path, MFA tokenises the corpus the same way the
        # aligner does (clitics such as 's are split off) and only generates
        # pronunciations for words that are still missing.
        mfa("g2p", str(CORPUS), g2p_name, str(oov_dict),
            "--dictionary_path", dict_name, "--clean", "-j", str(jobs), "--overwrite")
        custom = DICTS / f"{dict_name}_g2p.dict"
        shutil.copyfile(PRETRAINED / "dictionary" / f"{dict_name}.dict", custom)
        mfa("model", "add_words", str(custom), str(oov_dict))
        print(f"  {oov_dict.name}:")
        print("    " + oov_dict.read_text(encoding="utf-8").strip().replace("\n", "\n    "))


def step_align(jobs: int, only: list[str] | None) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    for name, acoustic, dictionary, extra, adapt, subset in EXPERIMENTS:
        if only and name not in only:
            continue
        out = OUTPUTS / name
        if out.exists():
            shutil.rmtree(out)
        corpus = CORPUS
        if subset:
            # MFA has no speaker filter, so give it a corpus with one speaker folder
            corpus = ROOT / "work" / f"corpus_{subset}"
            if corpus.exists():
                shutil.rmtree(corpus)
            shutil.copytree(CORPUS / subset, corpus / subset)
        common = [str(corpus), dictionary_path(dictionary)]
        if adapt:
            # MAP-adapt the pretrained model's Gaussians to these two speakers,
            # then align with the adapted model in the same run.
            model_out = MODELS / f"{acoustic}_adapted.zip"
            mfa("adapt", *common, acoustic, str(model_out), "--output_directory", str(out),
                "--clean", "-j", str(jobs), "--overwrite", *BEAM, *extra)
        else:
            mfa("align", *common, acoustic, str(out),
                "--clean", "-j", str(jobs), "--overwrite", *BEAM, *extra)


def step_analyse() -> None:
    for script in ("analyze_alignments.py", "plot_alignments.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                        "--outputs", str(OUTPUTS), "--corpus", str(CORPUS),
                        "--results", str(RESULTS)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="provided data folder (needed for the prepare step)")
    ap.add_argument("--steps", nargs="+", default=["prepare", "g2p", "align", "analyse"],
                    choices=["prepare", "g2p", "align", "analyse"])
    ap.add_argument("--experiments", nargs="+", help="align only these experiments")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    args = ap.parse_args()

    if "prepare" in args.steps:
        if args.src is None:
            ap.error("--src is required for the prepare step")
        step_prepare(args.src)
    if "g2p" in args.steps:
        step_g2p(args.jobs)
    if "align" in args.steps:
        step_align(args.jobs, args.experiments)
    if "analyse" in args.steps:
        step_analyse()


if __name__ == "__main__":
    main()
