# Forced alignment with the Montreal Forced Aligner (Assignment 2a)

A reproducible pipeline that aligns the six provided recordings with their transcripts using
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (MFA 3.4.3). It produces
word- and phone-level TextGrids, compares several acoustic models and dictionaries, and checks
the result for errors.

| | |
|---|---|
| **Outputs** | `outputs/<run>/<speaker>/<utterance>.TextGrid` (Praat TextGrids, tiers `words` and `phones`) |
| **Report** | [`report/Report_2a_Forced_Alignment.pdf`](report/Report_2a_Forced_Alignment.pdf) |
| **Tables** | [`results/summary.md`](results/summary.md) and the CSV files in `results/` |
| **Figures** | `results/plots/` (waveform, spectrogram, word and phone tiers) |
| **One command** | `python run_pipeline.py --src "<data folder>"` |

## The data

| Speaker folder | Files | What it is | Length |
|---|---|---|---|
| `f2b` | `F2BJRLP1-3` | BU Radio News corpus: a professional announcer reading three news paragraphs | 25-31 s each, 73-82 words |
| `isle_sess0131` | `ISLE_SESS0131_BLOCKD02_0[1-3]_sprt1` | ISLE corpus: a non-native (learner) speaker reading minimal-pair prompts, e.g. *"I said white not bait"* | 4 s each, 5 words |

Two problems in the provided files had to be fixed before MFA could use them:

1. **The three `F2BJRLP*.wav` files are not WAV files.** They are NIST SPHERE files (header `NIST_1A`,
   16 kHz, 16-bit) with a `.wav` name. `prepare_corpus.py` reads them with libsndfile and writes
   real RIFF WAV files.
2. **The news transcripts are formatted news copy, not aligner input.** They are line-wrapped and
   contain punctuation, dotted abbreviations (`S.J.C.'s`), acronyms (`WBUR's`) and numbers split
   across lines (`seventy- six`). They are normalised to lower-case dictionary words, with
   acronyms spelled out letter by letter as they are spoken: `s j c's`, `w b u r's`, `seventy six`.

## Setup

```bash
# 1. MFA environment (conda-forge build includes Kaldi, OpenFst and the G2P tools)
conda env create -f environment.yml          # or: conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.3 matplotlib librosa
conda activate mfa

# 2. (optional) keep MFA's models and temporary files somewhere other than ~/Documents/MFA
conda env config vars set MFA_ROOT_DIR=/path/to/MFA && conda activate mfa

# 3. pretrained models (about 210 MB in total)
mfa model download acoustic   english_us_arpa
mfa model download acoustic   english_mfa
mfa model download dictionary english_us_arpa
mfa model download dictionary english_us_mfa
mfa model download g2p        english_us_arpa
mfa model download g2p        english_us_mfa
```

## Running

Everything, in one command (about 7 minutes on a desktop CPU):

```bash
python run_pipeline.py --src "path/to/Assignment 2a"      # the folder that holds wav/ and transcripts/
```

The steps can also be run one at a time: `--steps prepare g2p align analyse`.
`--experiments arpa_g2p` re-runs a single alignment.

What the pipeline runs, written out as plain MFA commands:

```bash
# 1. data preparation: SPHERE -> WAV, transcript normalisation, corpus/<speaker>/<utt>.wav + .lab
python scripts/prepare_corpus.py --src "path/to/Assignment 2a" --out corpus

# 2. check the corpus and list words missing from the dictionary
mfa validate corpus english_us_arpa

# 3. custom dictionary: G2P pronunciations for the missing words, added to a copy of the stock dictionary
mfa g2p corpus english_us_arpa dictionaries/english_us_arpa_oov_g2p.dict --dictionary_path english_us_arpa
cp ~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict dictionaries/english_us_arpa_g2p.dict
mfa model add_words dictionaries/english_us_arpa_g2p.dict dictionaries/english_us_arpa_oov_g2p.dict

# 4. alignment (wide beams: the news paragraphs are ~30 s long)
mfa align corpus dictionaries/english_us_arpa_g2p.dict english_us_arpa outputs/arpa_g2p --beam 100 --retry_beam 400

# 5. tables and figures
python scripts/analyze_alignments.py && python scripts/plot_alignments.py && python scripts/formant_check.py
```

### Viewing a TextGrid in Praat

Open Praat, then *Open > Read from file...*, and load the `.wav` from `corpus/<speaker>/` and the
matching `.TextGrid` from `outputs/<run>/<speaker>/`. Select both and click *View & Edit*. The
figures in `results/plots/` show the same view (waveform, spectrogram, word tier, phone tier),
with several runs stacked so they can be compared.

## Experiments

| Run | Acoustic model | Dictionary | Purpose |
|---|---|---|---|
| `arpa_base` | `english_us_arpa` | `english_us_arpa` (stock) | baseline, as in the assignment example |
| `arpa_g2p` | `english_us_arpa` | stock + G2P pronunciations for missing words | **main run**: custom dictionary |
| `arpa_g2p_finetune` | `english_us_arpa` | same | `--fine_tune`: extra boundary refinement pass |
| `arpa_g2p_adapted` | `english_us_arpa` MAP-adapted to this corpus (`mfa adapt`) | same | speaker adaptation |
| `mfa_g2p` | `english_mfa` | `english_us_mfa` + G2P | second acoustic model and phone set (IPA-like, trained on more data) |
| `isle_variants` | `english_us_arpa` | [`isle_variants.dict`](dictionaries/isle_variants.dict) | pronunciation check of the non-native speaker |

**Missing words** (G2P'd, see `dictionaries/*_oov_g2p.dict`):
- `english_us_arpa` lacks *dukakis, maffy, melnicove, politicize*.
- `english_us_mfa` also lacks *judgeships* and *margo*.

The G2P model proposed five pronunciations for *Dukakis*. The aligner chose `D UW1 K AA1 K IH0 S` in
one paragraph and `D UW0 ...` in the other, selecting from the variants by acoustic fit.

## Results

All 6 files aligned in every run, with no utterance failing to align. All numbers are in [`results/summary.md`](results/summary.md).

| Run | Words without pronunciation (`spn`) | Mean phone length | Phones at the 30 ms minimum | Word boundaries within 20 ms of `arpa_g2p` |
|---|---|---|---|---|
| `arpa_base` | 5 | 75.5 ms | 7.8 % | 99 % |
| `arpa_g2p` | **0** | 75.8 ms | 7.8 % | (reference) |
| `arpa_g2p_finetune` | 0 | 75.8 ms | 7.9 % | 100 % |
| `arpa_g2p_adapted` | 0 | 75.8 ms | 7.8 % | 100 % |
| `mfa_g2p` | 0 | 82.1 ms | 9.4 % | 92 % (news), **63 %** (ISLE) |

### Key observations

1. **The custom dictionary removes all `spn` tokens.** Without it, the aligner cannot model
   *Dukakis, Melnicove, Maffy, politicize*. Each becomes one `spn` ("spoken noise") interval with no
   phones (`results/plots/compare_oov_*.png`). With the G2P pronunciations the word edges barely move
   (all within 20 ms), but the inside of each word now has a usable phone segmentation.
2. **Adaptation and fine-tuning change almost nothing here.** Boundaries move by at most 10 ms (one
   frame). About 100 s of audio from two speakers is too little to shift a model trained on
   thousands of hours. The stock `english_us_arpa` model is already a good fit for clear read speech.
3. **`english_mfa` makes a clear error on the ISLE files.** In all three it stretches the
   final stop over the trailing silence: the `t` of *bait* lasts 2.1 s and the `d` of *bed* 2.7 s
   (`compare_isle_*.png`). It also leaves out the pause between *white* and *not*. The ARPA model
   places both silences correctly. On the news paragraphs the two models agree well (92 % of word
   boundaries within 20 ms, median 10 ms).
4. **Phones at the 30 ms floor.** About 8 % of phones have the minimum duration a 3-state HMM allows
   (3 frames x 10 ms). These are mostly reduced function words in fast news speech (*the* = `DH AH0` in
   70 ms, *of*, *to*). The alignment is squeezed there, which limits precision for very short phones.
5. **Pauses are placed where the speaker breathes.** The aligner inserted 13 pauses of 300 ms or more
   in the news paragraphs. 8 of them fall at a full stop or comma of the original text
   (*march. It*, *court. For*, *houses. Despite*). The rest fall at phrase breaks the speaker made
   without punctuation.
6. **Forced alignment hides pronunciation errors.** Standard forced alignment assumes the canonical
   pronunciation, so it labels the learner's speech as correct even when it is not. Given a choice
   of variants (`isle_variants`), the aligner chose **`B EH1 T` ("bet") for *bad*** in *"I said bad
   not bed"*: the vowel contrast the prompt is testing, plus final devoicing. Formant
   measurements agree (`results/isle_formants.csv`, `scripts/formant_check.py`):
   - F1 of *bad* is 586 Hz vs 527 Hz for *bed*, a 60 Hz difference. Native male speakers show about
     130 Hz between these vowels (Peterson & Barney: 660 vs 530 Hz).
   - The final /d/ of both *bad* and *bed* has no voiced frames. That is final devoicing, typical of
     German-accented English.
   - For *new* the aligner chose `N Y UW1`. This is the British pronunciation, not an error.

## Repository layout

```
run_pipeline.py              one-command pipeline (prepare -> g2p -> align -> analyse)
environment.yml              conda environment
scripts/
  prepare_corpus.py          SPHERE -> WAV, transcript normalisation, MFA corpus layout
  analyze_alignments.py      run comparison, boundary agreement, red-flag detection
  plot_alignments.py         Praat-style figures
  formant_check.py           F1/F2 and final-stop voicing for the ISLE vowels
  textgrid_utils.py          minimal TextGrid reader
corpus/<speaker>/*.lab       normalised transcripts (audio not redistributed)
dictionaries/                G2P pronunciations for missing words, ISLE variant dictionary
outputs/<run>/               TextGrids + MFA's alignment_analysis.csv for every run
results/                     tables (CSV / Markdown) and plots/
report/                      PDF report
```

The audio files are not included in the repository (BU Radio News and ISLE are licensed corpora).
Put the provided `wav/` and `transcripts/` folders anywhere and pass that folder to `--src`.
