# Multimodal Emotion Recognition on TESS (Assignment 2b)

Three emotion classifiers for the Toronto Emotional Speech Set (TESS), built from the functional
blocks of the assignment (preprocessing, feature extraction, temporal/contextual modelling, fusion,
classifier):

| Pipeline | Input | Feature extraction | Temporal / contextual modelling | Fusion | Classifier |
|---|---|---|---|---|---|
| `models/speech_pipeline` | speech | frozen **WavLM-Base+**, learned mix of 13 layers | 2-layer **BiGRU** + attentive pooling | - | MLP |
| `models/text_pipeline` | transcript | frozen **BERT-base**, learned mix of 13 layers | 1-layer **BiGRU** + attentive pooling | - | MLP |
| `models/fusion_pipeline` | both | both of the above | both of the above | **Gated Multimodal Unit** | MLP |

**Report:** [`report/Report_2b_Multimodal_Emotion_Recognition.pdf`](report/Report_2b_Multimodal_Emotion_Recognition.pdf)
&nbsp;·&nbsp; **Tables:** [`Results/summary.md`](Results/summary.md) &nbsp;·&nbsp; **Figures:** [`Results/plots/`](Results/plots)

## Results

Test accuracy (%), mean ± std over 3 seeds (chance = 14.3 %). Full tables in [`Results/accuracy_table.md`](Results/accuracy_table.md).

| Model | Word-disjoint (main) | Cross-speaker OAF→YAF | Cross-speaker YAF→OAF |
|---|---|---|---|
| **Speech only** (WavLM + BiGRU) | **99.9 ± 0.1** | 53.8 ± 2.5 | 55.2 ± 6.4 |
| **Text only** (BERT + BiGRU) | 14.3 ± 0.0 | 14.3 ± 0.0 | 14.3 ± 0.0 |
| **Speech + text** (GMU fusion) | 99.8 ± 0.4 | 48.4 ± 4.8 | 56.5 ± 6.0 |
| Speech only, log-mel features | 99.8 ± 0.1 | 47.6 ± 5.9 | 35.6 ± 4.4 |
| Fusion, concatenation | 99.8 ± 0.1 | 56.8 ± 10.7 | 62.2 ± 6.0 |
| Speech only + speaker normalisation | 99.8 ± 0.1 | 52.5 ± 7.5 | 56.8 ± 5.9 |

What the analysis found (details and figures in the report):

- **Text-only is at exactly 1/7 on every seed.** This is a property of TESS, not a weak model: every test word occurs
  twice in each emotion, so any function of the transcript gets exactly 1/7 right.
- **Within the two speakers, speech solves the task** (1 error in 1,260 test predictions). **Across speakers it drops to
  about 55 %.** The high-arousal emotions (angry 7 %, happy 15 %, pleasant surprise 19 %) are expressed differently by the
  two actresses. Sad, neutral, fear and disgust hold up (82-89 %).
- **When does fusion help?** Not on clean speech (ceiling), and not across speakers (10 seeds, no significant
  difference; [`Results/seed_study.md`](Results/seed_study.md)). Under heavy noise the GMU fusion model is more robust
  (+8.7 points over speech-only at 0 dB, p = 0.07; +13.6 over the same model with zeroed text, p = 0.015). A control fed
  the *wrong* word's transcript keeps most of that gain, so it comes from the gated architecture, not from the text.
  Concatenation fusion is significantly *less* robust ([`Results/noise_robustness.md`](Results/noise_robustness.md)).
- **Representations:** the speech (temporal) block forms seven clean clusters (silhouette 0.95). The text
  (contextual) block has none (-0.02), and the fusion block inherits the speech clusters (0.98)
  ([`Results/plots/tsne_blocks_word.png`](Results/plots/tsne_blocks_word.png)).
- **Speaker normalisation** (per-speaker feature standardisation) does not close the cross-speaker gap. The unseen
  speaker's emotion clusters are rearranged relative to the training speaker's, not just shifted.
- **Error analysis:** five failure cases, each shown next to the same speaker saying the same word in the predicted
  emotion ([`Results/error_analysis.md`](Results/error_analysis.md), [`Results/plots/failure_cases.png`](Results/plots/failure_cases.png)).

## Two facts about TESS that shape the evaluation

- **The text carries no label information.** Each of the 200 target words is spoken in all seven
  emotions by both speakers, so *"Say the word bite"* appears with every label equally often. Any
  text-only model is at chance (1/7 = 14.3 %), whatever its architecture. The fusion model has to learn
  to ignore its text branch.
- **Random splits leak.** A clip-level random split puts the same speaker saying the same word (in
  another emotion) in train and test. The main protocol therefore splits by **word**: 140 / 30 / 30
  words for train / validation / test (1960 / 420 / 420 clips, both speakers and all emotions balanced in
  every split, see `data/metadata.csv`). A **cross-speaker** protocol (train on one actress, test on the
  other, in both directions) measures generalisation to a new voice.

## Setup

```bash
conda create -n emo python=3.10 && conda activate emo
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Download TESS from Kaggle (`ejlok1/toronto-emotional-speech-set-tess`) and unpack one copy of it so the
emotion folders (`OAF_angry/`, `YAF_sad/`, ...) sit in `../data/TESS/` next to this repository.
Paths can be changed with the environment variables `TESS_DIR` (dataset) and `EMO_WORK_DIR`
(cached features and checkpoints, default `../work`). Pretrained encoders are downloaded from Hugging Face
on first use. Set `HF_HOME` to choose where they are stored.

## Running

```bash
python scripts/prepare_data.py          # data/metadata.csv: labels, transcripts, word-disjoint split
python scripts/extract_features.py      # WavLM + log-mel + BERT features, cached once (~3 min on GPU, ~5 GB)

# each pipeline on its own (all protocols, seeds 0-2 by default)
python models/speech_pipeline/train.py   &&  python models/speech_pipeline/test.py
python models/text_pipeline/train.py     &&  python models/text_pipeline/test.py
python models/fusion_pipeline/train.py   &&  python models/fusion_pipeline/test.py

# or everything, including ablations and the analysis (about 1 h on one RTX 5060 Ti)
python scripts/run_all.py
python scripts/run_seed_study.py        # seeds 3-9 for the cross-speaker comparison (~1 h)
python scripts/run_noise_study.py       # extra seeds + text controls, then speech vs fusion on noisy speech (~40 min)
```

Useful options: `--protocols word oaf2yaf yaf2oaf`, `--seeds 0 1 2`, `--features wavlm|mel`
(speech and fusion), `--fusion gmu|concat` (fusion), `--speaker_norm` (speech and fusion),
`--text_input transcript|zeros|shuffled` (fusion; the last two are controls).
`test.py` writes `Results/metrics.csv` (one row per variant, protocol and seed) and per-clip
predictions in `Results/predictions/`.

## Repository layout

```
models/
  common/            data + split protocols, cached-feature loader, shared layers, train/test loops
  speech_pipeline/   model.py, train.py, test.py
  text_pipeline/     model.py, train.py, test.py
  fusion_pipeline/   model.py, train.py, test.py
scripts/
  prepare_data.py    metadata, word-disjoint split, fixes for the Kaggle copy
  extract_features.py
  run_all.py         every variant, then analyze_results.py
  run_seed_study.py  extra seeds for the cross-speaker comparison
  run_noise_study.py extra seeds and text controls, then noise_robustness.py
  analyze_results.py tables, confusion matrices, t-SNE, layer weights, gate, seed study, error analysis
  noise_robustness.py speech-only vs fusion on noisy test speech
data/metadata.csv    2800 clips: path, speaker, word, emotion, transcript, split
Results/             accuracy tables, per-class accuracy, analysis notes, predictions, training logs
Results/plots/       figures
report/              PDF report (+ HTML source)
```
