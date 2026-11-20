# Speech alignment and emotion recognition

Assignment 2, in two independent parts. Each folder has its own README with setup, commands and results.

| Part | Folder | Report |
|---|---|---|
| **2a** Forced alignment with the Montreal Forced Aligner | [`2a_forced_alignment/`](2a_forced_alignment) | [PDF](2a_forced_alignment/report/Report_2a_Forced_Alignment.pdf) |
| **2b** Multimodal (speech + text) emotion recognition on TESS | [`2b_emotion_recognition/`](2b_emotion_recognition) | [PDF](2b_emotion_recognition/report/Report_2b_Multimodal_Emotion_Recognition.pdf) |

**2a:** a one-command pipeline (`run_pipeline.py`) that converts the provided audio, normalises the transcripts,
builds a custom dictionary with G2P, aligns with two acoustic models plus an adapted one, and analyses the
TextGrids. All 6 files align in every run.

**2b:** speech-only, text-only and gated-fusion models, each with `train.py` / `test.py`. On a word-disjoint
split, speech reaches 99.9 % accuracy. Text sits at exactly chance (14.3 %), because every TESS word is spoken in every emotion.
The report analyses when fusion helps (only under noisy speech) and shows how emotions cluster in each block.

Audio is not included: the 2a recordings come from licensed corpora, and TESS (CC BY-NC-ND) is downloaded from
Kaggle. The READMEs explain where to put the data.
