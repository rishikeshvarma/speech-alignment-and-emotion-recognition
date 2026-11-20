# Where does fusion change the speech-only decision?

Paired comparison on the same test clips and seeds. *fixed* = speech-only wrong, fusion right; *broken* = speech-only right, fusion wrong.

| fusion | vs speech-only | protocol | speech acc | fusion acc | fixed / seed | broken / seed |
|---|---|---|---|---|---|---|
| Fusion: GMU (WavLM + BERT) | Speech: WavLM + BiGRU | Word-disjoint (main) | 99.9 | 99.8 | 0.3 | 1.0 |
| Fusion: GMU (WavLM + BERT) | Speech: WavLM + BiGRU | Cross-speaker OAF->YAF | 53.8 | 48.4 | 8.3 | 19.7 |
| Fusion: GMU (WavLM + BERT) | Speech: WavLM + BiGRU | Cross-speaker YAF->OAF | 55.2 | 56.5 | 24.3 | 21.7 |
| Fusion: concat (WavLM + BERT) | Speech: WavLM + BiGRU | Word-disjoint (main) | 99.9 | 99.8 | 0.3 | 0.7 |
| Fusion: concat (WavLM + BERT) | Speech: WavLM + BiGRU | Cross-speaker OAF->YAF | 53.8 | 56.8 | 25.7 | 19.3 |
| Fusion: concat (WavLM + BERT) | Speech: WavLM + BiGRU | Cross-speaker YAF->OAF | 55.2 | 62.2 | 28.3 | 13.7 |
| Fusion: GMU (log-mel + BERT) | Speech: log-mel + BiGRU | Word-disjoint (main) | 99.8 | 100.0 | 0.7 | 0.0 |
| Fusion: GMU (log-mel + BERT) | Speech: log-mel + BiGRU | Cross-speaker OAF->YAF | 47.6 | 52.2 | 31.7 | 22.0 |
| Fusion: GMU (log-mel + BERT) | Speech: log-mel + BiGRU | Cross-speaker YAF->OAF | 35.6 | 35.9 | 21.7 | 21.0 |
| Fusion: GMU + speaker norm. | Speech: WavLM + speaker norm. | Word-disjoint (main) | 99.8 | 99.9 | 0.7 | 0.3 |
| Fusion: GMU + speaker norm. | Speech: WavLM + speaker norm. | Cross-speaker OAF->YAF | 52.5 | 54.9 | 33.3 | 28.3 |
| Fusion: GMU + speaker norm. | Speech: WavLM + speaker norm. | Cross-speaker YAF->OAF | 56.8 | 55.6 | 26.0 | 28.7 |

## GMU fusion vs WavLM speech-only, cross-speaker, by emotion (summed over seeds and directions)

| emotion | fixed | broken |
|---|---|---|
| angry | 25 | 1 |
| disgust | 6 | 26 |
| fear | 26 | 9 |
| happy | 5 | 26 |
| neutral | 3 | 19 |
| pl. surprise | 28 | 13 |
| sad | 5 | 30 |

## GMU gate: mean of z over the 256 dimensions (1 = all speech, 0 = all text)

| protocol | mean | std | min | max |
|---|---|---|---|---|
| oaf2yaf | 0.722 | 0.036 | 0.637 | 0.822 |
| word | 0.744 | 0.042 | 0.638 | 0.802 |
| yaf2oaf | 0.831 | 0.047 | 0.709 | 0.913 |
