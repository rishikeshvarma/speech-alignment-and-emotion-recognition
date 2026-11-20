# Test accuracy (%) - mean ± std over 3 seeds

Chance = 14.3 % (7 balanced classes). Test sets: 420 clips (word-disjoint) / 1400 clips of the unseen speaker (cross-speaker).

| Model | Word-disjoint (main) | Cross-speaker OAF->YAF | Cross-speaker YAF->OAF | Cross-speaker mean |
|---|---|---|---|---|
| Speech: WavLM + BiGRU | 99.9 ± 0.1 | 53.8 ± 2.5 | 55.2 ± 6.4 | 54.5 |
| Speech: log-mel + BiGRU | 99.8 ± 0.1 | 47.6 ± 5.9 | 35.6 ± 4.4 | 41.6 |
| Text: BERT + BiGRU | 14.3 ± 0.0 | 14.3 ± 0.0 | 14.3 ± 0.0 | 14.3 |
| Fusion: GMU (WavLM + BERT) | 99.8 ± 0.4 | 48.4 ± 4.8 | 56.5 ± 6.0 | 52.5 |
| Fusion: concat (WavLM + BERT) | 99.8 ± 0.1 | 56.8 ± 10.7 | 62.2 ± 6.0 | 59.5 |
| Fusion: GMU (log-mel + BERT) | 100.0 ± 0.0 | 52.2 ± 8.7 | 35.9 ± 6.9 | 44.0 |
| Speech: WavLM + speaker norm. | 99.8 ± 0.1 | 52.5 ± 7.5 | 56.8 ± 5.9 | 54.7 |
| Fusion: GMU + speaker norm. | 99.9 ± 0.1 | 54.9 ± 14.5 | 55.6 ± 9.0 | 55.2 |
| Control: GMU, text = zeros | 99.9 ± 0.1 | - | - | - |
| Control: GMU, wrong word's text | 99.9 ± 0.1 | - | - | - |

# Macro-F1 (%)

| Model | Word-disjoint (main) | Cross-speaker OAF->YAF | Cross-speaker YAF->OAF |
|---|---|---|---|
| Speech: WavLM + BiGRU | 99.9 ± 0.1 | 50.7 ± 3.1 | 47.7 ± 10.4 |
| Speech: log-mel + BiGRU | 99.8 ± 0.1 | 46.6 ± 3.8 | 30.0 ± 6.7 |
| Text: BERT + BiGRU | 6.9 ± 2.4 | 5.5 ± 1.8 | 5.5 ± 1.8 |
| Fusion: GMU (WavLM + BERT) | 99.8 ± 0.4 | 45.4 ± 2.5 | 49.5 ± 8.1 |
| Fusion: concat (WavLM + BERT) | 99.8 ± 0.1 | 53.0 ± 9.3 | 56.2 ± 5.9 |
| Fusion: GMU (log-mel + BERT) | 100.0 ± 0.0 | 52.6 ± 9.6 | 28.7 ± 9.2 |
| Speech: WavLM + speaker norm. | 99.8 ± 0.1 | 49.9 ± 5.3 | 50.8 ± 4.5 |
| Fusion: GMU + speaker norm. | 99.9 ± 0.1 | 52.4 ± 11.7 | 47.5 ± 9.3 |
| Control: GMU, text = zeros | 99.9 ± 0.1 | - | - |
| Control: GMU, wrong word's text | 99.9 ± 0.1 | - | - |
