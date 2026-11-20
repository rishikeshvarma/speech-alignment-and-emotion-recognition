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


# Per-emotion test accuracy (recall, %), mean over seeds (and both directions for cross-speaker)

| Model | Protocol | angry | disgust | fear | happy | neutral | pl. surprise | sad |
|---|---|---|---|---|---|---|---|---|
| Speech only | word-disjoint | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.4 | 100.0 |
| Speech only | cross-speaker | 6.7 | 81.7 | 82.2 | 15.0 | 87.8 | 19.4 | 88.9 |
| Text only | word-disjoint | 0.0 | 14.4 | 13.3 | 28.9 | 0.0 | 2.2 | 41.1 |
| Text only | cross-speaker | 14.4 | 0.0 | 0.0 | 62.2 | 0.0 | 23.3 | 0.0 |
| Speech + text (GMU fusion) | word-disjoint | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 98.3 | 100.0 |
| Speech + text (GMU fusion) | cross-speaker | 20.0 | 70.6 | 91.7 | 3.3 | 78.9 | 27.8 | 75.0 |


# Seed study: cross-speaker accuracy over all seeds

Mean ± sample std over seeds; *difference* and Welch's t-test against the speech-only model with the same features.

| model | protocol | seeds | accuracy | min-max | vs | difference | p (Welch) |
|---|---|---|---|---|---|---|---|
| Speech: WavLM + BiGRU | Cross-speaker OAF->YAF | 10 | 54.7 ± 6.7 | 46-69 | - | - | - |
| Speech: WavLM + BiGRU | Cross-speaker YAF->OAF | 10 | 57.0 ± 9.3 | 41-75 | - | - | - |
| Fusion: GMU (WavLM + BERT) | Cross-speaker OAF->YAF | 10 | 51.3 ± 7.3 | 40-63 | Speech: WavLM + BiGRU | -3.4 | 0.289 |
| Fusion: GMU (WavLM + BERT) | Cross-speaker YAF->OAF | 10 | 55.1 ± 10.5 | 39-68 | Speech: WavLM + BiGRU | -1.9 | 0.672 |
| Fusion: concat (WavLM + BERT) | Cross-speaker OAF->YAF | 10 | 54.2 ± 7.4 | 46-67 | Speech: WavLM + BiGRU | -0.5 | 0.870 |
| Fusion: concat (WavLM + BERT) | Cross-speaker YAF->OAF | 10 | 57.6 ± 5.5 | 49-69 | Speech: WavLM + BiGRU | +0.5 | 0.881 |
| Speech: WavLM + speaker norm. | Cross-speaker OAF->YAF | 10 | 57.5 ± 5.8 | 44-67 | Speech: WavLM + BiGRU | +2.8 | 0.329 |
| Speech: WavLM + speaker norm. | Cross-speaker YAF->OAF | 10 | 54.3 ± 6.0 | 41-63 | Speech: WavLM + BiGRU | -2.8 | 0.442 |
| Fusion: GMU + speaker norm. | Cross-speaker OAF->YAF | 10 | 57.2 ± 9.3 | 42-70 | Speech: WavLM + speaker norm. | -0.3 | 0.935 |
| Fusion: GMU + speaker norm. | Cross-speaker YAF->OAF | 10 | 59.2 ± 7.5 | 47-70 | Speech: WavLM + speaker norm. | +4.9 | 0.122 |


# Separability of the learned representations

Silhouette score of the test-set embeddings with the true emotion as cluster label (cosine distance; 1 = tight, well-separated clusters, 0 = overlapping, < 0 = mixed). Mean ± std over 3 seeds.

| model | block | protocol | silhouette |
|---|---|---|---|
| fusion_gmu_wavlm | contextual | oaf2yaf | -0.033 ± 0.000 |
| fusion_gmu_wavlm | contextual | word | -0.017 ± 0.000 |
| fusion_gmu_wavlm | contextual | yaf2oaf | -0.033 ± 0.000 |
| fusion_gmu_wavlm | fusion | oaf2yaf | 0.577 ± 0.029 |
| fusion_gmu_wavlm | fusion | word | 0.983 ± 0.010 |
| fusion_gmu_wavlm | fusion | yaf2oaf | 0.270 ± 0.040 |
| fusion_gmu_wavlm | temporal | oaf2yaf | 0.617 ± 0.026 |
| fusion_gmu_wavlm | temporal | word | 0.951 ± 0.009 |
| fusion_gmu_wavlm | temporal | yaf2oaf | 0.313 ± 0.019 |
| speech_wavlm | temporal | oaf2yaf | 0.612 ± 0.107 |
| speech_wavlm | temporal | word | 0.965 ± 0.014 |
| speech_wavlm | temporal | yaf2oaf | 0.445 ± 0.185 |
| text_bert | contextual | oaf2yaf | -0.033 ± 0.000 |
| text_bert | contextual | word | -0.017 ± 0.000 |
| text_bert | contextual | yaf2oaf | -0.033 ± 0.000 |


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


# Error analysis: failure cases of the GMU fusion model

Most consistent errors (same mistake across seeds, then highest confidence), from each protocol.

| protocol | clip | speaker | word | true | predicted | seeds wrong | mean confidence | speech-only says |
|---|---|---|---|---|---|---|---|---|
| Word-disjoint (main) | OAF_make_ps | OAF | make | pleasant_surprise | happy | 1/3 | 0.95 | pleasant_surprise |
| Word-disjoint (main) | OAF_hit_ps | OAF | hit | pleasant_surprise | happy | 1/3 | 0.61 | pleasant_surprise |
| Word-disjoint (main) | OAF_bite_ps | OAF | bite | pleasant_surprise | happy | 1/3 | 0.53 | pleasant_surprise |
| Cross-speaker OAF->YAF | YAF_thought_happy | YAF | thought | happy | angry | 3/3 | 0.96 | angry |
| Cross-speaker OAF->YAF | YAF_which_happy | YAF | which | happy | angry | 3/3 | 0.96 | angry |
