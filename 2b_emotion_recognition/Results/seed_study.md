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
