# Accuracy (%) on noisy test speech (word-disjoint test set, white noise, mean ± std over seeds)

| Model | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| Speech only (WavLM) (10 seeds) | 99.9 ± 0.1 | 99.5 ± 0.7 | 92.3 ± 4.7 | 83.1 ± 7.7 | 60.5 ± 11.6 |
| Fusion: GMU (10 seeds) | 99.9 ± 0.2 | 99.6 ± 0.7 | 93.7 ± 3.3 | 86.4 ± 6.0 | 69.2 ± 8.5 |
| GMU, text replaced by zeros (10 seeds) | 99.8 ± 0.2 | 99.3 ± 1.1 | 89.4 ± 8.0 | 77.9 ± 12.1 | 55.6 ± 13.1 |
| GMU, transcript of a different word (10 seeds) | 100.0 ± 0.1 | 99.4 ± 0.9 | 93.0 ± 5.8 | 83.6 ± 8.5 | 65.5 ± 11.9 |
| Fusion: concat (10 seeds) | 99.8 ± 0.3 | 99.2 ± 0.7 | 85.2 ± 10.7 | 69.8 ± 15.1 | 45.6 ± 15.6 |

Difference to speech-only in points (Welch's t-test p-value):

| Model | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| Fusion: GMU | -0.0 (p=0.573) | +0.1 (p=0.707) | +1.4 (p=0.448) | +3.3 (p=0.303) | +8.7 (p=0.071) |
| GMU, text replaced by zeros | -0.1 (p=0.395) | -0.2 (p=0.698) | -2.9 (p=0.333) | -5.2 (p=0.269) | -4.8 (p=0.394) |
| GMU, transcript of a different word | +0.0 (p=0.356) | -0.0 (p=0.896) | +0.7 (p=0.758) | +0.5 (p=0.887) | +5.0 (p=0.350) |
| Fusion: concat | -0.1 (p=0.185) | -0.3 (p=0.343) | -7.1 (p=0.076) | -13.3 (p=0.027) | -14.8 (p=0.027) |

GMU speech gate (mean z): clean: 0.715, 20: 0.717, 10: 0.724, 5: 0.730, 0: 0.737
