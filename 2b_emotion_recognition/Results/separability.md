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
