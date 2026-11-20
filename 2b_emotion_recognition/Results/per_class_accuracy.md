# Per-emotion test accuracy (recall, %), mean over seeds (and both directions for cross-speaker)

| Model | Protocol | angry | disgust | fear | happy | neutral | pl. surprise | sad |
|---|---|---|---|---|---|---|---|---|
| Speech only | word-disjoint | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.4 | 100.0 |
| Speech only | cross-speaker | 6.7 | 81.7 | 82.2 | 15.0 | 87.8 | 19.4 | 88.9 |
| Text only | word-disjoint | 0.0 | 14.4 | 13.3 | 28.9 | 0.0 | 2.2 | 41.1 |
| Text only | cross-speaker | 14.4 | 0.0 | 0.0 | 62.2 | 0.0 | 23.3 | 0.0 |
| Speech + text (GMU fusion) | word-disjoint | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 98.3 | 100.0 |
| Speech + text (GMU fusion) | cross-speaker | 20.0 | 70.6 | 91.7 | 3.3 | 78.9 | 27.8 | 75.0 |
