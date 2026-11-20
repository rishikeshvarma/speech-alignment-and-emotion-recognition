# Error analysis: failure cases of the GMU fusion model

Most consistent errors (same mistake across seeds, then highest confidence), from each protocol.

| protocol | clip | speaker | word | true | predicted | seeds wrong | mean confidence | speech-only says |
|---|---|---|---|---|---|---|---|---|
| Word-disjoint (main) | OAF_make_ps | OAF | make | pleasant_surprise | happy | 1/3 | 0.95 | pleasant_surprise |
| Word-disjoint (main) | OAF_hit_ps | OAF | hit | pleasant_surprise | happy | 1/3 | 0.61 | pleasant_surprise |
| Word-disjoint (main) | OAF_bite_ps | OAF | bite | pleasant_surprise | happy | 1/3 | 0.53 | pleasant_surprise |
| Cross-speaker OAF->YAF | YAF_thought_happy | YAF | thought | happy | angry | 3/3 | 0.96 | angry |
| Cross-speaker OAF->YAF | YAF_which_happy | YAF | which | happy | angry | 3/3 | 0.96 | angry |
