# Alignment analysis

## Runs

| experiment | aligned | words | words_without_pron | phones | mean_phone_ms | median_phone_ms | phones_at_30ms | speech_loglik | phone_dur_dev |
|---|---|---|---|---|---|---|---|---|---|
| arpa_base | 6/6 | 244 | 5 | 1033 | 75.5 | 70.0 | 80 (7.8%) | -47.68 | 3.0 |
| arpa_g2p | 6/6 | 244 | 0 | 1063 | 75.8 | 70.0 | 83 (7.8%) | -47.74 | 2.96 |
| arpa_g2p_adapted | 6/6 | 244 | 0 | 1063 | 75.8 | 70.0 | 83 (7.8%) | -47.74 | 2.97 |
| arpa_g2p_finetune | 6/6 | 244 | 0 | 1063 | 75.8 | 70.0 | 84 (7.9%) | -47.74 | 2.96 |
| isle_variants | 3/6 | 15 | 0 | 38 | 136.6 | 120.0 | 0 (0.0%) | -49.14 | 1.4 |
| mfa_g2p | 6/6 | 244 | 0 | 1045 | 82.1 | 70.0 | 98 (9.4%) | -44.77 | 3.78 |


## Word-boundary agreement with `arpa_g2p`

| compared_to_arpa_g2p | speaker | boundaries | mean_ms | median_ms | max_ms | within_20ms | within_50ms |
|---|---|---|---|---|---|---|---|
| arpa_base | all | 488 | 0.6 | 0.0 | 20 | 99% | 100% |
| arpa_base | f2b | 458 | 0.7 | 0.0 | 20 | 99% | 100% |
| arpa_base | isle | 30 | 0.0 | 0.0 | 0 | 100% | 100% |
| arpa_g2p_adapted | all | 488 | 0.1 | 0.0 | 10 | 100% | 100% |
| arpa_g2p_adapted | f2b | 458 | 0.1 | 0.0 | 10 | 100% | 100% |
| arpa_g2p_adapted | isle | 30 | 0.0 | 0.0 | 0 | 100% | 100% |
| arpa_g2p_finetune | all | 488 | 0.1 | 0.0 | 10 | 100% | 100% |
| arpa_g2p_finetune | f2b | 458 | 0.1 | 0.0 | 10 | 100% | 100% |
| arpa_g2p_finetune | isle | 30 | 0.0 | 0.0 | 0 | 100% | 100% |
| isle_variants | all | 30 | 6.3 | 0.0 | 160 | 93% | 97% |
| isle_variants | isle | 30 | 6.3 | 0.0 | 160 | 93% | 97% |
| mfa_g2p | all | 488 | 18.4 | 10.0 | 2460 | 90% | 96% |
| mfa_g2p | f2b | 458 | 8.3 | 10.0 | 250 | 92% | 98% |
| mfa_g2p | isle | 30 | 172.8 | 10.0 | 2460 | 63% | 77% |


## Red flags (counts)

| experiment | type | count |
|---|---|---|
| arpa_base | no pronunciation (spn) | 5 |
| arpa_base | pause >= 300 ms | 13 |
| arpa_base | phone at 30 ms minimum | 80 |
| arpa_base | word < 80 ms | 3 |
| arpa_g2p | pause >= 300 ms | 13 |
| arpa_g2p | phone at 30 ms minimum | 83 |
| arpa_g2p | word < 80 ms | 2 |
| arpa_g2p_adapted | pause >= 300 ms | 13 |
| arpa_g2p_adapted | phone at 30 ms minimum | 83 |
| arpa_g2p_adapted | word < 80 ms | 3 |
| arpa_g2p_finetune | pause >= 300 ms | 13 |
| arpa_g2p_finetune | phone at 30 ms minimum | 84 |
| arpa_g2p_finetune | word < 80 ms | 3 |
| mfa_g2p | pause >= 300 ms | 11 |
| mfa_g2p | phone at 30 ms minimum | 98 |
| mfa_g2p | word < 80 ms | 6 |


## ISLE pronunciation check (variant chosen by the aligner)

| utterance | word | canonical | aligner chose | verdict | other options |
|---|---|---|---|---|---|
| D02_01_sprt1 | i | AY1 | AY1 | canonical |  |
| D02_01_sprt1 | said | S EH1 D | S EH1 D | canonical | S EH1 T |
| D02_01_sprt1 | white | W AY1 T | W AY1 T | canonical | V AY1 T / W AY1 D |
| D02_01_sprt1 | not | N AA1 T | N AA1 T | canonical | N AO1 T |
| D02_01_sprt1 | bait | B EY1 T | B EY1 T | canonical | B EH1 T / B EY1 D |
| D02_02_sprt1 | i | AY1 | AY1 | canonical |  |
| D02_02_sprt1 | said | S EH1 D | S EH1 D | canonical | S EH1 T |
| D02_02_sprt1 | new | N UW1 | N Y UW1 | LEARNER VARIANT | N Y UW1 |
| D02_02_sprt1 | not | N AA1 T | N AA1 T | canonical | N AO1 T |
| D02_02_sprt1 | no | N OW1 | N OW1 | canonical | N AO1 |
| D02_03_sprt1 | i | AY1 | AY1 | canonical |  |
| D02_03_sprt1 | said | S EH1 D | S EH1 D | canonical | S EH1 T |
| D02_03_sprt1 | bad | B AE1 D | B EH1 T | LEARNER VARIANT | B AE1 T / B EH1 D / B EH1 T |
| D02_03_sprt1 | not | N AA1 T | N AA1 T | canonical | N AO1 T |
| D02_03_sprt1 | bed | B EH1 D | B EH1 D | canonical | B EH1 T / B AE1 D |


## Focus words (G2P words and ISLE minimal pairs)

| experiment | utterance | word | start | dur_ms | phones (ms) |
|---|---|---|---|---|---|
| arpa_base | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 0.66 | 510 | B(40) AE1(220) D(250) |
| arpa_g2p | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 0.66 | 510 | B(40) AE1(220) D(250) |
| arpa_g2p_adapted | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 0.66 | 510 | B(40) AE1(220) D(250) |
| arpa_g2p_finetune | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 0.66 | 510 | B(40) AE1(220) D(250) |
| isle_variants | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 0.66 | 510 | B(40) EH1(220) T(250) |
| mfa_g2p | ISLE_SESS0131_BLOCKD02_03_sprt1 | bad | 0.66 | 510 | b(40) æ(220) d(250) |
| arpa_base | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 1.8 | 450 | B(50) EY1(200) T(200) |
| arpa_g2p | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 1.8 | 450 | B(50) EY1(200) T(200) |
| arpa_g2p_adapted | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 1.8 | 450 | B(50) EY1(200) T(200) |
| arpa_g2p_finetune | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 1.8 | 450 | B(50) EY1(200) T(200) |
| isle_variants | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 1.8 | 450 | B(50) EY1(200) T(200) |
| mfa_g2p | ISLE_SESS0131_BLOCKD02_01_sprt1 | bait | 1.7 | 2425 | b(150) ej(200) t(2075) |
| arpa_base | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 1.49 | 550 | B(120) EH1(180) D(250) |
| arpa_g2p | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 1.49 | 550 | B(120) EH1(180) D(250) |
| arpa_g2p_adapted | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 1.49 | 550 | B(120) EH1(180) D(250) |
| arpa_g2p_finetune | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 1.49 | 550 | B(120) EH1(180) D(250) |
| isle_variants | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 1.49 | 550 | B(120) EH1(190) D(240) |
| mfa_g2p | ISLE_SESS0131_BLOCKD02_03_sprt1 | bed | 1.47 | 3030 | b(140) ɛ(190) d(2700) |
| arpa_base | F2BJRLP1 | dukakis | 15.45 | 550 | spn(550) |
| arpa_g2p | F2BJRLP1 | dukakis | 15.44 | 570 | D(60) UW0(40) K(100) AA1(110) K(90) IH0(50) S(120) |
| arpa_g2p_adapted | F2BJRLP1 | dukakis | 15.44 | 570 | D(60) UW0(40) K(100) AA1(110) K(90) IH0(50) S(120) |
| arpa_g2p_finetune | F2BJRLP1 | dukakis | 15.44 | 570 | D(60) UW0(40) K(100) AA1(110) K(90) IH0(50) S(120) |
| mfa_g2p | F2BJRLP1 | dukakis | 15.44 | 560 | d(60) ɐ(40) kʰ(100) ɑː(120) c(80) i(40) z(120) |
| arpa_base | F2BJRLP2 | dukakis | 2.73 | 620 | spn(620) |
| arpa_g2p | F2BJRLP2 | dukakis | 2.72 | 640 | D(70) UW1(50) K(110) AA1(120) K(80) IH0(80) S(130) |
| arpa_g2p_adapted | F2BJRLP2 | dukakis | 2.72 | 640 | D(70) UW1(50) K(110) AA1(120) K(80) IH0(80) S(130) |
| arpa_g2p_finetune | F2BJRLP2 | dukakis | 2.72 | 640 | D(70) UW1(50) K(110) AA1(120) K(80) IH0(80) S(130) |
| mfa_g2p | F2BJRLP2 | dukakis | 2.72 | 640 | d(60) ɐ(50) k(110) ə(130) c(80) ɪ(80) s(130) |
| arpa_base | F2BJRLP3 | judgeships | 17.09 | 640 | JH(110) AH1(100) JH(50) SH(110) IH2(80) P(90) S(100) |
| arpa_g2p | F2BJRLP3 | judgeships | 17.09 | 640 | JH(110) AH1(100) JH(50) SH(110) IH2(80) P(90) S(100) |
| arpa_g2p_adapted | F2BJRLP3 | judgeships | 17.09 | 640 | JH(110) AH1(100) JH(50) SH(110) IH2(80) P(90) S(100) |
| arpa_g2p_finetune | F2BJRLP3 | judgeships | 17.09 | 640 | JH(110) AH1(100) JH(50) SH(110) IH2(80) P(90) S(100) |
| mfa_g2p | F2BJRLP3 | judgeships | 17.09 | 650 | dʒ(110) ɐ(100) dʒ(80) ʃ(80) ɪ(90) p(70) s(120) |
| arpa_base | F2BJRLP2 | maffy | 23.92 | 420 | spn(420) |
| arpa_g2p | F2BJRLP2 | maffy | 23.94 | 390 | M(50) AE1(120) F(100) IY0(120) |
| arpa_g2p_adapted | F2BJRLP2 | maffy | 23.94 | 390 | M(50) AE1(120) F(100) IY0(120) |
| arpa_g2p_finetune | F2BJRLP2 | maffy | 23.94 | 390 | M(50) AE1(120) F(100) IY0(120) |
| mfa_g2p | F2BJRLP2 | maffy | 23.94 | 390 | m(40) æ(140) fʲ(90) i(120) |
| arpa_base | F2BJRLP1 | margo | 21.44 | 310 | M(60) AA1(30) R(80) G(60) OW0(80) |
| arpa_g2p | F2BJRLP1 | margo | 21.44 | 300 | M(60) AA1(30) R(80) G(60) OW0(70) |
| arpa_g2p_adapted | F2BJRLP1 | margo | 21.44 | 300 | M(60) AA1(30) R(80) G(60) OW0(70) |
| arpa_g2p_finetune | F2BJRLP1 | margo | 21.44 | 300 | M(60) AA1(30) R(80) G(60) OW0(70) |
| mfa_g2p | F2BJRLP1 | margo | 21.44 | 300 | m(60) ɑ(40) ɹ(80) ɡ(60) ow(60) |
| arpa_base | F2BJRLP1 | melnicove | 21.75 | 530 | spn(530) |
| arpa_g2p | F2BJRLP1 | melnicove | 21.74 | 560 | M(60) EH1(40) L(60) N(70) IH0(50) K(100) OW2(90) V(90) |
| arpa_g2p_adapted | F2BJRLP1 | melnicove | 21.74 | 560 | M(60) EH1(40) L(60) N(70) IH0(50) K(100) OW2(90) V(90) |
| arpa_g2p_finetune | F2BJRLP1 | melnicove | 21.74 | 560 | M(60) EH1(40) L(60) N(70) IH0(50) K(100) OW2(90) V(90) |
| mfa_g2p | F2BJRLP1 | melnicove | 21.74 | 550 | m(60) ɛ(30) ɫ(60) n(80) ə(50) kʰ(110) ow(80) v(80) |
| arpa_base | F2BJRLP3 | new | 16.93 | 160 | N(60) Y(50) UW1(50) |
| arpa_g2p | F2BJRLP3 | new | 16.93 | 160 | N(60) Y(50) UW1(50) |
| arpa_g2p_adapted | F2BJRLP3 | new | 16.93 | 160 | N(60) Y(50) UW1(50) |
| arpa_g2p_finetune | F2BJRLP3 | new | 16.93 | 160 | N(60) Y(50) UW1(50) |
| mfa_g2p | F2BJRLP3 | new | 16.93 | 160 | ɲ(80) ʉː(80) |
| arpa_base | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 1.01 | 410 | N(100) Y(80) UW1(230) |
| arpa_g2p | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 1.01 | 410 | N(100) Y(80) UW1(230) |
| arpa_g2p_adapted | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 1.01 | 410 | N(100) Y(80) UW1(230) |
| arpa_g2p_finetune | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 1.01 | 410 | N(100) Y(80) UW1(230) |
| isle_variants | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 1.01 | 410 | N(100) Y(70) UW1(240) |
| mfa_g2p | ISLE_SESS0131_BLOCKD02_02_sprt1 | new | 1.01 | 280 | ɲ(180) ʉː(100) |
| arpa_base | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 1.75 | 470 | N(120) OW1(350) |
| arpa_g2p | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 1.75 | 470 | N(120) OW1(350) |
| arpa_g2p_adapted | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 1.75 | 470 | N(120) OW1(350) |
| arpa_g2p_finetune | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 1.75 | 470 | N(120) OW1(350) |
| isle_variants | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 1.75 | 490 | N(120) OW1(370) |
| mfa_g2p | ISLE_SESS0131_BLOCKD02_02_sprt1 | no | 1.75 | 500 | n(120) ow(380) |
| arpa_base | F2BJRLP2 | politicize | 5.66 | 870 | spn(870) |
| arpa_g2p | F2BJRLP2 | politicize | 5.65 | 860 | P(100) AH0(40) L(90) IH1(50) T(40) IH0(50) S(130) AY2(240) Z(120) |
| arpa_g2p_adapted | F2BJRLP2 | politicize | 5.65 | 860 | P(100) AH0(40) L(90) IH1(50) T(40) IH0(50) S(130) AY2(240) Z(120) |
| arpa_g2p_finetune | F2BJRLP2 | politicize | 5.65 | 860 | P(100) AH0(40) L(90) IH1(50) T(40) IH0(50) S(130) AY2(240) Z(120) |
| mfa_g2p | F2BJRLP2 | politicize | 5.65 | 870 | pʰ(110) ə(40) ʎ(100) ɪ(30) tʲ(30) ɪ(60) s(130) aj(260) z(110) |
| arpa_base | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 0.92 | 410 | W(100) AY1(150) T(160) |
| arpa_g2p | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 0.92 | 410 | W(100) AY1(150) T(160) |
| arpa_g2p_adapted | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 0.92 | 410 | W(100) AY1(150) T(160) |
| arpa_g2p_finetune | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 0.92 | 410 | W(100) AY1(150) T(160) |
| isle_variants | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 0.92 | 570 | W(110) AY1(140) T(320) |
| mfa_g2p | ISLE_SESS0131_BLOCKD02_01_sprt1 | white | 0.91 | 580 | w(80) aj(190) t(310) |
