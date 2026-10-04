# Phase 2 Tier 4 PCA Alignment Trial

This trial evaluates whether SPCC and PLAsTiCC are separated mainly by a coordinate-system mismatch.

## Separate PCA comparison

| component | SPCC var | PLAsTiCC var | cosine | abs cosine |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.463214 | 0.395334 | 0.995608 | 0.995608 |
| 2 | 0.154888 | 0.195560 | 0.923250 | 0.923250 |
| 3 | 0.094029 | 0.105248 | 0.532370 | 0.532370 |
| 4 | 0.075124 | 0.078637 | -0.380168 | 0.380168 |
| 5 | 0.068319 | 0.067281 | 0.740438 | 0.740438 |
| 6 | 0.043506 | 0.045678 | -0.397833 | 0.397833 |
| 7 | 0.025305 | 0.027440 | -0.120722 | 0.120722 |
| 8 | 0.023195 | 0.021714 | 0.239379 | 0.239379 |
| 9 | 0.016153 | 0.019591 | 0.097758 | 0.097758 |
| 10 | 0.014174 | 0.018780 | 0.021110 | 0.021110 |

## Cross projection
- SPCC PCA, SPCC train -> SPCC test F1: 0.815526
- SPCC PCA, SPCC train -> PLAsTiCC test F1: 0.521472

## Shared PCA
- Shared PCA, SPCC train -> SPCC test F1: 0.824879
- Shared PCA, SPCC train -> PLAsTiCC test F1: 0.518207
