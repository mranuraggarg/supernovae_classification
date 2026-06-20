# Phase 2 Tier 4 Windowed PLAsTiCC Audit

This audit tests whether peak-centered windowing makes PLAsTiCC event windows closer to SPCC-like transient windows.

## Main window comparison

| window | records | obs p50 | active obs p50 | active fraction p50 | time span p50 | active span p50 | peak time p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 7848 | 136 | 14 | 0.0928571 | 902.595 | 124.817 | 452.824 |
| pm60 | 7848 | 24 | 10 | 0.454545 | 87.7895 | 54.8655 | 49.8229 |
| pm90 | 7848 | 33 | 11 | 0.363636 | 116.886 | 70.9467 | 65.7825 |
| pm120 | 7848 | 38 | 12 | 0.310345 | 137.67 | 78.0021 | 70.8238 |
| pm180 | 7848 | 46 | 12 | 0.267192 | 200.712 | 83.868 | 101.785 |

## Per-band median observation counts

| window | g p50 | r p50 | i p50 | z p50 | g zero | r zero | i zero | z zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 12 | 23 | 23 | 31 | 0 | 0 | 0 | 0 |
| pm60 | 3 | 5 | 5 | 5 | 805 | 534 | 344 | 78 |
| pm90 | 3 | 7 | 6 | 7 | 453 | 396 | 131 | 21 |
| pm120 | 4 | 8 | 7 | 8 | 290 | 344 | 80 | 6 |
| pm180 | 4 | 9 | 8 | 10 | 168 | 228 | 40 | 2 |

## Label counts

Original: {'non-Ia': 5535, 'Ia': 2313}

pm60: {'non-Ia': 5535, 'Ia': 2313}

pm90: {'non-Ia': 5535, 'Ia': 2313}

pm120: {'non-Ia': 5535, 'Ia': 2313}

pm180: {'non-Ia': 5535, 'Ia': 2313}

## Interpretation guide

- If windowing reduces PLAsTiCC time_span p50 from ~900 days to ~100-200 days, the window hypothesis is supported.
- If active_time_span also becomes closer to SPCC, then compact features are more likely to describe the transient rather than the full survey history.
- If many g/r/i/z zero-count events appear under a narrow window, the window is too aggressive.
- The best trial window is usually the smallest window that preserves reasonable per-band support.
