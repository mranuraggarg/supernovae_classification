# Phase 2 Tier 4 Feature-Definition Audit

This audit compares the effective observation-level inputs used by the shared compact-feature builder.

SPCC records: 21319
PLAsTiCC records: 7848

## Label counts

SPCC: {'II': 12027, 'Ib': 1438, 'IIn': 789, 'Ia': 5088, 'IIL': 425, 'Ic': 1104, 'IIP': 189, 'Ibc': 259}

PLAsTiCC: {'non-Ia': 5535, 'Ia': 2313}

## Core observation-window quantities

| quantity | survey | min | p25 | p50 | p75 | p95 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_obs_count / SPCC | 16 | 67 | 101 | 123 | 149 | 161 |
| raw_obs_count / PLAsTiCC | 29 | 80 | 87 | 180 | 232 | 232 |
| active_obs_count / SPCC | 2 | 12 | 21 | 37 | 78 | 133 |
| active_obs_count / PLAsTiCC | 0 | 5 | 12 | 26 | 69 | 232 |
| active_fraction / SPCC | 0.0134228 | 0.151786 | 0.252033 | 0.404375 | 0.708386 | 1 |
| active_fraction / PLAsTiCC | 0 | 0.0491071 | 0.117647 | 0.229885 | 0.546242 | 1 |
| raw_time_span / SPCC | 30 | 95.105 | 125.852 | 146.984 | 173.824 | 174.907 |
| raw_time_span / PLAsTiCC | 748.808 | 853.834 | 887.598 | 1070 | 1084.89 | 1092.85 |
| active_time_span / SPCC | 0.004 | 40.043 | 66.887 | 94.0485 | 127.926 | 174.883 |
| active_time_span / PLAsTiCC | 0 | 60.8121 | 121.681 | 462.841 | 905.772 | 1092.84 |

## Per-band active-count medians

| band | SPCC active p50 | PLAsTiCC active p50 | SPCC raw p50 | PLAsTiCC raw p50 |
| --- | ---: | ---: | ---: | ---: |
| g | 1.0 | 1.0 | 27.0 | 12.0 |
| r | 6.0 | 4.0 | 26.0 | 23.0 |
| i | 6.0 | 3.0 | 25.0 | 23.0 |
| z | 7.0 | 3.0 | 24.0 | 31.0 |

## Peak-time medians

| feature | SPCC p50 | PLAsTiCC p50 |
| --- | ---: | ---: |
| g_peak_time | 49.85899999999674 | 464.84855000000243 |
| g_active_peak_time | 46.85199999999895 | 443.7894000000015 |
| r_peak_time | 52.85599999999977 | 458.8195999999989 |
| r_active_peak_time | 52.870999999999185 | 452.7924999999959 |
| i_peak_time | 54.0 | 439.71299999999974 |
| i_active_peak_time | 56.47849999999744 | 427.8628000000026 |
| z_peak_time | 58.93399999999383 | 433.9658000000054 |
| z_active_peak_time | 58.995999999999185 | 428.9366500000033 |

## PLAsTiCC detected-flag vs SNR-active rule

| quantity | min | p25 | p50 | p75 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| detected_obs_count | 0 | 4 | 9 | 26 | 82 | 232 |
| detected_fraction | 0 | 0.038961 | 0.0947368 | 0.212853 | 0.926724 | 1 |
| snr_detected_agreement_fraction | 0 | 0.926724 | 0.966292 | 0.988235 | 1 | 1 |
| detected_not_snr_count | 0 | 0 | 0 | 1 | 44 | 232 |
| snr_not_detected_count | 0 | 1 | 2 | 5 | 16 | 79 |

## Sparse active-support flags

SPCC: {'g_zero_active': 9397, 'g_lt2_active': 12656, 'i_lt2_active': 931, 'z_lt2_active': 1323, 'r_lt2_active': 1276, 'r_zero_active': 286, 'z_zero_active': 343, 'i_zero_active': 113}

PLAsTiCC: {'g_zero_active': 2278, 'g_lt2_active': 3962, 'r_zero_active': 1209, 'r_lt2_active': 2214, 'i_zero_active': 1341, 'i_lt2_active': 2428, 'z_lt2_active': 2486, 'z_zero_active': 1348}

## Interpretation guide

- Large raw/active time-span differences imply that the same compact feature formulas are operating over different effective event windows.
- Low PLAsTiCC detected/SNR agreement implies that ignoring the `detected` flag may change the effective PLAsTiCC event definition.
- Many zero-active or <2-active band flags imply that color and peak-time features may depend heavily on fallback behaviour.
- Large differences between raw peak-time and active peak-time summaries imply that the active-window rule changes the phase being measured.
