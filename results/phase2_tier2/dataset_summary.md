# Phase 2 Tier 2 Dataset Summary

This artifact summarizes the data table and fixed split used by the submitted Phase 2 Tier 2 compact-feature experiments.

## Sources
- Compact source CSV: `data/processed/phase2_tier1_compact_baseline.csv`
- Full Tier 1 source CSV: `data/processed/spcc_features_tier1.csv`

## Overall binary class balance
- Total objects: 21319
- Type Ia objects: 5088 (0.23866)
- Non-Ia objects: 16231 (0.76134)

## Original label balance
| label_name | count | fraction |
| --- | --- | --- |
| II | 12027 | 0.564145 |
| IIL | 425 | 0.0199353 |
| IIP | 189 | 0.00886533 |
| IIn | 789 | 0.0370092 |
| Ia | 5088 | 0.23866 |
| Ib | 1438 | 0.0674516 |
| Ibc | 259 | 0.0121488 |
| Ic | 1104 | 0.0517848 |

## Fixed split balance
| split | total | Ia | non-Ia | Ia fraction |
| --- | --- | --- | --- | --- |
| train | 13644 | 3256 | 10388 | 0.23864 |
| validation | 3411 | 814 | 2597 | 0.23864 |
| trainval | 17055 | 4070 | 12985 | 0.23864 |
| test | 4264 | 1018 | 3246 | 0.238743 |
| all_rows | 21319 | 5088 | 16231 | 0.23866 |

## Compact feature set
- Compact feature count: 16
- Compact features: z_peak_flux, r_mean_flux, peak_color_g_minus_r, i_peak_flux, peak_color_r_minus_i, peak_color_i_minus_z, g_mean_flux, r_peak_flux, z_std_flux, i_amplitude, i_std_flux, time_span, z_time_of_peak, i_time_of_peak, r_time_of_peak, r_std_flux

## Compact engineered feature ranges
| feature | min | median | max | mean | std |
| --- | --- | --- | --- | --- | --- |
| z_peak_flux | 0 | 1.49859 | 4.22585 | 1.53489 | 0.302897 |
| r_mean_flux | -0.382061 | 0.413542 | 3.32992 | 0.482159 | 0.3149 |
| peak_color_g_minus_r | -4.31109 | 0.0955305 | 5 | 0.101646 | 0.725676 |
| i_peak_flux | 0.403464 | 1.48487 | 4.01874 | 1.53447 | 0.314186 |
| peak_color_r_minus_i | -2.58647 | 0.155563 | 3.51914 | 0.218741 | 0.439189 |
| peak_color_i_minus_z | -5 | 0.010485 | 2.95866 | 0.00141847 | 0.368781 |
| g_mean_flux | -0.588372 | 0.194306 | 3.12205 | 0.260493 | 0.339634 |
| r_peak_flux | 0.406029 | 1.40875 | 4.13485 | 1.4513 | 0.341064 |
| z_std_flux | 0.296288 | 0.895127 | 3.68616 | 0.938495 | 0.291053 |
| i_amplitude | 0.751279 | 1.59868 | 4.01874 | 1.64218 | 0.283553 |
| i_std_flux | 0.253112 | 0.877328 | 3.50931 | 0.928252 | 0.300954 |
| time_span | 30 | 125.852 | 174.907 | 122.232 | 33.7313 |
| z_time_of_peak | 0 | 58.934 | 174.907 | 56.3001 | 27.4712 |
| i_time_of_peak | 0 | 54 | 157.864 | 51.5267 | 25.8584 |
| r_time_of_peak | 0 | 52.856 | 171.793 | 47.3572 | 23.6088 |
| r_std_flux | 0.188431 | 0.786718 | 3.59128 | 0.841297 | 0.324355 |

## Dataset context ranges
| feature | min | median | max | mean | std |
| --- | --- | --- | --- | --- | --- |
| sim_z | 0.023 | 0.6655 | 1.1 | 0.667536 | 0.236395 |
| observation_count | 16 | 101 | 161 | 96.2925 | 35.3594 |
| observed_band_count | 4 | 4 | 4 | 4 | 0 |
| time_span | 30 | 125.852 | 174.907 | 122.232 | 33.7313 |
| total_snr | 1.39777 | 2.33495 | 3.70385 | 2.37781 | 0.326774 |

Note: the feature ranges are computed from engineered features in the processed Tier 1 table. They should be described as feature ranges, not as raw survey magnitude limits.
