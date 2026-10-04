# Phase 2 Tier 3 Feature Importance Consistency

Cross-method comparison of XGBoost gain, permutation importance, SHAP, and Tier-2 ablation ranks.

| feature | group | gain_rank | perm_rank | shap_rank | ablation_rank | consensus_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r_mean_flux | brightness | 6 | 1 | 1 | 5 | 2.367 |
| time_span | temporal | 14 | 6 | 5 | 1 | 1.438 |
| z_peak_flux | brightness | 4 | 2 | 2 | 7 | 1.393 |
| i_peak_flux | brightness | 1 | 9 | 6 | 14 | 1.349 |
| peak_color_r_minus_i | color | 3 | 8 | 11 | 2 | 1.049 |
| i_std_flux | variability | 13 | 3 | 4 | 12 | 0.744 |
| peak_color_g_minus_r | color | 5 | 4 | 7 | 8 | 0.718 |
| g_mean_flux | brightness | 7 | 7 | 3 | 11 | 0.710 |
| i_amplitude | variability | 2 | 15 | 16 | 13 | 0.706 |
| peak_color_i_minus_z | color | 9 | 11 | 13 | 3 | 0.612 |

## Rank correlations

- gain vs permutation: 0.312
- gain vs shap: 0.194
- gain vs ablation: -0.094
- permutation vs shap: 0.806
- permutation vs ablation: 0.168
- shap vs ablation: 0.153

Plot: `plots/phase2_tier3/phase2_tier3_importance_consistency.png`
