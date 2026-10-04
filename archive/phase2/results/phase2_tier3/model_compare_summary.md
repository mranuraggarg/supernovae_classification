# Phase 2 Tier 3 Model Robustness

Compact Tier-1 features evaluated with multiple classifier families on the frozen split.

| model | f1 | roc_auc | pr_auc | delta_f1 | top permutation features |
| --- | ---: | ---: | ---: | ---: | --- |
| XGBoost | 0.844230 | 0.976588 | 0.927761 | +0.000000 | r_mean_flux, z_peak_flux, i_std_flux, peak_color_g_minus_r, r_peak_flux |
| Random Forest | 0.823834 | 0.973779 | 0.923555 | -0.020396 | z_peak_flux, i_peak_flux, peak_color_g_minus_r, r_mean_flux, peak_color_r_minus_i |
| Logistic Regression | 0.713548 | 0.894801 | 0.607943 | -0.130682 | peak_color_r_minus_i, r_peak_flux, i_std_flux, peak_color_i_minus_z, z_peak_flux |
| Support Vector Machine | 0.835271 | 0.970340 | 0.885895 | -0.008959 | i_std_flux, r_mean_flux, r_std_flux, g_mean_flux, r_peak_flux |

Best PR-AUC model: XGBoost (0.927761).
Plot: `plots/phase2_tier3/phase2_tier3_model_compare.png`
