# Phase 2 Tier 2 Minimal Core

Reduced-core compact subsets chosen from feature ablation plus Tier 1 importance context.

| subset_name | feature_count | selection_rule | feature_list | f1 | pr_auc | delta_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| top_5 | 5 | ablation-plus-importance with minimum one feature per physical family when possible | z_peak_flux, peak_color_i_minus_z, z_std_flux, time_span, peak_color_r_minus_i | 0.706973 | 0.745903 | -0.137257 |
| top_8 | 8 | ablation-plus-importance with minimum one feature per physical family when possible | z_peak_flux, peak_color_i_minus_z, z_std_flux, time_span, peak_color_r_minus_i, r_time_of_peak, i_time_of_peak, r_mean_flux | 0.796399 | 0.874361 | -0.047830 |
| top_10 | 10 | ablation-plus-importance with minimum one feature per physical family when possible | z_peak_flux, peak_color_i_minus_z, z_std_flux, time_span, peak_color_r_minus_i, r_time_of_peak, i_time_of_peak, r_mean_flux, peak_color_g_minus_r, z_time_of_peak | 0.838509 | 0.916900 | -0.005721 |
