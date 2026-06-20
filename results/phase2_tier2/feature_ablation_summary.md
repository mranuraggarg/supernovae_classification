# Phase 2 Tier 2 Feature Ablation

Leave-one-feature-out retraining on the frozen compact Tier 1 feature set.

| feature_removed | feature_group | num_features | f1 | roc_auc | pr_auc | delta_f1 | delta_pr_auc | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_span | temporal | 15 | 0.810278 | 0.969026 | 0.909497 | -0.033952 | -0.018264 | 1 |
| z_std_flux | variability | 15 | 0.836985 | 0.975219 | 0.924391 | -0.007245 | -0.003370 | 2 |
| z_peak_flux | brightness | 15 | 0.837577 | 0.975424 | 0.925365 | -0.006653 | -0.002396 | 3 |
| i_time_of_peak | temporal | 15 | 0.838371 | 0.974859 | 0.924749 | -0.005859 | -0.003011 | 4 |
| peak_color_i_minus_z | color | 15 | 0.839247 | 0.975226 | 0.920827 | -0.004983 | -0.006934 | 5 |
| peak_color_r_minus_i | color | 15 | 0.839387 | 0.974582 | 0.920446 | -0.004843 | -0.007315 | 6 |
| r_time_of_peak | temporal | 15 | 0.839876 | 0.974767 | 0.920199 | -0.004353 | -0.007562 | 7 |
| r_mean_flux | brightness | 15 | 0.840195 | 0.973784 | 0.921118 | -0.004035 | -0.006643 | 8 |
| i_amplitude | variability | 15 | 0.840247 | 0.976183 | 0.927419 | -0.003983 | -0.000342 | 9 |
| g_mean_flux | brightness | 15 | 0.841221 | 0.975016 | 0.924393 | -0.003009 | -0.003368 | 10 |
| z_time_of_peak | temporal | 15 | 0.841410 | 0.974919 | 0.923102 | -0.002820 | -0.004659 | 11 |
| peak_color_g_minus_r | color | 15 | 0.841736 | 0.975233 | 0.921381 | -0.002494 | -0.006380 | 12 |
| i_std_flux | variability | 15 | 0.841920 | 0.975728 | 0.923934 | -0.002310 | -0.003827 | 13 |
| r_std_flux | variability | 15 | 0.844759 | 0.976695 | 0.928654 | +0.000529 | +0.000893 | 14 |
| i_peak_flux | brightness | 15 | 0.845270 | 0.976774 | 0.926894 | +0.001040 | -0.000867 | 15 |
| r_peak_flux | brightness | 15 | 0.845579 | 0.976656 | 0.928603 | +0.001349 | +0.000842 | 16 |
