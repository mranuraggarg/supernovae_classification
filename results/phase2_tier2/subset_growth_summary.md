# Phase 2 Tier 2 Subset Growth

Cumulative compact-feature growth experiments plus single-family reference runs.

| subset_name | feature_count | included_blocks | f1 | roc_auc | pr_auc | delta_f1 | delta_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brightness_only | 5 | brightness | 0.750000 | 0.941865 | 0.826434 | -0.094230 | -0.101327 |
| color_only | 3 | color | 0.621033 | 0.840401 | 0.581675 | -0.223197 | -0.346085 |
| temporal_only | 4 | temporal | 0.533795 | 0.781509 | 0.495550 | -0.310434 | -0.432211 |
| brightness_plus_color | 8 | brightness,color | 0.768272 | 0.949428 | 0.849357 | -0.075958 | -0.078404 |
| brightness_plus_color_plus_variability | 12 | brightness,color,variability | 0.791579 | 0.957906 | 0.870509 | -0.052651 | -0.057252 |
| full_compact | 16 | brightness,color,variability,temporal | 0.842291 | 0.976513 | 0.928204 | -0.001939 | +0.000443 |
