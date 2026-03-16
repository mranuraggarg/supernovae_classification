# Phase 2 Tier 3 Cross-Validation Stability

Repeated evaluation of the compact XGBoost baseline under alternate resampling protocols.

| protocol | runs | f1_mean | f1_std | roc_auc_mean | pr_auc_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| kfold_cv | 5 | 0.841132 | 0.006213 | 0.975170 | 0.922780 |
| random_split | 5 | 0.845071 | 0.002608 | 0.975574 | 0.923226 |

Plot: `plots/phase2_tier3/phase2_tier3_cv_stability.png`
