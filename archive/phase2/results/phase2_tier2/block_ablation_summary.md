# Phase 2 Tier 2 Block Ablation

Leave-one-block-out retraining across brightness, color, variability, and temporal families.

| block_removed | remaining_features | f1 | roc_auc | pr_auc | delta_f1 | delta_pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| temporal | 12 | 0.797634 | 0.958770 | 0.872868 | -0.046596 | -0.054893 |
| brightness | 11 | 0.819149 | 0.968469 | 0.907833 | -0.025081 | -0.019928 |
| color | 13 | 0.831820 | 0.972364 | 0.907942 | -0.012410 | -0.019819 |
| variability | 12 | 0.831874 | 0.973228 | 0.918558 | -0.012356 | -0.009203 |
