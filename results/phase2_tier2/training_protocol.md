# Phase 2 Tier 2 Training Protocol

This artifact summarizes the model-training protocol used for the submitted Phase 2 Tier 2 compact-feature experiments.

## Data split
- Random seed: 42
- Held-out test fraction: 0.2
- Validation fraction within the train+validation pool: 0.2
- The train and validation partitions are used for model selection.
- The final model is refit on train+validation and evaluated once on the held-out test partition.

## Split counts
| split | total | Ia | non-Ia | Ia fraction |
| --- | --- | --- | --- | --- |
| train | 13644 | 3256 | 10388 | 0.23864 |
| validation | 3411 | 814 | 2597 | 0.23864 |
| trainval | 17055 | 4070 | 12985 | 0.23864 |
| test | 4264 | 1018 | 3246 | 0.238743 |
| all_rows | 21319 | 5088 | 16231 | 0.23866 |

## XGBoost protocol
- Objective: `binary:logistic`
- Evaluation metric during boosting: `logloss`
- Tree method: `hist`
- Maximum boosting rounds during model selection: 400
- Early stopping rounds: 30
- Candidate-selection metric: validation PR-AUC
- Final training rounds: selected candidate's best iteration from validation early stopping.
- Class imbalance handling: scale_pos_weight is computed as non-Ia count divided by Ia count in the current training partition
- Standardization: feature-wise z-score using the training partition mean and standard deviation; the same transform is applied to validation/test data

## Hyperparameter grid
| candidate | max_depth | eta | subsample | colsample_bytree | min_child_weight | lambda |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0.05 | 0.8 | 0.8 | 1.0 | 1.0 |
| 2 | 4 | 0.05 | 0.9 | 0.9 | 1.0 | 1.0 |
| 3 | 5 | 0.03 | 0.8 | 0.8 | 2.0 | 1.5 |

## Frozen compact baseline test metrics
| metric | value |
| --- | --- |
| F1 | 0.84423 |
| ROC-AUC | 0.976588 |
| PR-AUC | 0.927761 |

## Reproducibility note
The checked-in Tier 2 CSV/Markdown artifacts contain final metrics but not every selected per-run hyperparameter choice.
The source protocol above is exact; per-run selected candidate parameters and best iterations should be regenerated after installing XGBoost and rerunning the Tier 2 scripts.
