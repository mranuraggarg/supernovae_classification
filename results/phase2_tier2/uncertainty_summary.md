# Phase 2 Tier 2 Uncertainty Evaluation

Repeated stratified train/validation/test splits for the compact 16-feature XGBoost baseline.

## Protocol
- Seeds: 11, 22, 33, 44, 55
- Test split: 0.2
- Validation split within train+validation: 0.2
- Candidate-selection metric: validation PR-AUC
- Maximum boosting rounds: 400
- Early stopping rounds: 30

## Test metric summary
| metric | mean | std | min | max |
| --- | --- | --- | --- | --- |
| accuracy | 0.916135 | 0.002229 | 0.913696 | 0.918621 |
| precision | 0.767024 | 0.005617 | 0.759171 | 0.773431 |
| recall | 0.931827 | 0.008720 | 0.917485 | 0.941061 |
| f1 | 0.841402 | 0.004198 | 0.837293 | 0.846290 |
| roc_auc | 0.976008 | 0.001977 | 0.974092 | 0.978805 |
| pr_auc | 0.925644 | 0.005963 | 0.917162 | 0.933396 |

## Per-seed test metrics
| seed | selected_candidate | best_iteration | F1 | ROC-AUC | PR-AUC | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 3 | 400 | 0.846290 | 0.978805 | 0.933396 | 0.768860 | 0.941061 |
| 22 | 2 | 400 | 0.845434 | 0.975350 | 0.924992 | 0.773431 | 0.932220 |
| 33 | 3 | 400 | 0.839965 | 0.977252 | 0.928434 | 0.763666 | 0.933202 |
| 44 | 3 | 400 | 0.838028 | 0.974092 | 0.917162 | 0.759171 | 0.935167 |
| 55 | 3 | 400 | 0.837293 | 0.974541 | 0.924235 | 0.769992 | 0.917485 |

The submitted fixed-split result should remain the primary result; these repeated-split values quantify stability under alternate stratified splits.
