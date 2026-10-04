# XGBoost Hyperparameter Selection Rationale

This note documents the compact-baseline XGBoost selection procedure used by the Phase-2 manuscript workflow.

## Source of the grid

The compact-baseline grid is defined in `phase2_tier1_xgb_importance.py` as `XGB_PARAM_GRID` and is reused by the Tier-1 baseline finalization, Tier-2 shared helpers, Tier-3 model comparisons, and Tier-4 paired uncertainty script.

The three tested candidates are:

| Candidate | max_depth | eta | subsample | colsample_bytree | min_child_weight | lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0.05 | 0.8 | 0.8 | 1.0 | 1.0 |
| 2 | 4 | 0.05 | 0.9 | 0.9 | 1.0 | 1.0 |
| 3 | 5 | 0.03 | 0.8 | 0.8 | 2.0 | 1.5 |

## Fixed settings

| Setting | Value | Rationale |
| --- | --- | --- |
| objective | `binary:logistic` | Binary Ia versus non-Ia probability model. |
| eval_metric | `logloss` | XGBoost training/evaluation loss; validation PR-AUC is used for model selection. |
| tree_method | `hist` | Efficient tree construction for repeated tabular experiments. |
| num_boost_round | 400 | Computational cap used with early stopping. |
| early_stopping_rounds | 30 | Stops training when validation loss no longer improves. |
| scale_pos_weight | `N_negative / N_positive` | Computed from the training subset to compensate for class imbalance. |

## Selection rule

For each split, candidate models are trained on the training subset, evaluated on the validation subset, and selected by validation PR-AUC. The held-out test subset is not used for model-family selection, grid selection, early-stopping selection, or class-imbalance weighting.

For the frozen 16-feature compact baseline in `results/phase2_tier1/phase2_tier1_compact_baseline_curve_metrics.json`, the selected candidate is:

| max_depth | eta | subsample | colsample_bytree | min_child_weight | lambda | best_iteration | validation PR-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.05 | 0.9 | 0.9 | 1.0 | 1.0 | 400 | 0.919719 |

The paired uncertainty analysis in `phase2_tier4_compact_vs_full_uncertainty.py` reruns the same selection procedure independently for each fold and feature set. The tested values are engineering choices for shallow, regularized boosted trees; they are not physically fundamental quantities.

