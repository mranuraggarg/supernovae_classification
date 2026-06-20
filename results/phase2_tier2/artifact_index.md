# Phase 2 Tier 2 Artifact Index

This index maps generated code/supporting artifacts to analysis purposes and later manuscript sections.

- Indexed artifacts: 13
- Missing primary artifacts: 0

## Artifact Map
| artifact | type | path | analysis purpose | manuscript support | script | exists |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset summary | dataset | `results/phase2_tier2/dataset_summary.md` | Dataset support: dataset description, class balance, split counts | Data section; experimental setup; dataset-description support | `phase2_tier2_metadata.py` | yes |
| Class balance table | dataset | `results/phase2_tier2/class_balance.csv` | Dataset support: class balance and label distribution | Dataset statistics table | `phase2_tier2_metadata.py` | yes |
| Split class balance table | dataset/protocol | `results/phase2_tier2/split_class_balance.csv` | Protocol support: split details and held-out test usage | Train/validation/test protocol description | `phase2_tier2_metadata.py` | yes |
| Compact feature ranges | dataset/features | `results/phase2_tier2/compact_feature_ranges.csv` | Dataset support: engineered feature ranges | Dataset and feature-description details | `phase2_tier2_metadata.py` | yes |
| Training protocol | protocol | `results/phase2_tier2/training_protocol.md` | Protocol support: training details and hyperparameters | Methods; model-training protocol; hyperparameter table | `phase2_tier2_training_protocol.py` | yes |
| XGBoost hyperparameter grid | protocol | `results/phase2_tier2/xgb_hyperparameter_grid.csv` | Protocol support: hyperparameters not reported | Hyperparameter grid table | `phase2_tier2_training_protocol.py` | yes |
| Repeated-split uncertainty summary | uncertainty | `results/phase2_tier2/uncertainty_summary.md` | Uncertainty support: no uncertainty or cross-validation statistics | Robustness/stability paragraph; uncertainty table | `phase2_tier2_uncertainty.py` | yes |
| Repeated-split uncertainty runs | uncertainty | `results/phase2_tier2/uncertainty_runs.csv` | Uncertainty support: per-seed stability evidence | Supplementary repeated-split table | `phase2_tier2_uncertainty.py` | yes |
| Compact feature dictionary | feature dictionary | `results/phase2_tier2/compact_feature_dictionary.md` | Feature-definition support: full feature list, definitions, formulas, time_span clarification | Feature-definition table; appendix; correction of rise/decline wording | `phase2_tier2_feature_dictionary.py` | yes |
| Compact feature dictionary CSV | feature dictionary | `results/phase2_tier2/compact_feature_dictionary.csv` | Feature-definition support: tabular feature definitions | Feature table source | `phase2_tier2_feature_dictionary.py` | yes |
| Selected compact model | selected model | `results/phase2_tier2/selected_compact_model.md` | Protocol support: selected hyperparameters and final test metrics | Model-selection details; fixed-split baseline reproducibility | `phase2_tier2_selected_model.py` | yes |
| Tier 2 experiment audit | ablation audit | `results/phase2_tier2/tier2_experiment_audit.md` | Ablation support: ablation reproducibility and interpretation | Ablation results; compactness claim; analysis summary | `phase2_tier2_audit.py` | yes |
| Tier 2 experiment audit CSV | ablation audit | `results/phase2_tier2/tier2_experiment_audit.csv` | Ablation support: machine-readable ablation summary | Ablation table source | `phase2_tier2_audit.py` | yes |

## Scope Note
External-validation artifacts are intentionally excluded from this supporting artifact index because cross-survey generalization is handled in Phase 2 Tier 4.
