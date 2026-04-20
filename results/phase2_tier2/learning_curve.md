# Learning Curve

Learning curve for the fixed compact XGBoost configuration using increasing stratified subsets of the training partition.

- Selected hyperparameter candidate: {'max_depth': 4, 'eta': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 1.0, 'lambda': 1.0}
- Boosting rounds: 400

| train size | train F1 | validation F1 | train PR-AUC | validation PR-AUC |
| --- | --- | --- | --- | --- |
| 1365 | 0.998469 | 0.773286 | 1.000000 | 0.854788 |
| 2729 | 0.965159 | 0.809994 | 0.998551 | 0.886641 |
| 5457 | 0.937184 | 0.835267 | 0.988569 | 0.906097 |
| 8187 | 0.912438 | 0.849143 | 0.978785 | 0.913639 |
| 10915 | 0.902568 | 0.839367 | 0.973089 | 0.913649 |
| 13644 | 0.892857 | 0.845245 | 0.966213 | 0.919719 |

- PNG: `results/phase2_tier2/learning_curve.png`
- PDF: `results/phase2_tier2/learning_curve.pdf`
