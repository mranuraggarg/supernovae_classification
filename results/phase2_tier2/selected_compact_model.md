# Selected Compact Model

This artifact records the exact fixed-split XGBoost candidate selected for the compact 16-feature model.

## Protocol
- Seed: 42
- Test split: 0.2
- Validation split within train+validation: 0.2
- Selection metric: validation PR-AUC
- Final fit: train+validation
- Final evaluation: held-out test

## Split counts
| split | total | Ia | non-Ia |
| --- | --- | --- | --- |
| train | 13644 | 3256 | 10388 |
| validation | 3411 | 814 | 2597 |
| trainval | 17055 | 4070 | 12985 |
| test | 4264 | 1018 | 3246 |

## Selected XGBoost candidate
- Candidate: 2
- Best iteration: 400

| parameter | value |
| --- | --- |
| max_depth | 4 |
| eta | 0.05 |
| subsample | 0.9 |
| colsample_bytree | 0.9 |
| min_child_weight | 1.0 |
| lambda | 1.0 |

## Validation metrics for selected candidate
| metric | value |
| --- | --- |
| accuracy | 0.919378 |
| precision | 0.779855 |
| recall | 0.922604 |
| f1 | 0.845245 |
| roc_auc | 0.975164 |
| pr_auc | 0.919719 |

## Held-out test metrics
| metric | value |
| --- | --- |
| accuracy | 0.916745 |
| precision | 0.762887 |
| recall | 0.944990 |
| f1 | 0.844230 |
| roc_auc | 0.976588 |
| pr_auc | 0.927761 |

## Compact features
z_peak_flux, r_mean_flux, peak_color_g_minus_r, i_peak_flux, peak_color_r_minus_i, peak_color_i_minus_z, g_mean_flux, r_peak_flux, z_std_flux, i_amplitude, i_std_flux, time_span, z_time_of_peak, i_time_of_peak, r_time_of_peak, r_std_flux
