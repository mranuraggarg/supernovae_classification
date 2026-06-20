# Tier 2 Experiment Audit

This audit summarizes the regenerated Phase 2 Tier 2 experiment outputs for the Phase 2 Tier 2.

## Baseline reference
- F1: 0.844230
- ROC-AUC: 0.976588
- PR-AUC: 0.927761

## Baseline parity checks
| experiment | delta F1 | delta ROC-AUC | delta PR-AUC |
| --- | --- | --- | --- |
| feature_ablation | 0.000000 | 0.000000 | 0.000000 |
| block_ablation | 0.000000 | 0.000000 | 0.000000 |
| subset_growth | 0.000000 | 0.000000 | 0.000000 |
| minimal_core | 0.000000 | 0.000000 | 0.000000 |

## Feature Ablation
| item | features | F1 | PR-AUC | delta F1 | delta PR-AUC | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| time_span | 15 | 0.810278 | 0.909497 | -0.033952 | -0.018264 | large performance loss when removed |
| z_std_flux | 15 | 0.836985 | 0.924391 | -0.007245 | -0.003370 | small performance loss when removed |
| z_peak_flux | 15 | 0.837577 | 0.925365 | -0.006653 | -0.002396 | small performance loss when removed |
| i_time_of_peak | 15 | 0.838371 | 0.924749 | -0.005859 | -0.003011 | small performance loss when removed |
| peak_color_i_minus_z | 15 | 0.839247 | 0.920827 | -0.004983 | -0.006934 | small performance loss when removed |

## Block Ablation
| item | features | F1 | PR-AUC | delta F1 | delta PR-AUC | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| temporal | 12 | 0.797634 | 0.872868 | -0.046596 | -0.054893 | large performance loss when removed |
| brightness | 11 | 0.819149 | 0.907833 | -0.025081 | -0.019928 | moderate performance loss when removed |
| color | 13 | 0.831820 | 0.907942 | -0.012410 | -0.019819 | moderate performance loss when removed |
| variability | 12 | 0.831874 | 0.918558 | -0.012356 | -0.009203 | moderate performance loss when removed |

## Subset Growth
| item | features | F1 | PR-AUC | delta F1 | delta PR-AUC | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| color_only | 3 | 0.621033 | 0.581675 | -0.223197 | -0.346085 | substantial loss relative to compact baseline |
| temporal_only | 4 | 0.533795 | 0.495550 | -0.310434 | -0.432211 | substantial loss relative to compact baseline |
| brightness_only | 5 | 0.750000 | 0.826434 | -0.094230 | -0.101327 | substantial loss relative to compact baseline |
| brightness_plus_color | 8 | 0.768272 | 0.849357 | -0.075958 | -0.078404 | substantial loss relative to compact baseline |
| brightness_plus_color_plus_variability | 12 | 0.791579 | 0.870509 | -0.052651 | -0.057252 | moderate reduced-feature performance |
| full_compact | 16 | 0.842291 | 0.928204 | -0.001939 | 0.000443 | near-baseline compact performance |

## Minimal Core
| item | features | F1 | PR-AUC | delta F1 | delta PR-AUC | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| top_5 | 5 | 0.706973 | 0.745903 | -0.137257 | -0.181858 | substantial loss relative to compact baseline |
| top_8 | 8 | 0.796399 | 0.874361 | -0.047830 | -0.053400 | moderate reduced-feature performance |
| top_10 | 10 | 0.838509 | 0.916900 | -0.005721 | -0.010861 | near-baseline compact performance |

## Supporting takeaways
- The baseline parity checks reproduce the frozen compact baseline to numerical precision.
- Temporal features are the largest block-level contributor in the leave-one-block-out study.
- `time_span` is the strongest single-feature ablation loss, but it should be described as observational time coverage, not rise or decline time.
- The 10-feature minimal core remains close to the compact baseline by F1, supporting the compactness claim while retaining a small PR-AUC loss.

Note: this audit summarizes final experiment metrics and deltas. Exact selected XGBoost hyperparameters for the fixed compact baseline are recorded separately in `results/phase2_tier2/selected_compact_model.md`.
