# Phase 2 Tier 4 Trial Temporal Domain Swap

Trial experiment replacing peak-time features with r-anchored relative phase offsets.

| train | test | F1 | PR-AUC | ROC-AUC | delta F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| spcc | spcc | 0.834883 | 0.908937 | 0.972649 | -0.009347 |
| spcc | noise | 0.821760 | 0.902824 | 0.969802 | -0.022470 |
| spcc | no_z | 0.661496 | 0.680768 | 0.905689 | -0.182734 |
| spcc | no_i | 0.633389 | 0.709148 | 0.910589 | -0.210841 |
| spcc | short_span | 0.819014 | 0.898963 | 0.968972 | -0.025216 |
| spcc | flux_scale | 0.763119 | 0.829417 | 0.943205 | -0.081111 |
| spcc | plasticc | 0.483436 | 0.487285 | 0.670643 | -0.360794 |

Plot: `plots/phase2_tier4_trial_temporal/phase2_tier4_trial_temporal_domain_swap.png`
