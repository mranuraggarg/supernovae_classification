# Phase 2 Tier 4 Trial Domain Swap

Trial experiment replacing the three compact color features with ratio/log-ratio versions.

| train | test | F1 | PR-AUC | ROC-AUC | delta F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| spcc | spcc | 0.823426 | 0.892174 | 0.967867 | -0.020804 |
| spcc | noise | 0.770120 | 0.828182 | 0.944027 | -0.074110 |
| spcc | no_z | 0.698329 | 0.685967 | 0.908384 | -0.145901 |
| spcc | no_i | 0.503268 | 0.583060 | 0.857783 | -0.340962 |
| spcc | short_span | 0.792359 | 0.877053 | 0.962088 | -0.051871 |
| spcc | flux_scale | 0.734986 | 0.796186 | 0.927409 | -0.109244 |
| spcc | plasticc | 0.547739 | 0.559508 | 0.728460 | -0.296491 |

Plot: `plots/phase2_tier4_trial/phase2_tier4_trial_domain_swap.png`
