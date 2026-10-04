# Phase 2 Tier 4 Domain Swap

Train on frozen SPCC compact data and evaluate on Tier 4 variant tables plus PLAsTiCC when available.

| train | test | F1 | PR-AUC | ROC-AUC | delta F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| spcc | spcc | 0.840747 | 0.911988 | 0.973744 | -0.003483 |
| spcc | noise | 0.826299 | 0.905817 | 0.970775 | -0.017931 |
| spcc | no_z | 0.692105 | 0.683592 | 0.908259 | -0.152125 |
| spcc | no_i | 0.652222 | 0.725259 | 0.918069 | -0.192008 |
| spcc | short_span | 0.819421 | 0.902578 | 0.970567 | -0.024809 |
| spcc | flux_scale | 0.772947 | 0.837949 | 0.946018 | -0.071283 |
| spcc | plasticc | 0.593168 | 0.657812 | 0.751198 | -0.251062 |

Plot: `plots/phase2_tier4/phase2_tier4_domain_swap.png`
