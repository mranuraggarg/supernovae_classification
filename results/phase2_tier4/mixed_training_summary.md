# Phase 2 Tier 4 Mixed-Domain Training

Evaluate whether adding SPCC variants and PLAsTiCC to training improves average cross-domain performance.

| train | mean F1 | mean PR-AUC |
| --- | ---: | ---: |
| spcc | 0.704674 | 0.757530 |
| spcc_plus_noise | 0.711408 | 0.760781 |
| spcc_plus_no_z | 0.699238 | 0.769199 |
| spcc_plus_no_i | 0.709982 | 0.762528 |
| spcc_plus_short_span | 0.695854 | 0.746760 |
| spcc_plus_flux_scale | 0.708682 | 0.760015 |
| spcc_plus_plasticc | 0.746827 | 0.799853 |

Plot: `plots/phase2_tier4/phase2_tier4_mixed_training.png`
