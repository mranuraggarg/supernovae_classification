# Phase 2 Tier 3 Noise and Missing-Data Tests

Compact-feature perturbation proxies used to stress-test the frozen XGBoost baseline.

| scenario | f1 | pr_auc | delta_f1 | delta_pr_auc |
| --- | ---: | ---: | ---: | ---: |
| No perturbation | 0.844230 | 0.927761 | +0.000000 | +0.000000 |
| Reduced observations | 0.730242 | 0.843122 | -0.113988 | -0.084639 |
| Flux noise (+0.25 sigma) | 0.782128 | 0.846700 | -0.062102 | -0.081060 |
| Flux noise (+0.50 sigma) | 0.625885 | 0.686168 | -0.218345 | -0.241593 |
| Remove z-band proxies | 0.604339 | 0.689852 | -0.239891 | -0.237909 |
| Shortened time coverage | 0.704953 | 0.832808 | -0.139277 | -0.094953 |

These perturbations are feature-space proxies rather than full raw-light-curve reprocessing.
Plot: `plots/phase2_tier3/phase2_tier3_noise_test.png`
