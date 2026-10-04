# Compact versus Full Feature-Set Uncertainty

This analysis reruns the 31-feature full model and 16-feature compact model on identical stratified folds and seeds.

## Fold-Level Summary

| Model | Mean F1 | F1 std. | 95% CI |
| --- | ---: | ---: | ---: |
| 31-feature full | 0.843873 | 0.009480 | [0.832104, 0.855643] |
| 16-feature compact | 0.841132 | 0.006947 | [0.832508, 0.849756] |

## Paired Difference

- Mean paired F1 difference, compact minus full: -0.002742
- Fold-level 95% CI: [-0.007008, 0.001525]
- Exact paired sign-flip p-value over folds: 0.187500
- Cohen's dz over fold differences: -0.797796
- Out-of-fold paired bootstrap 95% CI for F1 difference: [-0.006099, 0.000541]

## Interpretation

The paired fold-level confidence interval includes zero, so the compact model should not be described as significantly better on F1. The supported conclusion is that the 16-feature representation preserves performance while using substantially fewer and more interpretable features.

## Output Files

- Fold metrics: `results/phase2_tier4/compact_vs_full_fold_metrics.csv`
- Out-of-fold predictions: `results/phase2_tier4/compact_vs_full_oof_predictions.csv`
- JSON summary: `results/phase2_tier4/compact_vs_full_uncertainty.json`
- Report: `results/phase2_tier4/compact_vs_full_uncertainty_report.md`
