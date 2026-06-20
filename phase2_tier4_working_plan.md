# Phase 2 Tier 4 Working Plan

> Status: COMPLETED
>
> Completion date: June 2026
>
> Final Tier-4 conclusion: The compact 16-feature representation remains physically meaningful and useful within a survey, but direct SPCC→PLAsTiCC transfer is limited by survey-dependent feature distributions. Event-window harmonization improves transfer, yet class-conditional centroid analysis demonstrates that the SPCC and PLAsTiCC Ia populations are not aligned in the shared compact feature space. Tier-4 is therefore closed as a domain-shift diagnosis phase, with follow-up work deferred to a future Tier-5 invariant-feature and harmonized-transfer study.

Branch: `phase2-tier4-domain-generalization`  
Date: March 16, 2026

## 1. Goal

Phase 2 Tier 4 asks whether the frozen compact 16-feature representation remains useful when the observing domain changes.

This repository contains SPCC compact features plus raw PLAsTiCC CSVs. The working implementation therefore writes concrete Tier 4 CSV variants to disk for SPCC and can also build a compact PLAsTiCC feature table for Dataset C.

## 2. Frozen Reference

Tier 4 reuses the Tier 3 compact baseline:

- Features: `16`
- F1: `0.8442299254`
- PR-AUC: `0.9277608810`
- ROC-AUC: `0.9765883838`

Source inputs:

- `data/processed/phase2_tier1_compact_baseline.csv`
- `results/phase2_tier1/phase2_tier1_compact_baseline_manifest.json`
- `results/phase2_tier1/phase2_tier1_compact_baseline_metrics.json`

## 3. Domain Strategy

Tier 4 will evaluate the following datasets and variants:

- `spcc`: frozen baseline domain.
- `noise`: additive noise on compact features.
- `no_z`: z-related features collapsed to reference values.
- `no_i`: i-related features collapsed to reference values.
- `short_span`: reduced `time_span` and compressed peak-time offsets.
- `flux_scale`: brightness features randomly rescaled.
- `plasticc`: compact feature table built from `data/PLAsTiCC/training_set.csv` and metadata when dependencies are available.

On-disk outputs:

- `data/spcc/features/compact_features.csv`
- `data/spcc/tier4_variants/noise/compact_features_noise.csv`
- `data/spcc/tier4_variants/no_z/compact_features_no_z.csv`
- `data/spcc/tier4_variants/no_i/compact_features_no_i.csv`
- `data/spcc/tier4_variants/short_span/compact_features_short_span.csv`
- `data/spcc/tier4_variants/flux_scale/compact_features_scaled_flux.csv`
- `data/PLAsTiCC/features/compact_features.csv`

## 4. Deliverables

Code:

- `phase2_tier4_common.py`
- `phase2_tier4_make_variants.py`
- `phase2_tier4_domain_swap.py`
- `phase2_tier4_mixed_training.py`
- `phase2_tier4_importance_domain.py`
- `phase2_tier4_minimal_domain.py`
- `phase2_tier4_shift_test.py`
- `phase2_tier4_summary.py`
- `phase2_tier4_plasticc_audit.py`
- `phase2_tier4_feature_definition_audit.py`
- `phase2_tier4_trial_lightcurve_normalization.py`
- `phase2_tier4_windowed_plasticc.py`
- `phase2_tier4_windowed_plasticc_audit.py`
- `phase2_tier4_window_sweep.py`
- `phase2_tier4_centroid_analysis.py`

Results:

- `results/phase2_tier4/domain_swap_metrics.csv`
- `results/phase2_tier4/mixed_training_metrics.csv`
- `results/phase2_tier4/importance_domain_metrics.csv`
- `results/phase2_tier4/minimal_domain_metrics.csv`
- `results/phase2_tier4/shift_test_metrics.csv`
- `results/phase2_tier4/phase2_tier4_master_summary.json`
- `results/phase2_tier4/phase2_tier4_report.md`

Plots:

- `plots/phase2_tier4/phase2_tier4_domain_swap.png`
- `plots/phase2_tier4/phase2_tier4_mixed_training.png`
- `plots/phase2_tier4/phase2_tier4_importance_domain.png`
- `plots/phase2_tier4/phase2_tier4_minimal_domain.png`
- `plots/phase2_tier4/phase2_tier4_shift_test.png`

## 5. Experiment Mapping

Experiment A (completed):
Class-conditional centroid analysis comparing SPCC and PLAsTiCC Ia/non-Ia populations in the compact feature space.

Result:
Cross-survey Ia centroid shift exceeded within-survey class separation, indicating that direct transfer is fundamentally limited by feature-space domain shift.

Experiment B (completed):
Direct SPCC→PLAsTiCC transfer and mixed-domain evaluation.

Result:
Transfer degradation was measurable but not catastrophic.

Experiment C (completed):
Feature-definition audit and PLAsTiCC compact-feature reconstruction review.

Result:
Several features had different effective meanings because PLAsTiCC compact features were initially extracted from full-history light curves.

Experiment D (completed):
Event-window harmonization and transient-window sweep.

Result:
SPCC-scale transient windows improved transfer performance, demonstrating that temporal-domain mismatch contributes significantly to the observed gap.

Experiment E (completed):
Synthetic domain-shift perturbation studies and normalization trials.

Result:
Feature-space robustness was generally preserved, but normalization alone did not resolve survey mismatch.

## 6. Final Tier-4 Interpretation

- The compact 16-feature representation remains scientifically meaningful and interpretable.
- Direct SPCC→PLAsTiCC transfer is limited by survey-dependent feature distributions.
- Event-window harmonization reduces, but does not eliminate, cross-survey mismatch.
- The dominant limitation is feature-space alignment rather than classifier instability.
- Class-conditional centroid analysis demonstrates that Type Ia populations occupy substantially different regions of the compact feature space across surveys.
- The Tier-4 outcome supports conditional portability rather than full survey universality.


## 7. Completion Summary

Completed analyses:

1. Domain-shift perturbation study
2. Mixed-domain training study
3. Feature-importance stability analysis
4. Minimal-feature transfer analysis
5. Distribution-shift diagnostics
6. Feature-definition audit
7. Event-window harmonization study
8. PLAsTiCC transient-window sweep
9. Class-conditional centroid analysis

Tier-4 is complete.

Future work will proceed under a separate Tier-5 phase focused on invariant-feature discovery, feature-space harmonization, and cross-survey transfer using survey-stable compact features.
