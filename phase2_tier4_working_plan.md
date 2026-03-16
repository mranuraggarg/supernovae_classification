# Phase 2 Tier 4 Working Plan

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

Experiment A:
Train on SPCC compact features and test on each shifted domain.

Experiment B:
Train on mixed-domain training sets and compare mean cross-domain performance.

Experiment C:
Compare gain, permutation, SHAP when available, and ablation ranks across domains.

Experiment D:
Retest `top_5`, `top_8`, `top_10`, and `compact` subsets across domains.

Experiment E:
Measure F1 drop under the five named on-disk SPCC shifts.

## 6. Interpretation Rules

- Stable cross-domain F1 with stable feature rankings suggests survey-independent astrophysical signal.
- Stable rankings with lower F1 suggests partial calibration or domain-calibration mismatch.
- Unstable rankings and large F1 drop suggest dataset-specific dependence.

## 7. Execution Order

1. `phase2_tier4_make_variants.py`
2. `phase2_tier4_domain_swap.py`
3. `phase2_tier4_mixed_training.py`
4. `phase2_tier4_importance_domain.py`
5. `phase2_tier4_minimal_domain.py`
6. `phase2_tier4_shift_test.py`
7. `phase2_tier4_summary.py`
