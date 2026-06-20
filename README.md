# Supernovae Type Ia Classification — Phase 2 Tier 2

This branch, `phase2-tier2-compact-feature-ablation`, is the canonical **Phase 2 Tier 2** branch for the project. It builds on the frozen Phase 2 Tier 1 compact 16-feature XGBoost baseline and explains why that compact model works through feature ablation, feature-block ablation, staged subset growth, reduced-core experiments, and supporting reproducibility artifacts.

The focus of this branch is not a new preprocessing pipeline. It reuses the Phase 2 Tier 1 processed SPCC/SNPhotCC feature table, compact feature set, fixed split protocol, and XGBoost evaluation setup so every Tier 2 result is directly comparable to the compact baseline.

## Current Status

Phase 2 Tier 2 is complete as a reproducible analysis branch. The branch now includes:

- the Phase 2 Tier 1 compact 16-feature baseline reference,
- Tier 2 ablation and subset-growth experiment scripts,
- generated CSV, JSON, Markdown, PNG, and PDF artifacts under `results/phase2_tier2/` and `plots/phase2_tier2/`,
- corrected compact-baseline PR and ROC curves for the 16-feature model,
- dataset, feature-dictionary, training-protocol, uncertainty, learning-curve, and artifact-index support files,
- and Zenodo archives for both the initial v1 package and the current v2 support package.

The current archived version of this branch is **phase2-tier2-v2.0**, which includes the normalized Phase 2 Tier 2 support package and updated branch documentation.

## Baseline Reference

The frozen compact baseline comes from Phase 2 Tier 1 and uses 16 physically interpretable features. It is the reference point for all Tier 2 deltas.

| Metric | Compact 16-feature baseline |
|---|---:|
| F1 | 0.844230 |
| ROC-AUC | 0.976588 |
| PR-AUC | 0.927761 |

The corrected compact-baseline curve artifacts are:

```text
results/phase2_tier1/phase2_tier1_compact_baseline_plots/phase2_tier1_compact_baseline_precision_recall_curve.png
results/phase2_tier1/phase2_tier1_compact_baseline_plots/phase2_tier1_compact_baseline_roc_curve.png
```

These curves belong to the **Phase 2 Tier 1 compact baseline model**, not the older original-paper model and not a separate Tier 2 model.

## Compact Feature Set

The compact model uses four physical feature families.

| Group | Features |
|---|---|
| Brightness | `r_mean_flux`, `g_mean_flux`, `z_peak_flux`, `i_peak_flux`, `r_peak_flux` |
| Color | `peak_color_g_minus_r`, `peak_color_r_minus_i`, `peak_color_i_minus_z` |
| Variability | `i_std_flux`, `z_std_flux`, `r_std_flux`, `i_amplitude` |
| Temporal | `r_time_of_peak`, `i_time_of_peak`, `z_time_of_peak`, `time_span` |

## Tier 2 Results

The main Tier 2 conclusion is that temporal information is the most decisive family for the compact classifier, while brightness, color, and variability provide complementary structure. The compact feature set can be reduced to a smaller interpretable core with only modest degradation, but the full compact set remains the strongest and most stable reference model.

Key generated results:

| Analysis | Main output | Interpretation |
|---|---|---|
| Single-feature ablation | `results/phase2_tier2/feature_ablation_metrics.csv` | `time_span` is the strongest individual feature by F1 loss when removed. |
| Block ablation | `results/phase2_tier2/block_ablation_metrics.csv` | Removing temporal features causes the largest block-level degradation. |
| Subset growth | `results/phase2_tier2/subset_growth_metrics.csv` | Performance grows as brightness, color, variability, and temporal information are added. |
| Minimal core | `results/phase2_tier2/minimal_core_metrics.csv` | A 10-feature core retains most of the compact baseline performance. |
| Master report | `results/phase2_tier2/phase2_tier2_report.md` | Human-readable summary of the Tier 2 ablation results. |

Highlights from `results/phase2_tier2/phase2_tier2_report.md`:

- Removing `time_span` gives the largest single-feature F1 drop: `delta F1 = -0.033952`.
- Removing the temporal block gives the largest block-level drop: `delta F1 = -0.046596`, `delta PR-AUC = -0.054893`.
- The best reduced core is the top-10 subset: `F1 = 0.838509`, `PR-AUC = 0.916900`.
- The full compact model remains the preferred reference: `F1 = 0.844230`, `PR-AUC = 0.927761`.

## Figures

The primary Tier 2 figures are:

```text
plots/phase2_tier2/feature_ablation_delta_f1.png
plots/phase2_tier2/block_ablation_delta_f1.png
plots/phase2_tier2/subset_growth_f1.png
plots/phase2_tier2/subset_growth_paper_ready.png
plots/phase2_tier2/minimal_core_tradeoff.png
```

The paper-ready subset-growth figure is:

```text
plots/phase2_tier2/subset_growth_paper_ready.png
```

It uses the four cumulative stages:

1. brightness only,
2. brightness + color,
3. brightness + color + variability,
4. full compact model.

It plots both F1 and PR-AUC and is intended for the ablation/growth-of-information discussion.

## Supporting Artifacts

Additional Phase 2 Tier 2 support material is stored under `results/phase2_tier2/`.

Important files include:

```text
results/phase2_tier2/artifact_index.md
results/phase2_tier2/dataset_summary.md
results/phase2_tier2/training_protocol.md
results/phase2_tier2/compact_feature_dictionary.md
results/phase2_tier2/selected_compact_model.md
results/phase2_tier2/uncertainty_summary.md
results/phase2_tier2/learning_curve.md
results/phase2_tier2/validation_logloss_curve.md
results/phase2_tier2/tier2_experiment_audit.md
```

The artifact index is the best entry point for the full support package:

```text
results/phase2_tier2/artifact_index.md
results/phase2_tier2/artifact_index.csv
```

## Running The Workflow

Use the project environment defined by `environment.yml` or `astro-ml.yml`.

Example:

```bash
conda env create -f environment.yml
conda activate astro-ml
```

The Tier 2 scripts expect the processed Phase 2 Tier 1 compact dataset to exist:

```text
data/processed/phase2_tier1_compact_baseline.csv
```

The core Tier 2 experiment order is:

```bash
python3 phase2_tier2_feature_ablation.py
python3 phase2_tier2_block_ablation.py
python3 phase2_tier2_subset_growth.py
python3 phase2_tier2_minimal_core.py
python3 phase2_tier2_summary.py
```

The support-artifact scripts can be run after the compact dataset and Tier 2 outputs are available:

```bash
python3 phase2_tier2_metadata.py
python3 phase2_tier2_training_protocol.py
python3 phase2_tier2_feature_dictionary.py
python3 phase2_tier2_selected_model.py
python3 phase2_tier2_uncertainty.py
python3 phase2_tier2_learning_curves.py
python3 phase2_tier2_example_light_curve.py
python3 phase2_tier2_audit.py
python3 phase2_tier2_artifact_index.py
```

To regenerate the corrected compact-baseline PR and ROC curves:

```bash
python3 phase2_tier1_compact_curves.py
```

## Repository Structure

The current Phase 2 Tier 2 branch centers on these files and folders:

```text
supernovae_classification/
├── README.md
├── data/
│   └── processed/
│       ├── phase2_tier1_compact_baseline.csv
│       └── spcc_features_tier1.csv
├── feature_pipeline/
├── plots/
│   └── phase2_tier2/
├── results/
│   ├── phase2_tier1/
│   └── phase2_tier2/
├── phase2_tier1_compact_curves.py
├── phase2_tier2_common.py
├── phase2_tier2_feature_ablation.py
├── phase2_tier2_block_ablation.py
├── phase2_tier2_subset_growth.py
├── phase2_tier2_minimal_core.py
├── phase2_tier2_summary.py
├── phase2_tier2_metadata.py
├── phase2_tier2_training_protocol.py
├── phase2_tier2_feature_dictionary.py
├── phase2_tier2_selected_model.py
├── phase2_tier2_uncertainty.py
├── phase2_tier2_learning_curves.py
├── phase2_tier2_example_light_curve.py
├── phase2_tier2_audit.py
└── phase2_tier2_artifact_index.py
```

Older Phase 1 and Phase 2 Tier 1 utilities remain in the repository for provenance, but this README documents the current Phase 2 Tier 2 branch state.

## Release And DOI

The current Phase 2 Tier 2 Zenodo release is:

- Version: `phase2-tier2-v2.0`
- DOI: [10.5281/zenodo.19666153](https://doi.org/10.5281/zenodo.19666153)
- Zenodo record: [https://zenodo.org/records/19666153](https://zenodo.org/records/19666153)
- GitHub tag: `phase2-tier2-v2.0`

Citation:

```text
Anurag Garg. (2026). mranuraggarg/supernovae_classification:
Phase 2 Tier 2 v2: Compact Feature Ablation Support Package
(phase2-tier2-v2.0). Zenodo. https://doi.org/10.5281/zenodo.19666153
```

The previous Phase 2 Tier 2 Zenodo release is:

- Version: `phase2-tier2-v1.0`
- DOI: [10.5281/zenodo.19665796](https://doi.org/10.5281/zenodo.19665796)
- GitHub tag: `phase2-tier2-v1.0`

The DOI for all Zenodo versions of this project is:

- Concept DOI: [10.5281/zenodo.15074652](https://doi.org/10.5281/zenodo.15074652)

## Data Source And Inspiration

This repository originally drew inspiration from Adam Moss's supernova classification work and data organization:

- [Adam Moss's Supernovae Dataset](https://github.com/adammoss/supernovae)

Phase 2 extends that direction by reconstructing a native SPCC/SNPhotCC preprocessing and feature-engineering workflow, then using compact physically interpretable features for controlled model analysis.

## License

This project is licensed under the **MIT License**.
