# Supernovae Type Ia Classification — Phase 2 Tier-1 (SPCC Preprocessing Branch)

This branch, `phase2-tier1-spcc-preprocessing`, represents **Phase-2 Tier-1** of the project. The focus of this phase is no longer repair of the original evaluation pipeline. Instead, this branch is dedicated to **owning the SPCC preprocessing pipeline**, constructing a **native feature-engineering workflow from raw light curves**, and producing a **compact, interpretable XGBoost baseline** for Type Ia supernova classification.

The outcome of this branch is a reproducible Tier-1 experimental pipeline that:

- ingests raw **SPCC / SNPhotCC** light-curve `.DAT` files,
- constructs physically interpretable photometric features,
- evaluates progressively reduced feature sets,
- identifies a **compact 16-feature baseline**, and
- explains classifier behavior using **permutation importance** and **SHAP analysis**.

A paper-length summary of this branch is included in:

```text
phase2_tier1_paper.pdf
```

---

## 1. What this branch does

Phase-2 Tier-1 consists of four linked tasks:

1. **Raw SPCC preprocessing**  
   Parse raw DES/SPCC light-curve files and organize multi-band observations.

2. **Native feature engineering**  
   Build brightness, color, variability, and temporal features directly from the light curves.

3. **Model benchmarking and reduction**  
   Train and evaluate XGBoost-based feature sets of different sizes.

4. **Interpretability and scientific analysis**  
   Use permutation importance and SHAP to understand which astrophysical signals drive classification.

This branch should be treated as the **canonical implementation of Phase-2 Tier-1**, not as a generic branch for all earlier project stages.

---

## 2. Branch status

### Phase-2 Tier-1 goal

Build a fully owned and scientifically interpretable preprocessing and modeling pipeline for SPCC data.

### Phase-2 Tier-1 outcome

This goal has been achieved in this branch through three experimental feature configurations:

| Configuration | Feature Count | Purpose |
|---|---:|---|
| Full baseline | 31 | Initial engineered feature pool |
| Working set | 30 | Removal of clearly redundant features |
| Compact baseline | 16 | Final interpretable Tier-1 baseline |

### Final compact baseline metrics

| Metric | Value |
|---|---:|
| F1 | **0.844230** |
| ROC-AUC | **0.976588** |
| PR-AUC | **0.927761** |

### Comparison across Phase-2 Tier-1 configurations

| Configuration | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| 31-feature full baseline | 0.836717 | 0.976449 | 0.928071 |
| 30-feature working set | 0.840529 | 0.976398 | 0.928204 |
| 16-feature compact baseline | **0.844230** | **0.976588** | 0.927761 |

The compact baseline preserves ranking performance while improving thresholded classification performance and reducing the feature count by almost half.

---

## 3. Scientific result of this branch

The main scientific result of Phase-2 Tier-1 is that a **compact 16-feature representation** is sufficient to capture the dominant photometric information needed for Type Ia classification on SPCC.

The retained features span four physically meaningful groups:

### Brightness features

- `r_mean_flux`
- `g_mean_flux`
- `z_peak_flux`
- `i_peak_flux`
- `r_peak_flux`

### Color features

- `peak_color_g_minus_r`
- `peak_color_r_minus_i`
- `peak_color_i_minus_z`

### Variability features

- `i_std_flux`
- `z_std_flux`
- `r_std_flux`
- `i_amplitude`

### Temporal features

- `r_time_of_peak`
- `i_time_of_peak`
- `z_time_of_peak`
- `time_span`

These features were selected after staged reduction, ablation-guided tightening, compact reruns, and interpretability review.

---

## 4. Astrophysical interpretation

Interpretability analysis shows that the compact model relies on three dominant physical signal families:

1. **Brightness scale across bands**  
   Captured by mean and peak flux features, especially in redder bands.

2. **Color gradients across filters**  
   Captured by peak color differences, which act as photometric proxies for spectral energy distribution evolution.

3. **Temporal structure of the light curve**  
   Captured by band-specific peak times and total time span, reflecting the ordering and evolution of emission across filters.

In other words, the classifier is not functioning as an arbitrary black box. It is learning physically interpretable signatures associated with Type Ia supernovae.

---

## 5. Interpretability analysis

Two complementary methods were used to interpret the compact baseline:

- **Permutation importance** — measures performance degradation when a feature is shuffled.
- **SHAP values** — quantify how each feature contributes to individual predictions.

The main interpretability outputs are located in:

```text
results/phase2_tier1/
```

Important files include:

```text
results/phase2_tier1/phase2_tier1_compact_baseline_importance.json
results/phase2_tier1/phase2_tier1_compact_baseline_interpretation_table.md
results/phase2_tier1/phase2_tier1_compact_baseline_metrics.json
results/phase2_tier1/phase2_tier1_compact_baseline_robustness.json
results/phase2_tier1/phase2_tier1_compact_baseline_comparison.md
```

The SHAP summary plot for the compact baseline is located at:

```text
results/phase2_tier1/phase2_tier1_compact_baseline_plots/phase2_tier1_compact_baseline_shap_summary.png
```

Additional dependence and probability plots for the top compact features are stored in the same plot directory.

---

## 6. Robustness

The compact baseline was also evaluated across multiple random seeds to ensure that the result is not an artifact of one favorable split.

Compact-baseline robustness summary:

- **F1 mean:** 0.842998 ± 0.001783
- **ROC-AUC mean:** 0.976723 ± 0.000219
- **PR-AUC mean:** 0.928472 ± 0.000985

This stability supports the use of the compact feature set as the frozen Tier-1 baseline for future work.

---

## 7. Repository structure for Phase-2 Tier-1

```text
supernovae_classification/
├── README.md
├── feature_pipeline/
│   ├── config.py
│   ├── policies.py
│   ├── schemas.py
│   ├── loaders/
│   │   └── spcc_raw.py
│   ├── cleaning/
│   │   └── spcc_clean.py
│   ├── interpolation/
│   │   ├── spcc_legacy_reference.py
│   │   └── spcc_native_reconstruct.py
│   ├── extraction/
│   │   ├── feature_registry.py
│   │   └── spcc_features.py
│   └── validation/
│       └── checks.py
├── data/
│   └── spcc/
│       └── raw/
├── plots/
│   └── phase2_tier1/
├── results/
│   └── phase2_tier1/
├── notebooks/
│   ├── colab_phase2_tier1_xgb_importance.ipynb
│   └── colab_phase2_tier1_review_ablation.ipynb
├── phase2_tier1_benchmarks.py
├── phase2_tier1_build_feature_manifest.py
├── phase2_tier1_xgb_importance.py
├── phase2_tier1_review_ablation.py
├── phase2_tier1_tighten_manifest.py
├── phase2_tier1_compact_rerun.py
├── phase2_tier1_finalize_compact_baseline.py
├── phase2_tier1_interpretability.py
├── phase2_tier1_paper.md
└── phase2_tier1_paper.pdf
```

This structure reflects the actual work done in this branch. Older Phase-1 utilities are still in the repository history, but they are no longer the main focus of this branch README.

---

## 8. Reproducing the Phase-2 Tier-1 workflow

### 8.1 Environment

Use the project Conda environment defined for this repository.

Example:

```bash
conda env create -f environment.yml
conda activate astro-ml
```

Adjust the environment name if your local setup differs.

### 8.2 Raw data

This branch assumes availability of raw SPCC/SNPhotCC files under:

```text
data/spcc/raw/
```

### 8.3 Recommended execution order

The Phase-2 Tier-1 workflow was developed through staged scripts rather than one monolithic entry point.

A practical execution order is:

```bash
python phase2_tier1_benchmarks.py
python phase2_tier1_build_feature_manifest.py
python phase2_tier1_xgb_importance.py
python phase2_tier1_review_ablation.py
python phase2_tier1_tighten_manifest.py
python phase2_tier1_compact_rerun.py
python phase2_tier1_finalize_compact_baseline.py
python phase2_tier1_interpretability.py
```

The exact command-line options may depend on local file paths and runtime environment, especially for Colab-based experiments.

### 8.4 Colab notebooks

Some heavier Phase-2 XGBoost experiments were also run using Colab notebooks:

```text
notebooks/colab_phase2_tier1_xgb_importance.ipynb
notebooks/colab_phase2_tier1_review_ablation.ipynb
```

---

## 9. Relation to Phase-1

Phase-1 focused on **repairing the original evaluation pipeline** and verifying the scientific integrity of the original results.

Phase-2 Tier-1, by contrast, focuses on:

- raw SPCC preprocessing,
- native feature engineering,
- compact baseline construction,
- and interpretability.

Therefore this README intentionally emphasizes **Phase-2 Tier-1 outputs**, not the original six-model Phase-1 comparison table.

---

## 10. Data source and inspiration

This repository originally drew inspiration from Adam Moss’s supernova classification work and data organization:

- [Adam Moss’s Supernovae Dataset](https://github.com/adammoss/supernovae)

Phase-2 Tier-1 extends that direction by reconstructing a native preprocessing and feature-engineering workflow for SPCC/SNPhotCC data and by emphasizing interpretability.

---

## 11. Branch deliverables

This branch should be considered complete when interpreted as a **Phase-2 Tier-1 research branch**. Its main deliverables are:

- a native SPCC preprocessing pipeline,
- a compact 16-feature interpretable baseline,
- comparison tables across feature configurations,
- compact-baseline robustness evaluation,
- permutation importance and SHAP analysis,
- and a paper-length summary (`phase2_tier1_paper.pdf`).

---

## 12. Future work

The next logical directions after this branch are:

- **Feature Ablation Study (Phase-2 Tier-2)**  
  Systematic removal of compact features to quantify astrophysical signal strength.

- **Early Classification**  
  Study how early reliable Ia classification can be achieved from partial light curves.

- **Uncertainty / Abstention Framework**  
  Allow the classifier to abstain on ambiguous events.

- **Modern Dataset Extension**  
  Extend the preprocessing and feature-engineering framework beyond SPCC to newer datasets.

---

## 13. License

This project is licensed under the **MIT License**.
