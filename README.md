# Supernovae Type Ia Classification — Phase 2 Compact Feature Robustness and Cross-Survey Transfer

This branch now contains the complete Phase‑2 workflow including Tier‑1, Tier‑2, Tier‑3, and the active Tier‑4 cross-survey transfer investigation.

Phase‑2 Tier‑1 constructed a compact interpretable feature representation for Type Ia supernova classification from SPCC/SNPhotCC light‑curve data.

Phase‑2 Tier‑2 performed systematic ablation experiments to quantify the physical importance of the compact photometric features.

Phase‑2 Tier‑3 extends the study by testing robustness, stability, and generalization of the compact feature representation across models, data splits, noise conditions, and reduced feature subsets.

Phase‑2 Tier‑4 extends the compact-feature study to cross-survey transfer between SPCC/SNPhotCC and PLAsTiCC. Tier‑4 tests whether the same compact, physically interpretable feature representation can be applied to a different simulated survey, and investigates why direct SPCC→PLAsTiCC transfer remains difficult even when the feature formulas are shared.

The goal of the current branch is therefore not only feature ablation, but verification that the compact representation captures stable astrophysical signal rather than dataset‑specific correlations.

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

Phase-2 Tier-2 consists of four linked tasks:

1. **Load the compact 16-feature baseline from Phase-2 Tier-1**  
   Use the frozen compact feature manifest as the reference model.

2. **Single-feature ablation experiments**  
   Remove each feature individually and measure the change in F1-score and PR-AUC.

3. **Subset growth experiments**  
   Evaluate staged feature groups (brightness, color, variability, temporal) to study how classification performance builds as physical information is added.

4. **Core-feature identification**  
   Determine the smallest physically interpretable subset that preserves nearly full performance.

---

## 2. Branch status

### Phase‑2 Tier‑3 outcome

| Experiment | Result | Interpretation |
|----------|---------|--------------|
| Model comparison | XGBoost best, ΔF1 ≈ 0 | compact features not model‑dependent |
| Split stability | F1 ≈ 0.845 ± small σ | stable across resampling |
| Noise test | large drop without z‑band | band information is physical |
| Importance comparison | SHAP / perm / gain agree | importance is consistent |
| Minimal core | best ≈ 10 features | 16‑feature set near minimal stable |

The results show that temporal features produce the largest performance drop, while color and brightness provide complementary information.

---

## 3. Scientific result of this branch

The main scientific result of Phase-2 Tier-2 is that the compact 16-feature representation can be reduced to a smaller core set of physically meaningful features with only minor loss in classification performance. The ablation analysis shows that temporal evolution provides the dominant discriminating signal, while brightness, color, and variability features refine the decision boundary.

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

The main deliverables of this branch are:

- compact 16-feature baseline from Phase-2 Tier-1
- single-feature ablation results
- subset growth experiments
- core-feature subset identification
- performance comparison tables
- figures used in the Phase-2 Tier-2 paper
- Phase-2 Tier-2 manuscript

---

## 11.1 Phase-2 Tier-3 extension in this branch

This branch now also includes a **Phase-2 Tier-3 robustness and generalization workflow** built on top of the frozen 16-feature compact baseline from Tier-1 and the ablation findings from Tier-2.

The new Tier-3 scripts are:

```text
phase2_tier3_model_compare.py
phase2_tier3_cv_stability.py
phase2_tier3_noise_test.py
phase2_tier3_importance_compare.py
phase2_tier3_minimal_generalization.py
phase2_tier3_summary.py
phase2_tier3_plan.tex
```

These scripts evaluate:

- classifier robustness across XGBoost, Random Forest, Logistic Regression, and SVM,
- stability under alternate cross-validation and random split protocols,
- compact-feature robustness under noise and missing-data proxy perturbations,
- consistency between gain, permutation, SHAP, and Tier-2 ablation importance views,
- and generalization of Tier-2 minimal-core subsets under new conditions.

Tier-3 outputs are written to:

```text
results/phase2_tier3/
plots/phase2_tier3/
```

---

## 11.2 Tier‑3 results summary

Results from the Tier‑3 scripts:

### Model comparison

| Model | PR‑AUC | Notes |
|------|--------|------|
| XGBoost | 0.928 | best performance |
| RandomForest | ~0.92 | similar behaviour |
| SVM | ~0.90 | slightly weaker |
| Logistic | lower | linear model insufficient |

### CV stability

| Protocol | Mean F1 | Std |
|----------|---------|-----|
| random_split | 0.845 | small |
| kfold | ~0.84 | small |
| repeated | ~0.84 | small |

### Noise / missing data

| Test | ΔF1 |
|------|------|
| remove z proxies | large drop |
| noise injection | moderate drop |
| reduced span | moderate drop |

### Importance consistency

Top consensus features:

- r_mean_flux
- time_span
- z_peak_flux
- i_peak_flux
- peak_color_r_minus_i

### Minimal core generalization

| subset | features | mean F1 |
|--------|----------|----------|
| top_5 | 5 | ~0.59 |
| top_8 | 8 | ~0.67 |
| top_10 | 10 | ~0.71 |
| compact | 16 | ~0.84 |

Conclusion: aggressive reduction weakens generalization.

---

## 12. Scientific scope after Tier-3

Phase-2 Tier-1, Tier-2, and Tier-3 together define the current scientific scope of this branch.

- Tier-1 established a compact, interpretable 16-feature baseline derived from SPCC/SNPhotCC light-curve data.
- Tier-2 quantified the physical importance of individual features and feature groups using systematic ablation experiments.
- Tier-3 tested whether the compact representation captures stable astrophysical information by evaluating robustness under changes in model, data split, noise, and feature reduction.

The main conclusions of the current branch are:

- The 16-feature compact representation preserves nearly all performance of the larger feature sets.
- Feature importance rankings are consistent across gain, permutation, SHAP, and ablation analysis.
- The compact feature model remains stable across different classifiers and resampling protocols.
- Performance degrades under strong perturbations, indicating that the retained features encode real photometric signal.
- Aggressive reduction below the compact baseline weakens generalization, suggesting that the 16-feature set is close to the minimal stable representation.

This branch therefore represents a complete compact-feature robustness study for Type Ia supernova classification using SPCC/SNPhotCC photometric data.

Future extensions beyond the current Tier‑4 work may include:

- cross-survey generalization tests,
- training on mixed datasets and testing on single surveys,
- early-time classification using partial light curves,
- and application of the compact feature framework to newer transient datasets.

---

## 12.1 Phase-2 Tier-4 cross-survey transfer investigation

Phase‑2 Tier‑4 evaluates whether the compact 16-feature representation learned from SPCC/SNPhotCC can transfer to PLAsTiCC under controlled feature-construction assumptions.

The Tier‑4 scripts include:

```text
phase2_tier4_make_variants.py
phase2_tier4_domain_swap.py
phase2_tier4_shift_test.py
phase2_tier4_plasticc_audit.py
phase2_tier4_feature_definition_audit.py
phase2_tier4_trial_lightcurve_normalization.py
phase2_tier4_windowed_plasticc_audit.py
phase2_tier4_windowed_plasticc.py
phase2_tier4_window_sweep.py
phase2_tier4_centroid_analysis.py
```

Tier‑4 outputs are written to:

```text
results/phase2_tier4/
results/phase2_tier4_windowed_plasticc/
results/phase2_tier4_windowed_plasticc_audit/
results/phase2_tier4_window_sweep/
results/phase2_tier4_centroid_analysis/
plots/phase2_tier4/
plots/phase2_tier4_centroid_analysis/
```

### Tier‑4 retained baseline

The retained Tier‑4 baseline uses a shared compact-feature builder for SPCC and PLAsTiCC. Earlier inconsistencies in PLAsTiCC peak-time definitions were corrected, and the audit now checks time consistency, feature-scale parity, label counts, and remaining median-scale mismatches before interpreting transfer results.

Under the retained non-windowed Tier‑4 setup, direct SPCC→PLAsTiCC transfer remains limited compared with SPCC→SPCC evaluation. This shows that compact physical features are meaningful but not automatically survey-invariant.

### Tier‑4 diagnostic experiments

Several diagnostic trials were run to isolate the source of the cross-survey gap:

| Diagnostic | Result | Interpretation |
|-----------|--------|----------------|
| ratio/log-ratio color trial | degraded robustness and did not solve transfer | color scaling alone is not the bottleneck |
| temporal-offset trial | reduced transfer performance | simple relative peak-time offsets do not solve the gap |
| PCA alignment trial | dominant axes were already aligned | the issue is not a simple linear rotation of feature space |
| event-level light-curve normalization | degraded SPCC and PLAsTiCC performance | amplitude normalization removes useful signal without fixing color mismatch |
| feature-definition audit | identified major PLAsTiCC full-history window mismatch | same formulas can have different effective meaning across surveys |
| PLAsTiCC transient-window sweep | improved transfer when using SPCC-scale transient windows | event-window parity is a real contributor to cross-survey transfer |
| class-conditional centroid analysis | SPCC and PLAsTiCC Ia centroids remain strongly separated | direct transfer is limited by feature-space domain shift, not only classifier choice |

### Windowed PLAsTiCC transfer result

The feature-definition audit showed that PLAsTiCC compact features were originally computed over much longer raw survey histories than SPCC. A science-constrained window sweep therefore tested PLAsTiCC transient-centered windows chosen to bracket SPCC event-duration scales:

```text
±60, ±75, ±90, ±105, ±120, ±180 days
```

The strongest transfer result came from the ±60 day window, which closely matches the central tendency of the SPCC transient duration. The ±105 day window also performed well and is retained as a sensitivity result because it approximates a broader, near-full transient-evolution regime.

The current interpretation is therefore:

- the compact 16-feature set remains physically meaningful within SPCC and remains useful as an interpretable diagnostic representation,
- direct cross-survey transfer is limited by survey-dependent feature distributions,
- PLAsTiCC full-history feature extraction is not comparable to SPCC event-scale feature extraction,
- restricting PLAsTiCC to SPCC-like transient windows improves transfer but does not fully align the surveys,
- class-conditional centroid analysis shows that the SPCC Ia and PLAsTiCC Ia populations do not occupy the same compact-feature region,
- the Ia centroid shift is larger than the within-survey Ia/non-Ia separation, indicating that transfer is fundamentally limited in the current compact feature space,
- and remaining mismatch is especially important in color and temporal/scale-sensitive features, including `peak_color_i_minus_z`.

The Tier‑4 result should therefore be read as evidence for **partial cross-survey transfer after event-window harmonization**, not as proof of a fully universal classifier.

### Tier‑4 class-conditional centroid result

The highest-priority Tier‑4 diagnostic was the class-conditional centroid comparison between SPCC and PLAsTiCC in the shared compact 16-feature space. The analysis measured four standardized centroids:

- SPCC Ia,
- PLAsTiCC Ia,
- SPCC non-Ia,
- PLAsTiCC non-Ia.

The key result is that the cross-survey Ia centroid shift is much larger than the class separation inside either survey:

| Quantity | Distance / Ratio |
|----------|------------------|
| SPCC Ia → PLAsTiCC Ia centroid distance | 6.125 |
| SPCC non-Ia → PLAsTiCC non-Ia centroid distance | 7.175 |
| SPCC Ia → SPCC non-Ia separation | 1.710 |
| PLAsTiCC Ia → PLAsTiCC non-Ia separation | 2.448 |
| Ia shift / SPCC class separation | 3.58× |
| Ia shift / PLAsTiCC class separation | 2.50× |

This result shows that the same astrophysical class, Type Ia, is not numerically aligned between SPCC and PLAsTiCC in the current compact feature representation. Therefore the SPCC→PLAsTiCC transfer limitation is not only a classifier-boundary problem; it is a feature-space alignment problem.

The Tier‑4 conclusion is therefore revised as follows:

> Compact physically interpretable features are useful within a survey and provide a strong diagnostic framework, but direct cross-survey transfer requires feature-space harmonization or domain adaptation. Interpretability alone is not sufficient for survey invariance.

This closes Tier‑4 as a domain-shift diagnosis rather than a failed transfer attempt.

## 12.2 Planned Phase‑2 Tier‑5 invariant-feature and harmonized-transfer study

Before claiming cross-survey portability, the next phase will identify which compact features are stable across surveys and test whether transfer improves after harmonization.

The Tier‑5 working question is:

> Which physically interpretable compact features are survey-invariant, and can a smaller harmonized subset improve SPCC→PLAsTiCC transfer?

Planned Tier‑5 tasks:

1. Rank compact features by class-conditional centroid shift.
2. Identify features whose Ia centroids are stable across SPCC and PLAsTiCC.
3. Remove or down-weight unstable features.
4. Test robust per-survey normalization and class-blind distribution alignment.
5. Compare direct transfer, harmonized transfer, and invariant-core transfer.
6. Decide whether the final claim should be partial portability, invariant-core portability, or survey-specific interpretability only.

The project hypothesis is therefore refined from direct universality to conditional portability:

> A compact physically interpretable feature model can support cross-survey Type Ia classification only when the retained features are both astrophysically meaningful and distributionally stable across survey domains.

---

## 13. License

This project is licensed under the **MIT License**.
