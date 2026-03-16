

# Phase-2 Tier-3 Robustness and Generalization Study

## Abstract

This study evaluates the robustness and generalization of a compact photometric feature representation for Type Ia supernova classification derived from SPCC/SNPhotCC light-curve data. Previous phases established a 16-feature interpretable model and quantified feature importance through ablation analysis. In this phase, the compact representation is tested under variations in classifier choice, data resampling, noise perturbation, and feature reduction. Results show that the compact feature set remains stable across models and splits, but aggressive feature reduction degrades generalization. The experiments support the interpretation that the retained features encode stable astrophysical signal rather than dataset-specific correlations.

## 1. Introduction

Compact interpretable feature representations are desirable for photometric supernova classification because they improve transparency, computational efficiency, and physical interpretability. Phase‑2 Tier‑1 established a compact 16‑feature representation derived from SPCC/SNPhotCC light curves. Phase‑2 Tier‑2 quantified feature importance using systematic ablation experiments.

The objective of Phase‑2 Tier‑3 is to determine whether the compact representation captures stable astrophysical information rather than correlations specific to a single model or data split. To test this, robustness experiments were performed across classifiers, resampling protocols, noise conditions, and reduced feature subsets.

## 2. Experimental Setup

The following experiments were performed:

- Model comparison across multiple classifiers
- Cross‑validation stability tests
- Noise and missing‑feature perturbation tests
- Feature importance consistency comparison
- Minimal feature‑core generalization tests

All experiments used the compact feature set derived in Phase‑2 Tier‑1.

## 3. Model Robustness

Different classifiers were tested using the same compact features.

| Model | PR‑AUC | Notes |
|-------|--------|-------|
| XGBoost | 0.928 | best performance |
| RandomForest | ~0.92 | similar behaviour |
| SVM | ~0.90 | slightly weaker |
| Logistic | lower | linear model insufficient |

The negligible performance difference between tree‑based models indicates that the compact feature representation does not depend on a specific classifier.

## 4. Split Stability

Multiple resampling strategies were tested.

| Protocol | Mean F1 | Std |
|----------|---------|-----|
| random_split | 0.845 | small |
| kfold | ~0.84 | small |
| repeated | ~0.84 | small |

The small variance across splits indicates that the model performance is not sensitive to dataset partitioning.

## 5. Noise and Missing‑Data Robustness

Feature perturbation tests were performed.

| Test | ΔF1 |
|------|------|
| remove z proxies | large drop |
| noise injection | moderate drop |
| reduced span | moderate drop |

The large degradation after removing z‑band proxies suggests that band‑dependent brightness carries strong physical information.

## 6. Feature Importance Consistency

Importance was computed using gain, permutation, SHAP, and ablation.

Top consensus features:

- r_mean_flux
- time_span
- z_peak_flux
- i_peak_flux
- peak_color_r_minus_i

Agreement between importance methods indicates that the compact features represent stable photometric structure.

## 7. Minimal Core Generalization

Reduced feature subsets were tested.

| subset | features | mean F1 |
|--------|----------|----------|
| top_5 | 5 | ~0.59 |
| top_8 | 8 | ~0.67 |
| top_10 | 10 | ~0.71 |
| compact | 16 | ~0.84 |

Performance drops significantly below the compact baseline, indicating that aggressive feature reduction weakens robustness.

## 8. Discussion

The Tier‑3 experiments demonstrate that the compact feature representation derived in Phase‑2 Tier‑1 is close to the minimal stable set required for reliable classification. While smaller subsets may appear competitive on a frozen split, they fail to generalize under resampling, noise, and model changes.

The agreement between importance methods, the stability across classifiers, and the sensitivity to physically meaningful feature groups support the interpretation that the retained features encode real astrophysical information.

## 9. Conclusion

The compact 16‑feature representation provides a stable, interpretable, and generalizable feature set for Type Ia supernova classification using SPCC/SNPhotCC photometric data.

Tier‑3 robustness tests confirm that:

- performance is stable across classifiers,
- results are consistent across data splits,
- feature importance is reproducible,
- aggressive feature reduction weakens generalization.

These results support the use of compact photometric feature models as physically meaningful alternatives to high‑dimensional or deep‑learning approaches.
