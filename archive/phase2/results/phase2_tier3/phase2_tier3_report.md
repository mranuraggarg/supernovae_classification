# Phase 2 Tier 3 Report

## Model robustness
- Best model by PR-AUC: XGBoost (0.927761).
- Frozen-split delta F1 for that model: +0.000000.

## Split stability
- Strongest resampling protocol by mean F1: random_split (0.845071 +/- 0.002608).

## Noise and missing-data robustness
- Largest F1 degradation: Remove z-band proxies (-0.239891).

## Importance consistency
- r_mean_flux (brightness): consensus score 2.367. Average r-band brightness encodes scale information beyond a single peak snapshot.
- time_span (temporal): consensus score 1.438. Observed temporal coverage contributes because different classes occupy different effective time windows.
- z_peak_flux (brightness): consensus score 1.393. Band-specific peak brightness is a strong classifier signal, consistent with transient luminosity structure.
- i_peak_flux (brightness): consensus score 1.349. i-band peak brightness contributes strongly to separating Ia from non-Ia events.
- peak_color_r_minus_i (color): consensus score 1.049. r-i color acts as a spectral-slope proxy near peak.

## Minimal core generalization
- Best reduced core: top_10 (10 features) with mean F1 0.715164.

## Interpretation
- Agreement among gain, permutation, SHAP, and ablation ranks supports the view that the retained compact features capture stable astrophysical signal rather than a single-model artifact.
