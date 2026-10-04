# Paper Sections 5 and 6 Draft

## 5 Do these features correspond to real light-curve physics?

The Tier-3 and Tier-4 results support a physical interpretation rather than a purely statistical one. Class separation appears directly in feature distributions, the feature families correlate in physically sensible ways, and performance under controlled shifts degrades gradually instead of collapsing immediately.

### 5.1 Feature distribution vs class

![Feature distribution vs class](/Users/anuraggarg/work/supernovae_classification/plots/phase2_tier4/paper_feature_distribution_vs_class.png)

Representative SPCC class splits:
- **Brightness (z-band peak flux)**: Ia median 1.618, non-Ia median 1.429, Cohen d 0.710.
- **Color (peak r-i color)**: Ia median 0.461, non-Ia median 0.079, Cohen d 0.983.
- **Time (time span)**: Ia median 58.028, non-Ia median 69.105, Cohen d -0.271.
- **Variability (z-band flux scatter)**: Ia median 0.979, non-Ia median 0.714, Cohen d 0.745.

These distributions show that the compact features are not random latent coordinates. Brightness and variability proxies shift upward for Ia events, peak-color features show a visibly different class locus, and the temporal coverage differs by class rather than collapsing onto one common distribution.

### 5.2 Correlation between features

![Feature correlation heatmap](/Users/anuraggarg/work/supernovae_classification/plots/phase2_tier4/paper_feature_correlation_heatmap.png)

Strongest correlations in the SPCC compact table:
- `i_amplitude` vs `i_std_flux`: Pearson r = 0.981.
- `r_mean_flux` vs `r_peak_flux`: Pearson r = 0.959.
- `r_peak_flux` vs `r_std_flux`: Pearson r = 0.922.
- `z_peak_flux` vs `i_peak_flux`: Pearson r = 0.920.
- `i_peak_flux` vs `i_std_flux`: Pearson r = 0.915.
- `z_peak_flux` vs `z_std_flux`: Pearson r = 0.914.

The correlation structure is also physically plausible. Flux-amplitude and flux-scatter features cluster strongly, same-band mean and peak brightness move together, and neighboring-band brightness features remain highly coupled. This is the pattern expected when the compact table is preserving light-curve scale and shape information instead of arbitrary dataset idiosyncrasies.

### 5.3 Physical interpretation

- **Brightness**: `z_peak_flux` and `r_mean_flux` remain among the strongest Tier-2/Tier-3 signals. `z_peak_flux` separates Ia from non-Ia with Cohen d 0.710, consistent with class-dependent luminosity scale near peak.
- **Color**: `peak_color_r_minus_i` is one of the most stable high-importance features across gain, permutation, SHAP, and ablation analyses. Ia events show a higher median `r-i` proxy (0.461 vs 0.079), which is consistent with color acting as a coarse spectral-slope indicator.
- **Time**: `time_span` carries a smaller but still real class effect. The Ia median span is 58.028, compared with 69.105 for non-Ia, showing that temporal coverage contributes useful discriminative structure instead of pure sampling noise.
- **Variability**: `z_std_flux` and `i_std_flux` stay important after multiple checks, with `z_std_flux` showing Cohen d 0.745. That supports the idea that the compact table is retaining information about rise/decline strength and band-wise curve structure.

### 5.4 Sensitivity to shift

![Shift sensitivity](/Users/anuraggarg/work/supernovae_classification/plots/phase2_tier4/paper_shift_sensitivity.png)

Tier-4 shift results show a graded rather than catastrophic failure mode. Noise and shortened time span produce only small drops, flux scaling produces a moderate drop, and removing full bands hurts much more strongly. This hierarchy is physically sensible: the model tolerates mild perturbations but depends strongly on cross-band structure.

### 5.5 Reduced training set test

![Reduced training set sensitivity](/Users/anuraggarg/work/supernovae_classification/plots/phase2_tier4/paper_reduced_training_set.png)

The PLAsTiCC compact training table shrinks from 7848 rows in the original extraction to 6235 rows in the parity-safe extraction. Across that reduction, F1 changes from 0.705151 to 0.728144 (delta +0.022993), while ROC-AUC changes by -0.009169 and PR-AUC changes by -0.028816.

The main Tier-4 interpretation is therefore not a collapse but a broad stability result: scores stay in the same range even after the usable training set is reduced. That is more consistent with real retained signal than with a fragile shortcut that disappears as soon as the sample changes.

## 6 Results summary

| Test | F1 | ROC | PR | Comment |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.844230 | 0.976588 | 0.927761 | Tier-1 frozen 16-feature compact baseline. |
| model compare | 0.844230 | 0.976588 | 0.927761 | Best Tier-3 model: XGBoost. |
| CV | 0.845071 | 0.975574 | 0.923226 | Best resampling protocol: random_split (mean over runs). |
| noise | 0.782128 | 0.949462 | 0.846700 | Tier-3 moderate flux-noise perturbation (+0.25 sigma). |
| reduced train | 0.728144 | 0.897319 | 0.766929 | Parity-safe PLAsTiCC table (6235 rows); vs 7848 rows: delta F1 +0.023, delta ROC -0.009, delta PR -0.029. |
| minimal features | 0.838509 | 0.973165 | 0.916900 | Best reduced subset: top_10 (10 features). |
| shift | 0.652222 | 0.918069 | 0.725259 | Hardest Tier-4 shift: no_i (F1 drop 0.192). |

Overall, the final Phase-2 picture is consistent across tiers: the compact features are competitive with the full baseline, remain stable under resampling, retain useful performance under noise and reduced feature subsets, and degrade in a structured way under domain shift. That combination supports the claim that the retained features correspond to real light-curve physics, even though Tier-4 still shows that survey transfer is not fully solved.
