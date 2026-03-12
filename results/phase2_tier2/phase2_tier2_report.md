# Phase 2 Tier 2 Report

## Baseline reference
- Source CSV: `data/processed/phase2_tier1_compact_baseline.csv`
- Compact feature count: 16
- Frozen baseline F1: 0.844230
- Frozen baseline ROC-AUC: 0.976588
- Frozen baseline PR-AUC: 0.927761

## Feature ablation highlights
- time_span (temporal): delta F1 -0.033952, delta PR-AUC -0.018264, label essential.
- z_std_flux (variability): delta F1 -0.007245, delta PR-AUC -0.003370, label marginal.
- z_peak_flux (brightness): delta F1 -0.006653, delta PR-AUC -0.002396, label marginal.
- i_time_of_peak (temporal): delta F1 -0.005859, delta PR-AUC -0.003011, label marginal.
- peak_color_i_minus_z (color): delta F1 -0.004983, delta PR-AUC -0.006934, label supportive.

## Near-redundant feature candidates
- r_peak_flux (brightness): delta F1 +0.001349, delta PR-AUC +0.000842, label redundant.
- i_peak_flux (brightness): delta F1 +0.001040, delta PR-AUC -0.000867, label redundant.
- r_std_flux (variability): delta F1 +0.000529, delta PR-AUC +0.000893, label redundant.
- i_std_flux (variability): delta F1 -0.002310, delta PR-AUC -0.003827, label marginal.
- peak_color_g_minus_r (color): delta F1 -0.002494, delta PR-AUC -0.006380, label supportive.

## Block ablation highlights
- Largest loss: removing temporal produced delta F1 -0.046596 and delta PR-AUC -0.054893, labeled essential.

## Subset growth
- Best subset-growth result: full_compact at F1 0.842291 (delta F1 -0.001939).

## Minimal core
- Best reduced core: top_10 with 10 features, F1 0.838509, PR-AUC 0.916900, delta F1 -0.005721.
- Feature list: z_peak_flux, peak_color_i_minus_z, z_std_flux, time_span, peak_color_r_minus_i, r_time_of_peak, i_time_of_peak, r_mean_flux, peak_color_g_minus_r, z_time_of_peak.

## Generated artifacts
- Feature ablation plot: `plots/phase2_tier2/feature_ablation_delta_f1.png`
- Block ablation plot: `plots/phase2_tier2/block_ablation_delta_f1.png`
- Subset growth plot: `plots/phase2_tier2/subset_growth_f1.png`
- Paper-ready subset growth plot: `plots/phase2_tier2/subset_growth_paper_ready.png`
- Minimal core plot: `plots/phase2_tier2/minimal_core_tradeoff.png`
