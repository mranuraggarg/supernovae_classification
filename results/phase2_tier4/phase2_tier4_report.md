# Phase 2 Tier 4 Report

## Domain swap
- Best test domain: spcc with F1 0.840747.
- Largest degradation: plasticc with delta F1 -0.327406.

## Mixed training
- Best mixed-training regime by mean F1: spcc_plus_plasticc (0.746827).

## Feature stability
- Feature stability score: 0.455215.
- Representative top features: peak_color_r_minus_i, peak_color_g_minus_r, peak_color_i_minus_z, z_std_flux, i_std_flux.

## Minimal-domain stability
- Best subset across domains: compact (16 features) with mean F1 0.798286.

## Distribution shifts
- Hardest shift: no_i with F1 drop 0.192008.

## Interpretation
- Tier 4 uses SPCC compact variants written to disk and includes PLAsTiCC whenever the compact PLAsTiCC table has been built.
- Stable feature rankings across noisy, missing-band, and cross-survey domains support the claim that the compact representation is not relying only on one clean-survey artifact.
