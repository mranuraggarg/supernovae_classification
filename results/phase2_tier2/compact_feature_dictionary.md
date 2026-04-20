# Compact Feature Dictionary

This table defines the 16 engineered features used by the compact Phase 2 Tier 2 model.

## Supporting cautions

- `time_span` is observational time coverage, not a rise-time or decline-time measurement.
- Flux-scale, spread, and amplitude features are engineered log-compressed summaries of reconstructed light curves.
- Color features are magnitude-style peak-flux-ratio proxies and should be described as color proxies rather than direct spectroscopic colors.

## Feature table
| order | feature | group | definition | formula / computation | interpretation | caution |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | z_peak_flux | brightness | Log-compressed maximum reconstructed z-band flux stored as log10(1 + peak flux). | log10(1 + max(reconstructed z-band flux, 0)) | Band-specific peak brightness is a strong classifier signal, consistent with transient luminosity structure. |  |
| 2 | r_mean_flux | brightness | Signed-log mean reconstructed r-band flux stored as sign(value) * log10(1 + abs(value)). | sign(mean r-band flux) * log10(1 + abs(mean r-band flux)) | Average r-band brightness encodes scale information beyond a single peak snapshot. |  |
| 3 | peak_color_g_minus_r | color | Magnitude-style color proxy computed as -2.5 * log10(g-band peak flux / r-band peak flux), then clipped to [-5, 5]. | -2.5 * log10(g_peak_flux / r_peak_flux), clipped to [-5, 5] | Color near peak captures temperature or spectral-slope differences characteristic of Ia evolution. |  |
| 4 | i_peak_flux | brightness | Log-compressed maximum reconstructed i-band flux stored as log10(1 + peak flux). | log10(1 + max(reconstructed i-band flux, 0)) | i-band peak brightness contributes strongly to separating Ia from non-Ia events. |  |
| 5 | peak_color_r_minus_i | color | Magnitude-style color proxy computed as -2.5 * log10(r-band peak flux / i-band peak flux), then clipped to [-5, 5]. | -2.5 * log10(r_peak_flux / i_peak_flux), clipped to [-5, 5] | r-i color acts as a spectral-slope proxy near peak. |  |
| 6 | peak_color_i_minus_z | color | Magnitude-style color proxy computed as -2.5 * log10(i-band peak flux / z-band peak flux), then clipped to [-5, 5]. | -2.5 * log10(i_peak_flux / z_peak_flux), clipped to [-5, 5] | i-z color contributes to distinguishing redder or later-phase light-curve behavior. |  |
| 7 | g_mean_flux | brightness | Signed-log mean reconstructed g-band flux stored as sign(value) * log10(1 + abs(value)). | sign(mean g-band flux) * log10(1 + abs(mean g-band flux)) | Mean g-band brightness adds complementary information to peak-only brightness proxies. |  |
| 8 | r_peak_flux | brightness | Log-compressed maximum reconstructed r-band flux stored as log10(1 + peak flux). | log10(1 + max(reconstructed r-band flux, 0)) | r-band peak brightness remains informative even after compact pruning. |  |
| 9 | z_std_flux | variability | Log-compressed standard deviation of reconstructed z-band flux stored as log10(1 + std). | log10(1 + standard deviation of reconstructed z-band flux) | Spread in z-band flux captures shape or variability around peak. |  |
| 10 | i_amplitude | variability | Log-compressed reconstructed i-band amplitude stored as log10(1 + max peak-to-trough amplitude). | log10(1 + max peak-to-trough reconstructed i-band amplitude) | i-band amplitude captures reconstructed peak-to-trough flux contrast as a light-curve-shape proxy. | This is a reconstructed i-band peak-to-trough amplitude proxy, not a direct bolometric amplitude. |
| 11 | i_std_flux | variability | Log-compressed standard deviation of reconstructed i-band flux stored as log10(1 + std). | log10(1 + standard deviation of reconstructed i-band flux) | i-band variability helps capture light-curve shape, not just scale. |  |
| 12 | time_span | temporal | Maximum normalized observation time minus minimum normalized observation time. | max(normalized observation time) - min(normalized observation time) | Observed temporal coverage contributes because different classes occupy different effective time windows. | This is observational time coverage, not a measured rise time or decline time. |
| 13 | z_time_of_peak | temporal | Normalized time at which reconstructed z-band flux reaches its maximum. | normalized time coordinate at maximum reconstructed z-band flux | Inter-band timing helps capture peak-lag structure in the transient. |  |
| 14 | i_time_of_peak | temporal | Normalized time at which reconstructed i-band flux reaches its maximum. | normalized time coordinate at maximum reconstructed i-band flux | Peak timing in i band contributes to the time-evolution signature of Ia light curves. |  |
| 15 | r_time_of_peak | temporal | Normalized time at which reconstructed r-band flux reaches its maximum. | normalized time coordinate at maximum reconstructed r-band flux | Peak timing in r band reflects band-lag physics and phase structure. |  |
| 16 | r_std_flux | variability | Log-compressed standard deviation of reconstructed r-band flux stored as log10(1 + std). | log10(1 + standard deviation of reconstructed r-band flux) | r-band spread adds another light-curve-shape cue after pruning redundant amplitudes. |  |
