| feature | physical meaning | SHAP rank | mean(|SHAP|) | perm PR-AUC drop | perm F1 drop | relative importance | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| r_mean_flux | Mean brightness level in r band across the reconstructed light curve. | 1 | 1.088698 | 0.123053 | 0.139880 | dominant | Average r-band brightness encodes scale information beyond a single peak snapshot. |
| z_peak_flux | Peak brightness scale in z band. | 2 | 0.817450 | 0.117791 | 0.126188 | dominant | Band-specific peak brightness is a strong classifier signal, consistent with transient luminosity structure. |
| g_mean_flux | Mean brightness level in g band across the reconstructed light curve. | 3 | 0.539386 | 0.046416 | 0.054645 | strong | Mean g-band brightness adds complementary information to peak-only brightness proxies. |
| i_std_flux | Variability or spread of reconstructed i-band flux values. | 4 | 0.529209 | 0.072550 | 0.045005 | strong | i-band variability helps capture light-curve shape, not just scale. |
| time_span | Observed temporal coverage of the transient. | 5 | 0.501941 | 0.052521 | 0.068187 | strong | Observed temporal coverage contributes because different classes occupy different effective time windows. |
| i_peak_flux | Peak brightness scale in i band. | 6 | 0.457098 | 0.042409 | 0.022163 | strong | i-band peak brightness contributes strongly to separating Ia from non-Ia events. |
| peak_color_g_minus_r | Color near peak from the g-to-r flux ratio. | 7 | 0.447699 | 0.056942 | 0.038534 | strong | Color near peak captures temperature or spectral-slope differences characteristic of Ia evolution. |
| z_time_of_peak | Time of peak in z band. | 8 | 0.404470 | 0.036689 | 0.028150 | moderate | Inter-band timing helps capture peak-lag structure in the transient. |
| i_time_of_peak | Time of peak in i band. | 9 | 0.401607 | 0.032582 | 0.024043 | moderate | Peak timing in i band contributes to the time-evolution signature of Ia light curves. |
| z_std_flux | Variability or spread of reconstructed z-band flux values. | 10 | 0.396916 | 0.039581 | 0.047152 | moderate | Spread in z-band flux captures shape or variability around peak. |
| peak_color_r_minus_i | Color near peak from the r-to-i flux ratio. | 11 | 0.379351 | 0.044868 | 0.033000 | moderate | r-i color acts as a spectral-slope proxy near peak. |
| r_peak_flux | Peak brightness scale in r band. | 12 | 0.273912 | 0.053109 | 0.032944 | moderate | r-band peak brightness remains informative even after compact pruning. |
| peak_color_i_minus_z | Color near peak from the i-to-z flux ratio. | 13 | 0.270055 | 0.036905 | 0.025370 | moderate | i-z color contributes to distinguishing redder or later-phase light-curve behavior. |
| r_time_of_peak | Time of peak in r band. | 14 | 0.254219 | 0.024717 | 0.021583 | moderate | Peak timing in r band reflects band-lag physics and phase structure. |
| r_std_flux | Variability or spread of reconstructed r-band flux values. | 15 | 0.193557 | 0.006068 | 0.006850 | weaker | r-band spread adds another light-curve-shape cue after pruning redundant amplitudes. |
| i_amplitude | Peak-to-trough amplitude in the i band. | 16 | 0.169791 | 0.008654 | -0.004014 | weaker | i-band amplitude reflects how strongly the light curve rises and falls around peak. |
