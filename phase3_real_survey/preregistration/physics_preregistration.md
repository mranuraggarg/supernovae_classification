# Physics-First SDSS-II to DES-SN5YR Pre-registration

Version: 1.0.0  
Frozen: 2026-10-04T14:44:46+0400  
Status: PRE-REGISTERED BEFORE CROSS-SURVEY COMPACT-FEATURE INSPECTION  
Gate: GO

## 1. Scientific question

For securely classified normal Type Ia and ordinary core-collapse supernovae in a fixed common redshift region, which differences in the frozen 16-feature representation are predicted from the documented SDSS-II and DES-SN5YR observing systems alone?

The tested claim is limited: some measurable feature shift may be attributable to documented passband, calibration, cadence, window, depth/noise, and photometric-pipeline differences. Agreement would not prove causality, and failure would not imply that the classifier or the source populations are equivalent.

## 2. Frozen survey products

### Survey A: SDSS-II

- Final three-season Scene Modelling Photometry product `SDSS_allCandidates+BOSS`.
- Processing: `Holtz/v6_reCalib`.
- Official release archive SHA-256: `b0861ce0d8bd5ab138bc2e9f41dd5321912d76cb43c718789a7bc637d56e3dbd`.
- Response/calibration: `SDSS_Doi2010/kcor_SDSS_Bessell90_BD17.fits.gz`.
- KCOR SHA-256: `7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb`.
- Generating input SHA-256: `51a2c555ae1359a9f76442a8776e0e834d9d7dec83953ec90019888c9c0a6d55`.
- Nominal response representation: KCOR-embedded Doi2010 CCD-averaged `griz`, photon-counting, AB referenced.
- Calibration offsets recorded by the response family: g `+0.02028`, r `+0.00493`, i `+0.01780`, z `+0.01015` mag. These belong to the prescribed analysis calibration and are not to be re-applied ad hoc to epoch FLUXCAL.

### Survey B: DES-SN5YR

- Full-transient SMP v1.2 product `DES-SN5YR_DES`.
- Immutable release: Zenodo DOI `10.5281/zenodo.12720778`, archive `DES-SN5YR-1.2.zip`, Zenodo MD5 `9019a6ddc569553bc323e9e1b68a55bf`.
- HEAD SHA-256: `08c1fb41dfe8b4e8a969f205144f94c9f6838a1a895a0fedc75b837fb87fd00a`.
- PHOT SHA-256: `10248f0364e90f88626ef19cf5ecf052500bbcbfd27f14f8b94cf8ffbc7dd01c`.
- Response/calibration: `calib_DES-SN5YR_DES.fits.gz`, SHA-256 `4412a60ca41eb2cd8b32c038b352a551949a386cdc48acc0b059899bf46496c8`.
- Generating input SHA-256: `0917ddc66647ff613421a38c5df9a67d2113a7983e28889206c85c29b5c9e1ee`.
- Responses: pinned `DES-SN3YR_DECam/DECam_[griz].dat`, including atmosphere at airmass 1.3.
- Calibration offsets encoded by the input: g `-0.0008`, r `-0.0116`, i `-0.0090`, z `+0.0071` mag.
- DES v1.2 already includes DCR/chromatic and host-surface-brightness error corrections. They must not be applied again.

Each photometry product and its named calibration artifact is treated as an inseparable versioned pair.

## 3. Frozen population and support

The first study is restricted to:

- `0.05 <= z_helio <= 0.30`.
- Definite normal Ia.
- Definite ordinary core-collapse II, IIb, Ib, Ic, or Ibc.
- Secure transient spectroscopy for class; host spectroscopy may supply redshift but not class.
- All four griz bands satisfy the existing frozen builder support rule.
- DES shallow and deep fields are reported separately before any pooled result.
- SDSS Stripe 82 north and south are reported separately as a stability check.
- No matching or weighting on any compact feature.

Observed `MWEBV`, spectroscopic subtype, field class, host-redshift source, released epoch count, and observing season remain declared confounders. Galactic extinction will not be corrected in only one survey. The primary analysis retains released observed fluxes and conditions or stratifies on `MWEBV`; a symmetric extinction correction is a sensitivity analysis only.

## 4. Frozen feature implementation

Authoritative builder: `phase2_tier4_make_variants.py`, SHA-256 `7d2b1b7c23d685e652b4b7df383441537162f9dc3829f8edba380a8ddf2a4d76`.

Rules that are part of every prediction:

1. Keep griz observations only and sort by time.
2. Active observations have flux `> 0`, flux error `> 0`, and flux/error `>= 3`.
3. If no active observation exists, fall back to all retained observations.
4. `time_span` is last minus first active/fallback time.
5. Per-band values use active observations; a missing active band falls back to observations in that band within the active window.
6. Peak flux is the maximum sampled band flux.
7. Peak time is sampled-maximum time minus first active time, clipped to `[0,time_span]`.
8. Mean and population standard deviation use the selected per-band fluxes.
9. `i_amplitude` is sampled maximum minus sampled minimum in the selected i-band group.
10. A colour representative is the mean of up to the three largest strictly positive selected fluxes in each band.
11. Colour proxies are `-2.5 log10(Fa/Fb)` and are clipped to `[-5,5]`. They are asynchronous operational proxies, not simultaneous physical colours.
12. Peak, standard-deviation, and amplitude features are stored as `log10(1 + max(value,0))`; g/r means use the signed logarithm `sign(value) log10(1+abs(value))`.
13. An event is rejected if any griz colour representative is non-positive.

The frozen ordered features are:

`z_peak_flux`, `r_mean_flux`, `peak_color_g_minus_r`, `i_peak_flux`, `peak_color_r_minus_i`, `peak_color_i_minus_z`, `g_mean_flux`, `r_peak_flux`, `z_std_flux`, `i_amplitude`, `i_std_flux`, `time_span`, `z_time_of_peak`, `i_time_of_peak`, `r_time_of_peak`, `r_std_flux`.

## 5. Dependency classification

The complete dependency map is frozen in `feature_prediction_registry.csv`.

- Highly survey sensitive: all three colour proxies, all three sampled peak fluxes, all three sampled peak times, and `time_span`.
- Moderately survey sensitive: `g_mean_flux`, `r_mean_flux`, `i_std_flux`, `r_std_flux`, `z_std_flux`, and `i_amplitude`.
- Weakly survey sensitive: none are declared weak because every statistic is computed after a survey-dependent active-window selection.
- Unknown: none at the dependency level, although six features have no safe marginal directional prediction.

## 6. Fixed passband prediction framework

Normal-Ia passband predictions use only:

- Hsiao07 SED, SHA-256 `6bd032004eccc7b5b52f7c021f0c4b541d801a70195028bf5af0a2578f554d78`;
- exact SDSS KCOR-embedded CCD-average responses;
- exact DES DECam responses;
- photon-counting AB synthetic photometry;
- redshift grid `0.05, 0.075, ..., 0.30`;
- rest-frame phases `-10,0,10,20,30,40` days for phase-conditioned checks;
- the full Hsiao daily phase grid `-20` through `+85` days for a dense noiseless analogue of sampled peaks and top-three representatives.

For response `T_b(lambda)` and observed SED `f_lambda(lambda;z,p,c)`, the AB-shape quantity is

`m_b = -2.5 log10[ integral f_lambda(lambda) lambda T_b(lambda) dlambda / integral T_b(lambda)/lambda dlambda ] + constant`.

The registered passband term is

`Delta_b(z,p,c) = m_DES,b(z,p,c) - m_SDSS,b(z,p,c)`

and for a colour proxy analogue

`Delta_(a-b) = Delta_a - Delta_b`.

The common additive constant and SED normalization cancel. Calibration-offset effects are tracked separately and must not be folded into the passband term twice.

The response pivot wavelengths are SDSS/DES respectively: g `4701.38/4809.40`, r `6177.21/6420.44`, i `7496.55/7816.03`, z `8904.30/9171.13` Angstrom.

The Hsiao dense-template analogue predicts:

- `peak_color_r_minus_i`: DES-SDSS is negative from z=0.05 through 0.275 on the frozen grid, changing to `+0.0295` mag at z=0.30; its most negative registered value is about `-0.153` mag.
- `peak_color_i_minus_z`: positive from z=0.05 through 0.225, then negative from z=0.25 through 0.30; registered values range from about `+0.217` to `-0.185` mag.
- `i_peak_flux`: the ideal DES i peak is fainter than the SDSS i peak for z=0.05 through 0.275, with the magnitude difference becoming approximately zero/slightly negative at z=0.30.
- `z_peak_flux`: sign-changing; DES is brighter at low z through about 0.15 and fainter above about 0.175 in the dense-template magnitude analogue.
- `r_peak_flux` and `peak_color_g_minus_r`: smaller, redshift-dependent and sign-changing terms. They are secondary checks.

These are normal-Ia passband-only predictions. No universal ordinary-core-collapse sign is registered because II/IIb/Ib/Ic/Ibc SED diversity is too large. CC outcomes are evaluated against a fixed subtype template library only after that library is version-frozen, without fitting to SDSS or DES feature distributions.

## 7. Cadence, window, depth, and noise predictions

Documented inputs are an effective SDSS cadence of about four nights over three autumn Stripe 82 seasons and roughly weekly DES cadence over five seasons, with eight DES shallow and two deep fields. Both releases retain forced SMP measurements and negative fluxes, while the frozen builder selects positive S/N-active measurements.

Predictions:

- `time_span`: DES deep fields should produce longer active spans than SDSS at fixed class, redshift, and extinction because greater depth retains earlier and later S/N-active phases. DES shallow versus SDSS has no unconditional sign. This is a supporting, partially informed prediction because an earlier metadata audit already summarized active-window support; it is not counted as an independent primary confirmation.
- `r/i/z_time_of_peak`: coarser DES cadence should increase sampling error and produce broader/less finely quantized peak-time residuals than SDSS under a fixed source template. A marginal median sign is not registered. In deep DES fields, earlier active onset should shift time-from-active-start upward, but this conditional direction can be cancelled by cadence and season boundaries.
- Peak flux: coarser cadence lowers the expected sampled maximum for a fixed noiseless light curve, while greater depth and noise extremes act in competing directions. The passband prediction is therefore tested after a physics-only cadence operator and by DES field stratum.
- Colour proxies: asynchronous top-three representatives inherit band-dependent cadence and depth. Cadence may attenuate or reverse a passband-only colour term; this is ambiguity, not permission to retune the synthetic prediction.
- Mean, standard deviation, and amplitude: no safe raw cross-survey sign is registered because window length, positive-S/N truncation, noise, number of samples, source evolution, and fallback behavior compete.

Measurement, selection, and extractor effects are kept separate:

| Layer | Meaning |
|---|---|
| Physical measurement | Different response, calibration, PSF, sky, host treatment, and per-epoch uncertainty change the measured flux distribution. |
| Selection | Discovery/spectroscopic targeting, field depth, quality masks, and four-band survival determine which objects and epochs enter. |
| Extractor | Positive S/N thresholding, fallback, sampled maxima, top-three positive representatives, clipping, and log compression map those measurements nonlinearly into features. |

## 8. Primary confirmatory features

The primary set was chosen from physics and extractor behavior, not feature importance:

1. `peak_color_r_minus_i`: redshift-dependent normal-Ia synthetic colour term with a near-upper-boundary sign change.
2. `peak_color_i_minus_z`: large, sign-changing normal-Ia synthetic colour term across the common redshift range.
3. `i_peak_flux`: passband-sensitive peak with a mostly one-sided normal-Ia prediction over z=0.05-0.275.
4. `z_peak_flux`: redshift-dependent sign reversal distinct from the i-band prediction.
5. `r_time_of_peak`: cadence-sensitive sampled timing feature with a predicted increase in sampling-error spread, but no unconditional median direction.

`time_span` is a predeclared supporting feature rather than an independent primary feature because prior metadata-level work exposed active-window summaries. `peak_color_g_minus_r` and `r_peak_flux` are secondary passband checks. The remaining six features are mechanism diagnostics with no predeclared marginal sign.

## 9. Primary nulls and falsifiers

| Feature | Physical prediction | Null expectation | Contradiction | Ambiguous |
|---|---|---|---|---|
| `peak_color_r_minus_i` | Normal-Ia DES-SDSS follows the frozen Hsiao redshift function: mostly negative, approaching/crossing zero near z=0.30. | No redshift-coherent agreement with the synthetic term; passband correction does not reduce the conditioned residual. | Opposite-sign conditioned curve over a contiguous redshift region where all registered Hsiao phases agree, with uncertainty excluding zero. | Phase/cadence uncertainty spans both signs or the trend changes materially by field/season. |
| `peak_color_i_minus_z` | Positive at z<=0.225 and negative at z>=0.25 in the dense-template analogue. | No sign reversal at the predeclared redshift location and no residual reduction after the synthetic correction. | A stable opposite ordering on both sides of the predicted reversal. | Broad uncertainty around the reversal or subtype/extinction sensitivity dominates. |
| `i_peak_flux` | For normal Ia, ideal DES i peak is lower in flux than SDSS for z<=0.275 after common flux convention; near equality/sign change at z=0.30. | Conditioned DES-SDSS peak relation is unrelated to the synthetic passband term. | Opposite-sign relation where phase-grid predictions agree, after the cadence operator and field stratification. | Flux-scale, cadence, saturation, or depth terms are comparable to the passband term. |
| `z_peak_flux` | Sign-changing with redshift: DES brighter at low z and fainter at higher z in the registered dense-template analogue. | No coherent redshift dependence matching the synthetic term. | A robust reversal in the opposite direction. | z-band noise/depth or phase dependence removes a stable sign. |
| `r_time_of_peak` | DES weekly sampling produces broader template-relative sampled-peak errors than SDSS four-night sampling; no marginal median sign. | Timing residual spread is unchanged under the documented cadence operators. | DES cadence degradation produces no added timing spread in fixed-template injection, and observed conditioned spread is systematically narrower without a documented depth/window explanation. | Boundary truncation, first-active-time shifts, or field depth dominates cadence. |

No classifier performance or p-value alone can satisfy a prediction. Uncertainty intervals and effect functions are reported even when they include zero.

## 10. Planned measurements

The full plan is in `planned_measurements.md`. Primary estimands are:

- redshift-conditioned class-conditional median curves and quantile bands;
- observed-minus-synthetic residual curves for passband predictions;
- 1-Wasserstein distance before and after the fixed physics operator as a magnitude summary;
- quantile displacement at 0.1, 0.5, and 0.9;
- template-relative absolute peak-time error and robust spread for cadence predictions;
- DES shallow/deep and SDSS north/south results reported separately;
- ordinary CC results by broad subtype where counts permit, never forced into the Ia template prediction.

Bootstrap intervals will use object-level resampling within survey, class, redshift stratum, and field stratum. Their purpose is uncertainty quantification, not a binary p-value gate.

## 11. Candidate physics-only transformations

No transformation is applied in this task.

### Synthetic passband operator

For an epoch/template state `(z,p,c)`:

`F_SDSS,b = F_DES,b * 10^(0.4 * Delta_b(z,p,c))`

or equivalently for a colour:

`C_SDSS,a-b = C_DES,a-b - Delta_(a-b)(z,p,c)`.

This acts before log compression and requires a fixed SED/template and phase treatment. It is class-, redshift-, phase-, and band-dependent, not a fitted scalar.

### Calibration-system operator

Use the exact KCOR/calibration artifacts to put synthetic calculations into one declared AB convention. Do not apply listed offsets independently to released FLUXCAL unless the SNANA calibration semantics show that conversion is required. No coefficient may be estimated from SDSS-DES feature alignment.

### Cadence operator

Forward-sample the same fixed template through the released/documented SDSS and DES observation times and uncertainties. A comparison operator may degrade the denser schedule to a predeclared schedule drawn from released unlabeled epoch metadata. It may not select a schedule because it improves class separation or survey alignment.

### Window operator

Apply a common observer-frame or rest-frame phase window only after its anchor and bounds are fixed from release metadata or the fixed template framework. No bounds may be selected from target feature distributions.

### Depth/noise operator

Forward-model per-epoch uncertainty using released sky, PSF, gain, read noise, zeropoint, and flux-error information, stratified by DES shallow/deep field. The first study will not brighten/dim real objects or censor epochs to minimize feature distance.

## 12. Leakage boundary

Allowed before outcome inspection: response curves, calibration files, fixed template spectra, survey metadata, unlabeled observation schedules, uncertainty metadata, field/depth labels, quality-mask documentation, and frozen redshift/support rules.

Held out until this registration is frozen: cross-survey feature summaries, feature distances, target-label-guided transformation choices, SHAP values, centroids, classifier scores, and any correction selected because it improves target classification.

Target labels may later be used only to apply the already frozen secure-label inclusion rule and to report class-conditioned outcomes.

## 13. What has not been inspected

During this pre-registration task:

- no SDSS-vs-DES compact-feature table was opened or summarized;
- no cross-survey means, medians, quantiles, distances, centroids, SHAP values, or classifier results were calculated;
- no transformation was fitted or applied;
- no target label was used to select a prediction or primary feature.

An earlier feasibility audit did report label counts and metadata-level active-window/epoch support. Therefore the `time_span` prediction is explicitly marked partially informed and supporting. The passband predictions above were calculated only from the frozen responses and fixed Hsiao template.

## 14. Stop and interpretation rules

Ten of sixteen features admit a defensible falsifiable prediction: six passband-conditioned peak/colour features, three cadence-sensitive sampled peak-time features, and `time_span`. This exceeds the minimum of three.

The study stops before physics normalization if any of the following occurs:

- the frozen real-survey builder cannot reproduce the exact registered definitions without survey-specific branches;
- epoch flux conventions cannot be placed in a documented common calibration interpretation;
- the fixed SED calculation cannot be reproduced from the checksummed artifacts;
- fewer than three predictions remain after quality-mask and support validation;
- transformation choices would require target feature distributions or labels.

Directional agreement alone is not causal evidence. A credible result requires coherent behavior across independent passband and cadence mechanisms, stability under predeclared confounder strata, and preservation of within-class structure.
