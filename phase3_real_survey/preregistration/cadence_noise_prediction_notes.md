# Cadence, Window, Depth, and Noise Prediction Notes

Version: 1.0.0  
Frozen: 2026-10-04T14:44:46+0400

## Documented inputs

| Quantity | SDSS-II | DES-SN5YR |
|---|---|---|
| Geometry | Stripe 82 | Ten SN fields: eight shallow, two deep |
| Seasons | Three autumn seasons, 2005-2007 | Five seasons |
| Effective cadence | Approximately four nights | Roughly weekly; field/band dependent |
| Photometry | Scene modelling, forced measurements, negative flux retained | SMP, nightly combination, forced measurements, negative flux retained |
| Per-epoch metadata | MJD, band, flux/error, PSF, sky, read noise, zeropoint, gain, flags | MJD, band, flux/error, PSF, sky, read noise, zeropoint, gain, flags, CCD/image/field |
| Selection context | Detection on at least two nights and approximately r<22.5; targeted spectroscopy | DiffImg discovery plus spectroscopic targeting; shallow/deep strategy |

Exact cadence calculations in the later study must use released MJD and field metadata without looking at compact-feature values or labels.

## Frozen extractor interaction

The builder first selects all griz epochs satisfying flux `>0`, uncertainty `>0`, and S/N `>=3`. If none exist, it uses every retained epoch. This single cross-band active group defines the first and last active times. Per-band features then use active observations in that band, with an in-window fallback only when the band has no active epoch.

Consequences:

- A deeper system can move the first active epoch earlier and the last active epoch later.
- A coarser cadence can miss the true maximum and move the sampled maximum in either direction.
- Noise affects both measured flux and inclusion in the active set.
- The top-three-positive colour statistic combines band-dependent sampling times.
- More available forced epochs do not automatically produce more selected epochs because the S/N rule intervenes.

## Feature predictions

### `time_span`

Physical measurement effect: greater depth permits threshold crossing farther from maximum.

Selection effect: four-band survival and spectroscopic targeting preferentially retain brighter/longer-lived events.

Extractor effect: `time_span` is defined by the first and last positive S/N>=3 observation across all bands; fallback changes the estimand discontinuously when no active epoch exists.

Prediction: at fixed class, redshift, extinction, and valid quality mask, DES deep fields should have longer `time_span` than SDSS. No unconditional DES-shallow versus SDSS sign is registered. A common-window/depth forward model should attenuate the deep-field difference if survey physics is responsible.

Integrity note: previous feasibility work reported metadata-level active-window summaries. This prediction is therefore partially informed and supporting, not an independent primary confirmation.

### `r_time_of_peak`, `i_time_of_peak`, `z_time_of_peak`

Physical measurement effect: band response changes the effective light-curve peak phase.

Selection effect: depth changes the first active epoch, which is the zero point for all three timing features.

Extractor effect: each feature is a sampled argmax minus the cross-band active start, clipped to the observed span.

Predictions:

- Weekly DES sampling produces larger fixed-template argmax error and broader timing residuals than four-night SDSS sampling.
- DES deep fields may yield larger time-from-active-start because the active start occurs earlier.
- No unconditional cross-survey median direction is registered because coarse sampling, filter phase shifts, and boundary truncation compete.
- z timing is expected to have the largest depth/noise ambiguity.

### Peak fluxes

Physical measurement effect: passband and calibration terms change the noiseless band flux.

Selection effect: shallow/deep thresholds and spectroscopy alter the population that survives four bands.

Extractor effect: a sampled maximum has upward noise bias but downward cadence bias relative to the continuous maximum.

Prediction: for a fixed noiseless template, degrading from four-night to weekly sampling cannot increase the expected accurately sampled maximum in the absence of noise; adding measurement noise can increase the observed maximum through an extreme-value effect. Therefore passband peak predictions must be evaluated after schedule/noise forward modelling and by DES field class.

### Colour proxies

Physical measurement effect: relative passbands produce the synthetic terms in `passband_prediction_notes.md`.

Selection effect: a band must have a positive representative and the event must survive all four bands.

Extractor effect: each representative is an average of up to three largest positive measurements, generally at different times between bands.

Prediction: cadence and depth can attenuate, broaden, or locally reverse the passband-only term. A successful physics account must reproduce the redshift-dependent shape without replacing the frozen statistic by simultaneous colour.

### Standard deviations and amplitude

No safe raw direction is declared for `r_std_flux`, `i_std_flux`, `z_std_flux`, or `i_amplitude`.

Competing terms are fixed in advance:

- longer phase support tends to add low-flux tails and can increase peak-to-trough amplitude;
- positive-S/N truncation removes low and negative values;
- more samples increase the chance of noise extremes;
- smaller uncertainties reduce noise variance;
- coarse cadence can miss both extrema;
- intrinsic duration and morphology differ by class/subtype.

These features are diagnostics of whether a forward model is coherent, not primary directional tests.

### Mean fluxes

No safe raw direction is declared for `g_mean_flux` or `r_mean_flux`. A longer selected window may reduce the mean by adding tails, but positive-S/N truncation, depth, passband response, and source luminosity selection can produce the opposite result.

## Physics-only operators to be frozen before execution

1. Quality masks: use each release's documented valid-epoch mask; no mask chosen from feature alignment.
2. Schedule operator: sample fixed SED light curves at actual unlabeled field/band MJD patterns or a deterministic documented cadence pattern.
3. Noise operator: use released per-epoch uncertainty metadata or documented error models; no variance coefficient fitted from feature distributions.
4. Field stratification: DES shallow and deep are separate primary strata.
5. Boundary diagnostic: flag events whose selected first/last epochs lie near a released season boundary; do not discard them based on outcome.
6. Common-window sensitivity: use one predeclared anchor and bounds from template/release metadata, identical in definition across surveys.
7. Direction reversal control: applying the inverse schedule mapping should reverse or remove the schedule-induced component in fixed-template simulations, not improve arbitrary features indiscriminately.

## Null and failure patterns

- If fixed-template schedule injection does not produce broader DES peak-time error, the cadence prediction fails before using real compact features.
- If deep and shallow field metadata do not yield distinguishable threshold/uncertainty operators, no depth-direction claim will be made.
- If a proposed window or noise operator requires choosing parameters from SDSS-DES feature alignment, it is rejected.
- If all apparent changes are explained by label/subtype/redshift composition and not stable within strata, the observing-system hypothesis is weakened.
