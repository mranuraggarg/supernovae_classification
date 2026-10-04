# Planned Measurements

Version: 1.0.0  
Frozen: 2026-10-04T14:44:46+0400

No measurement in this document has been executed.

## 1. Analysis populations

Primary population:

- `0.05 <= z_helio <= 0.30`;
- definite normal Ia and definite ordinary CC II/IIb/Ib/Ic/Ibc;
- secure transient-spectroscopic class;
- frozen four-band support rule;
- released observed fluxes and uncertainties;
- DES shallow and deep separate;
- SDSS north and south separate.

Fixed redshift bins are `[0.05,0.10)`, `[0.10,0.15)`, `[0.15,0.20)`, `[0.20,0.25)`, and `[0.25,0.30]`. These follow the common-support bounds and synthetic-prediction scale, not observed feature behavior. Continuous redshift curves will also be shown so conclusions do not depend on bin edges.

Primary normal-Ia tests evaluate the frozen Hsiao prediction. Ordinary CC is reported by broad II-like versus stripped-envelope grouping where counts permit; no Ia template correction is applied to CC.

## 2. Data handling before feature extraction

1. Verify file hashes and product versions against `preregistration_manifest.md`.
2. Apply documented release-specific valid-epoch quality masks.
3. Preserve negative forced-photometry measurements.
4. Do not require detection flags beyond the frozen S/N-active rule unless a release flag marks an invalid measurement.
5. Do not reapply DES DCR/chromatic or host-surface-brightness error corrections.
6. Do not directly add KCOR offsets to epoch FLUXCAL; calibration conversion must use the named artifact semantics.
7. Apply the unchanged compact builder.
8. Record fallback branch, selected epoch count, first/last active epoch, season-boundary proximity, and colour clipping only as audit diagnostics.

## 3. Primary estimands

### Passband features

For `peak_color_r_minus_i`, `peak_color_i_minus_z`, `i_peak_flux`, and `z_peak_flux`:

1. Estimate the class-conditional DES-SDSS median difference as a function of redshift.
2. Report 0.1, 0.5, and 0.9 quantile differences.
3. Evaluate the frozen synthetic function at each object's redshift, with the registered normal-Ia template treatment.
4. Define the residual function `R_f(z) = Delta_observed,f(z) - Delta_synthetic,f(z)`.
5. Report object-level 1-Wasserstein distance before and after the fixed synthetic operator within each redshift/field stratum.
6. Report the fraction of the original signed median displacement removed, but do not use it to tune the operator.

The primary evidence is shape and sign coherence of `Delta_observed(z)` with the preregistered function, not a single pooled distance.

### Cadence feature

For `r_time_of_peak`:

1. Forward-sample the fixed template through schedule patterns drawn from released unlabeled SDSS and DES epoch metadata.
2. Calculate sampled-argmax error relative to the continuous-template r maximum.
3. Compare median absolute error, median absolute deviation, interquartile range, and 0.1/0.9 residual quantiles.
4. In real data, report the class/redshift-conditioned distribution of `r_time_of_peak` and boundary flags, but do not interpret a median shift without the forward model.
5. Compare DES shallow and deep separately.

### Supporting `time_span`

Report class/redshift/field-conditioned median and 0.1/0.9 quantiles. Use a boundary-censoring diagnostic indicating whether the first or last selected epoch is adjacent to the available season/window. Because previous metadata work exposed active-span summaries, this result is supporting and not an independent confirmation.

## 4. Secondary and diagnostic metrics

| Purpose | Metric | Role |
|---|---|---|
| Marginal displacement | Median and quantile difference | Primary descriptive effect |
| Distribution magnitude | 1-Wasserstein distance | Primary magnitude summary after conditioning |
| CDF discrepancy | KS statistic | Secondary; never interpreted alone |
| Symmetric density discrepancy | Jensen-Shannon divergence using bins fixed before labels are read | Secondary sensitivity |
| Joint primary-feature shift | Energy distance and MMD with kernel bandwidth fixed from pooled unlabeled scale or a documented deterministic rule | Coherence diagnostic only |
| Covariance | Difference in robust correlation matrices | Secondary; no causal interpretation by itself |
| Class preservation | Within-survey Ia-versus-CC standardized separation before/after operator | Guard against erasing astrophysical structure |
| Downstream behavior | Frozen classifier discrimination/calibration without retraining | Supporting only, run after physics outcomes are frozen |

No metric will be selected or dropped because it produces a favorable outcome.

## 5. Uncertainty

- Object-level bootstrap, stratified by survey, secure class, fixed redshift bin, and field/depth stratum.
- Report pointwise and simultaneous uncertainty bands for redshift-conditioned curves.
- Keep all observations from one object in the same resample.
- Do not bootstrap individual epochs as independent objects.
- Report sensitivity to SDSS north/south and DES shallow/deep separately.
- Report small-sample limitations for ordinary CC and subtype strata.

Confidence intervals quantify uncertainty; no p-value alone defines success.

## 6. Confounder controls

Controls are specified without matching on the 16 features:

| Confounder | Primary control |
|---|---|
| Redshift/time dilation | Fixed support and bins; continuous z conditioning; rest-frame timing sensitivity |
| Class/subtype | Normal Ia primary; II-like and stripped-envelope reported separately where possible |
| Galactic extinction | Retain observed flux and condition on `MWEBV`; symmetric correction sensitivity only |
| Host extinction/population | Acknowledge unresolved; use subtype/host metadata only if defined independently of features |
| DES field depth | Shallow/deep separate before pooling |
| SDSS focal/stripe geometry | North/south separate; CCD-average nominal response with column sensitivity later |
| Spectroscopic targeting | Restrict to secure labels; do not claim population representativeness |
| Season boundary | Boundary indicator and sensitivity; no outcome-selected trimming |
| Epoch count | Report as metadata diagnostic; do not match on a compact feature |
| Pipeline/host background | Use released flags and host-surface-brightness metadata where available; report strata |
| Four-band survival | Report selection fractions using labels only after all inclusion rules are frozen |

## 7. Physics-only transformation sequence

The sequence is fixed:

1. Calibration interpretation: express both systems through their named AB/count artifacts without double application.
2. Passband operator: apply the fixed SED-derived band mapping before feature compression.
3. Cadence operator: forward-sample fixed templates through documented schedules; for a common-schedule sensitivity, degrade only according to a predeclared metadata-derived schedule.
4. Window operator: apply one common declared phase/window rule if the raw-release windows are not commensurate.
5. Noise/depth operator: use released per-epoch uncertainty/observing metadata, never fitted feature-distance coefficients.
6. Recompute frozen features unchanged.

Each operator is assessed separately before any combined operator. An operator is not retained merely because a classifier improves.

## 8. Interpretation rules

Strong coherent indication requires all of the following:

- at least three independent primary features show behavior consistent with their preregistered functions or spread predictions;
- the fixed physics operator reduces conditioned residuals/distances without being tuned to them;
- behavior is not confined to one DES field class or one redshift bin;
- class separation is not materially erased;
- conclusions survive the declared redshift, extinction, field, and boundary controls.

Weak/ambiguous indication applies when signs agree but uncertainty is broad, field dependence is unexplained, or only correlated colour features respond.

Evidence against the hypothesis applies when the observed conditioned functions are opposite to stable preregistered predictions, physics-only operators do not reduce the relevant residuals, or apparent alignment requires target-guided choices.

The preregistration does not set an arbitrary percentage reduction threshold. Effect sizes, uncertainty bands, and consistency across mechanisms will be reported in full.

## 9. Execution order and blinding

1. Reproduce synthetic calculations and schedule injections without loading class labels or compact features.
2. Freeze quality masks and all operator parameters.
3. Build features for included objects without cross-survey summaries.
4. Unblind secure classes only to apply the frozen sample definition and class-conditioned reports.
5. Compute the registered metrics.
6. Run the frozen classifier only after the physics analysis is immutable.

Any deviation requires a dated amendment that is visibly separate from this version and made before inspecting the affected outcome.
