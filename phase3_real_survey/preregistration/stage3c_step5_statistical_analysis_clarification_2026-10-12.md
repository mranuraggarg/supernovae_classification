# Stage 3C — Step 5 Statistical Analysis Clarification

**Date:** 2026-10-12

**Status:** POST-PRODUCTION, PRE-SCIENTIFIC-OUTCOME
METHODOLOGICAL CLARIFICATION

**Scope:** Stage 3C controlled SDSS-II versus DES-SN5YR
forward-model experiment.

## 1. Purpose and provenance

This document specifies the statistical analysis of the
completed Stage 3C production Monte Carlo experiment.

The production simulation was completed before this
clarification was written.

The clarification is not represented as part of the
original preregistration.

The production dataset contains 176,000 records:
172,822 accepted and 3,178 rejected.

The aggregate acceptance and rejection counts, code
structure, metadata availability, and simulation integrity
were examined before this clarification.

Scientific feature distributions, redshift-dependent
feature differences, and classifier outcomes have not
been examined for the purpose of choosing these rules.

This document does not authorize modification of the
frozen simulation, feature extractor, or classifier.

## 2. Scientific scope

Stage 3C tests whether differences between frozen
SDSS-II and DES-SN5YR survey operators can generate
differences in extracted features for a controlled
Hsiao SN Ia source model.

The experiment does not estimate real supernova
population differences.

It does not test Ia versus non-Ia classification
performance.

Its results must not be presented as an independent
real-survey validation.

## 3. Frozen experimental design

Redshift nodes:

0.050, 0.075, 0.100, 0.125, 0.150, 0.175,
0.200, 0.225, 0.250, 0.275, 0.300.

Host surface-brightness coordinates:

20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5.

Survey strata:

- SDSS-II 82N
- SDSS-II 82S
- DES-SN5YR SHALLOW
- DES-SN5YR DEEP

Each survey/redshift/surface-brightness design cell
contains 500 original Monte Carlo realizations.

All four cross-survey comparisons are retained:

1. DES SHALLOW minus SDSS 82N
2. DES SHALLOW minus SDSS 82S
3. DES DEEP minus SDSS 82N
4. DES DEEP minus SDSS 82S

No post hoc pooling of these four comparisons is
permitted in the primary analysis.

## 4. Registered feature outcomes

The five primary registered physics features are:

- peak_color_r_minus_i
- peak_color_i_minus_z
- i_peak_flux
- z_peak_flux
- r_time_of_peak

time_span is supportive.

The original registered predictions and their
directional interpretations remain unchanged.

No additional primary feature may be selected
after examining Stage 3C feature outcomes.

## 5. Primary location statistic

For feature f, redshift z, surface-brightness
coordinate b, and survey pair (D,S), define:

Delta_f(z,b,D,S) =
    median[f | DES D, accepted]
    -
    median[f | SDSS S, accepted].

This is the difference between two marginal medians.

It is NOT the median of realization-level paired
feature differences.

Each median uses accepted realizations from its
corresponding survey cell.

Rejected realizations are never assigned artificial
feature values.

## 6. Rejection and missingness rules

All 500 original realization identifiers remain
in the design-cell denominator.

Rejected realizations are not replaced.

The acceptance fraction for each survey cell is:

N_accepted / 500.

For each SDSS-DES comparison, report:

- SDSS acceptance count and fraction
- DES acceptance count and fraction
- Number of realization IDs accepted in both branches

Feature summaries use accepted observations only.

No imputation of rejected feature vectors is allowed.

If a bootstrap replicate has no accepted observations
in either required survey cell, its statistic is undefined.

Such occurrences must be counted and disclosed.

No undefined statistic may be silently replaced by zero,
discarded without accounting, or converted to a
scientific conclusion.

## 7. Bootstrap design

Use 5,000 bootstrap replicates.

Within each redshift and surface-brightness cell,
resample the original 500 realization identifiers
with replacement.

For a given bootstrap replicate, apply the SAME
resampled realization identifiers to all four
survey branches.

This preserves the registered shared-realization
design while retaining branch-specific acceptance
and rejection outcomes.

Compute each survey's marginal median using its
accepted observations in the resampled identifiers.

Compute DES-minus-SDSS differences of these
marginal medians.

Bootstrap resampling is independent across the
11 redshift nodes.

The same replicate index is used to assemble the
11-node redshift curve for each survey pair,
surface-brightness coordinate, and feature.

Use NumPy PCG64 with base seed 20261006.

Derive independent cell-level random streams using
SHA-256 of the UTF-8 string:

stage3c-step5-bootstrap-v1|20261006|z=<z:.3f>|sbmag=<b:.1f>

Interpret the first eight digest bytes as an
unsigned big-endian integer and initialize PCG64
with that integer.

Generate exactly 5,000 index vectors of length
500 per redshift/surface-brightness cell.

The resampling algorithm must be fixed before
scientific outcome analysis.

## 8. Pointwise confidence intervals

For each feature, survey pair, redshift, and
surface-brightness coordinate, calculate a
two-sided 95% percentile bootstrap interval.

Use the empirical 0.025 and 0.975 quantiles
of the 5,000 bootstrap differences.

Report the original-sample point estimate
alongside the interval.

These are conditional Monte Carlo intervals
under the frozen simulation design.

They do not quantify uncertainty in the true
astrophysical population or survey calibration.

## 9. Simultaneous redshift confidence bands

For each feature, surface-brightness coordinate,
and survey pair, construct a simultaneous
11-node redshift confidence band.

Let Delta_hat(z) be the original-sample
DES-minus-SDSS median difference.

For bootstrap replicate k define:

M_k = max_z |Delta_star_k(z) - Delta_hat(z)|.

Let c_0.95 be the empirical 95th percentile
of the 5,000 values M_k.

The simultaneous band is:

[Delta_hat(z) - c_0.95,
 Delta_hat(z) + c_0.95]

at every registered redshift node.

This is an unstandardized maximum-absolute-deviation
bootstrap band.

It provides simultaneous coverage across the
11 redshift nodes within one feature, one
surface-brightness coordinate, and one survey pair.

It does NOT provide familywise coverage across
different features, surface-brightness coordinates,
or survey pairs.

No outcome-guided selection between standardized
and unstandardized bands is permitted.

If any replicate contains an undefined node,
the simultaneous band is not calculated for
that combination; the failure is reported.

## 10. Distributional comparisons

For registered non-timing features, report:

- DES-minus-SDSS difference of medians
- Difference in the 0.1 quantiles
- Difference in the 0.9 quantiles
- First Wasserstein distance, W1

Use the accepted marginal feature distributions.

W1 is the uncorrected distance between the
two Stage 3C survey distributions.

Do not describe it as a post-correction residual
unless the registered synthetic correction has
actually been applied.

Any comparison with earlier passband-only
predictions must first establish that the
feature definitions and transformations match.

Incompatible quantities must not be subtracted
to construct a synthetic residual.

## 11. Timing-reference definition

The frozen r_time_of_peak feature is a time
relative to the first active observation MJD.

For each accepted production record:

t_r_selected =
    feature_audit.first_active_mjd
    +
    features.r_time_of_peak.

Use the noiseless Hsiao template and the frozen
survey-specific r-band response to define the
continuous-model reference peak.

For each survey S and redshift z, evaluate the
frozen synthetic-flux proxy at integer rest-frame
phases:

p = -20, -19, ..., 85.

Use the exact synthetic_flux_proxy implementation
from reproduce_passband_predictions.py and the
same frozen Hsiao template and response curves
used by the production simulation.

Define phi_r_max(S,z) as the earliest integer
phase attaining the maximum finite positive
synthetic flux.

If no valid positive flux exists, the reference
is undefined and must be flagged.

The physical r-band peak-recovery error is:

e_r =
    t_r_selected
    -
    peak_mjd
    -
    (1 + redshift) * phi_r_max(S,z).

The reference phase is survey-specific and
redshift-specific.

Do not assume that SDSS and DES share the same
r-band peak phase.

Noiseless reference evaluation is not a new
Monte Carlo realization.

It does not modify the frozen simulation.

## 12. Timing statistics

The registered timing prediction concerns
broader peak-recovery errors for DES, not a
directional shift in median peak time.

For each survey/redshift/surface-brightness cell,
report the following for e_r:

- Median signed error
- Median absolute error
- Median absolute deviation about the median
- Interquartile range
- 0.1 and 0.9 quantiles

The primary directional timing assessment is
the DES-minus-SDSS difference in median absolute
deviation of physical peak-recovery errors.

Positive values indicate broader DES timing
errors under this dispersion definition.

Report the DES-minus-SDSS difference in
interquartile range as a corroborating
dispersion statistic.

Do not use a difference in median signed
timing errors as evidence of broader uncertainty.

Bootstrap timing statistics using the same
paired-realization resampling procedure.

Timing dispersion differences receive the
same pointwise and simultaneous interval
construction as the location statistics.

## 13. Interpretation and reporting

Report all registered redshift nodes and all
surface-brightness coordinates.

Do not choose a favorable redshift interval,
host-brightness coordinate, or survey pairing
after examining outcomes.

Present acceptance fractions alongside
feature-distribution results.

Do not equate conditional simulation uncertainty
with uncertainty from template mismatch,
calibration error, or real population diversity.

A result inconsistent with a registered
directional prediction is a legitimate
scientific result.

A null or inconclusive result must be retained
and reported.

## 14. Analysis boundary

This clarification authorizes Step 5 analysis
only after the clarification and its integrity
record have been written and verified.

It does not authorize:

- Changes to the production simulation
- Changes to the frozen feature extractor
- Resampling rejected cases to replace them
- Outcome-guided feature selection
- Classifier training or evaluation
- Claims about real-survey population performance

Stage 6 remains separately gated.

## 15. Provenance statement

This document was written after production
completion and after aggregate production
integrity and metadata-availability checks.

It was written before examining the scientific
feature-distribution outcomes.

All deviations from the original preregistration
must be described as post-production,
pre-outcome statistical clarifications.
