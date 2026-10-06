# Stage 3C — Frozen Survey Forward-Observation Operator Specification

## Status

PRE-EXECUTION SPECIFICATION.

This document is written after Stage 3B provenance has been frozen and
before execution of the survey forward-observation operator.

No parameters in this specification may be estimated from Stage-1 or
Stage-2 cross-survey feature discrepancies.

No classifier results, secondary features, or class-separation outcomes
may be used to define or tune this operator.

---

# 1. Scientific question

For the same underlying supernova signal, how differently would SDSS-II
and DES-SN5YR observe that signal because of their documented:

- passbands;
- cadence;
- observing windows;
- sky background;
- PSF;
- gain/read noise;
- zeropoint and zeropoint uncertainty;
- measurement-error prescription;
- host-surface-brightness error behaviour;
- field-depth structure;
- released correction state?

The purpose is not to force the two surveys to agree.

The purpose is to propagate documented survey physics through the same
frozen feature extractor and quantify the survey-induced feature shift.

---

# 2. Fundamental operator

For an intrinsic spectral-temporal transient signal S(lambda,t), define

    O_survey[S] -> {MJD, band, FLUXCAL, FLUXCALERR}

where survey is either SDSS-II or DES-SN5YR.

The compact-feature vector is then

    X_survey = F_frozen(O_survey[S])

where F_frozen is the already validated and frozen compact-feature
extractor.

The comparison quantity is therefore

    Delta_X = X_DES - X_SDSS

for the same underlying input transient.

No direct feature-space regression between SDSS and DES is permitted.

---

# 3. Intrinsic transient source

The same intrinsic transient representation must be supplied to both
survey operators.

The transient source must be independent of Stage-1/Stage-2 empirical
cross-survey feature discrepancies.

A template/model may be used only if its identity and version are fixed
before execution.

Any intrinsic model parameters used for the primary forward comparison
must be specified before running the operator.

No parameters may be tuned to improve agreement with observed SDSS-DES
feature differences.

---

# 4. Redshift handling

Forward calculations are evaluated only in the already frozen common
redshift support:

    0.05 <= z <= 0.30

The registered grid is:

    0.050
    0.075
    0.100
    0.125
    0.150
    0.175
    0.200
    0.225
    0.250
    0.275
    0.300

The same intrinsic transient and redshift must be used for the SDSS and
DES branches of each paired forward calculation.

Cosmological or redshift-dependent transformations must therefore act
identically before the survey-specific observation branch except for
survey passband sampling.

---

# 5. Passband operator

Use the already pinned SDSS and DES response functions.

The passband transformation must operate on the spectral-temporal
signal before compact-feature construction.

No empirical correction derived from Stage-1 passband residuals may be
introduced.

The already frozen passband prediction machinery remains the registered
passband component of the operator.

---

# 6. Cadence and observing-window operator

## SDSS-II

Use the pinned SDSS 3-year survey cadence representation:

    SDSS_3year.SIMLIB

SHA-256:

    fad8558b04c2fc10e26d18f42d016ab52311078ea87d8cfb44f1c47e3fe7c6c9

The library represents seasons 2005, 2006 and 2007.

## DES-SN5YR

Use:

    DES-SN5YR_DES.SIMLIB

SHA-256:

    f5575ecd5b50c526ff63b758c5f6b88a710a185040dfd3c6e5008ca7b67d92a0

The library represents all five DES seasons.

The operator must preserve actual SIMLIB epoch structure rather than
replace it with a fitted mean cadence.

Survey observing-window boundaries are part of the operator.

---

# 7. Epoch-level instrumental state

Where supplied by the relevant SIMLIB, use the survey-specific epoch
metadata directly.

Potential components include:

- MJD;
- band;
- gain;
- read noise;
- sky noise;
- PSF;
- zeropoint;
- zeropoint uncertainty;
- field/depth stratum.

No linear combination of these variables will be fitted to real
compact-feature differences.

---

# 8. Measurement-noise operator

The measurement stage is stochastic.

A generic deterministic linear SDSS-to-DES correction is forbidden.

The nominal epoch measurement is represented conceptually as

    F_obs = F_true + epsilon

where epsilon is generated according to the documented survey-specific
measurement variance/error machinery.

The exact implementation must follow SNANA/SIMLIB semantics rather than
invent an independent Gaussian variance formula when SNANA already
defines the calculation.

Conditional Gaussian realization at the final flux-measurement stage is
permitted only as the realization implied by the documented
measurement uncertainty.

Poisson/source/sky/read-noise contributions remain part of the
survey-specific variance construction.

---

# 9. SDSS uncertainty model

Use the SDSS survey model:

    SDSS_fluxErrModel.DAT

SHA-256:

    a69b5a69fbbcaf54e6da2f6687c2020b97973c392000103553eb59674b9e05e7

The model is the external representation of the FLUXERR_ADD and
FLUXERR_COR maps contained in SDSS_3year.SIMLIB.

The model includes:

1. band-dependent additive FLUXCAL uncertainty;
2. S/N-dependent multiplicative uncertainty scaling.

This dependence is nonlinear in measurement S/N.

The model is treated as strong survey-level forward-model provenance,
not claimed as a byte-for-byte reconstruction of the released SMP
error-generation pipeline.

---

# 10. DES uncertainty model

For simulated DES observations use:

    DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT

SHA-256:

    2b6d0898fd1992a72cfa2322a79272a9557e8fc4cb98fa3193bd8ee00d2d80d9

Its error scaling depends on:

- field-depth group;
- band;
- host-galaxy surface brightness.

The shallow fields are:

    S1 S2 C1 C2 X1 X2 E1 E2

The deep fields are:

    X3 C3

The DATA/FAKE correction artifact

    DES-SN5YR_DES_FLUXERRMODEL_FAKE.DAT

SHA-256:

    1f1285b56d2d7edfe6ed4d293ee96d71acc05bbe03f6d52ccf8cba5d26538afd

documents the correction already represented in the released real-data
uncertainties.

It must NOT be reapplied to released DES FLUXCALERR values.

---

# 11. DES DCR/chromatic state

The artifact

    DES-SN5YR_DES_DCR+CHROM.DAT

SHA-256:

    1044cd913bfe28fed0982cc9a3f01d491bb211238ac51023858695fb30cd53c8

documents the DES DCR/chromatic correction state.

The released DES photometry already contains the documented
DCR/chromatic correction.

Therefore the correction must not be applied a second time to released
real photometry.

For synthetic forward observations, treatment must reproduce the
documented DES simulation/release convention rather than introduce a
new fitted chromatic correction.

---

# 12. Host surface brightness

DES forward uncertainty depends explicitly on host surface brightness.

Host surface brightness therefore cannot be silently replaced by a
single constant if the primary operator uses the DES flux-error model.

If a common host-surface-brightness distribution is required for paired
SDSS/DES simulation, that distribution must be fixed before execution
and must not be inferred from Stage-1 feature residuals.

Results must either:

1. condition on fixed host-SB values/grid; or
2. marginalize over a predeclared host-SB distribution.

The chosen primary treatment must be stated before execution.

---

# 13. Survey strata

DES shallow and deep observations remain separate identifiable strata.

SDSS 82N and 82S remain separate identifiable strata where supported by
the SIMLIB.

The primary operator must retain these identities.

Pooling may be reported only after stratum-level outputs exist.

---

# 14. Monte Carlo realization

Because the observation operator is stochastic, a single realization
must not define the survey effect.

For each fixed intrinsic transient/redshift/stratum condition, use a
fixed preregistered number of stochastic realizations.

Random seeds must be deterministic and recorded.

The same intrinsic input realization identifier must be paired across
surveys.

The operator must report distributions of resulting feature shifts,
not only one simulated value.

---

# 15. Frozen compact-feature extraction

Every synthetic observed light curve must be passed through the exact
same frozen feature extractor already validated before real-data
analysis.

No survey-specific feature definition is permitted.

The frozen:

- S/N-active rule;
- fallback behaviour;
- four-band support rule;
- representative-flux construction;
- compact-feature definitions

must remain unchanged.

---

# 16. Primary Stage-3C feature scope

The forward-model comparison remains restricted initially to the five
already preregistered primary quantities:

- peak_color_r_minus_i
- peak_color_i_minus_z
- i_peak_flux
- z_peak_flux
- r_time_of_peak

No secondary compact feature may be examined before the primary
forward-model analysis is complete.

---

# 17. Output quantities

For each feature, redshift and declared survey stratum, report the
paired distribution of

    Delta_forward = feature_DES - feature_SDSS

from the forward model.

For the four passband/flux features report at minimum:

- median;
- 10th percentile;
- 90th percentile;
- Monte Carlo uncertainty interval.

For r_time_of_peak report survey-induced differences in:

- median;
- MAD;
- IQR;
- 10th and 90th percentiles.

The complete simulated distribution must remain available for audit.

---

# 18. Comparison with Stage-1 observations

Only after the forward operator is frozen and executed may its
predictions be compared with the already frozen Stage-1 real-data
differences.

The forward model must not be refitted after that comparison.

The comparison asks whether the magnitude, direction and redshift
dependence of the observed survey differences are compatible with the
documented survey observation process.

Failure to reproduce an observed difference is a scientific result and
must not trigger post-hoc tuning of the primary operator.

---

# 19. Prohibited operations

The following are prohibited in the primary Stage-3 forward model:

- linear SDSS-to-DES feature normalization fitted from real data;
- polynomial feature-space normalization fitted from real data;
- spline normalization fitted from real data;
- quantile mapping fitted from SDSS/DES compact features;
- domain adaptation using survey labels;
- parameter optimization against Stage-1 residuals;
- classifier-driven tuning;
- secondary-feature inspection before completion of the primary test;
- reapplication of corrections already included in released
  photometry or uncertainties.

---

# 20. Interpretation boundary

The forward operator estimates the expected survey-induced component of
feature differences.

It does not prove that every remaining difference is intrinsic
astrophysics.

Residual differences may still arise from:

- population selection;
- spectroscopic targeting;
- host-population differences;
- extinction;
- subtype mixture;
- imperfect forward-model fidelity;
- finite-sample variation;
- other survey-selection effects.

These must remain distinct from the instrumental forward model.

---

# 21. Execution gate

No Stage-3C/3D forward simulation may begin until all of the following
are fixed:

1. intrinsic transient template/model and version;
2. intrinsic parameter treatment;
3. host-surface-brightness treatment;
4. number of stochastic realizations;
5. random-seed rule;
6. exact SIMLIB sampling rule;
7. exact implementation of SNANA error semantics;
8. output schema.

These choices must be frozen before examining the resulting synthetic
feature distributions.



# 22. Frozen execution choices

The following choices are fixed before execution and are also recorded in

    preregistration/stage3c_execution_lock.json

## 22.1 Intrinsic source

The primary normal-Ia source is:

    Hsiao07.dat

SHA-256:

    6bd032004eccc7b5b52f7c021f0c4b541d801a70195028bf5af0a2578f554d78

The model is used as a fixed normal-Ia mean spectral-temporal template.

Rest-frame phases are the already registered daily grid:

    -20 through +85 days inclusive

No SALT2/SALT3 x1, colour, or population parameter is fitted.

No intrinsic parameter is estimated from SDSS-DES feature differences.

## 22.2 Host-surface-brightness treatment

The DES host-surface-brightness dependence is evaluated conditionally
at the native SBMAG grid of the frozen DES simulation error model:

    20.5
    21.5
    22.5
    23.5
    24.5
    25.5
    26.5
    27.5

These values are not weighted by the observed DES host population in
the primary Stage-3C calculation.

Primary results therefore remain conditional on host surface brightness.

No empirical host-SB distribution is estimated from Stage-1 or Stage-2
feature outcomes.

## 22.3 Survey strata

SDSS-II:

    82N
    82S

DES-SN5YR:

    SHALLOW
    DEEP

The four SDSS-DES stratum combinations are retained separately.

## 22.4 SIMLIB sampling rule

Actual SIMLIB entries are used.

No fitted mean cadence or synthetic regular cadence is substituted.

Within each survey stratum:

1. eligible LIBIDs are sorted numerically;
2. a deterministic seeded permutation is constructed;
3. realization identifiers select entries from that permutation;
4. if the requested number exceeds the number of available LIBIDs,
   successive deterministic permutations are used.

The SDSS and DES schedule selections are independent because the
surveys have different observing strategies.

For a paired realization, however, both branches use exactly the same:

- intrinsic template;
- redshift;
- host-SB condition identifier;
- realization identifier.

## 22.5 Monte Carlo count

For every fixed

    redshift x SDSS stratum x DES stratum x host-SB

condition, use:

    500 paired stochastic realizations

with realization identifiers:

    0 ... 499

The number of realizations may not be changed after outcome inspection.

## 22.6 Random-number rule

The registered generator is:

    numpy.random.PCG64

All seeds are derived deterministically from SHA-256 strings beginning
with the namespace:

    phase3-stage3c-v1

The canonical seed key is:

    namespace|purpose|survey|stratum|z|sbmag|realization_id

The first eight bytes of the SHA-256 digest are interpreted as an
unsigned 64-bit integer.

Intrinsic realization identifiers are shared between paired SDSS and
DES calculations.

Survey measurement-noise realizations are independently seeded for
SDSS and DES; identical random noise is not imposed on the two surveys.

## 22.7 Real-data boundary

Stage 3C does not overwrite or correct released real-data FLUXCAL or
FLUXCALERR values.

DES DCR/chromatic and host-surface-brightness corrections already
contained in the released real-data product are not applied again.

The frozen survey-error artifacts are used only in the synthetic
forward-observation calculation.

## 22.8 Blinding boundary

None of the choices above may be changed because of:

- Stage-1 feature displacement;
- Stage-2 results;
- classifier performance;
- class separation;
- improved SDSS-DES agreement.

Any later alternative is a separately declared sensitivity analysis
and cannot replace the frozen primary Stage-3C operator.
