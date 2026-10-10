# Stage 3C Mechanistic Amendment v1.0.1

Date: 2026-10-10

## Status

This amendment is frozen before implementation of the affected Stage 3C
production operator.

It does not overwrite or modify the original preregistration or
`stage3c_execution_lock.json`.

No Stage 3C production Monte Carlo result, registered physics outcome,
or frozen-classifier result has been generated or inspected under the
mechanistic choice specified here.

## Scope

This amendment resolves one implementation ambiguity discovered while
validating the DES-SN5YR forward-noise operator:

> How should the preregistered scalar host-surface-brightness coordinate
> `SBMAG` be related to the physical DES host morphology, SN position,
> native band-specific host surface brightness, and host-photon variance?

No other Stage 3C parameter, grid, feature definition, hypothesis,
classifier, metric, sample definition, or interpretation rule is changed.

## Original Stage 3C treatment retained

The preregistered DES host-surface-brightness treatment remains the
fixed scalar conditional grid:

    SBMAG = 20.5, 21.5, 22.5, 23.5,
            24.5, 25.5, 26.5, 27.5

The primary treatment remains a conditional grid.

No empirical host-SBMAG distribution is substituted for this grid.
No primary marginal averaging over host surface brightness is introduced.

## Native SNANA findings motivating this clarification

Source inspection and native-SNANA validation established the following.

### 1. Native DES host surface brightness is band-specific

For an observation in filter `ifilt_obs`, pinned SNANA uses:

    SBmag = SNHOSTGAL.SB_MAG[ifilt_obs]

The native physical host therefore has a vector:

    (SBMAG_g, SBMAG_r, SBMAG_i, SBMAG_z)

rather than one common event-level scalar SBMAG.

### 2. DES FLUXERRMODEL is band- and field-specific

For each epoch, SNANA identifies the observation band and field and
passes the corresponding SBMAG value through the relevant DES
FLUXERRMODEL map.

The map parameter vector includes:

    SBMAG
    GALMAG
    SNSEP
    MJD
    PSF
    SKYSIG
    ZP
    LOGSNR

as required by the selected map.

### 3. Host-photon variance depends on the physical host independently

Native host-photon variance uses the epoch PSF scale and the
host-specific GALMAG-vs-PSF curve:

    NEA
      -> PSFSIG_ASEC
      -> GALMAG_NEA
      -> host-photon variance

with:

    HOSTVAR_PE2 =
        GAIN * 10**(0.4 * (ZPT - GALMAG_NEA))

This chain has been independently validated against the pinned native
SNANA executable for 102 DES epochs.

### 4. Scalar SBMAG cannot uniquely determine physical host morphology

Native host generation includes stochastic weighted host selection,
Sersic-component selection, radial enclosed-light position, and angular
position.

The relevant native random quantities include:

    FlatRan1_GALID
    FlatRan1_radius[0]
    FlatRan1_radius[1]
    FlatRan1_phi

Thus there is no unique inverse mapping:

    scalar SBMAG
        -> host morphology
        -> SN position
        -> GALMAG(PSF)

### 5. Native-host diagnostic population confirms the non-uniqueness

A 5000-event native-SNANA diagnostic pool was generated using
`SIMGEN_DUMPALL`.

The pool showed strongly correlated but substantially different
band-specific SBMAG values within the same physical host realization.
The scalar Stage 3C SBMAG coordinate therefore cannot be identified with
a unique physically realizable four-band host state.

The 5000-event pool is diagnostic evidence only and is not adopted as
the Stage 3C production host distribution.

## Frozen mechanistic interpretation

The following rule is adopted for Stage 3C.

### A. Scalar Stage 3C SBMAG

The preregistered scalar SBMAG value is interpreted strictly as a
controlled DES FLUXERRMODEL treatment coordinate.

For a Stage 3C cell with scalar value `s`, each DES epoch uses:

    ERRPARLIST[IPAR_FLUXERRMAP_SBMAG] = s

and the appropriate band- and field-specific DES FLUXERRMODEL map is
then evaluated.

Thus the same experimental scalar coordinate is supplied to the
corresponding g, r, i, or z error map.

This is an intentional controlled treatment and is not claimed to be a
native physical four-band host-SBMAG vector.

### B. Physical host nuisance realization

Physical host morphology and SN position are treated as stochastic
observational nuisance variables.

They are generated according to the pinned native DES SNANA HOSTLIB
semantics rather than inferred from the scalar Stage 3C SBMAG value.

The physical host realization supplies:

    GALID
    Sersic morphology
    SN position / host offset
    native GALMAG_b(PSF)
    epoch GALMAG_NEA
    host-photon variance

The physical host's native:

    SBMAG_g
    SBMAG_r
    SBMAG_i
    SBMAG_z

may be recorded as diagnostics but do not replace or redefine the
preregistered scalar Stage 3C treatment coordinate.

### C. No scalar-SBMAG inverse host construction

Stage 3C will not:

- solve for a synthetic host morphology from scalar SBMAG;
- assign scalar SBMAG to one arbitrarily selected physical band;
- use mean, median, nearest-band, or other derived combinations of
  native SBMAG_g/r/i/z as an inverse host prescription;
- require SBMAG_g = SBMAG_r = SBMAG_i = SBMAG_z;
- condition a production host pool on an ad hoc distance from the
  scalar SBMAG node.

These would introduce modeling assumptions absent from the original
preregistration.

## Randomness and deterministic reproducibility

The existing Stage 3C deterministic seed construction is retained
unchanged.

A distinct purpose namespace will be used for physical DES host
realizations:

    purpose = "des-host"

with the same registered seed inputs:

    namespace
    purpose
    survey
    stratum
    z
    sbmag
    realization_id

Thus host nuisance realization remains deterministic and reproducible
for every frozen Stage 3C condition.

The existing measurement/noise seed namespace remains separate.

No existing `derive_seed()` implementation or registered seed formula
is altered by this amendment.

## Pairing interpretation

The original requirement for a common intrinsic realization identifier
is retained.

The amendment does not require physical observational nuisance
realizations to be identical between different SBMAG treatment cells.

The scalar SBMAG coordinate remains part of the registered condition,
and therefore may participate in deterministic host-nuisance and
measurement-noise seed derivation.

The same intrinsic transient realization remains paired wherever
required by the original Stage 3C design.

## DES forward-noise decomposition after this amendment

For each DES epoch:

    intrinsic transient
        |
        +--> SIMLIB cadence / observing conditions
        |
        +--> native physical host nuisance realization
        |       |
        |       +--> GALMAG_b(PSF_epoch)
        |               |
        |               +--> host-photon variance
        |
        +--> preregistered scalar SBMAG treatment
                |
                +--> BAND + FIELD DES FLUXERRMODEL
                        |
                        +--> empirical uncertainty scale

The host-photon and scalar-SBMAG treatment paths therefore remain
explicitly separated.

## Native-SNANA provenance

Pinned SNANA commit:

    4113779f038eec613b6684d0d46a47019d7c9624

The native-SNANA host-photon validation established:

- NEA -> PSFSIG_ASEC: PASS
- PSFSIG_ASEC -> GALMAG_NEA interpolation: PASS
- GALMAG_NEA -> HOSTVAR_PE2: PASS
- 102-epoch shared variance-module regression: PASS
- controlled Stage 3D.6 DES integration: PASS
- 10-event native SBMAG dump instrumentation: PASS
- 5000-event native host diagnostic pool: PASS

## Production status

This amendment does NOT itself authorize the full Stage 3C production
Monte Carlo.

Production authorization requires, after this freeze:

1. implementation of this exact decomposition;
2. controlled validation against native SNANA semantics;
3. regression of previously validated Stage 3D operators;
4. final provenance/hash freeze;
5. explicit production-authorization gate.

No classifier result may be used to modify this mechanistic rule.

## Amendment rationale

The amendment resolves an ambiguity in translating the preregistered
scalar SBMAG treatment into the validated native-SNANA host/noise
machinery.

It narrows the interpretation of the existing scalar treatment rather
than adding an outcome-driven parameter.

No affected Stage 3C production or classifier outcome was used to
choose this rule.
