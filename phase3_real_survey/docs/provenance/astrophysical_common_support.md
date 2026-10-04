# Astrophysical Common-Support Audit

## Label definitions

Only definite transient-spectroscopic labels are eligible. Host spectroscopy supplies redshift, not class.

### SDSS-II

- Normal/definite Ia: `SNTYPE in {118,120}`: 499 objects in the inspected header.
- Ordinary non-Ia: `SNTYPE in {111,115,112,113,117}`: 86 objects (Ib, Ic, II).
- Exclude probable/photometric/z-confirmed classes, `SNIa?`, variables, AGN, unknown, and SLSN from the primary binary sample.

### DES-SN5YR

- Normal Ia: `SNTYPE == 1`: 353.
- Ordinary non-Ia: `SNTYPE in {23,29,32,33,39}`: 61 (IIb, II, Ib, Ic, Ibc).
- Exclude all question-mark classes, Ia-pec, ambiguous H-free SN, SLSN, AGN, TDE, stars, and unlabeled objects from the primary sample.

## Directly measured support

The existing local `HEAD` and `PHOT` FITS files were inspected. No feature distributions or classifier outcomes were used to choose the region.

| Survey/class | Total secure | Heliocentric-z range | Median z | In `0.05-0.30` | Survive frozen four-band positive-support rule in `0.05-0.30` |
|---|---:|---:|---:|---:|---:|
| SDSS Ia | 499 | 0.0130-0.5510 | 0.2064 | 402 | 305 |
| SDSS ordinary CC | 86 | 0.0136-0.2944 | 0.0803 | 68 | 61 |
| DES Ia | 353 | 0.0176-0.8500 | 0.3307 | 140 | 134 |
| DES ordinary CC | 61 | 0.0327-0.3900 | 0.1218 | 52 | 51 |

The four-band rule mirrors only the frozen extractor's input support: within the globally S/N-active group, or its existing fallback, every `g,r,i,z` band must contain a strictly positive representative flux. No feature values were compared.

## Conservative common-support region

Predeclare for the first feasibility experiment:

1. `0.05 <= REDSHIFT_HELIO <= 0.30`.
2. Definite normal Ia versus definite ordinary II/IIb/Ib/Ic/Ibc only.
3. All four `griz` bands must satisfy the frozen builder's existing support rule.
4. Use observed fluxes with supplied `MWEBV` retained as a stratification/confounder variable; do not silently deredden one survey only.
5. Report DES shallow and deep fields separately before pooling.
6. Report SDSS Stripe 82 north and south separately as a stability check.
7. Do not match or weight on any of the 16 compact features.

This region is broad enough for both directions and narrow enough to avoid the most obvious extrapolation. It is not claimed to equalize populations.

## Temporal and sampling support

Among objects in the common redshift range that pass the four-band rule, the S/N-active span and observation support are:

| Survey/class | Active-span 10/50/90 percentiles (days) | Median released `griz` epochs | Median S/N-active epochs |
|---|---:|---:|---:|
| SDSS Ia | 21.9 / 51.9 / 82.1 | 84 | 38 |
| SDSS ordinary CC | 21.0 / 59.9 / 86.9 | 80 | 46 |
| DES Ia | 70.3 / 123.8 / 236.8 | 102.5 | 55 |
| DES ordinary CC | 120.0 / 150.7 / 310.9 | 102 | 75 |

There is overlap, but the distributions are not close. This is scientifically informative because `time_span`, peak timing, means, standard deviations, amplitudes, and asynchronous colours all depend on the effective window. It is also a major confounder. A future physics analysis must model or stratify on documented observing windows without choosing the rule from target labels.

## Remaining population risks

- **Spectroscopic targeting:** both secure samples are selected subsets. DES non-Ia spectroscopy and SDSS follow-up were not random.
- **Subtype composition:** the binary ordinary-CC class has different II/Ibc mixtures. Subtype must be reported; counts are too small for fine-grained model claims.
- **Host population:** survey depth and target selection change host mass, surface brightness, and extinction distributions.
- **Redshift is not enough:** matching z does not equalize luminosity, extinction, phase coverage, or host background.
- **Field strategy:** DES deep fields over-represent faint/high-z events and have different cadence/depth from shallow fields.

## Permitted confounder controls

Before target-label evaluation, it is permissible to define fixed bins in redshift, `MWEBV`, field/depth class, observed epoch count, and documented observing season. These controls must use metadata only. Target labels may be used once to report class-stratified outcomes, not to choose transformations, passbands, windows, thresholds, or bins that improve classification.
