# Phase 3 Pre-Outcome Clarification: Four-Band Support and SDSS Filter Remapping

## Status

This clarification was written after implementation-level validation of the
pre-registered support rule and before inspection of SDSS-DES compact-feature
distributions, classifier performance, or cross-survey feature outcomes.

It does not replace or edit the original Phase 3 pre-registration. Its purpose
is to record an implementation ambiguity discovered during pre-outcome
validation and to freeze its treatment before outcome inspection.

## 1. Historical common-support calculation

The pre-registration provenance documents report the following common-support
counts after the "frozen four-band positive-support rule":

- SDSS: 305 secure normal Ia and 61 secure ordinary core-collapse SNe.
- DES: 134 secure Ia and 51 secure ordinary core-collapse SNe.

A forensic reconstruction reproduced the SDSS 305/61 count exactly when the
support calculation was implemented as:

1. use literal lowercase SDSS `g,r,i,z` filter tokens;
2. construct the globally S/N-active group using the frozen S/N >= 3 rule; and
3. require all four lowercase bands to have qualifying active support.

Therefore the historical 305/61 count has a reproducible computational origin
and is not a transcription error.

## 2. Difference from the frozen Tier-4 feature builder

Implementation validation showed that the historical common-support calculation
is stricter than the unchanged Tier-4 feature-building operator.

The Tier-4 operator contains its pre-existing fallback behavior. In particular,
lack of S/N-active support in an individual band does not necessarily reject an
object if the frozen fallback can supply the positive representative required
by the feature definitions.

Applying the actual frozen support/fallback semantics therefore admits a larger
SDSS sample than the historical 305/61 common-support calculation.

This difference was discovered before inspection of cross-survey feature
distributions or classifier outcomes.

## 3. SDSS uppercase/lowercase filter semantics

The official SDSS SNANA release explicitly contains ten filter identifiers:

`ugrizUGRIZ`

The release documentation states that the second `UGRIZ` set is used for a
small subset of SNe associated with the 82S/82N overlap. It also documents:

`FILTER_REMAP = 'UGRIZ -> ugriz'`

as the mechanism for including the overlap epochs in an `ugriz` analysis.

A pre-outcome audit of the secure common-redshift sample found:

- 402 Ia and 68 ordinary CC objects before four-band support selection;
- 21 Ia and 4 CC objects contain both lowercase and uppercase observations;
- no secure common-redshift object contains uppercase observations without
  lowercase observations;
- no same-object lowercase/uppercase near-epoch duplicate pairs were found.

Thus uppercase observations are legitimate release observations rather than
accidental duplicate rows. Excluding them cannot be justified merely by their
uppercase naming.

## 4. Frozen treatment for the confirmatory analysis

To avoid changing the originally registered primary population after seeing
empirical outcomes, the historical support definition is retained for the
primary confirmatory cohort:

- literal lowercase `g,r,i,z`;
- globally S/N-active support using the frozen S/N >= 3 criterion;
- qualifying active support required in all four bands.

This retains the originally documented SDSS primary cohort of 305 Ia and
61 ordinary CC objects.

The unchanged frozen 16-feature operator is then applied to this selected
cohort. No feature definition is changed to reproduce the cohort count.

## 5. Pre-declared sensitivity analyses

Two implementation alternatives identified during pre-outcome validation are
retained as sensitivity analyses rather than being used to redefine the primary
cohort:

1. **SDSS filter-remapping sensitivity**
   - remap `UGRIZ -> ugriz` as documented by the SDSS release before applying
     the support calculation;

2. **Frozen-builder fallback sensitivity**
   - use the unchanged Tier-4 builder's existing per-band fallback semantics
     rather than requiring active support independently in every band.

These analyses test whether conclusions depend materially on the historical
support implementation.

They must not be used to select whichever cohort gives stronger agreement with
the pre-registered physics predictions.

## 6. Interpretation

The distinction is therefore:

- **305/61** is the reconstructed historical pre-registered SDSS
  common-support cohort;
- SDSS `UGRIZ` observations are valid release observations for which the
  release provides an explicit remapping mechanism;
- the frozen Tier-4 feature builder permits broader survival through its
  pre-existing fallback behavior.

These facts are retained simultaneously rather than retroactively redefining
either the historical support calculation or the frozen feature operator.

## 7. Outcome-blinding statement

At the time this clarification was frozen:

- no SDSS-DES compact-feature distribution comparison had been inspected;
- no cross-survey classifier result had been inspected;
- no empirical feature shift had been used to choose among the support
  implementations;
- the clarification was motivated solely by provenance and implementation
  reconciliation during the pre-outcome validation stage.

Accordingly, the primary/sensitivity distinction above is frozen before
empirical outcome inspection.
