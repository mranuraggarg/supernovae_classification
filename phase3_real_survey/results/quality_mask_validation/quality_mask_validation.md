# Phase 3 Quality-Mask Validation

**No compact features, survey-domain distances, or classifier outcomes were calculated.**

## Gate: PASS

## Frozen structural mask

For both surveys:

1. Keep canonical `g,r,i,z` photometry rows only.
2. Require finite MJD.
3. Require finite FLUXCAL.
4. Require finite FLUXCALERR.
5. Require `FLUXCALERR > 0`.
6. Retain legitimate negative and zero FLUXCAL measurements.
7. Apply no S/N cut at this stage.
8. Apply no detection/PHOTPROB threshold.
9. Apply no empirical PHOTFLAG rejection.

The existing frozen builder will later apply its registered `flux > 0`, `flux_err > 0`, and `S/N >= 3` active-observation logic. That is a feature-extraction rule, not a release-quality mask.

## SDSS-II

- Total PHOT rows: 1,120,566
- Canonical griz rows: 894,594
- Separator '-' rows: 10,258
- Structural-valid rows: 894,594
- Non-positive-error rows outside '-' separators: 0
- Valid negative-flux rows retained: 161,158

## DES-SN5YR

- Total PHOT rows: 1,798,736
- Canonical griz rows: 1,779,030
- Separator '-' rows: 19,706
- Structural-valid rows: 1,779,030
- Non-positive-error rows outside '-' separators: 0
- Valid negative-flux rows retained: 517,801

## PHOTFLAG rule

PHOTFLAG values were enumerated descriptively only.

No PHOTFLAG bit is excluded unless the selected release documentation explicitly identifies it as an invalid measurement/image condition.

Observed flag frequency is not evidence of invalidity.

## Next step

If this gate passes, the next pre-outcome task is cadence-template injection using released unlabeled observing schedules.
