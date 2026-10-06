# Stage 3B — Survey Forward-Model Provenance

## Scientific purpose

Identify survey-specific observing and uncertainty machinery for a
forward-model comparison of SDSS-II and DES-SN5YR.

These artifacts are NOT corrections to be applied to the released real
photometry. Released FLUXCAL and FLUXCALERR remain unchanged.

The artifacts are used only to describe or simulate how an underlying
transient would be observed under each survey's observing conditions.

## SDSS-II

### Real-data lineage

The official released SDSS package contains:

- SMPv8+BOSS_2005
- SMPv8+BOSS_2006
- SMPv8+BOSS_2007

The release README identifies the source photometry as:

- Holtz/v6_reCalib/2005
- Holtz/v6_reCalib/2006
- Holtz/v6_reCalib/2007

### Forward-observation artifacts

`SDSS_3year.SIMLIB`

SHA-256:
`fad8558b04c2fc10e26d18f42d016ab52311078ea87d8cfb44f1c47e3fe7c6c9`

The artifact declares:

- PURPOSE: SDSS 3-year cadence for SNANA simulation
- seasons 2005+2006+2007
- nominal intent
- use by `snlc_sim.exe`
- validation in JLA and Pantheon cosmology analyses

It contains epoch-level observing quantities including cadence, gain,
read noise, sky noise, PSF, zeropoint and zeropoint uncertainty.

`SDSS_fluxErrModel.DAT`

SHA-256:
`a69b5a69fbbcaf54e6da2f6687c2020b97973c392000103553eb59674b9e05e7`

The file explicitly states that its maps are identical to the
FLUXERR_COR / FLUXERR_ADD maps in `SDSS_3year.SIMLIB`.

The SDSS error prescription includes:

- band-dependent additive FLUXCAL uncertainty;
- S/N-dependent multiplicative uncertainty scaling.

Multiple installed SNANA SDSS simulation configurations explicitly use
`SDSS_3year.SIMLIB` to specify the SDSS survey.

### Provenance strength

STRONG SURVEY-LEVEL PROVENANCE.

The released photometry and the simulation artifacts share the same
three SDSS observing seasons and survey lineage. The release README does
not explicitly state that `SDSS_3year.SIMLIB` generated the exact
released SMP uncertainties, so it is not claimed as an exact
release-generation recipe.

## DES-SN5YR

`DES-SN5YR_DES.SIMLIB`

SHA-256:
`f5575ecd5b50c526ff63b758c5f6b88a710a185040dfd3c6e5008ca7b67d92a0`

Nominal DES five-season cadence library containing epoch-level sky,
PSF, gain and zeropoint information.

`DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT`

SHA-256:
`2b6d0898fd1992a72cfa2322a79272a9557e8fc4cb98fa3193bd8ee00d2d80d9`

Simulation flux-error model. Its scaling depends on:

- field depth group;
- passband;
- host surface brightness.

`DES-SN5YR_DES_FLUXERRMODEL_FAKE.DAT`

SHA-256:
`1f1285b56d2d7edfe6ed4d293ee96d71acc05bbe03f6d52ccf8cba5d26538afd`

The DES release README explicitly identifies this artifact as the
host-surface-brightness-dependent magnitude-error correction applied to
FAKES/DATA.

`DES-SN5YR_DES_DCR+CHROM.DAT`

SHA-256:
`1044cd913bfe28fed0982cc9a3f01d491bb211238ac51023858695fb30cd53c8`

The DES release README explicitly states that chromatic and DCR
corrections are applied and points to this artifact.

### Provenance strength

VERIFIED RELEASE-SPECIFIC PROVENANCE.

## Frozen interpretation

The survey forward model is nonlinear and conditional.

Relevant mechanisms include:

- passband response;
- observing cadence;
- sky background;
- PSF;
- gain/read noise;
- zeropoint and zeropoint uncertainty;
- signal-dependent uncertainty;
- host-surface-brightness-dependent uncertainty;
- DES shallow/deep field structure;
- DES chromatic/DCR treatment.

No generic linear SDSS-to-DES feature correction is justified.

No empirical feature discrepancy will be used to fit the forward
operator.

No released real-data flux or uncertainty value will be overwritten or
retroactively corrected.
