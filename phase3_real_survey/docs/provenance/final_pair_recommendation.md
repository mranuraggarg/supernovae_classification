# Final Pair Recommendation

## Final gate

**CONDITIONAL GO**

## Recommended pair

**SURVEY A:** SDSS-II Supernova Survey final three-season Scene Modelling Photometry release associated with Sako et al. 2018; official NERSC SNANA product `SDSS_dataRelease-snana.tar.gz`, audited locally as `SDSS_allCandidates+BOSS`.

**SURVEY B:** DES-SN5YR full-transient Scene Modelling Photometry release, Zenodo DOI `10.5281/zenodo.12720778`; use the immutable SMP release files, not unpinned repository `main`.

**PRIMARY LABEL DEFINITION:** definite transient-spectroscopic normal Ia versus definite ordinary core-collapse II/IIb/Ib/Ic/Ibc. Exclude probable, peculiar, ambiguous, photometric, SLSN, AGN, TDE, stellar, and unlabeled events from the primary binary evaluation.

**COMMON REDSHIFT SUPPORT:** `0.05 <= z_helio <= 0.30`, predeclared from label availability and physical overlap rather than from compact-feature distributions.

**COMMON ASTROPHYSICAL POPULATION:** normal Ia and ordinary core-collapse SNe with secure transient spectra and four-band `griz` support in both releases. Subtypes and host/field strata remain reportable confounders rather than being silently pooled.

**KEY INSTRUMENTAL DIFFERENCES:** Sloan 2.5 m drift-scan imager versus Blanco/DECam; distinct measured `griz` responses and calibration systems; three autumn Stripe 82 seasons versus five DES seasons; approximately four-night versus roughly weekly cadence; single stripe versus shallow/deep field strategy; different depth, sky, PSF, and host-subtraction/error behavior.

## Why this pair is preferred

1. Both sides are real, homogeneous survey products with forced scene-modelling fluxes, negative measurements, exact epochs, uncertainties, and rich per-epoch observing metadata.
2. Both have usable definite Ia and ordinary non-Ia spectroscopy in a measured common-support region. After the frozen four-band support rule, the region contains SDSS 305 Ia/61 CC and DES 134 Ia/51 CC.
3. The observing systems differ enough to make a physics comparison meaningful even without a classifier, while retaining common `griz` wavelength coverage and the unchanged 16-feature formulas.

## Known limitations

- The final SDSS light-curve release is not yet explicitly linked, in one inspected official manifest, to the local Doi/KCOR response artifact. This is the condition preventing GO.
- DES release/correction state must be pinned because current mutable documentation and the 2024 paper disagree about applied chromatic/DCR corrections.
- Spectroscopic targeting, redshift, subtype, host, and DES shallow/deep composition can mimic survey effects.
- Active temporal support differs strongly: roughly 52-60 day medians in SDSS versus 124-151 days in DES for the common-support samples.
- The frozen colours are asynchronous top-three-positive flux ratios, not simultaneous physical colours.

## Label-blind transformation feasibility

**YES, conditional on provenance closure.** Instrument curves, calibration products, epoch metadata, field/depth indicators, and unlabeled cadence/noise summaries are available without target labels. No transformation needs to be selected using DES or SDSS classification outcomes.

## Physics-first test

If XGBoost were removed, this would still be a meaningful observational/instrumental experiment. It asks how two documented, calibrated observing systems map overlapping spectroscopic SN populations into sampled temporal, flux, colour-proxy, and dispersion measurements. The classifier is only a later fixed diagnostic.

## Single smallest next action

Recover the official `SDSS_dataRelease-snana.tar.gz` on a case-sensitive filesystem and inspect its bundled README/calibration assets for an explicit link from `SMPv8+BOSS` to the Doi 2010 response/KCOR set. Record archive and component SHA-256 checksums. If that link is absent, request only that exact calibration provenance statement from the SDSS release maintainers; do not substitute a plausible SDSS curve.

Until this action succeeds, passband-derived mathematical normalization should not begin.
