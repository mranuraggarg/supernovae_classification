# Open Questions and Hard Conditions

## Must be resolved before a physics-derived transformation

1. **SDSS release-to-response link.** Obtain direct evidence that the final 10,258-object `SMPv8+BOSS` release is calibrated for use with the Doi 2010 CCD-specific/CCD-averaged response functions embedded in `kcor_SDSS_Bessell90_BD17.fits.gz`. The current release README names `Holtz/v6_reCalib` but not the KCOR or passband files.

2. **Pin the DES product.** Select the immutable DES-SN5YR SMP release files and record their checksums. Do not use mutable repository `main` unless a commit is pinned. Resolve whether DCR/chromatic corrections and Fragilistic/AB offsets are already applied to that exact flux product.

3. **SDSS case-collision recovery.** The local macOS extraction cannot preserve distinct `u/U`, `g/G`, and `i/I` ASCII filenames. Use the embedded KCOR `FilterTrans` table or recover the original archive on a case-sensitive filesystem. Do not treat the surviving ASCII files as a complete response set.

4. **Flux-scale confirmation.** Verify the exact FLUXCAL convention and any applied AB offsets from the official SDSS SNANA archive README, then state the deterministic conversion needed to place SDSS and DES in a common physical-flux convention. Do not infer equivalence from the shared column name.

5. **Quality flags.** Freeze survey-specific flag masks from release documentation before inspecting cross-survey feature shifts. The frozen S/N rule is not a substitute for valid-epoch quality selection.

## Scientific decisions that must be predeclared

- Whether the unmodified extractor is applied to every released forced epoch or to an independently documented transient window. Any window must be based on survey metadata or a label-blind rule, not target outcomes.
- Whether Galactic extinction is left in observed fluxes and controlled by common support, or corrected identically using each release's `MWEBV`.
- Whether DES shallow and deep fields are separate primary strata or whether one field class is chosen a priori.
- How exact subtype labels are collapsed into the ordinary core-collapse class.
- Whether the asynchronous top-three-positive colour proxies are interpreted only operationally or used in passband-physics claims.

## Non-blocking but important uncertainties

- Detailed spectroscopic targeting efficiencies for ordinary core-collapse events in both surveys.
- Host-surface-brightness selection and error behavior across the two SMP implementations.
- Per-exposure atmospheric response versus each survey's nominal standard passband.
- Stability of results to camera-column response differences in SDSS.
- Small-sample uncertainty for the DES ordinary non-Ia subset, especially after field and redshift stratification.

## Alternative-survey findings

- YSE DR1 DOI metadata says 1,975 events and 472 spectroscopic SNe, but the downloaded archive contains 2,003 files and 494 non-`NA` broad spectroscopic labels. This must be version-resolved before YSE is used.
- YSE's released full archive removes negative flux measurements and mixes PS1 `griz` with ZTF `gr`; it is not observationally equivalent to the forced-flux DES/SDSS products.
- PS1-MDS has an attractive 518-object spectroscopic sample, but this audit did not establish an immutable raw light-curve product tied to a specific passband/calibration bundle.
- ZTF cannot support the unchanged frozen 16-feature set because general `z` coverage is absent.

## Leakage boundary

Allowed before unblinding target labels: response curves, calibration files, per-epoch metadata, unlabeled cadence/depth summaries, observing strategy, redshift availability, and fixed quality masks.

Held out until transformation and inclusion rules are frozen: target-survey class labels, target class-conditional feature distributions, classifier scores, and any choice among candidate corrections based on target performance.
