# Survey and Instrument Provenance Manifest

## Selected releases

- **Survey A:** SDSS-II Supernova Survey final three-season scene-modelling release, Sako et al. 2018, official NERSC SNANA product `SDSS_dataRelease-snana.tar.gz`; audited local copy `SDSS_allCandidates+BOSS`.
- **Survey B:** DES-SN5YR full-transient Scene Modelling Photometry release, Zenodo DOI `10.5281/zenodo.12720778`; audited local `DES-SN5YR_DES` copy. The original immutable SMP product, not an unpinned future `main`, must be selected for an experiment.

## Component manifest

| Component | Exact source/artifact | Evidence | Confidence |
|---|---|---|---|
| SDSS release landing page | `https://portal.nersc.gov/project/dessn/SDSS/dataRelease/` | Identifies final catalogs, SMP, spectra, and SNANA archive. | Verified |
| SDSS official SNANA archive | `SDSS_dataRelease-snana.tar.gz`; HTTP metadata on audit date: 49,007,071 bytes, last-modified 2016-04-04, ETag `2ebc9df-52fad22e78840` | Official page links the archive. It was not downloaded in this audit. | Strong |
| Audited SDSS header | `.../lcmerge/SDSS_allCandidates+BOSS/SDSS_allCandidates+BOSS_HEAD.FITS.gz`; 768,687 bytes; SHA-256 `708fa8b5aebbe727caaa82d07ca1dbc1a8a16b463b2bc724582c530f21675a9e` | 10,258 rows; labels, redshifts, NOBS, and pointers inspected directly. | Verified locally; Strong link to official archive |
| Audited SDSS photometry | `.../SDSS_allCandidates+BOSS_PHOT.FITS.gz`; 41,472,281 bytes; SHA-256 `bb75b4cdfbc85fdab171a01c5ce6c9103d4a3d3764eb54f5fc17176523cbbcfd` | 1,120,566 epochs; MJD, FLT, FIELD, TELESCOPE, flags, flux/error, magnitude/error, PSF, sky, read noise, zeropoint, gain. | Verified locally; Strong link to official archive |
| SDSS photometric method | Holtzman et al. 2008, DOI `10.1088/0004-6256/136/6/2306`; release README `Holtz/v6_reCalib` | Scene model jointly fits host and transient without image resampling/convolution; calibrated fluxes and uncertainties. | Verified |
| SDSS cadence/seasons | Sako et al. 2018; released MJDs | Stripe 82, 2005-2007 autumn seasons; full stripe in two nights, effective cadence about four nights. Exact epochs are released. | Verified |
| SDSS selection | Sako et al. 2018 | Catalog requires detection on at least two nights and approximately `r<22.5`; spectroscopy was targeted, not complete. | Verified |
| SDSS flux system | Released FLUXCAL, MAG, ZEROPT; SDSS calibration documentation | Calibrated scene-model fluxes include negative values; release conversion used asinh input magnitudes. Exact SNANA FLUXCAL zero-point convention must be read from the official archive README before transformation work. | Strong |
| SDSS response reference | Doi et al. 2010, DOI `10.1088/0004-6256/139/4/1628`; official SDSS response tables | Measured filter+optics+CCD response, per camera column; atmosphere documented separately/combined for survey response. `griz` temporal residuals are below roughly 0.01 mag after calibration. | Verified |
| Local SDSS KCOR | `.../kcor/SDSS/SDSS_Doi2010/kcor_SDSS_Bessell90_BD17.fits.gz`; 6,199,253 bytes; SHA-256 `7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb` | FITS header names `kcor_SDSS.input`, 16 filters, AB primary, Doi path; `FilterTrans` embeds both lowercase and uppercase SDSS curves. | Verified artifact; **Partial release linkage** |
| Local SDSS KCOR input | `.../kcor/SDSS/SDSS_Doi2010/kcor_SDSS.input`; SHA-256 `51a2c555ae1359a9f76442a8776e0e834d9d7dec83953ec90019888c9c0a6d55` | AB, photon-counting, CCDAVG Doi path, zero offsets, Hsiao SN SED, BD+17 reference. | Verified artifact; Partial release linkage |
| SDSS local ASCII curve caveat | `.../filters/SDSS/SDSS_Doi2010/CCDAVG/` | On the case-insensitive local filesystem, `u/U`, `g/G`, and `i/I` names collide; only one case survives. The KCOR FITS retains distinct embedded response columns. | Verified limitation |
| DES immutable SMP record | Zenodo DOI `10.5281/zenodo.12720778`, published 2024-07-11 | Official DES-SN5YR SMP data release. | Verified |
| DES audited header | `.../lcmerge/DES-SN5YR/DES-SN5YR_DES/DES-SN5YR_DES_HEAD.FITS.gz`; 2,929,622 bytes; SHA-256 `08c1fb41dfe8b4e8a969f205144f94c9f6838a1a895a0fedc75b837fb87fd00a` | 19,706 rows; 353 definite Ia and 61 definite ordinary CC. | Verified locally |
| DES audited photometry | `.../DES-SN5YR_DES_PHOT.FITS.gz`; 66,106,516 bytes; SHA-256 `10248f0364e90f88626ef19cf5ecf052500bbcbfd27f14f8b94cf8ffbc7dd01c` | 1,798,736 epochs; MJD, BAND, CCDNUM, IMGNUM, FIELD, flags, flux/error, PSF, sky, read noise, zeropoint, gain, detector position. Negative fluxes verified. | Verified locally |
| DES photometric method | Sanchez et al. 2024, arXiv `2406.05046`; Brout et al. 2019 | Forced scene modelling of host and transient, with nightly combination and documented uncertainty construction. | Verified |
| DES observing strategy | Sanchez et al. 2024; Kessler et al. 2015 | Ten SN fields, eight shallow and two deep, roughly weekly cadence over five seasons. Exact epochs and fields released. | Verified |
| DES calibration input and grid | `calib_DES-SN5YR_DES.input` SHA-256 `0917ddc66647ff613421a38c5df9a67d2113a7983e28889206c85c29b5c9e1ee`; generated FITS SHA-256 `4412a60ca41eb2cd8b32c038b352a551949a386cdc48acc0b059899bf46496c8` | AB/count system; explicitly names `filters/DES/DES-SN3YR_DECam/DECam_[griz].dat` and additive calibration-offset expressions. | Verified |
| DES response curves | `DECam_g.dat` SHA-256 `671d50b64e3efdf6b848f59e1625d63c20998b64c10ada8ddead5ca4fe0394f4`; `DECam_r.dat` `3dec154b19040b6fc8a3a08c94d213866114e062be93816e9a73ec0a301c3a21`; `DECam_i.dat` `e0fe7030e997ff51bc1933478dcc6619cbe06d3f07ee78ece0c84c9524f33175`; `DECam_z.dat` `078ebb6ea401ae22d55bcd35fa6e99eb0e06933dd877c02a5c00618b53dd9064` | Documentation identifies Y3A2 nominal DES-SN curves including atmosphere at airmass 1.3. | Verified |
| DES correction state | 2024 paper versus mutable repository README | Sources disagree on whether chromatic and DCR corrections are already applied to the released fluxes. | Partial; must pin product |
| DES flux convention | DES release documentation | `mag = 27.5 - 2.5 log10(FLUXCAL)`; AB and Fragilistic offsets are encoded in calibration products and must not be double-applied. | Strong/Verified by selected version |

## Parameter availability for the selected pair

| Quantity | SDSS-II | DES-SN5YR | Classification |
|---|---|---|---|
| Exact epochs and cadence distribution | Epoch MJD released | Epoch MJD released | Directly available |
| Seasons/window | Three autumn seasons, documented and derivable | Five seasons, documented and derivable | Directly available/derivable |
| Field structure | Stripe 82 north/south stripe identifiers | Ten named shallow/deep fields | Directly available |
| Exposure time | Drift-scan exposure documented externally | Band/field strategy documented externally; not a simple per-row exposure column | Documented externally/partial |
| Limiting depth | Published survey characterization; derivable empirically only with care | Published shallow/deep depths | Documented externally |
| PSF/seeing | `PSF_SIG1/2` per epoch | `PSF_SIG1/2` per epoch | Directly available |
| Sky background | `SKY_SIG`, `SKY_SIG_T` | `SKY_SIG`, `SKY_SIG_T` | Directly available |
| Gain/read noise/zeropoint | Released per epoch | Released per epoch | Directly available |
| Flux-error model | Released errors and SMP method | Released errors plus documented host-surface-brightness correction | Directly available/strong |
| Detection threshold/search efficiency | Published; catalog two-night and magnitude selection; detailed efficiency external | Published DiffImg trigger and simulations/efficiency assets | Documented externally/partial |
| Forced-photometry policy | Scene modelling across epochs, negative flux retained | SMP forced flux, negative flux retained | Verified |
| Galactic extinction | `MWEBV` header; observed flux not treated as dereddened | `MWEBV` header; observed flux not treated as dereddened | Directly available; correction policy must be predeclared |

## Provenance gate

The DES chain is sufficient once a specific immutable SMP product and correction state are pinned. The SDSS photometry and response systems are individually well documented, but the arrow from the **final 10,258-object SMP release** to the **specific Doi/KCOR artifact** remains only partial. This single link prevents an unconditional GO.
