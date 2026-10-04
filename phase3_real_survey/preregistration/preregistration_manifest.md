# Pre-registration Manifest

Version: 1.0.0  
Frozen: 2026-10-04T14:44:46+0400  
Gate: GO

## Pre-registration identity

| File | SHA-256 before installation |
|---|---|
| `physics_preregistration.md` | `85e22fa1515e8a0b00aa71d8f086506355eea0191ed5b7b10f6c0b528f1a088a` |
| `feature_prediction_registry.csv` | `5db2feba455c3b9b807e0b67aff23bfe01154a1cf85a58fb8a0c217170a2a5f6` |
| `passband_prediction_notes.md` | `8596c7477238293c7196e0e85b78ba5c7b3f9441c5470772637f59c2fad08d1a` |
| `cadence_noise_prediction_notes.md` | `856a28c486e07abea3656200ae2441044cf62dc72a6e3ae09763b67628f5b85d` |
| `planned_measurements.md` | `74f1943406bcd144565e64641302ebb8f1d71dceb45ec1739283ff148505e9d8` |

The manifest does not record its own hash because self-inclusion is recursive. Hashes are rechecked after installation. Any future amendment must be a new version and must not overwrite this record silently.

## Survey data products

| Artifact | Identity |
|---|---|
| SDSS official release archive | `SDSS_dataRelease-snana.tar.gz`; 49,007,071 bytes; SHA-256 `b0861ce0d8bd5ab138bc2e9f41dd5321912d76cb43c718789a7bc637d56e3dbd`; Last-Modified `Mon, 04 Apr 2016 18:45:45 GMT`; ETag `2ebc9df-52fad22e78840` |
| SDSS nested product | `SDSS_allCandidates+BOSS.tar.gz`; SHA-256 `033992473b0dace08233bf9b4ef395a2267148f64ee021cc725a475d70d34d42` |
| SDSS HEAD, uncompressed | 2,378,880 bytes; SHA-256 `488ed81fb93b7e783e34cf48ed66252d39556c56617b4ec32ab06e56ef0ca0ea` |
| SDSS PHOT, uncompressed | 114,307,200 bytes; SHA-256 `3c88f726182690df24858de35793a5e1c70155a7a4835ec762dd57a301c96152` |
| DES immutable release | DOI `10.5281/zenodo.12720778`; `DES-SN5YR-1.2.zip`; 1,534,568,826 bytes; Zenodo MD5 `9019a6ddc569553bc323e9e1b68a55bf` |
| DES v1.2 README | SHA-256 `75cd01a0786cbe983b71b154565c8158fcb97e4b7113633457b77350eb845544` |
| DES HEAD compressed member | SHA-256 `08c1fb41dfe8b4e8a969f205144f94c9f6838a1a895a0fedc75b837fb87fd00a`; ZIP CRC32 `4c4b8b65` |
| DES PHOT compressed member | SHA-256 `10248f0364e90f88626ef19cf5ecf052500bbcbfd27f14f8b94cf8ffbc7dd01c`; ZIP CRC32 `a401d06e` |

## Calibration and response artifacts

| Artifact | SHA-256 | Registered role |
|---|---|---|
| `kcor_SDSS_Bessell90_BD17.fits.gz` | `7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb` | Nominal final SDSS Doi2010 AB/count calibration and embedded CCDAVG response family |
| `kcor_SDSS.input` | `51a2c555ae1359a9f76442a8776e0e834d9d7dec83953ec90019888c9c0a6d55` | SDSS generating input |
| KCOR embedded `SDSS-g` text extraction | `ec214d078543c13642a1029cf22712ebc08a00e2a095d1512a5992419659d4c8` | Exact response column used by synthetic calculation |
| KCOR embedded `SDSS-r` text extraction | `0bece182fa49bb616037f172417424d957cfd865aa004f7996ff868953a50087` | Exact response column used by synthetic calculation |
| KCOR embedded `SDSS-i` text extraction | `2504acca188af4c7ac161d30b7af70026964d160c6d9637e4dcf7aa9a0634ffb` | Exact response column used by synthetic calculation |
| KCOR embedded `SDSS-z` text extraction | `5a29b480215d9f4868eea1539eb4d94941f24b04213eeffbce4eec531fd810f9` | Exact response column used by synthetic calculation |
| `calib_DES-SN5YR_DES.fits.gz` | `4412a60ca41eb2cd8b32c038b352a551949a386cdc48acc0b059899bf46496c8` | Nominal DES-SN5YR analysis calibration grid |
| `calib_DES-SN5YR_DES.input` | `0917ddc66647ff613421a38c5df9a67d2113a7983e28889206c85c29b5c9e1ee` | DES generating input and offsets |
| `DECam_g.dat` | `671d50b64e3efdf6b848f59e1625d63c20998b64c10ada8ddead5ca4fe0394f4` | DES g total response |
| `DECam_r.dat` | `3dec154b19040b6fc8a3a08c94d213866114e062be93816e9a73ec0a301c3a21` | DES r total response |
| `DECam_i.dat` | `e0fe7030e997ff51bc1933478dcc6619cbe06d3f07ee78ece0c84c9524f33175` | DES i total response |
| `DECam_z.dat` | `078ebb6ea401ae22d55bcd35fa6e99eb0e06933dd877c02a5c00618b53dd9064` | DES z total response |
| `DES-SN5YR_DES_DCR+CHROM.DAT` | `1044cd913bfe28fed0982cc9a3f01d491bb211238ac51023858695fb30cd53c8` | Table defining corrections already applied in v1.2 |
| `DES-SN5YR_DES_FLUXERRMODEL_FAKE.DAT` | `1f1285b56d2d7edfe6ed4d293ee96d71acc05bbe03f6d52ccf8cba5d26538afd` | Pinned host-surface-brightness error-model reference |

## Feature and template artifacts

| Artifact | SHA-256 | Role |
|---|---|---|
| `phase2_tier4_make_variants.py` | `7d2b1b7c23d685e652b4b7df383441537162f9dc3829f8edba380a8ddf2a4d76` | Frozen Tier-4 observation selection and feature builder |
| `phase2_tier2_common.py` | `cbe760930a6d1361ade159fa7accea6bae361cdcc3d7df46a61a6fcebdfa92f1` | Frozen ordered 16-feature list |
| `Hsiao07.dat` | `6bd032004eccc7b5b52f7c021f0c4b541d801a70195028bf5af0a2578f554d78` | Fixed normal-Ia passband prediction SED |

## Frozen synthetic calculation

- Photon-counting AB response integral.
- Redshift grid: `0.05` to `0.30` inclusive in steps of `0.025`.
- Phase grid: `-10,0,10,20,30,40` rest-frame days.
- Dense template analogue: all Hsiao daily phases `-20` through `+85`.
- No fitted coefficient.
- No observed compact-feature statistic.
- No class label used to choose a predicted function.

## Common support

- `0.05 <= z_helio <= 0.30`.
- Definite normal Ia and definite ordinary II/IIb/Ib/Ic/Ibc.
- Secure transient spectroscopy.
- All four griz bands pass the frozen builder support rule.
- DES shallow/deep and SDSS north/south are retained as declared strata.

## Blinding declaration

This pre-registration did not inspect or calculate SDSS-versus-DES compact-feature distributions, means, medians, Wasserstein distances, SHAP values, centroids, or classifier performance. Previous feasibility documentation contained secure-label counts and metadata-level active-window summaries; `time_span` is therefore marked partially informed and supporting. The five primary features were selected from response curves, fixed-template synthetic predictions, documented cadence, and frozen extractor behavior.

## Gate accounting

- Defensible predictions: 10 of 16 features.
- No safe marginal direction: 6 of 16 features.
- Independent primary confirmatory features: 5.
- Stop-rule minimum: 3.
- Pre-registration gate: GO.
