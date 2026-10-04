# Pinned SDSS-II and DES-SN5YR Products

Audit date: 2026-10-04

## Frozen comparison products

### SDSS-II

- Product: final three-season SNANA scene-modelling release, `SDSS_allCandidates+BOSS`.
- Official archive: `https://portal.nersc.gov/project/dessn/SDSS/dataRelease/SDSS_dataRelease-snana.tar.gz`.
- Archive size: 49,007,071 bytes.
- Archive SHA-256: `b0861ce0d8bd5ab138bc2e9f41dd5321912d76cb43c718789a7bc637d56e3dbd`.
- HTTP Last-Modified: `Mon, 04 Apr 2016 18:45:45 GMT`.
- HTTP ETag: `"2ebc9df-52fad22e78840"`.
- Response/calibration product: `SDSS_Doi2010/kcor_SDSS_Bessell90_BD17.fits`.
- KCOR SHA-256: `7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb`.
- Generating input SHA-256: `51a2c555ae1359a9f76442a8776e0e834d9d7dec83953ec90019888c9c0a6d55`.
- Response representation for first comparison: KCOR-embedded CCDAVG Doi2010 `ugriz` columns, not reconstructed filenames from the case-collided extraction.

The official archive's HEAD and PHOT members are byte-identical after decompression to the existing local installation. The release README directly prescribes the named Doi2010 KCOR.

### DES-SN5YR

- Product: immutable DES-SN5YR v1.2 release, full transient SMP product `DES-SN5YR_DES`.
- DOI: `10.5281/zenodo.12720778`.
- Zenodo record: `https://zenodo.org/records/12720778`.
- Archive: `DES-SN5YR-1.2.zip`.
- Archive size: 1,534,568,826 bytes.
- Archive MD5 recorded by Zenodo: `9019a6ddc569553bc323e9e1b68a55bf`.
- Publication date: 2024-07-11.
- The ZIP central directory was read using HTTP byte ranges; the full 1.53 GB archive was not downloaded.

## Exact data members

| Member | Bytes | ZIP CRC32 | SHA-256 / identity result |
|---|---:|---|---|
| `0_DATA/DES-SN5YR_DES/DES-SN5YR_DES.README` | 2,387 | `7417c19d` | Selectively extracted SHA-256 `75cd01a0786cbe983b71b154565c8158fcb97e4b7113633457b77350eb845544` |
| `0_DATA/DES-SN5YR_DES/DES-SN5YR_DES_HEAD.FITS.gz` | 2,929,622 | `4c4b8b65` | Existing local file matches immutable member by size and CRC32; SHA-256 `08c1fb41dfe8b4e8a969f205144f94c9f6838a1a895a0fedc75b837fb87fd00a` |
| `0_DATA/DES-SN5YR_DES/DES-SN5YR_DES_PHOT.FITS.gz` | 66,106,516 | `a401d06e` | Existing local file matches immutable member by size and CRC32; SHA-256 `10248f0364e90f88626ef19cf5ecf052500bbcbfd27f14f8b94cf8ffbc7dd01c` |
| `0_DATA/README.md` | 4,513 | `4d0496d7` | Selectively extracted SHA-256 `8744111c93d54b96be157d010f499d3c50e5c39f9234178d48002f2815b99aec` |
| root `README.md` | 3,135 | `94c34772` | Selectively extracted SHA-256 `89a92f1cf5c7d847aa9d8956fc0636e42bd882fe3ed581622dbf63db5b68fae4` |

The older local DES README is not byte-identical to the v1.2 member; the v1.2 README above is authoritative for the selected product. HEAD and PHOT are exact matches.

## DES calibration and response chain

The immutable v1.2 fit configuration `7_PIPPIN_FILES/base_files/lcfit/lcfit_desSMP_5yr.nml` names:

`$SNDATA_ROOT/kcor/DES/DES-SN5YR/calib_DES-SN5YR_DES.fits`

The corresponding pinned SNDATA_ROOT 2024-07-03 products are:

| Artifact | SHA-256 | Role |
|---|---|---|
| `calib_DES-SN5YR_DES.input` | `0917ddc66647ff613421a38c5df9a67d2113a7983e28889206c85c29b5c9e1ee` | Generating input; AB/count system, exact griz response filenames, calibration offsets. |
| `calib_DES-SN5YR_DES.fits.gz` | `4412a60ca41eb2cd8b32c038b352a551949a386cdc48acc0b059899bf46496c8` | Analysis calibration grid. |
| `DOCUMENTATION.README` | `2e0278993739bb4ccff19f5be88cf57a3d048740dafe8524d4fde521f5c50e74` | Identifies nominal DES-SN5YR calibration, version 2023-10. |
| `DECam_g.dat` | `671d50b64e3efdf6b848f59e1625d63c20998b64c10ada8ddead5ca4fe0394f4` | Y3A2 nominal g response including atmosphere at airmass 1.3. |
| `DECam_r.dat` | `3dec154b19040b6fc8a3a08c94d213866114e062be93816e9a73ec0a301c3a21` | Y3A2 nominal r response including atmosphere at airmass 1.3. |
| `DECam_i.dat` | `e0fe7030e997ff51bc1933478dcc6619cbe06d3f07ee78ece0c84c9524f33175` | Y3A2 nominal i response including atmosphere at airmass 1.3. |
| `DECam_z.dat` | `078ebb6ea401ae22d55bcd35fa6e99eb0e06933dd877c02a5c00618b53dd9064` | Y3A2 nominal z response including atmosphere at airmass 1.3. |

The calibration input gives net additive calibration expressions:

| Band | Expression | Net magnitude offset |
|---|---|---:|
| g | `0.0-0.003+0.0022` | -0.0008 |
| r | `0.0-0.003-0.0086` | -0.0116 |
| i | `0.0-0.002-0.007` | -0.0090 |
| z | `0.0+0.001+0.0061` | +0.0071 |

The input labels the second term as CALSPEC recalibration and the third as Supercal. Fragilistic calibration uncertainty is represented in the release's SALT3 systematic models; the epoch `FLUXCAL` columns must not receive an additional unregistered "Fragilistic correction." The exact nominal interpretation is the released flux product plus `calib_DES-SN5YR_DES`.

## Applied-correction state

The immutable v1.2 `DES-SN5YR_DES.README` states:

- chromatic and DCR magnitude corrections are applied;
- the defining lookup is `$SNDATA_ROOT/simlib/DES/DES-SN5YR_DES_DCR+CHROM.DAT`;
- the host-surface-brightness flux-error correction is applied.

The immutable fit configuration independently says the ad hoc `MAGCOR_FILE` and `FLUXERRMODEL_FILE` applications were removed on 2024-06-12 because the corrections are now in the data release. Therefore they must **not** be applied again.

Pinned referenced tables:

| Table | SHA-256 |
|---|---|
| `DES-SN5YR_DES_DCR+CHROM.DAT` | `1044cd913bfe28fed0982cc9a3f01d491bb211238ac51023858695fb30cd53c8` |
| `DES-SN5YR_DES_FLUXERRMODEL_FAKE.DAT` | `1f1285b56d2d7edfe6ed4d293ee96d71acc05bbe03f6d52ccf8cba5d26538afd` |

## Frozen handling rules for the next phase

1. Use the exact SDSS and DES photometry products listed above.
2. Treat each photometry product and its named KCOR/calibration artifact as one versioned system.
3. Do not apply the SDSS or DES calibration offsets directly to epoch fluxes as an extra empirical correction before a physical model is pre-registered.
4. Do not reapply DES DCR/chromatic or host-surface-brightness error corrections.
5. Use the embedded SDSS KCOR response columns for the nominal CCD-averaged comparison; preserve CCD-specific analysis as a separately predeclared sensitivity test.

## Provenance classification

| Arrow | Status |
|---|---|
| SDSS official archive -> released SMP | VERIFIED |
| SDSS released SMP -> Doi2010 KCOR | VERIFIED |
| SDSS KCOR -> generating input/embedded responses | VERIFIED |
| DES Zenodo v1.2 -> README/HEAD/PHOT | VERIFIED |
| DES v1.2 -> named `calib_DES-SN5YR_DES` path | VERIFIED |
| Named DES calibration path -> checksummed SNDATA_ROOT 2024 artifact | STRONG |
| DES calibration input -> DECam griz response files | VERIFIED |
| DES v1.2 -> applied DCR/chromatic state | VERIFIED |

The remaining `STRONG` rather than `VERIFIED` label reflects that the 1.2 GB DES ZIP names the external SNDATA artifact by path but does not bundle its checksum. The exact local artifact is separately frozen by SHA-256 and the contemporaneous SNDATA_ROOT Zenodo snapshot (`10.5281/zenodo.12655677`). This is not an undocumented-passband assumption.
