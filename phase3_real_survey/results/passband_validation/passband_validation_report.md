# Phase 3 Passband Prediction Validation

This validation uses only frozen response curves and the Hsiao07 template.

**No SDSS or DES real compact-feature distributions were read.**

## Artifact validation

- `sdss_kcor`: `/Users/anuraggarg/work/Feature normalization Phase/preliminary_hypothesis_test/SNDATA_ROOT_2024-07-03/kcor/SDSS/SDSS_Doi2010/kcor_SDSS_Bessell90_BD17.fits.gz` — SHA-256 verified
- `des_g`: `/Users/anuraggarg/work/Feature normalization Phase/preliminary_hypothesis_test/SNDATA_ROOT_2024-07-03/filters/DES/DES-SN3YR_DECam/DECam_g.dat` — SHA-256 verified
- `des_r`: `/Users/anuraggarg/work/Feature normalization Phase/preliminary_hypothesis_test/SNDATA_ROOT_2024-07-03/filters/DES/DES-SN3YR_DECam/DECam_r.dat` — SHA-256 verified
- `des_i`: `/Users/anuraggarg/work/Feature normalization Phase/preliminary_hypothesis_test/SNDATA_ROOT_2024-07-03/filters/DES/DES-SN3YR_DECam/DECam_i.dat` — SHA-256 verified
- `des_z`: `/Users/anuraggarg/work/Feature normalization Phase/preliminary_hypothesis_test/SNDATA_ROOT_2024-07-03/filters/DES/DES-SN3YR_DECam/DECam_z.dat` — SHA-256 verified
- `hsiao`: `/Users/anuraggarg/work/Feature normalization Phase/preliminary_hypothesis_test/SNDATA_ROOT_2024-07-03/snsed/Hsiao07.dat` — SHA-256 verified

## Pivot wavelengths

| Band | SDSS (A) | DES (A) |
|---|---:|---:|
| g | 4701.38 | 4809.40 |
| r | 6177.21 | 6420.44 |
| i | 7496.55 | 7816.03 |
| z | 8904.30 | 9171.13 |

## Reproduction gate

- Tolerance: `0.000500` mag
- Maximum absolute discrepancy: `0.00001370` mag
- Gate: **PASS**

The registered dense-template prediction table is reproduced independently from the frozen artifacts.

No observed survey feature statistic or class label was used.
