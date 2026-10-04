# Passband-Only Prediction Notes

Version: 1.0.0  
Frozen: 2026-10-04T14:44:46+0400

## Scope

These predictions use no SDSS or DES compact-feature values. They use only the checksummed response/calibration artifacts and a fixed normal-Ia SED template. They predict the passband component of a measurement, not the final observed cross-survey feature difference.

Ordinary core-collapse predictions are not assigned an Ia sign. The primary CC family contains II/IIb/Ib/Ic/Ibc, whose phase-dependent SED diversity requires a separately frozen subtype template library.

## Inputs

| Input | Frozen identity |
|---|---|
| SDSS responses | Lowercase griz columns embedded in `kcor_SDSS_Bessell90_BD17.fits.gz`, SHA-256 `7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb` |
| SDSS KCOR input | SHA-256 `51a2c555ae1359a9f76442a8776e0e834d9d7dec83953ec90019888c9c0a6d55` |
| DES calibration | `calib_DES-SN5YR_DES.fits.gz`, SHA-256 `4412a60ca41eb2cd8b32c038b352a551949a386cdc48acc0b059899bf46496c8` |
| DES generating input | SHA-256 `0917ddc66647ff613421a38c5df9a67d2113a7983e28889206c85c29b5c9e1ee` |
| DES g/r/i/z responses | SHA-256 `671d50...`, `3dec15...`, `e0fe70...`, `078ebb...`; full values in `preregistration_manifest.md` |
| Ia template | `Hsiao07.dat`, SHA-256 `6bd032004eccc7b5b52f7c021f0c4b541d801a70195028bf5af0a2578f554d78` |

The response pivot wavelengths are:

| Survey | g | r | i | z |
|---|---:|---:|---:|---:|
| SDSS Doi2010 CCDAVG | 4701.38 | 6177.21 | 7496.55 | 8904.30 |
| DES DECam | 4809.40 | 6420.44 | 7816.03 | 9171.13 |

Units are Angstrom. DES is redder in every nominal band under this definition.

## Synthetic-photometry definition

For photon-counting response `T_b(lambda)`, the source count shape is

`N_b proportional to integral f_lambda(lambda) lambda T_b(lambda) dlambda`.

The AB reference count shape is

`N_AB,b proportional to integral T_b(lambda)/lambda dlambda`.

The response-only magnitude difference is

`Delta_b(z,p,c) = m_DES,b - m_SDSS,b`.

Positive `Delta_b` means the fixed source is fainter in DES band b than in SDSS band b, and therefore has lower DES flux on a common AB scale. The corresponding flux mapping is

`F_SDSS,b / F_DES,b = 10^(0.4 Delta_b)`.

Calibration offsets are not included in `Delta_b`; they are a separate, documented analysis-calibration layer. This prevents response and zeropoint terms from being counted twice.

## Frozen grids

- Redshift: `0.05, 0.075, ..., 0.30`.
- Phase-conditioned grid: rest-frame `-10, 0, 10, 20, 30, 40` days.
- Dense peak/top-three analogue: all daily Hsiao phases from `-20` through `+85` days.
- No extinction, stretch, or colour perturbation is applied in the nominal synthetic calculation. These enter later sensitivity grids fixed independently of observed survey shifts.

## Dense-template analogue

The table records `DES-SDSS` synthetic magnitude differences. `dpeak_b` is the difference between ideal dense-template band maxima. `dtop3_a-b` reproduces the frozen operation on a noiseless daily template: average the three largest positive band fluxes independently, then form the magnitude-style ratio.

| z | dpeak_g | dpeak_r | dpeak_i | dpeak_z | dtop3_g-r | dtop3_r-i | dtop3_i-z |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.050 | -0.00136 | +0.03518 | +0.10122 | -0.03147 | -0.03667 | -0.06808 | +0.13476 |
| 0.075 | +0.02296 | +0.01191 | +0.13316 | -0.03141 | +0.01071 | -0.12469 | +0.16652 |
| 0.100 | +0.03064 | +0.02805 | +0.17902 | -0.03773 | +0.00229 | -0.15069 | +0.21670 |
| 0.125 | +0.01421 | +0.04039 | +0.13974 | -0.03702 | -0.02709 | -0.09757 | +0.17661 |
| 0.150 | -0.01074 | +0.02793 | +0.10752 | -0.00055 | -0.03992 | -0.07706 | +0.10562 |
| 0.175 | -0.03336 | -0.01624 | +0.13844 | +0.05068 | -0.01886 | -0.15310 | +0.08557 |
| 0.200 | -0.04228 | -0.02630 | +0.12042 | +0.07572 | -0.01609 | -0.14602 | +0.04299 |
| 0.225 | -0.02829 | -0.00547 | +0.10497 | +0.07550 | -0.02236 | -0.10977 | +0.02665 |
| 0.250 | -0.02118 | +0.00868 | +0.07993 | +0.09781 | -0.02926 | -0.07212 | -0.02051 |
| 0.275 | -0.04190 | +0.02140 | +0.02924 | +0.14018 | -0.06583 | -0.00865 | -0.11337 |
| 0.300 | -0.09037 | +0.02398 | -0.00697 | +0.17906 | -0.11309 | +0.02948 | -0.18474 |

## Registered feature predictions

### `peak_color_r_minus_i`

For normal Ia, the passband-only DES-SDSS term is negative through z=0.275 and crosses slightly positive at z=0.30 in the dense-template analogue. Phase-conditioned calculations range more widely and occasionally cross zero. Therefore the prediction is a function, not the statement "DES is always bluer."

### `peak_color_i_minus_z`

For normal Ia, the term is positive at z<=0.225 and negative at z>=0.25. The redshift sign reversal is the strongest preregistered passband diagnostic because it is difficult to explain with a single global zeropoint shift.

### `i_peak_flux`

The dense-template `Delta_i` is positive from z=0.05 through 0.275, implying lower ideal DES i-band peak flux on a common AB scale, and approximately zero/slightly negative at z=0.30. Sampling, noise, and field depth may obscure this response-only term.

### `z_peak_flux`

The dense-template sign changes: DES is brighter at low z and fainter at higher z relative to SDSS. The exact transition is template/phase dependent; the prediction is evaluated as a redshift curve.

### Secondary checks

`peak_color_g_minus_r` and `r_peak_flux` are redshift/phase dependent and sign-changing with smaller dense-template terms. They are useful checks but are not primary because blue extinction sensitivity and cadence/phase competition are large.

## Calibration term

The named systems record different analysis offsets. SDSS g/r/i/z values are `+0.02028,+0.00493,+0.01780,+0.01015` mag; DES net input expressions are `-0.0008,-0.0116,-0.0090,+0.0071` mag. These values are not applied directly to released epoch FLUXCAL in this pre-registration. Before implementation, the conversion must be expressed through the two named SNANA calibration artifacts so that already-applied corrections are neither omitted nor doubled.

## Limitations fixed in advance

1. Hsiao07 is a normal-Ia mean template, not a population model.
2. The frozen colour features are asynchronous top-three-positive proxies.
3. Host extinction, Milky Way extinction, intrinsic colour/stretch, and spectral diversity can alter or reverse a phase-specific term.
4. Peak flux is a sampled maximum and is biased by cadence and noise.
5. The SDSS response is CCD-averaged; camera-column responses are a predeclared sensitivity analysis, not a replacement nominal system.
6. DES nominal responses include atmosphere at airmass 1.3; per-exposure atmosphere is not reconstructed in the primary test.
7. Core-collapse results must remain subtype/template dependent.
