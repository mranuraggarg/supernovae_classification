# Candidate Pair Comparison

## Comparative gate matrix

`PASS` means the audited pair satisfies the requirement without changing the frozen feature definitions. `PARTIAL` means a bounded provenance or preprocessing question remains. `FAIL` is a hard failure for the proposed first experiment.

| Pair | Real photometry | Secure Ia | Secure non-Ia | Redshift overlap | Passband provenance | Calibration provenance | Cadence provenance | Noise/depth provenance | Feature compatibility | Bidirectional feasibility | Label-blind physics possible | Major confounders | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DES-SN5YR SMP / SDSS-II SMP** | PASS | PASS | PASS | PASS (`0.05-0.30`) | PARTIAL | PASS | PASS | PASS | PASS/PARTIAL | PASS | YES | spectroscopic targeting; DES shallow/deep fields; very different active windows; host and redshift demographics | **CONDITIONAL GO** |
| **DES-SN5YR SMP / YSE DR1 PS1-only** | PASS | PASS | PASS | PASS, mainly `0.02-0.18` | PARTIAL | PARTIAL | PASS | PARTIAL | PARTIAL | PASS | PARTIAL | YSE product removes negative flux, mixes parent instruments, release-count inconsistency, low-z dominance | PARTIAL |
| **SDSS-II SMP / YSE DR1 PS1-only** | PASS | PASS | PASS | PASS, mainly `0.02-0.18` | PARTIAL | PARTIAL | PASS | PARTIAL | PARTIAL | PASS | PARTIAL | YSE preprocessing; different transient selection; SDSS spectroscopic targeting | PARTIAL |
| **DES-SN5YR SMP / PS1-MDS 518** | PASS | PASS | PASS | Likely PASS | PARTIAL | PARTIAL | Strong | PARTIAL | Likely PASS | Likely PASS | PARTIAL | exact 518-object photometry release and curve bundle not pinned | FAIL at current evidence state |
| **Any frozen-16 pair using ZTF BTS as one side** | PASS | PASS | PASS | PASS | Strong | Strong | PASS | PASS | **FAIL** | FAIL | YES | no general `z`; incomplete `i`; cannot compute four frozen features and `i-z` unchanged | FAIL |

No weighted score was used. The leading pair is selected because it is the only one for which the released epoch schema, secure ordinary Ia/non-Ia labels, four-band support, and both surveys' observing metadata were all directly inspectable.

## Provenance-chain comparison

### DES-SN5YR SMP

Released Zenodo/repository light curves -> DES-SN5YR SMP product: **VERIFIED**  
SMP product -> scene-modelling pipeline and nightly combination: **VERIFIED**  
SMP product -> Y6/FGCM calibration and correction state: **STRONG/PARTIAL** because mutable `main` and the 2024 paper differ on whether DCR/chromatic corrections are already applied  
Calibration input -> Y3A2 `DECam_[griz].dat`: **VERIFIED**  
Response files -> nominal total system including atmosphere at airmass 1.3: **VERIFIED**

### SDSS-II final release

Official data-release page -> 10,258-object SMP product: **VERIFIED**  
SMP product -> `Holtz/v6_reCalib` scene-modelling pipeline: **VERIFIED** in the released README  
SMP product -> calibrated SDSS natural/AB-like photometric system: **VERIFIED/STRONG**  
Released photometry -> exact local `kcor_SDSS_Bessell90_BD17.fits.gz`: **PARTIAL**; the release README does not name the file  
KCOR -> embedded Doi 2010 CCD-averaged and per-column response functions: **VERIFIED** inside the FITS artifact  
Doi response -> optics/filter/detector plus atmosphere at airmass 1.3: **VERIFIED** by primary SDSS/Doi documentation

## Bidirectional feasibility

Using only definite transient-spectroscopic labels and the predeclared `0.05 <= z <= 0.30` region:

| Direction | Available after frozen four-band support rule | Feasibility judgment |
|---|---:|---|
| SDSS train -> SDSS test; SDSS train -> DES test | SDSS: 305 Ia, 61 ordinary CC; DES target: 134 Ia, 51 ordinary CC | Feasible for a binary feasibility diagnostic. Class-weighting and uncertainty intervals would be needed later, but no model work belongs in this audit. |
| DES train -> DES test; DES train -> SDSS test | DES: 134 Ia, 51 ordinary CC; SDSS target: 305 Ia, 61 ordinary CC | Feasible but statistically asymmetric. DES is the smaller training side; subtype-resolved evaluation is not supported. |

The two directions are not interchangeable. Any future result must report them separately.

## Ranking

### 1. DES-SN5YR SMP / SDSS-II final SMP

**Why scientifically attractive:** two real, untargeted rolling surveys; common `griz`; forced scene-modelling fluxes with negative measurements; secure Ia and ordinary core-collapse support; exact epochs and rich per-epoch observing metadata; different cameras, response functions, cadence, depth, and survey windows.

**Why it may fail:** the SDSS final-release README does not explicitly name the local Doi/KCOR artifact; DES correction state must be pinned; spectroscopic target selection and the longer DES active window may dominate some features.

**Dominant physical difference:** DECam versus the SDSS drift-scan camera, including distinct `griz` responses, depth, observing season, field strategy, and cadence.

**Dominant astrophysical confounder:** redshift- and subtype-dependent spectroscopic selection, with DES probing a higher-redshift population and deeper fields.

**Single evidence item that most changes the decision:** an official SDSS release manifest or calibration record explicitly linking the final `SMPv8+BOSS` light curves to the Doi 2010 response set/KCOR artifact.

### 2. DES-SN5YR SMP / YSE DR1 PS1-only

**Why scientifically attractive:** real `griz` surveys with secure Ia and core-collapse labels; strong low-redshift YSE support complements DES.

**Why it may fail:** YSE has already removed negative fluxes and applied a +/-757-day mask; its release combines PS1 and ZTF; the DOI says 1,975/472 while the archive contains 2,003 files/494 broad spectroscopic labels.

**Dominant physical difference:** DECam deep fields versus PS1 follow-up cadence and shallower GPC1 photometry.

**Dominant astrophysical confounder:** YSE's much lower redshift and young/fast-rising selection.

**Single evidence item that most changes the decision:** a versioned YSE manifest mapping each PS1 epoch to the exact photometric-calibration and system-response version before its negative-flux masking.

### 3. SDSS-II final SMP / YSE DR1 PS1-only

**Why scientifically attractive:** substantial low-redshift overlap and ordinary core-collapse support; distinct real instruments; common `griz`.

**Why it may fail:** YSE preprocessing changes the sampling distribution that drives several frozen features; PS1 calibration provenance is not carried per epoch in the released files.

**Dominant physical difference:** SDSS drift-scan four-night effective cadence versus targeted PS1/YSE roughly three-day scheduling with different band combinations.

**Dominant astrophysical confounder:** different discovery and spectroscopic follow-up selection.

**Single evidence item that most changes the decision:** an unmasked, release-versioned YSE PS1-only forced-photometry product retaining negative measurements and exact calibration identifiers.

## Physics-first test

**If XGBoost were removed, would the leading comparison remain meaningful? YES.**

The pair supports an observational experiment on how two calibrated, forced-photometry systems map the same broad SN populations into measured peak, colour-proxy, dispersion, and temporal-coverage statistics. The downstream classifier would only test whether those changes move events across a previously fixed empirical decision boundary.
