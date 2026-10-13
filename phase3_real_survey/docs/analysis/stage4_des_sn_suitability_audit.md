# Stage 4: DES-SN Real-Data Suitability Audit

Audit date: 2026-10-13. Repository baseline: `417871626386c655f9c4fd1aebb5175e467d8f74`.

## Executive decision

**CONDITIONAL GO for a limited, frozen SPCC-to-DES external classification evaluation.** DES-SN5YR supplies suitable observations and secure labels, but model/extractor lineage and the history of target-data exposure must be resolved before calling the evaluation independent. The previously failed **SPCC-to-DES passband-normalization gate remains NO-GO**. This audit does not reopen it.

Use the immutable full-transient SMP product in `DES-SN5YR-1.2.zip`, DOI [10.5281/zenodo.12720778](https://zenodo.org/records/12720778), as the primary target. The checked HEAD file contains **353 secure normal Ia and 61 ordinary core-collapse SNe**, not thousands of spectroscopically classified SNe. Within the existing `0.05 <= z_helio <= 0.30` support, there are **140 Ia and 52 ordinary CC** before the frozen band-support rule. Existing repository validation records **134 Ia and 51 CC** afterward.

The 185-object supported cohort can give a useful conditional performance estimate. It cannot establish accuracy for the entire DES transient population, precise subtype performance, or that instrument physics caused any classification failure. Existing Stage 1/2 manifests establish prior inspection of DES Ia feature outcomes; this target is independent of SPCC simulated objects, but is not a previously unseen research dataset.

No classifier, feature builder, statistical hypothesis test, or normalization was run in this audit. No dataset was downloaded. Only existing local documentation, code, a small HEAD table, earlier support-validation summaries, and official public documentation were inspected. Existing PHOT files were not reprocessed.

## 1. Evidence inventory

Paths below are relative to the repository unless absolute. Evidence is classified as direct verification, prior recorded verification, or unresolved; historical reports are not silently promoted to new measurements.

| Source | Role and evidence |
|---|---|
| `archive/phase2/scripts/phase2_tier4_make_variants.py`, `_build_compact_row_from_observations` and `build_spcc_rows` | Directly inspected frozen raw-observation operator and SPCC adapter. |
| `archive/phase2/scripts/phase2_tier2_common.py`, `COMPACT_FEATURES` | Exact ordered 16-column representation. |
| `feature_pipeline/loaders/spcc_raw.py`, `feature_pipeline/config.py`, `feature_pipeline/cleaning/spcc_clean.py` | SPCC observation parsing, simulated truth labels, numerical cleaning; default flux/time scaling both 1.0. |
| `feature_pipeline/extraction/spcc_features.py` | Earlier Tier-1 reconstructed-grid extractor; **not interchangeable** with the frozen Tier-4 operator despite shared names. |
| `phase3_real_survey/scripts/build_real_survey_features.py` | Real-survey adapter, secure-label mapping, support and structural-mask implementation. |
| `phase3_real_survey/preregistration/physics_preregistration.md` | Existing population, feature, calibration, and target-blinding rules. It concerns SDSS-DES, not a newly authorized SPCC model experiment. |
| `phase3_real_survey/results/feature_builder_equivalence.txt` | Prior synthetic equivalence: 10 edge cases and 1,000 randomized cases, zero failures; frozen builder hash `7d2b1b7c23d685e652b4b7df383441537162f9dc3829f8edba380a8ddf2a4d76`. |
| `phase3_real_survey/results/primary_cohort_validation/primary_cohort_validation.txt`; `results/real_feature_tables/feature_build_report.md` | Existing counts and four-band survival; no feature distributions inspected for this audit. |
| `phase3_real_survey/results/quality_mask_validation/quality_mask_validation.md`; `frozen_quality_mask_registry.csv` in that directory | Prior epoch validity, negative-flux retention, and unresolved flag semantics. |
| `phase3_real_survey/docs/provenance/pinned_des_sdss_products.md` | Prior immutable-release linkage, response/calibration hashes and correction state. |
| `phase3_real_survey/docs/analysis/stage3c_scientific_assessment.md` | Fixed-Hsiao operator study: mixed predictions, no real classification validation. |
| `phase3_real_survey/results/primary_confirmatory_stage1/stage1_manifest.json`; corresponding `stage2/stage2_manifest.json` | Direct evidence of previous DES Ia outcome inspection and operator comparisons; both record `classifier_run: false`. |
| `phase3_real_survey/data/raw/des/DES-SN5YR-1.2/0_DATA/DES-SN5YR_DES/` | Existing immutable SMP README and HEAD; HEAD counts and identity independently checked here. |
| `/Users/anuraggarg/work/Feature normalization Phase/preliminary_hypothesis_test/spcc_kcor_provenance/spcc_kcor_provenance_audit.md` | Final SPCC provenance NO-GO: actual challenge input names `kcor_DES_grizY.his`; missing main artifact/build chain. Later `.fits` examples do not close the link. |
| [SMP immutable record](https://zenodo.org/records/12720778) | Record metadata version v1; archive filename version 1.2; published 2024-07-11. |
| [DIFFIMG immutable record](https://zenodo.org/records/14332953) and [release README](https://zenodo.org/records/14332953/files/DES-SN5YR_DIFFIMG.README?download=1) | Version v1, published 2024-12-09; official label counts, schema and flag meanings. |
| [DES collaboration repository](https://github.com/des-science/DES-SN5YR) | Current main now describes DES-Dovekie and links the legacy release. It is not the identity of the frozen SMP target. |
| [Sanchez et al. release paper](https://arxiv.org/html/2406.05046), DOI `10.3847/1538-4357/ad739a` | Measurement methods and candidate-selection context. The HTML inspected is the June 2024 preprint; later product-specific correction statements take precedence. |
| [Official data dictionary](https://des-sn-dr.readthedocs.io/en/latest/0_DATA.html) | Column and SNTYPE definitions; generic descriptions require checking against the pinned product. |

## 2. Release comparison and identity

| Property | SMP primary | DIFFIMG sensitivity product |
|---|---|---|
| Immutable identifier | `10.5281/zenodo.12720778`, record v1, `DES-SN5YR-1.2.zip` | `10.5281/zenodo.14332953`, v1 |
| Product | `0_DATA/DES-SN5YR_DES` | `DES-SN5YR_DIFFIMG` |
| Released objects | 19,706 | 31,636 |
| Measurement | Scene model of transient plus host, forced flux measurements | Forced PSF measurements on subtracted images |
| Secure Ia / ordinary CC | 353 / 61, checked HEAD | 356 / 63, official manifest |
| Main limitation | Selected subset successfully processed by SMP | Operational photometry; release explicitly excludes cosmological use |
| Audit role | Primary calibrated real-data target | Potential paired photometry-method sensitivity set |

The release paper reports higher precision and accuracy for SMP and reduced host-surface-brightness effects. SMP processing is designed around single-season transients; the difference in object counts must not be attributed entirely to random fit failures. DIFFIMG and SMP share objects and observations, so they are not independent test surveys. The cosmology subset and photometric Ia probabilities are unsuitable substitutes for secure transient labels. [Release paper](https://arxiv.org/html/2406.05046).

Pinned SMP identity, from the prior immutable-member audit:

- HEAD SHA-256, reverified here: `08c1fb41dfe8b4e8a969f205144f94c9f6838a1a895a0fedc75b837fb87fd00a`.
- PHOT SHA-256, prior verified record: `10248f0364e90f88626ef19cf5ecf052500bbcbfd27f14f8b94cf8ffbc7dd01c`.
- Product README SHA-256, prior verified record: `75cd01a0786cbe983b71b154565c8158fcb97e4b7113633457b77350eb845544`.
- Zenodo archive MD5: `9019a6ddc569553bc323e9e1b68a55bf`.

DIFFIMG published MD5 values: README `6e703632a0f54a342d912d79c497c5bc`; HEAD `9686eeda30964237d56d76feb31f4620`; PHOT `2bd233b307e19ff56dd84aba10ff59a0`. These are publisher checksums, not locally recomputed checksums. No DIFFIMG binary was downloaded or used to infer unmeasured support. [Immutable release](https://zenodo.org/records/14332953).

## 3. Spectroscopic labels and joins

SMP HEAD has 19,706 rows, 19,706 unique SNIDs, and `FAKE=0` throughout. The following counts were directly recomputed from **SNTYPE metadata only**. DIFFIMG counts are from its immutable release manifest, not from a downloaded binary.

| SNTYPE | Meaning | SMP | DIFFIMG | Primary use |
|---:|---|---:|---:|---|
| 0 | No spectroscopic type | 19,138 | 31,057 | Exclude |
| 1 | Normal Ia | 353 | 356 | Ia |
| 4 | Ia-pec | 1 | 1 | Separate |
| 5 | H-free SN, Ia/Ib/Ibc ambiguous | 1 | 1 | Exclude |
| 23 | IIb | 5 | 5 | Ordinary CC |
| 29 | II | 44 | 46 | Ordinary CC |
| 32 | Ib | 1 | 1 | Ordinary CC |
| 33 | Ic | 4 | 4 | Ordinary CC |
| 39 | Ibc | 7 | 7 | Ordinary CC |
| 41 | SLSN-I | 15 | 16 | Separate |
| 66 | SLSN-II? | 1 | 1 | Uncertain, exclude |
| 80 | AGN | 49 | 54 | Contaminant, exclude |
| 81 | TDE | 1 | 1 | Contaminant, exclude |
| 82 | M-star | 2 | 2 | Contaminant, exclude |
| 101 | Ia? | 59 | 59 | Uncertain, exclude |
| 122 | IIn? | 1 | 1 | Uncertain, exclude |
| 129 | II? | 16 | 16 | Uncertain, exclude |
| 139 | Ibc? | 4 | 4 | Uncertain, exclude |
| 141 | SLSN-I? | 3 | 3 | Uncertain, exclude |
| 180 | AGN? | 1 | 1 | Uncertain, exclude |

Other documented codes absent from both count lists: 3 (Iax), 20 (IIP), 21 (IIL), 22 (IIn), and 42 (SLSN-II). Zero occurrence is not a reason to add these to the frozen class mapping. Definitions: [official dictionary](https://des-sn-dr.readthedocs.io/en/latest/0_DATA.html); DIFFIMG counts: [release record](https://zenodo.org/records/14332953).

There are 85 question-mark classifications in each product. SMP additionally contains one peculiar Ia, one ambiguous SNI, 15 definite SLSN-I, and 52 definite listed non-SN contaminants. Keep these categories disjoint. In particular, **76 definite non-Ia SNe includes 15 SLSNe; the ordinary CC denominator is 61**.

The pinned SMP README incorrectly lists 39 objects for type 39 and omits type 101. The checked binary contains seven type-39 and 59 type-101 objects. It also omits the question mark for code 66; the official dictionary and DIFFIMG nomenclature identify that code as uncertain. The conservative rule excludes it.

These are release-level transient spectroscopic tags, not independent reclassification of spectra. A host `HOSTGAL_SPECZ` establishes redshift, not SN class. The `REDSHIFT_HELIO` field can fall back to photometric redshift; its source/error must be retained. Do not treat every populated redshift as spectroscopic or use released classifier probabilities as truth.

The label is already attached to each HEAD row. `PTROBS_MIN/MAX` index that object's PHOT rows (SNANA one-based inclusive pointers); `MJD=-777` is a separator. SNID is the cross-product key. Uniqueness is verified for SMP HEAD; complete pointer validity has prior cohort-validation evidence. A future SMP-DIFFIMG join must still check unique IDs, coordinate agreement and label discrepancies; shared IDs do not justify counting two measurements as two SNe.

## 4. Photometry, calibration and metadata

| Quantity | Released representation and audit implication |
|---|---|
| Time | `MJD`, days; preserve observer-frame epochs. Header `PEAKMJD` is approximate and is not the frozen sampled-peak feature. |
| Bands | Canonical `g,r,i,z`; filter-token decoding is a schema adapter, not a change of passband. |
| Flux | `FLUXCAL`, fixed numerical convention `m = 27.5 - 2.5 log10(F)` for positive F. It is not raw ADU. |
| Uncertainty | `FLUXCALERR`, same units as flux; retain released errors, including corrections. |
| Forced observations | Zero/negative fluxes and nondetections retained before the feature operator. Prior local quality audit counted 517,801 valid negative SMP flux rows. |
| Image metadata | `CCDNUM`, `IMGNUM`, `FIELD`, `PSF_SIG1`, `SKY_SIG`, `ZEROPT`, `GAIN`, pixel position; available fields need not be complete for every epoch. |
| Flags | `PHOTFLAG`, `PHOTPROB`. Existing SMP rule is structural validity, not empirical bit rejection. Exact release-specific invalid-bit semantics remain a qualification. |
| Depth | Ten fields; C3/X3 deep, remaining survey fields shallow. Use released field values and validated mapping, not a guessed table from generic examples. |
| Host | Host IDs, separation/DDLR, confusion, SPECZ/PHOTOZ, masses/SFR and errors, griz magnitudes, local surface-brightness fields. Missing sentinels and host ambiguity are not physical measurements. |
| Galactic extinction | `MWEBV` and error; frozen primary policy retains observed fluxes and stratifies/conditions on extinction. No new dereddening. |

Schema evidence is the pinned local `0_DATA/README.md`, product README and existing validation outputs; the [official dictionary](https://des-sn-dr.readthedocs.io/en/latest/0_DATA.html) is a cross-check. In particular, the generic description of `HOSTGAL_SB_FLUXCAL_*` is inconsistent about flux versus magnitude wording: consult the actual units before treating it as SBMAG. It is not automatically Stage 3C's controlled host-brightness coordinate.

DIFFIMG has documented warning bits 1/2, fit/image error bits 8 through 512, image-quality bits 1024/2048, and detection bit 4096. A detection bit is not a forced-flux validity requirement; these definitions must not be transplanted into SMP without evidence. [DIFFIMG README](https://zenodo.org/records/14332953/files/DES-SN5YR_DIFFIMG.README?download=1).

### Versioned correction state

The local immutable SMP README states DCR/chromatic magnitude corrections and host-surface-brightness error corrections are applied. Its fit configuration independently removed additional correction-file application on 2024-06-12. Older paper prose and mutable main do not override this product state.

The calibration family is `calib_DES-SN5YR_DES.fits.gz` (SHA-256 `4412a60ca41eb2cd8b32c038b352a551949a386cdc48acc0b059899bf46496c8`), with generating input SHA-256 `0917ddc66647ff613421a38c5df9a67d2113a7983e28889206c85c29b5c9e1ee`. It names `DES-SN3YR_DECam/DECam_[griz].dat`: nominal Y3A2 photon-counting responses including atmosphere at airmass 1.3. Individual response hashes are retained in `docs/provenance/pinned_des_sdss_products.md`.

The released flux plus its named calibration artifact is the pinned system. Calibration offsets in that artifact (`g=-0.0008`, `r=-0.0116`, `i=-0.0090`, `z=+0.0071` mag) are not an instruction to apply a second flux correction. No ad hoc Fragilistic adjustment is justified. The release points to an external calibration filename, whose local checksum is separately frozen; this link is STRONG rather than a bundled checksum identity.

SPCC release documentation defines the same 27.5 FLUXCAL numerical convention and AB reference. Its loader defaults to flux scaling 1.0. **No unit rescaling is needed to feed the frozen raw-observation operator.** Equal numerical units do not establish identical physical response or calibration. The actual SPCC HBOOK KCOR and its generating chain remain unrecovered.

## 5. Frozen feature compatibility

The required numerical inputs are time, band, flux and flux error. Object ID, truth label and redshift are metadata. Both sources can supply these, through different parsers. The SPCC cleaner's simulated-truth requirement must not be imposed on real data; share the mathematical operator, not the simulation-specific parser.

The frozen Tier-4 operator sorts griz observations; selects positive flux/error with S/N >= 3; falls back globally when none is active; and uses within-active-window band fallback when necessary. All four colour representatives must remain positive. There is no additional minimum duration or observations-per-band rule: one positive selected observation per band can be mathematically sufficient but scientifically weak. No interpolation, rest-frame conversion or simultaneous-colour reconstruction is allowed.

Let L(x)=log10(1+max(x,0)), S(x)=sign(x)log10(1+abs(x)), and R_b be the mean of up to three largest positive selected fluxes in band b. Standard deviations use population normalization (`ddof=0`).

| Frozen feature | Exact operational quantity | DES compatibility / limitation |
|---|---|---|
| `z_peak_flux` | L(max z flux) | Compatible; calibration and sampled-max bias remain |
| `r_mean_flux` | S(mean r flux) | Compatible; active selection/window matters |
| `peak_color_g_minus_r` | clipped -2.5 log10(R_g/R_r) | Compatible; asynchronous, positive support required |
| `i_peak_flux` | L(max i flux) | Compatible; same flux convention required |
| `peak_color_r_minus_i` | clipped -2.5 log10(R_r/R_i) | Compatible; not a simultaneous colour |
| `peak_color_i_minus_z` | clipped -2.5 log10(R_i/R_z) | Compatible; z support can select sample |
| `g_mean_flux` | S(mean g flux) | Compatible; depth/SNR truncation matters |
| `r_peak_flux` | L(max r flux) | Compatible; no interpolation |
| `z_std_flux` | L(std z flux) | Compatible; cadence, noise and sample size matter |
| `i_amplitude` | L(max i - min i flux) | Compatible; window and extrema matter |
| `i_std_flux` | L(std i flux) | Compatible; same population-std definition |
| `time_span` | last minus first active/fallback time | Compatible; observed window, not intrinsic duration |
| `z_time_of_peak` | sampled z argmax time minus active start, clipped to span | Compatible; first maximum/tie behavior preserved |
| `i_time_of_peak` | corresponding i quantity | Compatible; observer-frame days |
| `r_time_of_peak` | corresponding r quantity | Compatible; not physical peak error |
| `r_std_flux` | L(std r flux) | Compatible; same selection dependency |

Colour clipping is [-5,5]. Negative fluxes can contribute in fallback cases; they must not be discarded at ingestion. Missing bands ultimately reject the object, not justify invented zero-valued accepted features. Previous equivalence tests support the Phase-3 implementation of this operator.

**Model-lineage condition:** the earlier Tier-1 extractor reconstructs a bandwise grid and uses peak-ratio colours. The current Tier-4 operator uses raw selected epochs and top-three colours. Identical 16 column names do not establish identical mathematics. Before evaluating an existing classifier, freeze its serialized model hash, training table, feature-builder version, feature order, scaler/imputer, binary label mapping and threshold. An old model trained on Tier-1 inputs cannot automatically consume Tier-4 DES features. This audit did not establish that model-specific chain.

## 6. Population support and selection

The frozen redshift range is retained; the older preliminary suggestion of 0.10-0.60 is not adopted.

| DES subset | Normal Ia | Ordinary CC |
|---|---:|---:|
| Full pinned SMP header | 353 | 61 |
| 0.05 <= z_helio <= 0.30, direct HEAD count | 140 | 52 |
| Four-band survivors, prior validated result | 134 | 51 |
| Lost to support rule in this region | 6 | 1 |

The 52 pre-support CC comprise II=39, IIb=3, Ib=1, Ic=3, Ibc=6. The present audit does not assign the single subsequent rejection to a subtype without an object-level selection manifest. Full-release NOBS ranges are Ia 3-235, IIb 93-139, II 70-206, Ib 112, Ic 98-112, Ibc 64-109. These are counts across bands, not cadence or phase coverage.

| Dimension | Evidence and remaining restriction |
|---|---|
| Class | SPCC has simulated Ia/II/Ibc families; binary Ia versus ordinary CC is plausible. Truth taxonomies and rare subclasses are not identical. |
| Redshift | Prior SPCC feasibility work records broad overlap and 20,992 compact survivors; the selected model's exact training sample within 0.05-0.30 remains to be pinned. DES best redshift is not uniformly spectroscopic. |
| Epochs and bands | DES epoch schema and four-band survivors verified in prior audits. No new SPCC-DES phase/cadence overlap calculation was made. |
| S/N | Same S/N-active rule is computable; equal S/N distributions or detection functions are not established. Structural validity is not equal observational support. |
| Features | Existing DES builder produced all 16 columns for 185 supported objects. Compatibility with the selected trained model remains conditional on lineage. |
| Field and host | Shallow/deep, extinction, host brightness and redshift provenance are available controls. Small CC counts limit simultaneous stratification. |

Spectroscopy is a selection mechanism: brightness, redshift, transient phase, host contrast, follow-up availability and target prioritization affect which objects obtain secure types. SMP processing and four-band survival impose additional selection. The secure cohort is neither a random subset of all candidates nor a volume-limited SN sample. Classification precision depends on class prevalence; the observed secure-label class ratio is not the DES population prior. Redshift/subtype matching cannot repair unobserved selection, host ambiguity or missing support. No matching on compact features is justified, because it can remove the survey effect of interest.

## 7. Independence and feasible scientific claims

Stage 3C used a fixed Hsiao Ia model and found mixed operator predictions. It supplies neither a real classification validation nor CC evidence. Its predictions cannot be reselected based on DES classification success.

The earlier Stage-1 manifest explicitly marks a first outcome reveal for 134 DES Ia objects; Stage 2 used their feature table in passband/cadence comparisons. Both say no classifier was run. Therefore:

- DES observations are external to SPCC simulations, but the existing DES cohort is not wholly unseen by the research process.
- A previously frozen SPCC model can still receive an external evaluation if no DES exposure changed the model, feature policy, calibration choices, threshold or sample definition. That timeline needs a model-specific record.
- An adaptively revised method evaluated on these same objects is exploratory. A retrospectively chosen split cannot erase previous exposure.
- Future label-blind construction must use documented instrument/metadata inputs. Target classification success must not choose corrections or exclusions.

| Proposed use | Feasibility | Defensible claim |
|---|---|---|
| A. Independent SPCC-to-DES classification test | CONDITIONAL | External evaluation on secure, supported DES SNe if model lineage and absence of target-driven tuning are documented; not a newly blinded confirmation automatically. |
| B. Survey-transfer robustness | CONDITIONAL | Measure performance of a fixed simulation-trained decision function on real selected observations; failure combines simulation realism, population, selection, pipeline and instrument effects. |
| C. Matched populations | PARTIAL | Predefined redshift, broad subtype, extinction and field controls; no assurance of adequate counts in every joint stratum or complete selection correction. |
| D. Uncertainty for ordinary CC | FEASIBLE BUT COARSE | Object-level binomial intervals for recall and stratified object bootstrap for appropriate aggregate metrics; intervals conditional on this selected sample. |
| Physics-based SPCC passband normalization | NO-GO | Missing historical SPCC response-generation link still blocks physical attribution; modern DES curves are not a substitute. |

For scale only, not as a measured performance interval: with n=51 CC, one error changes recall by about 1.96 percentage points. The worst-case binomial normal-approximation 95% half-width is about 14 percentage points (1.96*sqrt(0.25/51)); exact or Wilson intervals should be used later, especially near boundaries. Even all 61 CC give about 12.5 percentage points by this planning approximation. Ia n=134 gives about 8.5 points. Subtype Ib n=1 cannot support a stable rate estimate. Bootstrap resampling does not manufacture additional independent SNe or account for unknown selection bias. No actual classification interval was calculated.

## 8. Closure conditions and recommendation

The available DES observations are sufficient to justify a carefully bounded external evaluation, subject to three explicit conditions:

1. Pin the exact existing SPCC model and its training/preprocessing lineage to the intended 16-feature operator, and record whether any choices postdate DES outcome exposure.
2. Freeze an evaluation protocol for the existing support and secure-label rules, object-level uncertainty and abstention/rejection reporting, without changing the earlier methodology retrospectively. Resolve or explicitly retain the SMP quality-flag limitation; do not borrow DIFFIMG masks.
3. Restrict interpretation to the spectroscopically selected common-support cohort and separate empirical transfer performance from the still-blocked SPCC passband-normalization claim.

**Final recommendation: CONDITIONAL GO.** The smallest next action is a read-only model-and-training-artifact lineage check against the frozen Tier-4 builder and the dated DES-exposure manifests. No additional large download is needed for that decision. If the requirement is instead a fully untouched confirmatory test or a provenance-complete physical SPCC-to-DES passband correction, current evidence does not meet it.

This report completes the suitability audit. It does not authorize model execution, a new transformation, alteration of the frozen population, or a manuscript.
