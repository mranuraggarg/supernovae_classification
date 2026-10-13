# Stage 4: SPCC Model and Training-Lineage Audit

Date: 2026-10-13. Repository baseline: `417871626386c655f9c4fd1aebb5175e467d8f74`.

## Decision

**NO-GO for evaluation of an existing SPCC classifier on the frozen DES Tier-4 feature table using the currently available artifacts.**

No serialized classifier or fitted preprocessing artifact was located. Historical results establish completed training experiments, but do not supply the saved decision function. The compact Tier-1 training path is mathematically incompatible with the Tier-4 DES representation despite identical column names. The Tier-4 training path is compatible at the feature-extraction level, but also applies temporal transformations and training-set standardization whose fitted state is not preserved as an artifact.

This is a model-artifact and lineage decision, not a statement that DES observations are unsuitable. No model was loaded, trained, retrained, evaluated, or modified. No predictions or DES classification outcomes were inspected. No replacement model was created.

## 1. Search scope and evidence standard

The audit searched the repository working tree including ignored files, historical script references, current and archived result records, and paths in all locally reachable Git history. Companion project resources under `/Users/anuraggarg/work/Feature normalization Phase` were searched for conventional model/preprocessing filenames. The `/Users/anuraggarg/work/supernovae_classification` location is a symlink to the current repository, not a second independent artifact store.

The repository has no `models/` directory. Filename searches found no `.pkl`, `.pickle`, `.joblib`, `.ubj`, `.model`, `.bst`, `.pt`, `.pth`, `.h5`, or `.onnx` model candidates. A content check of 89 JSON files in artifact areas found no XGBoost `learner`/`gradient_booster` signatures. Git object-path searches found no historical `models/`, saved-baseline-XGBoost, or scaler artifacts. No archive files were found in these artifact areas. Raw astronomical dataset directories were excluded from the final JSON-content scan; a broader scan was interrupted while reading raw transient JSON files. Filename matches such as `AT2020xgb.json` are astronomical objects, not evidence of a classifier.

This establishes absence in the inspected local resources and reachable history, not that models never existed elsewhere. Referenced Colab/download locations and missing relative output paths cannot be treated as recovered artifacts. No filesystem modification time was used as proof of a training date. Git commit timestamps below establish recorded history, not independently certified wall-clock training times.

## 2. Artifact inventory

All paths below are repository-relative unless stated otherwise.

| Candidate or resource | Located evidence | Missing artifact / implication |
|---|---|---|
| Original-paper optimized XGBoost | `archive/original_paper/notebooks/Supernovae_xgboost_classification_cuda.ipynb`, including saved-output/download records; `archive/original_paper/scripts/spcc_f1_score.py`; original result JSONs | Referenced `models/xgboost_optimized/xgboost_optimized.json` and `xgboost_scaler.pkl` absent. Notebook outputs do not constitute the model bytes. |
| Phase-1 repair XGBoost, with/without SMOTE | `archive/original_paper/scripts/xgboost_training.py`; `results/phase1_repair/xgboost_{with,without}_SMOTE_results.json`; repair notebook | `models/phase1_repair/xgboost/{with_SMOTE,without_SMOTE}/model.pkl` and `scaler.pkl` absent. Completed result records exist. |
| Original/repair random forest and linear models | Training scripts and summary JSONs under `archive/original_paper/scripts/` and `results/{original_paper,phase1_repair}/` | No fitted estimator/scaler recovered. These use the legacy representation and are not the proposed compact-XGBoost model. |
| Full Tier-1 XGBoost | `archive/phase2/scripts/phase2_tier1_xgb_importance.py` explicitly saves a booster | `models/phase2_tier1/xgboost_importance/full_tier1_baseline_xgb.json` absent. A save instruction alone is not proof that these exact bytes were produced. |
| Compact Tier-1 XGBoost | `archive/phase2/scripts/phase2_tier1_finalize_compact_baseline.py`; compact table, comparison, curve and interpretation outputs | Expected `models/phase2_tier1/compact_baseline/phase2_tier1_compact_baseline_xgb.json` absent. Finalizer's manifest/selection/metrics artifact set is not present in the inspected Tier-1 results directory. |
| Tier-2/Tier-3 compact experiments | Shared helpers and ablation/model-comparison results under `archive/phase2/` | Results are evidence of experiments, not a recoverable fitted estimator. Tier-1 source-table lineage remains relevant. |
| Tier-4 domain-transfer XGBoost | `archive/phase2/scripts/phase2_tier4_domain_swap.py`, `phase2_tier4_common.py`, `phase2_tier3_model_compare.py`; `archive/phase2/results/phase2_tier4/domain_swap_metrics.{csv,json}` and `variant_manifest.json` | Trained booster and means/stds are returned in an in-memory bundle; domain-swap script writes summaries, not that bundle. No serialized Tier-4 model found. |
| Tier-4 variants, reduced-feature, windowed and mixed-domain experiments | Scripts and archived results | Multiple experimental decision functions, not one deployable checkpoint. Mixed SPCC/PLAsTiCC training is not SPCC-only training. No saved model found. |
| Frozen DES target | `phase3_real_survey/results/real_feature_tables/des_compact_features.csv` and validated Phase-3 builder | Target table exists; availability does not solve missing source-model provenance. |

The Tier-4 domain-swap JSON explicitly records `baseline_manifest_found: false` and `baseline_metrics_found: false`. Its baseline-parity result is not a hash or identity proof for an earlier saved model. The domain-swap loop calls training separately for each test domain; it does not load one previously frozen checkpoint.

## 3. Training tables and checksums

Hashes below were computed read-only during this audit. They identify the current bytes, not their unproven identity with every historical training run.

| Table | Rows and class metadata | SHA-256 |
|---|---|---|
| `data/processed/spcc_features_tier1.csv` | 21,319; Ia 5,088; all other simulated classes 16,231 | `6abb0faa98963587c07cfb803cb7c074988a922051bf4926b65a0f892ef3af1a` |
| `data/processed/phase2_tier1_compact_baseline.csv` | Same 21,319 objects/class counts; selected Tier-1 columns | `50424499da2c2bcc2bc570d645f57024ad42dc7c048592b27e9a1af944648bfd` |
| `data/spcc/features/compact_features.csv` | 20,992 Tier-4 survivors; Ia 4,975; other classes 16,017 | `d628b4187f0f25d027907f37168167c6afea64faf7cc62ddfef5937fe8ce2bb2` |
| `phase3_real_survey/results/real_feature_tables/des_compact_features.csv` | Existing target, not used for model fitting or predictions in this audit | `8963b9e1fb4fe0b30cec58599ae01c53ecf7088a3ddd555b5aed73098d39f669` |

Tier-1 metadata counts: II 12,027; Ib 1,438; IIn 789; Ia 5,088; IIL 425; Ic 1,104; IIP 189; Ibc 259. Tier-4 counts: II 11,916; Ib 1,391; IIn 786; Ia 4,975; IIL 408; Ic 1,078; IIP 185; Ibc 253. Only class metadata and row counts were summarized, not feature values.

The original Tier-1 table stores multiclass IDs (Ia=1, II=2, Ibc=3, IIn=21, IIP=22, IIL=23, Ib=32, Ic=33). Binary training helpers reconstruct `1 if label_name == 'Ia' else 0`; they do not use those multiclass integers directly as binary truth. The Tier-4 table already stores binary 1/0. Training negatives include IIn and other simulated subclasses, whereas the DES primary set is restricted to definite normal Ia and ordinary II/IIb/Ib/Ic/Ibc. A restricted DES evaluation is possible in principle, but the training population must be described as its actual broader simulated non-Ia mixture.

## 4. Feature lineage and exact order

The frozen raw-observation builder is now at `archive/phase2/scripts/phase2_tier4_make_variants.py`, SHA-256 `7d2b1b7c23d685e652b4b7df383441537162f9dc3829f8edba380a8ddf2a4d76`. Its Phase-3 equivalent is `phase3_real_survey/scripts/build_real_survey_features.py`. Prior synthetic equivalence evidence is in `phase3_real_survey/results/feature_builder_equivalence.txt`; no equivalence test was rerun here.

The exact compact order, shared by the Tier-1 finalizer and Tier-4 configuration, is:

```text
z_peak_flux, r_mean_flux, peak_color_g_minus_r, i_peak_flux,
peak_color_r_minus_i, peak_color_i_minus_z, g_mean_flux,
r_peak_flux, z_std_flux, i_amplitude, i_std_flux, time_span,
z_time_of_peak, i_time_of_peak, r_time_of_peak, r_std_flux
```

This shared order conceals a change in mathematical definitions:

| Feature(s) | Tier-1 implementation | Frozen Tier-4 / DES implementation | Tier-1 compatibility |
|---|---|---|---|
| `z_peak_flux`, `i_peak_flux`, `r_peak_flux` | Maximum of zero-filled shared-time grid | Maximum of S/N-active or fallback selected band epochs | Not generally equivalent |
| `r_mean_flux`, `g_mean_flux` | Mean including inserted band-grid zeros | Mean of selected band observations | Not equivalent |
| `peak_color_g_minus_r`, `peak_color_r_minus_i`, `peak_color_i_minus_z` | Ratios of band maxima with small positive floor | Ratios of means of up to three largest positive selected fluxes | Not equivalent |
| `z_std_flux`, `i_std_flux`, `r_std_flux` | Population std over zero-filled grid | Population std over selected band epochs | Not equivalent |
| `i_amplitude` | Maximum minus minimum including grid fill | Maximum minus minimum of selected i observations | Not equivalent |
| `time_span` | Full retained observation-grid duration | Active/fallback-window duration | Not equivalent |
| `z_time_of_peak`, `i_time_of_peak`, `r_time_of_peak` | Maximum time relative to first retained observation | Selected maximum time relative to first active/fallback time, clipped to span | Not equivalent |

Tier-1 evidence: `feature_pipeline/extraction/spcc_features.py` and `feature_pipeline/interpolation/spcc_native_reconstruct.py`. The executable reconstruction uses unique observed times and explicit zeros for absent bands; it does not interpolate missing fluxes. Generic policy prose about filling/interpolation is not authoritative over this code.

Tier-4 uses positive flux/error and S/N >= 3 to define active epochs, global fallback if none are active, and within-window band fallback. It rejects events without positive representatives in all griz bands. Colours are clipped to [-5,5]. Both families compress peak/std/amplitude as `log10(1+max(x,0))` and means as `sign(x)*log10(1+abs(x))`; sharing those outer transforms does not make their inputs equivalent. Default SPCC loader flux/time scaling is 1.0. Raw flux is FLUXCAL; time is in days. No rest-frame time transformation occurs in the feature builder.

No fitted imputer was found. Tier-1's zero filling is an extractor operation, not a learned imputer. Tier-4 support rejection is not missing-value imputation. Available model-training helpers do not document an independently serialized missing-value policy; one cannot substitute a new imputer for an absent training-time state.

### Additional Tier-4 training transformations

`phase2_tier4_common.py:train_test_arrays` casts to float32 and then applies:

```text
span = max(time_span, 1e-6)
r/i/z_time_of_peak = clip(time_of_peak / span, 0, 1)
time_span = log10(1 + span)
```

It then calls the Tier-3 XGBoost helper, which standardizes using the final training-set mean and population standard deviation, replacing zero standard deviations by one. These transformations must be applied in that order, using the original training statistics. The raw DES feature table is therefore not directly the model input matrix. Recovering and applying documented preprocessing would not redefine the extractor, but the fitted means/stds are absent. Refitting them on DES would invalidate the intended external test.

The compact Tier-1 finalizer also standardizes on its train+validation subset. It saves only the booster; its code does not save the fitted means/stds. Recovering that booster alone would still not provide a complete deployment artifact, and its extractor mismatch would remain.

## 5. Model-lineage table

For every candidate, **serialized-model path exists: no; serialized-model SHA-256: unavailable; fitted-preprocessing SHA-256: unavailable; exact training-library version: not established**. Current environment versions cannot certify historical versions.

| Candidate | Training configuration recoverable from source/records | Threshold and labels | Reproducibility / eligibility |
|---|---|---|---|
| Legacy optimized XGBoost | Flattened light-curve arrays, notebook optimization and scaler references; no recovered exact model/input bundle | Legacy evaluator uses >=0.5; binary semantics must be checked against its original one-hot class encoding | Not compatible with a 16-column Tier-4 matrix; absent model and scaler |
| Phase-1 repair XGBoost | Flattened arrays; StandardScaler; optional SMOTE; split/model/SMOTE seed 42; Optuna 50 trials without a pinned sampler seed in inspected script | Uses estimator `predict`; argmax of stored one-hot training labels, not the explicit compact label-name mapping | Completed result records, but no model/scaler or complete versioned training-array identity |
| Full Tier-1 XGBoost | 31 ordered features from `FEATURE_SETS['full_tier1_baseline']`; reconstructed-grid table; binary logistic, hist, logloss; seed 42; validation search | Ia=1, all other simulated labels=0; >=0.5 | Different dimensionality and definitions; no checkpoint |
| Compact Tier-1 XGBoost | 16 columns above from Tier-1 CSV; train/test 80/20; inner validation 20%; seed 42; train+validation standardization; validation-selected rounds/parameters | Ia=1, others=0; >=0.5 | Historical comparisons establish an experiment, not recovered exact booster; incompatible feature definitions |
| Tier-2/Tier-3 compact variants | Tier-1 compact source plus experiment-specific subsets/model families; train-set standardization | Binary helper >=0.5 | Not an interchangeable Tier-4 model; no fitted artifact |
| Tier-4 SPCC-only domain experiment | Raw-epoch Tier-4 table; split IDs inherited from baseline context; temporal transforms above; train-set scaling; seed 42; validation-selected XGBoost | Ia=1, others=0; >=0.5 | Closest candidate mathematically, but model, scaler state and exact selected-run configuration not serialized in domain-swap output |
| Tier-4 mixed/ablation/windowed variants | Distinct populations, feature subsets or preprocessing choices documented in their scripts/results | Binary helper mapping | No recovered checkpoint; mixed-domain variants are not SPCC-only; cannot choose a variant retrospectively to fit DES |

The full Tier-1 order is the concatenation, in `phase2_tier1_benchmarks.py`, of 23 `PHOTOMETRIC_CORE_FEATURES`, five `PEAK_TIME_FEATURES`, and `observation_count,time_span,total_snr`. It is not a 16-feature model.

The shared compact XGBoost search uses three configurations: depth/eta/subsample/colsample/min_child_weight/lambda = `(3,.05,.8,.8,1,1)`, `(4,.05,.9,.9,1,1)`, `(5,.03,.8,.8,2,1.5)`. Compact finalization uses a 400-round cap and 30-round early stopping, selecting validation PR-AUC; final refit uses selected rounds and training negative/positive class ratio. These are a **search recipe**, not recovered final hyperparameters. Domain-swap summaries omit the per-run fitted bundle and selected hyperparameters, so its baseline reference cannot fill that gap.

Phase-1 repair result JSONs do record final parameter choices. Without SMOTE: n_estimators=900, max_depth=8, learning_rate=0.093782495488377, subsample=0.952069378682005, colsample_bytree=0.6718735884319078, gamma=0.05704574956513833, reg_lambda=0.9974838333013512. With SMOTE: 700, 9, 0.09411172683787414, 0.8336019992984126, 0.534162333222795, 0.01507870520170107, 1.7761585068293582 respectively. These records do not recover fitted trees or scaler arrays.

Recipes could potentially support a future reproduction effort, but an exact historical decision function is not reproducible from the audited records alone. Training again would create a new artifact and is outside this task.

## 6. Target-exposure timeline

| Recorded event | Git evidence | Interpretation |
|---|---|---|
| Phase-1 repair result summaries | `a3840dac`, 2026-03-11 00:31:06 +04:00 | Recorded experiments precede the recorded DES exposure |
| Tier-1 implementation/experiments | `a58d3669`, 2026-03-11 20:04:30 +04:00 | Recipe and historical evidence precede exposure; no recovered checkpoint |
| Processed Tier-1 data | `a5e35629`, 2026-03-12 20:39:31 +04:00 | Table history exists; current hash is not a historical training manifest |
| Tier-4 domain results | `ee7f888d`, 2026-03-16 20:09:21 +04:00; later changes `86693d44`, `15859499`, 2026-06-20 | Multiple pre-exposure result versions; not one identified deployable model |
| Phase-2 archive move | `98ffc1bd`, 2026-10-04 15:01:33 +04:00 | Path relocation, not a training date |
| DES primary Stage-1 outcome reveal | `3e6878f0`, 2026-10-06 19:10:46 +04:00 | Manifest states `first_outcome_reveal: true`, 134 DES Ia, classifier_run=false |
| DES Stage-2 operator comparison | `93155710`, 2026-10-06 19:21:40 +04:00 | Manifest records use of DES feature table, no classifier execution, no outcome-fitted coefficients |

These commit times are the earliest inspected recorded evidence of the relevant artifacts/events, not exact process start times. DES exposure had occurred by the Stage-1 commit; the audit cannot prove that nobody inspected data earlier.

**Candidate classifications:** legacy, repair, full Tier-1, compact Tier-1, Tier-2/3, and Tier-4 candidates are all **C: timing of the exact frozen classifier cannot be established**. Their documented recipes/results generally precede exposure, but no serialized model plus preprocessing hash can be linked to that boundary. None qualifies as A merely because its script or result is old. No recovered artifact establishes B either.

There is no DES-driven tuning in the inspected pre-exposure training scripts; this is limited positive evidence. It does not prove the independence of an unidentified artifact or justify selecting one of several historical variants after target outcomes were known. Stage-1/2 manifests prove the target has been exposed even though classifier metrics were not calculated there.

## 7. Eligibility matrix

| Candidate | Tier-4 mathematical input | Binary target | Reproducible fitted preprocessing | Threshold evidence | Pre-DES model freeze | Bounded evaluation now |
|---|---|---|---|---|---|---|
| Legacy / repair | FAIL | PARTIAL: original encoding required | FAIL | Script only | C | NO |
| Full Tier-1 | FAIL | Broad binary mapping available | FAIL | 0.5 recipe | C | NO |
| Compact Tier-1 / Tier-2/3 | FAIL despite 16 names | Broad binary mapping available | FAIL | 0.5 recipe | C | NO |
| Tier-4 SPCC-only | PASS at extraction; preprocessing required | PASS for restricted DES evaluation with training-mixture caveat | PARTIAL recipe, absent fitted state | 0.5 recipe | C | NO: no checkpoint |
| Tier-4 mixed/subset/variants | Depends on variant; not interchangeable | Broad binary mapping available | Absent fitted state | Recipe only | C | NO |

## 8. Provenance gaps and scientific implications

The minimum missing bundle for the nearest candidate is an existing Tier-4 SPCC-only booster, its exact training mean/std arrays, ordered input schema including the temporal transforms, training-row identity and table hash, selected hyperparameters/round count, library version, binary mapping and threshold, and dated evidence linking that bundle to the pre-DES freeze. An independently checksummed archived bundle could change the decision; a path in a script cannot.

A recovered compact Tier-1 booster would not resolve the feature-definition mismatch. A recovered Tier-4 booster without scaler arrays could only be assessed further if the exact training subset, ordering and numerical preprocessing were demonstrably recoverable without retraining or target fitting. Current result summaries do not establish that complete chain.

The earlier suitability audit's conditional go concerned DES data availability. This narrower audit fails the existing-model gate. The classifier cannot be evaluated faithfully merely by selecting the same column names, and the historical model cannot be recreated by silently rerunning a training recipe. Such a run would require a separately specified new experiment and a truthful post-exposure status.

**Final recommendation: NO-GO with current artifacts.** No suitable existing model was found. Stop before prediction or hypothesis testing. The only artifact-recovery route identified here is locating the original trained Tier-4 bundle and its contemporaneous provenance; this audit neither creates nor substitutes a replacement.
