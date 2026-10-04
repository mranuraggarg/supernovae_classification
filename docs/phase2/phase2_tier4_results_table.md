# Phase-2 Tier-4 Results Tables

## 1. Domain Swap Results

| Train | Test | F1 | PR-AUC | ROC-AUC |
|--------|--------|--------|--------|--------|
| SPCC | SPCC | 0.840747 | 0.912002 | 0.973911 |
| SPCC | noise | 0.826299 | 0.905817 | 0.971004 |
| SPCC | flux_scale | 0.772947 | 0.837949 | 0.951203 |
| SPCC | short_span | 0.819421 | 0.902578 | 0.969842 |
| SPCC | no_z | 0.692105 | 0.683592 | 0.915440 |
| SPCC | no_i | 0.652222 | 0.725259 | 0.908337 |
| SPCC | PLAsTiCC | 0.516821 | 0.548993 | 0.736502 |
| SPCC + PLAsTiCC | PLAsTiCC | 0.690312 | 0.720417 | 0.880317 |
| SPCC + PLAsTiCC | SPCC | 0.822431 | 0.903002 | 0.971228 |


## 2. Mixed Training Summary

| Training set | Mean F1 | Mean PR-AUC |
|-------------|---------|------------|
| spcc | 0.702118 | 0.780441 |
| spcc + noise | 0.709332 | 0.789554 |
| spcc + flux_scale | 0.707992 | 0.784331 |
| spcc + short_span | 0.695004 | 0.748992 |
| spcc + no_i | 0.709881 | 0.763442 |
| spcc + no_z | 0.698774 | 0.770552 |
| spcc + plasticc | 0.746827 | 0.799876 |


## 3. External Domain Transfer

| Model | Train | Test | F1 | PR-AUC | ROC-AUC |
|--------|--------|--------|--------|--------|--------|
| compact | SPCC | PLAsTiCC | 0.3909 | 0.4218 | 0.6054 |
| compact | SPCC + PLAsTiCC | PLAsTiCC | 0.6903 | 0.7204 | 0.8803 |


## 4. Perturbation Sensitivity Summary

| Variant | ΔF1 from baseline |
|---------|------------------|
| noise | very small |
| short_span | very small |
| flux_scale | moderate |
| no_z | large |
| no_i | largest |
| plasticc | very large |


## 5. Notes

- Baseline compact model reproduced Tier-1 results exactly.
- Performance degrades gradually under simulated perturbations.
- Missing bands cause large performance loss.
- External dataset transfer is weak without mixed training.
- Mixed training with PLAsTiCC gives best overall generalization.
- Results in this table correspond to the **first parity upgrade** feature pipeline.
- A stricter second upgrade using same-epoch color interpolation was tested but reduced the usable PLAsTiCC sample size from ~7800 to ~2700 events, which was considered too restrictive for the main Tier-4 evaluation.
- The stricter pipeline improved parity but introduced strong sample-selection effects, so it is treated as a sensitivity test rather than the primary result.
- Updated shift-test results show moderate degradation under flux scaling and missing-band conditions, with the largest drop for the `no_i` variant.
- Feature-importance stability across domains is moderate rather than perfect, consistent with a stability score of ~0.46.
- Mixed-domain training with PLAsTiCC gives the best overall cross-domain performance but does not imply full survey independence.


## 6. Possible Sources of Error

The current Tier-4 results are informative, but cross-domain failure can still arise from multiple causes that are not yet fully separated.

- **Feature representation limitation**: the compact 16-feature set may be sufficient for SPCC and SPCC-derived perturbations but may not capture all survey-dependent structure needed for PLAsTiCC.
- **Feature distribution mismatch**: the same named features may have different numerical distributions across SPCC and PLAsTiCC because of cadence, depth, noise model, redshift coverage, and band behavior.
- **Decision-boundary specialization**: the classifier trained on SPCC may learn a boundary that works well in SPCC feature space but does not transfer to PLAsTiCC even if the features themselves remain physically meaningful.
- **Preprocessing mismatch**: feature extraction definitions may not be fully equivalent across datasets, leading to artificial domain shift.
- **Class-composition mismatch**: differences in class balance or subtype composition between SPCC and PLAsTiCC may alter the apparent transfer behavior.
- **Calibration mismatch**: model probabilities may become poorly calibrated on the external domain even when some discriminative signal is still present.
- **Missing-band sensitivity**: the strong drops under `no_z` and `no_i` indicate that cross-band structure is central to performance, so domain shifts affecting band coverage may strongly affect transfer.

## 6A. Parity Upgrade Sensitivity Test

A stricter feature-construction rule was evaluated in which peak colors were required to be measured at the same epoch using interpolation around the r-band peak.  
This reduced feature-definition mismatch between SPCC and PLAsTiCC, but it also caused heavy filtering of PLAsTiCC events due to cadence differences.

Approximate row counts during testing:

| Stage | Train row count |
|--------|----------------|
| original compact extraction | ~7848 |
| first parity upgrade | ~6235 |
| strict same-epoch upgrade | ~2757 |

Because the strict rule removed a large fraction of PLAsTiCC events, the main Tier-4 results are reported using the first parity upgrade.  
The strict upgrade is retained only as evidence that improved feature parity can increase transfer performance, but it does not represent the full external-domain distribution.

## 7. Missing Diagnostic Tests

The current Tier-4 experiment establishes that transfer changes across domains, but it does not yet fully explain why. The following diagnostic tests are needed to separate representation failure from boundary mismatch.

1. **Feature-space overlap test**
   - Apply PCA, UMAP, or t-SNE to the 16-feature space.
   - Color points by dataset and then by class.
   - Check whether Ia objects from SPCC and PLAsTiCC overlap in feature space, and likewise for non-Ia objects.

2. **Per-feature distribution comparison**
   - Compare SPCC Ia vs PLAsTiCC Ia and SPCC non-Ia vs PLAsTiCC non-Ia for each of the 16 compact features.
   - Use histograms or KDE plots together with summary statistics such as median, IQR, KS statistic, or Wasserstein distance.
   - Identify which features remain aligned across surveys and which shift strongly.

3. **Frozen-model probability diagnostics**
   - Evaluate the SPCC-trained compact model directly on PLAsTiCC.
   - Inspect probability histograms, calibration curves, confusion matrices, and PR curves.
   - Determine whether the failure is due to uncertainty, overconfidence, or systematic class shift.

4. **Cross-dataset nearest-neighbor sanity check**
   - In the 16-feature space, find the nearest SPCC neighbors for PLAsTiCC Ia objects and for PLAsTiCC non-Ia objects.
   - Check whether same-class neighbors dominate.
   - If same-class neighbors cluster together but the classifier still fails, the decision boundary is likely the problem; if not, the feature representation itself is unstable across surveys.

## 8. Future Tests

The next Tier-4 experiments should focus on diagnosing the source of cross-domain failure rather than only adding more training combinations.

1. Run PCA/UMAP feature-space visualization across SPCC and PLAsTiCC.
2. Perform per-feature SPCC–PLAsTiCC distribution comparisons for both Ia and non-Ia classes.
3. Evaluate frozen SPCC-model calibration and probability behavior on PLAsTiCC.
4. Perform nearest-neighbor feature-space consistency checks across datasets.
5. Reassess the compact feature set only after the above diagnostics determine whether the main issue is representation mismatch or classifier specialization.


## 9. Working Interpretation

At the current stage, the Tier-4 experiment should be interpreted as follows:

- The compact feature representation is robust under moderate SPCC-derived perturbations.
- Direct cross-survey transfer from SPCC to PLAsTiCC remains limited.
- Mixed-domain training improves transfer substantially, but this does not by itself imply a universal survey-independent model.
- The remaining unresolved question is whether the cross-survey gap is caused primarily by feature-space mismatch, preprocessing mismatch, or a survey-specific decision boundary.


## 10. Implementation Findings and Remaining Risks

During Tier-4 validation, several implementation-level checks were performed to ensure that cross-survey results are not caused by coding errors.

[P1] A real PLAsTiCC time-feature bug was identified and fixed in `phase2_tier4_make_variants.py` (lines ~102 and ~127). The band peak times (`r/i/z_time_of_peak`) are now measured from the event start, not from each band’s own first timestamp. The previous implementation could make band peak times incomparable to SPCC features.

[P2] A residual PLAsTiCC consistency risk remains in `phase2_tier4_make_variants.py` (lines ~113 and ~131). If a band is missing in the detected subset, the code falls back to that band’s full history, while `time_span` still comes from the detected window. This can produce cases where a band peak time exceeds the event `time_span`. An explicit audit for this condition was added in `phase2_tier4_plasticc_audit.py`.

[P2] The Tier-4 evaluation pipeline itself is mathematically sound. Split validity is checked in `phase2_tier4_common.py`, and model selection is performed on the training data only before final evaluation in `phase2_tier3_model_compare.py`. The main remaining caution concerns feature-definition parity, not train/test leakage.

[P2] Cross-survey feature parity is still not exact. SPCC compact features are derived from reconstructed bandwise grids (`spcc_features.py`), while PLAsTiCC compact features are computed directly from raw/detected observations (`phase2_tier4_make_variants.py`). Therefore, SPCC→PLAsTiCC transfer results are now interpretable, but they are not yet a perfect apples-to-apples survey comparison.

[P1] The SPCC-side variant math behaves as intended. Noise injection, `no_z`, `no_i`, `short_span`, and `flux_scale` are applied directly in compact-feature space in `phase2_tier4_common.py`, and the domain-swap pipeline correctly trains on the SPCC split and evaluates on the requested target domain.

These checks indicate that the current Tier-4 conclusions are not dominated by coding errors, but some remaining differences in feature construction between surveys may still contribute to the observed cross-domain performance gap.