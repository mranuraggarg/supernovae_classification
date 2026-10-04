

# Phase 2 Tier 5 Working Plan

> Status: PLANNED
>
> Created: June 2026
>
> Parent phase: Phase 2 Tier 4
>
> Motivation: Tier-4 demonstrated that direct SPCC→PLAsTiCC transfer is limited by class-conditional feature-space domain shift. The next objective is to identify which compact features remain stable across surveys and whether transfer can be improved through feature-space harmonization.

Branch: `phase2-tier5-invariant-features`

---

## 1. Objective

Determine whether a subset of compact physically interpretable features is sufficiently survey-invariant to support improved SPCC→PLAsTiCC transfer.

Primary question:

> Which compact features are both astrophysically meaningful and distributionally stable across survey domains?

---

## 2. Scientific Background

Tier-4 established that:

- SPCC and PLAsTiCC Type Ia centroids are not aligned in the compact 16-feature space.
- Cross-survey centroid shift exceeds within-survey class separation.
- Event-window harmonization improves transfer but does not eliminate mismatch.
- The dominant limitation is feature-space alignment rather than classifier instability.

Therefore, Tier-5 focuses on feature invariance rather than classifier optimization.

---

## 3. Working Hypothesis

A smaller subset of compact features may remain stable across survey domains.

If unstable features are removed or harmonized, cross-survey transfer performance should improve relative to the full compact feature set.

---

## 4. Planned Experiments

### Experiment A — Feature Shift Ranking

Measure class-conditional centroid shifts for every compact feature.

Outputs:

- feature stability ranking
- Ia shift ranking
- non-Ia shift ranking

Goal:

Identify the most survey-stable and most survey-unstable compact features.

### Experiment B — Invariant Core Discovery

Construct progressively smaller feature subsets:

- top 4 most stable features
- top 6 most stable features
- top 8 most stable features
- top 10 most stable features

Goal:

Determine whether an invariant core exists.

### Experiment C — Harmonized Transfer

Evaluate:

- original compact feature set
- invariant-core feature sets
- harmonized feature variants

Goal:

Measure transfer improvement after feature selection.

### Experiment D — Distribution Alignment

Test survey-independent transformations:

- robust scaling
- median/IQR normalization
- quantile alignment

Goal:

Determine whether simple harmonization reduces centroid mismatch.

### Experiment E — Final Transfer Evaluation

Compare:

- direct transfer
- mixed-domain training
- invariant-core transfer
- harmonized transfer

Goal:

Establish the strongest defensible cross-survey result.

---

## 5. Deliverables

Code:

- `phase2_tier5_feature_shift_ranking.py`
- `phase2_tier5_invariant_core.py`
- `phase2_tier5_harmonized_transfer.py`
- `phase2_tier5_distribution_alignment.py`
- `phase2_tier5_summary.py`

Results:

- invariant feature rankings
- harmonized-transfer benchmarks
- centroid-alignment diagnostics
- final Tier-5 summary report

---

## 6. Success Criteria

Tier-5 will be considered successful if at least one of the following is demonstrated:

1. A compact invariant feature subset produces improved SPCC→PLAsTiCC transfer.
2. Feature-space harmonization substantially reduces centroid mismatch.
3. The project can clearly identify which compact features are portable and which are survey-specific.

---

## 7. Expected Outcomes

Possible outcome A:

A survey-invariant compact core exists and improves transfer.

Possible outcome B:

Partial improvement is possible but survey-specific adaptation remains necessary.

Possible outcome C:

No stable invariant core exists, implying that compact interpretability and survey portability are fundamentally separate objectives.

---

## 8. Exit Condition

Tier-5 concludes when the project can make a defensible statement regarding:

> Whether compact physically interpretable features can be transferred across surveys directly, after harmonization, or only within survey-specific domains.
