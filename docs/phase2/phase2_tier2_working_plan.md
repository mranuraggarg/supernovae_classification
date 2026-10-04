# Phase 2 Tier 2 Working Plan

Branch: `phase2-tier2-compact-feature-ablation`  
Date: March 11, 2026

## 1. Goal

Phase 2 Tier 2 will explain why the frozen 16-feature compact baseline works by measuring the performance contribution of:

- each individual retained feature,
- each scientific feature block,
- progressive feature-set growth, and
- smaller candidate core subsets.

This phase is analysis-driven. We are not building a new preprocessing pipeline. We are reusing the Tier 1 compact dataset, split logic, and XGBoost evaluation setup so that every Tier 2 result is directly comparable to the frozen baseline.

## 2. Frozen Baseline

All Tier 2 experiments must be evaluated against the existing Tier 1 compact baseline artifacts already present in this branch.

Baseline source files:

- `results/phase2_tier1/phase2_tier1_compact_baseline_artifacts.json`
- `results/phase2_tier1/phase2_tier1_compact_baseline_metrics.json`
- `results/phase2_tier1/phase2_tier1_compact_baseline_manifest.json`
- `data/processed/phase2_tier1_compact_baseline.csv`
- `data/processed/phase2_tier1_compact_baseline.npz`
- `models/phase2_tier1/compact_baseline/phase2_tier1_compact_baseline_xgb.json`

Frozen reference metrics:

- F1: `0.8442299254`
- ROC-AUC: `0.9765883838`
- PR-AUC: `0.9277608810`

Frozen compact feature set:

- Brightness: `r_mean_flux`, `g_mean_flux`, `z_peak_flux`, `i_peak_flux`, `r_peak_flux`
- Color: `peak_color_g_minus_r`, `peak_color_r_minus_i`, `peak_color_i_minus_z`
- Variability: `i_std_flux`, `z_std_flux`, `r_std_flux`, `i_amplitude`
- Temporal: `r_time_of_peak`, `i_time_of_peak`, `z_time_of_peak`, `time_span`

## 3. Working Principles

- Reuse Tier 1 split logic and model-selection conventions wherever possible.
- Keep comparison fair: same compact dataset, same train/validation/test protocol, same primary metrics.
- Prefer small reusable utilities over four completely independent scripts if that reduces drift.
- Every experiment must emit machine-readable JSON or CSV plus one human-readable Markdown summary.
- Store all new outputs under `results/phase2_tier2/` and `plots/phase2_tier2/`.

## 4. Deliverables

### 4.1 Code deliverables

- `phase2_tier2_feature_ablation.py`
  Purpose: leave-one-feature-out retraining across all 16 compact features.
- `phase2_tier2_block_ablation.py`
  Purpose: leave-one-block-out retraining across brightness, color, variability, temporal blocks.
- `phase2_tier2_subset_growth.py`
  Purpose: cumulative subset experiments showing performance growth as blocks are added.
- `phase2_tier2_minimal_core.py`
  Purpose: evaluate reduced candidate subsets such as top 5, top 8, and top 10.
- `phase2_tier2_summary.py`
  Purpose: consolidate outputs, compute rankings and category labels, and write final tables/plots.

Recommended implementation note:

- Put shared functions in one helper module if repetition starts growing. The obvious shared pieces are dataset loading, standardization, metric computation, baseline comparison, and result serialization.

### 4.2 Result deliverables

- `results/phase2_tier2/feature_ablation_metrics.csv`
- `results/phase2_tier2/feature_ablation_summary.md`
- `results/phase2_tier2/block_ablation_metrics.csv`
- `results/phase2_tier2/block_ablation_summary.md`
- `results/phase2_tier2/subset_growth_metrics.csv`
- `results/phase2_tier2/subset_growth_summary.md`
- `results/phase2_tier2/minimal_core_metrics.csv`
- `results/phase2_tier2/minimal_core_summary.md`
- `results/phase2_tier2/phase2_tier2_master_summary.json`
- `results/phase2_tier2/phase2_tier2_report.md`

### 4.3 Plot deliverables

- Feature ablation delta bar plot
- Block ablation delta bar plot
- Subset growth line plot
- Minimal core tradeoff plot

## 5. Experiment Design

### Experiment A: Leave-One-Feature-Out

Question: which individual compact features are essential versus replaceable?

Run 16 retraining jobs:

- each run drops exactly one compact feature,
- all remaining settings stay aligned with the frozen Tier 1 compact baseline,
- record F1, ROC-AUC, PR-AUC, and delta versus baseline.

Expected output columns:

- `feature_removed`
- `feature_group`
- `num_features`
- `f1`
- `roc_auc`
- `pr_auc`
- `delta_f1`
- `delta_roc_auc`
- `delta_pr_auc`
- `rank_by_delta_f1`

Decision use:

- input to essential/supportive/marginal/redundant labels,
- input to minimal-core candidate selection.

### Experiment B: Leave-One-Block-Out

Question: which physical information family drives the bulk of the classification signal?

Run 4 retraining jobs:

- remove all brightness features,
- remove all color features,
- remove all variability features,
- remove all temporal features.

Expected output columns:

- `block_removed`
- `remaining_feature_count`
- `f1`
- `roc_auc`
- `pr_auc`
- `delta_f1`
- `delta_roc_auc`
- `delta_pr_auc`

Decision use:

- determine which astrophysical signal family is dominant,
- define the most informative progression for subset-growth plots.

### Experiment C: Incremental Subset Growth

Question: how quickly does predictive power saturate as information is added back?

Base progression from the PDF:

1. brightness only
2. brightness + color
3. brightness + color + variability
4. full compact feature set

Recommended extension:

- also test `color only` and `temporal only` if time permits, because these provide cleaner interpretation of single-family signal strength.

Expected output columns:

- `subset_name`
- `feature_count`
- `included_blocks`
- `f1`
- `roc_auc`
- `pr_auc`
- `delta_f1`
- `delta_roc_auc`
- `delta_pr_auc`

Decision use:

- identify the information plateau,
- show whether most performance is recovered before all 16 features are present.

### Experiment D: Minimal Core Search

Question: can we define a smaller core subset that retains most of the compact baseline?

Candidate subsets should be chosen from:

- top-performing features from leave-one-feature-out,
- top compact features from Tier 1 permutation importance,
- scientific coverage across the four feature families.

Initial target subsets:

- top 5
- top 8
- top 10

Selection rule:

- do not choose purely by importance rank if that collapses the subset into one physical family,
- prefer candidates that preserve both performance and interpretability.

Expected output columns:

- `subset_name`
- `selection_rule`
- `feature_count`
- `feature_list`
- `f1`
- `roc_auc`
- `pr_auc`
- `delta_f1`
- `delta_roc_auc`
- `delta_pr_auc`

Decision use:

- determine whether a stable, scientifically interpretable mini-core exists.

## 6. Interpretation Framework

Each feature and block will be assigned one of four labels:

- Essential: clear negative delta when removed.
- Supportive: consistent but moderate negative delta.
- Marginal: very small negative delta.
- Redundant: near-zero or positive delta.

Proposed operational rule for the first pass:

- use delta F1 as primary ranking,
- use delta PR-AUC as tie-breaker,
- confirm that labels do not contradict the broader scientific interpretation.

These thresholds should be finalized only after seeing the empirical delta distribution.

## 7. Execution Plan For The Next Few Days

### Day 1: Freeze inputs and build shared Tier 2 scaffolding

- Verify the compact baseline files load cleanly from `results/phase2_tier1/` and `data/processed/`.
- Create `results/phase2_tier2/` and `plots/phase2_tier2/`.
- Implement shared Tier 2 helpers for:
  - loading compact rows,
  - building feature matrices from arbitrary subsets,
  - reusing stratified split logic,
  - training and evaluating XGBoost with consistent metrics,
  - computing baseline deltas,
  - writing CSV/JSON summaries.
- Add a quick baseline rerun check to confirm the Tier 2 helper path reproduces the frozen compact metrics closely enough.

Exit criteria:

- one reusable training/evaluation path exists,
- baseline parity check is passing,
- output directories and schema are fixed.

### Day 2: Finish Experiments A and B

- Implement and run leave-one-feature-out ablation.
- Implement and run leave-one-block-out ablation.
- Save clean tables and first-pass plots.
- Produce an initial ranking of most sensitive features and blocks.

Exit criteria:

- 16 feature-drop rows complete,
- 4 block-drop rows complete,
- deltas versus baseline are computed and stored.

### Day 3: Finish Experiments C and D

- Implement subset growth experiments.
- Use Experiment A plus Tier 1 importance outputs to define candidate minimal-core subsets.
- Run top 5, top 8, and top 10 core experiments.
- Compare compact baseline versus reduced cores.

Exit criteria:

- subset-growth table and curve exist,
- minimal-core table exists,
- at least one candidate subset is identified for scientific discussion.

### Day 4: Summarize and interpret

- Implement `phase2_tier2_summary.py`.
- Aggregate all Tier 2 outputs into one master summary.
- Assign feature and block labels.
- Draft `results/phase2_tier2/phase2_tier2_report.md`.
- Extract the main scientific statement for the branch README or paper update.

Exit criteria:

- all experiments are consolidated,
- final report is readable,
- key claim about dominant signal families is explicit.

## 8. Risks And Controls

- Risk: metric drift caused by silent changes in split logic or preprocessing.
  Control: reuse Tier 1 data sources and split helpers directly where possible.

- Risk: overinterpreting tiny metric differences.
  Control: compare both delta F1 and delta PR-AUC, and report absolute values alongside deltas.

- Risk: minimal-core selection becoming arbitrary.
  Control: tie subset choice to both empirical ablation results and scientific coverage across feature groups.

- Risk: script duplication across experiments.
  Control: centralize shared training and result-writing logic early.

## 9. Immediate Next Task

Start with the shared Tier 2 evaluation scaffold and the baseline parity check. That is the dependency that makes all four experiments comparable and keeps the branch from drifting into ad hoc one-off analyses.
