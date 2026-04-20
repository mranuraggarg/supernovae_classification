# Phase 2 Tier 2 Code Checklist

This checklist is for code and results changes only. Manuscript-only edits are intentionally left out until the Phase 2 Tier 2 experiments and artifacts are stable.

## Worktree

- Revision worktree: `/Users/anuraggarg/work/supernovae_classification_phase2_tier2`
- Branch: `phase2-tier2-compact-feature-ablation`
- Base branch: `phase2-tier2-compact-feature-ablation`
- Base commit: `50a263e3`

The current Phase 2 Tier 4 work remains in `/Users/anuraggarg/work/supernovae_classification`.

## Reviewer-Driven Code Tasks

### 1. Dataset description and class balance

Reviewer items:

- Dataset support: dataset description insufficient.

Code action:

- Add a supporting metadata script that reads the Tier 1 processed feature CSV and writes dataset statistics to `results/phase2_tier2/`.
- Include total object count, Ia/non-Ia counts, class fractions, split counts, split class counts, feature count, compact feature list, and basic feature ranges.
- If magnitude-like or flux-derived ranges are needed for the text, compute them from the compact/Tier 1 feature table and label them clearly as engineered feature ranges.

Likely files:

- Add `phase2_tier2_metadata.py`.
- Reuse `phase2_tier2_common.create_context()`.

Status:

- Done in `phase2_tier2_metadata.py`.
- Generated `results/phase2_tier2/dataset_summary.md`.
- Generated `results/phase2_tier2/dataset_summary.json`.
- Generated `results/phase2_tier2/class_balance.csv`.
- Generated `results/phase2_tier2/split_class_balance.csv`.
- Generated `results/phase2_tier2/compact_feature_ranges.csv`.
- Generated `results/phase2_tier2/dataset_context_ranges.csv`.
- Implementation note: the metadata script is standalone and does not import XGBoost, so dataset reporting can run even before model dependencies are installed.

### 2. Model training details and hyperparameters

Reviewer items:

- Reviewer 1.7: missing model training details.
- Protocol support: missing hyperparameters.

Code action:

- Export the exact split protocol, random seed, validation selection metric, XGBoost parameter grid, selected parameter set, early-stopping setting, and final boosting rounds for the compact baseline and Tier 2 experiments.
- Ensure every Tier 2 output JSON includes enough training metadata to reproduce the run without reading source code.

Likely files:

- Update `phase2_tier2_common.py`.
- Add a summary artifact such as `results/phase2_tier2/training_protocol.md`.

Status:

- Partly done in `phase2_tier2_training_protocol.py`.
- Generated `results/phase2_tier2/training_protocol.md`.
- Generated `results/phase2_tier2/training_protocol.json`.
- Generated `results/phase2_tier2/xgb_hyperparameter_grid.csv`.
- The exported protocol includes the split seed, train/validation/test protocol, XGBoost base parameters, hyperparameter grid, early stopping rounds, selection metric, standardization rule, class weighting rule, and frozen compact baseline metrics.
- Remaining follow-up: regenerate Tier 2 JSON artifacts after XGBoost is available so the exact selected candidate and best iteration for each run can be reported directly rather than inferred from source protocol.

### 3. Test-set usage clarification

Reviewer items:

- Reviewer 1.8: why no test set used.

Code action:

- Verify and report the existing protocol: train and validation are used for selection, then train+validation is refit and evaluated once on the held-out test set.
- Add an explicit machine-readable split manifest and human-readable summary so the manuscript can state the test set was used.

Likely files:

- Update or extend `baseline_reference_payload()` in `phase2_tier2_common.py`.
- Add `results/phase2_tier2/split_manifest.md`.

### 4. Cross-validation or uncertainty statistics

Reviewer items:

- Uncertainty support: no uncertainty / CV statistics.

Code action:

- Add repeated-seed evaluation or stratified K-fold CV for the compact 16-feature baseline and the key reduced-core subsets.
- Report mean and standard deviation for F1, ROC-AUC, PR-AUC, precision, and recall.
- Keep the original fixed held-out test result as the main result, and use repeated/CV results as stability evidence.

Likely files:

- Add `phase2_tier2_uncertainty.py`.
- Reuse `evaluate_feature_subset()` where possible, but avoid leaking the fixed test set into model selection.

Status:

- Implemented in `phase2_tier2_uncertainty.py`.
- The script runs repeated stratified train/validation/test splits for the compact 16-feature XGBoost baseline.
- Default seeds: 11, 22, 33, 44, 55.
- For each seed, the script selects the XGBoost candidate by validation PR-AUC, refits on train+validation, evaluates on the held-out test split, and records the selected candidate and best iteration.
- Intended outputs: `results/phase2_tier2/uncertainty_runs.csv`, `results/phase2_tier2/uncertainty_summary.json`, and `results/phase2_tier2/uncertainty_summary.md`.
- Syntax check passed with `PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile phase2_tier2_uncertainty.py`.
- Executed successfully by the user in the `astro-ml` conda environment.
- Generated `results/phase2_tier2/uncertainty_runs.csv`.
- Generated `results/phase2_tier2/uncertainty_summary.json`.
- Generated `results/phase2_tier2/uncertainty_summary.md`.
- The script now prints the repeated-split uncertainty summary to stdout after writing the artifacts.

### 5. Feature definitions and formulas

Reviewer items:

- Reviewer 2.6: feature definitions / formulas missing.
- Reviewer 1.10: missing full feature list.
- Reviewer 1.13: feature mismatch, rise/decline vs `time_span`.

Code action:

- Export a compact feature dictionary with feature name, group, definition, and formula/derivation.
- Explicitly define `time_span` as observation-time coverage, not a rise or decline time.
- Include all 16 compact features used by the submitted model.

Likely files:

- Add `phase2_tier2_feature_dictionary.py`, or add feature metadata constants to `phase2_tier2_common.py`.
- Output `results/phase2_tier2/compact_feature_dictionary.csv` and `.md`.

Status:

- Implemented in `phase2_tier2_feature_dictionary.py`.
- The script joins the compact feature list and feature groups from `phase2_tier2_common.py`, formal definitions from `feature_pipeline/extraction/feature_registry.py`, and physical interpretation fields from `results/phase2_tier1/phase2_tier1_compact_baseline_interpretation_table.csv`.
- Intended outputs: `results/phase2_tier2/compact_feature_dictionary.csv`, `results/phase2_tier2/compact_feature_dictionary.json`, and `results/phase2_tier2/compact_feature_dictionary.md`.
- Syntax check passed with `PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile phase2_tier2_feature_dictionary.py`.
- Execution left to the user in the `astro-ml` conda environment by request.

### 6. Real-data validation, optional but useful

Reviewer items:

- Reviewer 2.4: no real-data validation.

Code action:

- Treat this as optional unless we can do it cleanly without changing the paper's scope.
- If using local SNE/PLAsTiCC data, keep it as an external validation or feasibility check with clear caveats about survey/domain differences.
- Do not merge Phase 2 Tier 4 logic by default; cherry-pick only a small, analysis-relevant validation script if needed.

Likely files:

- Prefer a new isolated script such as `phase2_tier2_external_validation.py`.
- Use `results/phase2_tier2/external_validation_*` artifacts.

### 7. Fixed-split selected-model reproducibility

Reviewer items:

- Reviewer 1.7: missing model training details.
- Protocol support: hyperparameters not reported.

Code action:

- Record the exact XGBoost candidate selected on the submitted fixed split.
- Include selected hyperparameters, validation metrics, best iteration, final held-out test metrics, split counts, and compact feature list.
- Keep this as a reproducibility artifact under `results/phase2_tier2/`.

Likely files:

- Add `phase2_tier2_selected_model.py`.
- Output `results/phase2_tier2/selected_compact_model.json` and `results/phase2_tier2/selected_compact_model.md`.

Status:

- Implemented in `phase2_tier2_selected_model.py`.
- Syntax check passed with `PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile phase2_tier2_selected_model.py`.
- Execution left to the user in the `astro-ml` conda environment.

### 8. Tier 2 experiment audit

Reviewer items:

- Reviewer 1.7: missing model training details.
- Uncertainty support: uncertainty and reproducibility context.
- Reviewer 2.6: feature-ablation interpretation clarity.

Code action:

- Summarize regenerated Tier 2 JSON outputs in tracked supporting Markdown/CSV artifacts.
- Include baseline parity checks, feature-ablation highlights, block-ablation highlights, subset-growth results, and minimal-core results.

Likely files:

- Add `phase2_tier2_audit.py`.
- Output `results/phase2_tier2/tier2_experiment_audit.md` and `results/phase2_tier2/tier2_experiment_audit.csv`.

Status:

- Implemented in `phase2_tier2_audit.py`.
- Generated `results/phase2_tier2/tier2_experiment_audit.md`.
- Generated `results/phase2_tier2/tier2_experiment_audit.csv`.
- The audit confirms baseline parity to numerical precision and identifies temporal features/time-span as the strongest ablation signals while preserving the wording caution that `time_span` is observational coverage.

### 9. Revision artifact index

Reviewer items:

- Supports the code-to-writing transition for all code/experiment analysis purposes.

Code action:

- Create a single index mapping generated artifacts to analysis purposes and manuscript sections.
- Explicitly mark external validation as out of scope for this Phase 2 Tier 2 work because it belongs to Phase 2 Tier 4.

Likely files:

- Add `phase2_tier2_artifact_index.py`.
- Output `results/phase2_tier2/artifact_index.md` and `results/phase2_tier2/artifact_index.csv`.

Status:

- Implemented in `phase2_tier2_artifact_index.py`.
- Generated `results/phase2_tier2/artifact_index.md`.
- Generated `results/phase2_tier2/artifact_index.csv`.
- The index records 13 primary artifacts with no missing primary or machine-readable companion artifacts.

### 10. Final code-package housekeeping

Code action:

- Verify new supporting scripts compile.
- Verify the artifact index has no missing files.
- Ensure generated Phase 2 Tier 2 JSON evidence is visible to Git without unignoring unrelated JSON outputs.

Status:

- Syntax check passed for all new `phase2_tier2_*.py` scripts.
- `results/phase2_tier2/artifact_index.csv` reports 13 artifacts, 0 missing primary artifacts, and 0 missing machine-readable companion artifacts.
- `.gitignore` now explicitly allows `results/phase2_tier2/*.json` while leaving regenerated `results/phase2_tier2/*.json` ignored.
- Trackable Phase 2 Tier 2 support package currently consists of the checklist, 7 supporting scripts, 21 files under `results/phase2_tier2/`, and the narrow `.gitignore` exception.

### 11. Learning and validation-logloss curves

Reviewer items:

- Reviewer 1.7: missing model training details.
- Uncertainty support: model stability and overfitting concern.
- Protocol support: early-stopping/logloss training details.

Code action:

- Generate a compact-model learning curve using increasing stratified training subsets.
- Generate a training/validation logloss curve over XGBoost boosting rounds.
- Keep the curves classical and XGBoost-specific; do not add epoch-style deep-learning plots.

Likely files:

- Add `phase2_tier2_learning_curves.py`.
- Output `results/phase2_tier2/learning_curve.{csv,json,md,png,pdf}`.
- Output `results/phase2_tier2/validation_logloss_curve.{csv,json,md,png,pdf}`.

Status:

- Implemented in `phase2_tier2_learning_curves.py`.
- Syntax check passed with `PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile phase2_tier2_learning_curves.py`.
- Execution left to the user in the `astro-ml` conda environment.

## Recommended Execution Order

1. Add metadata and split reporting.
2. Add feature dictionary export.
3. Add training protocol export.
4. Add uncertainty/repeated-seed evaluation.
5. Add fixed-split selected-model reproducibility artifact.
6. Decide whether external validation is scientifically clean enough for this Phase 2 Tier 2 package.
7. Rerun Tier 2 scripts and collect final `results/phase2_tier2/` artifacts.

## Guardrails

- Do not edit the Phase 2 Tier 4 worktree for this Phase 2 Tier 2 package.
- Do not overwrite existing `results/phase2_tier2/` artifacts until the supporting scripts are stable.
- Keep supporting artifacts under `results/phase2_tier2/` and `plots/phase2_tier2/`.
- Do not add copyrighted or large external data to Git.
- Avoid importing broader Tier 3/Tier 4 claims unless they directly answer a Phase 2 Tier 2 scope.
