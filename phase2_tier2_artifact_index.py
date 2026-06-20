"""Create an index of generated Phase 2 Tier 2 artifacts."""

from __future__ import annotations

import csv
import os
from typing import Any


TIER2_RESULTS_DIR = "results/phase2_tier2"
ARTIFACT_INDEX_CSV = f"{TIER2_RESULTS_DIR}/artifact_index.csv"
ARTIFACT_INDEX_MD = f"{TIER2_RESULTS_DIR}/artifact_index.md"


ARTIFACTS = [
    {
        "artifact": "Dataset summary",
        "path": "results/phase2_tier2/dataset_summary.md",
        "machine_path": "results/phase2_tier2/dataset_summary.json",
        "type": "dataset",
        "analysis_purpose": "Dataset support: dataset description, class balance, split counts",
        "manuscript_support": "Data section; experimental setup; dataset-description support",
        "source_script": "phase2_tier2_metadata.py",
        "status": "generated",
    },
    {
        "artifact": "Class balance table",
        "path": "results/phase2_tier2/class_balance.csv",
        "machine_path": "",
        "type": "dataset",
        "analysis_purpose": "Dataset support: class balance and label distribution",
        "manuscript_support": "Dataset statistics table",
        "source_script": "phase2_tier2_metadata.py",
        "status": "generated",
    },
    {
        "artifact": "Split class balance table",
        "path": "results/phase2_tier2/split_class_balance.csv",
        "machine_path": "",
        "type": "dataset/protocol",
        "analysis_purpose": "Protocol support: split details and held-out test usage",
        "manuscript_support": "Train/validation/test protocol description",
        "source_script": "phase2_tier2_metadata.py",
        "status": "generated",
    },
    {
        "artifact": "Compact feature ranges",
        "path": "results/phase2_tier2/compact_feature_ranges.csv",
        "machine_path": "",
        "type": "dataset/features",
        "analysis_purpose": "Dataset support: engineered feature ranges",
        "manuscript_support": "Dataset and feature-description details",
        "source_script": "phase2_tier2_metadata.py",
        "status": "generated",
    },
    {
        "artifact": "Training protocol",
        "path": "results/phase2_tier2/training_protocol.md",
        "machine_path": "results/phase2_tier2/training_protocol.json",
        "type": "protocol",
        "analysis_purpose": "Protocol support: training details and hyperparameters",
        "manuscript_support": "Methods; model-training protocol; hyperparameter table",
        "source_script": "phase2_tier2_training_protocol.py",
        "status": "generated",
    },
    {
        "artifact": "XGBoost hyperparameter grid",
        "path": "results/phase2_tier2/xgb_hyperparameter_grid.csv",
        "machine_path": "",
        "type": "protocol",
        "analysis_purpose": "Protocol support: hyperparameters not reported",
        "manuscript_support": "Hyperparameter grid table",
        "source_script": "phase2_tier2_training_protocol.py",
        "status": "generated",
    },
    {
        "artifact": "Repeated-split uncertainty summary",
        "path": "results/phase2_tier2/uncertainty_summary.md",
        "machine_path": "results/phase2_tier2/uncertainty_summary.json",
        "type": "uncertainty",
        "analysis_purpose": "Uncertainty support: no uncertainty or cross-validation statistics",
        "manuscript_support": "Robustness/stability paragraph; uncertainty table",
        "source_script": "phase2_tier2_uncertainty.py",
        "status": "generated",
    },
    {
        "artifact": "Repeated-split uncertainty runs",
        "path": "results/phase2_tier2/uncertainty_runs.csv",
        "machine_path": "",
        "type": "uncertainty",
        "analysis_purpose": "Uncertainty support: per-seed stability evidence",
        "manuscript_support": "Supplementary repeated-split table",
        "source_script": "phase2_tier2_uncertainty.py",
        "status": "generated",
    },
    {
        "artifact": "Compact feature dictionary",
        "path": "results/phase2_tier2/compact_feature_dictionary.md",
        "machine_path": "results/phase2_tier2/compact_feature_dictionary.json",
        "type": "feature dictionary",
        "analysis_purpose": "Feature-definition support: full feature list, definitions, formulas, time_span clarification",
        "manuscript_support": "Feature-definition table; appendix; correction of rise/decline wording",
        "source_script": "phase2_tier2_feature_dictionary.py",
        "status": "generated",
    },
    {
        "artifact": "Compact feature dictionary CSV",
        "path": "results/phase2_tier2/compact_feature_dictionary.csv",
        "machine_path": "",
        "type": "feature dictionary",
        "analysis_purpose": "Feature-definition support: tabular feature definitions",
        "manuscript_support": "Feature table source",
        "source_script": "phase2_tier2_feature_dictionary.py",
        "status": "generated",
    },
    {
        "artifact": "Selected compact model",
        "path": "results/phase2_tier2/selected_compact_model.md",
        "machine_path": "results/phase2_tier2/selected_compact_model.json",
        "type": "selected model",
        "analysis_purpose": "Protocol support: selected hyperparameters and final test metrics",
        "manuscript_support": "Model-selection details; fixed-split baseline reproducibility",
        "source_script": "phase2_tier2_selected_model.py",
        "status": "generated",
    },
    {
        "artifact": "Tier 2 experiment audit",
        "path": "results/phase2_tier2/tier2_experiment_audit.md",
        "machine_path": "",
        "type": "ablation audit",
        "analysis_purpose": "Ablation support: ablation reproducibility and interpretation",
        "manuscript_support": "Ablation results; compactness claim; analysis summary",
        "source_script": "phase2_tier2_audit.py",
        "status": "generated",
    },
    {
        "artifact": "Tier 2 experiment audit CSV",
        "path": "results/phase2_tier2/tier2_experiment_audit.csv",
        "machine_path": "",
        "type": "ablation audit",
        "analysis_purpose": "Ablation support: machine-readable ablation summary",
        "manuscript_support": "Ablation table source",
        "source_script": "phase2_tier2_audit.py",
        "status": "generated",
    },
]


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def artifact_exists(path: str) -> str:
    if not path:
        return ""
    return "yes" if os.path.exists(path) else "missing"


def enriched_artifacts() -> list[dict[str, Any]]:
    rows = []
    for artifact in ARTIFACTS:
        row = dict(artifact)
        row["path_exists"] = artifact_exists(row["path"])
        row["machine_path_exists"] = artifact_exists(row["machine_path"])
        rows.append(row)
    return rows


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "artifact",
        "path",
        "path_exists",
        "machine_path",
        "machine_path_exists",
        "type",
        "analysis_purpose",
        "manuscript_support",
        "source_script",
        "status",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_markdown(path: str, rows: list[dict[str, Any]]) -> None:
    missing = [row for row in rows if row["path_exists"] != "yes"]
    lines = [
        "# Phase 2 Tier 2 Artifact Index",
        "",
        "This index maps generated code/supporting artifacts to analysis purposes and later manuscript sections.",
        "",
        f"- Indexed artifacts: {len(rows)}",
        f"- Missing primary artifacts: {len(missing)}",
        "",
        "## Artifact Map",
    ]
    lines.extend(
        markdown_table(
            ["artifact", "type", "path", "analysis purpose", "manuscript support", "script", "exists"],
            [
                [
                    row["artifact"],
                    row["type"],
                    f"`{row['path']}`",
                    row["analysis_purpose"],
                    row["manuscript_support"],
                    f"`{row['source_script']}`",
                    row["path_exists"],
                ]
                for row in rows
            ],
        )
    )
    if missing:
        lines.extend(["", "## Missing Artifacts"])
        lines.extend(f"- `{row['path']}` from `{row['source_script']}`" for row in missing)
    lines.extend(
        [
            "",
            "## Scope Note",
            "External-validation artifacts are intentionally excluded from this supporting artifact index because cross-survey generalization is handled in Phase 2 Tier 4.",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_results_dir()
    rows = enriched_artifacts()
    write_csv(ARTIFACT_INDEX_CSV, rows)
    write_markdown(ARTIFACT_INDEX_MD, rows)
    print(f"Wrote {ARTIFACT_INDEX_CSV}")
    print(f"Wrote {ARTIFACT_INDEX_MD}")


if __name__ == "__main__":
    main()
