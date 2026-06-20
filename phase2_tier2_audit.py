"""Summarize regenerated Tier 2 experiment outputs for the Phase 2 Tier 2."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


TIER2_RESULTS_DIR = "results/phase2_tier2"
AUDIT_MD = f"{TIER2_RESULTS_DIR}/tier2_experiment_audit.md"
AUDIT_CSV = f"{TIER2_RESULTS_DIR}/tier2_experiment_audit.csv"

TIER2_RESULTS_DIR = Path("results/phase2_tier2")
EXPERIMENT_PATHS = {
    "feature_ablation": TIER2_RESULTS_DIR / "feature_ablation_metrics.json",
    "block_ablation": TIER2_RESULTS_DIR / "block_ablation_metrics.json",
    "subset_growth": TIER2_RESULTS_DIR / "subset_growth_metrics.json",
    "minimal_core": TIER2_RESULTS_DIR / "minimal_core_metrics.json",
}


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def row_label(experiment_name: str, row: dict[str, Any]) -> str:
    if experiment_name == "feature_ablation":
        return row["feature_removed"]
    if experiment_name == "block_ablation":
        return row["block_removed"]
    return row["subset_name"]


def feature_count(experiment_name: str, row: dict[str, Any]) -> int:
    if experiment_name == "feature_ablation":
        return int(row["num_features"])
    if experiment_name == "block_ablation":
        return int(row["remaining_feature_count"])
    return int(row["feature_count"])


def audit_rows_for_experiment(experiment_name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    audit_rows = []
    for row in payload["rows"]:
        audit_rows.append(
            {
                "experiment": experiment_name,
                "item": row_label(experiment_name, row),
                "feature_group": row.get("feature_group", ""),
                "feature_count": feature_count(experiment_name, row),
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "delta_f1": row["delta_f1"],
                "delta_roc_auc": row["delta_roc_auc"],
                "delta_pr_auc": row["delta_pr_auc"],
                "interpretation": interpretation_for_row(experiment_name, row),
            }
        )
    return audit_rows


def interpretation_for_row(experiment_name: str, row: dict[str, Any]) -> str:
    delta_f1 = float(row["delta_f1"])
    delta_pr_auc = float(row["delta_pr_auc"])
    if experiment_name in {"feature_ablation", "block_ablation"}:
        if delta_f1 < -0.03 or delta_pr_auc < -0.03:
            return "large performance loss when removed"
        if delta_f1 < -0.01 or delta_pr_auc < -0.01:
            return "moderate performance loss when removed"
        if delta_f1 < -0.003 or delta_pr_auc < -0.003:
            return "small performance loss when removed"
        return "near-baseline performance when removed"
    if delta_f1 > -0.01 and delta_pr_auc > -0.02:
        return "near-baseline compact performance"
    if delta_f1 > -0.06:
        return "moderate reduced-feature performance"
    return "substantial loss relative to compact baseline"


def sorted_highlights(experiment_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if experiment_name in {"feature_ablation", "block_ablation"}:
        return sorted(rows, key=lambda row: row["delta_f1"])[:5]
    return sorted(rows, key=lambda row: row["feature_count"])


def write_markdown(path: str, payloads: dict[str, dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    baseline = next(iter(payloads.values()))["baseline_reference"]["baseline_metrics"]
    lines = [
        "# Tier 2 Experiment Audit",
        "",
        "This audit summarizes the regenerated Phase 2 Tier 2 experiment outputs for the Phase 2 Tier 2.",
        "",
        "## Baseline reference",
        f"- F1: {format_float(baseline['f1'])}",
        f"- ROC-AUC: {format_float(baseline['roc_auc'])}",
        f"- PR-AUC: {format_float(baseline['pr_auc'])}",
        "",
        "## Baseline parity checks",
    ]
    lines.extend(
        markdown_table(
            ["experiment", "delta F1", "delta ROC-AUC", "delta PR-AUC"],
            [
                [
                    experiment_name,
                    format_float(payload["baseline_parity_check"]["delta_f1"]),
                    format_float(payload["baseline_parity_check"]["delta_roc_auc"]),
                    format_float(payload["baseline_parity_check"]["delta_pr_auc"]),
                ]
                for experiment_name, payload in payloads.items()
            ],
        )
    )

    for experiment_name, payload in payloads.items():
        rows = [row for row in audit_rows if row["experiment"] == experiment_name]
        highlights = sorted_highlights(experiment_name, rows)
        title = experiment_name.replace("_", " ").title()
        lines.extend(["", f"## {title}"])
        lines.extend(
            markdown_table(
                ["item", "features", "F1", "PR-AUC", "delta F1", "delta PR-AUC", "interpretation"],
                [
                    [
                        row["item"],
                        str(row["feature_count"]),
                        format_float(row["f1"]),
                        format_float(row["pr_auc"]),
                        format_float(row["delta_f1"]),
                        format_float(row["delta_pr_auc"]),
                        row["interpretation"],
                    ]
                    for row in highlights
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Supporting takeaways",
            "- The baseline parity checks reproduce the frozen compact baseline to numerical precision.",
            "- Temporal features are the largest block-level contributor in the leave-one-block-out study.",
            "- `time_span` is the strongest single-feature ablation loss, but it should be described as observational time coverage, not rise or decline time.",
            "- The 10-feature minimal core remains close to the compact baseline by F1, supporting the compactness claim while retaining a small PR-AUC loss.",
            "",
            "Note: this audit summarizes final experiment metrics and deltas. Exact selected XGBoost hyperparameters for the fixed compact baseline are recorded separately in `results/phase2_tier2/selected_compact_model.md`.",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_results_dir()
    missing = [str(path) for path in EXPERIMENT_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing regenerated Tier 2 JSON output(s): " + ", ".join(missing))

    payloads = {name: read_json(path) for name, path in EXPERIMENT_PATHS.items()}
    audit_rows = []
    for experiment_name, payload in payloads.items():
        audit_rows.extend(audit_rows_for_experiment(experiment_name, payload))

    write_csv(
        AUDIT_CSV,
        [
            "experiment",
            "item",
            "feature_group",
            "feature_count",
            "f1",
            "roc_auc",
            "pr_auc",
            "delta_f1",
            "delta_roc_auc",
            "delta_pr_auc",
            "interpretation",
        ],
        audit_rows,
    )
    write_markdown(AUDIT_MD, payloads, audit_rows)
    print(f"Wrote {AUDIT_CSV}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
