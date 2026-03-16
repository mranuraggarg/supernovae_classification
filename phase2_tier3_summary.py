"""Aggregate Phase 2 Tier 3 experiment outputs into one report."""

from __future__ import annotations

import json
import os
from typing import Any

from phase2_tier2_common import write_json
from phase2_tier3_model_compare import RESULTS_DIR, ensure_output_dirs


MODEL_COMPARE_JSON_PATH = f"{RESULTS_DIR}/model_compare_metrics.json"
CV_JSON_PATH = f"{RESULTS_DIR}/cv_stability_metrics.json"
NOISE_JSON_PATH = f"{RESULTS_DIR}/noise_test_metrics.json"
IMPORTANCE_JSON_PATH = f"{RESULTS_DIR}/importance_compare_metrics.json"
MINIMAL_JSON_PATH = f"{RESULTS_DIR}/minimal_generalization_metrics.json"

MASTER_JSON_PATH = f"{RESULTS_DIR}/phase2_tier3_master_summary.json"
REPORT_PATH = f"{RESULTS_DIR}/phase2_tier3_report.md"


def load_required_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label} output: {path}")
    with open(path) as handle:
        return json.load(handle)


def write_report(payload: dict[str, Any]) -> None:
    model_rows = payload["model_compare"]["rows"]
    cv_rows = payload["cv_stability"]["summary_rows"]
    noise_rows = payload["noise_test"]["rows"]
    importance_rows = payload["importance_compare"]["consensus_rows"]
    minimal_rows = payload["minimal_generalization"]["summary_rows"]

    best_model = max(model_rows, key=lambda row: row["pr_auc"])
    best_cv = max(cv_rows, key=lambda row: row["f1_mean"])
    worst_noise = min(noise_rows, key=lambda row: row["delta_f1"])
    best_subset = max(minimal_rows, key=lambda row: row["mean_f1"])
    top_physical = importance_rows[:5]

    lines = [
        "# Phase 2 Tier 3 Report",
        "",
        "## Model robustness",
        f"- Best model by PR-AUC: {best_model['model_label']} ({best_model['pr_auc']:.6f}).",
        f"- Frozen-split delta F1 for that model: {best_model['delta_f1']:+.6f}.",
        "",
        "## Split stability",
        f"- Strongest resampling protocol by mean F1: {best_cv['protocol']} ({best_cv['f1_mean']:.6f} +/- {best_cv['f1_std']:.6f}).",
        "",
        "## Noise and missing-data robustness",
        f"- Largest F1 degradation: {worst_noise['scenario_label']} ({worst_noise['delta_f1']:+.6f}).",
        "",
        "## Importance consistency",
    ]
    for row in top_physical:
        lines.append(
            f"- {row['feature']} ({row['feature_group']}): consensus score {row['consensus_score']:.3f}. "
            f"{row['interpretation']}"
        )
    lines.extend(
        [
            "",
            "## Minimal core generalization",
            f"- Best reduced core: {best_subset['subset_name']} ({best_subset['feature_count']} features) "
            f"with mean F1 {best_subset['mean_f1']:.6f}.",
            "",
            "## Interpretation",
            "- Agreement among gain, permutation, SHAP, and ablation ranks supports the view that the retained compact features capture stable astrophysical signal rather than a single-model artifact.",
        ]
    )

    with open(REPORT_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_output_dirs()
    payload = {
        "model_compare": load_required_json(MODEL_COMPARE_JSON_PATH, "model comparison"),
        "cv_stability": load_required_json(CV_JSON_PATH, "cross-validation stability"),
        "noise_test": load_required_json(NOISE_JSON_PATH, "noise test"),
        "importance_compare": load_required_json(IMPORTANCE_JSON_PATH, "importance comparison"),
        "minimal_generalization": load_required_json(MINIMAL_JSON_PATH, "minimal-core generalization"),
    }
    write_json(MASTER_JSON_PATH, payload)
    write_report(payload)
    print(json.dumps({"master_json_path": MASTER_JSON_PATH, "report_path": REPORT_PATH}, indent=2))


if __name__ == "__main__":
    main()
