"""Record the fixed-split selected compact XGBoost model for the Phase 2 Tier 2."""

from __future__ import annotations

import json
import os
from typing import Any

from phase2_tier2_uncertainty import (
    COMPACT_CSV_PATH,
    TIER1_BENCHMARKS_PATH,
    TIER1_XGB_IMPORTANCE_PATH,
    TIER2_COMMON_PATH,
    fit_final_model,
    parse_feature_rows,
    read_csv_rows,
    read_python_assignment,
    select_best_model,
    split_count_summary,
    split_rows,
)


TIER2_RESULTS_DIR = "results/phase2_tier2"
SELECTED_MODEL_JSON = f"{TIER2_RESULTS_DIR}/selected_compact_model.json"
SELECTED_MODEL_MD = f"{TIER2_RESULTS_DIR}/selected_compact_model.md"


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def format_float(value: float) -> str:
    return f"{value:.6f}"


def write_markdown(path: str, payload: dict[str, Any]) -> None:
    selected = payload["selected_model"]
    test_metrics = payload["test_metrics"]
    validation_metrics = selected["validation_metrics"]
    split_summary = payload["split_summary"]
    params = selected["params"]

    lines = [
        "# Selected Compact Model",
        "",
        "This artifact records the exact fixed-split XGBoost candidate selected for the compact 16-feature model.",
        "",
        "## Protocol",
        f"- Seed: {payload['seed']}",
        f"- Test split: {payload['test_split']}",
        f"- Validation split within train+validation: {payload['validation_split']}",
        "- Selection metric: validation PR-AUC",
        "- Final fit: train+validation",
        "- Final evaluation: held-out test",
        "",
        "## Split counts",
    ]
    lines.extend(
        markdown_table(
            ["split", "total", "Ia", "non-Ia"],
            [
                [
                    split_name,
                    str(counts["total"]),
                    str(counts["ia_count"]),
                    str(counts["non_ia_count"]),
                ]
                for split_name, counts in split_summary.items()
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Selected XGBoost candidate",
            f"- Candidate: {selected['candidate']}",
            f"- Best iteration: {selected['best_iteration']}",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["parameter", "value"],
            [
                ["max_depth", str(params["max_depth"])],
                ["eta", str(params["eta"])],
                ["subsample", str(params["subsample"])],
                ["colsample_bytree", str(params["colsample_bytree"])],
                ["min_child_weight", str(params["min_child_weight"])],
                ["lambda", str(params["lambda"])],
            ],
        )
    )
    lines.extend(["", "## Validation metrics for selected candidate"])
    lines.extend(
        markdown_table(
            ["metric", "value"],
            [[name, format_float(value)] for name, value in validation_metrics.items()],
        )
    )
    lines.extend(["", "## Held-out test metrics"])
    lines.extend(
        markdown_table(
            ["metric", "value"],
            [[name, format_float(value)] for name, value in test_metrics.items()],
        )
    )
    lines.extend(
        [
            "",
            "## Compact features",
            ", ".join(payload["compact_features"]),
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_results_dir()
    seed = read_python_assignment(TIER1_BENCHMARKS_PATH, "RANDOM_STATE")
    test_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "TEST_SPLIT")
    validation_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "VALIDATION_SPLIT")
    compact_features = read_python_assignment(TIER2_COMMON_PATH, "COMPACT_FEATURES")
    baseline_metrics = read_python_assignment(TIER2_COMMON_PATH, "FROZEN_BASELINE_METRICS")
    param_grid = read_python_assignment(TIER1_XGB_IMPORTANCE_PATH, "XGB_PARAM_GRID")

    rows = parse_feature_rows(read_csv_rows(COMPACT_CSV_PATH))
    split_data = split_rows(rows, test_split=test_split, validation_split=validation_split, seed=seed)
    selected = select_best_model(
        split_data["train"],
        split_data["validation"],
        compact_features,
        param_grid,
        seed=seed,
        num_boost_round=400,
        early_stopping_rounds=30,
    )
    test_metrics = fit_final_model(
        split_data["trainval"],
        split_data["test"],
        compact_features,
        seed=seed,
        params=selected["params"],
        num_boost_round=selected["best_iteration"],
    )
    payload = {
        "artifact": "phase2_tier2_selected_compact_model",
        "csv_path": COMPACT_CSV_PATH,
        "seed": seed,
        "test_split": test_split,
        "validation_split": validation_split,
        "compact_feature_count": len(compact_features),
        "compact_features": compact_features,
        "selection_metric": "validation PR-AUC",
        "num_boost_round": 400,
        "early_stopping_rounds": 30,
        "selected_model": selected,
        "test_metrics": test_metrics,
        "frozen_baseline_metrics": baseline_metrics,
        "delta_from_frozen_baseline": {
            "f1": test_metrics["f1"] - baseline_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"] - baseline_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"] - baseline_metrics["pr_auc"],
        },
        "split_summary": split_count_summary(split_data),
        "notes": [
            "This is the fixed submitted split selected-model artifact.",
            "The model is selected on validation PR-AUC, refit on train+validation, and evaluated on the held-out test split.",
        ],
    }

    write_json(SELECTED_MODEL_JSON, payload)
    write_markdown(SELECTED_MODEL_MD, payload)
    print(
        f"Selected candidate {selected['candidate']} with best_iteration={selected['best_iteration']} "
        f"and test F1={test_metrics['f1']:.6f}, ROC-AUC={test_metrics['roc_auc']:.6f}, "
        f"PR-AUC={test_metrics['pr_auc']:.6f}"
    )
    print(f"Wrote {SELECTED_MODEL_JSON}")
    print(f"Wrote {SELECTED_MODEL_MD}")


if __name__ == "__main__":
    main()
