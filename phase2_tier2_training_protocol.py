"""Export supporting training protocol and hyperparameter details."""

from __future__ import annotations

import ast
import csv
import json
import os
from typing import Any


TIER2_RESULTS_DIR = "results/phase2_tier2"
TRAINING_PROTOCOL_JSON = f"{TIER2_RESULTS_DIR}/training_protocol.json"
TRAINING_PROTOCOL_MD = f"{TIER2_RESULTS_DIR}/training_protocol.md"
XGB_GRID_CSV = f"{TIER2_RESULTS_DIR}/xgb_hyperparameter_grid.csv"

TIER1_BENCHMARKS_PATH = "phase2_tier1_benchmarks.py"
TIER1_XGB_IMPORTANCE_PATH = "phase2_tier1_xgb_importance.py"
TIER2_COMMON_PATH = "phase2_tier2_common.py"
DATASET_SUMMARY_JSON = f"{TIER2_RESULTS_DIR}/dataset_summary.json"


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def read_python_assignment(path: str, name: str) -> Any:
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"Could not find assignment {name!r} in {path}.")


def read_json(path: str) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parameter_grid_rows(param_grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, params in enumerate(param_grid, start=1):
        rows.append(
            {
                "candidate": index,
                "max_depth": params["max_depth"],
                "eta": params["eta"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "min_child_weight": params["min_child_weight"],
                "lambda": params["lambda"],
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def format_float(value: float) -> str:
    return f"{value:.6g}"


def write_markdown(path: str, payload: dict[str, Any]) -> None:
    split_protocol = payload["split_protocol"]
    split_balance = payload["split_balance"]
    model_protocol = payload["model_protocol"]
    grid_rows = payload["xgboost_hyperparameter_grid"]
    baseline_metrics = payload["frozen_compact_baseline_metrics"]

    lines = [
        "# Phase 2 Tier 2 Training Protocol",
        "",
        "This artifact summarizes the model-training protocol used for the submitted Phase 2 Tier 2 compact-feature experiments.",
        "",
        "## Data split",
        f"- Random seed: {split_protocol['random_state']}",
        f"- Held-out test fraction: {format_float(split_protocol['test_split'])}",
        f"- Validation fraction within the train+validation pool: {format_float(split_protocol['validation_split_within_trainval'])}",
        "- The train and validation partitions are used for model selection.",
        "- The final model is refit on train+validation and evaluated once on the held-out test partition.",
        "",
        "## Split counts",
    ]
    lines.extend(
        markdown_table(
            ["split", "total", "Ia", "non-Ia", "Ia fraction"],
            [
                [
                    row["split"],
                    str(row["total"]),
                    str(row["ia_count"]),
                    str(row["non_ia_count"]),
                    format_float(row["ia_fraction"]),
                ]
                for row in split_balance
            ],
        )
    )
    lines.extend(
        [
            "",
            "## XGBoost protocol",
            f"- Objective: `{model_protocol['base_xgb_params']['objective']}`",
            f"- Evaluation metric during boosting: `{model_protocol['base_xgb_params']['eval_metric']}`",
            f"- Tree method: `{model_protocol['base_xgb_params']['tree_method']}`",
            f"- Maximum boosting rounds during model selection: {model_protocol['num_boost_round']}",
            f"- Early stopping rounds: {model_protocol['early_stopping_rounds']}",
            f"- Candidate-selection metric: {model_protocol['selection_metric']}",
            f"- Final training rounds: selected candidate's best iteration from validation early stopping.",
            f"- Class imbalance handling: {model_protocol['class_weighting']}",
            f"- Standardization: {model_protocol['standardization']}",
            "",
            "## Hyperparameter grid",
        ]
    )
    lines.extend(
        markdown_table(
            ["candidate", "max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight", "lambda"],
            [
                [
                    str(row["candidate"]),
                    str(row["max_depth"]),
                    str(row["eta"]),
                    str(row["subsample"]),
                    str(row["colsample_bytree"]),
                    str(row["min_child_weight"]),
                    str(row["lambda"]),
                ]
                for row in grid_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Frozen compact baseline test metrics",
        ]
    )
    lines.extend(
        markdown_table(
            ["metric", "value"],
            [
                ["F1", format_float(baseline_metrics["f1"])],
                ["ROC-AUC", format_float(baseline_metrics["roc_auc"])],
                ["PR-AUC", format_float(baseline_metrics["pr_auc"])],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Reproducibility note",
            "The checked-in Tier 2 CSV/Markdown artifacts contain final metrics but not every selected per-run hyperparameter choice.",
            "The source protocol above is exact; per-run selected candidate parameters and best iterations should be regenerated after installing XGBoost and rerunning the Tier 2 scripts.",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_results_dir()
    random_state = read_python_assignment(TIER1_BENCHMARKS_PATH, "RANDOM_STATE")
    validation_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "VALIDATION_SPLIT")
    test_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "TEST_SPLIT")
    xgb_param_grid = read_python_assignment(TIER1_XGB_IMPORTANCE_PATH, "XGB_PARAM_GRID")
    compact_features = read_python_assignment(TIER2_COMMON_PATH, "COMPACT_FEATURES")
    frozen_baseline_metrics = read_python_assignment(TIER2_COMMON_PATH, "FROZEN_BASELINE_METRICS")

    dataset_summary = read_json(DATASET_SUMMARY_JSON) if os.path.exists(DATASET_SUMMARY_JSON) else {}
    split_balance = dataset_summary.get("split_balance", [])

    payload = {
        "artifact": "phase2_tier2_training_protocol",
        "source_files": {
            "tier1_benchmarks": TIER1_BENCHMARKS_PATH,
            "tier1_xgb_importance": TIER1_XGB_IMPORTANCE_PATH,
            "tier2_common": TIER2_COMMON_PATH,
        },
        "compact_feature_count": len(compact_features),
        "compact_features": compact_features,
        "split_protocol": {
            "random_state": random_state,
            "test_split": test_split,
            "validation_split_within_trainval": validation_split,
            "split_type": "stratified by binary Ia/non-Ia label",
            "model_selection_partitions": ["train", "validation"],
            "final_fit_partition": "trainval",
            "final_evaluation_partition": "test",
        },
        "split_balance": split_balance,
        "model_protocol": {
            "model": "XGBoost binary logistic classifier",
            "base_xgb_params": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "verbosity": 0,
                "seed": random_state,
                "scale_pos_weight": "negative_count / positive_count, computed from the current training partition",
            },
            "num_boost_round": 400,
            "early_stopping_rounds": 30,
            "selection_metric": "validation PR-AUC",
            "threshold_for_class_metrics": 0.5,
            "standardization": "feature-wise z-score using the training partition mean and standard deviation; the same transform is applied to validation/test data",
            "class_weighting": "scale_pos_weight is computed as non-Ia count divided by Ia count in the current training partition",
            "final_refit": "after validation selection, refit on train+validation using the selected parameter set and best iteration, then evaluate on held-out test",
        },
        "xgboost_hyperparameter_grid": parameter_grid_rows(xgb_param_grid),
        "frozen_compact_baseline_metrics": frozen_baseline_metrics,
        "notes": [
            "This export reads source constants without importing XGBoost.",
            "Per-run selected parameters and best iterations are not present in checked-in Tier 2 CSV/Markdown outputs and should be regenerated with the Tier 2 scripts after XGBoost is installed.",
        ],
    }

    write_json(TRAINING_PROTOCOL_JSON, payload)
    write_csv(
        XGB_GRID_CSV,
        ["candidate", "max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight", "lambda"],
        payload["xgboost_hyperparameter_grid"],
    )
    write_markdown(TRAINING_PROTOCOL_MD, payload)

    print(f"Wrote {TRAINING_PROTOCOL_JSON}")
    print(f"Wrote {XGB_GRID_CSV}")
    print(f"Wrote {TRAINING_PROTOCOL_MD}")


if __name__ == "__main__":
    main()
