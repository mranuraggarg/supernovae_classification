"""Repeated-split uncertainty evaluation for the Phase 2 Tier 2.

This script evaluates the compact 16-feature XGBoost baseline across repeated
stratified train/validation/test splits. It preserves the submitted protocol:
select candidates on validation PR-AUC, refit on train+validation, and evaluate
once on the held-out test split for each seed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from typing import Any

import numpy as np
import xgboost as xgb


TIER2_RESULTS_DIR = "results/phase2_tier2"
RUNS_CSV = f"{TIER2_RESULTS_DIR}/uncertainty_runs.csv"
SUMMARY_JSON = f"{TIER2_RESULTS_DIR}/uncertainty_summary.json"
SUMMARY_MD = f"{TIER2_RESULTS_DIR}/uncertainty_summary.md"

TIER1_BENCHMARKS_PATH = "phase2_tier1_benchmarks.py"
TIER1_XGB_IMPORTANCE_PATH = "phase2_tier1_xgb_importance.py"
TIER2_COMMON_PATH = "phase2_tier2_common.py"
COMPACT_CSV_PATH = "data/processed/phase2_tier1_compact_baseline.csv"

DEFAULT_SEEDS = [11, 22, 33, 44, 55]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


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


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_feature_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key == "label_name":
                parsed[key] = value
                continue
            if value in (None, ""):
                parsed[key] = value
                continue
            try:
                parsed[key] = int(value)
                continue
            except ValueError:
                pass
            try:
                parsed[key] = float(value)
                continue
            except ValueError:
                parsed[key] = value
        parsed_rows.append(parsed)
    return parsed_rows


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def stratified_split_indices(labels: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        test_count = int(round(len(shuffled) * test_size))
        test_count = min(max(test_count, 1), len(shuffled) - 1)
        test_indices.extend(shuffled[:test_count])
        train_indices.extend(shuffled[test_count:])
    return np.array(sorted(train_indices)), np.array(sorted(test_indices))


def split_rows(
    rows: list[dict[str, Any]],
    *,
    test_split: float,
    validation_split: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    trainval_idx, test_idx = stratified_split_indices(labels, test_split, seed)
    train_idx, validation_idx = stratified_split_indices(labels[trainval_idx], validation_split, seed)
    return {
        "train": [rows[trainval_idx[index]] for index in train_idx],
        "validation": [rows[trainval_idx[index]] for index in validation_idx],
        "trainval": [rows[index] for index in trainval_idx],
        "test": [rows[index] for index in test_idx],
    }


def build_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    return x, y


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std


def roc_auc_score_numpy(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(pos) * len(neg)))


def average_precision_numpy(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted == 1)
    fp_cumsum = np.cumsum(y_sorted == 0)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
    positive_total = max(int(np.sum(y_true == 1)), 1)
    recall = tp_cumsum / positive_total

    ap = 0.0
    previous_recall = 0.0
    for precision_value, recall_value, label in zip(precision, recall, y_sorted):
        if label == 1:
            ap += precision_value * (recall_value - previous_recall)
            previous_recall = recall_value
    return float(ap)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_numpy(y_true, probs),
        "pr_auc": average_precision_numpy(y_true, probs),
    }


def base_xgb_params(seed: int, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": seed,
        "scale_pos_weight": scale_pos_weight,
    }


def select_best_model(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_names: list[str],
    param_grid: list[dict[str, Any]],
    *,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> dict[str, Any]:
    x_train_raw, y_train = build_matrix(train_rows, feature_names)
    x_validation_raw, y_validation = build_matrix(validation_rows, feature_names)
    x_train, x_validation = standardize(x_train_raw, x_validation_raw)

    positive_count = float(np.sum(y_train == 1))
    negative_count = float(np.sum(y_train == 0))
    scale_pos_weight = negative_count / max(positive_count, 1.0)
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dvalidation = xgb.DMatrix(x_validation, label=y_validation, feature_names=feature_names)

    candidates = []
    for candidate_index, params in enumerate(param_grid, start=1):
        booster = xgb.train(
            params={**base_xgb_params(seed, scale_pos_weight), **params},
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalidation, "validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        probs = booster.predict(dvalidation, iteration_range=(0, booster.best_iteration + 1))
        metrics = binary_metrics(y_validation, probs)
        candidates.append(
            {
                "candidate": candidate_index,
                "params": params,
                "best_iteration": int(booster.best_iteration + 1),
                "validation_metrics": metrics,
            }
        )

    return max(candidates, key=lambda candidate: candidate["validation_metrics"]["pr_auc"])


def fit_final_model(
    trainval_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    seed: int,
    params: dict[str, Any],
    num_boost_round: int,
) -> dict[str, float]:
    x_trainval_raw, y_trainval = build_matrix(trainval_rows, feature_names)
    x_test_raw, y_test = build_matrix(test_rows, feature_names)
    x_trainval, x_test = standardize(x_trainval_raw, x_test_raw)

    positive_count = float(np.sum(y_trainval == 1))
    negative_count = float(np.sum(y_trainval == 0))
    scale_pos_weight = negative_count / max(positive_count, 1.0)
    dtrainval = xgb.DMatrix(x_trainval, label=y_trainval, feature_names=feature_names)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
    booster = xgb.train(
        params={**base_xgb_params(seed, scale_pos_weight), **params},
        dtrain=dtrainval,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    probs = booster.predict(dtest)
    return binary_metrics(y_test, probs)


def summarize_metrics(run_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric_name in METRIC_NAMES:
        values = np.array([float(row[metric_name]) for row in run_rows], dtype=np.float64)
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def split_count_summary(split_data: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    summary = {}
    for split_name, split_rows_value in split_data.items():
        ia_count = sum(row["label_name"] == "Ia" for row in split_rows_value)
        total = len(split_rows_value)
        summary[split_name] = {
            "total": total,
            "ia_count": ia_count,
            "non_ia_count": total - ia_count,
        }
    return summary


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


def write_markdown_summary(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 Tier 2 Uncertainty Evaluation",
        "",
        "Repeated stratified train/validation/test splits for the compact 16-feature XGBoost baseline.",
        "",
        "## Protocol",
        f"- Seeds: {', '.join(str(seed) for seed in payload['seeds'])}",
        f"- Test split: {payload['protocol']['test_split']}",
        f"- Validation split within train+validation: {payload['protocol']['validation_split']}",
        f"- Candidate-selection metric: {payload['protocol']['selection_metric']}",
        f"- Maximum boosting rounds: {payload['protocol']['num_boost_round']}",
        f"- Early stopping rounds: {payload['protocol']['early_stopping_rounds']}",
        "",
        "## Test metric summary",
    ]
    metric_summary = payload["metric_summary"]
    lines.extend(
        markdown_table(
            ["metric", "mean", "std", "min", "max"],
            [
                [
                    metric_name,
                    format_float(metric_summary[metric_name]["mean"]),
                    format_float(metric_summary[metric_name]["std"]),
                    format_float(metric_summary[metric_name]["min"]),
                    format_float(metric_summary[metric_name]["max"]),
                ]
                for metric_name in METRIC_NAMES
            ],
        )
    )
    lines.extend(["", "## Per-seed test metrics"])
    lines.extend(
        markdown_table(
            ["seed", "selected_candidate", "best_iteration", "F1", "ROC-AUC", "PR-AUC", "precision", "recall"],
            [
                [
                    str(row["seed"]),
                    str(row["selected_candidate"]),
                    str(row["best_iteration"]),
                    format_float(row["f1"]),
                    format_float(row["roc_auc"]),
                    format_float(row["pr_auc"]),
                    format_float(row["precision"]),
                    format_float(row["recall"]),
                ]
                for row in payload["runs"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "The submitted fixed-split result should remain the primary result; these repeated-split values quantify stability under alternate stratified splits.",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def print_metric_summary(payload: dict[str, Any]) -> None:
    print("\nRepeated-split uncertainty summary")
    print("metric, mean, std, min, max")
    for metric_name in METRIC_NAMES:
        metric = payload["metric_summary"][metric_name]
        print(
            f"{metric_name}, "
            f"{metric['mean']:.6f}, "
            f"{metric['std']:.6f}, "
            f"{metric['min']:.6f}, "
            f"{metric['max']:.6f}"
        )


def run_uncertainty(args: argparse.Namespace) -> dict[str, Any]:
    ensure_results_dir()
    rows = parse_feature_rows(read_csv_rows(args.csv_path))
    compact_features = read_python_assignment(TIER2_COMMON_PATH, "COMPACT_FEATURES")
    test_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "TEST_SPLIT")
    validation_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "VALIDATION_SPLIT")
    param_grid = read_python_assignment(TIER1_XGB_IMPORTANCE_PATH, "XGB_PARAM_GRID")

    missing_features = [feature for feature in compact_features if feature not in rows[0]]
    if missing_features:
        raise KeyError("Missing compact features: " + ", ".join(missing_features))

    run_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for seed in args.seeds:
        split_data = split_rows(rows, test_split=test_split, validation_split=validation_split, seed=seed)
        split_summaries[str(seed)] = split_count_summary(split_data)
        selection = select_best_model(
            split_data["train"],
            split_data["validation"],
            compact_features,
            param_grid,
            seed=seed,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        test_metrics = fit_final_model(
            split_data["trainval"],
            split_data["test"],
            compact_features,
            seed=seed,
            params=selection["params"],
            num_boost_round=selection["best_iteration"],
        )
        run_row = {
            "seed": seed,
            "selected_candidate": selection["candidate"],
            "best_iteration": selection["best_iteration"],
            "max_depth": selection["params"]["max_depth"],
            "eta": selection["params"]["eta"],
            "subsample": selection["params"]["subsample"],
            "colsample_bytree": selection["params"]["colsample_bytree"],
            "min_child_weight": selection["params"]["min_child_weight"],
            "lambda": selection["params"]["lambda"],
            "validation_pr_auc": selection["validation_metrics"]["pr_auc"],
            **test_metrics,
        }
        run_rows.append(run_row)
        print(
            f"seed={seed} candidate={selection['candidate']} best_iteration={selection['best_iteration']} "
            f"f1={test_metrics['f1']:.6f} pr_auc={test_metrics['pr_auc']:.6f}"
        )

    payload = {
        "artifact": "phase2_tier2_uncertainty_summary",
        "csv_path": args.csv_path,
        "seeds": args.seeds,
        "compact_feature_count": len(compact_features),
        "compact_features": compact_features,
        "protocol": {
            "split_type": "repeated stratified train/validation/test split by binary Ia/non-Ia label",
            "test_split": test_split,
            "validation_split": validation_split,
            "selection_metric": "validation PR-AUC",
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "final_fit": "train+validation",
            "final_evaluation": "held-out test for each seed",
        },
        "metric_summary": summarize_metrics(run_rows),
        "runs": run_rows,
        "split_summaries": split_summaries,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default=COMPACT_CSV_PATH)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_uncertainty(args)
    write_json(SUMMARY_JSON, payload)
    write_csv(
        RUNS_CSV,
        [
            "seed",
            "selected_candidate",
            "best_iteration",
            "max_depth",
            "eta",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "lambda",
            "validation_pr_auc",
            *METRIC_NAMES,
        ],
        payload["runs"],
    )
    write_markdown_summary(SUMMARY_MD, payload)
    print_metric_summary(payload)
    print(f"Wrote {RUNS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
