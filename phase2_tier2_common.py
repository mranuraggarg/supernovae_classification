"""Shared utilities for Phase 2 Tier 2 compact-feature analyses."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from phase2_tier1_benchmarks import (
    RANDOM_STATE,
    TEST_SPLIT,
    VALIDATION_SPLIT,
    load_rows,
    stratified_split_indices,
)
from phase2_tier1_xgb_importance import XGB_PARAM_GRID


RESULTS_DIR = "results/phase2_tier2"
PLOTS_DIR = "plots/phase2_tier2"

COMPACT_FEATURES = [
    "z_peak_flux",
    "r_mean_flux",
    "peak_color_g_minus_r",
    "i_peak_flux",
    "peak_color_r_minus_i",
    "peak_color_i_minus_z",
    "g_mean_flux",
    "r_peak_flux",
    "z_std_flux",
    "i_amplitude",
    "i_std_flux",
    "time_span",
    "z_time_of_peak",
    "i_time_of_peak",
    "r_time_of_peak",
    "r_std_flux",
]

FEATURE_GROUPS = {
    "brightness": [
        "r_mean_flux",
        "g_mean_flux",
        "z_peak_flux",
        "i_peak_flux",
        "r_peak_flux",
    ],
    "color": [
        "peak_color_g_minus_r",
        "peak_color_r_minus_i",
        "peak_color_i_minus_z",
    ],
    "variability": [
        "i_std_flux",
        "z_std_flux",
        "r_std_flux",
        "i_amplitude",
    ],
    "temporal": [
        "r_time_of_peak",
        "i_time_of_peak",
        "z_time_of_peak",
        "time_span",
    ],
}

GROUP_ORDER = ["brightness", "color", "variability", "temporal"]

FROZEN_BASELINE_METRICS = {
    "f1": 0.8442299254,
    "roc_auc": 0.9765883838,
    "pr_auc": 0.9277608810,
}

BASELINE_MANIFEST_PATH = "results/phase2_tier1/phase2_tier1_compact_baseline_manifest.json"
BASELINE_METRICS_PATH = "results/phase2_tier1/phase2_tier1_compact_baseline_metrics.json"
BASELINE_COMPACT_CSV_PATH = "data/processed/phase2_tier1_compact_baseline.csv"
FULL_TIER1_CSV_PATH = "data/processed/spcc_features_tier1.csv"
INTERPRETATION_TABLE_PATH = "results/phase2_tier1/phase2_tier1_compact_baseline_interpretation_table.csv"
COMPACT_IMPORTANCE_PATH = "results/phase2_tier1/phase2_tier1_compact_baseline_importance.json"


@dataclass
class Tier2Context:
    rows: list[dict[str, Any]]
    split_data: dict[str, list[dict[str, Any]]]
    baseline_metrics: dict[str, float]
    compact_features: list[str]
    source_csv: str
    baseline_manifest_found: bool
    baseline_metrics_found: bool


def ensure_output_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def feature_to_group(feature_name: str) -> str:
    for group_name, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return group_name
    raise KeyError(f"Feature {feature_name!r} is not in the compact Tier 1 feature groups.")


def parse_numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                parsed[key] = value
                continue
            if key in {"feature_removed", "feature_group", "block_removed", "subset_name", "selection_rule", "feature_list", "included_blocks", "label"}:
                parsed[key] = value
                continue
            if value == "":
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


def load_baseline_reference() -> tuple[list[str], dict[str, float], bool, bool]:
    compact_features = list(COMPACT_FEATURES)
    baseline_metrics = dict(FROZEN_BASELINE_METRICS)
    manifest_found = False
    metrics_found = False

    if os.path.exists(BASELINE_MANIFEST_PATH):
        manifest_payload = read_json(BASELINE_MANIFEST_PATH)
        compact_features = list(manifest_payload.get("feature_names", compact_features))
        manifest_found = True

    if os.path.exists(BASELINE_METRICS_PATH):
        metrics_payload = read_json(BASELINE_METRICS_PATH)
        test_metrics = metrics_payload.get("test_metrics", {})
        baseline_metrics = {
            "f1": float(test_metrics.get("f1", baseline_metrics["f1"])),
            "roc_auc": float(test_metrics.get("roc_auc", baseline_metrics["roc_auc"])),
            "pr_auc": float(test_metrics.get("pr_auc", baseline_metrics["pr_auc"])),
        }
        metrics_found = True

    return compact_features, baseline_metrics, manifest_found, metrics_found


def resolve_compact_source_csv() -> str:
    for candidate in (BASELINE_COMPACT_CSV_PATH, FULL_TIER1_CSV_PATH):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Phase 2 Tier 2 requires a processed Tier 1 CSV. Expected one of "
        f"{BASELINE_COMPACT_CSV_PATH!r} or {FULL_TIER1_CSV_PATH!r}."
    )


def load_compact_rows(compact_features: list[str]) -> tuple[list[dict[str, Any]], str]:
    source_csv = resolve_compact_source_csv()
    rows = load_rows(source_csv)
    if not rows:
        raise ValueError(f"No rows found in {source_csv}.")

    missing_features = [name for name in compact_features if name not in rows[0]]
    if missing_features:
        raise KeyError(
            "The processed Tier 1 CSV is missing compact features required for Tier 2: "
            + ", ".join(missing_features)
        )
    return rows, source_csv


def split_compact_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    trainval_idx, test_idx = stratified_split_indices(labels, TEST_SPLIT, RANDOM_STATE)
    train_idx, validation_idx = stratified_split_indices(labels[trainval_idx], VALIDATION_SPLIT, RANDOM_STATE)
    return {
        "all_rows": rows,
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
    *,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    x_train_raw, y_train = build_matrix(train_rows, feature_names)
    x_validation_raw, y_validation = build_matrix(validation_rows, feature_names)
    x_train, x_validation = standardize(x_train_raw, x_validation_raw)

    positive_count = float(np.sum(y_train == 1))
    negative_count = float(np.sum(y_train == 0))
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dvalidation = xgb.DMatrix(x_validation, label=y_validation, feature_names=feature_names)

    best_candidate: dict[str, Any] | None = None
    for params in XGB_PARAM_GRID:
        booster = xgb.train(
            params={**base_xgb_params(seed, negative_count / max(positive_count, 1.0)), **params},
            dtrain=dtrain,
            num_boost_round=400,
            evals=[(dvalidation, "validation")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        probs = booster.predict(dvalidation, iteration_range=(0, booster.best_iteration + 1))
        metrics = binary_metrics(y_validation, probs)
        candidate = {
            "params": params,
            "best_iteration": int(booster.best_iteration + 1),
            "validation_metrics": metrics,
        }
        if best_candidate is None or metrics["pr_auc"] > best_candidate["validation_metrics"]["pr_auc"]:
            best_candidate = candidate

    assert best_candidate is not None
    return best_candidate


def fit_final_model(
    trainval_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    seed: int,
    params: dict[str, Any],
    num_boost_round: int,
) -> tuple[xgb.Booster, dict[str, float], np.ndarray]:
    x_trainval_raw, y_trainval = build_matrix(trainval_rows, feature_names)
    x_test_raw, y_test = build_matrix(test_rows, feature_names)
    x_trainval, x_test = standardize(x_trainval_raw, x_test_raw)

    positive_count = float(np.sum(y_trainval == 1))
    negative_count = float(np.sum(y_trainval == 0))
    dtrainval = xgb.DMatrix(x_trainval, label=y_trainval, feature_names=feature_names)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
    booster = xgb.train(
        params={**base_xgb_params(seed, negative_count / max(positive_count, 1.0)), **params},
        dtrain=dtrainval,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    probs = booster.predict(dtest)
    return booster, binary_metrics(y_test, probs), y_test


def compute_delta_metrics(metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "delta_f1": metrics["f1"] - baseline_metrics["f1"],
        "delta_roc_auc": metrics["roc_auc"] - baseline_metrics["roc_auc"],
        "delta_pr_auc": metrics["pr_auc"] - baseline_metrics["pr_auc"],
    }


def evaluate_feature_subset(
    context: Tier2Context,
    feature_names: list[str],
    *,
    subset_name: str,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    selection = select_best_model(context.split_data["train"], context.split_data["validation"], feature_names, seed=seed)
    _, metrics, _ = fit_final_model(
        context.split_data["trainval"],
        context.split_data["test"],
        feature_names,
        seed=seed,
        params=selection["params"],
        num_boost_round=selection["best_iteration"],
    )
    return {
        "subset_name": subset_name,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "selection_summary": selection,
        "metrics": metrics,
        **compute_delta_metrics(metrics, context.baseline_metrics),
    }


def create_context() -> Tier2Context:
    ensure_output_dirs()
    compact_features, baseline_metrics, manifest_found, metrics_found = load_baseline_reference()
    rows, source_csv = load_compact_rows(compact_features)
    split_data = split_compact_rows(rows)
    return Tier2Context(
        rows=rows,
        split_data=split_data,
        baseline_metrics=baseline_metrics,
        compact_features=compact_features,
        source_csv=source_csv,
        baseline_manifest_found=manifest_found,
        baseline_metrics_found=metrics_found,
    )


def baseline_reference_payload(context: Tier2Context) -> dict[str, Any]:
    return {
        "baseline_metrics": context.baseline_metrics,
        "baseline_manifest_found": context.baseline_manifest_found,
        "baseline_metrics_found": context.baseline_metrics_found,
        "compact_feature_count": len(context.compact_features),
        "compact_features": context.compact_features,
        "source_csv": context.source_csv,
        "split_manifest": {
            "random_state": RANDOM_STATE,
            "train_count": len(context.split_data["train"]),
            "validation_count": len(context.split_data["validation"]),
            "trainval_count": len(context.split_data["trainval"]),
            "test_count": len(context.split_data["test"]),
        },
    }


def run_baseline_parity_check(context: Tier2Context) -> dict[str, Any]:
    baseline_run = evaluate_feature_subset(context, context.compact_features, subset_name="compact_baseline_parity")
    return {
        "metrics": baseline_run["metrics"],
        "delta_f1": baseline_run["delta_f1"],
        "delta_roc_auc": baseline_run["delta_roc_auc"],
        "delta_pr_auc": baseline_run["delta_pr_auc"],
    }


def load_interpretation_rows() -> list[dict[str, Any]]:
    if not os.path.exists(INTERPRETATION_TABLE_PATH):
        return []
    return parse_numeric_rows(read_csv_rows(INTERPRETATION_TABLE_PATH))


def load_compact_importance_rows() -> list[dict[str, Any]]:
    if not os.path.exists(COMPACT_IMPORTANCE_PATH):
        return []
    payload = read_json(COMPACT_IMPORTANCE_PATH)
    return list(payload.get("permutation_importance", []))


def rank_rows_by_delta(rows: list[dict[str, Any]], key_name: str, rank_name: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: (row[key_name], row.get("delta_pr_auc", 0.0)))
    for rank, row in enumerate(sorted_rows, start=1):
        row[rank_name] = rank


def safe_label_from_loss(loss_f1: float, loss_pr_auc: float, max_loss_f1: float, max_loss_pr_auc: float) -> str:
    if loss_f1 <= 0.001 and loss_pr_auc <= 0.001:
        return "redundant"
    essential_f1 = max(0.02, 0.6 * max_loss_f1)
    essential_pr = max(0.02, 0.6 * max_loss_pr_auc)
    supportive_f1 = max(0.006, 0.25 * max_loss_f1)
    supportive_pr = max(0.006, 0.25 * max_loss_pr_auc)
    if loss_f1 >= essential_f1 or loss_pr_auc >= essential_pr:
        return "essential"
    if loss_f1 >= supportive_f1 or loss_pr_auc >= supportive_pr:
        return "supportive"
    return "marginal"


def write_simple_markdown_table(path: str, headers: list[str], rows: list[list[str]], intro_lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    with open(path, "w") as handle:
        handle.write("\n".join(intro_lines + ["", *table_lines, ""]) )


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate Tier 2 plots. Install project dependencies and rerun."
        ) from exc
    return plt


def sorted_feature_groups() -> list[tuple[str, list[str]]]:
    return [(group_name, FEATURE_GROUPS[group_name]) for group_name in GROUP_ORDER]


def select_ranked_core_subsets(
    compact_features: list[str],
    feature_ablation_rows: list[dict[str, Any]],
    interpretation_rows: list[dict[str, Any]],
    subset_sizes: list[int],
) -> list[dict[str, Any]]:
    ablation_loss = {
        row["feature_removed"]: max(-float(row["delta_f1"]), 0.0) + 0.35 * max(-float(row["delta_pr_auc"]), 0.0)
        for row in feature_ablation_rows
    }
    shap_rank = {row["feature"]: int(row["SHAP rank"]) for row in interpretation_rows if "SHAP rank" in row}
    perm_drop = {
        row["feature"]: float(row.get("perm PR-AUC drop", 0.0)) + float(row.get("perm F1 drop", 0.0))
        for row in interpretation_rows
    }

    scored_features: list[dict[str, Any]] = []
    for feature_name in compact_features:
        score = ablation_loss.get(feature_name, 0.0)
        if feature_name in shap_rank:
            score += 1.0 / shap_rank[feature_name]
        score += perm_drop.get(feature_name, 0.0)
        scored_features.append(
            {
                "feature": feature_name,
                "group": feature_to_group(feature_name),
                "score": score,
            }
        )

    ranked_features = [row["feature"] for row in sorted(scored_features, key=lambda row: row["score"], reverse=True)]
    group_best = {
        group_name: next((feature for feature in ranked_features if feature_to_group(feature) == group_name), None)
        for group_name in GROUP_ORDER
    }

    subsets: list[dict[str, Any]] = []
    for subset_size in subset_sizes:
        selected: list[str] = []
        for group_name in GROUP_ORDER:
            feature_name = group_best[group_name]
            if feature_name and feature_name not in selected and len(selected) < subset_size:
                selected.append(feature_name)
        for feature_name in ranked_features:
            if feature_name not in selected and len(selected) < subset_size:
                selected.append(feature_name)
        subsets.append(
            {
                "subset_name": f"top_{subset_size}",
                "selection_rule": "ablation-plus-importance with minimum one feature per physical family when possible",
                "feature_names": selected,
            }
        )
    return subsets


def format_feature_list(feature_names: list[str]) -> str:
    return ", ".join(feature_names)


def normalize_path(path: str) -> str:
    return str(Path(path))
