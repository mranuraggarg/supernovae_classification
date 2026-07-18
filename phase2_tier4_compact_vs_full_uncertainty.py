"""Paired uncertainty analysis for 16-feature compact vs 31-feature full models.

This script reruns both XGBoost models on identical stratified folds and seeds.
It is intentionally self-contained so that the paired comparison does not depend
on unavailable saved predictions from earlier one-off runs.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

import numpy as np


CSV_PATH = "data/processed/spcc_features_tier1.csv"
RESULTS_DIR = "results/phase2_tier4"
FOLD_METRICS_CSV = os.path.join(RESULTS_DIR, "compact_vs_full_fold_metrics.csv")
OOF_PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "compact_vs_full_oof_predictions.csv")
UNCERTAINTY_JSON = os.path.join(RESULTS_DIR, "compact_vs_full_uncertainty.json")
REPORT_MD = os.path.join(RESULTS_DIR, "compact_vs_full_uncertainty_report.md")

RANDOM_STATE = 42
N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260718

FULL_FEATURES = [
    "peak_flux_all",
    "amplitude_all",
    "mean_flux_all",
    "std_flux_all",
    "g_peak_flux",
    "g_mean_flux",
    "g_std_flux",
    "g_amplitude",
    "r_peak_flux",
    "r_mean_flux",
    "r_std_flux",
    "r_amplitude",
    "i_peak_flux",
    "i_mean_flux",
    "i_std_flux",
    "i_amplitude",
    "z_peak_flux",
    "z_mean_flux",
    "z_std_flux",
    "z_amplitude",
    "peak_color_g_minus_r",
    "peak_color_r_minus_i",
    "peak_color_i_minus_z",
    "time_of_peak_all",
    "g_time_of_peak",
    "r_time_of_peak",
    "i_time_of_peak",
    "z_time_of_peak",
    "observation_count",
    "time_span",
    "total_snr",
]

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

XGB_PARAM_GRID = [
    {
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
    },
    {
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "lambda": 1.0,
    },
    {
        "max_depth": 5,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2.0,
        "lambda": 1.5,
    },
]

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


@dataclass(frozen=True)
class Dataset:
    ids: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, Any]]


def load_rows(path: str = CSV_PATH) -> Dataset:
    rows: list[dict[str, Any]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, Any] = {
                "snid": int(raw_row["snid"]),
                "label_name": raw_row["label_name"],
            }
            for key, value in raw_row.items():
                if key in {"snid", "label_name"}:
                    continue
                row[key] = float(value)
            rows.append(row)
    ids = np.array([row["snid"] for row in rows], dtype=np.int64)
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    return Dataset(ids=ids, labels=labels, rows=rows)


def validate_features(rows: list[dict[str, Any]], feature_names: list[str]) -> None:
    missing = [name for name in feature_names if name not in rows[0]]
    if missing:
        raise KeyError(f"Missing required features in {CSV_PATH}: {missing}")


def build_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    return x, y


def stratified_kfold_indices(labels: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fold_buckets = [[] for _ in range(n_splits)]
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        for fold_index, index in enumerate(shuffled):
            fold_buckets[fold_index % n_splits].append(int(index))
    folds = []
    all_indices = np.arange(len(labels), dtype=np.int32)
    for fold_indices in fold_buckets:
        test_idx = np.array(sorted(fold_indices), dtype=np.int32)
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        folds.append((train_idx, test_idx))
    return folds


def stratified_split_indices(labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
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
    return np.array(sorted(train_indices), dtype=np.int32), np.array(sorted(test_indices), dtype=np.int32)


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mean = train_x.mean(axis=0)
    train_std = train_x.std(axis=0)
    train_std[train_std == 0.0] = 1.0
    return (train_x - train_mean) / train_std, (other_x - train_mean) / train_std, train_mean, train_std


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
    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc_score_numpy(y_true, probs),
        "pr_auc": average_precision_numpy(y_true, probs),
    }


def f1_score_only(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def base_xgb_params(seed: int, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": 4,
        "verbosity": 0,
        "seed": seed,
        "scale_pos_weight": scale_pos_weight,
    }


def train_model_for_fold(
    *,
    xgb: Any,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
    seed: int,
) -> dict[str, Any]:
    train_x_raw, train_y = build_matrix(train_rows, feature_names)
    test_x_raw, test_y = build_matrix(test_rows, feature_names)

    inner_train_idx, validation_idx = stratified_split_indices(train_y, 0.2, seed)
    inner_train_x_raw = train_x_raw[inner_train_idx]
    inner_train_y = train_y[inner_train_idx]
    validation_x_raw = train_x_raw[validation_idx]
    validation_y = train_y[validation_idx]

    inner_train_x, validation_x, _, _ = standardize(inner_train_x_raw, validation_x_raw)
    positive_count = float(np.sum(inner_train_y == 1))
    negative_count = float(np.sum(inner_train_y == 0))
    dtrain = xgb.DMatrix(inner_train_x, label=inner_train_y, feature_names=feature_names)
    dvalidation = xgb.DMatrix(validation_x, label=validation_y, feature_names=feature_names)

    best: dict[str, Any] | None = None
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
        metrics = binary_metrics(validation_y, probs)
        candidate = {
            "params": params,
            "best_iteration": int(booster.best_iteration + 1),
            "validation_metrics": metrics,
        }
        if best is None or metrics["pr_auc"] > best["validation_metrics"]["pr_auc"]:
            best = candidate
    assert best is not None

    train_x, test_x, train_mean, train_std = standardize(train_x_raw, test_x_raw)
    positive_count = float(np.sum(train_y == 1))
    negative_count = float(np.sum(train_y == 0))
    dtrain_final = xgb.DMatrix(train_x, label=train_y, feature_names=feature_names)
    dtest = xgb.DMatrix(test_x, label=test_y, feature_names=feature_names)
    booster = xgb.train(
        params={**base_xgb_params(seed, negative_count / max(positive_count, 1.0)), **best["params"]},
        dtrain=dtrain_final,
        num_boost_round=best["best_iteration"],
        verbose_eval=False,
    )
    probs = booster.predict(dtest)
    return {
        "metrics": binary_metrics(test_y, probs),
        "probabilities": probs.astype(float),
        "labels": test_y.astype(int),
        "selection": best,
        "train_mean": train_mean,
        "train_std": train_std,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    n = len(values)
    value_mean = mean(values)
    value_sd = stdev(values) if n > 1 else 0.0
    t_critical = T_CRITICAL_95.get(n - 1, 1.96)
    half_width = t_critical * value_sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": float(value_mean),
        "std": float(value_sd),
        "ci95_low": float(value_mean - half_width),
        "ci95_high": float(value_mean + half_width),
    }


def paired_sign_flip_p_value(differences: list[float]) -> float:
    observed = abs(mean(differences))
    if not differences:
        return 1.0
    count = 0
    extreme = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(differences)):
        signed_mean = abs(mean([sign * diff for sign, diff in zip(signs, differences)]))
        count += 1
        if signed_mean >= observed - 1e-15:
            extreme += 1
    return float(extreme / count)


def paired_bootstrap_diff_ci(y_true: np.ndarray, full_probs: np.ndarray, compact_probs: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = np.arange(len(y_true))
    diffs = np.zeros(BOOTSTRAP_ITERATIONS, dtype=float)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(indices, size=len(indices), replace=True)
        full_f1 = f1_score_only(y_true[sample], full_probs[sample])
        compact_f1 = f1_score_only(y_true[sample], compact_probs[sample])
        diffs[iteration] = compact_f1 - full_f1
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "mean": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
    }


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis() -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required to rerun the paired compact-vs-full uncertainty analysis. "
            "Install the project dependency and rerun this script."
        ) from exc

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = load_rows()
    validate_features(dataset.rows, FULL_FEATURES)
    validate_features(dataset.rows, COMPACT_FEATURES)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    full_oof = np.zeros(len(dataset.rows), dtype=float)
    compact_oof = np.zeros(len(dataset.rows), dtype=float)

    folds = stratified_kfold_indices(dataset.labels, N_SPLITS, RANDOM_STATE)
    for fold_index, (train_idx, test_idx) in enumerate(folds, start=1):
        seed = RANDOM_STATE + fold_index
        train_rows = [dataset.rows[index] for index in train_idx]
        test_rows = [dataset.rows[index] for index in test_idx]

        print(f"[INFO] Fold {fold_index}/{N_SPLITS}: training 31-feature full model")
        full_result = train_model_for_fold(
            xgb=xgb,
            train_rows=train_rows,
            test_rows=test_rows,
            feature_names=FULL_FEATURES,
            seed=seed,
        )
        print(f"[INFO] Fold {fold_index}/{N_SPLITS}: training 16-feature compact model")
        compact_result = train_model_for_fold(
            xgb=xgb,
            train_rows=train_rows,
            test_rows=test_rows,
            feature_names=COMPACT_FEATURES,
            seed=seed,
        )
        print(
            f"[OK] Fold {fold_index}: full F1={full_result['metrics']['f1']:.6f}, "
            f"compact F1={compact_result['metrics']['f1']:.6f}"
        )

        full_metrics = full_result["metrics"]
        compact_metrics = compact_result["metrics"]
        fold_rows.append(
            {
                "fold": fold_index,
                "seed": seed,
                "test_events": len(test_idx),
                "full_f1": full_metrics["f1"],
                "compact_f1": compact_metrics["f1"],
                "paired_delta_f1_compact_minus_full": compact_metrics["f1"] - full_metrics["f1"],
                "full_roc_auc": full_metrics["roc_auc"],
                "compact_roc_auc": compact_metrics["roc_auc"],
                "full_pr_auc": full_metrics["pr_auc"],
                "compact_pr_auc": compact_metrics["pr_auc"],
                "full_best_iteration": full_result["selection"]["best_iteration"],
                "compact_best_iteration": compact_result["selection"]["best_iteration"],
                "full_params": json.dumps(full_result["selection"]["params"], sort_keys=True),
                "compact_params": json.dumps(compact_result["selection"]["params"], sort_keys=True),
            }
        )

        full_oof[test_idx] = full_result["probabilities"]
        compact_oof[test_idx] = compact_result["probabilities"]
        for local_index, global_index in enumerate(test_idx):
            label = int(dataset.labels[global_index])
            full_prob = float(full_result["probabilities"][local_index])
            compact_prob = float(compact_result["probabilities"][local_index])
            oof_rows.append(
                {
                    "snid": int(dataset.ids[global_index]),
                    "fold": fold_index,
                    "label": label,
                    "full_probability": full_prob,
                    "compact_probability": compact_prob,
                    "full_prediction": int(full_prob >= 0.5),
                    "compact_prediction": int(compact_prob >= 0.5),
                }
            )

    full_f1_values = [float(row["full_f1"]) for row in fold_rows]
    compact_f1_values = [float(row["compact_f1"]) for row in fold_rows]
    differences = [float(row["paired_delta_f1_compact_minus_full"]) for row in fold_rows]
    diff_sd = stdev(differences) if len(differences) > 1 else 0.0
    effect_size_dz = mean(differences) / diff_sd if diff_sd > 0.0 else 0.0

    y_true = dataset.labels.astype(int)
    oof_full_metrics = binary_metrics(y_true, full_oof)
    oof_compact_metrics = binary_metrics(y_true, compact_oof)
    bootstrap = paired_bootstrap_diff_ci(y_true, full_oof, compact_oof)
    p_value = paired_sign_flip_p_value(differences)

    payload = {
        "analysis": "phase2_tier4_compact_vs_full_uncertainty",
        "data_source": CSV_PATH,
        "feature_sets": {
            "full": {"feature_count": len(FULL_FEATURES), "feature_names": FULL_FEATURES},
            "compact": {"feature_count": len(COMPACT_FEATURES), "feature_names": COMPACT_FEATURES},
        },
        "protocol": {
            "folds": N_SPLITS,
            "fold_seed": RANDOM_STATE,
            "validation_fraction_within_training_fold": 0.2,
            "model": "XGBoost binary:logistic",
            "param_grid": XGB_PARAM_GRID,
            "selection_metric": "validation PR-AUC",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "fold_rows": fold_rows,
        "summary": {
            "full_f1": summarize_values(full_f1_values),
            "compact_f1": summarize_values(compact_f1_values),
            "paired_delta_f1_compact_minus_full": summarize_values(differences),
            "paired_sign_flip_p_value": p_value,
            "effect_size_cohens_dz": float(effect_size_dz),
            "oof_full_metrics": oof_full_metrics,
            "oof_compact_metrics": oof_compact_metrics,
            "oof_delta_f1_compact_minus_full": float(oof_compact_metrics["f1"] - oof_full_metrics["f1"]),
            "paired_bootstrap_oof_delta_f1": bootstrap,
        },
        "outputs": {
            "fold_metrics_csv": FOLD_METRICS_CSV,
            "oof_predictions_csv": OOF_PREDICTIONS_CSV,
            "uncertainty_json": UNCERTAINTY_JSON,
            "report_md": REPORT_MD,
        },
    }

    write_csv(
        FOLD_METRICS_CSV,
        [
            "fold",
            "seed",
            "test_events",
            "full_f1",
            "compact_f1",
            "paired_delta_f1_compact_minus_full",
            "full_roc_auc",
            "compact_roc_auc",
            "full_pr_auc",
            "compact_pr_auc",
            "full_best_iteration",
            "compact_best_iteration",
            "full_params",
            "compact_params",
        ],
        fold_rows,
    )
    write_csv(
        OOF_PREDICTIONS_CSV,
        [
            "snid",
            "fold",
            "label",
            "full_probability",
            "compact_probability",
            "full_prediction",
            "compact_prediction",
        ],
        oof_rows,
    )
    with open(UNCERTAINTY_JSON, "w") as handle:
        json.dump(payload, handle, indent=2)
    write_report(payload)
    return payload


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    full = summary["full_f1"]
    compact = summary["compact_f1"]
    diff = summary["paired_delta_f1_compact_minus_full"]
    bootstrap = summary["paired_bootstrap_oof_delta_f1"]
    lines = [
        "# Compact versus Full Feature-Set Uncertainty",
        "",
        "This analysis reruns the 31-feature full model and 16-feature compact model on identical stratified folds and seeds.",
        "",
        "## Fold-Level Summary",
        "",
        "| Model | Mean F1 | F1 std. | 95% CI |",
        "| --- | ---: | ---: | ---: |",
        f"| 31-feature full | {_fmt(full['mean'])} | {_fmt(full['std'])} | [{_fmt(full['ci95_low'])}, {_fmt(full['ci95_high'])}] |",
        f"| 16-feature compact | {_fmt(compact['mean'])} | {_fmt(compact['std'])} | [{_fmt(compact['ci95_low'])}, {_fmt(compact['ci95_high'])}] |",
        "",
        "## Paired Difference",
        "",
        f"- Mean paired F1 difference, compact minus full: {_fmt(diff['mean'])}",
        f"- Fold-level 95% CI: [{_fmt(diff['ci95_low'])}, {_fmt(diff['ci95_high'])}]",
        f"- Exact paired sign-flip p-value over folds: {_fmt(summary['paired_sign_flip_p_value'])}",
        f"- Cohen's dz over fold differences: {_fmt(summary['effect_size_cohens_dz'])}",
        f"- Out-of-fold paired bootstrap 95% CI for F1 difference: [{_fmt(bootstrap['ci95_low'])}, {_fmt(bootstrap['ci95_high'])}]",
        "",
        "## Interpretation",
        "",
    ]
    if diff["ci95_low"] <= 0.0 <= diff["ci95_high"]:
        lines.append(
            "The paired fold-level confidence interval includes zero, so the compact model should not be described as significantly better on F1. The supported conclusion is that the 16-feature representation preserves performance while using substantially fewer and more interpretable features."
        )
    else:
        lines.append(
            "The paired fold-level confidence interval excludes zero. Interpret this result together with the bootstrap interval and the small number of folds before making any strong performance-superiority claim."
        )
    lines += [
        "",
        "## Output Files",
        "",
        f"- Fold metrics: `{FOLD_METRICS_CSV}`",
        f"- Out-of-fold predictions: `{OOF_PREDICTIONS_CSV}`",
        f"- JSON summary: `{UNCERTAINTY_JSON}`",
        f"- Report: `{REPORT_MD}`",
        "",
    ]
    with open(REPORT_MD, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    payload = run_analysis()
    print(
        json.dumps(
            {
                "fold_metrics_csv": payload["outputs"]["fold_metrics_csv"],
                "uncertainty_json": payload["outputs"]["uncertainty_json"],
                "report_md": payload["outputs"]["report_md"],
                "mean_delta_f1": payload["summary"]["paired_delta_f1_compact_minus_full"]["mean"],
                "p_value": payload["summary"]["paired_sign_flip_p_value"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
