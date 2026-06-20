"""Finalize the Phase 2 Tier-1 compact baseline and write official artifacts."""

from __future__ import annotations

import csv
import json
import os
from statistics import mean, pstdev

import numpy as np
import xgboost as xgb

from phase2_tier1_benchmarks import CSV_PATH, load_rows, stratified_split_indices
from phase2_tier1_xgb_importance import XGB_PARAM_GRID, gain_importance, permutation_importance


RANDOM_STATE = 42
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
SEED_SWEEP = [42, 52, 62]

RESULTS_DIR = "results/phase2_tier1"
MODELS_DIR = "models/phase2_tier1/compact_baseline"
DATA_DIR = "data/processed"

BASELINE_NAME = "phase2_tier1_compact_baseline"
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


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std


def build_matrix(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    return x, y


def split_rows() -> dict[str, list[dict]]:
    rows = load_rows(CSV_PATH)
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    trainval_idx, test_idx = stratified_split_indices(labels, TEST_SPLIT, RANDOM_STATE)
    train_idx, val_idx = stratified_split_indices(labels[trainval_idx], VALIDATION_SPLIT, RANDOM_STATE)
    return {
        "all_rows": rows,
        "train": [rows[trainval_idx[index]] for index in train_idx],
        "validation": [rows[trainval_idx[index]] for index in val_idx],
        "trainval": [rows[index] for index in trainval_idx],
        "test": [rows[index] for index in test_idx],
    }


def base_xgb_params(seed: int, scale_pos_weight: float) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": seed,
        "scale_pos_weight": scale_pos_weight,
    }


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
    for p_value, r_value, label in zip(precision, recall, y_sorted):
        if label == 1:
            ap += p_value * (r_value - previous_recall)
            previous_recall = r_value
    return float(ap)


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
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


def select_best_model(
    train_rows: list[dict],
    validation_rows: list[dict],
    feature_names: list[str],
    *,
    seed: int,
) -> dict:
    x_train_raw, y_train = build_matrix(train_rows, feature_names)
    x_val_raw, y_val = build_matrix(validation_rows, feature_names)
    x_train, x_val = standardize(x_train_raw, x_val_raw)

    positive_count = float(np.sum(y_train == 1))
    negative_count = float(np.sum(y_train == 0))
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_names)

    best = None
    for params in XGB_PARAM_GRID:
        booster = xgb.train(
            params={**base_xgb_params(seed, negative_count / max(positive_count, 1.0)), **params},
            dtrain=dtrain,
            num_boost_round=400,
            evals=[(dval, "validation")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        probs = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        metrics = binary_metrics(y_val, probs)
        candidate = {
            "params": params,
            "best_iteration": int(booster.best_iteration + 1),
            "validation_metrics": metrics,
        }
        if best is None or metrics["pr_auc"] > best["validation_metrics"]["pr_auc"]:
            best = candidate
    assert best is not None
    return best


def fit_final_model(
    trainval_rows: list[dict],
    test_rows: list[dict],
    feature_names: list[str],
    *,
    seed: int,
    params: dict,
    num_boost_round: int,
) -> tuple[xgb.Booster, dict, np.ndarray, np.ndarray, np.ndarray]:
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
    return booster, binary_metrics(y_test, probs), probs, y_test, x_test


def write_compact_dataset(rows: list[dict], feature_names: list[str]) -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{BASELINE_NAME}.csv")
    npz_path = os.path.join(DATA_DIR, f"{BASELINE_NAME}.npz")

    fieldnames = ["snid", "label_name", "label_id", "sim_z", *feature_names]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    feature_matrix = np.array([[row[name] for name in feature_names] for row in rows], dtype=np.float32)
    labels = np.array([row["label_id"] for row in rows], dtype=np.int32)
    ids = np.array([row["snid"] for row in rows], dtype=np.int32)
    np.savez(
        npz_path,
        ids=ids,
        labels=labels,
        feature_names=np.array(feature_names, dtype=object),
        feature_matrix=feature_matrix,
    )
    return {"csv_path": csv_path, "npz_path": npz_path}


def write_comparison_table(compact_payload: dict) -> dict:
    compact_results = compact_payload["results"]["compact_working_set"]
    previous_results = compact_payload["results"]["previous_working_set"]
    full_results = compact_payload["results"]["full_baseline"]
    full_metrics = full_results["metrics"]

    table_rows = [
        ("31-feature full baseline", full_results["feature_count"], full_metrics),
        ("30-feature working set", previous_results["feature_count"], previous_results["metrics"]),
        ("16-feature compact baseline", compact_results["feature_count"], compact_results["metrics"]),
    ]

    csv_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_comparison.csv")
    md_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_comparison.md")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "feature_count", "f1", "roc_auc", "pr_auc", "delta_f1_vs_full", "delta_roc_auc_vs_full", "delta_pr_auc_vs_full"])
        for name, count, metrics in table_rows:
            writer.writerow([
                name,
                count,
                metrics["f1"],
                metrics["roc_auc"],
                metrics["pr_auc"],
                metrics["f1"] - full_metrics["f1"],
                metrics["roc_auc"] - full_metrics["roc_auc"],
                metrics["pr_auc"] - full_metrics["pr_auc"],
            ])

    lines = [
        "| name | feature_count | f1 | roc_auc | pr_auc | delta_f1_vs_full | delta_roc_auc_vs_full | delta_pr_auc_vs_full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, count, metrics in table_rows:
        lines.append(
            f"| {name} | {count} | {metrics['f1']:.6f} | {metrics['roc_auc']:.6f} | {metrics['pr_auc']:.6f} | "
            f"{metrics['f1'] - full_metrics['f1']:+.6f} | {metrics['roc_auc'] - full_metrics['roc_auc']:+.6f} | {metrics['pr_auc'] - full_metrics['pr_auc']:+.6f} |"
        )
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return {"csv_path": csv_path, "markdown_path": md_path}


def run_seed_robustness(
    split_data: dict[str, list[dict]],
    feature_names: list[str],
) -> dict:
    rows = []
    for seed in SEED_SWEEP:
        selection = select_best_model(split_data["train"], split_data["validation"], feature_names, seed=seed)
        _, metrics, _, _, _ = fit_final_model(
            split_data["trainval"],
            split_data["test"],
            feature_names,
            seed=seed,
            params=selection["params"],
            num_boost_round=selection["best_iteration"],
        )
        rows.append({
            "seed": seed,
            "best_iteration": selection["best_iteration"],
            "params": selection["params"],
            "metrics": metrics,
        })

    summary = {}
    for metric_name in ("f1", "roc_auc", "pr_auc"):
        values = [row["metrics"][metric_name] for row in rows]
        summary[metric_name] = {
            "mean": mean(values),
            "std": pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return {"seeds": rows, "summary": summary}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    split_data = split_rows()
    compact_dataset_paths = write_compact_dataset(split_data["all_rows"], COMPACT_FEATURES)

    selection = select_best_model(split_data["train"], split_data["validation"], COMPACT_FEATURES, seed=RANDOM_STATE)
    booster, metrics, _, y_test, x_test = fit_final_model(
        split_data["trainval"],
        split_data["test"],
        COMPACT_FEATURES,
        seed=RANDOM_STATE,
        params=selection["params"],
        num_boost_round=selection["best_iteration"],
    )

    model_path = os.path.join(MODELS_DIR, f"{BASELINE_NAME}_xgb.json")
    booster.save_model(model_path)

    compact_manifest = {
        "baseline_name": BASELINE_NAME,
        "feature_count": len(COMPACT_FEATURES),
        "feature_names": COMPACT_FEATURES,
        "source_csv": CSV_PATH,
        "dataset_artifacts": compact_dataset_paths,
    }
    compact_manifest_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_manifest.json")
    with open(compact_manifest_path, "w") as handle:
        json.dump(compact_manifest, handle, indent=2)

    metrics_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(
            {
                "baseline_name": BASELINE_NAME,
                "selection_summary": selection,
                "test_metrics": metrics,
                "model_path": model_path,
            },
            handle,
            indent=2,
        )

    importance_payload = {
        "baseline_name": BASELINE_NAME,
        "selection_summary": selection,
        "test_metrics": metrics,
        "importance_scope_note": "Feature importance is conditional on this preprocessing chain, this XGBoost model family, and this fixed split.",
        "permutation_importance": permutation_importance(booster, x_test, y_test, COMPACT_FEATURES),
        "gain_importance": gain_importance(booster, COMPACT_FEATURES),
        "model_path": model_path,
    }
    importance_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_importance.json")
    with open(importance_path, "w") as handle:
        json.dump(importance_payload, handle, indent=2)

    compact_rerun_path = os.path.join(RESULTS_DIR, "phase2_tier1_compact_rerun.json")
    with open(compact_rerun_path) as handle:
        compact_rerun_payload = json.load(handle)
    comparison_paths = write_comparison_table(compact_rerun_payload)

    robustness_payload = run_seed_robustness(split_data, COMPACT_FEATURES)
    robustness_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_robustness.json")
    with open(robustness_path, "w") as handle:
        json.dump(robustness_payload, handle, indent=2)

    summary = {
        "baseline_name": BASELINE_NAME,
        "compact_manifest_path": compact_manifest_path,
        "compact_dataset_paths": compact_dataset_paths,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "importance_path": importance_path,
        "comparison_paths": comparison_paths,
        "robustness_path": robustness_path,
    }
    summary_path = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_artifacts.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
