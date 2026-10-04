"""Phase 2 Tier 1 nonlinear feature-importance workflow.

This script is designed for environments where `xgboost` loads correctly.
It uses the fixed SPCC Tier 1 split, trains an XGBoost baseline on the
`full_tier1_baseline` feature block, computes permutation importance on the
held-out test split, records gain importance as a secondary view, builds an
initial accept/review/reject list, and compares reduced feature sets back
against the full baseline.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import xgboost as xgb

from phase2_tier1_benchmarks import (
    CSV_PATH,
    FEATURE_SETS,
    RANDOM_STATE,
    TEST_SPLIT,
    VALIDATION_SPLIT,
    average_precision_numpy,
    binary_metrics,
    build_matrix,
    load_rows,
    roc_auc_score_numpy,
    stratified_split_indices,
)


RESULTS_DIR = "results/phase2_tier1"
MODELS_DIR = "models/phase2_tier1/xgboost_importance"

FULL_BASELINE_NAME = "full_tier1_baseline"
FULL_BASELINE_FEATURES = FEATURE_SETS[FULL_BASELINE_NAME]

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


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def split_rows() -> dict[str, list[dict]]:
    rows = load_rows(CSV_PATH)
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    trainval_idx, test_idx = stratified_split_indices(labels, TEST_SPLIT, RANDOM_STATE)
    train_idx, val_idx = stratified_split_indices(labels[trainval_idx], VALIDATION_SPLIT, RANDOM_STATE)
    return {
        "train": [rows[trainval_idx[index]] for index in train_idx],
        "validation": [rows[trainval_idx[index]] for index in val_idx],
        "trainval": [rows[index] for index in trainval_idx],
        "test": [rows[index] for index in test_idx],
    }


def _base_xgb_params(scale_pos_weight: float) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
    }


def _prepare_data(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return build_matrix(rows, feature_names)


def _train_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    *,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 30,
) -> tuple[xgb.Booster, dict]:
    positive_count = float(np.sum(y_train == 1.0))
    negative_count = float(np.sum(y_train == 0.0))
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    booster = xgb.train(
        params={**_base_xgb_params(negative_count / max(positive_count, 1.0)), **params},
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    probs = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    metrics = binary_metrics(y_val, probs)
    return booster, metrics


def select_best_xgb_model(
    train_rows: list[dict],
    validation_rows: list[dict],
    feature_names: list[str],
) -> tuple[dict, dict]:
    x_train_raw, y_train = _prepare_data(train_rows, feature_names)
    x_val_raw, y_val = _prepare_data(validation_rows, feature_names)
    x_train, x_val, mean, std = standardize(x_train_raw, x_val_raw)

    best = None
    for params in XGB_PARAM_GRID:
        booster, metrics = _train_candidate(x_train, y_train, x_val, y_val, params)
        candidate = {
            "params": params,
            "booster": booster,
            "validation_metrics": metrics,
            "best_iteration": int(booster.best_iteration + 1),
            "mean": mean,
            "std": std,
        }
        if best is None or metrics["pr_auc"] > best["validation_metrics"]["pr_auc"]:
            best = candidate

    assert best is not None
    return best, {
        "feature_names": feature_names,
        "validation_metrics": best["validation_metrics"],
        "best_iteration": best["best_iteration"],
        "params": best["params"],
    }


def fit_final_xgb_model(
    trainval_rows: list[dict],
    test_rows: list[dict],
    feature_names: list[str],
    params: dict,
    num_boost_round: int,
) -> tuple[xgb.Booster, dict, np.ndarray, np.ndarray, np.ndarray]:
    x_trainval_raw, y_trainval = _prepare_data(trainval_rows, feature_names)
    x_test_raw, y_test = _prepare_data(test_rows, feature_names)
    x_trainval, x_test, mean, std = standardize(x_trainval_raw, x_test_raw)

    positive_count = float(np.sum(y_trainval == 1.0))
    negative_count = float(np.sum(y_trainval == 0.0))
    dtrainval = xgb.DMatrix(x_trainval, label=y_trainval, feature_names=feature_names)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
    booster = xgb.train(
        params={**_base_xgb_params(negative_count / max(positive_count, 1.0)), **params},
        dtrain=dtrainval,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    probs = booster.predict(dtest)
    metrics = binary_metrics(y_test, probs)
    return booster, metrics, probs, y_test, x_test


def permutation_importance(
    booster: xgb.Booster,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    num_repeats: int = 5,
) -> list[dict]:
    base_metrics = binary_metrics(y_test, booster.predict(xgb.DMatrix(x_test, feature_names=feature_names)))
    rng = np.random.default_rng(RANDOM_STATE)
    results = []
    for feature_index, feature_name in enumerate(feature_names):
        pr_auc_drops = []
        f1_drops = []
        for _ in range(num_repeats):
            shuffled = x_test.copy()
            rng.shuffle(shuffled[:, feature_index])
            probs = booster.predict(xgb.DMatrix(shuffled, feature_names=feature_names))
            metrics = binary_metrics(y_test, probs)
            pr_auc_drops.append(base_metrics["pr_auc"] - metrics["pr_auc"])
            f1_drops.append(base_metrics["f1"] - metrics["f1"])
        results.append(
            {
                "feature": feature_name,
                "mean_pr_auc_drop": float(np.mean(pr_auc_drops)),
                "mean_f1_drop": float(np.mean(f1_drops)),
                "max_pr_auc_drop": float(np.max(pr_auc_drops)),
                "max_f1_drop": float(np.max(f1_drops)),
            }
        )
    results.sort(key=lambda item: (item["mean_pr_auc_drop"], item["mean_f1_drop"]), reverse=True)
    return results


def gain_importance(booster: xgb.Booster, feature_names: list[str]) -> list[dict]:
    raw_gain = booster.get_score(importance_type="gain")
    raw_weight = booster.get_score(importance_type="weight")
    output = []
    for feature_name in feature_names:
        output.append(
            {
                "feature": feature_name,
                "gain": float(raw_gain.get(feature_name, 0.0)),
                "split_count": int(raw_weight.get(feature_name, 0)),
            }
        )
    output.sort(key=lambda item: item["gain"], reverse=True)
    return output


def build_feature_buckets(permutation_rows: list[dict], gain_rows: list[dict]) -> dict:
    gain_rank = {row["feature"]: index for index, row in enumerate(gain_rows)}
    buckets = {"accept": [], "review": [], "reject": []}
    for index, row in enumerate(permutation_rows):
        feature = row["feature"]
        gain_position = gain_rank.get(feature, len(gain_rows))
        strong_perm = row["mean_pr_auc_drop"] >= 0.003 or row["mean_f1_drop"] >= 0.003
        weak_perm = row["mean_pr_auc_drop"] <= 0.0005 and row["mean_f1_drop"] <= 0.0005
        strong_gain = gain_position < 10
        weak_gain = gain_position >= 20

        if strong_perm and strong_gain:
            buckets["accept"].append(feature)
        elif weak_perm and weak_gain:
            buckets["reject"].append(feature)
        else:
            buckets["review"].append(feature)

    return buckets


def compare_feature_sets(
    split_data: dict[str, list[dict]],
    params: dict,
    full_iteration_count: int,
    buckets: dict[str, list[str]],
) -> list[dict]:
    comparisons = []
    candidate_sets = {
        "accepted_only": buckets["accept"],
        "accepted_plus_review": buckets["accept"] + buckets["review"],
        "full_baseline": FULL_BASELINE_FEATURES,
    }
    for name, feature_names in candidate_sets.items():
        booster, metrics, _, _, _ = fit_final_xgb_model(
            split_data["trainval"],
            split_data["test"],
            feature_names,
            params,
            full_iteration_count,
        )
        comparisons.append(
            {
                "name": name,
                "feature_count": len(feature_names),
                "features": feature_names,
                "metrics": metrics,
                "gain_importance_top10": gain_importance(booster, feature_names)[:10],
            }
        )
    return comparisons


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def run_workflow() -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    split_data = split_rows()

    selected_model, validation_summary = select_best_xgb_model(
        split_data["train"],
        split_data["validation"],
        FULL_BASELINE_FEATURES,
    )
    final_booster, final_metrics, test_probs, y_test, x_test = fit_final_xgb_model(
        split_data["trainval"],
        split_data["test"],
        FULL_BASELINE_FEATURES,
        selected_model["params"],
        selected_model["best_iteration"],
    )

    permutation_rows = permutation_importance(final_booster, x_test, y_test, FULL_BASELINE_FEATURES)
    gain_rows = gain_importance(final_booster, FULL_BASELINE_FEATURES)
    buckets = build_feature_buckets(permutation_rows, gain_rows)
    reduced_comparisons = compare_feature_sets(
        split_data,
        selected_model["params"],
        selected_model["best_iteration"],
        buckets,
    )

    model_path = os.path.join(MODELS_DIR, "full_tier1_baseline_xgb.json")
    final_booster.save_model(model_path)

    output = {
        "task": "phase2_tier1_xgboost_importance",
        "artifact": CSV_PATH,
        "split_manifest": {
            "random_state": RANDOM_STATE,
            "train_count": len(split_data["train"]),
            "validation_count": len(split_data["validation"]),
            "test_count": len(split_data["test"]),
        },
        "full_baseline_features": FULL_BASELINE_FEATURES,
        "validation_selection": validation_summary,
        "full_baseline_test_metrics": final_metrics,
        "importance_scope_note": "Feature importance is conditional on this preprocessing chain, this XGBoost model family, and this fixed split.",
        "permutation_importance": permutation_rows,
        "gain_importance": gain_rows,
        "feature_buckets": buckets,
        "reduced_set_comparisons": reduced_comparisons,
        "saved_model_path": model_path,
    }
    write_json(os.path.join(RESULTS_DIR, "phase2_tier1_xgb_importance.json"), output)
    return output


if __name__ == "__main__":
    results = run_workflow()
    print(json.dumps(results, indent=2))
