"""Phase 2 Tier 3 Experiment A: model robustness on compact features.

This module also provides shared utilities used by the other Tier 3 scripts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from phase2_tier1_benchmarks import RANDOM_STATE, TEST_SPLIT, VALIDATION_SPLIT, stratified_split_indices
from phase2_tier1_xgb_importance import XGB_PARAM_GRID
from phase2_tier2_common import (
    baseline_reference_payload,
    binary_metrics,
    build_matrix,
    compute_delta_metrics,
    create_context,
    feature_to_group,
    format_feature_list,
    run_baseline_parity_check,
    write_csv,
    write_json,
)


RESULTS_DIR = "results/phase2_tier3"
PLOTS_DIR = "plots/phase2_tier3"

MODEL_COMPARE_CSV_PATH = f"{RESULTS_DIR}/model_compare_metrics.csv"
MODEL_COMPARE_JSON_PATH = f"{RESULTS_DIR}/model_compare_metrics.json"
MODEL_COMPARE_SUMMARY_PATH = f"{RESULTS_DIR}/model_compare_summary.md"

MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression", "svm_rbf"]
TOP_K_IMPORTANCE = 5


@dataclass
class SplitBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    trainval_x: np.ndarray
    trainval_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    feature_names: list[str]


def ensure_output_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate Tier 3 plots. Install project dependencies and rerun."
        ) from exc
    return plt


def standardize_from_train(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def split_train_validation(trainval_x: np.ndarray, trainval_y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx, validation_idx = stratified_split_indices(trainval_y, VALIDATION_SPLIT, seed)
    return train_idx, validation_idx


def rows_to_xy(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x, y = build_matrix(rows, feature_names)
    return x.astype(np.float32), y.astype(np.int32)


def split_bundle_from_context(context: Any, feature_names: list[str]) -> SplitBundle:
    train_x, train_y = rows_to_xy(context.split_data["train"], feature_names)
    validation_x, validation_y = rows_to_xy(context.split_data["validation"], feature_names)
    trainval_x, trainval_y = rows_to_xy(context.split_data["trainval"], feature_names)
    test_x, test_y = rows_to_xy(context.split_data["test"], feature_names)
    return SplitBundle(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        trainval_x=trainval_x,
        trainval_y=trainval_y,
        test_x=test_x,
        test_y=test_y,
        feature_names=list(feature_names),
    )


def build_train_test_split(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    test_size: float = TEST_SPLIT,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_all, y_all = rows_to_xy(rows, feature_names)
    train_idx, test_idx = stratified_split_indices(y_all, test_size, seed)
    return x_all[train_idx], y_all[train_idx], x_all[test_idx], y_all[test_idx]


def permutation_importance_scores(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    *,
    num_repeats: int = 5,
    seed: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    base_probs = predict_fn(x_eval)
    base_metrics = binary_metrics(y_eval, base_probs)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(feature_names):
        f1_drops = []
        pr_auc_drops = []
        roc_auc_drops = []
        for _ in range(num_repeats):
            shuffled = x_eval.copy()
            rng.shuffle(shuffled[:, feature_index])
            probs = predict_fn(shuffled)
            metrics = binary_metrics(y_eval, probs)
            f1_drops.append(base_metrics["f1"] - metrics["f1"])
            pr_auc_drops.append(base_metrics["pr_auc"] - metrics["pr_auc"])
            roc_auc_drops.append(base_metrics["roc_auc"] - metrics["roc_auc"])
        rows.append(
            {
                "feature": feature_name,
                "feature_group": feature_to_group(feature_name),
                "mean_f1_drop": float(np.mean(f1_drops)),
                "mean_pr_auc_drop": float(np.mean(pr_auc_drops)),
                "mean_roc_auc_drop": float(np.mean(roc_auc_drops)),
            }
        )
    rows.sort(key=lambda row: (row["mean_pr_auc_drop"], row["mean_f1_drop"]), reverse=True)
    return rows


def native_importance_rows(model_name: str, model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    if model_name == "xgboost":
        raw_gain = model.get_score(importance_type="gain")
        rows = [
            {
                "feature": feature_name,
                "feature_group": feature_to_group(feature_name),
                "score": float(raw_gain.get(feature_name, 0.0)),
            }
            for feature_name in feature_names
        ]
    elif model_name == "random_forest":
        rows = [
            {
                "feature": feature_name,
                "feature_group": feature_to_group(feature_name),
                "score": float(score),
            }
            for feature_name, score in zip(feature_names, model.feature_importances_)
        ]
    elif model_name == "logistic_regression":
        rows = [
            {
                "feature": feature_name,
                "feature_group": feature_to_group(feature_name),
                "score": float(abs(score)),
            }
            for feature_name, score in zip(feature_names, model.coef_[0])
        ]
    else:
        rows = []
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def rank_rows(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row[score_key], reverse=True)
    ranked = []
    for rank, row in enumerate(ordered, start=1):
        ranked.append({**row, "rank": rank})
    return ranked


def _xgb_base_params(seed: int, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": seed,
        "scale_pos_weight": scale_pos_weight,
    }


def _select_best_xgb(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    train_x_scaled, validation_x_scaled, _, _ = standardize_from_train(train_x, validation_x)
    positive_count = float(np.sum(train_y == 1))
    negative_count = float(np.sum(train_y == 0))
    dtrain = xgb.DMatrix(train_x_scaled, label=train_y, feature_names=feature_names)
    dvalidation = xgb.DMatrix(validation_x_scaled, label=validation_y, feature_names=feature_names)
    best: dict[str, Any] | None = None
    for params in XGB_PARAM_GRID:
        booster = xgb.train(
            params={**_xgb_base_params(seed, negative_count / max(positive_count, 1.0)), **params},
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
    return best


def _fit_final_xgb(
    trainval_x: np.ndarray,
    trainval_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
    params: dict[str, Any],
    num_boost_round: int,
) -> tuple[Any, np.ndarray]:
    trainval_x_scaled, test_x_scaled, _, _ = standardize_from_train(trainval_x, test_x)
    positive_count = float(np.sum(trainval_y == 1))
    negative_count = float(np.sum(trainval_y == 0))
    dtrainval = xgb.DMatrix(trainval_x_scaled, label=trainval_y, feature_names=feature_names)
    dtest = xgb.DMatrix(test_x_scaled, label=test_y, feature_names=feature_names)
    booster = xgb.train(
        params={**_xgb_base_params(seed, negative_count / max(positive_count, 1.0)), **params},
        dtrain=dtrainval,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    probs = booster.predict(dtest)
    return (
        {
            "model_name": "xgboost",
            "booster": booster,
            "train_mean": trainval_x.mean(axis=0),
            "train_std": np.where(trainval_x.std(axis=0) == 0.0, 1.0, trainval_x.std(axis=0)),
        },
        probs,
    )


def _fit_eval_sklearn_classifier(
    estimator_factory: Callable[[dict[str, Any]], Any],
    param_grid: list[dict[str, Any]],
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    train_x_scaled, validation_x_scaled, mean, std = standardize_from_train(train_x, validation_x)
    best: dict[str, Any] | None = None
    for params in param_grid:
        estimator = estimator_factory(params)
        estimator.fit(train_x_scaled, train_y)
        probs = estimator.predict_proba(validation_x_scaled)[:, 1]
        metrics = binary_metrics(validation_y, probs)
        candidate = {
            "params": params,
            "validation_metrics": metrics,
            "model": estimator,
            "mean": mean,
            "std": std,
        }
        if best is None or metrics["pr_auc"] > best["validation_metrics"]["pr_auc"]:
            best = candidate
    assert best is not None
    return best


def _fit_final_sklearn_classifier(
    estimator_factory: Callable[[dict[str, Any]], Any],
    params: dict[str, Any],
    trainval_x: np.ndarray,
    trainval_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, np.ndarray]:
    trainval_x_scaled, test_x_scaled, mean, std = standardize_from_train(trainval_x, test_x)
    estimator = estimator_factory(params)
    estimator.fit(trainval_x_scaled, trainval_y)
    probs = estimator.predict_proba(test_x_scaled)[:, 1]
    return (
        {
            "model_name": estimator.__class__.__name__,
            "estimator": estimator,
            "train_mean": mean,
            "train_std": std,
        },
        probs,
    )


def model_label(model_name: str) -> str:
    return {
        "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "svm_rbf": "Support Vector Machine",
    }[model_name]


def train_and_evaluate_model(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = RANDOM_STATE,
    validation_fraction: float = VALIDATION_SPLIT,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    train_idx, validation_idx = stratified_split_indices(train_y, validation_fraction, seed)
    inner_train_x = train_x[train_idx]
    inner_train_y = train_y[train_idx]
    validation_x = train_x[validation_idx]
    validation_y = train_y[validation_idx]

    if model_name == "xgboost":
        selection = _select_best_xgb(inner_train_x, inner_train_y, validation_x, validation_y, feature_names, seed=seed)
        model_bundle, probs = _fit_final_xgb(
            train_x,
            train_y,
            test_x,
            test_y,
            feature_names,
            seed=seed,
            params=selection["params"],
            num_boost_round=selection["best_iteration"],
        )
        mean = model_bundle["train_mean"]
        std = model_bundle["train_std"]

        def predict_fn(x_values: np.ndarray) -> np.ndarray:
            x_scaled = (x_values - mean) / std
            return model_bundle["booster"].predict(xgb.DMatrix(x_scaled, feature_names=feature_names))

        native_rows = native_importance_rows(model_name, model_bundle["booster"], feature_names)
        hyperparams = {
            **selection["params"],
            "num_boost_round": selection["best_iteration"],
        }
    elif model_name == "random_forest":
        param_grid = [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2},
        ]

        def factory(params: dict[str, Any]) -> RandomForestClassifier:
            return RandomForestClassifier(
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
                **params,
            )

        selection = _fit_eval_sklearn_classifier(factory, param_grid, inner_train_x, inner_train_y, validation_x, validation_y, seed=seed)
        model_bundle, probs = _fit_final_sklearn_classifier(factory, selection["params"], train_x, train_y, test_x, seed=seed)
        mean = model_bundle["train_mean"]
        std = model_bundle["train_std"]

        def predict_fn(x_values: np.ndarray) -> np.ndarray:
            x_scaled = (x_values - mean) / std
            return model_bundle["estimator"].predict_proba(x_scaled)[:, 1]

        native_rows = native_importance_rows(model_name, model_bundle["estimator"], feature_names)
        hyperparams = selection["params"]
    elif model_name == "logistic_regression":
        param_grid = [{"C": 0.1}, {"C": 1.0}, {"C": 3.0}]

        def factory(params: dict[str, Any]) -> LogisticRegression:
            return LogisticRegression(
                random_state=seed,
                class_weight="balanced",
                max_iter=3000,
                solver="lbfgs",
                **params,
            )

        selection = _fit_eval_sklearn_classifier(factory, param_grid, inner_train_x, inner_train_y, validation_x, validation_y, seed=seed)
        model_bundle, probs = _fit_final_sklearn_classifier(factory, selection["params"], train_x, train_y, test_x, seed=seed)
        mean = model_bundle["train_mean"]
        std = model_bundle["train_std"]

        def predict_fn(x_values: np.ndarray) -> np.ndarray:
            x_scaled = (x_values - mean) / std
            return model_bundle["estimator"].predict_proba(x_scaled)[:, 1]

        native_rows = native_importance_rows(model_name, model_bundle["estimator"], feature_names)
        hyperparams = selection["params"]
    elif model_name == "svm_rbf":
        param_grid = [
            {"C": 0.5, "gamma": "scale"},
            {"C": 1.0, "gamma": "scale"},
            {"C": 2.0, "gamma": "scale"},
        ]

        def factory(params: dict[str, Any]) -> SVC:
            return SVC(
                probability=True,
                class_weight="balanced",
                random_state=seed,
                kernel="rbf",
                **params,
            )

        selection = _fit_eval_sklearn_classifier(factory, param_grid, inner_train_x, inner_train_y, validation_x, validation_y, seed=seed)
        model_bundle, probs = _fit_final_sklearn_classifier(factory, selection["params"], train_x, train_y, test_x, seed=seed)
        mean = model_bundle["train_mean"]
        std = model_bundle["train_std"]

        def predict_fn(x_values: np.ndarray) -> np.ndarray:
            x_scaled = (x_values - mean) / std
            return model_bundle["estimator"].predict_proba(x_scaled)[:, 1]

        native_rows = []
        hyperparams = selection["params"]
    else:
        raise KeyError(f"Unsupported model: {model_name}")

    metrics = binary_metrics(test_y, probs)
    permutation_rows = permutation_importance_scores(predict_fn, test_x, test_y, feature_names, seed=seed)
    return {
        "model_name": model_name,
        "model_label": model_label(model_name),
        "metrics": metrics,
        "model_bundle": model_bundle,
        "hyperparameters": hyperparams,
        "native_importance": rank_rows(native_rows, "score") if native_rows else [],
        "permutation_importance": rank_rows(permutation_rows, "mean_pr_auc_drop"),
    }


def top_features_from_ranked(rows: list[dict[str, Any]], *, key: str, top_k: int = TOP_K_IMPORTANCE) -> list[str]:
    ordered = sorted(rows, key=lambda row: row[key], reverse=True)
    return [row["feature"] for row in ordered[:top_k]]


def write_markdown_summary(path: str, lines: list[str]) -> None:
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_model_compare(rows: list[dict[str, Any]]) -> str:
    plt = maybe_import_matplotlib()
    labels = [row["model_label"] for row in rows]
    f1_values = [row["f1"] for row in rows]
    pr_auc_values = [row["pr_auc"] for row in rows]
    x_values = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    ax.bar(x_values - width / 2, f1_values, width=width, color="#2a6f97", label="F1")
    ax.bar(x_values + width / 2, pr_auc_values, width=width, color="#c84c09", label="PR-AUC")
    ax.axhline(0.8442299254, color="#1f2937", linestyle="--", linewidth=1.2, label="Frozen Tier-1 F1")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Phase 2 Tier 3 Model Robustness on Compact Features")
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = f"{PLOTS_DIR}/phase2_tier3_model_compare.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def run_model_robustness_experiment() -> dict[str, Any]:
    ensure_output_dirs()
    context = create_context()
    bundle = split_bundle_from_context(context, context.compact_features)

    rows = []
    detailed_rows = []
    for model_name in MODEL_ORDER:
        result = train_and_evaluate_model(
            model_name,
            bundle.trainval_x,
            bundle.trainval_y,
            bundle.test_x,
            bundle.test_y,
            bundle.feature_names,
            seed=RANDOM_STATE,
        )
        delta_metrics = compute_delta_metrics(result["metrics"], context.baseline_metrics)
        top_perm_features = top_features_from_ranked(result["permutation_importance"], key="mean_pr_auc_drop")
        top_native_features = (
            top_features_from_ranked(result["native_importance"], key="score")
            if result["native_importance"]
            else top_perm_features
        )
        rows.append(
            {
                "model_name": model_name,
                "model_label": result["model_label"],
                "feature_count": len(bundle.feature_names),
                "f1": result["metrics"]["f1"],
                "roc_auc": result["metrics"]["roc_auc"],
                "pr_auc": result["metrics"]["pr_auc"],
                "delta_f1": delta_metrics["delta_f1"],
                "delta_roc_auc": delta_metrics["delta_roc_auc"],
                "delta_pr_auc": delta_metrics["delta_pr_auc"],
                "top_permutation_features": format_feature_list(top_perm_features),
                "top_native_features": format_feature_list(top_native_features),
            }
        )
        detailed_rows.append(
            {
                "model_name": model_name,
                "model_label": result["model_label"],
                "metrics": result["metrics"],
                "delta_metrics": delta_metrics,
                "hyperparameters": result["hyperparameters"],
                "permutation_importance": result["permutation_importance"],
                "native_importance": result["native_importance"],
            }
        )

    plot_path = plot_model_compare(rows)
    write_csv(
        MODEL_COMPARE_CSV_PATH,
        [
            "model_name",
            "model_label",
            "feature_count",
            "f1",
            "roc_auc",
            "pr_auc",
            "delta_f1",
            "delta_roc_auc",
            "delta_pr_auc",
            "top_permutation_features",
            "top_native_features",
        ],
        rows,
    )

    payload = {
        "experiment": "model_robustness",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "rows": rows,
        "detailed_rows": detailed_rows,
        "plot_path": plot_path,
    }
    write_json(MODEL_COMPARE_JSON_PATH, payload)

    best_row = max(rows, key=lambda row: row["pr_auc"])
    lines = [
        "# Phase 2 Tier 3 Model Robustness",
        "",
        "Compact Tier-1 features evaluated with multiple classifier families on the frozen split.",
        "",
        "| model | f1 | roc_auc | pr_auc | delta_f1 | top permutation features |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['f1']:.6f} | {row['roc_auc']:.6f} | {row['pr_auc']:.6f} | "
            f"{row['delta_f1']:+.6f} | {row['top_permutation_features']} |"
        )
    lines.extend(
        [
            "",
            f"Best PR-AUC model: {best_row['model_label']} ({best_row['pr_auc']:.6f}).",
            f"Plot: `{plot_path}`",
        ]
    )
    write_markdown_summary(MODEL_COMPARE_SUMMARY_PATH, lines)
    return payload


def main() -> None:
    payload = run_model_robustness_experiment()
    print(
        json.dumps(
            {
                "csv_path": MODEL_COMPARE_CSV_PATH,
                "json_path": MODEL_COMPARE_JSON_PATH,
                "summary_path": MODEL_COMPARE_SUMMARY_PATH,
                "rows": len(payload["rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
