"""Generate learning and validation-logloss curves for the Phase 2 Tier 2."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
import xgboost as xgb

from phase2_tier2_uncertainty import (
    COMPACT_CSV_PATH,
    TIER1_BENCHMARKS_PATH,
    TIER2_COMMON_PATH,
    binary_metrics,
    build_matrix,
    parse_feature_rows,
    read_csv_rows,
    read_python_assignment,
    split_rows,
    standardize,
)


TIER2_RESULTS_DIR = "results/phase2_tier2"
LEARNING_CURVE_CSV = f"{TIER2_RESULTS_DIR}/learning_curve.csv"
LEARNING_CURVE_JSON = f"{TIER2_RESULTS_DIR}/learning_curve.json"
LEARNING_CURVE_MD = f"{TIER2_RESULTS_DIR}/learning_curve.md"
LEARNING_CURVE_PNG = f"{TIER2_RESULTS_DIR}/learning_curve.png"
LEARNING_CURVE_PDF = f"{TIER2_RESULTS_DIR}/learning_curve.pdf"
VALIDATION_LOGLOSS_CSV = f"{TIER2_RESULTS_DIR}/validation_logloss_curve.csv"
VALIDATION_LOGLOSS_JSON = f"{TIER2_RESULTS_DIR}/validation_logloss_curve.json"
VALIDATION_LOGLOSS_MD = f"{TIER2_RESULTS_DIR}/validation_logloss_curve.md"
VALIDATION_LOGLOSS_PNG = f"{TIER2_RESULTS_DIR}/validation_logloss_curve.png"
VALIDATION_LOGLOSS_PDF = f"{TIER2_RESULTS_DIR}/validation_logloss_curve.pdf"

TRAIN_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
SELECTED_COMPACT_PARAMS = {
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1.0,
    "lambda": 1.0,
}
SELECTED_BEST_ITERATION = 400


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def xgb_base_params(seed: int, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": seed,
        "scale_pos_weight": scale_pos_weight,
    }


def stratified_subsample(rows: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    if fraction >= 1.0:
        return list(rows)
    rng = np.random.default_rng(seed)
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    selected_indices: list[int] = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        sample_count = max(1, int(round(len(label_indices) * fraction)))
        selected = rng.choice(label_indices, size=sample_count, replace=False)
        selected_indices.extend(int(index) for index in selected)
    return [rows[index] for index in sorted(selected_indices)]


def train_booster_with_eval(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    seed: int,
    params: dict[str, Any],
    num_boost_round: int,
) -> tuple[xgb.Booster, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_raw, y_train = build_matrix(train_rows, feature_names)
    x_validation_raw, y_validation = build_matrix(validation_rows, feature_names)
    x_train, x_validation = standardize(x_train_raw, x_validation_raw)

    positive_count = float(np.sum(y_train == 1))
    negative_count = float(np.sum(y_train == 0))
    scale_pos_weight = negative_count / max(positive_count, 1.0)
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dvalidation = xgb.DMatrix(x_validation, label=y_validation, feature_names=feature_names)
    evals_result: dict[str, Any] = {}
    booster = xgb.train(
        params={**xgb_base_params(seed, scale_pos_weight), **params},
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        evals_result=evals_result,
        verbose_eval=False,
    )
    return booster, evals_result, x_train, y_train, x_validation, y_validation


def build_learning_curve(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for fraction in TRAIN_FRACTIONS:
        subset = stratified_subsample(train_rows, fraction, seed + int(fraction * 1000))
        booster, _, x_train, y_train, x_validation, y_validation = train_booster_with_eval(
            subset,
            validation_rows,
            feature_names,
            seed=seed,
            params=SELECTED_COMPACT_PARAMS,
            num_boost_round=SELECTED_BEST_ITERATION,
        )
        train_probs = booster.predict(xgb.DMatrix(x_train, feature_names=feature_names))
        validation_probs = booster.predict(xgb.DMatrix(x_validation, feature_names=feature_names))
        train_metrics = binary_metrics(y_train, train_probs)
        validation_metrics = binary_metrics(y_validation, validation_probs)
        rows.append(
            {
                "train_fraction": fraction,
                "train_size": len(subset),
                "validation_size": len(validation_rows),
                "train_f1": train_metrics["f1"],
                "validation_f1": validation_metrics["f1"],
                "train_pr_auc": train_metrics["pr_auc"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "train_roc_auc": train_metrics["roc_auc"],
                "validation_roc_auc": validation_metrics["roc_auc"],
            }
        )
    return rows


def build_validation_logloss_curve(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    _, evals_result, *_ = train_booster_with_eval(
        train_rows,
        validation_rows,
        feature_names,
        seed=seed,
        params=SELECTED_COMPACT_PARAMS,
        num_boost_round=SELECTED_BEST_ITERATION,
    )
    rows = []
    train_logloss = evals_result["train"]["logloss"]
    validation_logloss = evals_result["validation"]["logloss"]
    for iteration, (train_loss, validation_loss) in enumerate(zip(train_logloss, validation_logloss), start=1):
        rows.append(
            {
                "iteration": iteration,
                "train_logloss": float(train_loss),
                "validation_logloss": float(validation_loss),
            }
        )
    return rows


def maybe_import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def plot_learning_curve(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    x_values = [row["train_size"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(x_values, [row["train_f1"] for row in rows], marker="o", linewidth=2.0, label="Training F1", color="#2a6f97")
    ax.plot(x_values, [row["validation_f1"] for row in rows], marker="s", linewidth=2.0, label="Validation F1", color="#c84c09")
    ax.plot(x_values, [row["train_pr_auc"] for row in rows], marker="o", linestyle="--", linewidth=1.7, label="Training PR-AUC", color="#5a9f68")
    ax.plot(x_values, [row["validation_pr_auc"] for row in rows], marker="s", linestyle="--", linewidth=1.7, label="Validation PR-AUC", color="#8b5e3c")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Score")
    ax.set_title("Learning curve for compact XGBoost model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(LEARNING_CURVE_PNG, dpi=300)
    fig.savefig(LEARNING_CURVE_PDF)
    plt.close(fig)


def plot_validation_logloss(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    x_values = [row["iteration"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x_values, [row["train_logloss"] for row in rows], linewidth=2.0, label="Training logloss", color="#2a6f97")
    ax.plot(x_values, [row["validation_logloss"] for row in rows], linewidth=2.0, label="Validation logloss", color="#c84c09")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Logloss")
    ax.set_title("XGBoost convergence for compact feature model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(VALIDATION_LOGLOSS_PNG, dpi=300)
    fig.savefig(VALIDATION_LOGLOSS_PDF)
    plt.close(fig)


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


def write_learning_curve_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Learning Curve",
        "",
        "Learning curve for the fixed compact XGBoost configuration using increasing stratified subsets of the training partition.",
        "",
        f"- Selected hyperparameter candidate: {SELECTED_COMPACT_PARAMS}",
        f"- Boosting rounds: {SELECTED_BEST_ITERATION}",
        "",
    ]
    lines.extend(
        markdown_table(
            ["train size", "train F1", "validation F1", "train PR-AUC", "validation PR-AUC"],
            [
                [
                    str(row["train_size"]),
                    format_float(row["train_f1"]),
                    format_float(row["validation_f1"]),
                    format_float(row["train_pr_auc"]),
                    format_float(row["validation_pr_auc"]),
                ]
                for row in rows
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- PNG: `{LEARNING_CURVE_PNG}`",
            f"- PDF: `{LEARNING_CURVE_PDF}`",
        ]
    )
    with open(LEARNING_CURVE_MD, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_validation_logloss_markdown(rows: list[dict[str, Any]]) -> None:
    best_validation = min(rows, key=lambda row: row["validation_logloss"])
    final = rows[-1]
    lines = [
        "# Validation Logloss Curve",
        "",
        "Training and validation logloss across XGBoost boosting rounds for the fixed compact model configuration.",
        "",
        f"- Best validation logloss: {format_float(best_validation['validation_logloss'])} at round {best_validation['iteration']}",
        f"- Final validation logloss: {format_float(final['validation_logloss'])} at round {final['iteration']}",
        "",
        f"- PNG: `{VALIDATION_LOGLOSS_PNG}`",
        f"- PDF: `{VALIDATION_LOGLOSS_PDF}`",
    ]
    with open(VALIDATION_LOGLOSS_MD, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_results_dir()
    seed = read_python_assignment(TIER1_BENCHMARKS_PATH, "RANDOM_STATE")
    test_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "TEST_SPLIT")
    validation_split = read_python_assignment(TIER1_BENCHMARKS_PATH, "VALIDATION_SPLIT")
    compact_features = read_python_assignment(TIER2_COMMON_PATH, "COMPACT_FEATURES")
    all_rows = parse_feature_rows(read_csv_rows(COMPACT_CSV_PATH))
    split_data = split_rows(all_rows, test_split=test_split, validation_split=validation_split, seed=seed)

    learning_rows = build_learning_curve(split_data["train"], split_data["validation"], compact_features, seed=seed)
    logloss_rows = build_validation_logloss_curve(split_data["train"], split_data["validation"], compact_features, seed=seed)

    write_csv(
        LEARNING_CURVE_CSV,
        [
            "train_fraction",
            "train_size",
            "validation_size",
            "train_f1",
            "validation_f1",
            "train_pr_auc",
            "validation_pr_auc",
            "train_roc_auc",
            "validation_roc_auc",
        ],
        learning_rows,
    )
    write_json(
        LEARNING_CURVE_JSON,
        {
            "artifact": "phase2_tier2_learning_curve",
            "protocol": {
                "seed": seed,
                "test_split": test_split,
                "validation_split": validation_split,
                "train_fractions": TRAIN_FRACTIONS,
                "selected_params": SELECTED_COMPACT_PARAMS,
                "boosting_rounds": SELECTED_BEST_ITERATION,
            },
            "rows": learning_rows,
            "outputs": {"png": LEARNING_CURVE_PNG, "pdf": LEARNING_CURVE_PDF, "csv": LEARNING_CURVE_CSV},
        },
    )
    write_csv(VALIDATION_LOGLOSS_CSV, ["iteration", "train_logloss", "validation_logloss"], logloss_rows)
    write_json(
        VALIDATION_LOGLOSS_JSON,
        {
            "artifact": "phase2_tier2_validation_logloss_curve",
            "protocol": {
                "seed": seed,
                "selected_params": SELECTED_COMPACT_PARAMS,
                "boosting_rounds": SELECTED_BEST_ITERATION,
            },
            "rows": logloss_rows,
            "outputs": {
                "png": VALIDATION_LOGLOSS_PNG,
                "pdf": VALIDATION_LOGLOSS_PDF,
                "csv": VALIDATION_LOGLOSS_CSV,
            },
        },
    )
    plot_learning_curve(learning_rows)
    plot_validation_logloss(logloss_rows)
    write_learning_curve_markdown(learning_rows)
    write_validation_logloss_markdown(logloss_rows)

    print(f"Wrote {LEARNING_CURVE_CSV}")
    print(f"Wrote {LEARNING_CURVE_JSON}")
    print(f"Wrote {LEARNING_CURVE_MD}")
    print(f"Wrote {LEARNING_CURVE_PNG}")
    print(f"Wrote {LEARNING_CURVE_PDF}")
    print(f"Wrote {VALIDATION_LOGLOSS_CSV}")
    print(f"Wrote {VALIDATION_LOGLOSS_JSON}")
    print(f"Wrote {VALIDATION_LOGLOSS_MD}")
    print(f"Wrote {VALIDATION_LOGLOSS_PNG}")
    print(f"Wrote {VALIDATION_LOGLOSS_PDF}")


if __name__ == "__main__":
    main()
