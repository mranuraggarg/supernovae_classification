"""Generate ROC and PR curves for the Phase 2 Tier 1 compact 16-feature baseline."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from phase2_tier2_common import (
    COMPACT_FEATURES,
    FROZEN_BASELINE_METRICS,
    average_precision_numpy,
    binary_metrics,
    build_matrix,
    create_context,
    fit_final_model,
    select_best_model,
    standardize,
)


BASELINE_NAME = "phase2_tier1_compact_baseline"
RESULTS_DIR = "results/phase2_tier1"
PLOTS_DIR = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_plots")
CURVE_METRICS_PATH = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_curve_metrics.json")
PR_CURVE_PATH = os.path.join(PLOTS_DIR, f"{BASELINE_NAME}_precision_recall_curve.png")
ROC_CURVE_PATH = os.path.join(PLOTS_DIR, f"{BASELINE_NAME}_roc_curve.png")


def roc_curve_numpy(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    positives = max(int(np.sum(y_true == 1)), 1)
    negatives = max(int(np.sum(y_true == 0)), 1)

    tpr_values = []
    fpr_values = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int32)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)
    return np.array(fpr_values, dtype=float), np.array(tpr_values, dtype=float)


def precision_recall_curve_numpy(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted == 1)
    fp_cumsum = np.cumsum(y_sorted == 0)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
    positives = max(int(np.sum(y_true == 1)), 1)
    recall = tp_cumsum / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return recall.astype(float), precision.astype(float)


def predict_compact_baseline() -> tuple[np.ndarray, np.ndarray, dict]:
    context = create_context()
    selection = select_best_model(context.split_data["train"], context.split_data["validation"], COMPACT_FEATURES)

    x_trainval_raw, _ = build_matrix(context.split_data["trainval"], COMPACT_FEATURES)
    x_test_raw, y_test = build_matrix(context.split_data["test"], COMPACT_FEATURES)
    x_trainval, x_test = standardize(x_trainval_raw, x_test_raw)
    dtrainval = xgb.DMatrix(x_trainval, feature_names=COMPACT_FEATURES)
    dtest = xgb.DMatrix(x_test, feature_names=COMPACT_FEATURES)

    booster, metrics, _ = fit_final_model(
        context.split_data["trainval"],
        context.split_data["test"],
        COMPACT_FEATURES,
        seed=42,
        params=selection["params"],
        num_boost_round=selection["best_iteration"],
    )
    probs = booster.predict(dtest)
    return y_test, probs, {
        "selection_summary": selection,
        "test_metrics": metrics,
        "baseline_reference_metrics": context.baseline_metrics,
    }


def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, pr_auc: float) -> None:
    plt.figure(figsize=(7.5, 6))
    plt.step(recall, precision, where="post", color="#c84c09", linewidth=2.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Phase 2 Tier 1 Compact Baseline PR Curve (PR-AUC = {pr_auc:.3f})")
    plt.ylim(0.0, 1.02)
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(PR_CURVE_PATH, dpi=220)
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
    plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, color="#1d4e89", linewidth=2.0, label=f"Compact baseline (ROC-AUC = {roc_auc:.3f})")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.0, label="No-skill")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Phase 2 Tier 1 Compact Baseline ROC Curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(ROC_CURVE_PATH, dpi=220)
    plt.close()


def main() -> None:
    y_test, probs, metadata = predict_compact_baseline()
    metrics = binary_metrics(y_test, probs)
    recall, precision = precision_recall_curve_numpy(y_test, probs)
    fpr, tpr = roc_curve_numpy(y_test, probs)

    plot_pr_curve(recall, precision, metrics["pr_auc"])
    plot_roc_curve(fpr, tpr, metrics["roc_auc"])

    payload = {
        "baseline_name": BASELINE_NAME,
        "feature_count": len(COMPACT_FEATURES),
        "feature_names": COMPACT_FEATURES,
        "selection_summary": metadata["selection_summary"],
        "curve_metrics": {
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "reference_frozen_f1": FROZEN_BASELINE_METRICS["f1"],
            "reference_frozen_roc_auc": FROZEN_BASELINE_METRICS["roc_auc"],
            "reference_frozen_pr_auc": FROZEN_BASELINE_METRICS["pr_auc"],
        },
        "output_files": {
            "precision_recall_curve": PR_CURVE_PATH,
            "roc_curve": ROC_CURVE_PATH,
        },
    }
    with open(CURVE_METRICS_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
