"""Phase 2 Tier 3 Experiment B: cross-validation and split stability."""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

import numpy as np

from phase2_tier1_benchmarks import RANDOM_STATE, stratified_split_indices
from phase2_tier2_common import baseline_reference_payload, create_context, run_baseline_parity_check, write_csv, write_json
from phase2_tier3_model_compare import PLOTS_DIR, RESULTS_DIR, ensure_output_dirs, maybe_import_matplotlib, train_and_evaluate_model


CSV_PATH = f"{RESULTS_DIR}/cv_stability_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/cv_stability_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/cv_stability_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier3_cv_stability.png"

CV_FOLDS = 5
RANDOM_SPLIT_SEEDS = [42, 52, 62, 72, 82]


def stratified_kfold_indices(labels: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    label_buckets = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    fold_buckets = [[] for _ in range(n_splits)]
    for label in sorted(label_buckets):
        indices = label_buckets[label].copy()
        rng.shuffle(indices)
        for fold_index, index in enumerate(indices):
            fold_buckets[fold_index % n_splits].append(int(index))

    splits = []
    all_indices = set(range(len(labels)))
    for fold_indices in fold_buckets:
        test_idx = np.array(sorted(fold_indices), dtype=np.int32)
        train_idx = np.array(sorted(all_indices - set(test_idx.tolist())), dtype=np.int32)
        splits.append((train_idx, test_idx))
    return splits


def summarize_runs(rows: list[dict[str, Any]], protocol_name: str) -> dict[str, Any]:
    metrics = {metric: [float(row[metric]) for row in rows] for metric in ("f1", "roc_auc", "pr_auc")}
    return {
        "protocol": protocol_name,
        "run_count": len(rows),
        "f1_mean": mean(metrics["f1"]),
        "f1_std": pstdev(metrics["f1"]) if len(metrics["f1"]) > 1 else 0.0,
        "roc_auc_mean": mean(metrics["roc_auc"]),
        "roc_auc_std": pstdev(metrics["roc_auc"]) if len(metrics["roc_auc"]) > 1 else 0.0,
        "pr_auc_mean": mean(metrics["pr_auc"]),
        "pr_auc_std": pstdev(metrics["pr_auc"]) if len(metrics["pr_auc"]) > 1 else 0.0,
    }


def write_markdown(path: str, lines: list[str]) -> None:
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_summary(summary_rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    labels = [row["protocol"] for row in summary_rows]
    means = [row["f1_mean"] for row in summary_rows]
    stds = [row["f1_std"] for row in summary_rows]
    x_values = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_values, means, yerr=stds, color="#3a7d44", capsize=6)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("F1 mean ± std")
    ax.set_title("Phase 2 Tier 3 Stability Across Resampling Protocols")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    x_all = np.array([[row[name] for name in context.compact_features] for row in context.rows], dtype=np.float32)
    y_all = np.array([1 if row["label_name"] == "Ia" else 0 for row in context.rows], dtype=np.int32)

    run_rows = []

    for fold_index, (train_idx, test_idx) in enumerate(stratified_kfold_indices(y_all, CV_FOLDS, RANDOM_STATE), start=1):
        result = train_and_evaluate_model(
            "xgboost",
            x_all[train_idx],
            y_all[train_idx],
            x_all[test_idx],
            y_all[test_idx],
            context.compact_features,
            seed=RANDOM_STATE + fold_index,
        )
        run_rows.append(
            {
                "protocol": "kfold_cv",
                "run_id": f"fold_{fold_index}",
                "seed": RANDOM_STATE + fold_index,
                **result["metrics"],
            }
        )

    for seed in RANDOM_SPLIT_SEEDS:
        train_idx, test_idx = stratified_split_indices(y_all, 0.2, seed)
        result = train_and_evaluate_model(
            "xgboost",
            x_all[train_idx],
            y_all[train_idx],
            x_all[test_idx],
            y_all[test_idx],
            context.compact_features,
            seed=seed,
        )
        run_rows.append(
            {
                "protocol": "random_split",
                "run_id": f"seed_{seed}",
                "seed": seed,
                **result["metrics"],
            }
        )

    write_csv(
        CSV_PATH,
        ["protocol", "run_id", "seed", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        run_rows,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[row["protocol"]].append(row)
    summary_rows = [summarize_runs(grouped[protocol], protocol) for protocol in ("kfold_cv", "random_split")]
    plot_summary(summary_rows)

    payload = {
        "experiment": "cv_stability",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "run_rows": run_rows,
        "summary_rows": summary_rows,
        "plot_path": PLOT_PATH,
    }
    write_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 3 Cross-Validation Stability",
        "",
        "Repeated evaluation of the compact XGBoost baseline under alternate resampling protocols.",
        "",
        "| protocol | runs | f1_mean | f1_std | roc_auc_mean | pr_auc_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['protocol']} | {row['run_count']} | {row['f1_mean']:.6f} | {row['f1_std']:.6f} | "
            f"{row['roc_auc_mean']:.6f} | {row['pr_auc_mean']:.6f} |"
        )
    lines.extend(["", f"Plot: `{PLOT_PATH}`"])
    write_markdown(SUMMARY_PATH, lines)

    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()
