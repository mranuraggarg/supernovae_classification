"""Phase 2 Tier 3 Experiment E: minimal core generalization under new conditions."""

from __future__ import annotations

import json
import os
from statistics import mean
from typing import Any

import numpy as np

from phase2_tier1_benchmarks import stratified_split_indices
from phase2_tier2_common import (
    baseline_reference_payload,
    create_context,
    load_interpretation_rows,
    parse_numeric_rows,
    read_csv_rows,
    run_baseline_parity_check,
    select_ranked_core_subsets,
    write_csv,
    write_json,
)
from phase2_tier3_model_compare import PLOTS_DIR, RESULTS_DIR, ensure_output_dirs, maybe_import_matplotlib, train_and_evaluate_model


CSV_PATH = f"{RESULTS_DIR}/minimal_generalization_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/minimal_generalization_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/minimal_generalization_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier3_minimal_generalization.png"

TIER2_ABLATION_CSV_PATH = "results/phase2_tier2/feature_ablation_metrics.csv"
SUBSET_SIZES = [5, 8, 10]
ALT_SPLIT_SEEDS = [42, 52, 62]
MODEL_NAMES = ["xgboost", "random_forest", "logistic_regression"]


def load_feature_ablation_rows() -> list[dict[str, Any]]:
    if not os.path.exists(TIER2_ABLATION_CSV_PATH):
        raise FileNotFoundError(
            "Minimal-core generalization depends on Tier-2 feature ablation output. "
            "Run phase2_tier2_feature_ablation.py first."
        )
    return parse_numeric_rows(read_csv_rows(TIER2_ABLATION_CSV_PATH))


def add_noise(x_values: np.ndarray, train_x: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    noise = rng.normal(0.0, scale, size=x_values.shape).astype(np.float32)
    return x_values + noise * std


def write_markdown(path: str, lines: list[str]) -> None:
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_summary(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    subset_names = [row["subset_name"] for row in rows]
    mean_f1 = [row["mean_f1"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(subset_names, mean_f1, color="#1d4e89")
    ax.set_ylabel("Mean F1 across stress tests")
    ax.set_title("Phase 2 Tier 3 Minimal Core Generalization")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    interpretation_rows = load_interpretation_rows()
    feature_ablation_rows = load_feature_ablation_rows()
    subset_specs = select_ranked_core_subsets(context.compact_features, feature_ablation_rows, interpretation_rows, SUBSET_SIZES)

    detailed_rows = []
    summary_rows = []
    label_array = np.array([1 if row["label_name"] == "Ia" else 0 for row in context.rows], dtype=np.int32)

    for subset_spec in subset_specs:
        subset_name = subset_spec["subset_name"]
        feature_names = subset_spec["feature_names"]
        subset_run_metrics = []
        for seed in ALT_SPLIT_SEEDS:
            train_idx, test_idx = stratified_split_indices(label_array, 0.2, seed)
            train_rows = [context.rows[index] for index in train_idx]
            test_rows = [context.rows[index] for index in test_idx]
            train_x = np.array([[row[name] for name in feature_names] for row in train_rows], dtype=np.float32)
            train_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in train_rows], dtype=np.int32)
            test_x = np.array([[row[name] for name in feature_names] for row in test_rows], dtype=np.float32)
            test_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in test_rows], dtype=np.int32)

            scenarios = [
                ("clean", test_x),
                ("noisy", add_noise(test_x, train_x, 0.30, seed + len(feature_names))),
            ]
            for model_name in MODEL_NAMES:
                for scenario_name, scenario_x in scenarios:
                    result = train_and_evaluate_model(
                        model_name,
                        train_x,
                        train_y,
                        scenario_x,
                        test_y,
                        feature_names,
                        seed=seed,
                    )
                    detailed_rows.append(
                        {
                            "subset_name": subset_name,
                            "feature_count": len(feature_names),
                            "feature_list": ", ".join(feature_names),
                            "selection_rule": subset_spec["selection_rule"],
                            "seed": seed,
                            "model_name": model_name,
                            "scenario_name": scenario_name,
                            "f1": result["metrics"]["f1"],
                            "roc_auc": result["metrics"]["roc_auc"],
                            "pr_auc": result["metrics"]["pr_auc"],
                        }
                    )
                    subset_run_metrics.append(result["metrics"]["f1"])

        summary_rows.append(
            {
                "subset_name": subset_name,
                "feature_count": len(feature_names),
                "feature_list": ", ".join(feature_names),
                "selection_rule": subset_spec["selection_rule"],
                "mean_f1": mean(subset_run_metrics),
                "min_f1": min(subset_run_metrics),
                "max_f1": max(subset_run_metrics),
            }
        )

    summary_rows.sort(key=lambda row: row["feature_count"])
    plot_summary(summary_rows)

    write_csv(
        CSV_PATH,
        [
            "subset_name",
            "feature_count",
            "feature_list",
            "selection_rule",
            "seed",
            "model_name",
            "scenario_name",
            "f1",
            "roc_auc",
            "pr_auc",
        ],
        detailed_rows,
    )

    payload = {
        "experiment": "minimal_generalization",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "summary_rows": summary_rows,
        "detailed_rows": detailed_rows,
        "plot_path": PLOT_PATH,
    }
    write_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 3 Minimal Core Generalization",
        "",
        "Tier-2 reduced-core subsets retested across alternative splits, models, and a compact-space noise stress test.",
        "",
        "| subset | feature_count | mean_f1 | min_f1 | max_f1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['subset_name']} | {row['feature_count']} | {row['mean_f1']:.6f} | {row['min_f1']:.6f} | {row['max_f1']:.6f} |"
        )
    lines.extend(["", f"Plot: `{PLOT_PATH}`"])
    write_markdown(SUMMARY_PATH, lines)

    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()
