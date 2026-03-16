"""Phase 2 Tier 3 Experiment C: compact-feature robustness to perturbations."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from phase2_tier2_common import FEATURE_GROUPS, baseline_reference_payload, create_context, run_baseline_parity_check, write_csv, write_json
from phase2_tier3_model_compare import (
    PLOTS_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    maybe_import_matplotlib,
    train_and_evaluate_model,
)


CSV_PATH = f"{RESULTS_DIR}/noise_test_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/noise_test_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/noise_test_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier3_noise_test.png"


def standardize_reference(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def feature_index_map(feature_names: list[str]) -> dict[str, int]:
    return {feature_name: index for index, feature_name in enumerate(feature_names)}


def mask_features_to_train_median(x_values: np.ndarray, train_x: np.ndarray, indices: list[int]) -> np.ndarray:
    modified = x_values.copy()
    medians = np.median(train_x, axis=0)
    for index in indices:
        modified[:, index] = medians[index]
    return modified


def add_gaussian_noise(x_values: np.ndarray, std_reference: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=scale, size=x_values.shape).astype(np.float32)
    return x_values + noise * std_reference


def make_sparse_observation_proxy(
    x_values: np.ndarray,
    train_x: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
    mask_fraction: float = 0.45,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    modified = x_values.copy()
    medians = np.median(train_x, axis=0)
    relevant = FEATURE_GROUPS["variability"] + FEATURE_GROUPS["temporal"]
    index_lookup = feature_index_map(feature_names)
    for feature_name in relevant:
        feature_index = index_lookup[feature_name]
        mask = rng.random(len(modified)) < mask_fraction
        modified[mask, feature_index] = medians[feature_index]
    return modified


def make_shortened_time_coverage(x_values: np.ndarray, train_x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    modified = x_values.copy()
    lookup = feature_index_map(feature_names)
    span_index = lookup["time_span"]
    modified[:, span_index] *= 0.6
    medians = np.median(train_x, axis=0)
    for feature_name in FEATURE_GROUPS["temporal"]:
        feature_index = lookup[feature_name]
        if feature_name == "time_span":
            continue
        modified[:, feature_index] = 0.5 * modified[:, feature_index] + 0.5 * medians[feature_index]
    return modified


def write_markdown(path: str, lines: list[str]) -> None:
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_results(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    labels = [row["scenario_label"] for row in rows]
    deltas = [row["delta_f1"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, deltas, color="#b33f62")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("delta F1 vs frozen baseline")
    ax.set_title("Phase 2 Tier 3 Noise and Missing-Data Stress Tests")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    feature_names = context.compact_features
    train_x = np.array([[row[name] for name in feature_names] for row in context.split_data["trainval"]], dtype=np.float32)
    train_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in context.split_data["trainval"]], dtype=np.int32)
    test_x = np.array([[row[name] for name in feature_names] for row in context.split_data["test"]], dtype=np.float32)
    test_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in context.split_data["test"]], dtype=np.int32)

    mean_ref, std_ref = standardize_reference(train_x)
    _ = mean_ref
    lookup = feature_index_map(feature_names)
    z_band_indices = [lookup[name] for name in ("z_peak_flux", "peak_color_i_minus_z", "z_std_flux", "z_time_of_peak")]

    scenarios = [
        ("baseline_compact", "No perturbation", test_x),
        ("sparse_observation_proxy", "Reduced observations", make_sparse_observation_proxy(test_x, train_x, feature_names, seed=42)),
        ("flux_noise_0p25sigma", "Flux noise (+0.25 sigma)", add_gaussian_noise(test_x, std_ref, 0.25, 52)),
        ("flux_noise_0p50sigma", "Flux noise (+0.50 sigma)", add_gaussian_noise(test_x, std_ref, 0.50, 62)),
        ("drop_z_band", "Remove z-band proxies", mask_features_to_train_median(test_x, train_x, z_band_indices)),
        ("shortened_time_coverage", "Shortened time coverage", make_shortened_time_coverage(test_x, train_x, feature_names)),
    ]

    rows = []
    detailed_rows = []
    for scenario_name, scenario_label, scenario_test_x in scenarios:
        result = train_and_evaluate_model(
            "xgboost",
            train_x,
            train_y,
            scenario_test_x,
            test_y,
            feature_names,
            seed=42,
        )
        delta_f1 = result["metrics"]["f1"] - context.baseline_metrics["f1"]
        delta_roc_auc = result["metrics"]["roc_auc"] - context.baseline_metrics["roc_auc"]
        delta_pr_auc = result["metrics"]["pr_auc"] - context.baseline_metrics["pr_auc"]
        row = {
            "scenario_name": scenario_name,
            "scenario_label": scenario_label,
            "f1": result["metrics"]["f1"],
            "roc_auc": result["metrics"]["roc_auc"],
            "pr_auc": result["metrics"]["pr_auc"],
            "delta_f1": delta_f1,
            "delta_roc_auc": delta_roc_auc,
            "delta_pr_auc": delta_pr_auc,
        }
        rows.append(row)
        detailed_rows.append({**row, "permutation_importance": result["permutation_importance"]})

    plot_results(rows)
    write_csv(
        CSV_PATH,
        ["scenario_name", "scenario_label", "f1", "roc_auc", "pr_auc", "delta_f1", "delta_roc_auc", "delta_pr_auc"],
        rows,
    )

    payload = {
        "experiment": "noise_test",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "rows": rows,
        "detailed_rows": detailed_rows,
        "notes": {
            "method": "Perturbations are applied in compact-feature space as survey-condition proxies.",
            "warning": "These tests do not re-extract features from degraded raw light curves.",
        },
        "plot_path": PLOT_PATH,
    }
    write_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 3 Noise and Missing-Data Tests",
        "",
        "Compact-feature perturbation proxies used to stress-test the frozen XGBoost baseline.",
        "",
        "| scenario | f1 | pr_auc | delta_f1 | delta_pr_auc |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_label']} | {row['f1']:.6f} | {row['pr_auc']:.6f} | {row['delta_f1']:+.6f} | {row['delta_pr_auc']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "These perturbations are feature-space proxies rather than full raw-light-curve reprocessing.",
            f"Plot: `{PLOT_PATH}`",
        ]
    )
    write_markdown(SUMMARY_PATH, lines)

    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()
