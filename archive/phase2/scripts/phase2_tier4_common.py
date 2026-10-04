"""Shared utilities for Phase 2 Tier 4 domain-generalization experiments."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from phase2_tier1_benchmarks import RANDOM_STATE
from phase2_tier2_common import (
    COMPACT_FEATURES,
    FEATURE_GROUPS,
    baseline_reference_payload,
    build_matrix,
    create_context,
    feature_to_group,
    load_interpretation_rows,
    parse_numeric_rows,
    read_csv_rows,
    run_baseline_parity_check,
    select_ranked_core_subsets,
    write_csv,
    write_json,
)
from phase2_tier3_model_compare import maybe_import_matplotlib, train_and_evaluate_model


RESULTS_DIR = "results/phase2_tier4"
PLOTS_DIR = "plots/phase2_tier4"

SPCC_FEATURES_DIR = "data/spcc/features"
SPCC_TIER4_VARIANTS_DIR = "data/spcc/tier4_variants"
PLASTICC_FEATURES_DIR = "data/PLAsTiCC/features"

SPCC_COMPACT_CSV_PATH = f"{SPCC_FEATURES_DIR}/compact_features.csv"
SPCC_NOISE_CSV_PATH = f"{SPCC_TIER4_VARIANTS_DIR}/noise/compact_features_noise.csv"
SPCC_NO_Z_CSV_PATH = f"{SPCC_TIER4_VARIANTS_DIR}/no_z/compact_features_no_z.csv"
SPCC_NO_I_CSV_PATH = f"{SPCC_TIER4_VARIANTS_DIR}/no_i/compact_features_no_i.csv"
SPCC_SHORT_SPAN_CSV_PATH = f"{SPCC_TIER4_VARIANTS_DIR}/short_span/compact_features_short_span.csv"
SPCC_SCALED_FLUX_CSV_PATH = f"{SPCC_TIER4_VARIANTS_DIR}/flux_scale/compact_features_scaled_flux.csv"
PLASTICC_TRAIN_COMPACT_CSV_PATH = f"{PLASTICC_FEATURES_DIR}/train_compact_features.csv"
PLASTICC_TEST_COMPACT_CSV_PATH = f"{PLASTICC_FEATURES_DIR}/test_compact_features.csv"

SPCC_DOMAIN_ORDER = [
    "spcc",
    "noise",
    "no_z",
    "no_i",
    "short_span",
    "flux_scale",
]
DOMAIN_ORDER = [
    "spcc",
    "noise",
    "no_z",
    "no_i",
    "short_span",
    "flux_scale",
    "plasticc",
]
SHIFT_ORDER = ["noise", "no_z", "no_i", "short_span", "flux_scale"]

MODEL_NAME = "xgboost"
SUBSET_SIZES = [5, 8, 10]
TOP_K_IMPORTANCE = 5
FEATURE_ABLATION_CSV_PATH = "results/phase2_tier2/feature_ablation_metrics.csv"
TEMPORAL_PEAK_FEATURES = ["z_time_of_peak", "i_time_of_peak", "r_time_of_peak"]

BRIGHTNESS_FEATURES = list(FEATURE_GROUPS["brightness"])
VARIABILITY_FEATURES = list(FEATURE_GROUPS["variability"])
TEMPORAL_FEATURES = list(FEATURE_GROUPS["temporal"])
Z_RELATED_FEATURES = ["z_peak_flux", "z_std_flux", "z_time_of_peak", "peak_color_i_minus_z"]
I_RELATED_FEATURES = ["i_peak_flux", "i_std_flux", "i_amplitude", "i_time_of_peak", "peak_color_r_minus_i", "peak_color_i_minus_z"]

VARIANT_PATHS = {
    "spcc": SPCC_COMPACT_CSV_PATH,
    "noise": SPCC_NOISE_CSV_PATH,
    "no_z": SPCC_NO_Z_CSV_PATH,
    "no_i": SPCC_NO_I_CSV_PATH,
    "short_span": SPCC_SHORT_SPAN_CSV_PATH,
    "flux_scale": SPCC_SCALED_FLUX_CSV_PATH,
}


def ensure_output_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def ensure_variant_dirs() -> None:
    os.makedirs(SPCC_FEATURES_DIR, exist_ok=True)
    for subdir in ("noise", "no_z", "no_i", "short_span", "flux_scale"):
        os.makedirs(os.path.join(SPCC_TIER4_VARIANTS_DIR, subdir), exist_ok=True)
    os.makedirs(PLASTICC_FEATURES_DIR, exist_ok=True)


def write_markdown(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def maybe_import_shap():
    try:
        import shap  # type: ignore
    except ModuleNotFoundError:
        return None
    return shap


def load_feature_ablation_rows() -> list[dict[str, Any]]:
    if not os.path.exists(FEATURE_ABLATION_CSV_PATH):
        raise FileNotFoundError(
            "Tier 4 minimal-domain testing depends on Tier 2 feature ablation output. "
            "Run phase2_tier2_feature_ablation.py first."
        )
    return parse_numeric_rows(read_csv_rows(FEATURE_ABLATION_CSV_PATH))


def compact_fieldnames() -> list[str]:
    return ["snid", "label_name", "label_id", "sim_z", *COMPACT_FEATURES]


def parse_feature_csv(path: str) -> list[dict[str, Any]]:
    rows = []
    for raw_row in read_csv_rows(path):
        parsed: dict[str, Any] = {}
        for key, value in raw_row.items():
            if key == "label_name":
                parsed[key] = value
            elif key in {"snid", "label_id"}:
                parsed[key] = int(float(value))
            else:
                parsed[key] = float(value)
        rows.append(parsed)
    return rows


def sort_rows_by_snid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row["snid"]))


def domain_reference_stats(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for feature_name in feature_names:
        values = np.array([float(row[feature_name]) for row in rows], dtype=np.float32)
        stats[feature_name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std() if values.std() > 0.0 else 1.0),
        }
    return stats


def _copy_row(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def _add_feature_noise(
    row: dict[str, Any],
    feature_names: list[str],
    stats: dict[str, dict[str, float]],
    rng: np.random.Generator,
    scale: float,
) -> dict[str, Any]:
    updated = _copy_row(row)
    for feature_name in feature_names:
        feature_scale = scale
        if feature_name in BRIGHTNESS_FEATURES:
            feature_scale *= 1.4
        elif feature_name in VARIABILITY_FEATURES:
            feature_scale *= 1.25
        elif feature_name in TEMPORAL_FEATURES:
            feature_scale *= 1.10
        updated[feature_name] = float(updated[feature_name] + rng.normal(0.0, feature_scale * stats[feature_name]["std"]))
    return updated


def _collapse_to_reference(
    row: dict[str, Any],
    feature_names: list[str],
    stats: dict[str, dict[str, float]],
    rng: np.random.Generator,
    *,
    jitter_scale: float = 0.05,
) -> dict[str, Any]:
    updated = _copy_row(row)
    for feature_name in feature_names:
        updated[feature_name] = float(
            stats[feature_name]["median"] + rng.normal(0.0, jitter_scale * stats[feature_name]["std"])
        )
    return updated


def _scale_flux(row: dict[str, Any], rng: np.random.Generator, *, low: float, high: float) -> dict[str, Any]:
    updated = _copy_row(row)
    scale = float(rng.uniform(low, high))
    for feature_name in BRIGHTNESS_FEATURES:
        updated[feature_name] = float(updated[feature_name] * scale)
    return updated


def build_spcc_variant_rows(base_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    feature_names = list(COMPACT_FEATURES)
    stats = domain_reference_stats(base_rows, feature_names)

    noise_rng = np.random.default_rng(RANDOM_STATE + 400)
    no_z_rng = np.random.default_rng(RANDOM_STATE + 401)
    no_i_rng = np.random.default_rng(RANDOM_STATE + 402)
    flux_rng = np.random.default_rng(RANDOM_STATE + 403)

    variants = {"spcc": [_copy_row(row) for row in base_rows]}
    variants["noise"] = [_add_feature_noise(row, feature_names, stats, noise_rng, 0.12) for row in base_rows]
    variants["no_z"] = [_collapse_to_reference(row, Z_RELATED_FEATURES, stats, no_z_rng) for row in base_rows]
    variants["no_i"] = [_collapse_to_reference(row, I_RELATED_FEATURES, stats, no_i_rng) for row in base_rows]

    short_rows = []
    for row in base_rows:
        updated = _copy_row(row)
        updated["time_span"] = float(updated["time_span"] * 0.50)
        for feature_name in ["r_time_of_peak", "i_time_of_peak", "z_time_of_peak"]:
            updated[feature_name] = float(stats[feature_name]["median"] + 0.60 * (updated[feature_name] - stats[feature_name]["median"]))
        short_rows.append(updated)
    variants["short_span"] = short_rows
    variants["flux_scale"] = [_scale_flux(row, flux_rng, low=0.75, high=1.25) for row in base_rows]
    return variants


def write_feature_rows(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fieldnames())
        writer.writeheader()
        for row in sort_rows_by_snid(rows):
            writer.writerow({name: row[name] for name in compact_fieldnames()})


def expected_variant_paths() -> dict[str, str]:
    return dict(VARIANT_PATHS)


def available_domain_paths() -> dict[str, str]:
    return {domain_name: path for domain_name, path in VARIANT_PATHS.items() if os.path.exists(path)}


def load_variant_rows(require_plasticc: bool = False) -> dict[str, list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    for domain_name, path in VARIANT_PATHS.items():
        if not os.path.exists(path):
            if domain_name == "plasticc" and not require_plasticc:
                continue
            if domain_name != "plasticc":
                raise FileNotFoundError(
                    f"Missing Tier 4 variant file for {domain_name}: {path}. "
                    "Run phase2_tier4_make_variants.py first."
                )
            raise FileNotFoundError(f"Missing required Tier 4 feature table: {path}")
        loaded[domain_name] = parse_feature_csv(path)
    plasticc_train_path = Path(PLASTICC_TRAIN_COMPACT_CSV_PATH)
    plasticc_test_path = Path(PLASTICC_TEST_COMPACT_CSV_PATH)
    if plasticc_train_path.exists():
        loaded["plasticc_train"] = parse_feature_csv(str(plasticc_train_path))
        if plasticc_test_path.exists():
            loaded["plasticc_test"] = parse_feature_csv(str(plasticc_test_path))
    elif require_plasticc:
        raise FileNotFoundError(
            "Missing PLAsTiCC training compact feature table under data/PLAsTiCC/features/."
        )
    return loaded


def split_snid_lookup(context: Any) -> dict[str, set[int]]:
    return {
        split_name: {int(row["snid"]) for row in rows}
        for split_name, rows in context.split_data.items()
        if split_name in {"train", "validation", "trainval", "test"}
    }


def split_rows_like_context(context: Any, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup = split_snid_lookup(context)
    split_rows = {split_name: [] for split_name in lookup}
    for row in rows:
        snid = int(row["snid"])
        for split_name, snids in lookup.items():
            if snid in snids:
                split_rows[split_name].append(row)
    return split_rows


def split_external_rows(rows: list[dict[str, Any]], *, test_fraction: float = 0.2, seed: int = RANDOM_STATE) -> dict[str, list[dict[str, Any]]]:
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    indices = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for label in np.unique(labels):
        label_indices = indices[labels == label]
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        test_count = int(round(len(shuffled) * test_fraction))
        test_count = min(max(test_count, 1), max(len(shuffled) - 1, 1))
        test_idx.extend(shuffled[:test_count])
        train_idx.extend(shuffled[test_count:])
    return {
        "trainval": [rows[index] for index in sorted(train_idx)],
        "test": [rows[index] for index in sorted(test_idx)],
    }


def domain_splits_from_variants(context: Any, variant_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    splits = {}
    for domain_name, rows in variant_rows.items():
        if domain_name == "plasticc_train":
            splits[domain_name] = split_external_rows(rows)
        elif domain_name == "plasticc_test":
            splits[domain_name] = {"test": rows}
        else:
            splits[domain_name] = split_rows_like_context(context, rows)
    if "plasticc_train" in splits and "plasticc_test" in splits:
        splits["plasticc"] = {
            "trainval": splits["plasticc_train"]["trainval"],
            "test": splits["plasticc_test"]["test"],
        }
    elif "plasticc_train" in splits:
        splits["plasticc"] = splits["plasticc_train"]
    return splits


def train_test_arrays(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x, train_y = build_matrix(train_rows, feature_names)
    test_x, test_y = build_matrix(test_rows, feature_names)
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)

    name_to_index = {name: index for index, name in enumerate(feature_names)}
    if "time_span" in name_to_index:
        span_index = name_to_index["time_span"]
        train_span = np.maximum(train_x[:, span_index], 1e-6)
        test_span = np.maximum(test_x[:, span_index], 1e-6)

        for feature_name in TEMPORAL_PEAK_FEATURES:
            if feature_name in name_to_index:
                feature_index = name_to_index[feature_name]
                train_x[:, feature_index] = np.clip(train_x[:, feature_index] / train_span, 0.0, 1.0)
                test_x[:, feature_index] = np.clip(test_x[:, feature_index] / test_span, 0.0, 1.0)

        train_x[:, span_index] = np.log10(1.0 + train_span)
        test_x[:, span_index] = np.log10(1.0 + test_span)
    return train_x, train_y, test_x, test_y


def delta_from_baseline(metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "delta_f1": float(metrics["f1"] - baseline_metrics["f1"]),
        "delta_pr_auc": float(metrics["pr_auc"] - baseline_metrics["pr_auc"]),
        "delta_roc_auc": float(metrics["roc_auc"] - baseline_metrics["roc_auc"]),
    }


def run_domain_experiment(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
    baseline_metrics: dict[str, float],
    *,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    train_x, train_y, test_x, test_y = train_test_arrays(train_rows, test_rows, feature_names)
    if len(train_y) == 0 or len(test_y) == 0:
        raise ValueError("Tier 4 domain experiment received an empty train or test split.")
    if len(np.unique(train_y)) < 2:
        raise ValueError("Tier 4 domain experiment needs both classes in the training split.")
    if len(np.unique(test_y)) < 2:
        raise ValueError("Tier 4 domain experiment needs both classes in the test split.")
    result = train_and_evaluate_model(MODEL_NAME, train_x, train_y, test_x, test_y, feature_names, seed=seed)
    return {**result, **delta_from_baseline(result["metrics"], baseline_metrics)}


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(mean(float(row[key]) for row in rows)) if rows else 0.0


def load_subset_specs(context: Any) -> list[dict[str, Any]]:
    interpretation_rows = load_interpretation_rows()
    feature_ablation_rows = load_feature_ablation_rows()
    subsets = select_ranked_core_subsets(
        context.compact_features,
        feature_ablation_rows,
        interpretation_rows,
        SUBSET_SIZES,
    )
    subsets.append(
        {
            "subset_name": "compact",
            "selection_rule": "frozen Tier-3 compact baseline",
            "feature_names": list(context.compact_features),
        }
    )
    return subsets


def ablation_importance_rows(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
    baseline_metrics: dict[str, float],
    *,
    seed: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    rows = []
    for feature_name in feature_names:
        reduced = [name for name in feature_names if name != feature_name]
        result = run_domain_experiment(train_rows, test_rows, reduced, baseline_metrics, seed=seed)
        rows.append(
            {
                "feature": feature_name,
                "feature_group": feature_to_group(feature_name),
                "delta_f1": float(result["metrics"]["f1"] - baseline_metrics["f1"]),
                "delta_pr_auc": float(result["metrics"]["pr_auc"] - baseline_metrics["pr_auc"]),
                "ablation_score": float(
                    max(baseline_metrics["f1"] - result["metrics"]["f1"], 0.0)
                    + 0.35 * max(baseline_metrics["pr_auc"] - result["metrics"]["pr_auc"], 0.0)
                ),
            }
        )
    rows.sort(key=lambda row: row["ablation_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def shap_importance_rows(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    seed: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    shap = maybe_import_shap()
    if shap is None:
        return []
    train_x, train_y, test_x, test_y = train_test_arrays(train_rows, test_rows, feature_names)
    result = train_and_evaluate_model(MODEL_NAME, train_x, train_y, test_x, test_y, feature_names, seed=seed)
    booster = result["model_bundle"].get("booster")
    if booster is None:
        return []
    train_mean = result["model_bundle"]["train_mean"]
    train_std = result["model_bundle"]["train_std"]
    x_scaled = (test_x - train_mean) / train_std
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x_scaled)
    mean_abs = np.mean(np.abs(np.asarray(shap_values)), axis=0)
    order = np.argsort(-mean_abs)
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append(
            {
                "feature": feature_names[index],
                "feature_group": feature_to_group(feature_names[index]),
                "score": float(mean_abs[index]),
                "rank": rank,
            }
        )
    return rows


def mean_rank_from_methods(method_rows: dict[str, list[dict[str, Any]]], feature_names: list[str]) -> list[dict[str, Any]]:
    aggregate_rows = []
    for feature_name in feature_names:
        per_method_ranks = []
        for rows in method_rows.values():
            rank_lookup = {row["feature"]: int(row["rank"]) for row in rows}
            if feature_name in rank_lookup:
                per_method_ranks.append(rank_lookup[feature_name])
        aggregate_rows.append(
            {
                "feature": feature_name,
                "feature_group": feature_to_group(feature_name),
                "mean_rank": float(np.mean(per_method_ranks)) if per_method_ranks else float(len(feature_names)),
                "method_count": len(per_method_ranks),
            }
        )
    aggregate_rows.sort(key=lambda row: row["mean_rank"])
    for rank, row in enumerate(aggregate_rows, start=1):
        row["aggregate_rank"] = rank
    return aggregate_rows


def pairwise_overlap_score(feature_sets: list[list[str]]) -> float:
    if len(feature_sets) < 2:
        return 1.0
    overlaps = []
    for left_index in range(len(feature_sets)):
        for right_index in range(left_index + 1, len(feature_sets)):
            left = set(feature_sets[left_index])
            right = set(feature_sets[right_index])
            overlaps.append(len(left & right) / max(len(left | right), 1))
    return float(np.mean(overlaps)) if overlaps else 1.0


def tier4_reference_payload(context: Any) -> dict[str, Any]:
    return {
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "variant_paths": available_domain_paths(),
        "domain_strategy": "SPCC compact baseline plus on-disk Tier 4 variants and optional PLAsTiCC compact features",
    }


def plot_grouped_bars(
    rows: list[dict[str, Any]],
    labels: list[str],
    value_keys: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
    output_path: str,
) -> str:
    plt = maybe_import_matplotlib()
    x_values = np.arange(len(labels))
    width = 0.8 / max(len(value_keys), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (key, color) in enumerate(zip(value_keys, colors)):
        offset = (index - (len(value_keys) - 1) / 2) * width
        ax.bar(x_values + offset, [row[key] for row in rows], width=width, color=color, label=key)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_json(path: str, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def save_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    write_csv(path, fieldnames, rows)


def file_label(path: str) -> str:
    return str(Path(path))
