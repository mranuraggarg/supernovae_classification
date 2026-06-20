"""Export supporting dataset and split metadata for the Phase 2 Tier 2."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from typing import Any

import numpy as np


TIER2_RESULTS_DIR = "results/phase2_tier2"
DATASET_SUMMARY_JSON = f"{TIER2_RESULTS_DIR}/dataset_summary.json"
CLASS_BALANCE_CSV = f"{TIER2_RESULTS_DIR}/class_balance.csv"
SPLIT_CLASS_BALANCE_CSV = f"{TIER2_RESULTS_DIR}/split_class_balance.csv"
FEATURE_RANGES_CSV = f"{TIER2_RESULTS_DIR}/compact_feature_ranges.csv"
CONTEXT_RANGES_CSV = f"{TIER2_RESULTS_DIR}/dataset_context_ranges.csv"
DATASET_SUMMARY_MD = f"{TIER2_RESULTS_DIR}/dataset_summary.md"


CONTEXT_FEATURES = [
    "sim_z",
    "observation_count",
    "observed_band_count",
    "time_span",
    "total_snr",
]

RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
BASELINE_COMPACT_CSV_PATH = "data/processed/phase2_tier1_compact_baseline.csv"
FULL_TIER1_CSV_PATH = "data/processed/spcc_features_tier1.csv"

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


def ensure_results_dir() -> None:
    os.makedirs(TIER2_RESULTS_DIR, exist_ok=True)


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key == "label_name":
                parsed[key] = value
                continue
            if value in (None, ""):
                parsed[key] = value
                continue
            try:
                parsed[key] = int(value)
                continue
            except ValueError:
                pass
            try:
                parsed[key] = float(value)
                continue
            except ValueError:
                parsed[key] = value
        parsed_rows.append(parsed)
    return parsed_rows


def load_compact_rows() -> tuple[list[dict[str, Any]], str]:
    for path in [BASELINE_COMPACT_CSV_PATH, FULL_TIER1_CSV_PATH]:
        if os.path.exists(path):
            rows = parse_numeric_rows(read_csv_rows(path))
            missing_features = [name for name in COMPACT_FEATURES if name not in rows[0]]
            if missing_features:
                raise KeyError(
                    "The processed CSV is missing compact features required for the Phase 2 Tier 2 summary: "
                    + ", ".join(missing_features)
                )
            return rows, path
    raise FileNotFoundError(
        f"Expected {BASELINE_COMPACT_CSV_PATH!r} or {FULL_TIER1_CSV_PATH!r}."
    )


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_balance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["label_name"]) for row in rows)
    total = sum(counts.values())
    balance_rows = []
    for label_name, count in sorted(counts.items()):
        balance_rows.append(
            {
                "label_name": label_name,
                "count": count,
                "fraction": count / total if total else 0.0,
            }
        )
    return balance_rows


def binary_class_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ia_count = sum(str(row["label_name"]) == "Ia" for row in rows)
    total = len(rows)
    non_ia_count = total - ia_count
    return {
        "total": total,
        "ia_count": ia_count,
        "non_ia_count": non_ia_count,
        "ia_fraction": ia_count / total if total else 0.0,
        "non_ia_fraction": non_ia_count / total if total else 0.0,
    }


def split_balance_rows(split_data: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for split_name in ["train", "validation", "trainval", "test", "all_rows"]:
        summary = binary_class_counts(split_data[split_name])
        rows.append({"split": split_name, **summary})
    return rows


def stratified_split_indices(labels: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
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
    return np.array(sorted(train_indices)), np.array(sorted(test_indices))


def split_compact_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    labels = np.array([1 if row["label_name"] == "Ia" else 0 for row in rows], dtype=np.int32)
    trainval_idx, test_idx = stratified_split_indices(labels, TEST_SPLIT, RANDOM_STATE)
    train_idx, validation_idx = stratified_split_indices(labels[trainval_idx], VALIDATION_SPLIT, RANDOM_STATE)
    return {
        "all_rows": rows,
        "train": [rows[trainval_idx[index]] for index in train_idx],
        "validation": [rows[trainval_idx[index]] for index in validation_idx],
        "trainval": [rows[index] for index in trainval_idx],
        "test": [rows[index] for index in test_idx],
    }


def numeric_values(rows: list[dict[str, Any]], feature_name: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(feature_name)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return np.array(values, dtype=np.float64)


def feature_range_rows(rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    range_rows = []
    for feature_name in feature_names:
        values = numeric_values(rows, feature_name)
        if values.size == 0:
            continue
        range_rows.append(
            {
                "feature": feature_name,
                "count": int(values.size),
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
                "std": float(np.std(values, ddof=0)),
            }
        )
    return range_rows


def format_float(value: float) -> str:
    return f"{value:.6g}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_markdown_summary(
    path: str,
    payload: dict[str, Any],
    class_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    compact_range_rows: list[dict[str, Any]],
    context_range_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 2 Tier 2 Dataset Summary",
        "",
        "This artifact summarizes the data table and fixed split used by the submitted Phase 2 Tier 2 compact-feature experiments.",
        "",
        "## Sources",
        f"- Compact source CSV: `{payload['sources']['compact_source_csv']}`",
        f"- Full Tier 1 source CSV: `{payload['sources']['full_tier1_csv']}`",
        "",
        "## Overall binary class balance",
        f"- Total objects: {payload['binary_class_balance']['total']}",
        f"- Type Ia objects: {payload['binary_class_balance']['ia_count']} ({format_float(payload['binary_class_balance']['ia_fraction'])})",
        f"- Non-Ia objects: {payload['binary_class_balance']['non_ia_count']} ({format_float(payload['binary_class_balance']['non_ia_fraction'])})",
        "",
        "## Original label balance",
    ]
    lines.extend(
        markdown_table(
            ["label_name", "count", "fraction"],
            [
                [row["label_name"], str(row["count"]), format_float(row["fraction"])]
                for row in class_rows
            ],
        )
    )
    lines.extend(["", "## Fixed split balance"])
    lines.extend(
        markdown_table(
            ["split", "total", "Ia", "non-Ia", "Ia fraction"],
            [
                [
                    row["split"],
                    str(row["total"]),
                    str(row["ia_count"]),
                    str(row["non_ia_count"]),
                    format_float(row["ia_fraction"]),
                ]
                for row in split_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Compact feature set",
            f"- Compact feature count: {payload['compact_feature_count']}",
            f"- Compact features: {', '.join(payload['compact_features'])}",
            "",
            "## Compact engineered feature ranges",
        ]
    )
    lines.extend(
        markdown_table(
            ["feature", "min", "median", "max", "mean", "std"],
            [
                [
                    row["feature"],
                    format_float(row["min"]),
                    format_float(row["median"]),
                    format_float(row["max"]),
                    format_float(row["mean"]),
                    format_float(row["std"]),
                ]
                for row in compact_range_rows
            ],
        )
    )
    if context_range_rows:
        lines.extend(["", "## Dataset context ranges"])
        lines.extend(
            markdown_table(
                ["feature", "min", "median", "max", "mean", "std"],
                [
                    [
                        row["feature"],
                        format_float(row["min"]),
                        format_float(row["median"]),
                        format_float(row["max"]),
                        format_float(row["mean"]),
                        format_float(row["std"]),
                    ]
                    for row in context_range_rows
                ],
            )
        )
    lines.extend(
        [
            "",
            "Note: the feature ranges are computed from engineered features in the processed Tier 1 table. They should be described as feature ranges, not as raw survey magnitude limits.",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_results_dir()
    rows, compact_source_csv = load_compact_rows()
    split_data = split_compact_rows(rows)
    class_rows = class_balance(rows)
    split_rows = split_balance_rows(split_data)
    compact_range_rows = feature_range_rows(rows, COMPACT_FEATURES)

    full_rows = read_csv_rows(FULL_TIER1_CSV_PATH) if os.path.exists(FULL_TIER1_CSV_PATH) else []
    context_features = [name for name in CONTEXT_FEATURES if full_rows and name in full_rows[0]]
    context_range_rows = feature_range_rows(full_rows, context_features) if full_rows else []

    payload = {
        "artifact": "phase2_tier2_dataset_summary",
        "sources": {
            "compact_source_csv": compact_source_csv,
            "full_tier1_csv": FULL_TIER1_CSV_PATH if full_rows else None,
        },
        "binary_class_balance": binary_class_counts(rows),
        "original_label_balance": class_rows,
        "split_balance": split_rows,
        "split_protocol": {
            "random_state": RANDOM_STATE,
            "test_split": TEST_SPLIT,
            "validation_split_within_trainval": VALIDATION_SPLIT,
            "selection_split": "train/validation",
            "final_evaluation_split": "held-out test",
        },
        "compact_feature_count": len(COMPACT_FEATURES),
        "compact_features": COMPACT_FEATURES,
        "compact_feature_ranges": compact_range_rows,
        "dataset_context_ranges": context_range_rows,
        "notes": [
            "Split statistics use the existing Phase 2 Tier 2 stratified split protocol.",
            "Feature ranges are computed from engineered processed features, not raw survey magnitude limits.",
        ],
    }

    write_json(DATASET_SUMMARY_JSON, payload)
    write_csv(CLASS_BALANCE_CSV, ["label_name", "count", "fraction"], class_rows)
    write_csv(
        SPLIT_CLASS_BALANCE_CSV,
        ["split", "total", "ia_count", "non_ia_count", "ia_fraction", "non_ia_fraction"],
        split_rows,
    )
    write_csv(
        FEATURE_RANGES_CSV,
        ["feature", "count", "min", "q25", "median", "mean", "q75", "max", "std"],
        compact_range_rows,
    )
    write_csv(
        CONTEXT_RANGES_CSV,
        ["feature", "count", "min", "q25", "median", "mean", "q75", "max", "std"],
        context_range_rows,
    )
    write_markdown_summary(
        DATASET_SUMMARY_MD,
        payload,
        class_rows,
        split_rows,
        compact_range_rows,
        context_range_rows,
    )
    print(f"Wrote {DATASET_SUMMARY_JSON}")
    print(f"Wrote {CLASS_BALANCE_CSV}")
    print(f"Wrote {SPLIT_CLASS_BALANCE_CSV}")
    print(f"Wrote {FEATURE_RANGES_CSV}")
    print(f"Wrote {CONTEXT_RANGES_CSV}")
    print(f"Wrote {DATASET_SUMMARY_MD}")


if __name__ == "__main__":
    main()
