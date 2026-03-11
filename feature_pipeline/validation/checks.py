"""Validation helpers for generated SPCC Tier 1 feature artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from statistics import median

from feature_pipeline.extraction.feature_registry import FEATURE_REGISTRY


def load_feature_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {"label_name"}:
                    parsed[key] = value
                elif key in {"snid", "label_id"}:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _feature_names() -> list[str]:
    return [spec.name for spec in FEATURE_REGISTRY]


def invalid_value_counts(rows: list[dict], feature_names: list[str]) -> dict:
    counts = {name: {"nan": 0, "inf": 0, "none": 0} for name in feature_names}
    for row in rows:
        for name in feature_names:
            value = row.get(name)
            if value is None:
                counts[name]["none"] += 1
            elif isinstance(value, float) and math.isnan(value):
                counts[name]["nan"] += 1
            elif isinstance(value, float) and math.isinf(value):
                counts[name]["inf"] += 1
    return counts


def collapsed_columns(rows: list[dict], feature_names: list[str]) -> list[str]:
    collapsed = []
    for name in feature_names:
        values = {row[name] for row in rows}
        if len(values) <= 1:
            collapsed.append(name)
    return collapsed


def per_feature_summary(rows: list[dict], feature_names: list[str]) -> dict:
    summary = {}
    for name in feature_names:
        values = sorted(row[name] for row in rows)
        summary[name] = {
            "min": values[0],
            "median": median(values),
            "max": values[-1],
            "mean": sum(values) / len(values),
        }
    return summary


def class_range_summary(rows: list[dict], feature_names: list[str]) -> dict:
    ia_rows = [row for row in rows if row["label_name"] == "Ia"]
    non_ia_rows = [row for row in rows if row["label_name"] != "Ia"]
    summary = {}
    for name in feature_names:
        ia_values = sorted(row[name] for row in ia_rows)
        non_ia_values = sorted(row[name] for row in non_ia_rows)
        summary[name] = {
            "Ia": {
                "min": ia_values[0],
                "median": median(ia_values),
                "max": ia_values[-1],
                "mean": sum(ia_values) / len(ia_values),
            },
            "non_Ia": {
                "min": non_ia_values[0],
                "median": median(non_ia_values),
                "max": non_ia_values[-1],
                "mean": sum(non_ia_values) / len(non_ia_values),
            },
        }
    return summary


def largest_median_gaps(rows: list[dict], feature_names: list[str], top_k: int = 10) -> list[dict]:
    ranges = class_range_summary(rows, feature_names)
    gaps = []
    for name, summary in ranges.items():
        ia_median = summary["Ia"]["median"]
        non_ia_median = summary["non_Ia"]["median"]
        gaps.append(
            {
                "feature": name,
                "ia_median": ia_median,
                "non_ia_median": non_ia_median,
                "absolute_median_gap": abs(ia_median - non_ia_median),
            }
        )
    gaps.sort(key=lambda item: item["absolute_median_gap"], reverse=True)
    return gaps[:top_k]


def candidate_feature_summary(rows: list[dict], feature_names: list[str], top_k: int = 15) -> dict:
    collapsed = set(collapsed_columns(rows, feature_names))
    invalid = invalid_value_counts(rows, feature_names)
    top_gaps = largest_median_gaps(rows, feature_names, top_k=top_k)
    shortlisted = []
    for item in top_gaps:
        name = item["feature"]
        shortlisted.append(
            {
                **item,
                "collapsed": name in collapsed,
                "invalid_counts": invalid[name],
                "status_recommendation": "drop" if name in collapsed else "keep_for_importance_screen",
            }
        )
    return {
        "selection_rule": "Shortlist features with largest Ia vs non-Ia median gaps, excluding collapsed columns from recommended keep set.",
        "shortlisted_features": shortlisted,
        "collapsed_features": sorted(collapsed),
    }


def _plot_feature(rows: list[dict], feature: str, output_path: str) -> None:
    import matplotlib.pyplot as plt

    ia = [row[feature] for row in rows if row["label_name"] == "Ia"]
    non_ia = [row[feature] for row in rows if row["label_name"] != "Ia"]
    plt.figure(figsize=(7, 4))
    plt.hist(non_ia, bins=40, alpha=0.6, label="non-Ia")
    plt.hist(ia, bins=40, alpha=0.6, label="Ia")
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_distribution_plots(
    rows: list[dict],
    features: list[str],
    *,
    output_dir: str = "plots/phase2_tier1",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    histogram_data_paths = []
    plot_error = None
    for feature in features:
        ia = [row[feature] for row in rows if row["label_name"] == "Ia"]
        non_ia = [row[feature] for row in rows if row["label_name"] != "Ia"]
        histogram_data = {
            "feature": feature,
            "Ia_values": ia,
            "non_Ia_values": non_ia,
        }
        histogram_path = os.path.join(output_dir, f"{feature}_distribution_data.json")
        with open(histogram_path, "w") as handle:
            json.dump(histogram_data, handle)
        histogram_data_paths.append(histogram_path)

        if plot_error is None:
            try:
                output_path = os.path.join(output_dir, f"{feature}_distribution.png")
                _plot_feature(rows, feature, output_path)
                paths.append(output_path)
            except Exception as exc:
                plot_error = str(exc)

    return {
        "plot_paths": paths,
        "histogram_data_paths": histogram_data_paths,
        "plot_error": plot_error,
    }


def update_tier1_metadata(
    report: dict,
    *,
    metadata_path: str = "results/phase2_tier1/spcc_tier1_metadata.json",
    candidate_summary_path: str,
    plot_result: dict,
) -> None:
    with open(metadata_path) as handle:
        metadata = json.load(handle)

    collapsed = set(report["collapsed_columns"])
    for spec in metadata.get("feature_registry", []):
        spec["collapsed"] = spec["name"] in collapsed

    metadata["validation"] = {
        "report_path": "results/phase2_tier1/spcc_feature_validation_report.json",
        "candidate_summary_path": candidate_summary_path,
        "plot_paths": plot_result["plot_paths"],
        "histogram_data_paths": plot_result["histogram_data_paths"],
        "plot_error": plot_result["plot_error"],
        "collapsed_columns": report["collapsed_columns"],
    }

    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)


def build_validation_report(csv_path: str) -> dict:
    rows = load_feature_rows(csv_path)
    feature_names = _feature_names()
    invalid = invalid_value_counts(rows, feature_names)
    collapsed = collapsed_columns(rows, feature_names)
    overall = per_feature_summary(rows, feature_names)
    class_ranges = class_range_summary(rows, feature_names)
    top_gaps = largest_median_gaps(rows, feature_names)
    candidates = candidate_feature_summary(rows, feature_names)

    return {
        "csv_path": csv_path,
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "invalid_value_counts": invalid,
        "collapsed_columns": collapsed,
        "overall_summary": overall,
        "class_range_summary": class_ranges,
        "top_class_median_gaps": top_gaps,
        "candidate_feature_summary": candidates,
    }


def write_validation_report(
    csv_path: str,
    *,
    output_path: str = "results/phase2_tier1/spcc_feature_validation_report.json",
) -> dict:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = build_validation_report(csv_path)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)

    shortlisted = [
        item["feature"]
        for item in report["candidate_feature_summary"]["shortlisted_features"]
        if item["status_recommendation"] == "keep_for_importance_screen"
    ][:8]
    plot_result = write_distribution_plots(load_feature_rows(csv_path), shortlisted)

    candidate_path = "results/phase2_tier1/spcc_candidate_feature_summary.json"
    with open(candidate_path, "w") as handle:
        json.dump(report["candidate_feature_summary"], handle, indent=2)

    update_tier1_metadata(
        report,
        candidate_summary_path=candidate_path,
        plot_result=plot_result,
    )

    return {
        "report_path": output_path,
        "candidate_summary_path": candidate_path,
        "plot_paths": plot_result["plot_paths"],
        "histogram_data_paths": plot_result["histogram_data_paths"],
        "plot_error": plot_result["plot_error"],
    }


if __name__ == "__main__":
    output = write_validation_report("data/processed/spcc_features_tier1.csv")
    print(json.dumps(output, indent=2))
