"""Progressive subset growth experiments for Phase 2 Tier 2."""

from __future__ import annotations

from phase2_tier2_common import (
    FEATURE_GROUPS,
    RESULTS_DIR,
    baseline_reference_payload,
    create_context,
    evaluate_feature_subset,
    run_baseline_parity_check,
    write_csv,
    write_json,
    write_simple_markdown_table,
)


CSV_PATH = f"{RESULTS_DIR}/subset_growth_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/subset_growth_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/subset_growth_summary.md"


SUBSET_SPECS = [
    ("brightness_only", ["brightness"]),
    ("color_only", ["color"]),
    ("temporal_only", ["temporal"]),
    ("brightness_plus_color", ["brightness", "color"]),
    ("brightness_plus_color_plus_variability", ["brightness", "color", "variability"]),
    ("full_compact", ["brightness", "color", "variability", "temporal"]),
]


def features_for_blocks(block_names: list[str]) -> list[str]:
    feature_names: list[str] = []
    for block_name in block_names:
        feature_names.extend(FEATURE_GROUPS[block_name])
    return feature_names


def main() -> None:
    context = create_context()
    rows = []

    for subset_name, block_names in SUBSET_SPECS:
        feature_names = features_for_blocks(block_names)
        evaluation = evaluate_feature_subset(context, feature_names, subset_name=subset_name)
        rows.append(
            {
                "subset_name": subset_name,
                "feature_count": len(feature_names),
                "included_blocks": ",".join(block_names),
                "f1": evaluation["metrics"]["f1"],
                "roc_auc": evaluation["metrics"]["roc_auc"],
                "pr_auc": evaluation["metrics"]["pr_auc"],
                "delta_f1": evaluation["delta_f1"],
                "delta_roc_auc": evaluation["delta_roc_auc"],
                "delta_pr_auc": evaluation["delta_pr_auc"],
            }
        )

    write_csv(
        CSV_PATH,
        [
            "subset_name",
            "feature_count",
            "included_blocks",
            "f1",
            "roc_auc",
            "pr_auc",
            "delta_f1",
            "delta_roc_auc",
            "delta_pr_auc",
        ],
        rows,
    )

    payload = {
        "experiment": "subset_growth",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "rows": rows,
    }
    write_json(JSON_PATH, payload)

    markdown_rows = [
        [
            row["subset_name"],
            str(row["feature_count"]),
            row["included_blocks"],
            f"{row['f1']:.6f}",
            f"{row['roc_auc']:.6f}",
            f"{row['pr_auc']:.6f}",
            f"{row['delta_f1']:+.6f}",
            f"{row['delta_pr_auc']:+.6f}",
        ]
        for row in rows
    ]
    write_simple_markdown_table(
        SUMMARY_PATH,
        ["subset_name", "feature_count", "included_blocks", "f1", "roc_auc", "pr_auc", "delta_f1", "delta_pr_auc"],
        markdown_rows,
        [
            "# Phase 2 Tier 2 Subset Growth",
            "",
            "Cumulative compact-feature growth experiments plus single-family reference runs.",
        ],
    )
    print(f"Wrote {CSV_PATH}, {JSON_PATH}, and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
