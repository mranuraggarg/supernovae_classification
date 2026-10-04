"""Leave-one-feature-out retraining for Phase 2 Tier 2."""

from __future__ import annotations

from phase2_tier2_common import (
    RESULTS_DIR,
    baseline_reference_payload,
    create_context,
    evaluate_feature_subset,
    feature_to_group,
    rank_rows_by_delta,
    run_baseline_parity_check,
    write_csv,
    write_json,
    write_simple_markdown_table,
)


CSV_PATH = f"{RESULTS_DIR}/feature_ablation_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/feature_ablation_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/feature_ablation_summary.md"


def main() -> None:
    context = create_context()
    rows = []

    for feature_name in context.compact_features:
        remaining_features = [name for name in context.compact_features if name != feature_name]
        evaluation = evaluate_feature_subset(context, remaining_features, subset_name=f"drop_{feature_name}")
        rows.append(
            {
                "feature_removed": feature_name,
                "feature_group": feature_to_group(feature_name),
                "num_features": len(remaining_features),
                "f1": evaluation["metrics"]["f1"],
                "roc_auc": evaluation["metrics"]["roc_auc"],
                "pr_auc": evaluation["metrics"]["pr_auc"],
                "delta_f1": evaluation["delta_f1"],
                "delta_roc_auc": evaluation["delta_roc_auc"],
                "delta_pr_auc": evaluation["delta_pr_auc"],
            }
        )

    rank_rows_by_delta(rows, "delta_f1", "rank_by_delta_f1")
    rows.sort(key=lambda row: row["rank_by_delta_f1"])

    write_csv(
        CSV_PATH,
        [
            "feature_removed",
            "feature_group",
            "num_features",
            "f1",
            "roc_auc",
            "pr_auc",
            "delta_f1",
            "delta_roc_auc",
            "delta_pr_auc",
            "rank_by_delta_f1",
        ],
        rows,
    )

    payload = {
        "experiment": "leave_one_feature_out",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "rows": rows,
    }
    write_json(JSON_PATH, payload)

    markdown_rows = [
        [
            row["feature_removed"],
            row["feature_group"],
            str(row["num_features"]),
            f"{row['f1']:.6f}",
            f"{row['roc_auc']:.6f}",
            f"{row['pr_auc']:.6f}",
            f"{row['delta_f1']:+.6f}",
            f"{row['delta_pr_auc']:+.6f}",
            str(row["rank_by_delta_f1"]),
        ]
        for row in rows
    ]
    write_simple_markdown_table(
        SUMMARY_PATH,
        ["feature_removed", "feature_group", "num_features", "f1", "roc_auc", "pr_auc", "delta_f1", "delta_pr_auc", "rank"],
        markdown_rows,
        [
            "# Phase 2 Tier 2 Feature Ablation",
            "",
            "Leave-one-feature-out retraining on the frozen compact Tier 1 feature set.",
        ],
    )
    print(f"Wrote {CSV_PATH}, {JSON_PATH}, and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
