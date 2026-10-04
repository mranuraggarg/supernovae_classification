"""Leave-one-block-out retraining for Phase 2 Tier 2."""

from __future__ import annotations

from phase2_tier2_common import (
    FEATURE_GROUPS,
    GROUP_ORDER,
    RESULTS_DIR,
    baseline_reference_payload,
    create_context,
    evaluate_feature_subset,
    run_baseline_parity_check,
    write_csv,
    write_json,
    write_simple_markdown_table,
)


CSV_PATH = f"{RESULTS_DIR}/block_ablation_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/block_ablation_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/block_ablation_summary.md"


def main() -> None:
    context = create_context()
    rows = []

    for block_name in GROUP_ORDER:
        removed_features = set(FEATURE_GROUPS[block_name])
        remaining_features = [name for name in context.compact_features if name not in removed_features]
        evaluation = evaluate_feature_subset(context, remaining_features, subset_name=f"drop_{block_name}")
        rows.append(
            {
                "block_removed": block_name,
                "remaining_feature_count": len(remaining_features),
                "f1": evaluation["metrics"]["f1"],
                "roc_auc": evaluation["metrics"]["roc_auc"],
                "pr_auc": evaluation["metrics"]["pr_auc"],
                "delta_f1": evaluation["delta_f1"],
                "delta_roc_auc": evaluation["delta_roc_auc"],
                "delta_pr_auc": evaluation["delta_pr_auc"],
            }
        )

    rows.sort(key=lambda row: row["delta_f1"])
    write_csv(
        CSV_PATH,
        [
            "block_removed",
            "remaining_feature_count",
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
        "experiment": "leave_one_block_out",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "rows": rows,
    }
    write_json(JSON_PATH, payload)

    markdown_rows = [
        [
            row["block_removed"],
            str(row["remaining_feature_count"]),
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
        ["block_removed", "remaining_features", "f1", "roc_auc", "pr_auc", "delta_f1", "delta_pr_auc"],
        markdown_rows,
        [
            "# Phase 2 Tier 2 Block Ablation",
            "",
            "Leave-one-block-out retraining across brightness, color, variability, and temporal families.",
        ],
    )
    print(f"Wrote {CSV_PATH}, {JSON_PATH}, and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
