"""Reduced-core subset search for Phase 2 Tier 2."""

from __future__ import annotations

import os

from phase2_tier2_common import (
    RESULTS_DIR,
    baseline_reference_payload,
    create_context,
    evaluate_feature_subset,
    format_feature_list,
    load_interpretation_rows,
    parse_numeric_rows,
    read_csv_rows,
    run_baseline_parity_check,
    select_ranked_core_subsets,
    write_csv,
    write_json,
    write_simple_markdown_table,
)


CSV_PATH = f"{RESULTS_DIR}/minimal_core_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/minimal_core_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/minimal_core_summary.md"
FEATURE_ABLATION_CSV_PATH = f"{RESULTS_DIR}/feature_ablation_metrics.csv"


def load_feature_ablation_rows() -> list[dict]:
    if not os.path.exists(FEATURE_ABLATION_CSV_PATH):
        raise FileNotFoundError(
            "Minimal core search depends on feature ablation output. Run phase2_tier2_feature_ablation.py first."
        )
    return parse_numeric_rows(read_csv_rows(FEATURE_ABLATION_CSV_PATH))


def main() -> None:
    context = create_context()
    feature_ablation_rows = load_feature_ablation_rows()
    interpretation_rows = load_interpretation_rows()
    subset_specs = select_ranked_core_subsets(context.compact_features, feature_ablation_rows, interpretation_rows, [5, 8, 10])

    rows = []
    for subset_spec in subset_specs:
        evaluation = evaluate_feature_subset(context, subset_spec["feature_names"], subset_name=subset_spec["subset_name"])
        rows.append(
            {
                "subset_name": subset_spec["subset_name"],
                "selection_rule": subset_spec["selection_rule"],
                "feature_count": len(subset_spec["feature_names"]),
                "feature_list": format_feature_list(subset_spec["feature_names"]),
                "f1": evaluation["metrics"]["f1"],
                "roc_auc": evaluation["metrics"]["roc_auc"],
                "pr_auc": evaluation["metrics"]["pr_auc"],
                "delta_f1": evaluation["delta_f1"],
                "delta_roc_auc": evaluation["delta_roc_auc"],
                "delta_pr_auc": evaluation["delta_pr_auc"],
            }
        )

    rows.sort(key=lambda row: row["feature_count"])
    write_csv(
        CSV_PATH,
        [
            "subset_name",
            "selection_rule",
            "feature_count",
            "feature_list",
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
        "experiment": "minimal_core",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "feature_ablation_dependency": FEATURE_ABLATION_CSV_PATH,
        "rows": rows,
    }
    write_json(JSON_PATH, payload)

    markdown_rows = [
        [
            row["subset_name"],
            str(row["feature_count"]),
            row["selection_rule"],
            row["feature_list"],
            f"{row['f1']:.6f}",
            f"{row['pr_auc']:.6f}",
            f"{row['delta_f1']:+.6f}",
        ]
        for row in rows
    ]
    write_simple_markdown_table(
        SUMMARY_PATH,
        ["subset_name", "feature_count", "selection_rule", "feature_list", "f1", "pr_auc", "delta_f1"],
        markdown_rows,
        [
            "# Phase 2 Tier 2 Minimal Core",
            "",
            "Reduced-core compact subsets chosen from feature ablation plus Tier 1 importance context.",
        ],
    )
    print(f"Wrote {CSV_PATH}, {JSON_PATH}, and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
