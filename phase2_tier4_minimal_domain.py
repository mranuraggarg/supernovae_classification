"""Phase 2 Tier 4 Experiment D: domain stability of reduced compact subsets."""

from __future__ import annotations

import json

from phase2_tier2_common import create_context
from phase2_tier4_common import (
    PLOTS_DIR,
    RESULTS_DIR,
    domain_splits_from_variants,
    ensure_output_dirs,
    load_variant_rows,
    load_subset_specs,
    mean_metric,
    run_domain_experiment,
    save_csv,
    save_json,
    tier4_reference_payload,
    write_markdown,
)
from phase2_tier4_common import plot_grouped_bars


CSV_PATH = f"{RESULTS_DIR}/minimal_domain_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/minimal_domain_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/minimal_domain_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier4_minimal_domain.png"


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    subset_specs = load_subset_specs(context)
    variant_rows = load_variant_rows(require_plasticc=False)
    domain_splits = domain_splits_from_variants(context, variant_rows)

    detailed_rows = []
    summary_rows = []
    for subset_spec in subset_specs:
        subset_runs = []
        for domain_name in domain_splits:
            result = run_domain_experiment(
                domain_splits[domain_name]["trainval"],
                domain_splits[domain_name]["test"],
                subset_spec["feature_names"],
                context.baseline_metrics,
            )
            row = {
                "subset_name": subset_spec["subset_name"],
                "selection_rule": subset_spec["selection_rule"],
                "domain": domain_name,
                "feature_count": len(subset_spec["feature_names"]),
                "feature_list": ", ".join(subset_spec["feature_names"]),
                "f1": result["metrics"]["f1"],
                "pr_auc": result["metrics"]["pr_auc"],
                "roc_auc": result["metrics"]["roc_auc"],
                "delta_f1": result["delta_f1"],
            }
            detailed_rows.append(row)
            subset_runs.append(row)
        summary_rows.append(
            {
                "subset_name": subset_spec["subset_name"],
                "feature_count": len(subset_spec["feature_names"]),
                "selection_rule": subset_spec["selection_rule"],
                "mean_f1": mean_metric(subset_runs, "f1"),
                "min_f1": min(float(row["f1"]) for row in subset_runs),
                "max_f1": max(float(row["f1"]) for row in subset_runs),
            }
        )

    plot_grouped_bars(
        summary_rows,
        [row["subset_name"] for row in summary_rows],
        ["mean_f1"],
        ["#0f766e"],
        "Phase 2 Tier 4 Minimal-Core Domain Stability",
        "Mean F1 across domains",
        PLOT_PATH,
    )

    save_csv(
        CSV_PATH,
        ["subset_name", "selection_rule", "domain", "feature_count", "feature_list", "f1", "pr_auc", "roc_auc", "delta_f1"],
        detailed_rows,
    )
    payload = {
        "experiment": "minimal_domain",
        "reference": tier4_reference_payload(context),
        "detailed_rows": detailed_rows,
        "summary_rows": summary_rows,
        "plot_path": PLOT_PATH,
    }
    save_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 4 Minimal-Core Domain Stability",
        "",
        "| subset | feature_count | mean F1 | min F1 | max F1 |",
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
