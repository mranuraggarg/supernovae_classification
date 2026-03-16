"""Phase 2 Tier 4 Experiment B: mixed-domain training for robustness."""

from __future__ import annotations

import json

from phase2_tier2_common import create_context
from phase2_tier4_common import (
    PLOTS_DIR,
    RESULTS_DIR,
    domain_splits_from_variants,
    ensure_output_dirs,
    load_variant_rows,
    plot_grouped_bars,
    run_domain_experiment,
    save_csv,
    save_json,
    tier4_reference_payload,
    write_markdown,
)


CSV_PATH = f"{RESULTS_DIR}/mixed_training_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/mixed_training_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/mixed_training_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier4_mixed_training.png"


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    variant_rows = load_variant_rows(require_plasticc=False)
    domain_splits = domain_splits_from_variants(context, variant_rows)
    test_domains = list(domain_splits.keys())

    training_specs = [
        ("spcc", domain_splits["spcc"]["trainval"]),
        ("spcc_plus_noise", domain_splits["spcc"]["trainval"] + domain_splits["noise"]["trainval"]),
        ("spcc_plus_no_z", domain_splits["spcc"]["trainval"] + domain_splits["no_z"]["trainval"]),
        ("spcc_plus_no_i", domain_splits["spcc"]["trainval"] + domain_splits["no_i"]["trainval"]),
        ("spcc_plus_short_span", domain_splits["spcc"]["trainval"] + domain_splits["short_span"]["trainval"]),
        ("spcc_plus_flux_scale", domain_splits["spcc"]["trainval"] + domain_splits["flux_scale"]["trainval"]),
    ]
    if "plasticc" in domain_splits:
        training_specs.append(("spcc_plus_plasticc", domain_splits["spcc"]["trainval"] + domain_splits["plasticc"]["trainval"]))

    rows = []
    for train_label, train_rows in training_specs:
        for test_domain in test_domains:
            result = run_domain_experiment(
                train_rows,
                domain_splits[test_domain]["test"],
                context.compact_features,
                context.baseline_metrics,
            )
            rows.append(
                {
                    "train_domain": train_label,
                    "test_domain": test_domain,
                    "feature_count": len(context.compact_features),
                    "f1": result["metrics"]["f1"],
                    "pr_auc": result["metrics"]["pr_auc"],
                    "roc_auc": result["metrics"]["roc_auc"],
                    "delta_f1": result["delta_f1"],
                    "delta_pr_auc": result["delta_pr_auc"],
                    "delta_roc_auc": result["delta_roc_auc"],
                }
            )

    summary_rows = []
    for train_label, _ in training_specs:
        subset_rows = [row for row in rows if row["train_domain"] == train_label]
        summary_rows.append(
            {
                "train_domain": train_label,
                "mean_f1": sum(row["f1"] for row in subset_rows) / len(subset_rows),
                "mean_pr_auc": sum(row["pr_auc"] for row in subset_rows) / len(subset_rows),
            }
        )

    plot_grouped_bars(
        summary_rows,
        [row["train_domain"] for row in summary_rows],
        ["mean_f1", "mean_pr_auc"],
        ["#14532d", "#b45309"],
        "Phase 2 Tier 4 Mixed Training",
        "Mean cross-domain score",
        PLOT_PATH,
    )

    save_csv(
        CSV_PATH,
        ["train_domain", "test_domain", "feature_count", "f1", "pr_auc", "roc_auc", "delta_f1", "delta_pr_auc", "delta_roc_auc"],
        rows,
    )
    payload = {
        "experiment": "mixed_training",
        "reference": tier4_reference_payload(context),
        "rows": rows,
        "summary_rows": summary_rows,
        "plot_path": PLOT_PATH,
    }
    save_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 4 Mixed-Domain Training",
        "",
        "Evaluate whether adding SPCC variants and PLAsTiCC to training improves average cross-domain performance.",
        "",
        "| train | mean F1 | mean PR-AUC |",
        "| --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(f"| {row['train_domain']} | {row['mean_f1']:.6f} | {row['mean_pr_auc']:.6f} |")
    lines.extend(["", f"Plot: `{PLOT_PATH}`"])
    write_markdown(SUMMARY_PATH, lines)

    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()
