"""Trial domain-swap experiment using temporal-alignment features."""

from __future__ import annotations

import json

from phase2_tier4_common import DOMAIN_ORDER, plot_grouped_bars, save_csv, save_json
from phase2_tier4_trial_temporal_common import (
    PLOTS_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    load_trial_domain_splits,
    run_domain_experiment,
    trial_reference_payload,
    write_markdown,
)


CSV_PATH = f"{RESULTS_DIR}/trial_temporal_domain_swap_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/trial_temporal_domain_swap_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/trial_temporal_domain_swap_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier4_trial_temporal_domain_swap.png"


def main() -> None:
    ensure_output_dirs()
    context, domain_splits = load_trial_domain_splits()

    rows = []
    for test_domain in DOMAIN_ORDER:
        if test_domain not in domain_splits:
            continue
        result = run_domain_experiment(
            domain_splits["spcc"]["trainval"],
            domain_splits[test_domain]["test"],
            context.compact_features,
            context.baseline_metrics,
        )
        rows.append(
            {
                "train_domain": "spcc",
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

    plot_grouped_bars(
        rows,
        [row["test_domain"] for row in rows],
        ["f1", "pr_auc"],
        ["#0f766e", "#b45309"],
        "Phase 2 Tier 4 Trial Temporal Domain Swap",
        "Score",
        PLOT_PATH,
    )

    save_csv(
        CSV_PATH,
        ["train_domain", "test_domain", "feature_count", "f1", "pr_auc", "roc_auc", "delta_f1", "delta_pr_auc", "delta_roc_auc"],
        rows,
    )
    payload = {
        "experiment": "trial_temporal_domain_swap",
        "reference": trial_reference_payload(context),
        "rows": rows,
        "plot_path": PLOT_PATH,
    }
    save_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 4 Trial Temporal Domain Swap",
        "",
        "Trial experiment replacing peak-time features with r-anchored relative phase offsets.",
        "",
        "| train | test | F1 | PR-AUC | ROC-AUC | delta F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['train_domain']} | {row['test_domain']} | {row['f1']:.6f} | {row['pr_auc']:.6f} | {row['roc_auc']:.6f} | {row['delta_f1']:+.6f} |"
        )
    lines.extend(["", f"Plot: `{PLOT_PATH}`"])
    write_markdown(SUMMARY_PATH, lines)
    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()

