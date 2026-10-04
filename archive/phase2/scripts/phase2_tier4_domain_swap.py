"""Phase 2 Tier 4 Experiment A: train-test domain swap on compact features."""

from __future__ import annotations

import json

from phase2_tier2_common import create_context
from phase2_tier4_common import (
    DOMAIN_ORDER,
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


CSV_PATH = f"{RESULTS_DIR}/domain_swap_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/domain_swap_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/domain_swap_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier4_domain_swap.png"


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    variant_rows = load_variant_rows(require_plasticc=False)
    domain_splits = domain_splits_from_variants(context, variant_rows)

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
        ["#1d4e89", "#d97706"],
        "Phase 2 Tier 4 Domain Swap",
        "Score",
        PLOT_PATH,
    )

    save_csv(
        CSV_PATH,
        ["train_domain", "test_domain", "feature_count", "f1", "pr_auc", "roc_auc", "delta_f1", "delta_pr_auc", "delta_roc_auc"],
        rows,
    )
    payload = {
        "experiment": "domain_swap",
        "reference": tier4_reference_payload(context),
        "rows": rows,
        "plot_path": PLOT_PATH,
    }
    save_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 4 Domain Swap",
        "",
        "Train on frozen SPCC compact data and evaluate on Tier 4 variant tables plus PLAsTiCC when available.",
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
