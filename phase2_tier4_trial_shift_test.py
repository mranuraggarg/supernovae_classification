"""Trial shift test using ratio-based color features."""

from __future__ import annotations

import json

from phase2_tier4_common import SHIFT_ORDER, plot_grouped_bars
from phase2_tier4_trial_common import (
    PLOTS_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    load_trial_domain_splits,
    run_domain_experiment,
    save_csv,
    save_json,
    trial_reference_payload,
    write_markdown,
)


CSV_PATH = f"{RESULTS_DIR}/trial_shift_test_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/trial_shift_test_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/trial_shift_test_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier4_trial_shift_test.png"


def main() -> None:
    ensure_output_dirs()
    context, shift_splits = load_trial_domain_splits()

    rows = []
    for shift_name in SHIFT_ORDER:
        if shift_name not in shift_splits:
            continue
        result = run_domain_experiment(
            shift_splits["spcc"]["trainval"],
            shift_splits[shift_name]["test"],
            context.compact_features,
            context.baseline_metrics,
        )
        rows.append(
            {
                "shift_name": shift_name,
                "f1": result["metrics"]["f1"],
                "pr_auc": result["metrics"]["pr_auc"],
                "roc_auc": result["metrics"]["roc_auc"],
                "delta_f1": result["delta_f1"],
                "delta_pr_auc": result["delta_pr_auc"],
                "delta_roc_auc": result["delta_roc_auc"],
                "f1_drop": -result["delta_f1"],
            }
        )

    plot_grouped_bars(
        rows,
        [row["shift_name"] for row in rows],
        ["f1_drop"],
        ["#1d4ed8"],
        "Phase 2 Tier 4 Trial Shift Test",
        "F1 drop vs frozen baseline",
        PLOT_PATH,
    )

    save_csv(
        CSV_PATH,
        ["shift_name", "f1", "pr_auc", "roc_auc", "delta_f1", "delta_pr_auc", "delta_roc_auc", "f1_drop"],
        rows,
    )
    payload = {
        "experiment": "trial_shift_test",
        "reference": trial_reference_payload(context),
        "rows": rows,
        "plot_path": PLOT_PATH,
    }
    save_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 4 Trial Shift Test",
        "",
        "| shift | F1 | F1 drop | PR-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['shift_name']} | {row['f1']:.6f} | {row['f1_drop']:.6f} | {row['pr_auc']:.6f} |")
    lines.extend(["", f"Plot: `{PLOT_PATH}`"])
    write_markdown(SUMMARY_PATH, lines)
    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()
