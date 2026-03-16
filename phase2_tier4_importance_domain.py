"""Phase 2 Tier 4 Experiment C: feature-importance stability across domains."""

from __future__ import annotations

import json

import numpy as np

from phase2_tier2_common import create_context
from phase2_tier4_common import (
    PLOTS_DIR,
    RESULTS_DIR,
    TOP_K_IMPORTANCE,
    ablation_importance_rows,
    domain_splits_from_variants,
    ensure_output_dirs,
    load_variant_rows,
    mean_rank_from_methods,
    pairwise_overlap_score,
    save_csv,
    save_json,
    shap_importance_rows,
    tier4_reference_payload,
    write_markdown,
)
from phase2_tier3_model_compare import train_and_evaluate_model


CSV_PATH = f"{RESULTS_DIR}/importance_domain_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/importance_domain_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/importance_domain_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier4_importance_domain.png"


def main() -> None:
    ensure_output_dirs()
    context = create_context()
    variant_rows = load_variant_rows(require_plasticc=False)
    domain_splits = domain_splits_from_variants(context, variant_rows)

    flat_rows = []
    domain_payload = []
    aggregate_top_sets = []

    for domain_name in domain_splits:
        train_rows = domain_splits[domain_name]["trainval"]
        test_rows = domain_splits[domain_name]["test"]
        train_x = np.array([[row[name] for name in context.compact_features] for row in train_rows], dtype=np.float32)
        train_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in train_rows], dtype=np.int32)
        test_x = np.array([[row[name] for name in context.compact_features] for row in test_rows], dtype=np.float32)
        test_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in test_rows], dtype=np.int32)

        result = train_and_evaluate_model(
            "xgboost",
            train_x,
            train_y,
            test_x,
            test_y,
            context.compact_features,
        )
        gain_rows = result["native_importance"]
        perm_rows = result["permutation_importance"]
        ablation_rows = ablation_importance_rows(
            train_rows,
            test_rows,
            context.compact_features,
            result["metrics"],
        )
        shap_rows = shap_importance_rows(train_rows, test_rows, context.compact_features)
        method_rows = {
            "gain": gain_rows,
            "permutation": perm_rows,
            "ablation": ablation_rows,
        }
        if shap_rows:
            method_rows["shap"] = shap_rows

        aggregate_rows = mean_rank_from_methods(method_rows, context.compact_features)
        aggregate_top_sets.append([row["feature"] for row in aggregate_rows[:TOP_K_IMPORTANCE]])

        for method_name, rows in method_rows.items():
            for row in rows:
                score_key = "score"
                if method_name == "permutation":
                    score_key = "mean_pr_auc_drop"
                elif method_name == "ablation":
                    score_key = "ablation_score"
                flat_rows.append(
                    {
                        "domain": domain_name,
                        "method": method_name,
                        "feature": row["feature"],
                        "feature_group": row["feature_group"],
                        "rank": row["rank"],
                        "score": float(row.get(score_key, row.get("score", 0.0))),
                    }
                )

        domain_payload.append(
            {
                "domain": domain_name,
                "metrics": result["metrics"],
                "method_rows": method_rows,
                "aggregate_rows": aggregate_rows,
                "top_features": [row["feature"] for row in aggregate_rows[:TOP_K_IMPORTANCE]],
            }
        )

    stability_score = pairwise_overlap_score(aggregate_top_sets)

    plot_rows = []
    for domain_info in domain_payload:
        plot_rows.append(
            {
                "domain": domain_info["domain"],
                "mean_rank_score": sum(1.0 / row["aggregate_rank"] for row in domain_info["aggregate_rows"][:TOP_K_IMPORTANCE]),
            }
        )

    from phase2_tier4_common import plot_grouped_bars

    plot_grouped_bars(
        plot_rows,
        [row["domain"] for row in plot_rows],
        ["mean_rank_score"],
        ["#7c3aed"],
        "Phase 2 Tier 4 Importance Stability",
        "Top-feature consensus score",
        PLOT_PATH,
    )

    save_csv(CSV_PATH, ["domain", "method", "feature", "feature_group", "rank", "score"], flat_rows)
    payload = {
        "experiment": "importance_domain",
        "reference": tier4_reference_payload(context),
        "domain_rows": domain_payload,
        "feature_stability_score": stability_score,
        "plot_path": PLOT_PATH,
    }
    save_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 4 Domain Importance Stability",
        "",
        f"Feature stability score (mean top-{TOP_K_IMPORTANCE} overlap across domains): {stability_score:.6f}",
        "",
        "| domain | top features |",
        "| --- | --- |",
    ]
    for domain_info in domain_payload:
        lines.append(f"| {domain_info['domain']} | {', '.join(domain_info['top_features'])} |")
    lines.extend(["", f"Plot: `{PLOT_PATH}`"])
    write_markdown(SUMMARY_PATH, lines)

    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()
