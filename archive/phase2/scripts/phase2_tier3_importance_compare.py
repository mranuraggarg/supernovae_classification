"""Phase 2 Tier 3 Experiment D/F: feature-importance consistency and physical interpretation."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import xgboost as xgb

from phase2_tier1_benchmarks import stratified_split_indices
from phase2_tier2_common import (
    baseline_reference_payload,
    create_context,
    feature_to_group,
    load_interpretation_rows,
    parse_numeric_rows,
    read_csv_rows,
    run_baseline_parity_check,
    write_csv,
    write_json,
)
from phase2_tier3_model_compare import (
    PLOTS_DIR,
    RESULTS_DIR,
    _fit_final_xgb,
    _select_best_xgb,
    ensure_output_dirs,
    maybe_import_matplotlib,
    train_and_evaluate_model,
)


CSV_PATH = f"{RESULTS_DIR}/importance_compare_metrics.csv"
JSON_PATH = f"{RESULTS_DIR}/importance_compare_metrics.json"
SUMMARY_PATH = f"{RESULTS_DIR}/importance_compare_summary.md"
PLOT_PATH = f"{PLOTS_DIR}/phase2_tier3_importance_consistency.png"

TIER2_ABLATION_CSV_PATH = "results/phase2_tier2/feature_ablation_metrics.csv"


def load_tier2_ablation_rows() -> list[dict[str, Any]]:
    if not os.path.exists(TIER2_ABLATION_CSV_PATH):
        raise FileNotFoundError(
            "Tier-3 importance comparison depends on Tier-2 feature ablation output. "
            "Run phase2_tier2_feature_ablation.py first."
        )
    rows = parse_numeric_rows(read_csv_rows(TIER2_ABLATION_CSV_PATH))
    for row in rows:
        row["ablation_loss"] = max(-float(row["delta_f1"]), 0.0) + max(-float(row["delta_pr_auc"]), 0.0)
    return rows


def spearman_rank_correlation(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    common = sorted(set(rank_a) & set(rank_b))
    if len(common) < 2:
        return 0.0
    a = np.array([rank_a[name] for name in common], dtype=float)
    b = np.array([rank_b[name] for name in common], dtype=float)
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def build_rank_map(
    rows: list[dict[str, Any]],
    score_key: str,
    *,
    feature_key: str = "feature",
    reverse: bool = True,
) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: row[score_key], reverse=reverse)
    return {row[feature_key]: rank for rank, row in enumerate(ordered, start=1)}


def plot_top_feature_scores(consensus_rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    top_rows = consensus_rows[:8]
    labels = [row["feature"] for row in reversed(top_rows)]
    values = [row["consensus_score"] for row in reversed(top_rows)]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(labels, values, color="#8c5e58")
    ax.set_xlabel("Consensus score")
    ax.set_title("Phase 2 Tier 3 Cross-Method Feature Importance Consensus")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def write_markdown(path: str, lines: list[str]) -> None:
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_output_dirs()
    context = create_context()

    train_x = np.array([[row[name] for name in context.compact_features] for row in context.split_data["trainval"]], dtype=np.float32)
    train_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in context.split_data["trainval"]], dtype=np.int32)
    test_x = np.array([[row[name] for name in context.compact_features] for row in context.split_data["test"]], dtype=np.float32)
    test_y = np.array([1 if row["label_name"] == "Ia" else 0 for row in context.split_data["test"]], dtype=np.int32)

    train_idx, validation_idx = stratified_split_indices(train_y, 0.2, 42)
    selection = _select_best_xgb(
        train_x[train_idx],
        train_y[train_idx],
        train_x[validation_idx],
        train_y[validation_idx],
        context.compact_features,
        seed=42,
    )
    model_bundle, _ = _fit_final_xgb(
        train_x,
        train_y,
        test_x,
        test_y,
        context.compact_features,
        seed=42,
        params=selection["params"],
        num_boost_round=selection["best_iteration"],
    )
    mean = model_bundle["train_mean"]
    std = model_bundle["train_std"]
    test_x_scaled = (test_x - mean) / std
    dtest = xgb.DMatrix(test_x_scaled, label=test_y, feature_names=context.compact_features)
    shap_values = model_bundle["booster"].predict(dtest, pred_contribs=True)[:, :-1]

    model_result = train_and_evaluate_model(
        "xgboost",
        train_x,
        train_y,
        test_x,
        test_y,
        context.compact_features,
        seed=42,
    )
    xgb_native = model_result["native_importance"]

    shap_rows = []
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    for feature_name, score in zip(context.compact_features, mean_abs_shap):
        shap_rows.append({"feature": feature_name, "mean_abs_shap": float(score)})

    permutation_rows = model_result["permutation_importance"]
    ablation_rows = load_tier2_ablation_rows()
    interpretation_rows = load_interpretation_rows()

    gain_ranks = build_rank_map(xgb_native, "score")
    perm_ranks = build_rank_map(permutation_rows, "mean_pr_auc_drop")
    shap_ranks = build_rank_map(shap_rows, "mean_abs_shap")
    ablation_ranks = build_rank_map(ablation_rows, "ablation_loss", feature_key="feature_removed")

    consensus_rows = []
    interpretation_map = {row["feature"]: row for row in interpretation_rows}
    ablation_map = {row["feature_removed"]: row for row in ablation_rows}
    perm_map = {row["feature"]: row for row in permutation_rows}
    gain_map = {row["feature"]: row for row in xgb_native}
    shap_map = {row["feature"]: row for row in shap_rows}

    for feature_name in context.compact_features:
        rank_values = [
            gain_ranks.get(feature_name, len(context.compact_features)),
            perm_ranks.get(feature_name, len(context.compact_features)),
            shap_ranks.get(feature_name, len(context.compact_features)),
            ablation_ranks.get(feature_name, len(context.compact_features)),
        ]
        consensus_score = float(sum(1.0 / rank for rank in rank_values))
        interp = interpretation_map.get(feature_name, {})
        ablation = ablation_map.get(feature_name, {})
        perm = perm_map.get(feature_name, {})
        gain = gain_map.get(feature_name, {})
        shap = shap_map.get(feature_name, {})
        consensus_rows.append(
            {
                "feature": feature_name,
                "feature_group": feature_to_group(feature_name),
                "gain_rank": gain_ranks.get(feature_name, len(context.compact_features)),
                "permutation_rank": perm_ranks.get(feature_name, len(context.compact_features)),
                "shap_rank": shap_ranks.get(feature_name, len(context.compact_features)),
                "ablation_rank": ablation_ranks.get(feature_name, len(context.compact_features)),
                "gain_score": float(gain.get("score", 0.0)),
                "mean_pr_auc_drop": float(perm.get("mean_pr_auc_drop", 0.0)),
                "mean_abs_shap": float(shap.get("mean_abs_shap", 0.0)),
                "ablation_loss": float(ablation.get("ablation_loss", 0.0)),
                "consensus_score": consensus_score,
                "physical_meaning": interp.get("physical_meaning", ""),
                "interpretation": interp.get("interpretation", ""),
            }
        )

    consensus_rows.sort(key=lambda row: row["consensus_score"], reverse=True)
    plot_top_feature_scores(consensus_rows)

    rank_correlation_rows = [
        {
            "method_a": "gain",
            "method_b": "permutation",
            "spearman_rank_correlation": spearman_rank_correlation(gain_ranks, perm_ranks),
        },
        {
            "method_a": "gain",
            "method_b": "shap",
            "spearman_rank_correlation": spearman_rank_correlation(gain_ranks, shap_ranks),
        },
        {
            "method_a": "gain",
            "method_b": "ablation",
            "spearman_rank_correlation": spearman_rank_correlation(gain_ranks, ablation_ranks),
        },
        {
            "method_a": "permutation",
            "method_b": "shap",
            "spearman_rank_correlation": spearman_rank_correlation(perm_ranks, shap_ranks),
        },
        {
            "method_a": "permutation",
            "method_b": "ablation",
            "spearman_rank_correlation": spearman_rank_correlation(perm_ranks, ablation_ranks),
        },
        {
            "method_a": "shap",
            "method_b": "ablation",
            "spearman_rank_correlation": spearman_rank_correlation(shap_ranks, ablation_ranks),
        },
    ]

    write_csv(
        CSV_PATH,
        [
            "feature",
            "feature_group",
            "gain_rank",
            "permutation_rank",
            "shap_rank",
            "ablation_rank",
            "gain_score",
            "mean_pr_auc_drop",
            "mean_abs_shap",
            "ablation_loss",
            "consensus_score",
            "physical_meaning",
            "interpretation",
        ],
        consensus_rows,
    )

    payload = {
        "experiment": "importance_compare",
        "baseline_reference": baseline_reference_payload(context),
        "baseline_parity_check": run_baseline_parity_check(context),
        "consensus_rows": consensus_rows,
        "rank_correlations": rank_correlation_rows,
        "plot_path": PLOT_PATH,
    }
    write_json(JSON_PATH, payload)

    lines = [
        "# Phase 2 Tier 3 Feature Importance Consistency",
        "",
        "Cross-method comparison of XGBoost gain, permutation importance, SHAP, and Tier-2 ablation ranks.",
        "",
        "| feature | group | gain_rank | perm_rank | shap_rank | ablation_rank | consensus_score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in consensus_rows[:10]:
        lines.append(
            f"| {row['feature']} | {row['feature_group']} | {row['gain_rank']} | {row['permutation_rank']} | "
            f"{row['shap_rank']} | {row['ablation_rank']} | {row['consensus_score']:.3f} |"
        )
    lines.extend(["", "## Rank correlations", ""])
    for row in rank_correlation_rows:
        lines.append(
            f"- {row['method_a']} vs {row['method_b']}: {row['spearman_rank_correlation']:.3f}"
        )
    lines.extend(["", f"Plot: `{PLOT_PATH}`"])
    write_markdown(SUMMARY_PATH, lines)

    print(json.dumps({"csv_path": CSV_PATH, "json_path": JSON_PATH, "summary_path": SUMMARY_PATH}, indent=2))


if __name__ == "__main__":
    main()
