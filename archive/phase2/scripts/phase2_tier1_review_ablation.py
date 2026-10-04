"""Second-pass review-feature ablation for Phase 2 Tier 1 XGBoost."""

from __future__ import annotations

import json
import os

from phase2_tier1_xgb_importance import (
    RESULTS_DIR,
    compare_feature_sets,
    fit_final_xgb_model,
    select_best_xgb_model,
    split_rows,
)


MANIFEST_PATH = "results/phase2_tier1/phase2_tier1_feature_manifest.json"
OUTPUT_PATH = "results/phase2_tier1/phase2_tier1_review_ablation.json"


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path) as handle:
        return json.load(handle)


def feature_subsets_for_ablation(manifest: dict) -> dict[str, list[str]]:
    working = manifest["working_feature_set"]
    review = manifest["review_features"]
    blocks = manifest.get("review_ablation_blocks", {})
    subsets = {}

    for feature in review:
        subsets[f"drop_feature__{feature}"] = [name for name in working if name != feature]
    for block_name, block_features in blocks.items():
        subsets[f"drop_block__{block_name}"] = [name for name in working if name not in block_features]
    return subsets


def recommendation(pr_auc_delta: float, f1_delta: float) -> str:
    if pr_auc_delta >= -0.002 and f1_delta >= -0.002:
        return "safe_to_drop"
    if pr_auc_delta <= -0.01 or f1_delta <= -0.01:
        return "keep"
    return "review_again"


def main() -> None:
    manifest = load_manifest()
    split_data = split_rows()

    baseline_selection, selection_summary = select_best_xgb_model(
        split_data["train"],
        split_data["validation"],
        manifest["working_feature_set"],
    )
    _, baseline_metrics, _, _, _ = fit_final_xgb_model(
        split_data["trainval"],
        split_data["test"],
        manifest["working_feature_set"],
        baseline_selection["params"],
        baseline_selection["best_iteration"],
    )

    ablation_rows = []
    for name, feature_subset in feature_subsets_for_ablation(manifest).items():
        selection, _ = select_best_xgb_model(
            split_data["train"],
            split_data["validation"],
            feature_subset,
        )
        _, metrics, _, _, _ = fit_final_xgb_model(
            split_data["trainval"],
            split_data["test"],
            feature_subset,
            selection["params"],
            selection["best_iteration"],
        )
        pr_auc_delta = metrics["pr_auc"] - baseline_metrics["pr_auc"]
        f1_delta = metrics["f1"] - baseline_metrics["f1"]
        ablation_rows.append(
            {
                "name": name,
                "feature_count": len(feature_subset),
                "removed_features": [feature for feature in manifest["working_feature_set"] if feature not in feature_subset],
                "metrics": metrics,
                "pr_auc_delta_vs_working_set": pr_auc_delta,
                "f1_delta_vs_working_set": f1_delta,
                "recommendation": recommendation(pr_auc_delta, f1_delta),
            }
        )

    ablation_rows.sort(key=lambda row: (row["pr_auc_delta_vs_working_set"], row["f1_delta_vs_working_set"]), reverse=True)
    payload = {
        "source_manifest": MANIFEST_PATH,
        "baseline_working_set_metrics": baseline_metrics,
        "baseline_selection": selection_summary,
        "ablation_results": ablation_rows,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved ablation report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
