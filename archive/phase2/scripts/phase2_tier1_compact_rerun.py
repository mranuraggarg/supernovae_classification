"""Run a clean XGBoost comparison using the tightened compact working set."""

from __future__ import annotations

import json
import os

from phase2_tier1_xgb_importance import fit_final_xgb_model, select_best_xgb_model, split_rows


TIGHT_MANIFEST_PATH = "results/phase2_tier1/phase2_tier1_feature_manifest_tightened.json"
OUTPUT_PATH = "results/phase2_tier1/phase2_tier1_compact_rerun.json"


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def train_and_score(split_data: dict, feature_names: list[str]) -> dict:
    selection, selection_summary = select_best_xgb_model(
        split_data["train"],
        split_data["validation"],
        feature_names,
    )
    _, metrics, _, _, _ = fit_final_xgb_model(
        split_data["trainval"],
        split_data["test"],
        feature_names,
        selection["params"],
        selection["best_iteration"],
    )
    return {
        "feature_count": len(feature_names),
        "features": feature_names,
        "selection_summary": selection_summary,
        "metrics": metrics,
    }


def main() -> None:
    manifest = load_json(TIGHT_MANIFEST_PATH)
    split_data = split_rows()

    compact = manifest["compact_working_feature_set"]
    previous_working = manifest["accept_features"] + manifest["compact_review_features"] + manifest["safe_drop_features"]
    full_baseline = manifest["full_baseline_features"]

    results = {
        "compact_working_set": train_and_score(split_data, compact),
        "previous_working_set": train_and_score(split_data, previous_working),
        "full_baseline": train_and_score(split_data, full_baseline),
    }
    compact_metrics = results["compact_working_set"]["metrics"]
    comparison = {}
    for name in ("previous_working_set", "full_baseline"):
        metrics = results[name]["metrics"]
        comparison[name] = {
            "pr_auc_delta_vs_compact": metrics["pr_auc"] - compact_metrics["pr_auc"],
            "f1_delta_vs_compact": metrics["f1"] - compact_metrics["f1"],
            "roc_auc_delta_vs_compact": metrics["roc_auc"] - compact_metrics["roc_auc"],
        }

    payload = {
        "source_manifest": TIGHT_MANIFEST_PATH,
        "results": results,
        "comparison": comparison,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved compact rerun: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
