"""Build a Phase 2 feature manifest from saved XGBoost importance output."""

from __future__ import annotations

import json
import os


INPUT_PATH = "results/phase2_tier1/phase2_tier1_xgb_importance.json"
OUTPUT_PATH = "results/phase2_tier1/phase2_tier1_feature_manifest.json"


def block_members(feature_names: list[str], candidates: list[str]) -> list[str]:
    return [feature for feature in candidates if feature in feature_names]


def build_manifest(payload: dict) -> dict:
    full_features = payload["full_baseline_features"]
    buckets = payload["feature_buckets"]
    accept = buckets["accept"]
    review = buckets["review"]
    reject = buckets["reject"]
    working = accept + review

    review_blocks = {
        "peak_flux_block": block_members(review, [name for name in full_features if name.endswith("_peak_flux")] + ["peak_flux_all"]),
        "amplitude_block": block_members(review, [name for name in full_features if name.endswith("_amplitude")] + ["amplitude_all"]),
        "timing_block": block_members(review, [name for name in full_features if "time_of_peak" in name] + ["time_span"]),
        "mean_flux_block": block_members(review, [name for name in full_features if name.endswith("_mean_flux")] + ["mean_flux_all"]),
        "color_block": block_members(review, [name for name in full_features if name.startswith("peak_color_")]),
        "context_block": block_members(review, ["observation_count", "time_span", "total_snr"]),
    }
    review_blocks = {name: features for name, features in review_blocks.items() if features}

    return {
        "manifest_version": "phase2_tier1_v1",
        "source_importance_file": INPUT_PATH,
        "importance_scope_note": payload["importance_scope_note"],
        "full_baseline_features": full_features,
        "accept_features": accept,
        "review_features": review,
        "reject_features": reject,
        "working_feature_set": working,
        "top_permutation_features": [row["feature"] for row in payload["permutation_importance"][:10]],
        "review_ablation_blocks": review_blocks,
        "drop_rule": {
            "primary_metric": "pr_auc",
            "secondary_metric": "f1",
            "guidance": "Keep accept + review as the provisional working set. Drop only reject by default. Prune review features only through second-pass ablation.",
        },
    }


def main() -> None:
    with open(INPUT_PATH) as handle:
        payload = json.load(handle)
    manifest = build_manifest(payload)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
