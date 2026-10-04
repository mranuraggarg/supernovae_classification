"""Tighten the Phase 2 feature manifest using review-ablation results."""

from __future__ import annotations

import json
import os


MANIFEST_PATH = "results/phase2_tier1/phase2_tier1_feature_manifest.json"
ABLATION_PATH = "results/phase2_tier1/phase2_tier1_review_ablation.json"
OUTPUT_PATH = "results/phase2_tier1/phase2_tier1_feature_manifest_tightened.json"


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def main() -> None:
    manifest = load_json(MANIFEST_PATH)
    ablation = load_json(ABLATION_PATH)

    safe_to_drop = set()
    review_again = set()
    keep = set()
    for row in ablation["ablation_results"]:
        if not row["name"].startswith("drop_feature__"):
            continue
        feature = row["removed_features"][0]
        if row["recommendation"] == "safe_to_drop":
            safe_to_drop.add(feature)
        elif row["recommendation"] == "review_again":
            review_again.add(feature)
        elif row["recommendation"] == "keep":
            keep.add(feature)

    compact_review = [feature for feature in manifest["review_features"] if feature not in safe_to_drop]
    compact_working = manifest["accept_features"] + compact_review
    payload = {
        "manifest_version": "phase2_tier1_v2_tightened",
        "source_manifest": MANIFEST_PATH,
        "source_ablation": ABLATION_PATH,
        "importance_scope_note": manifest["importance_scope_note"],
        "full_baseline_features": manifest["full_baseline_features"],
        "accept_features": manifest["accept_features"],
        "safe_drop_features": sorted(safe_to_drop),
        "review_again_features": [feature for feature in compact_review if feature in review_again],
        "keep_features_from_review": [feature for feature in compact_review if feature in keep],
        "compact_review_features": compact_review,
        "compact_working_feature_set": compact_working,
        "reject_features": sorted(set(manifest["reject_features"]) | safe_to_drop),
        "guidance": {
            "status": "tightened_after_second_pass_ablation",
            "rule": "Use compact_working_feature_set for the next clean rerun. Reject only the original reject set plus features marked safe_to_drop.",
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved tightened manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
