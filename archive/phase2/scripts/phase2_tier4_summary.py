"""Aggregate Phase 2 Tier 4 outputs into one report."""

from __future__ import annotations

import json
import os
from typing import Any

from phase2_tier2_common import write_json
from phase2_tier4_common import RESULTS_DIR, ensure_output_dirs


DOMAIN_SWAP_JSON_PATH = f"{RESULTS_DIR}/domain_swap_metrics.json"
MIXED_JSON_PATH = f"{RESULTS_DIR}/mixed_training_metrics.json"
IMPORTANCE_JSON_PATH = f"{RESULTS_DIR}/importance_domain_metrics.json"
MINIMAL_JSON_PATH = f"{RESULTS_DIR}/minimal_domain_metrics.json"
SHIFT_JSON_PATH = f"{RESULTS_DIR}/shift_test_metrics.json"

MASTER_JSON_PATH = f"{RESULTS_DIR}/phase2_tier4_master_summary.json"
REPORT_PATH = f"{RESULTS_DIR}/phase2_tier4_report.md"


def load_required_json(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label} output: {path}")
    with open(path) as handle:
        return json.load(handle)


def write_report(payload: dict[str, Any]) -> None:
    domain_rows = payload["domain_swap"]["rows"]
    mixed_rows = payload["mixed_training"]["summary_rows"]
    importance_score = payload["importance_domain"]["feature_stability_score"]
    importance_domains = payload["importance_domain"]["domain_rows"]
    minimal_rows = payload["minimal_domain"]["summary_rows"]
    shift_rows = payload["shift_test"]["rows"]

    best_generalization = max(domain_rows, key=lambda row: row["f1"])
    worst_generalization = min(domain_rows, key=lambda row: row["f1"])
    best_mixed = max(mixed_rows, key=lambda row: row["mean_f1"])
    best_subset = max(minimal_rows, key=lambda row: row["mean_f1"])
    worst_shift = max(shift_rows, key=lambda row: row["f1_drop"])
    top_domain_features = importance_domains[0]["top_features"] if importance_domains else []

    lines = [
        "# Phase 2 Tier 4 Report",
        "",
        "## Domain swap",
        f"- Best test domain: {best_generalization['test_domain']} with F1 {best_generalization['f1']:.6f}.",
        f"- Largest degradation: {worst_generalization['test_domain']} with delta F1 {worst_generalization['delta_f1']:+.6f}.",
        "",
        "## Mixed training",
        f"- Best mixed-training regime by mean F1: {best_mixed['train_domain']} ({best_mixed['mean_f1']:.6f}).",
        "",
        "## Feature stability",
        f"- Feature stability score: {importance_score:.6f}.",
        f"- Representative top features: {', '.join(top_domain_features)}.",
        "",
        "## Minimal-domain stability",
        f"- Best subset across domains: {best_subset['subset_name']} ({best_subset['feature_count']} features) with mean F1 {best_subset['mean_f1']:.6f}.",
        "",
        "## Distribution shifts",
        f"- Hardest shift: {worst_shift['shift_name']} with F1 drop {worst_shift['f1_drop']:.6f}.",
        "",
        "## Interpretation",
        "- Tier 4 uses SPCC compact variants written to disk and includes PLAsTiCC whenever the compact PLAsTiCC table has been built.",
        "- Stable feature rankings across noisy, missing-band, and cross-survey domains support the claim that the compact representation is not relying only on one clean-survey artifact.",
    ]

    with open(REPORT_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_output_dirs()
    payload = {
        "domain_swap": load_required_json(DOMAIN_SWAP_JSON_PATH, "domain swap"),
        "mixed_training": load_required_json(MIXED_JSON_PATH, "mixed training"),
        "importance_domain": load_required_json(IMPORTANCE_JSON_PATH, "importance by domain"),
        "minimal_domain": load_required_json(MINIMAL_JSON_PATH, "minimal-domain stability"),
        "shift_test": load_required_json(SHIFT_JSON_PATH, "distribution shift"),
    }
    write_json(MASTER_JSON_PATH, payload)
    write_report(payload)
    print(json.dumps({"master_json_path": MASTER_JSON_PATH, "report_path": REPORT_PATH}, indent=2))


if __name__ == "__main__":
    main()
