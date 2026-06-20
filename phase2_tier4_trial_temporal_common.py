"""Trial utilities for temporal-alignment Tier 4 features."""

from __future__ import annotations

import os
from typing import Any

from phase2_tier2_common import create_context
from phase2_tier4_common import (
    domain_splits_from_variants,
    load_variant_rows,
    run_domain_experiment,
    tier4_reference_payload,
    write_markdown,
)


RESULTS_DIR = "results/phase2_tier4_trial_temporal"
PLOTS_DIR = "plots/phase2_tier4_trial_temporal"


def ensure_output_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def transform_rows_temporal_alignment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed = []
    for row in rows:
        updated = dict(row)
        span = max(float(updated["time_span"]), 1e-6)
        r_peak = float(updated["r_time_of_peak"])
        updated["z_time_of_peak"] = (float(updated["z_time_of_peak"]) - r_peak) / span
        updated["i_time_of_peak"] = (float(updated["i_time_of_peak"]) - r_peak) / span
        updated["r_time_of_peak"] = 0.0
        transformed.append(updated)
    return transformed


def load_trial_domain_splits() -> tuple[Any, dict[str, dict[str, list[dict[str, Any]]]]]:
    context = create_context()
    variant_rows = load_variant_rows(require_plasticc=False)
    transformed_variants = {
        domain_name: transform_rows_temporal_alignment(rows)
        for domain_name, rows in variant_rows.items()
    }
    splits = domain_splits_from_variants(context, transformed_variants)
    return context, splits


def trial_reference_payload(context: Any) -> dict[str, Any]:
    payload = tier4_reference_payload(context)
    payload["trial"] = {
        "name": "temporal_alignment_trial",
        "description": (
            "Replace band-specific peak times with relative phase offsets anchored to r-band peak time."
        ),
        "notes": [
            "z_time_of_peak becomes (z_time_of_peak - r_time_of_peak) / time_span",
            "i_time_of_peak becomes (i_time_of_peak - r_time_of_peak) / time_span",
            "r_time_of_peak is set to 0 as the anchor reference",
        ],
    }
    return payload

