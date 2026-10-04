"""Trial utilities for ratio-based Tier 4 color features."""

from __future__ import annotations

import json
import os
from typing import Any

from phase2_tier2_common import create_context
from phase2_tier4_common import (
    PLOTS_DIR as BASE_PLOTS_DIR,
    RESULTS_DIR as BASE_RESULTS_DIR,
    domain_splits_from_variants,
    load_variant_rows,
    run_domain_experiment,
    save_csv,
    save_json,
    tier4_reference_payload,
    write_markdown,
)


RESULTS_DIR = "results/phase2_tier4_trial"
PLOTS_DIR = "plots/phase2_tier4_trial"


def ensure_output_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _invert_positive_log(value: float) -> float:
    return max((10.0 ** float(value)) - 1.0, 0.0)


def _invert_signed_log(value: float) -> float:
    sign = -1.0 if float(value) < 0 else 1.0
    return sign * ((10.0 ** abs(float(value))) - 1.0)


def _safe_log_ratio(numerator: float, denominator: float, floor: float = 1e-6) -> float:
    safe_num = max(float(numerator), floor)
    safe_den = max(float(denominator), floor)
    return float(max(min(__import__("math").log10(safe_num / safe_den), 5.0), -5.0))


def transform_rows_ratio_colors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed = []
    for row in rows:
        updated = dict(row)
        r_peak = _invert_positive_log(updated["r_peak_flux"])
        i_peak = _invert_positive_log(updated["i_peak_flux"])
        z_peak = _invert_positive_log(updated["z_peak_flux"])
        # Current compact table does not include g_peak_flux, so the trial uses mean-flux ratio for g/r.
        g_mean = max(_invert_signed_log(updated["g_mean_flux"]), 0.0)
        r_mean = max(_invert_signed_log(updated["r_mean_flux"]), 0.0)

        updated["peak_color_g_minus_r"] = _safe_log_ratio(g_mean + 1.0, r_mean + 1.0)
        updated["peak_color_r_minus_i"] = _safe_log_ratio(r_peak + 1.0, i_peak + 1.0)
        updated["peak_color_i_minus_z"] = _safe_log_ratio(i_peak + 1.0, z_peak + 1.0)
        transformed.append(updated)
    return transformed


def load_trial_domain_splits() -> tuple[Any, dict[str, dict[str, list[dict[str, Any]]]]]:
    context = create_context()
    variant_rows = load_variant_rows(require_plasticc=False)
    transformed_variants = {
        domain_name: transform_rows_ratio_colors(rows)
        for domain_name, rows in variant_rows.items()
    }
    splits = domain_splits_from_variants(context, transformed_variants)
    return context, splits


def trial_reference_payload(context: Any) -> dict[str, Any]:
    payload = tier4_reference_payload(context)
    payload["trial"] = {
        "name": "ratio_color_trial",
        "description": (
            "Replace the three compact color features with ratio/log-ratio versions while "
            "keeping the remaining compact features unchanged."
        ),
        "notes": [
            "peak_color_r_minus_i uses r_peak_flux / i_peak_flux",
            "peak_color_i_minus_z uses i_peak_flux / z_peak_flux",
            "peak_color_g_minus_r uses g_mean_flux / r_mean_flux because g_peak_flux is not part of the compact Tier 4 table",
        ],
    }
    return payload

