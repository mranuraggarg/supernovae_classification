#!/usr/bin/env python3
"""
Phase 2 Tier 4 small PLAsTiCC windowing audit.

Purpose
-------
Before building a full windowed PLAsTiCC compact-feature pipeline, test whether
cropping each PLAsTiCC object around its transient peak actually changes the
problematic observation-window quantities.

This script compares the original PLAsTiCC raw object history against simple
peak-centered windows:

    peak_time +/- 60 days
    peak_time +/- 90 days
    peak_time +/- 120 days
    peak_time +/- 180 days

The goal is to check whether windowing brings PLAsTiCC closer to the SPCC-like
event duration before rerunning domain-swap experiments.

Outputs
-------
results/phase2_tier4_windowed_plasticc_audit/windowed_plasticc_audit.json
results/phase2_tier4_windowed_plasticc_audit/windowed_plasticc_audit_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase2_tier4_make_variants import (
    PLASTICC_LIGHTCURVE_PATH,
    PLASTICC_METADATA_PATH,
    PLASTICC_PASSBANDS,
)


OUT_DIR = Path("results/phase2_tier4_windowed_plasticc_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUT_DIR / "windowed_plasticc_audit.json"
REPORT_OUT = OUT_DIR / "windowed_plasticc_audit_report.md"

WINDOW_HALFWIDTHS_DAYS = [60.0, 90.0, 120.0, 180.0]
BANDS = ["g", "r", "i", "z"]


def _passband_lookup() -> dict[int, str]:
    return {band_id: band_name for band_name, band_id in PLASTICC_PASSBANDS.items()}


def _safe_quantiles(values: list[float | None]) -> dict[str, float | None]:
    clean = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=float,
    )
    if clean.size == 0:
        return {
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    return {
        "min": float(np.nanmin(clean)),
        "p25": float(np.nanpercentile(clean, 25)),
        "p50": float(np.nanpercentile(clean, 50)),
        "p75": float(np.nanpercentile(clean, 75)),
        "p95": float(np.nanpercentile(clean, 95)),
        "max": float(np.nanmax(clean)),
    }


def _time_span(group: pd.DataFrame) -> float | None:
    if len(group) == 0:
        return None
    return float(group["mjd"].max() - group["mjd"].min())


def _anchor_time(group: pd.DataFrame, detected_only: bool = True) -> float | None:
    """
    Find transient-centered anchor time.

    Default:
    - use detected observations if available
    - choose the positive maximum-flux point
    - fallback to maximum flux among all observations
    """

    candidate = group.copy()

    if detected_only and "detected" in candidate.columns:
        detected = candidate[candidate["detected"].astype(int) == 1]
        if len(detected) > 0:
            candidate = detected

    positive = candidate[candidate["flux"] > 0]
    if len(positive) > 0:
        candidate = positive

    if len(candidate) == 0:
        return None

    idx = candidate["flux"].idxmax()
    return float(candidate.loc[idx, "mjd"])


def _band_counts(group: pd.DataFrame) -> dict[str, int]:
    lookup = _passband_lookup()
    counts = {band: 0 for band in BANDS}

    for passband, count in group["passband"].value_counts().items():
        band = lookup.get(int(passband))
        if band in counts:
            counts[band] = int(count)

    return counts


def _active_mask(group: pd.DataFrame, snr_threshold: float = 3.0) -> pd.Series:
    flux = group["flux"].astype(float)
    flux_err = group["flux_err"].astype(float)
    return (flux > 0) & (flux_err > 0) & ((flux / flux_err) >= snr_threshold)


def _active_time_span(group: pd.DataFrame) -> float | None:
    if len(group) == 0:
        return None

    active = group[_active_mask(group)]
    if len(active) == 0:
        return 0.0

    return float(active["mjd"].max() - active["mjd"].min())


def _active_count(group: pd.DataFrame) -> int:
    if len(group) == 0:
        return 0
    return int(_active_mask(group).sum())


def _peak_time_relative(group: pd.DataFrame) -> float | None:
    """
    Peak time relative to first observation inside this group.
    This is comparable to the compact-feature convention.
    """

    if len(group) == 0:
        return None

    positive = group[group["flux"] > 0]
    candidate = positive if len(positive) > 0 else group

    idx = candidate["flux"].idxmax()
    return float(candidate.loc[idx, "mjd"] - group["mjd"].min())


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    numeric_keys = [
        "obs_count",
        "active_count",
        "active_fraction",
        "time_span",
        "active_time_span",
        "peak_time_relative",
    ]

    for key in numeric_keys:
        summary[key] = _safe_quantiles([record.get(key) for record in records])

    for band in BANDS:
        summary[f"{band}_count"] = _safe_quantiles(
            [record["band_counts"].get(band, 0) for record in records]
        )
        summary[f"{band}_zero_count"] = int(
            sum(1 for record in records if record["band_counts"].get(band, 0) == 0)
        )

    labels = pd.Series([record["label_name"] for record in records], dtype=str)
    summary["label_counts"] = labels.value_counts().to_dict()

    return summary


def _record_for_group(
    object_id: int,
    label_name: str,
    group: pd.DataFrame,
) -> dict[str, Any]:
    obs_count = len(group)
    active_count = _active_count(group)

    return {
        "object_id": int(object_id),
        "label_name": label_name,
        "obs_count": int(obs_count),
        "active_count": int(active_count),
        "active_fraction": float(active_count / obs_count) if obs_count else 0.0,
        "time_span": _time_span(group),
        "active_time_span": _active_time_span(group),
        "peak_time_relative": _peak_time_relative(group),
        "band_counts": _band_counts(group),
    }


def build_audit() -> dict[str, Any]:
    print("[INFO] Loading PLAsTiCC light curves...")
    lightcurves = pd.read_csv(PLASTICC_LIGHTCURVE_PATH)
    metadata = pd.read_csv(PLASTICC_METADATA_PATH)

    meta_lookup = metadata.set_index("object_id").to_dict("index")

    payload: dict[str, Any] = {
        "source_lightcurve_path": PLASTICC_LIGHTCURVE_PATH,
        "source_metadata_path": PLASTICC_METADATA_PATH,
        "window_halfwidths_days": WINDOW_HALFWIDTHS_DAYS,
        "baseline": {},
        "windowed": {},
    }

    baseline_records: list[dict[str, Any]] = []
    windowed_records: dict[str, list[dict[str, Any]]] = {
        f"pm{int(width)}": [] for width in WINDOW_HALFWIDTHS_DAYS
    }

    for object_id, group in lightcurves.groupby("object_id", sort=True):
        if object_id not in meta_lookup:
            continue

        meta = meta_lookup[object_id]
        label_name = "Ia" if int(meta["target"]) == 90 else "non-Ia"

        baseline_records.append(
            _record_for_group(
                object_id=int(object_id),
                label_name=label_name,
                group=group,
            )
        )

        anchor = _anchor_time(group, detected_only=True)
        if anchor is None:
            continue

        for width in WINDOW_HALFWIDTHS_DAYS:
            key = f"pm{int(width)}"
            lo = anchor - width
            hi = anchor + width

            window = group[(group["mjd"] >= lo) & (group["mjd"] <= hi)].copy()

            windowed_records[key].append(
                _record_for_group(
                    object_id=int(object_id),
                    label_name=label_name,
                    group=window,
                )
            )

    payload["baseline"] = {
        "record_count": len(baseline_records),
        "summary": _summarize_records(baseline_records),
    }

    for key, records in windowed_records.items():
        payload["windowed"][key] = {
            "record_count": len(records),
            "summary": _summarize_records(records),
        }

    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _q(summary: dict[str, Any], key: str, q: str = "p50") -> str:
    return _fmt(summary[key].get(q))


def write_report(payload: dict[str, Any]) -> None:
    lines: list[str] = []

    lines += [
        "# Phase 2 Tier 4 Windowed PLAsTiCC Audit",
        "",
        "This audit tests whether peak-centered windowing makes PLAsTiCC event windows closer to SPCC-like transient windows.",
        "",
        "## Main window comparison",
        "",
        "| window | records | obs p50 | active obs p50 | active fraction p50 | time span p50 | active span p50 | peak time p50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    baseline = payload["baseline"]
    baseline_summary = baseline["summary"]

    lines.append(
        "| original | "
        f"{baseline['record_count']} | "
        f"{_q(baseline_summary, 'obs_count')} | "
        f"{_q(baseline_summary, 'active_count')} | "
        f"{_q(baseline_summary, 'active_fraction')} | "
        f"{_q(baseline_summary, 'time_span')} | "
        f"{_q(baseline_summary, 'active_time_span')} | "
        f"{_q(baseline_summary, 'peak_time_relative')} |"
    )

    for key, item in payload["windowed"].items():
        summary = item["summary"]
        lines.append(
            f"| {key} | "
            f"{item['record_count']} | "
            f"{_q(summary, 'obs_count')} | "
            f"{_q(summary, 'active_count')} | "
            f"{_q(summary, 'active_fraction')} | "
            f"{_q(summary, 'time_span')} | "
            f"{_q(summary, 'active_time_span')} | "
            f"{_q(summary, 'peak_time_relative')} |"
        )

    lines += [
        "",
        "## Per-band median observation counts",
        "",
        "| window | g p50 | r p50 | i p50 | z p50 | g zero | r zero | i zero | z zero |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    lines.append(
        "| original | "
        f"{_q(baseline_summary, 'g_count')} | "
        f"{_q(baseline_summary, 'r_count')} | "
        f"{_q(baseline_summary, 'i_count')} | "
        f"{_q(baseline_summary, 'z_count')} | "
        f"{baseline_summary['g_zero_count']} | "
        f"{baseline_summary['r_zero_count']} | "
        f"{baseline_summary['i_zero_count']} | "
        f"{baseline_summary['z_zero_count']} |"
    )

    for key, item in payload["windowed"].items():
        summary = item["summary"]
        lines.append(
            f"| {key} | "
            f"{_q(summary, 'g_count')} | "
            f"{_q(summary, 'r_count')} | "
            f"{_q(summary, 'i_count')} | "
            f"{_q(summary, 'z_count')} | "
            f"{summary['g_zero_count']} | "
            f"{summary['r_zero_count']} | "
            f"{summary['i_zero_count']} | "
            f"{summary['z_zero_count']} |"
        )

    lines += [
        "",
        "## Label counts",
        "",
        f"Original: {baseline_summary['label_counts']}",
        "",
    ]

    for key, item in payload["windowed"].items():
        lines.append(f"{key}: {item['summary']['label_counts']}")
        lines.append("")

    lines += [
        "## Interpretation guide",
        "",
        "- If windowing reduces PLAsTiCC time_span p50 from ~900 days to ~100-200 days, the window hypothesis is supported.",
        "- If active_time_span also becomes closer to SPCC, then compact features are more likely to describe the transient rather than the full survey history.",
        "- If many g/r/i/z zero-count events appear under a narrow window, the window is too aggressive.",
        "- The best trial window is usually the smallest window that preserves reasonable per-band support.",
        "",
    ]

    REPORT_OUT.write_text("\n".join(lines))


def main() -> None:
    payload = build_audit()

    JSON_OUT.write_text(json.dumps(payload, indent=2))
    write_report(payload)

    print(
        json.dumps(
            {
                "json_path": str(JSON_OUT),
                "report_path": str(REPORT_OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
