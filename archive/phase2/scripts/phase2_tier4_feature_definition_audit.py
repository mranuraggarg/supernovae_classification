#!/usr/bin/env python3
"""
Phase 2 Tier 4 feature-definition audit.

Purpose
-------
Audit whether the shared compact-feature builder has the same effective meaning
for SPCC and PLAsTiCC.

This does not evaluate model performance. It checks the observation-level inputs
that lead to compact features:

- raw observation count
- active observation count
- active fraction
- detected-flag agreement for PLAsTiCC
- event time span
- active-window time span
- per-band raw counts
- per-band active counts
- per-band detected counts
- per-band peak timing
- fallback risk from sparse active support

Output
------
results/phase2_tier4/feature_definition_audit.json
results/phase2_tier4/feature_definition_audit_report.md
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase2_tier4_make_variants import (
    PLASTICC_LIGHTCURVE_PATH,
    PLASTICC_METADATA_PATH,
    PLASTICC_PASSBANDS,
    SPCC_BANDS,
    SPCC_RAW_GLOB,
    clean_event,
    iter_spcc_files,
    load_spcc_raw_event,
)


OUT_DIR = Path("results/phase2_tier4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUT_DIR / "feature_definition_audit.json"
REPORT_OUT = OUT_DIR / "feature_definition_audit_report.md"

SNR_THRESHOLD = 3.0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _snr_active(obs: dict[str, Any]) -> bool:
    flux = float(obs["flux"])
    flux_err = float(obs["flux_err"])
    if not _finite(flux) or not _finite(flux_err):
        return False
    if flux_err <= 0:
        return False
    return flux > 0 and (flux / flux_err) >= SNR_THRESHOLD


def _safe_quantiles(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([v for v in values if _finite(v)], dtype=float)
    if clean.size == 0:
        return {"min": None, "p25": None, "p50": None, "p75": None, "p95": None, "max": None}
    return {
        "min": float(np.nanmin(clean)),
        "p25": float(np.nanpercentile(clean, 25)),
        "p50": float(np.nanpercentile(clean, 50)),
        "p75": float(np.nanpercentile(clean, 75)),
        "p95": float(np.nanpercentile(clean, 95)),
        "max": float(np.nanmax(clean)),
    }


def _summarize_records(records: list[dict[str, Any]], survey_name: str) -> dict[str, Any]:
    numeric_keys = [
        "raw_obs_count",
        "active_obs_count",
        "active_fraction",
        "raw_time_span",
        "active_time_span",
        "g_raw_count",
        "r_raw_count",
        "i_raw_count",
        "z_raw_count",
        "g_active_count",
        "r_active_count",
        "i_active_count",
        "z_active_count",
        "g_peak_time",
        "r_peak_time",
        "i_peak_time",
        "z_peak_time",
        "g_active_peak_time",
        "r_active_peak_time",
        "i_active_peak_time",
        "z_active_peak_time",
    ]

    summary: dict[str, Any] = {
        "survey": survey_name,
        "row_count": len(records),
        "label_counts": dict(Counter(str(r.get("label_name", "unknown")) for r in records)),
        "quantiles": {},
    }

    for key in numeric_keys:
        summary["quantiles"][key] = _safe_quantiles([float(r[key]) for r in records if r.get(key) is not None])

    if survey_name.lower() == "plasticc":
        extra_keys = [
            "detected_obs_count",
            "detected_fraction",
            "snr_detected_agreement_fraction",
            "detected_not_snr_count",
            "snr_not_detected_count",
            "g_detected_count",
            "r_detected_count",
            "i_detected_count",
            "z_detected_count",
        ]
        for key in extra_keys:
            summary["quantiles"][key] = _safe_quantiles([float(r[key]) for r in records if r.get(key) is not None])

    sparse_flags = Counter()
    for r in records:
        for band in SPCC_BANDS:
            if int(r[f"{band}_active_count"]) == 0:
                sparse_flags[f"{band}_zero_active"] += 1
            if int(r[f"{band}_active_count"]) < 2:
                sparse_flags[f"{band}_lt2_active"] += 1
    summary["sparse_active_flags"] = dict(sparse_flags)

    return summary


def _peak_time(observations: list[dict[str, Any]], band: str, active_only: bool) -> float | None:
    rows = [obs for obs in observations if obs["band"] == band]
    if active_only:
        rows = [obs for obs in rows if _snr_active(obs)]
    if not rows:
        return None

    # Match compact-builder spirit: use maximum flux among available rows.
    best = max(rows, key=lambda obs: float(obs["flux"]))
    first_time = min(float(obs["time"]) for obs in observations)
    return float(best["time"]) - first_time


def _event_record(
    *,
    snid: int,
    label_name: str,
    observations: list[dict[str, Any]],
    is_plasticc: bool,
) -> dict[str, Any]:
    finite_obs = [
        obs for obs in observations
        if _finite(obs.get("time")) and _finite(obs.get("flux")) and _finite(obs.get("flux_err")) and obs.get("band") in SPCC_BANDS
    ]

    raw_times = [float(obs["time"]) for obs in finite_obs]
    active_obs = [obs for obs in finite_obs if _snr_active(obs)]
    active_times = [float(obs["time"]) for obs in active_obs]

    record: dict[str, Any] = {
        "snid": int(snid),
        "label_name": str(label_name),
        "raw_obs_count": len(finite_obs),
        "active_obs_count": len(active_obs),
        "active_fraction": len(active_obs) / len(finite_obs) if finite_obs else 0.0,
        "raw_time_span": max(raw_times) - min(raw_times) if raw_times else None,
        "active_time_span": max(active_times) - min(active_times) if active_times else None,
    }

    for band in SPCC_BANDS:
        raw_band = [obs for obs in finite_obs if obs["band"] == band]
        active_band = [obs for obs in raw_band if _snr_active(obs)]
        record[f"{band}_raw_count"] = len(raw_band)
        record[f"{band}_active_count"] = len(active_band)
        record[f"{band}_peak_time"] = _peak_time(finite_obs, band, active_only=False)
        record[f"{band}_active_peak_time"] = _peak_time(finite_obs, band, active_only=True)

    if is_plasticc:
        detected_obs = [obs for obs in finite_obs if int(obs.get("detected", 0)) == 1]
        record["detected_obs_count"] = len(detected_obs)
        record["detected_fraction"] = len(detected_obs) / len(finite_obs) if finite_obs else 0.0

        agreements = 0
        detected_not_snr = 0
        snr_not_detected = 0

        for obs in finite_obs:
            detected = int(obs.get("detected", 0)) == 1
            active = _snr_active(obs)
            if detected == active:
                agreements += 1
            elif detected and not active:
                detected_not_snr += 1
            elif active and not detected:
                snr_not_detected += 1

        record["snr_detected_agreement_fraction"] = agreements / len(finite_obs) if finite_obs else 0.0
        record["detected_not_snr_count"] = detected_not_snr
        record["snr_not_detected_count"] = snr_not_detected

        for band in SPCC_BANDS:
            detected_band = [obs for obs in detected_obs if obs["band"] == band]
            record[f"{band}_detected_count"] = len(detected_band)

    return record


def build_spcc_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for path in iter_spcc_files(SPCC_RAW_GLOB):
        cleaning_result = clean_event(load_spcc_raw_event(path), min_observations_per_event=1)
        if not cleaning_result.accepted or cleaning_result.event is None or cleaning_result.event.sim_type is None:
            continue

        observations = [
            {
                "time": obs.mjd,
                "band": obs.band,
                "flux": obs.flux,
                "flux_err": obs.flux_err,
            }
            for obs in cleaning_result.event.observations
        ]

        records.append(
            _event_record(
                snid=int(cleaning_result.event.snid),
                label_name=str(cleaning_result.event.sim_type),
                observations=observations,
                is_plasticc=False,
            )
        )

    return records


def build_plasticc_records() -> list[dict[str, Any]]:
    lightcurves = pd.read_csv(PLASTICC_LIGHTCURVE_PATH)
    metadata = pd.read_csv(PLASTICC_METADATA_PATH)

    passband_to_name = {value: key for key, value in PLASTICC_PASSBANDS.items()}
    meta_lookup = metadata.set_index("object_id").to_dict("index")

    records: list[dict[str, Any]] = []

    for object_id, group in lightcurves.groupby("object_id", sort=True):
        if object_id not in meta_lookup:
            continue

        meta = meta_lookup[object_id]
        observations = []

        for _, row in group.iterrows():
            passband = int(row["passband"])
            if passband not in passband_to_name:
                continue

            observations.append(
                {
                    "time": float(row["mjd"]),
                    "band": passband_to_name[passband],
                    "flux": float(row["flux"]),
                    "flux_err": float(row["flux_err"]),
                    "detected": int(row["detected"]),
                }
            )

        label_name = "Ia" if int(meta["target"]) == 90 else "non-Ia"

        records.append(
            _event_record(
                snid=int(object_id),
                label_name=label_name,
                observations=observations,
                is_plasticc=True,
            )
        )

    return records


def _format_quantile_line(label: str, q: dict[str, Any]) -> str:
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        return f"{float(x):.6g}"

    return (
        f"| {label} | {fmt(q.get('min'))} | {fmt(q.get('p25'))} | "
        f"{fmt(q.get('p50'))} | {fmt(q.get('p75'))} | {fmt(q.get('p95'))} | {fmt(q.get('max'))} |"
    )


def write_report(payload: dict[str, Any]) -> None:
    spcc = payload["spcc_summary"]
    plasticc = payload["plasticc_summary"]

    lines = [
        "# Phase 2 Tier 4 Feature-Definition Audit",
        "",
        "This audit compares the effective observation-level inputs used by the shared compact-feature builder.",
        "",
        f"SPCC records: {spcc['row_count']}",
        f"PLAsTiCC records: {plasticc['row_count']}",
        "",
        "## Label counts",
        "",
        f"SPCC: {spcc['label_counts']}",
        "",
        f"PLAsTiCC: {plasticc['label_counts']}",
        "",
        "## Core observation-window quantities",
        "",
        "| quantity | survey | min | p25 | p50 | p75 | p95 | max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key in [
        "raw_obs_count",
        "active_obs_count",
        "active_fraction",
        "raw_time_span",
        "active_time_span",
    ]:
        lines.append(_format_quantile_line(f"{key} / SPCC", spcc["quantiles"][key]))
        lines.append(_format_quantile_line(f"{key} / PLAsTiCC", plasticc["quantiles"][key]))

    lines += [
        "",
        "## Per-band active-count medians",
        "",
        "| band | SPCC active p50 | PLAsTiCC active p50 | SPCC raw p50 | PLAsTiCC raw p50 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for band in SPCC_BANDS:
        lines.append(
            f"| {band} | "
            f"{spcc['quantiles'][f'{band}_active_count']['p50']} | "
            f"{plasticc['quantiles'][f'{band}_active_count']['p50']} | "
            f"{spcc['quantiles'][f'{band}_raw_count']['p50']} | "
            f"{plasticc['quantiles'][f'{band}_raw_count']['p50']} |"
        )

    lines += [
        "",
        "## Peak-time medians",
        "",
        "| feature | SPCC p50 | PLAsTiCC p50 |",
        "| --- | ---: | ---: |",
    ]

    for band in SPCC_BANDS:
        lines.append(
            f"| {band}_peak_time | "
            f"{spcc['quantiles'][f'{band}_peak_time']['p50']} | "
            f"{plasticc['quantiles'][f'{band}_peak_time']['p50']} |"
        )
        lines.append(
            f"| {band}_active_peak_time | "
            f"{spcc['quantiles'][f'{band}_active_peak_time']['p50']} | "
            f"{plasticc['quantiles'][f'{band}_active_peak_time']['p50']} |"
        )

    lines += [
        "",
        "## PLAsTiCC detected-flag vs SNR-active rule",
        "",
        "| quantity | min | p25 | p50 | p75 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key in [
        "detected_obs_count",
        "detected_fraction",
        "snr_detected_agreement_fraction",
        "detected_not_snr_count",
        "snr_not_detected_count",
    ]:
        lines.append(_format_quantile_line(key, plasticc["quantiles"][key]))

    lines += [
        "",
        "## Sparse active-support flags",
        "",
        f"SPCC: {spcc['sparse_active_flags']}",
        "",
        f"PLAsTiCC: {plasticc['sparse_active_flags']}",
        "",
        "## Interpretation guide",
        "",
        "- Large raw/active time-span differences imply that the same compact feature formulas are operating over different effective event windows.",
        "- Low PLAsTiCC detected/SNR agreement implies that ignoring the `detected` flag may change the effective PLAsTiCC event definition.",
        "- Many zero-active or <2-active band flags imply that color and peak-time features may depend heavily on fallback behaviour.",
        "- Large differences between raw peak-time and active peak-time summaries imply that the active-window rule changes the phase being measured.",
        "",
    ]

    REPORT_OUT.write_text("\n".join(lines))


def main() -> None:
    print("[INFO] Building SPCC feature-definition audit records...")
    spcc_records = build_spcc_records()
    print(f"[INFO] SPCC records: {len(spcc_records)}")

    print("[INFO] Building PLAsTiCC feature-definition audit records...")
    plasticc_records = build_plasticc_records()
    print(f"[INFO] PLAsTiCC records: {len(plasticc_records)}")

    payload = {
        "snr_threshold": SNR_THRESHOLD,
        "spcc_summary": _summarize_records(spcc_records, "SPCC"),
        "plasticc_summary": _summarize_records(plasticc_records, "PLAsTiCC"),
    }

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