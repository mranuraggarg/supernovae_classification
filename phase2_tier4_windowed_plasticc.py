#!/usr/bin/env python3
"""
Phase 2 Tier 4 PLAsTiCC windowed compact-feature builder.

Purpose
-------
Build alternative PLAsTiCC compact-feature tables after cropping each object
to a transient-centered window.

This tests whether SPCC -> PLAsTiCC transfer is hindered because PLAsTiCC
compact features were previously computed over the full survey history rather
than a transient-scale event window.

Outputs
-------
data/PLAsTiCC/features/windowed/compact_features_window_detected_max_flux_pm90.csv
data/PLAsTiCC/features/windowed/compact_features_window_detected_max_flux_pm120.csv
data/PLAsTiCC/features/windowed/compact_features_window_detected_max_flux_pm180.csv

results/phase2_tier4_windowed_plasticc/windowed_plasticc_manifest.json
results/phase2_tier4_windowed_plasticc/windowed_plasticc_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase2_tier4_make_variants import (
    COMPACT_FEATURES,
    PLASTICC_LIGHTCURVE_PATH,
    PLASTICC_METADATA_PATH,
    PLASTICC_PASSBANDS,
    _build_compact_row_from_observations,
    write_feature_rows,
)


OUT_DIR = Path("results/phase2_tier4_windowed_plasticc")
FEATURE_DIR = Path("data/PLAsTiCC/features/windowed")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_HALFWIDTHS_DAYS = [90.0, 120.0, 180.0]
ANCHOR_METHOD = "detected_max_flux"


def _label_from_target(target: int) -> str:
    return "Ia" if int(target) == 90 else "non-Ia"


def _redshift_from_meta(meta: dict[str, Any]) -> float:
    specz = meta.get("hostgal_specz")
    photoz = meta.get("hostgal_photoz")

    if pd.notna(specz) and float(specz) > 0:
        return float(specz)

    if pd.notna(photoz):
        return float(photoz)

    return float("nan")


def _passband_lookup() -> dict[int, str]:
    return {band_id: band_name for band_name, band_id in PLASTICC_PASSBANDS.items()}


def _group_to_observations(group: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = _passband_lookup()
    observations: list[dict[str, Any]] = []

    for _, row in group.iterrows():
        passband = int(row["passband"])
        if passband not in lookup:
            continue

        observations.append(
            {
                "time": float(row["mjd"]),
                "band": lookup[passband],
                "flux": float(row["flux"]),
                "flux_err": float(row["flux_err"]),
                "detected": int(row["detected"]),
            }
        )

    return observations


def _anchor_time(observations: list[dict[str, Any]]) -> float | None:
    """
    Anchor on the strongest detected positive-flux observation.

    Fallback order:
    1. detected == 1 and flux > 0
    2. detected == 1
    3. flux > 0
    4. all observations
    """

    if not observations:
        return None

    detected_positive = [
        obs for obs in observations
        if int(obs.get("detected", 0)) == 1 and float(obs["flux"]) > 0
    ]
    detected_any = [
        obs for obs in observations
        if int(obs.get("detected", 0)) == 1
    ]
    positive_any = [
        obs for obs in observations
        if float(obs["flux"]) > 0
    ]

    if detected_positive:
        candidates = detected_positive
    elif detected_any:
        candidates = detected_any
    elif positive_any:
        candidates = positive_any
    else:
        candidates = observations

    best = max(candidates, key=lambda obs: float(obs["flux"]))
    return float(best["time"])


def _window_observations(
    observations: list[dict[str, Any]],
    anchor_time: float,
    halfwidth_days: float,
) -> list[dict[str, Any]]:
    lo = anchor_time - halfwidth_days
    hi = anchor_time + halfwidth_days

    return [
        dict(obs)
        for obs in observations
        if lo <= float(obs["time"]) <= hi
    ]


def _time_span(observations: list[dict[str, Any]]) -> float | None:
    if not observations:
        return None

    times = [float(obs["time"]) for obs in observations]
    return float(max(times) - min(times))


def _count_by_band(observations: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"g": 0, "r": 0, "i": 0, "z": 0}

    for obs in observations:
        band = str(obs["band"])
        if band in counts:
            counts[band] += 1

    return counts


def _active_count(observations: list[dict[str, Any]]) -> int:
    count = 0

    for obs in observations:
        flux = float(obs["flux"])
        flux_err = float(obs["flux_err"])

        if flux > 0 and flux_err > 0 and (flux / flux_err) >= 3.0:
            count += 1

    return count


def _safe_quantiles(values: list[float | None]) -> dict[str, float | None]:
    clean = np.asarray(
        [
            float(v)
            for v in values
            if v is not None and np.isfinite(float(v))
        ],
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


def _summarize_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    summary["object_count"] = len(records)
    summary["feature_row_count"] = int(sum(1 for r in records if r["feature_built"]))

    labels = pd.Series([r["label_name"] for r in records], dtype=str)
    built_labels = pd.Series(
        [r["label_name"] for r in records if r["feature_built"]],
        dtype=str,
    )

    summary["label_counts"] = labels.value_counts().to_dict()
    summary["built_label_counts"] = built_labels.value_counts().to_dict()

    for key in [
        "raw_obs_count",
        "window_obs_count",
        "active_obs_count",
        "raw_time_span",
        "window_time_span",
        "anchor_relative_time",
    ]:
        summary[key] = _safe_quantiles([r[key] for r in records])

    for band in ["g", "r", "i", "z"]:
        summary[f"{band}_window_count"] = _safe_quantiles(
            [r["window_band_counts"].get(band, 0) for r in records]
        )
        summary[f"{band}_zero_window_count"] = int(
            sum(1 for r in records if r["window_band_counts"].get(band, 0) == 0)
        )

    return summary


def build_windowed_features(
    *,
    halfwidth_days: float,
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_lookup = metadata.set_index("object_id").to_dict("index")

    feature_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for object_id, group in lightcurves.groupby("object_id", sort=True):
        if object_id not in meta_lookup:
            continue

        meta = meta_lookup[object_id]
        label_name = _label_from_target(int(meta["target"]))

        observations = _group_to_observations(group)
        raw_span = _time_span(observations)
        raw_count = len(observations)

        anchor = _anchor_time(observations)
        if anchor is None:
            continue

        first_time = min(float(obs["time"]) for obs in observations) if observations else anchor
        anchor_relative_time = float(anchor - first_time)

        windowed = _window_observations(
            observations=observations,
            anchor_time=anchor,
            halfwidth_days=halfwidth_days,
        )

        compact_row = _build_compact_row_from_observations(
            snid=int(object_id),
            label_name=label_name,
            sim_z=_redshift_from_meta(meta),
            observations=windowed,
        )

        feature_built = compact_row is not None
        if compact_row is not None:
            feature_rows.append(compact_row)

        diagnostics.append(
            {
                "object_id": int(object_id),
                "label_name": label_name,
                "raw_obs_count": raw_count,
                "window_obs_count": len(windowed),
                "active_obs_count": _active_count(windowed),
                "raw_time_span": raw_span,
                "window_time_span": _time_span(windowed),
                "anchor_relative_time": anchor_relative_time,
                "window_band_counts": _count_by_band(windowed),
                "feature_built": feature_built,
            }
        )

    summary = _summarize_diagnostics(diagnostics)
    return feature_rows, summary


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _q(summary: dict[str, Any], key: str, q: str = "p50") -> str:
    return _fmt(summary[key].get(q))


def write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Phase 2 Tier 4 Windowed PLAsTiCC Feature Tables",
        "",
        "This report summarizes PLAsTiCC compact-feature tables built after peak-centered windowing.",
        "",
        "| window | feature rows | Ia rows | non-Ia rows | obs p50 | active p50 | time span p50 | g zero | r zero | i zero | z zero |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, entry in manifest["outputs"].items():
        summary = entry["summary"]
        labels = summary.get("built_label_counts", {})

        lines.append(
            f"| {key} | "
            f"{summary.get('feature_row_count')} | "
            f"{labels.get('Ia', 0)} | "
            f"{labels.get('non-Ia', 0)} | "
            f"{_q(summary, 'window_obs_count')} | "
            f"{_q(summary, 'active_obs_count')} | "
            f"{_q(summary, 'window_time_span')} | "
            f"{summary.get('g_zero_window_count')} | "
            f"{summary.get('r_zero_window_count')} | "
            f"{summary.get('i_zero_window_count')} | "
            f"{summary.get('z_zero_window_count')} |"
        )

    lines += [
        "",
        "## Output paths",
        "",
    ]

    for key, entry in manifest["outputs"].items():
        lines += [
            f"### {key}",
            "",
            f"Feature table: `{entry['feature_path']}`",
            "",
        ]

    (OUT_DIR / "windowed_plasticc_report.md").write_text("\n".join(lines))


def main() -> None:
    print("[INFO] Loading PLAsTiCC light curves and metadata...")
    lightcurves = pd.read_csv(PLASTICC_LIGHTCURVE_PATH)
    metadata = pd.read_csv(PLASTICC_METADATA_PATH)

    manifest: dict[str, Any] = {
        "trial": "windowed_plasticc",
        "anchor_method": ANCHOR_METHOD,
        "source_lightcurve_path": PLASTICC_LIGHTCURVE_PATH,
        "source_metadata_path": PLASTICC_METADATA_PATH,
        "compact_features": list(COMPACT_FEATURES),
        "outputs": {},
    }

    for halfwidth_days in WINDOW_HALFWIDTHS_DAYS:
        tag = f"{ANCHOR_METHOD}_pm{int(halfwidth_days)}"
        print(f"[INFO] Building {tag}...")

        feature_rows, summary = build_windowed_features(
            halfwidth_days=halfwidth_days,
            lightcurves=lightcurves,
            metadata=metadata,
        )

        feature_path = FEATURE_DIR / f"compact_features_window_{tag}.csv"
        write_feature_rows(str(feature_path), feature_rows)

        manifest["outputs"][tag] = {
            "halfwidth_days": halfwidth_days,
            "feature_path": str(feature_path),
            "summary": summary,
        }

        print(
            f"[OK] {tag}: wrote {len(feature_rows)} rows to {feature_path}"
        )

    manifest_path = OUT_DIR / "windowed_plasticc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    write_report(manifest)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "report_path": str(OUT_DIR / "windowed_plasticc_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()