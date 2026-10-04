"""Build on-disk Tier 4 SPCC variants and optional PLAsTiCC compact features."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np

from feature_pipeline.cleaning.spcc_clean import clean_event
from feature_pipeline.loaders.spcc_raw import iter_spcc_files, load_spcc_raw_event
from phase2_tier2_common import COMPACT_FEATURES
from phase2_tier4_common import (
    PLASTICC_TEST_COMPACT_CSV_PATH,
    PLASTICC_TRAIN_COMPACT_CSV_PATH,
    SPCC_COMPACT_CSV_PATH,
    SPCC_NOISE_CSV_PATH,
    SPCC_NO_I_CSV_PATH,
    SPCC_NO_Z_CSV_PATH,
    SPCC_SCALED_FLUX_CSV_PATH,
    SPCC_SHORT_SPAN_CSV_PATH,
    build_spcc_variant_rows,
    compact_fieldnames,
    ensure_variant_dirs,
    write_feature_rows,
)


SPCC_RAW_GLOB = "data/spcc/raw/DES_*.DAT"
PLASTICC_LIGHTCURVE_PATH = "data/PLAsTiCC/training_set.csv"
PLASTICC_METADATA_PATH = "data/PLAsTiCC/training_set_metadata.csv"
PLASTICC_TEST_LIGHTCURVE_PATH = "data/PLAsTiCC/test_set.csv"
SUMMARY_PATH = "results/phase2_tier4/variant_manifest.json"

COLOR_CLIP_RANGE = (-5.0, 5.0)
PLASTICC_PASSBANDS = {"g": 1, "r": 2, "i": 3, "z": 4}
SPCC_BANDS = ("g", "r", "i", "z")
SNR_ACTIVE_THRESHOLD = 3.0
MIN_COLOR_FLUX_THRESHOLD = 0.0
LC_NORMALIZATION_MODES = ("none", "event_peak", "event_p95", "band_peak")
def _safe_scale(value: float, eps: float = 1e-6) -> float:
    if not math.isfinite(float(value)) or abs(float(value)) < eps:
        return 1.0
    return float(value)


def _normalize_observations(
    observations: list[dict[str, float | str | int]],
    mode: str,
) -> list[dict[str, float | str | int]]:
    if mode == "none":
        return [dict(obs) for obs in observations]
    if mode not in LC_NORMALIZATION_MODES:
        raise ValueError(f"Unknown light-curve normalization mode: {mode}")

    normalized = [dict(obs) for obs in observations]

    if mode == "event_peak":
        scale = _safe_scale(max(abs(float(obs["flux"])) for obs in normalized) if normalized else 1.0)
        for obs in normalized:
            obs["flux"] = float(obs["flux"]) / scale
            obs["flux_err"] = float(obs["flux_err"]) / scale
        return normalized

    if mode == "event_p95":
        flux_values = np.array([abs(float(obs["flux"])) for obs in normalized], dtype=float)
        scale = _safe_scale(float(np.nanpercentile(flux_values, 95)) if len(flux_values) else 1.0)
        for obs in normalized:
            obs["flux"] = float(obs["flux"]) / scale
            obs["flux_err"] = float(obs["flux_err"]) / scale
        return normalized

    if mode == "band_peak":
        for band_name in SPCC_BANDS:
            band_indices = [index for index, obs in enumerate(normalized) if obs["band"] == band_name]
            if not band_indices:
                continue
            scale = _safe_scale(max(abs(float(normalized[index]["flux"])) for index in band_indices))
            for index in band_indices:
                normalized[index]["flux"] = float(normalized[index]["flux"]) / scale
                normalized[index]["flux_err"] = float(normalized[index]["flux_err"]) / scale
        return normalized

    raise ValueError(f"Unhandled light-curve normalization mode: {mode}")


def _positive_log10_1p(value: float) -> float:
    return math.log10(1.0 + max(float(value), 0.0))


def _signed_log10_1p(value: float) -> float:
    magnitude = math.log10(1.0 + abs(float(value)))
    return math.copysign(magnitude, float(value))


def _clip_value(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _safe_color_from_peak_fluxes(flux_a: float, flux_b: float, floor: float = 1e-6) -> float:
    safe_a = max(float(flux_a), floor)
    safe_b = max(float(flux_b), floor)
    return -2.5 * math.log10(safe_a / safe_b)


def _compress_feature_row(feature_row: dict[str, Any]) -> dict[str, Any]:
    positive_log_features = {
        "g_peak_flux",
        "r_peak_flux",
        "i_peak_flux",
        "z_peak_flux",
        "g_std_flux",
        "r_std_flux",
        "i_std_flux",
        "z_std_flux",
        "i_amplitude",
    }
    signed_log_features = {"g_mean_flux", "r_mean_flux"}
    transformed = dict(feature_row)
    for name in positive_log_features:
        transformed[name] = _positive_log10_1p(transformed[name])
    for name in signed_log_features:
        transformed[name] = _signed_log10_1p(transformed[name])
    for name in ("peak_color_g_minus_r", "peak_color_r_minus_i", "peak_color_i_minus_z"):
        transformed[name] = _clip_value(transformed[name], *COLOR_CLIP_RANGE)
    return transformed


def _representative_flux_for_color(fluxes: np.ndarray) -> float:
    positive_fluxes = np.sort(fluxes[fluxes > MIN_COLOR_FLUX_THRESHOLD])[::-1]
    if len(positive_fluxes) == 0:
        return 0.0
    top_k = positive_fluxes[: min(3, len(positive_fluxes))]
    return float(np.mean(top_k))


def _build_compact_row_from_observations(
    *,
    snid: int,
    label_name: str,
    sim_z: float,
    observations: list[dict[str, float | str | int]],
) -> dict[str, Any] | None:
    filtered = [obs for obs in observations if obs["band"] in SPCC_BANDS]
    if not filtered:
        return None
    filtered.sort(key=lambda obs: float(obs["time"]))

    active = [
        obs
        for obs in filtered
        if float(obs["flux"]) > 0.0 and float(obs["flux_err"]) > 0.0 and float(obs["flux"]) / float(obs["flux_err"]) >= SNR_ACTIVE_THRESHOLD
    ]
    active_group = active if active else filtered
    event_start_time = float(active_group[0]["time"])
    event_end_time = float(active_group[-1]["time"])
    event_span = max(event_end_time - event_start_time, 0.0)

    feature_row: dict[str, Any] = {
        "snid": int(snid),
        "label_name": label_name,
        "label_id": 1 if label_name == "Ia" else 0,
        "sim_z": float(sim_z),
    }

    band_peaks = {}
    color_reference_fluxes = {}
    for band_name in SPCC_BANDS:
        band_group = [obs for obs in active_group if obs["band"] == band_name]
        if not band_group:
            band_group = [
                obs
                for obs in filtered
                if obs["band"] == band_name and event_start_time <= float(obs["time"]) <= event_end_time
            ]
        if not band_group:
            fluxes = np.zeros(1, dtype=float)
            times = np.zeros(1, dtype=float)
        else:
            fluxes = np.array([float(obs["flux"]) for obs in band_group], dtype=float)
            times = np.array([float(obs["time"]) for obs in band_group], dtype=float)
        peak_index = int(np.argmax(fluxes))
        peak_flux = float(fluxes[peak_index]) if len(fluxes) else 0.0
        band_peaks[band_name] = peak_flux
        color_reference_fluxes[band_name] = _representative_flux_for_color(fluxes)
        feature_row[f"{band_name}_peak_flux"] = peak_flux
        feature_row[f"{band_name}_mean_flux"] = float(np.mean(fluxes)) if len(fluxes) else 0.0
        feature_row[f"{band_name}_std_flux"] = float(np.std(fluxes)) if len(fluxes) else 0.0
        relative_peak_time = float(times[peak_index] - event_start_time) if len(times) else 0.0
        feature_row[f"{band_name}_time_of_peak"] = min(max(relative_peak_time, 0.0), event_span)
        if band_name == "i":
            feature_row["i_amplitude"] = float(np.max(fluxes) - np.min(fluxes)) if len(fluxes) else 0.0

    # Keep only events with usable band support for all compact color features.
    if any(color_reference_fluxes[band_name] <= 0.0 for band_name in SPCC_BANDS):
        return None

    feature_row["time_span"] = event_span
    feature_row["peak_color_g_minus_r"] = _safe_color_from_peak_fluxes(color_reference_fluxes["g"], color_reference_fluxes["r"])
    feature_row["peak_color_r_minus_i"] = _safe_color_from_peak_fluxes(color_reference_fluxes["r"], color_reference_fluxes["i"])
    feature_row["peak_color_i_minus_z"] = _safe_color_from_peak_fluxes(color_reference_fluxes["i"], color_reference_fluxes["z"])
    return _compress_feature_row(feature_row)


def build_spcc_rows(normalization_mode: str = "none") -> list[dict[str, Any]]:
    rows = []
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
        observations = _normalize_observations(observations, normalization_mode)
        row = _build_compact_row_from_observations(
            snid=int(cleaning_result.event.snid),
            label_name=str(cleaning_result.event.sim_type),
            sim_z=float(cleaning_result.event.sim_z),
            observations=observations,
        )
        if row is not None:
            rows.append(row)
    return rows


def build_plasticc_rows(lightcurve_path: str, metadata_path: str, normalization_mode: str = "none") -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required to build PLAsTiCC compact features from the raw CSVs."
        ) from exc

    if not os.path.exists(lightcurve_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"PLAsTiCC raw files not found: {lightcurve_path}, {metadata_path}")

    lightcurves = pd.read_csv(lightcurve_path)
    metadata_frame = pd.read_csv(metadata_path)
    if "target" not in metadata_frame.columns:
        raise ValueError(f"PLAsTiCC metadata file does not include target labels: {metadata_path}")
    metadata = metadata_frame[["object_id", "hostgal_photoz", "hostgal_specz", "target"]]
    meta_lookup = metadata.set_index("object_id").to_dict("index")

    grouped = lightcurves.groupby("object_id", sort=True)
    rows = []
    for object_id, group in grouped:
        if object_id not in meta_lookup:
            continue
        meta = meta_lookup[object_id]
        observations = [
            {
                "time": float(row["mjd"]),
                "band": next((name for name, band_id in PLASTICC_PASSBANDS.items() if band_id == int(row["passband"])), None),
                "flux": float(row["flux"]),
                "flux_err": float(row["flux_err"]),
                "detected": int(row["detected"]),
            }
            for _, row in group.iterrows()
            if int(row["passband"]) in PLASTICC_PASSBANDS.values()
        ]
        observations = _normalize_observations(observations, normalization_mode)
        row = _build_compact_row_from_observations(
            snid=int(object_id),
            label_name="Ia" if int(meta["target"]) == 90 else "non-Ia",
            sim_z=float(meta["hostgal_specz"] if pd.notna(meta["hostgal_specz"]) and float(meta["hostgal_specz"]) > 0 else meta["hostgal_photoz"]),
            observations=observations,
        )
        if row is not None:
            rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 Tier 4 variant CSVs.")
    parser.add_argument(
        "--skip-plasticc",
        action="store_true",
        help="Build SPCC variants only and skip PLAsTiCC compact feature export.",
    )
    parser.add_argument(
        "--lc-norm-mode",
        choices=LC_NORMALIZATION_MODES,
        default="none",
        help="Apply optional per-event light-curve normalization before compact-feature extraction.",
    )
    args = parser.parse_args()

    ensure_variant_dirs()
    os.makedirs("results/phase2_tier4", exist_ok=True)

    base_rows = build_spcc_rows(normalization_mode=args.lc_norm_mode)
    variant_rows = build_spcc_variant_rows(base_rows)

    write_feature_rows(SPCC_COMPACT_CSV_PATH, variant_rows["spcc"])
    write_feature_rows(SPCC_NOISE_CSV_PATH, variant_rows["noise"])
    write_feature_rows(SPCC_NO_Z_CSV_PATH, variant_rows["no_z"])
    write_feature_rows(SPCC_NO_I_CSV_PATH, variant_rows["no_i"])
    write_feature_rows(SPCC_SHORT_SPAN_CSV_PATH, variant_rows["short_span"])
    write_feature_rows(SPCC_SCALED_FLUX_CSV_PATH, variant_rows["flux_scale"])

    plasticc_status = {
        "train_built": False,
        "train_path": PLASTICC_TRAIN_COMPACT_CSV_PATH,
        "test_built": False,
        "test_path": PLASTICC_TEST_COMPACT_CSV_PATH,
        "test_labels_available": False,
    }
    if not args.skip_plasticc:
        plasticc_train_rows = build_plasticc_rows(
            PLASTICC_LIGHTCURVE_PATH,
            PLASTICC_METADATA_PATH,
            normalization_mode=args.lc_norm_mode,
        )
        write_feature_rows(PLASTICC_TRAIN_COMPACT_CSV_PATH, plasticc_train_rows)
        plasticc_status["train_built"] = True
        plasticc_status["train_row_count"] = len(plasticc_train_rows)

    payload = {
        "lightcurve_normalization_mode": args.lc_norm_mode,
        "spcc_row_count": len(base_rows),
        "compact_features": list(COMPACT_FEATURES),
        "spcc_outputs": {
            "compact_features": SPCC_COMPACT_CSV_PATH,
            "compact_features_noise": SPCC_NOISE_CSV_PATH,
            "compact_features_no_z": SPCC_NO_Z_CSV_PATH,
            "compact_features_no_i": SPCC_NO_I_CSV_PATH,
            "compact_features_short_span": SPCC_SHORT_SPAN_CSV_PATH,
            "compact_features_scaled_flux": SPCC_SCALED_FLUX_CSV_PATH,
        },
        "plasticc": plasticc_status,
    }
    with open(SUMMARY_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps({"manifest_path": SUMMARY_PATH, **payload}, indent=2))


if __name__ == "__main__":
    main()
