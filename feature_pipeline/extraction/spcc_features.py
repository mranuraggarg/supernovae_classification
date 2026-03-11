"""Native owned feature extraction for SPCC Tier 1."""

from __future__ import annotations

import csv
import json
import os
import argparse
from collections import Counter

import numpy as np

from feature_pipeline.cleaning.spcc_clean import clean_event, summarize_cleaning_results
from feature_pipeline.extraction.feature_registry import FEATURE_REGISTRY
from feature_pipeline.interpolation.spcc_native_reconstruct import reconstruct_bandwise_grid
from feature_pipeline.loaders.spcc_raw import DEFAULT_SPCC_RAW_GLOB, iter_spcc_files, load_spcc_raw_event
from feature_pipeline.policies import DEFAULT_SPCC_POLICY, KEY_TYPES


def _band_fluxes(reconstructed: np.ndarray, band_index: int) -> np.ndarray:
    return reconstructed[:, 1 + band_index]


COLOR_CLIP_RANGE = (-5.0, 5.0)


def _positive_log10_1p(value: float) -> float:
    return float(np.log10(1.0 + max(float(value), 0.0)))


def _signed_log10_1p(value: float) -> float:
    magnitude = np.log10(1.0 + abs(float(value)))
    return float(np.sign(value) * magnitude)


def _clip_value(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), lower, upper))


def _safe_color_from_peak_fluxes(flux_a: float, flux_b: float, floor: float = 1e-6) -> float:
    """Compute a magnitude-style color proxy from two peak fluxes.

    We clip non-positive fluxes to a small floor so the feature stays finite while
    preserving the directionality of the flux ratio.
    """
    safe_a = max(float(flux_a), floor)
    safe_b = max(float(flux_b), floor)
    return float(-2.5 * np.log10(safe_a / safe_b))


def _compress_training_features(feature_row: dict) -> dict:
    positive_log_features = {
        "peak_flux_all",
        "amplitude_all",
        "std_flux_all",
        "total_snr",
        "g_peak_flux",
        "g_std_flux",
        "g_amplitude",
        "r_peak_flux",
        "r_std_flux",
        "r_amplitude",
        "i_peak_flux",
        "i_std_flux",
        "i_amplitude",
        "z_peak_flux",
        "z_std_flux",
        "z_amplitude",
    }
    signed_log_features = {
        "mean_flux_all",
        "g_mean_flux",
        "r_mean_flux",
        "i_mean_flux",
        "z_mean_flux",
    }
    color_features = {
        "peak_color_g_minus_r",
        "peak_color_r_minus_i",
        "peak_color_i_minus_z",
    }

    transformed = dict(feature_row)
    for name in positive_log_features:
        transformed[name] = _positive_log10_1p(transformed[name])
    for name in signed_log_features:
        transformed[name] = _signed_log10_1p(transformed[name])
    for name in color_features:
        transformed[name] = _clip_value(transformed[name], *COLOR_CLIP_RANGE)
    return transformed


def extract_native_features(event) -> dict:
    artifact = reconstruct_bandwise_grid(event)
    reconstructed = np.array(artifact.reconstructed_sequence, dtype=float)
    times = reconstructed[:, 0]
    all_flux = reconstructed[:, 1:5]
    all_errors = reconstructed[:, 5:9]
    flat_flux = all_flux.reshape(-1)
    valid_error_mask = all_errors > 0
    total_snr = float(np.sum(np.abs(all_flux[valid_error_mask]) / all_errors[valid_error_mask])) if np.any(valid_error_mask) else 0.0

    feature_row = {
        "snid": event.snid,
        "label_name": event.sim_type,
        "label_id": KEY_TYPES[event.sim_type],
        "sim_z": event.sim_z,
        "observation_count": len(event.observations),
        "time_span": float(times.max() - times.min()) if len(times) else 0.0,
        "observed_band_count": len({obs.band for obs in event.observations}),
        "peak_flux_all": float(np.max(flat_flux)),
        "time_of_peak_all": float(times[np.argmax(np.max(all_flux, axis=1))]),
        "amplitude_all": float(np.max(flat_flux) - np.min(flat_flux)),
        "mean_flux_all": float(np.mean(flat_flux)),
        "std_flux_all": float(np.std(flat_flux)),
        "total_snr": total_snr,
    }

    band_names = ("g", "r", "i", "z")
    band_peaks = {}
    for idx, band in enumerate(band_names):
        fluxes = _band_fluxes(reconstructed, idx)
        peak_index = int(np.argmax(fluxes))
        band_peaks[band] = float(fluxes[peak_index])
        feature_row[f"{band}_peak_flux"] = band_peaks[band]
        feature_row[f"{band}_time_of_peak"] = float(times[peak_index])
        feature_row[f"{band}_mean_flux"] = float(np.mean(fluxes))
        feature_row[f"{band}_std_flux"] = float(np.std(fluxes))
        feature_row[f"{band}_amplitude"] = float(np.max(fluxes) - np.min(fluxes))

    feature_row["peak_color_g_minus_r"] = _safe_color_from_peak_fluxes(band_peaks["g"], band_peaks["r"])
    feature_row["peak_color_r_minus_i"] = _safe_color_from_peak_fluxes(band_peaks["r"], band_peaks["i"])
    feature_row["peak_color_i_minus_z"] = _safe_color_from_peak_fluxes(band_peaks["i"], band_peaks["z"])

    return _compress_training_features(feature_row)


def build_native_feature_rows(input_glob: str = DEFAULT_SPCC_RAW_GLOB) -> tuple[list[dict], dict]:
    files = iter_spcc_files(input_glob=input_glob)
    cleaning_results = []
    feature_rows = []
    for path in files:
        raw_event = load_spcc_raw_event(path)
        cleaning_result = clean_event(
            raw_event,
            min_observations_per_event=DEFAULT_SPCC_POLICY.min_observations_per_event,
        )
        cleaning_results.append(cleaning_result)
        if not cleaning_result.accepted:
            continue
        if cleaning_result.event.sim_type not in KEY_TYPES:
            continue
        feature_rows.append(extract_native_features(cleaning_result.event))

    summary = summarize_cleaning_results(cleaning_results)
    summary["raw_input_glob"] = input_glob
    summary["raw_file_count"] = len(files)
    summary["feature_row_count"] = len(feature_rows)
    summary["feature_count"] = len(FEATURE_REGISTRY)
    summary["label_distribution"] = dict(Counter(row["label_name"] for row in feature_rows))
    return feature_rows, summary


def write_native_feature_artifacts(
    feature_rows: list[dict],
    summary: dict,
    *,
    output_dir: str = "data/processed",
    results_dir: str = "results/phase2_tier1",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "spcc_features_tier1.csv")
    npz_path = os.path.join(output_dir, "spcc_features_tier1.npz")
    metadata_path = os.path.join(results_dir, "spcc_tier1_metadata.json")

    fieldnames = list(feature_rows[0].keys()) if feature_rows else ["snid", "label_name", "label_id", "sim_z"]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    numeric_feature_names = [spec.name for spec in FEATURE_REGISTRY]
    ids = np.array([row["snid"] for row in feature_rows], dtype=object)
    labels = np.array([row["label_id"] for row in feature_rows], dtype=int)
    feature_matrix = np.array([[row[name] for name in numeric_feature_names] for row in feature_rows], dtype=float)
    np.savez(
        npz_path,
        ids=ids,
        labels=labels,
        feature_names=np.array(numeric_feature_names, dtype=object),
        feature_matrix=feature_matrix,
    )

    metadata = {
        "artifact_version": "phase2_tier1_v1",
        "feature_registry": [spec.to_dict() for spec in FEATURE_REGISTRY],
        "summary": summary,
        "output_files": {
            "csv": csv_path,
            "npz": npz_path,
        },
        "training_value_policy": {
            "positive_flux_like_features": "Stored as log10(1 + max(value, 0)) to compress heavy tails before training.",
            "mean_flux_features": "Stored as signed log10(1 + abs(value)) with original sign preserved.",
            "color_features": {
                "formula": "Computed as magnitude-style flux ratios: -2.5 * log10(flux_a / flux_b).",
                "clip_range": list(COLOR_CLIP_RANGE),
                "purpose": "Bound extreme ratios caused by near-zero peak fluxes before training.",
            },
        },
        "selection_policy": {
            "current_status": "candidate_only",
            "acceptance_rule": "Features remain candidates until retained or rejected using proven importance from training.",
        },
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "csv_path": csv_path,
        "npz_path": npz_path,
        "metadata_path": metadata_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build native owned SPCC Tier 1 feature artifacts.")
    parser.add_argument("--input-glob", default=DEFAULT_SPCC_RAW_GLOB, help="Glob for raw SPCC DES_*.DAT files.")
    args = parser.parse_args()

    rows, summary = build_native_feature_rows(input_glob=args.input_glob)
    paths = write_native_feature_artifacts(rows, summary)
    print("Native SPCC feature artifacts written:")
    for key, value in paths.items():
        print(f"{key}: {value}")
