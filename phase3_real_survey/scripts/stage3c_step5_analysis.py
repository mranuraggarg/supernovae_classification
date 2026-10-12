"""Integrated Stage 3C Step 5 analysis engine.

The engine implements the frozen post-production statistical clarification. It
does not select features, tune transformations, train classifiers, or interpret
scientific outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from stage3c_step5_records import (
    BRANCHES,
    N_REALIZATIONS,
    PRIMARY_FEATURES,
    REDSHIFTS,
    REGISTERED_FEATURES,
    SBMAGS,
    SUPPORTIVE_FEATURES,
    validate_complete_grid,
    validate_grid,
    validate_record,
)
from stage3c_step5_statistics import (
    bootstrap_frequency_counts,
    bootstrap_timing_statistics,
    bootstrap_weighted_quantile,
    interquartile_range,
    median_absolute_deviation,
    median_difference,
    paired_bootstrap_indices,
    pointwise_bootstrap_interval,
    quantile_difference,
    simultaneous_redshift_band,
    wasserstein_one,
)
from stage3c_step5_timing import r_peak_timing_error


COMPARISONS = (
    ("DES-SN5YR", "SHALLOW", "SDSS-II", "82N"),
    ("DES-SN5YR", "SHALLOW", "SDSS-II", "82S"),
    ("DES-SN5YR", "DEEP", "SDSS-II", "82N"),
    ("DES-SN5YR", "DEEP", "SDSS-II", "82S"),
)

TIMING_METRICS = (
    "signed_median_days",
    "median_absolute_error_days",
    "mad_days",
    "iqr_days",
    "q10_days",
    "q90_days",
)

PHYSICAL_PREDICTIONS = (
    {
        "feature": "peak_color_r_minus_i",
        "role": "primary",
        "prediction": "redshift-dependent Hsiao top-three colour term",
        "comparison_compatibility": "compatible",
        "reason": "Both quantities use -2.5 log10 of independent top-three positive band representatives.",
    },
    {
        "feature": "peak_color_i_minus_z",
        "role": "primary",
        "prediction": "redshift-dependent sign-changing Hsiao top-three colour term",
        "comparison_compatibility": "compatible",
        "reason": "Both quantities use the frozen asynchronous top-three-positive definition.",
    },
    {
        "feature": "i_peak_flux",
        "role": "primary",
        "prediction": "redshift-dependent passband peak-flux response",
        "comparison_compatibility": "not_directly_subtractable",
        "reason": "The extracted feature is log10(1+FLUXCAL), while the registered prediction is an AB magnitude difference without an absolute FLUXCAL normalization.",
    },
    {
        "feature": "z_peak_flux",
        "role": "primary",
        "prediction": "redshift-dependent sign-changing passband peak-flux response",
        "comparison_compatibility": "not_directly_subtractable",
        "reason": "The extracted feature and synthetic magnitude prediction have different normalization conventions.",
    },
    {
        "feature": "r_time_of_peak",
        "role": "primary",
        "prediction": "DES has broader physical peak-recovery errors under the registered MAD definition",
        "comparison_compatibility": "compatible_via_physical_timing_error",
        "reason": "Survey-specific Hsiao r-band peak phases define a common timing-error quantity in days.",
    },
    {
        "feature": "time_span",
        "role": "supportive",
        "prediction": "conditional active-window effect; no unconditional shallow-field sign",
        "comparison_compatibility": "descriptive_only",
        "reason": "The earlier support audit partially informed this prediction.",
    },
)

PASSBAND_PREDICTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "passband_validation"
    / "reproduced_passband_predictions.csv"
)
PASSBAND_COLUMNS = {
    "peak_color_r_minus_i": ("dtop3_r-i", True, "mag"),
    "peak_color_i_minus_z": ("dtop3_i-z", True, "mag"),
    "i_peak_flux": ("dpeak_i", False, "mag"),
    "z_peak_flux": ("dpeak_z", False, "mag"),
}
FEATURE_UNITS = {
    "peak_color_r_minus_i": "mag",
    "peak_color_i_minus_z": "mag",
    "i_peak_flux": "log10(1+FLUXCAL)",
    "z_peak_flux": "log10(1+FLUXCAL)",
    "r_time_of_peak": "days",
    "time_span": "days",
}


@dataclass(frozen=True)
class AnalysisConfig:
    branches: tuple = BRANCHES
    comparisons: tuple = COMPARISONS
    redshifts: tuple = REDSHIFTS
    sbmags: tuple = SBMAGS
    n_realizations: int = N_REALIZATIONS
    n_bootstrap: int = 5000
    confidence: float = 0.95
    bootstrap_base_seed: int = 20261006
    bootstrap_rng: str = "numpy.random.PCG64"
    bootstrap_seed_namespace: str = "stage3c-step5-bootstrap-v1"
    timing_bootstrap_backend: str = "python"
    primary_features: tuple = PRIMARY_FEATURES
    supportive_features: tuple = SUPPORTIVE_FEATURES

    @property
    def features(self):
        return self.primary_features + self.supportive_features

    def validate(self):
        if self.n_realizations <= 0 or self.n_bootstrap <= 0:
            raise ValueError("Analysis dimensions must be positive.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie between zero and one.")
        if tuple(self.branches) != BRANCHES:
            raise ValueError("The four registered branches may not be changed.")
        if tuple(self.comparisons) != COMPARISONS:
            raise ValueError("The four registered comparisons may not be changed.")
        if tuple(self.features) != REGISTERED_FEATURES:
            raise ValueError("The registered feature set may not be changed.")
        if self.timing_bootstrap_backend not in {"python", "native"}:
            raise ValueError("Invalid timing bootstrap backend.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registered_passband_predictions(path=PASSBAND_PREDICTION_PATH):
    """Load only the already-frozen Hsiao passband prediction coordinates."""
    path = Path(path)
    with path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    nodes = []
    lookup = {}
    for source in source_rows:
        redshift = round(float(source["z"]), 3)
        node = {"redshift": redshift}
        for feature, (column, compatible, unit) in PASSBAND_COLUMNS.items():
            value = float(source[column])
            node[feature] = {
                "des_minus_sdss": value,
                "source_column": column,
                "unit": unit,
                "mathematically_compatible_with_extracted_feature": compatible,
            }
            lookup[(redshift, feature)] = node[feature]
        nodes.append(node)

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "template": "Hsiao SN Ia",
        "quantity_convention": "DES minus SDSS synthetic AB magnitude term",
        "nodes": nodes,
    }, lookup


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {line_number}.")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc


def comparison_name(comparison):
    des_survey, des_stratum, sdss_survey, sdss_stratum = comparison
    return f"{des_survey}/{des_stratum} - {sdss_survey}/{sdss_stratum}"


def _interval_payload(replicates, confidence):
    values = np.asarray(replicates, dtype=float)
    undefined = int(np.count_nonzero(~np.isfinite(values)))
    interval = pointwise_bootstrap_interval(values, confidence)
    return {
        "replicates": int(values.size),
        "defined_replicates": int(values.size - undefined),
        "undefined_replicates": undefined,
        "confidence_level": float(confidence),
        "lower": None if interval is None else float(interval[0]),
        "upper": None if interval is None else float(interval[1]),
        "available": interval is not None,
        "unavailable_reason": (
            None
            if interval is not None
            else "At least one frozen bootstrap replicate is undefined."
        ),
    }


def _point_feature_statistics(des_values, sdss_values):
    if len(des_values) == 0 or len(sdss_values) == 0:
        return {
            "median_difference_des_minus_sdss": None,
            "q10_difference_des_minus_sdss": None,
            "q90_difference_des_minus_sdss": None,
            "wasserstein_1": None,
            "available": False,
            "unavailable_reason": "At least one marginal feature sample is empty.",
        }
    return {
        "median_difference_des_minus_sdss": median_difference(
            des_values, sdss_values
        ),
        "q10_difference_des_minus_sdss": quantile_difference(
            des_values, sdss_values, 0.10
        ),
        "q90_difference_des_minus_sdss": quantile_difference(
            des_values, sdss_values, 0.90
        ),
        "wasserstein_1": wasserstein_one(des_values, sdss_values),
        "available": True,
        "unavailable_reason": None,
    }


def _timing_summary(values):
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return {name: None for name in TIMING_METRICS}
    return {
        "signed_median_days": float(np.median(sample)),
        "median_absolute_error_days": float(np.median(np.abs(sample))),
        "mad_days": median_absolute_deviation(sample),
        "iqr_days": interquartile_range(sample),
        "q10_days": float(np.quantile(sample, 0.10)),
        "q90_days": float(np.quantile(sample, 0.90)),
    }


class Stage3CData:
    """Compact branch/cell/realization representation of Stage 3C records."""

    def __init__(self, config: AnalysisConfig, timing_reference_phases):
        config.validate()
        self.config = config
        self.branch_index = {branch: i for i, branch in enumerate(config.branches)}
        self.z_index = {round(float(z), 3): i for i, z in enumerate(config.redshifts)}
        self.sb_index = {round(float(b), 1): i for i, b in enumerate(config.sbmags)}
        self.feature_index = {name: i for i, name in enumerate(config.features)}
        shape = (
            len(config.branches),
            len(config.redshifts),
            len(config.sbmags),
            config.n_realizations,
        )
        self.seen = np.zeros(shape, dtype=bool)
        self.accepted = np.zeros(shape, dtype=bool)
        self.features = np.full(shape + (len(config.features),), np.nan, dtype=float)
        self.feature_available = np.zeros(
            shape + (len(config.features),), dtype=bool
        )
        self.timing_error = np.full(shape, np.nan, dtype=float)
        self.timing_available = np.zeros(shape, dtype=bool)
        self.rejection_reasons = {}
        self.timing_reference_phases = dict(timing_reference_phases)

    def add(self, record):
        survey, stratum, z, sbmag, realization_id = validate_record(record)
        branch = (survey, stratum)
        z_key = round(z, 3)
        sb_key = round(sbmag, 1)

        if branch not in self.branch_index:
            raise ValueError(f"Unexpected branch: {branch}")
        if z_key not in self.z_index or sb_key not in self.sb_index:
            raise ValueError(f"Unexpected analysis cell: z={z}, sbmag={sbmag}")
        if realization_id >= self.config.n_realizations:
            raise ValueError(f"Unexpected realization ID: {realization_id}")

        key = (
            self.branch_index[branch],
            self.z_index[z_key],
            self.sb_index[sb_key],
            realization_id,
        )
        if self.seen[key]:
            raise ValueError(f"Duplicate record key: {branch}, {z_key}, {sb_key}, {realization_id}")
        self.seen[key] = True

        if record["feature_status"] == "rejected":
            reason = str(record["feature_rejection_reason"])
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
            return

        self.accepted[key] = True
        values = record["features"]
        for feature, feature_i in self.feature_index.items():
            value = values[feature]
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                self.features[key + (feature_i,)] = numeric
                self.feature_available[key + (feature_i,)] = True

        if self.feature_available[key + (self.feature_index["r_time_of_peak"],)]:
            reference_key = (survey, stratum, z_key)
            phase = self.timing_reference_phases.get(reference_key)
            if phase is not None:
                error = r_peak_timing_error(
                    first_active_mjd=record["feature_audit"]["first_active_mjd"],
                    recovered_r_time_of_peak=values["r_time_of_peak"],
                    peak_mjd=record["peak_mjd"],
                    redshift=z_key,
                    physical_peak_phase=phase,
                )
                self.timing_error[key] = error
                self.timing_available[key] = True

    def finalize(self):
        if not np.all(self.seen):
            missing = int(np.count_nonzero(~self.seen))
            raise ValueError(f"Accumulated grid is missing {missing} records.")


def accumulate_records(records, config, timing_reference_phases):
    data = Stage3CData(config, timing_reference_phases)
    for record in records:
        data.add(record)
    data.finalize()
    return data


def _grid_summary(config):
    return {
        "complete": True,
        "total_records": (
            len(config.branches)
            * len(config.redshifts)
            * len(config.sbmags)
            * config.n_realizations
        ),
        "branch_count": len(config.branches),
        "redshift_count": len(config.redshifts),
        "sbmag_count": len(config.sbmags),
        "realizations_per_cell": config.n_realizations,
        "duplicate_records": 0,
        "missing_cells": 0,
        "missing_realization_ids": 0,
        "unexpected_realization_ids": 0,
    }


def analyze_data(data: Stage3CData, *, input_provenance=None, code_provenance=None):
    config = data.config
    passband_metadata, passband_lookup = load_registered_passband_predictions()
    acceptance_rows = []
    availability_rows = []
    feature_rows = []
    feature_bands = []
    timing_rows = []
    timing_comparison_rows = []
    timing_bands = []
    undefined_diagnostics = []

    branch_index = data.branch_index

    for sb_i, sbmag in enumerate(config.sbmags):
        feature_curves = {
            (comparison_name(c), feature): np.full(
                (config.n_bootstrap, len(config.redshifts)), np.nan
            )
            for c in config.comparisons
            for feature in config.features
        }
        feature_points = {
            key: np.full(len(config.redshifts), np.nan)
            for key in feature_curves
        }
        timing_curves = {
            (comparison_name(c), metric): np.full(
                (config.n_bootstrap, len(config.redshifts)), np.nan
            )
            for c in config.comparisons
            for metric in TIMING_METRICS
        }
        timing_points = {
            key: np.full(len(config.redshifts), np.nan)
            for key in timing_curves
        }

        for z_i, redshift in enumerate(config.redshifts):
            indices = paired_bootstrap_indices(
                redshift,
                sbmag,
                n_realizations=config.n_realizations,
                n_bootstrap=config.n_bootstrap,
            )
            counts = bootstrap_frequency_counts(indices, config.n_realizations)

            branch_feature_bootstrap = {}
            branch_timing_bootstrap = {}
            branch_timing_points = {}

            for branch in config.branches:
                b_i = branch_index[branch]
                accepted = data.accepted[b_i, z_i, sb_i]
                accepted_count = int(np.count_nonzero(accepted))
                acceptance_rows.append({
                    "survey": branch[0],
                    "stratum": branch[1],
                    "redshift": float(redshift),
                    "sbmag_mag_per_arcsec2": float(sbmag),
                    "total_realizations": config.n_realizations,
                    "accepted_realizations": accepted_count,
                    "rejected_realizations": config.n_realizations - accepted_count,
                    "acceptance_fraction": accepted_count / config.n_realizations,
                })

                for feature, f_i in data.feature_index.items():
                    available = data.feature_available[b_i, z_i, sb_i, :, f_i]
                    values = data.features[b_i, z_i, sb_i, :, f_i]
                    available_count = int(np.count_nonzero(available))
                    availability_rows.append({
                        "survey": branch[0],
                        "stratum": branch[1],
                        "redshift": float(redshift),
                        "sbmag_mag_per_arcsec2": float(sbmag),
                        "feature": feature,
                        "accepted_realizations": accepted_count,
                        "feature_available_realizations": available_count,
                        "feature_unavailable_among_accepted": accepted_count - available_count,
                    })
                    branch_feature_bootstrap[(branch, feature)] = (
                        bootstrap_weighted_quantile(values, available, counts, 0.5)
                    )

                timing_available = data.timing_available[b_i, z_i, sb_i]
                timing_values = data.timing_error[b_i, z_i, sb_i]
                timing_sample = timing_values[timing_available]
                summary = _timing_summary(timing_sample)
                branch_timing_points[branch] = summary
                timing_rows.append({
                    "survey": branch[0],
                    "stratum": branch[1],
                    "redshift": float(redshift),
                    "sbmag_mag_per_arcsec2": float(sbmag),
                    "timing_available_realizations": int(timing_sample.size),
                    "unit": "days",
                    **summary,
                })
                branch_timing_bootstrap[branch] = bootstrap_timing_statistics(
                    timing_values,
                    timing_available,
                    indices,
                    backend=config.timing_bootstrap_backend,
                )

            for comparison in config.comparisons:
                des_branch = (comparison[0], comparison[1])
                sdss_branch = (comparison[2], comparison[3])
                name = comparison_name(comparison)
                d_i = branch_index[des_branch]
                s_i = branch_index[sdss_branch]
                accepted_complete = int(np.count_nonzero(
                    data.accepted[d_i, z_i, sb_i]
                    & data.accepted[s_i, z_i, sb_i]
                ))

                for feature, f_i in data.feature_index.items():
                    des_available = data.feature_available[d_i, z_i, sb_i, :, f_i]
                    sdss_available = data.feature_available[s_i, z_i, sb_i, :, f_i]
                    des_values = data.features[d_i, z_i, sb_i, :, f_i][des_available]
                    sdss_values = data.features[s_i, z_i, sb_i, :, f_i][sdss_available]
                    complete_pair_count = int(np.count_nonzero(
                        des_available & sdss_available
                    ))
                    points = _point_feature_statistics(des_values, sdss_values)
                    replicates = (
                        branch_feature_bootstrap[(des_branch, feature)]
                        - branch_feature_bootstrap[(sdss_branch, feature)]
                    )
                    interval = _interval_payload(replicates, config.confidence)
                    estimate = points["median_difference_des_minus_sdss"]
                    if estimate is not None:
                        feature_points[(name, feature)][z_i] = estimate
                    feature_curves[(name, feature)][:, z_i] = replicates
                    if interval["undefined_replicates"]:
                        undefined_diagnostics.append({
                            "analysis": "feature_median_difference",
                            "comparison": name,
                            "feature": feature,
                            "redshift": float(redshift),
                            "sbmag_mag_per_arcsec2": float(sbmag),
                            "undefined_replicates": interval["undefined_replicates"],
                        })
                    prediction = passband_lookup.get(
                        (round(float(redshift), 3), feature)
                    )
                    prediction_payload = None
                    if prediction is not None:
                        compatible = prediction[
                            "mathematically_compatible_with_extracted_feature"
                        ]
                        prediction_payload = {
                            **prediction,
                            "observed_minus_prediction": (
                                estimate - prediction["des_minus_sdss"]
                                if compatible and estimate is not None
                                else None
                            ),
                            "residual_available": bool(
                                compatible and estimate is not None
                            ),
                            "residual_unavailable_reason": (
                                None
                                if compatible and estimate is not None
                                else (
                                    "Synthetic AB-magnitude term and frozen flux feature use incompatible normalization conventions."
                                    if not compatible
                                    else "Observed median difference is unavailable."
                                )
                            ),
                        }
                    feature_rows.append({
                        "comparison": name,
                        "des_survey": des_branch[0],
                        "des_stratum": des_branch[1],
                        "sdss_survey": sdss_branch[0],
                        "sdss_stratum": sdss_branch[1],
                        "redshift": float(redshift),
                        "sbmag_mag_per_arcsec2": float(sbmag),
                        "feature": feature,
                        "feature_role": "primary" if feature in config.primary_features else "supportive",
                        "feature_unit": FEATURE_UNITS[feature],
                        "des_available_count": int(des_values.size),
                        "sdss_available_count": int(sdss_values.size),
                        "accepted_in_both_branches": accepted_complete,
                        "feature_available_in_both_branches": complete_pair_count,
                        **points,
                        "median_difference_bootstrap": interval,
                        "registered_passband_prediction": prediction_payload,
                    })

                des_point = branch_timing_points[des_branch]
                sdss_point = branch_timing_points[sdss_branch]
                for metric in TIMING_METRICS:
                    d_value = des_point[metric]
                    s_value = sdss_point[metric]
                    point_difference = (
                        None if d_value is None or s_value is None else d_value - s_value
                    )
                    replicates = (
                        branch_timing_bootstrap[des_branch][metric]
                        - branch_timing_bootstrap[sdss_branch][metric]
                    )
                    interval = _interval_payload(replicates, config.confidence)
                    if point_difference is not None:
                        timing_points[(name, metric)][z_i] = point_difference
                    timing_curves[(name, metric)][:, z_i] = replicates
                    if interval["undefined_replicates"]:
                        undefined_diagnostics.append({
                            "analysis": "timing_difference",
                            "comparison": name,
                            "metric": metric,
                            "redshift": float(redshift),
                            "sbmag_mag_per_arcsec2": float(sbmag),
                            "undefined_replicates": interval["undefined_replicates"],
                        })
                    timing_comparison_rows.append({
                        "comparison": name,
                        "redshift": float(redshift),
                        "sbmag_mag_per_arcsec2": float(sbmag),
                        "metric": metric,
                        "unit": "days",
                        "metric_role": (
                            "primary_dispersion" if metric == "mad_days"
                            else "corroborative_dispersion" if metric == "iqr_days"
                            else "descriptive_no_directional_hypothesis"
                        ),
                        "difference_des_minus_sdss": point_difference,
                        "bootstrap": interval,
                    })

        for (name, feature), matrix in feature_curves.items():
            estimates = feature_points[(name, feature)]
            band = simultaneous_redshift_band(
                estimates, matrix, config.confidence
            )
            if band is None:
                undefined_by_redshift = [
                    int(np.count_nonzero(~np.isfinite(matrix[:, z_i])))
                    for z_i in range(len(config.redshifts))
                ]
                feature_bands.append({
                    "comparison": name,
                    "feature": feature,
                    "unit": FEATURE_UNITS[feature],
                    "sbmag_mag_per_arcsec2": float(sbmag),
                    "available": False,
                    "unavailable_reason": "At least one point estimate or aligned bootstrap node is undefined.",
                    "undefined_replicates_by_redshift": undefined_by_redshift,
                    "critical_value": None,
                    "nodes": [],
                })
            else:
                feature_bands.append({
                    "comparison": name,
                    "feature": feature,
                    "unit": FEATURE_UNITS[feature],
                    "sbmag_mag_per_arcsec2": float(sbmag),
                    "available": True,
                    "unavailable_reason": None,
                    "undefined_replicates_by_redshift": [0] * len(config.redshifts),
                    "critical_value": float(band["critical_value"]),
                    "nodes": [
                        {
                            "redshift": float(redshift),
                            "point_estimate": float(estimates[z_i]),
                            "lower": float(band["lower"][z_i]),
                            "upper": float(band["upper"][z_i]),
                        }
                        for z_i, redshift in enumerate(config.redshifts)
                    ],
                })

        for (name, metric), matrix in timing_curves.items():
            estimates = timing_points[(name, metric)]
            band = simultaneous_redshift_band(
                estimates, matrix, config.confidence
            )
            if band is None:
                timing_bands.append({
                    "comparison": name,
                    "metric": metric,
                    "unit": "days",
                    "sbmag_mag_per_arcsec2": float(sbmag),
                    "available": False,
                    "unavailable_reason": "At least one point estimate or aligned bootstrap node is undefined.",
                    "undefined_replicates_by_redshift": [
                        int(np.count_nonzero(~np.isfinite(matrix[:, z_i])))
                        for z_i in range(len(config.redshifts))
                    ],
                    "critical_value": None,
                    "nodes": [],
                })
            else:
                timing_bands.append({
                    "comparison": name,
                    "metric": metric,
                    "unit": "days",
                    "sbmag_mag_per_arcsec2": float(sbmag),
                    "available": True,
                    "unavailable_reason": None,
                    "undefined_replicates_by_redshift": [0] * len(config.redshifts),
                    "critical_value": float(band["critical_value"]),
                    "nodes": [
                        {
                            "redshift": float(redshift),
                            "point_estimate": float(estimates[z_i]),
                            "lower": float(band["lower"][z_i]),
                            "upper": float(band["upper"][z_i]),
                        }
                        for z_i, redshift in enumerate(config.redshifts)
                    ],
                })

    timing_reference_rows = []
    for branch in config.branches:
        for redshift in config.redshifts:
            phase = data.timing_reference_phases.get(
                (branch[0], branch[1], round(float(redshift), 3))
            )
            timing_reference_rows.append({
                "survey": branch[0],
                "stratum": branch[1],
                "redshift": float(redshift),
                "phi_r_max_rest_frame_days": None if phase is None else int(phase),
                "available": phase is not None,
            })

    return {
        "schema": "stage3c-step5-analysis-v1",
        "status": "analysis_complete_no_scientific_interpretation",
        "configuration": {
            **asdict(config),
            "features": list(config.features),
            "statistical_definition": "DES minus SDSS difference of marginal accepted medians",
            "timing_primary_definition": "MAD(DES physical timing error) - MAD(SDSS physical timing error)",
            "simultaneous_band": "unstandardized_max_absolute_deviation",
        },
        "input_provenance": input_provenance or {},
        "code_provenance": code_provenance or {},
        "grid_validation": _grid_summary(config),
        "rejection_reason_counts": dict(sorted(data.rejection_reasons.items())),
        "acceptance_statistics": acceptance_rows,
        "feature_availability_statistics": availability_rows,
        "primary_and_supportive_feature_statistics": feature_rows,
        "feature_simultaneous_confidence_bands": feature_bands,
        "timing_reference_phases": timing_reference_rows,
        "timing_error_statistics": timing_rows,
        "timing_comparisons": timing_comparison_rows,
        "timing_simultaneous_confidence_bands": timing_bands,
        "undefined_statistic_diagnostics": undefined_diagnostics,
        "physical_prediction_metadata": {
            "registry": list(PHYSICAL_PREDICTIONS),
            "synthetic_passband_predictions": passband_metadata,
        },
        "reproducibility": {
            "input_order_invariant": True,
            "rng": config.bootstrap_rng,
            "bootstrap_seed_namespace": config.bootstrap_seed_namespace,
            "bootstrap_base_seed": config.bootstrap_base_seed,
            "shared_realization_resampling_across_branches": True,
            "independent_stream_per_redshift_sbmag_cell": True,
            "undefined_replicates_discarded": False,
            "missing_features_imputed": False,
            "timing_bootstrap_backend": config.timing_bootstrap_backend,
        },
    }


def analyze_records(
    records,
    timing_reference_phases: Mapping,
    *,
    config=None,
    input_provenance=None,
    code_provenance=None,
):
    """Validate and analyze an in-memory artificial or registered dataset."""
    config = config or AnalysisConfig()
    config.validate()
    records = list(records)
    validate_grid(
        records,
        branches=config.branches,
        redshifts=config.redshifts,
        sbmags=config.sbmags,
        n_realizations=config.n_realizations,
    )
    data = accumulate_records(records, config, timing_reference_phases)
    return analyze_data(
        data,
        input_provenance=input_provenance,
        code_provenance=code_provenance,
    )


def analyze_jsonl(
    path: Path,
    timing_reference_phases: Mapping,
    *,
    config=None,
    expected_sha256=None,
    code_provenance=None,
):
    """Two-pass streaming validation and analysis for the production JSONL."""
    config = config or AnalysisConfig()
    config.validate()
    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"Input SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )

    frozen_grid = (
        tuple(config.branches) == BRANCHES
        and tuple(config.redshifts) == REDSHIFTS
        and tuple(config.sbmags) == SBMAGS
        and config.n_realizations == N_REALIZATIONS
    )
    if frozen_grid:
        validate_complete_grid(iter_jsonl(path))
    else:
        validate_grid(
            iter_jsonl(path),
            branches=config.branches,
            redshifts=config.redshifts,
            sbmags=config.sbmags,
            n_realizations=config.n_realizations,
        )

    data = accumulate_records(iter_jsonl(path), config, timing_reference_phases)
    provenance = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "expected_sha256": expected_sha256,
        "sha256_matches_expected": expected_sha256 is None or digest == expected_sha256,
    }
    return analyze_data(
        data,
        input_provenance=provenance,
        code_provenance=code_provenance,
    )
