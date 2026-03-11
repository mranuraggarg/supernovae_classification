"""Explicit preprocessing policy definitions for the SPCC pipeline."""

from dataclasses import dataclass


KEY_TYPES = {
    "Ia": 1,
    "II": 2,
    "Ibc": 3,
    "IIn": 21,
    "IIP": 22,
    "IIL": 23,
    "Ib": 32,
    "Ic": 33,
}


@dataclass(frozen=True)
class PreprocessingPolicy:
    missing_data_strategy: str
    time_axis_normalization: str
    band_grouping_strategy: str
    flux_normalization: str
    outlier_treatment: str
    min_observations_per_event: int
    sparse_curve_policy: str


DEFAULT_SPCC_POLICY = PreprocessingPolicy(
    missing_data_strategy="Bandwise fill between nearest valid observations; edge gaps are forward/back filled.",
    time_axis_normalization="Observation times are shifted so the first observation occurs at t=0.",
    band_grouping_strategy="Observations are grouped onto a shared time basis using the configurable grouping fraction.",
    flux_normalization="No scaling beyond the configured normalization constants.",
    outlier_treatment="No explicit outlier clipping in the legacy-compatible path.",
    min_observations_per_event=1,
    sparse_curve_policy="Retain sparse curves and fall back to padded/interpolated values depending on parser mode.",
)
