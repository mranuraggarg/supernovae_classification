"""Owned native SPCC reconstruction utilities for the Phase 2 pipeline.

This module is intentionally distinct from the legacy reference path. The logic here
is policy-driven and kept simple, explicit, and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from feature_pipeline.loaders.spcc_raw import normalize_event_observations
from feature_pipeline.schemas import SPCCRawEvent


@dataclass(frozen=True)
class NativeReconstructionArtifact:
    mode: str
    raw_sequence: list[list[float | str]]
    reconstructed_sequence: list[list[float]]
    notes: list[str]


def normalized_raw_sequence(event: SPCCRawEvent) -> list[list[float | str]]:
    return [
        [obs.time, obs.band, obs.flux, obs.flux_err]
        for obs in normalize_event_observations(event)
    ]


def reconstruct_bandwise_grid(event: SPCCRawEvent) -> NativeReconstructionArtifact:
    """Build a simple owned shared-time-grid representation without borrowed interpolation logic.

    Policy:
    - use the distinct normalized observation times already present in the event
    - emit one row per unique time
    - preserve observed values only; missing band entries are explicit zeros
    - preserve measurement uncertainty only where observed
    """

    raw = normalized_raw_sequence(event)
    unique_times = sorted({row[0] for row in raw})
    observed = {(row[0], row[1]): (row[2], row[3]) for row in raw}
    reconstructed = []
    for t in unique_times:
        row = [t]
        for band in ("g", "r", "i", "z"):
            flux, flux_err = observed.get((t, band), (0.0, 0.0))
            row.append(flux)
        for band in ("g", "r", "i", "z"):
            flux, flux_err = observed.get((t, band), (0.0, 0.0))
            row.append(flux_err)
        reconstructed.append(row)

    return NativeReconstructionArtifact(
        mode="native_bandwise_grid",
        raw_sequence=raw,
        reconstructed_sequence=reconstructed,
        notes=[
            "Shared grid uses unique observed times only.",
            "Missing band values are explicit zeros rather than filled by legacy interpolation.",
            "This path is owned by the repository and intentionally separate from the legacy benchmark code.",
        ],
    )
