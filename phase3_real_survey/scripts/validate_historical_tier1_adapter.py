#!/usr/bin/env python3
"""Validate label-independent DES observation preparation against historical SPCC.

This script deliberately performs no DES classification or feature extraction.
It validates numerical cleaning, stable ordering, normalization and reconstruction
against the original historical implementation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from math import isfinite
from pathlib import Path


DEFAULT_BACKUP = Path(
    "/Volumes/AstroSSD/share/github/"
    "supernovae_classification.backup.20260510-200046"
)


def prepare_observations(observations):
    """Apply the numerical portion of historical SPCC cleaning."""
    ordered = sorted(
        observations,
        key=lambda observation: (
            observation.mjd,
            observation.band,
        ),
    )

    return [
        observation
        for observation in ordered
        if (
            isfinite(observation.mjd)
            and isfinite(observation.flux)
            and isfinite(observation.flux_err)
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=DEFAULT_BACKUP,
    )
    args = parser.parse_args()

    historical_root = args.historical_root.resolve()

    if not (historical_root / "feature_pipeline").is_dir():
        raise SystemExit(
            f"Historical feature pipeline not found: {historical_root}"
        )

    sys.path.insert(0, str(historical_root))

    from feature_pipeline.schemas import (
        SPCCRawEvent,
        SPCCRawObservation,
    )
    from feature_pipeline.cleaning.spcc_clean import clean_event
    from feature_pipeline.extraction.spcc_features import (
        extract_native_features,
    )
    from feature_pipeline.interpolation.spcc_native_reconstruct import (
        reconstruct_bandwise_grid,
    )

    # Synthetic observations deliberately include:
    # - unsorted epochs
    # - negative flux
    # - duplicate MJD/filter
    # - nonfinite flux
    # - missing bands at individual epochs
    observations = [
        SPCCRawObservation(
            mjd=60002.0,
            band="r",
            flux=-4.0,
            flux_err=2.0,
        ),
        SPCCRawObservation(
            mjd=60000.0,
            band="g",
            flux=10.0,
            flux_err=1.0,
        ),
        SPCCRawObservation(
            mjd=60001.0,
            band="i",
            flux=20.0,
            flux_err=2.0,
        ),
        SPCCRawObservation(
            mjd=60000.0,
            band="g",
            flux=12.0,
            flux_err=1.5,
        ),
        SPCCRawObservation(
            mjd=60003.0,
            band="z",
            flux=float("nan"),
            flux_err=1.0,
        ),
    ]

    # These are synthetic SPCC metadata values used ONLY to run
    # the original cleaning function as a reference.
    # They are never derived from DES labels or metadata.
    reference_event = SPCCRawEvent(
        survey="SPCC",
        snid=999999,
        sn_type=None,
        sim_type="Ia",
        sim_z=0.1,
        ra=10.0,
        decl=10.0,
        mwebv=0.01,
        hostid=None,
        hostz=None,
        spec=None,
        observations=observations,
    )

    reference_result = clean_event(
        reference_event,
        min_observations_per_event=1,
    )

    assert reference_result.accepted, (
        "Historical reference cleaning rejected synthetic event: "
        f"{reference_result.reasons}"
    )

    reference_cleaned = reference_result.event

    adapter_cleaned = replace(
        reference_event,
        observations=prepare_observations(observations),
    )

    assert (
        adapter_cleaned.observations
        == reference_cleaned.observations
    ), "Adapter cleaning differs from historical cleaning"

    reference_grid = reconstruct_bandwise_grid(
        reference_cleaned
    )

    adapter_grid = reconstruct_bandwise_grid(
        adapter_cleaned
    )

    assert (
        adapter_grid.raw_sequence
        == reference_grid.raw_sequence
    ), "Normalized observation sequences differ"

    assert (
        adapter_grid.reconstructed_sequence
        == reference_grid.reconstructed_sequence
    ), "Reconstructed grids differ"

    grid = adapter_grid.reconstructed_sequence

    assert len(grid) == 3, (
        f"Expected three unique valid epochs, got {len(grid)}"
    )

    # The later duplicate g observation must overwrite the first.
    assert grid[0][1] == 12.0, (
        "Historical last-observation-wins rule was not preserved"
    )

    # Negative flux must survive.
    assert grid[2][2] == -4.0, (
        "Negative flux was unexpectedly removed"
    )

    # Missing bands must remain zero-filled.
    assert grid[0][2] == 0.0
    assert grid[0][3] == 0.0
    assert grid[0][4] == 0.0

    # Normalized time must start at zero.
    assert [row[0] for row in grid] == [
        0.0,
        1.0,
        2.0,
    ]

    # Compare the original numerical feature extractor on both
    # independently prepared representations.
    reference_features = extract_native_features(
        reference_cleaned
    )

    adapter_features = extract_native_features(
        adapter_cleaned
    )

    feature_names = [
        "z_peak_flux",
        "r_mean_flux",
        "peak_color_g_minus_r",
        "i_peak_flux",
        "peak_color_r_minus_i",
        "peak_color_i_minus_z",
        "g_mean_flux",
        "r_peak_flux",
        "z_std_flux",
        "i_amplitude",
        "i_std_flux",
        "time_span",
        "z_time_of_peak",
        "i_time_of_peak",
        "r_time_of_peak",
        "r_std_flux",
    ]

    for name in feature_names:
        reference_value = reference_features[name]
        adapter_value = adapter_features[name]

        assert reference_value == adapter_value, (
            f"Feature mismatch: {name}: "
            f"{reference_value} != {adapter_value}"
        )

    print("Historical 16-feature equivalence: PASS")

    print("=" * 65)
    print("HISTORICAL TIER-1 ADAPTER VALIDATION")
    print("=" * 65)
    print("Historical root:", historical_root)
    print()
    print("Synthetic observations:", len(observations))
    print("Cleaned observations:", len(adapter_cleaned.observations))
    print("Unique reconstructed epochs:", len(grid))
    print()
    print("Historical cleaning equivalence: PASS")
    print("Historical normalization equivalence: PASS")
    print("Historical reconstruction equivalence: PASS")
    print("Duplicate handling: PASS")
    print("Negative-flux preservation: PASS")
    print("Missing-band zero filling: PASS")
    print()
    print("OVERALL: PASS")
    print()
    print("No DES observations or labels were accessed.")
    print("No classifier predictions were generated.")


if __name__ == "__main__":
    main()
