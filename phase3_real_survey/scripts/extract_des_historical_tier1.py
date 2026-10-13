#!/usr/bin/env python3
"""Extract historical Tier-1 features from the frozen DES cohort.

Uses original historical feature-extraction code.
Does not read DES classification labels.
Does not load or execute the classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


DEFAULT_BACKUP = Path(
    "/Volumes/AstroSSD/share/github/"
    "supernovae_classification.backup.20260510-200046"
)

FEATURES = [
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


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def decode_band(value):
    if isinstance(value, bytes):
        value = value.decode()

    return str(value).strip().lower()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--historical-root",
        type=Path,
        default=DEFAULT_BACKUP,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "phase3_real_survey/results/"
            "historical_tier1_des"
        ),
    )

    args = parser.parse_args()

    historical_root = args.historical_root.resolve()

    sys.path.insert(0, str(historical_root))

    from feature_pipeline.schemas import (
        SPCCRawEvent,
        SPCCRawObservation,
    )

    from feature_pipeline.extraction.spcc_features import (
        extract_native_features,
    )

    from validate_historical_tier1_adapter import (
        prepare_observations,
    )

    root = Path(
        "phase3_real_survey/data/raw/des/"
        "DES-SN5YR-1.2/0_DATA/DES-SN5YR_DES"
    )

    head_path = root / "DES-SN5YR_DES_HEAD.FITS.gz"
    phot_path = root / "DES-SN5YR_DES_PHOT.FITS.gz"

    cohort_path = Path(
        "phase3_real_survey/results/"
        "real_feature_tables/des_compact_features.csv"
    )

    cohort = pd.read_csv(
        cohort_path,
        usecols=["object_id"],
    )

    ids = pd.to_numeric(
        cohort["object_id"],
        errors="raise",
    ).astype(int).tolist()

    assert len(ids) == 185
    assert len(set(ids)) == 185

    cohort_ids = set(ids)

    with fits.open(head_path, memmap=False) as h:
        head = h[1].data.copy()

    with fits.open(phot_path, memmap=False) as h:
        phot = h[1].data.copy()

    results = []
    failures = []
    found = set()

    for event in head:
        snid = int(event["SNID"])

        if snid not in cohort_ids:
            continue

        found.add(snid)

        lo = int(event["PTROBS_MIN"]) - 1
        hi = int(event["PTROBS_MAX"])

        observations = []

        for row in phot[lo:hi]:
            band = decode_band(row["BAND"])

            if band not in ("g", "r", "i", "z"):
                continue

            observations.append(
                SPCCRawObservation(
                    mjd=float(row["MJD"]),
                    band=band,
                    flux=float(row["FLUXCAL"]),
                    flux_err=float(row["FLUXCALERR"]),
                )
            )

        cleaned = prepare_observations(observations)

        if not cleaned:
            failures.append({
                "snid": snid,
                "reason": "No valid observations",
            })
            continue

        # Compatibility metadata is deliberately constant.
        # It is NOT taken from DES classifications.
        #
        # The historical extractor requires sim_type to populate
        # metadata columns. Those columns are discarded below.
        #
        # This is an interface sentinel, not a scientific label.
        raw_event = SPCCRawEvent(
            survey="SPCC",
            snid=snid,
            sn_type=None,
            sim_type="Ia",
            sim_z=0.1,
            ra=0.0,
            decl=0.0,
            mwebv=0.0,
            hostid=None,
            hostz=None,
            spec=None,
            observations=cleaned,
        )

        try:
            extracted = extract_native_features(raw_event)

            feature_row = {
                "object_id": snid,
            }

            for name in FEATURES:
                feature_row[name] = float(extracted[name])

            results.append(feature_row)

        except Exception as exc:
            failures.append({
                "snid": snid,
                "reason": repr(exc),
            })

    results.sort(key=lambda row: row["object_id"])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "des_historical_tier1_16.csv"
    audit_path = output_dir / "extraction_audit.json"

    with open(feature_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["object_id"] + FEATURES,
        )

        writer.writeheader()
        writer.writerows(results)

    feature_values = np.array(
        [
            [row[name] for name in FEATURES]
            for row in results
        ],
        dtype=float,
    )

    finite = (
        bool(np.isfinite(feature_values).all())
        if len(feature_values)
        else False
    )

    audit = {
        "stage": "DES historical Tier-1 feature extraction",
        "classifier_executed": False,
        "des_labels_used": False,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "cohort_count": len(ids),
        "cohort_ids_found": len(found),
        "missing_cohort_ids": sorted(cohort_ids - found),
        "successful_extractions": len(results),
        "failed_extractions": failures,
        "all_features_finite": finite,
        "historical_root": str(historical_root),
        "inputs_sha256": {
            "head": sha256(head_path),
            "phot": sha256(phot_path),
            "cohort": sha256(cohort_path),
        },
    }

    with open(audit_path, "w") as handle:
        json.dump(audit, handle, indent=2)

    print("=" * 65)
    print("DES HISTORICAL TIER-1 FEATURE EXTRACTION")
    print("=" * 65)

    print("Frozen cohort:", len(ids))
    print("Objects found:", len(found))
    print("Successful extractions:", len(results))
    print("Failed extractions:", len(failures))
    print("Feature count:", len(FEATURES))
    print("All feature values finite:", finite)

    print()
    print("Feature table:", feature_path)
    print("Audit report:", audit_path)

    if failures:
        print()
        print("FAILURES")
        for failure in failures[:20]:
            print(failure)

    assert found == cohort_ids, "Missing cohort objects"
    assert len(results) == 185, "Not all objects extracted"
    assert not failures, "Feature extraction failures"
    assert finite, "Nonfinite feature values"

    print()
    print("EXTRACTION STATUS: PASS")
    print("No classifier predictions generated.")


if __name__ == "__main__":
    main()
