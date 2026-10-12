"""Stage 3C Step 5 record validation.

No production data is loaded by this module.
"""

import math


BRANCHES = (
    ("SDSS-II", "82N"),
    ("SDSS-II", "82S"),
    ("DES-SN5YR", "SHALLOW"),
    ("DES-SN5YR", "DEEP"),
)

PRIMARY_FEATURES = (
    "peak_color_r_minus_i",
    "peak_color_i_minus_z",
    "i_peak_flux",
    "z_peak_flux",
    "r_time_of_peak",
)

SUPPORTIVE_FEATURES = ("time_span",)
REGISTERED_FEATURES = PRIMARY_FEATURES + SUPPORTIVE_FEATURES

REDSHIFTS = tuple(round(0.05 + 0.025 * i, 3) for i in range(11))
SBMAGS = tuple(20.5 + i for i in range(8))
N_REALIZATIONS = 500


def validate_record(record):
    """Validate one Stage 3C production-shaped record."""

    if not isinstance(record, dict):
        raise ValueError("Record must be a dictionary.")

    required = (
        "survey",
        "stratum",
        "redshift",
        "sbmag",
        "realization_id",
        "peak_mjd",
        "feature_status",
        "feature_rejection_reason",
        "feature_audit",
        "features",
    )

    for field in required:
        if field not in record:
            raise ValueError(f"Missing required field: {field}")

    branch = (record["survey"], record["stratum"])

    if branch not in BRANCHES:
        raise ValueError(f"Unknown survey branch: {branch}")

    z = float(record["redshift"])
    sbmag = float(record["sbmag"])

    if not math.isfinite(z) or not math.isfinite(sbmag):
        raise ValueError("Nonfinite redshift or sbmag.")

    if z < 0:
        raise ValueError("Negative redshift.")

    realization_id = record["realization_id"]

    if (
        isinstance(realization_id, bool)
        or not isinstance(realization_id, int)
        or realization_id < 0
    ):
        raise ValueError("Invalid realization_id.")

    if not math.isfinite(float(record["peak_mjd"])):
        raise ValueError("Invalid peak_mjd.")

    status = record["feature_status"]

    if status not in ("accepted", "rejected"):
        raise ValueError("Invalid feature_status.")

    features = record["features"]
    audit = record["feature_audit"]

    if not isinstance(audit, dict):
        raise ValueError("feature_audit must be a dictionary.")

    if status == "rejected":
        if features is not None:
            raise ValueError(
                "Rejected record must have features=None."
            )

        if record["feature_rejection_reason"] is None:
            raise ValueError(
                "Rejected record requires rejection reason."
            )

    else:
        if not isinstance(features, dict):
            raise ValueError(
                "Accepted record requires feature dictionary."
            )

        if record["feature_rejection_reason"] is not None:
            raise ValueError(
                "Accepted record must not have rejection reason."
            )

        for name in REGISTERED_FEATURES:
            if name not in features:
                raise ValueError(
                    f"Missing primary feature: {name}"
                )

            value = features[name]

            # Individual features may be unavailable in accepted records.
            if value is None:
                continue

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Nonfinite primary feature: {name}"
                )

        first_active = audit.get("first_active_mjd")

        if first_active is None:
            raise ValueError("Missing first_active_mjd.")

        if not math.isfinite(float(first_active)):
            raise ValueError("Invalid first_active_mjd.")

    return (
        record["survey"],
        record["stratum"],
        z,
        sbmag,
        realization_id,
    )


def validate_grid(
    records,
    *,
    branches=BRANCHES,
    redshifts=REDSHIFTS,
    sbmags=SBMAGS,
    n_realizations=N_REALIZATIONS,
):
    """Validate an exact registered-style grid, including rejected records."""

    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive.")

    branches = tuple(branches)
    redshifts = tuple(float(value) for value in redshifts)
    sbmags = tuple(float(value) for value in sbmags)
    expected_ids = set(range(n_realizations))

    expected_cells = {
        (survey, stratum, z, sbmag)
        for survey, stratum in branches
        for z in redshifts
        for sbmag in sbmags
    }

    observed = {}
    total = 0

    for record in records:
        survey, stratum, z, sbmag, realization_id = validate_record(record)

        cell = (survey, stratum, z, sbmag)

        if cell not in expected_cells:
            raise ValueError(f"Unexpected grid cell: {cell}")

        ids = observed.setdefault(cell, set())

        if realization_id in ids:
            raise ValueError(
                f"Duplicate realization: {cell}, ID={realization_id}"
            )

        ids.add(realization_id)
        total += 1

    missing_cells = expected_cells - observed.keys()

    if missing_cells:
        raise ValueError(
            f"Missing grid cells: {len(missing_cells)}"
        )

    for cell in sorted(expected_cells):
        ids = observed[cell]

        missing = expected_ids - ids
        unexpected = ids - expected_ids

        if missing or unexpected:
            raise ValueError(
                f"Incomplete grid cell {cell}: "
                f"missing={sorted(missing)[:10]}, "
                f"unexpected={sorted(unexpected)[:10]}"
            )

    expected_total = len(expected_cells) * n_realizations

    if total != expected_total:
        raise ValueError(
            f"Unexpected total record count: {total}"
        )

    return {
        "total_records": total,
        "expected_records": expected_total,
        "cell_count": len(expected_cells),
        "realizations_per_cell": n_realizations,
        "complete": True,
    }


def validate_complete_grid(records):
    """Validate the exact frozen 176,000-record Stage 3C grid."""

    summary = validate_grid(records)

    if summary["total_records"] != 176000:
        raise ValueError(
            "Frozen Stage 3C grid must contain 176000 records."
        )

    # Preserve the original validator API: callers historically received the
    # validated record count rather than the generic validator's rich summary.
    return summary["total_records"]


def validate_records(records):
    """Validate records and reject duplicate branch/cell/ID keys."""

    seen = set()

    for record in records:
        key = validate_record(record)

        if key in seen:
            raise ValueError(f"Duplicate record key: {key}")

        seen.add(key)

    return len(seen)
