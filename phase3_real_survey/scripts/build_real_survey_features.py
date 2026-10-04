#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


# =====================================================================
# Phase 3 empirical stage — real compact-feature construction
#
# SCIENTIFIC BOUNDARY
# ---------------------------------------------------------------------
# This script MAY:
#   - read real SDSS-II and DES-SN5YR HEAD/PHOT data
#   - read secure class labels for the frozen inclusion rule
#   - read released redshifts and metadata
#   - apply the frozen structural quality mask
#   - apply the validated SDSS official IGNORE list
#   - calculate the frozen 16 compact features
#   - write separate SDSS and DES feature tables
#
# This script MUST NOT:
#   - compare SDSS and DES feature distributions
#   - calculate cross-survey medians or distances
#   - fit normalization coefficients
#   - tune feature rules
#   - train/evaluate XGBoost
#   - calculate SHAP values
# =====================================================================


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"
DES_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "des"

OUTDIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "real_feature_tables"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SDSS_OUT = OUTDIR / "sdss_compact_features.csv"
DES_OUT = OUTDIR / "des_compact_features.csv"
AUDIT_OUT = OUTDIR / "feature_build_audit.json"
REPORT_OUT = OUTDIR / "feature_build_report.md"


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

Z_MIN = 0.05
Z_MAX = 0.30

SDSS_IGNORE_TOL_DAYS = 0.0030


# =====================================================================
# Release-specific secure spectroscopic mappings
# =====================================================================

# SDSS-II final release:
#
# 118,120 : secure spectroscopic normal Ia
# 111,115 : Ib
# 112     : Ic
# 113,117 : II
#
# Explicitly excluded:
# 101-106 : photometric / host-z-assisted photometric classes
# 119     : Ia?
# other peculiar/unsupported categories
#
SDSS_SECURE_TYPES = {
    118: ("Ia", "Ia"),
    120: ("Ia", "Ia"),
    111: ("CC", "Ib"),
    115: ("CC", "Ib"),
    112: ("CC", "Ic"),
    113: ("CC", "II"),
    117: ("CC", "II"),
}

# DES-SN5YR:
#
# 1  : Ia
# 23 : IIb
# 29 : II
# 32 : Ib
# 33 : Ic
# 39 : Ibc
#
# Explicitly excluded:
# 4    : Ia-pec
# 5    : ambiguous SNI
# 129  : II?
# 139  : Ibc?
# SLSN, AGN, TDE, etc.
#
DES_SECURE_TYPES = {
    1: ("Ia", "Ia"),
    23: ("CC", "IIb"),
    29: ("CC", "II"),
    32: ("CC", "Ib"),
    33: ("CC", "Ic"),
    39: ("CC", "Ibc"),
}


# =====================================================================
# Helpers
# =====================================================================

def decode(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def find_first(root: Path, names: list[str]) -> Path | None:
    matches = []

    for name in names:
        matches.extend(
            p
            for p in root.rglob(name)
            if p.is_file()
        )

    if not matches:
        return None

    return sorted(
        set(matches),
        key=lambda p: (len(str(p)), str(p)),
    )[0]


def resolve_column(
    columns: list[str],
    names: list[str],
) -> str | None:
    lookup = {
        col.upper(): col
        for col in columns
    }

    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]

    return None


def first_table_hdu(path: Path):
    hdul = fits.open(path, memmap=True)

    for hdu in hdul:
        if (
            hasattr(hdu, "columns")
            and hdu.columns is not None
            and hdu.data is not None
        ):
            return hdul, hdu

    hdul.close()
    raise RuntimeError(
        f"No binary table found in {path}"
    )


def signed_log10_1p(x: float) -> float:
    return math.copysign(
        math.log10(1.0 + abs(x)),
        x,
    )


def positive_log10_1p(x: float) -> float:
    return math.log10(
        1.0 + max(float(x), 0.0)
    )


def colour_proxy(
    flux_a: float,
    flux_b: float,
) -> float:

    if flux_a <= 0 or flux_b <= 0:
        return float("nan")

    value = (
        -2.5
        * math.log10(
            flux_a / flux_b
        )
    )

    return float(
        np.clip(
            value,
            -5.0,
            5.0,
        )
    )


def top_positive_mean(
    values: np.ndarray,
    n: int = 3,
) -> float:

    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
        & (arr > 0)
    ]

    if len(arr) == 0:
        return float("nan")

    arr = np.sort(arr)

    return float(
        np.mean(
            arr[-min(n, len(arr)):]
        )
    )


# =====================================================================
# SDSS IGNORE handling
# =====================================================================

IGNORE_PATTERN = re.compile(
    r"^\s*IGNORE:\s+(\S+)\s+([0-9.]+)\s+([A-Za-z])(?:\s+.*)?$"
)


def locate_sdss_ignore_file() -> Path:
    matches = list(
        SDSS_ROOT.rglob(
            "SDSS_allCandidates+BOSS.IGNORE"
        )
    )

    matches = [
        p for p in matches
        if p.is_file()
    ]

    if not matches:
        raise RuntimeError(
            "Validated SDSS .IGNORE file not found."
        )

    return sorted(
        matches,
        key=lambda p: (len(str(p)), str(p)),
    )[0]


def parse_sdss_ignore_file(
    path: Path,
) -> list[dict[str, Any]]:

    entries = []

    for lineno, line in enumerate(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines(),
        start=1,
    ):
        match = IGNORE_PATTERN.match(line)

        if not match:
            continue

        cid, mjd, filt = match.groups()

        entries.append(
            {
                "line": lineno,
                "cid": cid,
                "mjd": float(mjd),
                "filter": filt.lower(),
            }
        )

    return entries


def build_sdss_ignore_map(
    head: Any,
    head_cols: list[str],
    phot: Any,
    phot_cols: list[str],
    entries: list[dict[str, Any]],
) -> tuple[dict[str, set[int]], dict[str, Any]]:

    id_col = resolve_column(
        head_cols,
        ["SNID", "CID"],
    )

    pmin_col = resolve_column(
        head_cols,
        ["PTROBS_MIN"],
    )

    pmax_col = resolve_column(
        head_cols,
        ["PTROBS_MAX"],
    )

    mjd_col = resolve_column(
        phot_cols,
        ["MJD"],
    )

    band_col = resolve_column(
        phot_cols,
        ["FLT", "BAND", "FILTER"],
    )

    if None in (
        id_col,
        pmin_col,
        pmax_col,
        mjd_col,
        band_col,
    ):
        raise RuntimeError(
            "Cannot construct SDSS ignore map."
        )

    head_index = {
        decode(row[id_col]): i
        for i, row in enumerate(head)
    }

    ignored_local_indices: dict[str, set[int]] = {}

    applied = 0
    absent_cid = 0
    ambiguous = 0
    unmatched_present = 0
    max_delta = 0.0

    for entry in entries:

        cid = entry["cid"]

        if cid not in head_index:
            absent_cid += 1
            continue

        hrow = head[
            head_index[cid]
        ]

        pmin = int(
            hrow[pmin_col]
        )

        pmax = int(
            hrow[pmax_col]
        )

        prows = phot[
            pmin - 1:pmax
        ]

        mjd = np.asarray(
            prows[mjd_col],
            dtype=float,
        )

        band = np.asarray(
            [
                decode(v).lower()
                for v in prows[band_col]
            ],
            dtype=object,
        )

        delta = np.abs(
            mjd - entry["mjd"]
        )

        matches = np.where(
            (band == entry["filter"])
            & np.isfinite(mjd)
            & (delta <= SDSS_IGNORE_TOL_DAYS)
        )[0]

        if len(matches) == 1:

            local_index = int(
                matches[0]
            )

            ignored_local_indices.setdefault(
                cid,
                set(),
            ).add(
                local_index
            )

            applied += 1

            max_delta = max(
                max_delta,
                float(
                    delta[
                        local_index
                    ]
                ),
            )

        elif len(matches) == 0:
            unmatched_present += 1

        else:
            ambiguous += 1

    audit = {
        "parsed": len(entries),
        "applied": applied,
        "absent_release_cid": absent_cid,
        "ambiguous": ambiguous,
        "unmatched_present_cid": unmatched_present,
        "tolerance_days": SDSS_IGNORE_TOL_DAYS,
        "maximum_applied_delta_days": max_delta,
    }

    if ambiguous != 0 or unmatched_present != 0:
        raise RuntimeError(
            "SDSS IGNORE mapping became inconsistent with "
            "the validated pre-outcome state."
        )

    return (
        ignored_local_indices,
        audit,
    )


# =====================================================================
# Frozen compact feature builder
# =====================================================================

SNR_ACTIVE_THRESHOLD = 3.0

def frozen_features(
    mjd: np.ndarray,
    band: np.ndarray,
    flux: np.ndarray,
    ferr: np.ndarray,
) -> tuple[
    dict[str, float] | None,
    dict[str, Any]
]:

    order = np.argsort(
        mjd
    )

    mjd = np.asarray(
        mjd,
        dtype=float,
    )[order]

    band = np.asarray(
        band,
        dtype=object,
    )[order]

    flux = np.asarray(
        flux,
        dtype=float,
    )[order]

    ferr = np.asarray(
        ferr,
        dtype=float,
    )[order]

    active = (
        np.isfinite(flux)
        & np.isfinite(ferr)
        & (flux > 0)
        & (ferr > 0)
        & ((flux / ferr) >= 3.0)
    )

    global_fallback = False

    if np.any(active):
        active_idx = np.where(
            active
        )[0]
    else:
        active_idx = np.arange(
            len(mjd)
        )

        global_fallback = True

    if len(active_idx) == 0:
        return None, {
            "reason": "no epochs"
        }

    t0 = float(
        mjd[
            active_idx[0]
        ]
    )

    t1 = float(
        mjd[
            active_idx[-1]
        ]
    )

    time_span = max(
        t1 - t0,
        0.0,
    )

    selected_by_band = {}
    band_fallback = {}

    for b in "griz":

        active_band = (
            active
            & (band == b)
        )

        if np.any(active_band):

            idx = np.where(
                active_band
            )[0]

            band_fallback[b] = False

        else:

            fallback = (
                (band == b)
                & (mjd >= t0)
                & (mjd <= t1)
            )

            idx = np.where(
                fallback
            )[0]

            band_fallback[b] = True

        if len(idx) == 0:
            return None, {
                "reason": (
                    f"missing usable {b} band"
                ),
                "global_fallback": global_fallback,
                "band_fallback": band_fallback,
            }

        selected_by_band[b] = idx

    representative = {}

    for b in "griz":

        representative[b] = (
            top_positive_mean(
                flux[
                    selected_by_band[b]
                ]
            )
        )

        if (
            not np.isfinite(
                representative[b]
            )
            or representative[b] <= 0
        ):
            return None, {
                "reason": (
                    f"nonpositive colour representative {b}"
                ),
                "global_fallback": global_fallback,
                "band_fallback": band_fallback,
            }

    def stats_for_band(b: str):
        idx = selected_by_band[b]

        fb = flux[idx]
        tb = mjd[idx]

        peak_index = int(
            np.argmax(fb)
        )

        return {
            "mean": float(
                np.mean(fb)
            ),
            "std": float(
                np.std(
                    fb,
                    ddof=0,
                )
            ),
            "peak": float(
                fb[
                    peak_index
                ]
            ),
            "peak_time": float(
                np.clip(
                    tb[
                        peak_index
                    ] - t0,
                    0.0,
                    time_span,
                )
            ),
            "amplitude": float(
                np.max(fb)
                - np.min(fb)
            ),
            "n": int(
                len(idx)
            ),
        }

    stats = {
        b: stats_for_band(b)
        for b in "griz"
    }

    features = {
        "z_peak_flux":
            positive_log10_1p(
                stats["z"]["peak"]
            ),

        "r_mean_flux":
            signed_log10_1p(
                stats["r"]["mean"]
            ),

        "peak_color_g_minus_r":
            colour_proxy(
                representative["g"],
                representative["r"],
            ),

        "i_peak_flux":
            positive_log10_1p(
                stats["i"]["peak"]
            ),

        "peak_color_r_minus_i":
            colour_proxy(
                representative["r"],
                representative["i"],
            ),

        "peak_color_i_minus_z":
            colour_proxy(
                representative["i"],
                representative["z"],
            ),

        "g_mean_flux":
            signed_log10_1p(
                stats["g"]["mean"]
            ),

        "r_peak_flux":
            positive_log10_1p(
                stats["r"]["peak"]
            ),

        "z_std_flux":
            positive_log10_1p(
                stats["z"]["std"]
            ),

        "i_amplitude":
            positive_log10_1p(
                stats["i"]["amplitude"]
            ),

        "i_std_flux":
            positive_log10_1p(
                stats["i"]["std"]
            ),

        "time_span":
            float(
                time_span
            ),

        "z_time_of_peak":
            stats["z"]["peak_time"],

        "i_time_of_peak":
            stats["i"]["peak_time"],

        "r_time_of_peak":
            stats["r"]["peak_time"],

        "r_std_flux":
            positive_log10_1p(
                stats["r"]["std"]
            ),
    }

    if not all(
        np.isfinite(
            features[name]
        )
        for name in FEATURES
    ):
        return None, {
            "reason": (
                "nonfinite compact feature"
            )
        }

    diagnostics = {
        "global_fallback":
            global_fallback,

        "band_fallback":
            band_fallback,

        "first_active_mjd":
            t0,

        "last_active_mjd":
            t1,

        "selected_epochs_g":
            stats["g"]["n"],

        "selected_epochs_r":
            stats["r"]["n"],

        "selected_epochs_i":
            stats["i"]["n"],

        "selected_epochs_z":
            stats["z"]["n"],

        "colour_clipped_g_r":
            abs(
                features[
                    "peak_color_g_minus_r"
                ]
            ) >= 5.0,

        "colour_clipped_r_i":
            abs(
                features[
                    "peak_color_r_minus_i"
                ]
            ) >= 5.0,

        "colour_clipped_i_z":
            abs(
                features[
                    "peak_color_i_minus_z"
                ]
            ) >= 5.0,
    }

    return (
        features,
        diagnostics,
    )


# =====================================================================
# Metadata extraction
# =====================================================================

def secure_class(
    survey: str,
    value: Any,
) -> tuple[str | None, str]:

    text = decode(value)

    try:
        code = int(
            float(text)
        )
    except Exception:
        return None, text

    if survey == "SDSS-II":
        return SDSS_SECURE_TYPES.get(
            code,
            (None, f"code:{code}"),
        )

    if survey == "DES-SN5YR":
        return DES_SECURE_TYPES.get(
            code,
            (None, f"code:{code}"),
        )

    return None, text


def extract_redshift(
    row: Any,
    columns: list[str],
) -> tuple[
    float | None,
    str | None
]:

    candidates = [
        "REDSHIFT_HELIO",
        "REDSHIFT_FINAL",
        "REDSHIFT_SPEC",
        "HOSTGAL_SPECZ",
        "REDSHIFT",
    ]

    for name in candidates:

        col = resolve_column(
            columns,
            [name],
        )

        if col is None:
            continue

        try:
            value = float(
                row[col]
            )
        except Exception:
            continue

        if (
            np.isfinite(value)
            and value > 0
        ):
            return (
                value,
                col,
            )

    return (
        None,
        None,
    )


def classify_field(
    survey: str,
    field: str,
) -> str:

    if survey != "DES-SN5YR":
        return field or "UNKNOWN"

    value = field.upper()

    if value in {"C3", "X3"}:
        return "DEEP"

    if value in {
        "C1", "C2",
        "E1", "E2",
        "S1", "S2",
        "X1", "X2",
    }:
        return "SHALLOW"

    return "UNKNOWN"


# =====================================================================
# Survey builder
# =====================================================================

def build_survey(
    survey: str,
    head_path: Path,
    phot_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any]
]:

    head_hdul, head_hdu = (
        first_table_hdu(
            head_path
        )
    )

    phot_hdul, phot_hdu = (
        first_table_hdu(
            phot_path
        )
    )

    try:

        head = head_hdu.data
        phot = phot_hdu.data

        head_cols = list(
            head_hdu.columns.names
            or []
        )

        phot_cols = list(
            phot_hdu.columns.names
            or []
        )

        id_col = resolve_column(
            head_cols,
            ["SNID", "CID"],
        )

        type_col = resolve_column(
            head_cols,
            ["SNTYPE", "TYPE"],
        )

        pmin_col = resolve_column(
            head_cols,
            ["PTROBS_MIN"],
        )

        pmax_col = resolve_column(
            head_cols,
            ["PTROBS_MAX"],
        )

        mw_col = resolve_column(
            head_cols,
            ["MWEBV"],
        )

        head_field_col = resolve_column(
            head_cols,
            ["FIELD"],
        )

        mjd_col = resolve_column(
            phot_cols,
            ["MJD"],
        )

        band_col = resolve_column(
            phot_cols,
            ["FLT", "BAND", "FILTER"],
        )

        flux_col = resolve_column(
            phot_cols,
            ["FLUXCAL"],
        )

        err_col = resolve_column(
            phot_cols,
            ["FLUXCALERR", "FLUXCAL_ERR"],
        )

        phot_field_col = resolve_column(
            phot_cols,
            ["FIELD"],
        )

        required = {
            "SNTYPE": type_col,
            "PTROBS_MIN": pmin_col,
            "PTROBS_MAX": pmax_col,
            "MJD": mjd_col,
            "BAND": band_col,
            "FLUXCAL": flux_col,
            "FLUXCALERR": err_col,
        }

        missing = [
            key
            for key, value
            in required.items()
            if value is None
        ]

        if missing:
            raise RuntimeError(
                f"{survey}: missing required columns: {missing}"
            )

        ignore_map = {}
        ignore_audit = None

        if survey == "SDSS-II":

            ignore_file = (
                locate_sdss_ignore_file()
            )

            ignore_entries = (
                parse_sdss_ignore_file(
                    ignore_file
                )
            )

            (
                ignore_map,
                ignore_audit,
            ) = build_sdss_ignore_map(
                head,
                head_cols,
                phot,
                phot_cols,
                ignore_entries,
            )

        counts = Counter()
        rejection_reasons = Counter()
        redshift_sources = Counter()

        output = []

        for index in range(
            len(head)
        ):

            counts["head_total"] += 1

            hrow = head[index]

            class_name, subtype = (
                secure_class(
                    survey,
                    hrow[type_col],
                )
            )

            if class_name is None:
                counts[
                    "excluded_nonsecure_or_nonprimary_type"
                ] += 1
                continue

            counts[
                f"secure_before_z_{class_name}"
            ] += 1

            z, z_source = extract_redshift(
                hrow,
                head_cols,
            )

            if z is None:
                counts[
                    "excluded_missing_redshift"
                ] += 1
                continue

            redshift_sources[
                z_source
            ] += 1

            if not (
                Z_MIN
                <= z
                <= Z_MAX
            ):
                counts[
                    "excluded_outside_redshift_support"
                ] += 1
                continue

            counts[
                f"in_redshift_support_{class_name}"
            ] += 1

            pmin = int(
                hrow[
                    pmin_col
                ]
            )

            pmax = int(
                hrow[
                    pmax_col
                ]
            )

            if (
                pmin < 1
                or pmax < pmin
                or pmax > len(phot)
            ):
                counts[
                    "excluded_invalid_pointer"
                ] += 1
                continue

            prows = phot[
                pmin - 1:pmax
            ]

            mjd = np.asarray(
                prows[mjd_col],
                dtype=float,
            )

            flux = np.asarray(
                prows[flux_col],
                dtype=float,
            )

            ferr = np.asarray(
                prows[err_col],
                dtype=float,
            )

            bands = np.asarray(
                [
                    decode(v)
                    for v in prows[
                        band_col
                    ]
                ],
                dtype=object,
            )

            object_id = (
                decode(
                    hrow[id_col]
                )
                if id_col is not None
                else (
                    f"{survey}_{index:06d}"
                )
            )

            # ---------------------------------------------------------
            # Apply official SDSS IGNORE rows BEFORE structural mask.
            # ---------------------------------------------------------

            ignored_count = 0

            keep_ignore = np.ones(
                len(prows),
                dtype=bool,
            )

            if (
                survey == "SDSS-II"
                and object_id in ignore_map
            ):

                for local_index in (
                    ignore_map[
                        object_id
                    ]
                ):

                    if (
                        0
                        <= local_index
                        < len(
                            keep_ignore
                        )
                    ):
                        keep_ignore[
                            local_index
                        ] = False

                        ignored_count += 1

            # ---------------------------------------------------------
            # Frozen structural quality mask.
            # ---------------------------------------------------------

            structural = (
                keep_ignore
                & np.isfinite(mjd)
                & np.isfinite(flux)
                & np.isfinite(ferr)
                & (ferr > 0)
                & np.isin(
                    bands,
                    np.asarray(
                        ["g", "r", "i", "z"],
                        dtype=object,
                    ),
                )
            )

            mjd = mjd[
                structural
            ]

            flux = flux[
                structural
            ]

            ferr = ferr[
                structural
            ]

            bands = bands[
                structural
            ]

            if len(mjd) == 0:
                counts[
                    "excluded_no_structural_photometry"
                ] += 1
                continue

            if not all(
                np.any(
                    bands == b
                )
                for b in "griz"
            ):
                counts[
                    "excluded_missing_raw_griz"
                ] += 1
                continue

            # ---------------------------------------------------------
            # Frozen Phase-3 primary cohort support gate.
            #
            # Historical preregistered common-support definition:
            # literal lowercase g,r,i,z and >=1 S/N-active epoch
            # independently in every band.
            #
            # IMPORTANT:
            # This is a population-selection rule only.
            # The validated frozen feature operator below remains
            # completely unchanged, including its fallback semantics.
            # ---------------------------------------------------------
            primary_active = (
                (flux > 0.0)
                & (ferr > 0.0)
                & ((flux / ferr) >= SNR_ACTIVE_THRESHOLD)
            )

            primary_band_support = {
                b: bool(
                    np.any(
                        primary_active
                        & (bands == b)
                    )
                )
                for b in "griz"
            }

            if not all(
                primary_band_support.values()
            ):
                counts[
                    "excluded_primary_four_band_support"
                ] += 1
                continue

            counts[
                "passed_primary_four_band_support"
            ] += 1

            features, diag = (
                frozen_features(
                    mjd,
                    bands,
                    flux,
                    ferr,
                )
            )

            if features is None:

                counts[
                    "excluded_feature_builder"
                ] += 1

                rejection_reasons[
                    diag.get(
                        "reason",
                        "unknown",
                    )
                ] += 1

                continue

            if head_field_col is not None:

                field = decode(
                    hrow[
                        head_field_col
                    ]
                )

            elif phot_field_col is not None:

                available = [
                    decode(v)
                    for v in prows[
                        phot_field_col
                    ][structural]
                    if decode(v)
                    not in {
                        "",
                        "-",
                        "NULL",
                    }
                ]

                field = (
                    Counter(
                        available
                    ).most_common(1)[0][0]
                    if available
                    else "UNKNOWN"
                )

            else:
                field = "UNKNOWN"

            field_stratum = (
                classify_field(
                    survey,
                    field,
                )
            )

            try:
                mwebv = (
                    float(
                        hrow[
                            mw_col
                        ]
                    )
                    if mw_col is not None
                    else float("nan")
                )
            except Exception:
                mwebv = float("nan")

            row = {
                "survey":
                    survey,

                "object_id":
                    object_id,

                "class":
                    class_name,

                "subtype":
                    subtype,

                "z_helio":
                    float(z),

                "redshift_source":
                    z_source,

                "mwebv":
                    mwebv,

                "field":
                    field,

                "field_stratum":
                    field_stratum,

                "ignored_sdss_epochs":
                    ignored_count,

                "structural_epoch_count":
                    int(
                        len(mjd)
                    ),

                "global_fallback":
                    bool(
                        diag[
                            "global_fallback"
                        ]
                    ),

                "fallback_g":
                    bool(
                        diag[
                            "band_fallback"
                        ]["g"]
                    ),

                "fallback_r":
                    bool(
                        diag[
                            "band_fallback"
                        ]["r"]
                    ),

                "fallback_i":
                    bool(
                        diag[
                            "band_fallback"
                        ]["i"]
                    ),

                "fallback_z":
                    bool(
                        diag[
                            "band_fallback"
                        ]["z"]
                    ),

                "selected_epochs_g":
                    diag[
                        "selected_epochs_g"
                    ],

                "selected_epochs_r":
                    diag[
                        "selected_epochs_r"
                    ],

                "selected_epochs_i":
                    diag[
                        "selected_epochs_i"
                    ],

                "selected_epochs_z":
                    diag[
                        "selected_epochs_z"
                    ],

                "first_active_mjd":
                    diag[
                        "first_active_mjd"
                    ],

                "last_active_mjd":
                    diag[
                        "last_active_mjd"
                    ],

                "colour_clipped_g_r":
                    diag[
                        "colour_clipped_g_r"
                    ],

                "colour_clipped_r_i":
                    diag[
                        "colour_clipped_r_i"
                    ],

                "colour_clipped_i_z":
                    diag[
                        "colour_clipped_i_z"
                    ],
            }

            for name in FEATURES:
                row[name] = (
                    features[name]
                )

            output.append(
                row
            )

            counts[
                "accepted"
            ] += 1

            counts[
                f"accepted_{class_name}"
            ] += 1

            counts[
                f"accepted_subtype_{subtype}"
            ] += 1

            if ignored_count:
                counts[
                    "accepted_objects_with_ignore_epoch"
                ] += 1

                counts[
                    "ignored_epochs_on_accepted_objects"
                ] += ignored_count

        audit = {
            "survey":
                survey,

            "rows_written":
                len(output),

            "counts":
                dict(counts),

            "redshift_sources":
                dict(
                    redshift_sources
                ),

            "feature_rejection_reasons":
                dict(
                    rejection_reasons
                ),

            "sdss_ignore":
                ignore_audit,
        }

        return (
            output,
            audit,
        )

    finally:
        head_hdul.close()
        phot_hdul.close()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        raise RuntimeError(
            f"No rows generated for {path}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def print_audit(
    audit: dict[str, Any],
) -> None:

    survey = audit["survey"]
    counts = audit["counts"]

    print()
    print(f"  {survey}")
    print(
        f"    accepted total : "
        f"{audit['rows_written']:,}"
    )
    print(
        f"    accepted Ia    : "
        f"{counts.get('accepted_Ia', 0):,}"
    )
    print(
        f"    accepted CC    : "
        f"{counts.get('accepted_CC', 0):,}"
    )

    print("    accepted subtypes:")

    for key, value in sorted(
        counts.items()
    ):
        if key.startswith(
            "accepted_subtype_"
        ):
            subtype = key.replace(
                "accepted_subtype_",
                "",
            )

            print(
                f"      {subtype:6s}: "
                f"{value:,}"
            )

    if audit.get(
        "sdss_ignore"
    ):

        ig = audit[
            "sdss_ignore"
        ]

        print(
            "    SDSS IGNORE:"
        )
        print(
            f"      parsed             : "
            f"{ig['parsed']}"
        )
        print(
            f"      applicable/applied : "
            f"{ig['applied']}"
        )
        print(
            f"      absent-release CID : "
            f"{ig['absent_release_cid']}"
        )
        print(
            f"      ambiguous          : "
            f"{ig['ambiguous']}"
        )
        print(
            f"      unmatched-present  : "
            f"{ig['unmatched_present_cid']}"
        )
        print(
            f"      max MJD delta      : "
            f"{ig['maximum_applied_delta_days']:.6f} d"
        )


def main() -> int:

    print("=" * 78)
    print("PHASE 3 EMPIRICAL STAGE")
    print("REAL COMPACT-FEATURE TABLE CONSTRUCTION")
    print("NO CROSS-SURVEY FEATURE COMPARISON")
    print("=" * 78)

    sdss_head = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_HEAD.FITS",
            "SDSS_allCandidates+BOSS_HEAD.FITS.gz",
        ],
    )

    sdss_phot = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_PHOT.FITS",
            "SDSS_allCandidates+BOSS_PHOT.FITS.gz",
        ],
    )

    des_head = find_first(
        DES_ROOT,
        [
            "DES-SN5YR_DES_HEAD.FITS.gz",
            "DES-SN5YR_DES_HEAD.FITS",
        ],
    )

    des_phot = find_first(
        DES_ROOT,
        [
            "DES-SN5YR_DES_PHOT.FITS.gz",
            "DES-SN5YR_DES_PHOT.FITS",
        ],
    )

    for label, path in {
        "SDSS HEAD": sdss_head,
        "SDSS PHOT": sdss_phot,
        "DES HEAD": des_head,
        "DES PHOT": des_phot,
    }.items():

        if path is None:
            print(
                f"ERROR: {label} not found."
            )
            return 1

        print(
            f"{label:10s}: {path}"
        )

    print(
        "\n[1] Building SDSS-II real feature table..."
    )

    sdss_rows, sdss_audit = (
        build_survey(
            "SDSS-II",
            sdss_head,
            sdss_phot,
        )
    )

    print_audit(
        sdss_audit
    )

    print(
        "\n[2] Building DES-SN5YR real feature table..."
    )

    des_rows, des_audit = (
        build_survey(
            "DES-SN5YR",
            des_head,
            des_phot,
        )
    )

    print_audit(
        des_audit
    )

    print(
        "\n[3] Writing separate feature tables..."
    )

    write_csv(
        SDSS_OUT,
        sdss_rows,
    )

    write_csv(
        DES_OUT,
        des_rows,
    )

    audit = {
        "stage":
            "empirical_feature_construction",

        "cross_survey_feature_comparison":
            False,

        "classifier_run":
            False,

        "normalization_applied":
            False,

        "frozen_redshift_support": {
            "min": Z_MIN,
            "max": Z_MAX,
        },

        "frozen_features":
            FEATURES,

        "sdss":
            sdss_audit,

        "des":
            des_audit,
    }

    AUDIT_OUT.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        "# Phase 3 Real Feature Tables",
        "",
        "**This stage calculates real compact features but does not "
        "compare SDSS and DES feature distributions.**",
        "",
        "## SDSS-II",
        "",
        f"- Accepted total: {sdss_audit['rows_written']}",
        f"- Ia: {sdss_audit['counts'].get('accepted_Ia', 0)}",
        f"- ordinary CC: {sdss_audit['counts'].get('accepted_CC', 0)}",
        "",
        "## DES-SN5YR",
        "",
        f"- Accepted total: {des_audit['rows_written']}",
        f"- Ia: {des_audit['counts'].get('accepted_Ia', 0)}",
        f"- ordinary CC: {des_audit['counts'].get('accepted_CC', 0)}",
        "",
        "## Frozen controls",
        "",
        "- `0.05 <= z_helio <= 0.30`",
        "- secure transient-spectroscopic primary classes only",
        "- ordinary CC restricted to II/IIb/Ib/Ic/Ibc",
        "- canonical griz structural mask",
        "- official validated SDSS IGNORE list applied",
        "- frozen S/N-active feature extraction",
        "- no feature matching or weighting",
        "- no cross-survey feature statistic calculated",
        "",
        "## Next gate",
        "",
        "Perform table-integrity and selection-count validation only.",
        "Do not inspect SDSS-DES feature differences until that gate passes.",
        "",
    ]

    REPORT_OUT.write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("REAL FEATURE TABLE BUILD: COMPLETE")
    print("=" * 78)

    print(
        f"\nSDSS table : {SDSS_OUT}"
    )

    print(
        f"DES table  : {DES_OUT}"
    )

    print(
        f"Audit      : {AUDIT_OUT}"
    )

    print(
        f"Report     : {REPORT_OUT}"
    )

    print()
    print(
        "STOP HERE: inspect only counts/integrity next. "
        "Do not calculate SDSS-DES feature differences yet."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
