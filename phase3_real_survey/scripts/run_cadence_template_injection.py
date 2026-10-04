#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------
# Phase 3 pre-outcome cadence-template injection — OBJECT LEVEL
#
# Scientific purpose
# ------------------
# Test the preregistered prediction that the realized DES cadence produces
# broader sampled r-band peak-time error than SDSS when the SAME fixed
# supernova template is passed through each survey's actual unlabeled
# object-level observing schedule.
#
# Critical safeguards
# -------------------
# - SNTYPE / class labels are NEVER read.
# - Object redshifts are NEVER read.
# - Real flux values are used ONLY to identify structurally valid rows;
#   they are NEVER injected into the synthetic template.
# - No compact feature is calculated.
# - No classifier is run.
# - No feature-distribution comparison is made.
# - No parameter is tuned using scientific outcomes.
#
# Why object-level?
# -----------------
# SDSS Stripe 82 is a drift-scan survey. Pooling MJDs from all objects into
# one field-level schedule creates tens of thousands of artificial "epochs".
# The scientifically relevant cadence is the schedule actually experienced
# by each transient. HEAD PTROBS_MIN/PTROBS_MAX pointers recover that schedule.
# ---------------------------------------------------------------------


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"
DES_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "des"

OUTDIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "cadence_template_injection"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTDIR / "cadence_injection_summary.csv"
FIELD_CSV = OUTDIR / "cadence_injection_by_field.csv"
REPORT_MD = OUTDIR / "cadence_injection_report.md"
REPORT_JSON = OUTDIR / "cadence_injection_report.json"


# ---------------------------------------------------------------------
# Frozen preregistered artifacts
# ---------------------------------------------------------------------

SDSS_KCOR_NAME = "kcor_SDSS_Bessell90_BD17.fits.gz"
SDSS_KCOR_SHA256 = (
    "7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb"
)

HSIAO_NAME = "Hsiao07.dat"
HSIAO_SHA256 = (
    "6bd032004eccc7b5b52f7c021f0c4b541d801a70195028bf5af0a2578f554d78"
)

REDSHIFTS = np.round(
    np.arange(0.05, 0.3000001, 0.025),
    3,
)

PHASE_MIN = -20.0
PHASE_MAX = 85.0

# Synthetic true peak is inserted at fixed fractions of each REAL observed gap.
GAP_ANCHOR_FRACTIONS = np.array(
    [0.25, 0.50, 0.75],
    dtype=float,
)

# Prevent synthetic transients being centred inside seasonal gaps.
SEASON_GAP_DAYS = 45.0

# Very large within-season holes are not treated as ordinary cadence gaps.
MAX_ANCHOR_GAP_DAYS = 30.0

# Dense interpolation grid for establishing the continuous template peak.
DENSE_PHASE_STEP = 0.02


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def search_roots() -> list[Path]:
    roots = [
        REPO,
        Path(
            "/Users/anuraggarg/work/Feature normalization Phase/"
            "preliminary_hypothesis_test/SNDATA_ROOT_2024-07-03"
        ),
    ]

    out = []
    seen = set()

    for root in roots:
        if root.exists() and str(root) not in seen:
            out.append(root)
            seen.add(str(root))

    return out


def locate_verified(filename: str, expected_hash: str) -> Path:
    candidates = []

    for root in search_roots():
        candidates.extend(
            p for p in root.rglob(filename)
            if p.is_file()
        )

    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"Frozen artifact not found: {filename}"
        )

    mismatches = []

    for path in candidates:
        digest = sha256(path)

        if digest == expected_hash:
            return path

        mismatches.append((path, digest))

    lines = [
        f"No copy of {filename} matches the preregistered SHA-256.",
        f"Expected: {expected_hash}",
    ]

    for path, digest in mismatches:
        lines.append(f"{digest}  {path}")

    raise RuntimeError("\n".join(lines))


def find_first(root: Path, names: list[str]) -> Path | None:
    found = []

    for name in names:
        found.extend(
            p for p in root.rglob(name)
            if p.is_file()
        )

    if not found:
        return None

    return sorted(
        set(found),
        key=lambda p: (len(str(p)), str(p)),
    )[0]


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        ).strip()

    return str(value).strip()


def normalize_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


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
    raise RuntimeError(f"No FITS table found: {path}")


# ---------------------------------------------------------------------
# Frozen response + Hsiao template
# ---------------------------------------------------------------------

def load_sdss_r_response(
    kcor_path: Path,
) -> tuple[np.ndarray, np.ndarray]:

    targets = {
        normalize_name("SDSS-r"),
        normalize_name("SDSS_r"),
        normalize_name("SDSSr"),
    }

    with fits.open(kcor_path, memmap=True) as hdul:

        for hdu in hdul:

            if (
                not hasattr(hdu, "columns")
                or hdu.columns is None
                or hdu.data is None
            ):
                continue

            columns = list(hdu.columns.names or [])

            normalized = {
                normalize_name(c): c
                for c in columns
            }

            filter_col = None

            for target in targets:
                if target in normalized:
                    filter_col = normalized[target]
                    break

            if filter_col is None:
                continue

            wave_col = None

            for candidate in columns:
                if normalize_name(candidate) in {
                    "wave",
                    "wavelength",
                    "lambda",
                    "lam",
                }:
                    wave_col = candidate
                    break

            if wave_col is None:

                for candidate in columns:
                    try:
                        values = np.asarray(
                            hdu.data[candidate],
                            dtype=float,
                        ).reshape(-1)
                    except Exception:
                        continue

                    finite = values[np.isfinite(values)]

                    if (
                        finite.size > 100
                        and finite.min() > 100
                        and finite.max() > 1000
                    ):
                        wave_col = candidate
                        break

            if wave_col is None:
                raise RuntimeError(
                    "Could not identify wavelength column "
                    "for SDSS-r response."
                )

            wave = np.asarray(
                hdu.data[wave_col],
                dtype=float,
            ).reshape(-1)

            trans = np.asarray(
                hdu.data[filter_col],
                dtype=float,
            ).reshape(-1)

            valid = (
                np.isfinite(wave)
                & np.isfinite(trans)
                & (wave > 0)
                & (trans >= 0)
            )

            wave = wave[valid]
            trans = trans[valid]

            order = np.argsort(wave)

            return wave[order], trans[order]

    raise RuntimeError(
        "Frozen SDSS-r response not found inside KCOR."
    )


def load_hsiao(
    path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:

    arr = np.loadtxt(path)

    phases = np.asarray(arr[:, 0], dtype=float)
    waves = np.asarray(arr[:, 1], dtype=float)
    fluxes = np.asarray(arr[:, 2], dtype=float)

    rounded = np.rint(phases).astype(int)

    result = {}

    for phase in sorted(set(rounded)):

        mask = np.isclose(
            phases,
            float(phase),
            atol=1e-8,
        )

        wave = waves[mask]
        flux = fluxes[mask]

        valid = (
            np.isfinite(wave)
            & np.isfinite(flux)
            & (wave > 0)
        )

        wave = wave[valid]
        flux = flux[valid]

        order = np.argsort(wave)

        if len(wave) > 10:
            result[int(phase)] = (
                wave[order],
                flux[order],
            )

    return result


def synthetic_flux(
    sed_wave_rest: np.ndarray,
    sed_flux_rest: np.ndarray,
    response_wave_obs: np.ndarray,
    response: np.ndarray,
    redshift: float,
) -> float:

    rest_wave = (
        response_wave_obs
        / (1.0 + redshift)
    )

    source = np.interp(
        rest_wave,
        sed_wave_rest,
        sed_flux_rest,
        left=0.0,
        right=0.0,
    )

    numerator = np.trapezoid(
        source
        * response_wave_obs
        * response,
        response_wave_obs,
    )

    denominator = np.trapezoid(
        response / response_wave_obs,
        response_wave_obs,
    )

    if numerator <= 0 or denominator <= 0:
        return float("nan")

    return float(
        numerator / denominator
    )


def build_template_curve(
    hsiao: dict[int, tuple[np.ndarray, np.ndarray]],
    response_wave: np.ndarray,
    response: np.ndarray,
    redshift: float,
) -> tuple[np.ndarray, np.ndarray, float]:

    integer_phases = np.array(
        [
            p for p in sorted(hsiao)
            if PHASE_MIN <= p <= PHASE_MAX
        ],
        dtype=float,
    )

    fluxes = []

    for phase in integer_phases:
        wave, flux = hsiao[int(phase)]

        fluxes.append(
            synthetic_flux(
                wave,
                flux,
                response_wave,
                response,
                redshift,
            )
        )

    fluxes = np.asarray(
        fluxes,
        dtype=float,
    )

    dense_phase = np.arange(
        PHASE_MIN,
        PHASE_MAX + DENSE_PHASE_STEP / 2,
        DENSE_PHASE_STEP,
    )

    dense_flux = np.interp(
        dense_phase,
        integer_phases,
        fluxes,
    )

    peak_phase = float(
        dense_phase[
            np.nanargmax(dense_flux)
        ]
    )

    return (
        dense_phase,
        dense_flux,
        peak_phase,
    )


# ---------------------------------------------------------------------
# Object-level schedules
# ---------------------------------------------------------------------

def modal_string(values: np.ndarray) -> str:
    cleaned = [
        decode(v)
        for v in values
        if decode(v) not in {"", "-", "NULL"}
    ]

    if not cleaned:
        return "UNKNOWN"

    return Counter(cleaned).most_common(1)[0][0]


def classify_des_field(field: str) -> str:
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


def split_seasons(times: np.ndarray) -> list[np.ndarray]:
    times = np.sort(
        np.asarray(times, dtype=float)
    )

    if len(times) < 2:
        return []

    gaps = np.diff(times)

    cut = np.where(
        gaps > SEASON_GAP_DAYS
    )[0]

    pieces = []
    start = 0

    for index in cut:
        part = times[start:index + 1]

        if len(part) >= 2:
            pieces.append(part)

        start = index + 1

    part = times[start:]

    if len(part) >= 2:
        pieces.append(part)

    return pieces


def extract_object_schedules(
    head_path: Path,
    phot_path: Path,
    survey: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    head_hdul, head_hdu = first_table_hdu(head_path)
    phot_hdul, phot_hdu = first_table_hdu(phot_path)

    try:
        head_cols = list(head_hdu.columns.names or [])
        phot_cols = list(phot_hdu.columns.names or [])

        ptr_min_col = resolve_column(
            head_cols,
            ["PTROBS_MIN"],
        )

        ptr_max_col = resolve_column(
            head_cols,
            ["PTROBS_MAX"],
        )

        id_col = resolve_column(
            head_cols,
            ["SNID", "CID", "OBJECT_ID"],
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
            "PTROBS_MIN": ptr_min_col,
            "PTROBS_MAX": ptr_max_col,
            "MJD": mjd_col,
            "BAND": band_col,
            "FLUXCAL": flux_col,
            "FLUXCALERR": err_col,
        }

        missing = [
            name
            for name, value in required.items()
            if value is None
        ]

        if missing:
            raise RuntimeError(
                f"{survey}: required columns missing: {missing}"
            )

        head = head_hdu.data
        phot = phot_hdu.data

        nphot = len(phot)

        pmins = np.asarray(
            head[ptr_min_col],
            dtype=int,
        )

        pmaxs = np.asarray(
            head[ptr_max_col],
            dtype=int,
        )

        # SNANA pointers are 1-based and inclusive.
        pointer_ok = (
            np.all(pmins >= 1)
            and np.all(pmaxs >= pmins)
            and np.all(pmaxs <= nphot)
        )

        if not pointer_ok:
            raise RuntimeError(
                f"{survey}: PTROBS pointer convention failed validation."
            )

        schedules = []

        no_r = 0
        one_r = 0
        usable = 0

        for index in range(len(head)):

            pmin = int(pmins[index])
            pmax = int(pmaxs[index])

            rows = phot[
                pmin - 1:pmax
            ]

            mjd = np.asarray(
                rows[mjd_col],
                dtype=float,
            )

            flux = np.asarray(
                rows[flux_col],
                dtype=float,
            )

            ferr = np.asarray(
                rows[err_col],
                dtype=float,
            )

            bands = np.asarray(
                [
                    decode(v).lower()
                    for v in rows[band_col]
                ],
                dtype=object,
            )

            structural = (
                np.isfinite(mjd)
                & np.isfinite(flux)
                & np.isfinite(ferr)
                & (ferr > 0)
                & (bands == "r")
            )

            r_times = mjd[structural]

            if len(r_times) == 0:
                no_r += 1
                continue

            # Duplicate MJDs within one transient do not constitute
            # independent cadence samples.
            r_times = np.unique(
                np.round(
                    r_times,
                    5,
                )
            )

            if len(r_times) < 2:
                one_r += 1
                continue

            if head_field_col is not None:
                field = decode(
                    head[index][head_field_col]
                )
            elif phot_field_col is not None:
                field = modal_string(
                    rows[phot_field_col][structural]
                )
            else:
                field = "UNKNOWN"

            if field in {"", "NULL", "-"}:
                if phot_field_col is not None:
                    field = modal_string(
                        rows[phot_field_col][structural]
                    )
                else:
                    field = "UNKNOWN"

            object_id = (
                decode(head[index][id_col])
                if id_col is not None
                else f"{survey}_{index:06d}"
            )

            if survey == "DES-SN5YR":
                stratum = classify_des_field(field)
            else:
                stratum = field

            schedules.append(
                {
                    "object_id": object_id,
                    "field": field,
                    "stratum": stratum,
                    "times": r_times,
                }
            )

            usable += 1

        diagnostics = {
            "survey": survey,
            "head_rows": int(len(head)),
            "phot_rows": int(len(phot)),
            "pointer_convention": "1-based inclusive",
            "pointer_validation": True,
            "objects_without_r": no_r,
            "objects_with_one_r_epoch": one_r,
            "objects_with_usable_r_schedule": usable,
        }

        return schedules, diagnostics

    finally:
        head_hdul.close()
        phot_hdul.close()


# ---------------------------------------------------------------------
# Vectorized cadence injection
# ---------------------------------------------------------------------

def inject_season_vectorized(
    times: np.ndarray,
    redshift: float,
    dense_phase: np.ndarray,
    dense_flux: np.ndarray,
    template_peak_phase: float,
) -> tuple[np.ndarray, np.ndarray]:

    times = np.sort(
        np.asarray(times, dtype=float)
    )

    if len(times) < 2:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    gaps = np.diff(times)

    good_gap = (
        (gaps > 0)
        & (gaps <= MAX_ANCHOR_GAP_DAYS)
    )

    indices = np.where(good_gap)[0]

    if len(indices) == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    left = times[indices]
    gap = gaps[indices]

    true_peaks = (
        left[:, None]
        + gap[:, None]
        * GAP_ANCHOR_FRACTIONS[None, :]
    ).reshape(-1)

    # Matrix dimensions:
    # n synthetic peak anchors × n real r-band epochs
    rest_phase = (
        (
            times[None, :]
            - true_peaks[:, None]
        )
        / (1.0 + redshift)
        + template_peak_phase
    )

    interpolated = np.interp(
        rest_phase.ravel(),
        dense_phase,
        dense_flux,
        left=np.nan,
        right=np.nan,
    ).reshape(rest_phase.shape)

    finite_counts = np.sum(
        np.isfinite(interpolated),
        axis=1,
    )

    usable = finite_counts >= 2

    if not np.any(usable):
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    matrix = interpolated[usable]
    peaks = true_peaks[usable]

    # Replace NaN by -inf solely for argmax.
    safe = np.where(
        np.isfinite(matrix),
        matrix,
        -np.inf,
    )

    sampled_indices = np.argmax(
        safe,
        axis=1,
    )

    sampled_times = times[
        sampled_indices
    ]

    errors = (
        sampled_times
        - peaks
    )

    return (
        errors,
        np.abs(errors),
    )


def robust_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "mad": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "iqr": float("nan"),
        }

    median = float(np.median(arr))

    q10, q25, q75, q90 = np.quantile(
        arr,
        [0.10, 0.25, 0.75, 0.90],
    )

    return {
        "n": int(len(arr)),
        "median": median,
        "mad": float(
            np.median(
                np.abs(arr - median)
            )
        ),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "iqr": float(q75 - q25),
    }


def main() -> int:

    print("=" * 78)
    print("PHASE 3 CADENCE-TEMPLATE INJECTION — OBJECT LEVEL")
    print("UNLABELED SCHEDULES ONLY - NO REAL COMPACT FEATURES")
    print("=" * 78)

    print("\n[1] Verifying frozen artifacts...")

    kcor = locate_verified(
        SDSS_KCOR_NAME,
        SDSS_KCOR_SHA256,
    )

    hsiao_path = locate_verified(
        HSIAO_NAME,
        HSIAO_SHA256,
    )

    print(f"  SDSS-r KCOR : PASS")
    print(f"    {kcor}")

    print(f"  Hsiao07     : PASS")
    print(f"    {hsiao_path}")

    print("\n[2] Locating survey products...")

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

    for name, path in {
        "SDSS HEAD": sdss_head,
        "SDSS PHOT": sdss_phot,
        "DES HEAD": des_head,
        "DES PHOT": des_phot,
    }.items():

        if path is None:
            print(f"ERROR: {name} not found.")
            return 1

        print(f"  {name:10s}: {path}")

    print("\n[3] Recovering object-level unlabeled schedules...")

    sdss_sched, sdss_diag = extract_object_schedules(
        sdss_head,
        sdss_phot,
        "SDSS-II",
    )

    des_sched, des_diag = extract_object_schedules(
        des_head,
        des_phot,
        "DES-SN5YR",
    )

    for diag in (sdss_diag, des_diag):

        print(f"\n  {diag['survey']}")
        print(
            f"    HEAD objects                 : "
            f"{diag['head_rows']:,}"
        )
        print(
            f"    usable object r schedules    : "
            f"{diag['objects_with_usable_r_schedule']:,}"
        )
        print(
            f"    objects without r            : "
            f"{diag['objects_without_r']:,}"
        )
        print(
            f"    objects with only one r epoch: "
            f"{diag['objects_with_one_r_epoch']:,}"
        )

    print("\n  Schedule-size distribution")

    for survey, schedules in (
        ("SDSS-II", sdss_sched),
        ("DES-SN5YR", des_sched),
    ):

        counts = np.array(
            [
                len(item["times"])
                for item in schedules
            ],
            dtype=float,
        )

        q10, med, q90 = np.quantile(
            counts,
            [0.10, 0.50, 0.90],
        )

        print(
            f"    {survey:10s}: "
            f"10/50/90% = "
            f"{q10:.0f}/{med:.0f}/{q90:.0f} r epochs"
        )

    print("\n[4] Building fixed common r-band template...")

    response_wave, response = load_sdss_r_response(
        kcor
    )

    hsiao = load_hsiao(
        hsiao_path
    )

    templates = {}

    for z in REDSHIFTS:

        templates[float(z)] = build_template_curve(
            hsiao,
            response_wave,
            response,
            float(z),
        )

        peak_phase = templates[float(z)][2]

        print(
            f"  z={z:.3f}: continuous peak phase "
            f"{peak_phase:+.3f} rest-frame days"
        )

    print("\n[5] Running object-level cadence injections...")

    # Aggregation only. We intentionally do not write millions of
    # individual synthetic injections to disk.
    signed = defaultdict(list)
    absolute = defaultdict(list)

    survey_sets = [
        ("SDSS-II", sdss_sched),
        ("DES-SN5YR", des_sched),
    ]

    for survey, schedules in survey_sets:

        print(
            f"\n  Processing {survey}: "
            f"{len(schedules):,} object schedules"
        )

        for object_number, item in enumerate(
            schedules,
            start=1,
        ):

            seasons = split_seasons(
                item["times"]
            )

            if not seasons:
                continue

            for z in REDSHIFTS:

                (
                    dense_phase,
                    dense_flux,
                    peak_phase,
                ) = templates[float(z)]

                for season in seasons:

                    errors, abs_errors = inject_season_vectorized(
                        season,
                        float(z),
                        dense_phase,
                        dense_flux,
                        peak_phase,
                    )

                    if len(errors) == 0:
                        continue

                    key_all = (
                        survey,
                        float(z),
                        "ALL",
                    )

                    key_field = (
                        survey,
                        float(z),
                        item["stratum"],
                    )

                    signed[key_all].extend(
                        errors.tolist()
                    )

                    absolute[key_all].extend(
                        abs_errors.tolist()
                    )

                    signed[key_field].extend(
                        errors.tolist()
                    )

                    absolute[key_field].extend(
                        abs_errors.tolist()
                    )

            if (
                object_number % 2000 == 0
                or object_number == len(schedules)
            ):
                print(
                    f"    {object_number:,}/"
                    f"{len(schedules):,} objects complete"
                )

    print("\n[6] Computing preregistered summaries...")

    summary_rows = []

    for z in REDSHIFTS:

        for survey in (
            "SDSS-II",
            "DES-SN5YR",
        ):

            key = (
                survey,
                float(z),
                "ALL",
            )

            s = robust_stats(
                signed[key]
            )

            a = robust_stats(
                absolute[key]
            )

            summary_rows.append(
                {
                    "survey": survey,
                    "redshift": float(z),
                    "stratum": "ALL",
                    "n_injections": s["n"],
                    "median_error_days": s["median"],
                    "mad_error_days": s["mad"],
                    "q10_error_days": s["q10"],
                    "q90_error_days": s["q90"],
                    "iqr_error_days": s["iqr"],
                    "median_abs_error_days": a["median"],
                    "q90_abs_error_days": a["q90"],
                }
            )

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    field_rows = []

    strata = sorted(
        {
            key[2]
            for key in signed
            if key[2] != "ALL"
        }
    )

    for z in REDSHIFTS:

        for survey in (
            "SDSS-II",
            "DES-SN5YR",
        ):

            survey_strata = sorted(
                {
                    key[2]
                    for key in signed
                    if (
                        key[0] == survey
                        and key[2] != "ALL"
                    )
                }
            )

            for stratum in survey_strata:

                key = (
                    survey,
                    float(z),
                    stratum,
                )

                s = robust_stats(
                    signed[key]
                )

                a = robust_stats(
                    absolute[key]
                )

                if s["n"] == 0:
                    continue

                field_rows.append(
                    {
                        "survey": survey,
                        "redshift": float(z),
                        "stratum": stratum,
                        "n_injections": s["n"],
                        "median_error_days": s["median"],
                        "mad_error_days": s["mad"],
                        "q10_error_days": s["q10"],
                        "q90_error_days": s["q90"],
                        "iqr_error_days": s["iqr"],
                        "median_abs_error_days": a["median"],
                        "q90_abs_error_days": a["q90"],
                    }
                )

    if field_rows:

        with FIELD_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    field_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(field_rows)

    print(
        "\nz      SDSS_med_abs  DES_med_abs   "
        "SDSS_IQR   DES_IQR"
    )

    comparison = []

    for z in REDSHIFTS:

        srow = next(
            row
            for row in summary_rows
            if (
                row["survey"] == "SDSS-II"
                and math.isclose(
                    row["redshift"],
                    float(z),
                )
            )
        )

        drow = next(
            row
            for row in summary_rows
            if (
                row["survey"] == "DES-SN5YR"
                and math.isclose(
                    row["redshift"],
                    float(z),
                )
            )
        )

        d_medabs = (
            drow["median_abs_error_days"]
            - srow["median_abs_error_days"]
        )

        d_iqr = (
            drow["iqr_error_days"]
            - srow["iqr_error_days"]
        )

        comparison.append(
            {
                "z": float(z),
                "des_minus_sdss_median_abs_error_days": d_medabs,
                "des_minus_sdss_iqr_days": d_iqr,
            }
        )

        print(
            f"{z:.3f}   "
            f"{srow['median_abs_error_days']:12.4f}  "
            f"{drow['median_abs_error_days']:11.4f}   "
            f"{srow['iqr_error_days']:8.4f}  "
            f"{drow['iqr_error_days']:8.4f}"
        )

    n_z = len(comparison)

    medabs_larger = sum(
        row[
            "des_minus_sdss_median_abs_error_days"
        ] > 0
        for row in comparison
    )

    iqr_larger = sum(
        row[
            "des_minus_sdss_iqr_days"
        ] > 0
        for row in comparison
    )

    if medabs_larger == 0 and iqr_larger == 0:
        interpretation = "CONTRADICTS_REGISTERED_DIRECTION"
    elif medabs_larger == n_z and iqr_larger == n_z:
        interpretation = "FULLY_CONSISTENT_WITH_REGISTERED_DIRECTION"
    else:
        interpretation = "MIXED_DIRECTIONAL_SUPPORT"

    print("\n[7] Registered-direction diagnostic")

    print(
        f"  DES median absolute error larger: "
        f"{medabs_larger}/{n_z} redshifts"
    )

    print(
        f"  DES signed-error IQR larger: "
        f"{iqr_larger}/{n_z} redshifts"
    )

    print(
        f"  Interpretation: {interpretation}"
    )

    report = {
        "method_version": "object_level_v2",
        "class_labels_read": False,
        "object_redshifts_read": False,
        "real_flux_used_for_science": False,
        "compact_features_calculated": False,
        "same_response_for_both_surveys": (
            "SDSS Doi2010 CCDAVG r"
        ),
        "redshift_grid": [
            float(z)
            for z in REDSHIFTS
        ],
        "gap_anchor_fractions": (
            GAP_ANCHOR_FRACTIONS.tolist()
        ),
        "season_gap_days": SEASON_GAP_DAYS,
        "max_anchor_gap_days": MAX_ANCHOR_GAP_DAYS,
        "sdss_schedule_diagnostics": sdss_diag,
        "des_schedule_diagnostics": des_diag,
        "des_median_abs_larger_count": medabs_larger,
        "des_iqr_larger_count": iqr_larger,
        "redshift_count": n_z,
        "interpretation": interpretation,
        "comparison": comparison,
    }

    REPORT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        "# Phase 3 Object-Level Cadence Injection",
        "",
        "## Scientific boundary",
        "",
        "- No class labels were read.",
        "- No object redshifts were read.",
        "- No real compact features were calculated.",
        "- Real flux values were not used as template measurements.",
        "- The same frozen SDSS Doi2010 r response was used for both surveys.",
        "- Cadence was reconstructed from each object's own HEAD/PHOT pointer range.",
        "",
        "## Registered-direction result",
        "",
        f"- DES median absolute timing error larger at "
        f"**{medabs_larger}/{n_z}** frozen redshift points.",
        f"- DES signed-error IQR larger at "
        f"**{iqr_larger}/{n_z}** frozen redshift points.",
        f"- Interpretation: **{interpretation}**.",
        "",
        "## Important correction",
        "",
        "The original field-union cadence implementation was abandoned before "
        "scientific use because SDSS drift-scan MJDs are object/location-specific. "
        "This implementation uses each transient's actual released observing "
        "schedule and therefore does not manufacture a global SDSS cadence.",
        "",
        "No arbitrary success threshold was introduced after seeing the result.",
        "",
    ]

    REPORT_MD.write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(
        f"CADENCE-TEMPLATE INJECTION: {interpretation}"
    )
    print("=" * 78)

    print(f"\nSummary : {SUMMARY_CSV}")
    print(f"Strata  : {FIELD_CSV}")
    print(f"Report  : {REPORT_MD}")
    print(f"JSON    : {REPORT_JSON}")

    if interpretation == "CONTRADICTS_REGISTERED_DIRECTION":
        print(
            "\nSTOP: the preregistered cadence mechanism is contradicted "
            "before real compact-feature inspection."
        )
        return 1

    print(
        "\nIf this mechanism is not contradicted, the next step is to freeze "
        "the validated pre-outcome machinery before generating real SDSS/DES "
        "compact-feature tables."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
