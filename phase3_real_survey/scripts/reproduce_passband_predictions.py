#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------
# Phase 3 pre-outcome validation:
# reproduce the preregistered SDSS-II <-> DES-SN5YR passband predictions.
#
# THIS SCRIPT MUST NOT:
#   - read SDSS or DES PHOT/HEAD data
#   - calculate real compact features
#   - inspect labels
#   - calculate classifier outputs
#   - fit any coefficient to survey data
#
# It uses only:
#   - frozen SDSS Doi2010 KCOR responses
#   - frozen DES DECam responses
#   - frozen Hsiao07 normal-Ia SED
#   - preregistered redshift/phase grids
# ---------------------------------------------------------------------


REPO = Path(__file__).resolve().parents[2]

OUTDIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "passband_validation"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Frozen artifact identities from preregistration
# ---------------------------------------------------------------------

EXPECTED = {
    "sdss_kcor": {
        "filename": "kcor_SDSS_Bessell90_BD17.fits.gz",
        "sha256": "7febdd9faa7d6032a3a264446773a1032d8cb71160240d8d8f8c4709a47355eb",
    },
    "des_g": {
        "filename": "DECam_g.dat",
        "sha256": "671d50b64e3efdf6b848f59e1625d63c20998b64c10ada8ddead5ca4fe0394f4",
    },
    "des_r": {
        "filename": "DECam_r.dat",
        "sha256": "3dec154b19040b6fc8a3a08c94d213866114e062be93816e9a73ec0a301c3a21",
    },
    "des_i": {
        "filename": "DECam_i.dat",
        "sha256": "e0fe7030e997ff51bc1933478dcc6619cbe06d3f07ee78ece0c84c9524f33175",
    },
    "des_z": {
        "filename": "DECam_z.dat",
        "sha256": "078ebb6ea401ae22d55bcd35fa6e99eb0e06933dd877c02a5c00618b53dd9064",
    },
    "hsiao": {
        "filename": "Hsiao07.dat",
        "sha256": "6bd032004eccc7b5b52f7c021f0c4b541d801a70195028bf5af0a2578f554d78",
    },
}


# ---------------------------------------------------------------------
# Registered synthetic values from passband_prediction_notes.md
#
# DES - SDSS magnitude differences.
# These values were preregistered before inspection of real compact features.
# ---------------------------------------------------------------------

REGISTERED = {
    0.050: [-0.00136, +0.03518, +0.10122, -0.03147, -0.03667, -0.06808, +0.13476],
    0.075: [+0.02296, +0.01191, +0.13316, -0.03141, +0.01071, -0.12469, +0.16652],
    0.100: [+0.03064, +0.02805, +0.17902, -0.03773, +0.00229, -0.15069, +0.21670],
    0.125: [+0.01421, +0.04039, +0.13974, -0.03702, -0.02709, -0.09757, +0.17661],
    0.150: [-0.01074, +0.02793, +0.10752, -0.00055, -0.03992, -0.07706, +0.10562],
    0.175: [-0.03336, -0.01624, +0.13844, +0.05068, -0.01886, -0.15310, +0.08557],
    0.200: [-0.04228, -0.02630, +0.12042, +0.07572, -0.01609, -0.14602, +0.04299],
    0.225: [-0.02829, -0.00547, +0.10497, +0.07550, -0.02236, -0.10977, +0.02665],
    0.250: [-0.02118, +0.00868, +0.07993, +0.09781, -0.02926, -0.07212, -0.02051],
    0.275: [-0.04190, +0.02140, +0.02924, +0.14018, -0.06583, -0.00865, -0.11337],
    0.300: [-0.09037, +0.02398, -0.00697, +0.17906, -0.11309, +0.02948, -0.18474],
}

REGISTERED_COLUMNS = [
    "dpeak_g",
    "dpeak_r",
    "dpeak_i",
    "dpeak_z",
    "dtop3_g-r",
    "dtop3_r-i",
    "dtop3_i-z",
]

# Comparison tolerance against values rounded to 5 decimal places
TOLERANCE_MAG = 5.0e-4


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

    env_root = None
    try:
        import os
        if os.environ.get("SNDATA_ROOT"):
            env_root = Path(os.environ["SNDATA_ROOT"])
    except Exception:
        pass

    if env_root is not None:
        roots.append(env_root)

    # Remove duplicates while preserving order.
    result = []
    seen = set()
    for r in roots:
        key = str(r)
        if key not in seen and r.exists():
            seen.add(key)
            result.append(r)

    return result


def locate_frozen_artifact(key: str) -> Path:
    spec = EXPECTED[key]
    filename = spec["filename"]
    expected_hash = spec["sha256"]

    candidates = []

    for root in search_roots():
        try:
            candidates.extend(root.rglob(filename))
        except Exception:
            continue

    candidates = sorted(set(p for p in candidates if p.is_file()))

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate frozen artifact: {filename}"
        )

    wrong = []

    for path in candidates:
        digest = sha256(path)
        if digest == expected_hash:
            return path
        wrong.append((path, digest))

    lines = [
        f"Found {filename}, but no copy has the preregistered SHA-256.",
        f"Expected: {expected_hash}",
        "",
        "Found:",
    ]

    for path, digest in wrong:
        lines.append(f"  {digest}  {path}")

    raise RuntimeError("\n".join(lines))


def load_des_response(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Unexpected DES response format: {path}")

    wave = np.asarray(arr[:, 0], dtype=float)
    trans = np.asarray(arr[:, 1], dtype=float)

    mask = (
        np.isfinite(wave)
        & np.isfinite(trans)
        & (wave > 0)
        & (trans >= 0)
    )

    wave = wave[mask]
    trans = trans[mask]

    order = np.argsort(wave)

    return wave[order], trans[order]


def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def load_sdss_response(
    kcor_path: Path,
    band: str,
) -> tuple[np.ndarray, np.ndarray]:

    target_variants = {
        normalize_name(f"SDSS-{band}"),
        normalize_name(f"SDSS_{band}"),
        normalize_name(f"SDSS{band}"),
    }

    with fits.open(kcor_path, memmap=True) as hdul:
        for hdu in hdul:
            if not hasattr(hdu, "columns") or hdu.columns is None:
                continue

            columns = list(hdu.columns.names or [])
            if not columns:
                continue

            normalized = {
                normalize_name(c): c
                for c in columns
            }

            filter_col = None

            for target in target_variants:
                if target in normalized:
                    filter_col = normalized[target]
                    break

            if filter_col is None:
                continue

            wave_col = None

            for candidate in columns:
                n = normalize_name(candidate)
                if n in {
                    "wave",
                    "wavelength",
                    "lambda",
                    "lam",
                }:
                    wave_col = candidate
                    break

            if wave_col is None:
                # KCOR FilterTrans normally has wavelength first.
                for candidate in columns:
                    values = np.asarray(hdu.data[candidate]).reshape(-1)
                    if values.size > 100:
                        try:
                            vv = values.astype(float)
                            finite = vv[np.isfinite(vv)]
                            if (
                                finite.size > 100
                                and np.nanmin(finite) > 100
                                and np.nanmax(finite) > 1000
                            ):
                                wave_col = candidate
                                break
                        except Exception:
                            pass

            if wave_col is None:
                raise RuntimeError(
                    "Found SDSS filter column "
                    f"{filter_col} but could not identify wavelength column."
                )

            wave = np.asarray(hdu.data[wave_col], dtype=float).reshape(-1)
            trans = np.asarray(hdu.data[filter_col], dtype=float).reshape(-1)

            mask = (
                np.isfinite(wave)
                & np.isfinite(trans)
                & (wave > 0)
                & (trans >= 0)
            )

            wave = wave[mask]
            trans = trans[mask]

            order = np.argsort(wave)

            return wave[order], trans[order]

    # Diagnostic if not found
    available = []

    with fits.open(kcor_path, memmap=True) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.columns is not None:
                available.extend(list(hdu.columns.names or []))

    raise RuntimeError(
        f"Could not find SDSS-{band} response in {kcor_path}\n"
        f"Available FITS columns include:\n{available}"
    )


def load_hsiao(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    raw = np.loadtxt(path)

    if raw.ndim != 2 or raw.shape[1] < 3:
        raise RuntimeError(
            "Hsiao07.dat was expected to have at least "
            "phase, wavelength, flux columns."
        )

    phase = np.asarray(raw[:, 0], dtype=float)
    wave = np.asarray(raw[:, 1], dtype=float)
    flux = np.asarray(raw[:, 2], dtype=float)

    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    rounded_phases = np.rint(phase).astype(int)

    for p in sorted(set(rounded_phases)):
        mask = np.isclose(phase, float(p), atol=1e-8)

        w = wave[mask]
        f = flux[mask]

        valid = (
            np.isfinite(w)
            & np.isfinite(f)
            & (w > 0)
        )

        w = w[valid]
        f = f[valid]

        order = np.argsort(w)

        if len(w) > 10:
            result[p] = (w[order], f[order])

    return result


def pivot_wavelength(
    wave: np.ndarray,
    trans: np.ndarray,
) -> float:
    numerator = np.trapezoid(trans * wave, wave)
    denominator = np.trapezoid(trans / wave, wave)

    return math.sqrt(numerator / denominator)


def synthetic_flux_proxy(
    sed_wave_rest: np.ndarray,
    sed_flux_rest: np.ndarray,
    response_wave_obs: np.ndarray,
    response: np.ndarray,
    redshift: float,
) -> float:

    rest_wave_needed = response_wave_obs / (1.0 + redshift)

    source = np.interp(
        rest_wave_needed,
        sed_wave_rest,
        sed_flux_rest,
        left=0.0,
        right=0.0,
    )

    # The 1/(1+z) f_lambda factor is common to all bands/surveys
    # for a fixed source/redshift and therefore cancels in all registered
    # relative quantities. We omit it intentionally.
    numerator = np.trapezoid(
        source * response_wave_obs * response,
        response_wave_obs,
    )

    denominator = np.trapezoid(
        response / response_wave_obs,
        response_wave_obs,
    )

    if numerator <= 0 or denominator <= 0:
        return float("nan")

    return numerator / denominator


def magnitude_from_proxy(flux: float) -> float:
    if not np.isfinite(flux) or flux <= 0:
        return float("nan")

    return -2.5 * math.log10(flux)


def top_three_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]

    if arr.size == 0:
        return float("nan")

    arr = np.sort(arr)

    return float(np.mean(arr[-min(3, arr.size):]))


def colour_from_fluxes(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("nan")

    return -2.5 * math.log10(a / b)


def main() -> int:

    print("=" * 78)
    print("PHASE 3 PASSBAND PREDICTION REPRODUCTION")
    print("NO REAL SDSS/DES FEATURE DATA ARE READ")
    print("=" * 78)

    print("\n[1] Locating and verifying frozen artifacts...")

    paths = {}

    for key in EXPECTED:
        paths[key] = locate_frozen_artifact(key)
        print(f"  {key:12s}: PASS")
        print(f"                 {paths[key]}")

    print("\n[2] Loading response curves...")

    sdss = {}
    des = {}

    for band in "griz":
        sdss[band] = load_sdss_response(
            paths["sdss_kcor"],
            band,
        )

        des[band] = load_des_response(
            paths[f"des_{band}"]
        )

    print("\nResponse pivot wavelengths [Angstrom]")
    print("        SDSS       DES")

    pivots = {}

    for band in "griz":
        ps = pivot_wavelength(*sdss[band])
        pd = pivot_wavelength(*des[band])

        pivots[band] = {
            "sdss": ps,
            "des": pd,
        }

        print(
            f"  {band}: "
            f"{ps:9.2f}  "
            f"{pd:9.2f}"
        )

    print("\n[3] Loading Hsiao07 template...")

    hsiao = load_hsiao(paths["hsiao"])

    required_phases = set(range(-20, 86))

    missing = sorted(required_phases - set(hsiao))

    if missing:
        print(
            "ERROR: Hsiao template does not contain all preregistered "
            "daily phases -20..+85."
        )
        print("Missing:", missing)
        return 1

    print(
        f"  Loaded {len(hsiao)} integer phases; "
        "required dense grid is complete."
    )

    redshifts = np.round(
        np.arange(0.05, 0.3000001, 0.025),
        3,
    )

    dense_phases = list(range(-20, 86))

    rows = []

    print("\n[4] Reproducing dense-template preregistered predictions...")

    for z in redshifts:

        survey_fluxes = {
            "sdss": {b: [] for b in "griz"},
            "des": {b: [] for b in "griz"},
        }

        for phase in dense_phases:
            sed_wave, sed_flux = hsiao[phase]

            for band in "griz":

                fs = synthetic_flux_proxy(
                    sed_wave,
                    sed_flux,
                    sdss[band][0],
                    sdss[band][1],
                    float(z),
                )

                fd = synthetic_flux_proxy(
                    sed_wave,
                    sed_flux,
                    des[band][0],
                    des[band][1],
                    float(z),
                )

                survey_fluxes["sdss"][band].append(fs)
                survey_fluxes["des"][band].append(fd)

        dpeak = {}

        for band in "griz":

            max_sdss = np.nanmax(
                survey_fluxes["sdss"][band]
            )

            max_des = np.nanmax(
                survey_fluxes["des"][band]
            )

            dpeak[band] = (
                magnitude_from_proxy(max_des)
                - magnitude_from_proxy(max_sdss)
            )

        top = {
            survey: {
                band: top_three_mean(
                    survey_fluxes[survey][band]
                )
                for band in "griz"
            }
            for survey in ("sdss", "des")
        }

        colours = {}

        for name, a, b in [
            ("g-r", "g", "r"),
            ("r-i", "r", "i"),
            ("i-z", "i", "z"),
        ]:

            c_sdss = colour_from_fluxes(
                top["sdss"][a],
                top["sdss"][b],
            )

            c_des = colour_from_fluxes(
                top["des"][a],
                top["des"][b],
            )

            colours[name] = c_des - c_sdss

        calculated = [
            dpeak["g"],
            dpeak["r"],
            dpeak["i"],
            dpeak["z"],
            colours["g-r"],
            colours["r-i"],
            colours["i-z"],
        ]

        registered = REGISTERED[float(z)]

        differences = [
            calculated[i] - registered[i]
            for i in range(len(calculated))
        ]

        max_abs_difference = max(
            abs(v)
            for v in differences
        )

        passed = max_abs_difference <= TOLERANCE_MAG

        row = {
            "z": float(z),
            "dpeak_g": calculated[0],
            "dpeak_r": calculated[1],
            "dpeak_i": calculated[2],
            "dpeak_z": calculated[3],
            "dtop3_g-r": calculated[4],
            "dtop3_r-i": calculated[5],
            "dtop3_i-z": calculated[6],
            "max_abs_difference_from_registered": max_abs_difference,
            "pass": passed,
        }

        rows.append(row)

        print(
            f"  z={z:.3f}  "
            f"max |delta|={max_abs_difference:.6f} mag  "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # -----------------------------------------------------------------
    # Write reproduced prediction table
    # -----------------------------------------------------------------

    csv_path = OUTDIR / "reproduced_passband_predictions.csv"

    fieldnames = [
        "z",
        *REGISTERED_COLUMNS,
        "max_abs_difference_from_registered",
        "pass",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------------------
    # Phase-conditioned predictions
    # -----------------------------------------------------------------

    print("\n[5] Computing preregistered phase-conditioned grid...")

    phase_grid = [-10, 0, 10, 20, 30, 40]

    phase_rows = []

    for z in redshifts:
        for phase in phase_grid:

            sed_wave, sed_flux = hsiao[phase]

            delta_band = {}

            for band in "griz":

                fs = synthetic_flux_proxy(
                    sed_wave,
                    sed_flux,
                    sdss[band][0],
                    sdss[band][1],
                    float(z),
                )

                fd = synthetic_flux_proxy(
                    sed_wave,
                    sed_flux,
                    des[band][0],
                    des[band][1],
                    float(z),
                )

                delta_band[band] = (
                    magnitude_from_proxy(fd)
                    - magnitude_from_proxy(fs)
                )

            phase_rows.append(
                {
                    "z": float(z),
                    "phase_rest_days": phase,
                    "delta_g": delta_band["g"],
                    "delta_r": delta_band["r"],
                    "delta_i": delta_band["i"],
                    "delta_z": delta_band["z"],
                    "delta_g-r": (
                        delta_band["g"]
                        - delta_band["r"]
                    ),
                    "delta_r-i": (
                        delta_band["r"]
                        - delta_band["i"]
                    ),
                    "delta_i-z": (
                        delta_band["i"]
                        - delta_band["z"]
                    ),
                }
            )

    phase_path = OUTDIR / "phase_conditioned_predictions.csv"

    with phase_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(phase_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(phase_rows)

    # -----------------------------------------------------------------
    # Gate
    # -----------------------------------------------------------------

    all_pass = all(r["pass"] for r in rows)

    global_max_difference = max(
        r["max_abs_difference_from_registered"]
        for r in rows
    )

    md_path = OUTDIR / "passband_validation_report.md"

    lines = [
        "# Phase 3 Passband Prediction Validation",
        "",
        "This validation uses only frozen response curves and the "
        "Hsiao07 template.",
        "",
        "**No SDSS or DES real compact-feature distributions were read.**",
        "",
        "## Artifact validation",
        "",
    ]

    for key, path in paths.items():
        lines.append(
            f"- `{key}`: `{path}` — SHA-256 verified"
        )

    lines.extend(
        [
            "",
            "## Pivot wavelengths",
            "",
            "| Band | SDSS (A) | DES (A) |",
            "|---|---:|---:|",
        ]
    )

    for band in "griz":
        lines.append(
            f"| {band} | "
            f"{pivots[band]['sdss']:.2f} | "
            f"{pivots[band]['des']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Reproduction gate",
            "",
            f"- Tolerance: `{TOLERANCE_MAG:.6f}` mag",
            f"- Maximum absolute discrepancy: "
            f"`{global_max_difference:.8f}` mag",
            f"- Gate: **{'PASS' if all_pass else 'FAIL'}**",
            "",
            "The registered dense-template prediction table is "
            "reproduced independently from the frozen artifacts.",
            "",
            "No observed survey feature statistic or class label was used.",
            "",
        ]
    )

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)

    if all_pass:
        print("PASSBAND REPRODUCTION GATE: PASS")
    else:
        print("PASSBAND REPRODUCTION GATE: FAIL")

    print(
        f"Maximum absolute discrepancy: "
        f"{global_max_difference:.8f} mag"
    )

    print("=" * 78)

    print(f"\nDense table : {csv_path}")
    print(f"Phase table : {phase_path}")
    print(f"Report      : {md_path}")

    if not all_pass:
        print(
            "\nSTOP: do not inspect real cross-survey compact features. "
            "Resolve the synthetic-photometry discrepancy first."
        )
        return 1

    print(
        "\nNext permitted step: validate/freeze release-specific "
        "quality masks and then perform cadence-template injections."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
