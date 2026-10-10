#!/usr/bin/env python3

"""
Stage 3D.7 native-SNANA GALMAG interpolation validation.

Validates the middle link of the host-photon chain:

    PSFSIG_ASEC
        -> interp_GALMAG_HOSTLIB()
        -> GALMAG_NEA

against epoch-level GALMAG_NEA values emitted directly from the
pinned native SNANA gen_fluxNoise_calc().

SNANA source semantics reproduced here:

    NPSF = NMAGPSF_HOSTLIB
    PSFSIGmin = Aperture_PSFSIG[1]
    PSFSIGmax = Aperture_PSFSIG[NPSF]

    if PSFSIG < min:
        PSFSIG_local = min + 0.0001
    if PSFSIG > max:
        PSFSIG_local = max - 0.0001

followed by 1-D linear interpolation of GALMAG versus PSFSIG.

No Stage-1/Stage-2 outputs are read.
No classifier is run.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


ROOT = Path("phase3_real_survey")

ORACLE_DIR = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d7_production_semantics"
    / "native_snana_oracle"
)

HOST_BLOCK = (
    ORACLE_DIR
    / "des5yr_ia_host_oracle_accept_all_snhost_block.txt"
)

HOSTPHOT = (
    ORACLE_DIR
    / "des5yr_ia_host_oracle_hostphot_diag.txt"
)

OUTDIR = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d7_production_semantics"
)

BANDS = "griz"

# Exact fixed SNANA host-aperture PSF grid from sntools_host.c.
# The final value is normally PSFMAX_SNANA/2.35; for this validation
# we prefer the values printed in DUMP_SNHOST when available.
SOURCE_PSF_GRID = np.asarray(
    [
        0.03 / 2.35,
        0.07 / 2.35,
        0.10 / 2.35,
        0.20 / 2.35,
        0.40 / 2.35,
        0.80 / 2.35,
        1.30 / 2.35,
        2.10 / 2.35,
        5.00 / 2.35,
    ],
    dtype=float,
)

HOSTPHOT_PATTERN = re.compile(
    r"^STAGE3D7_HOSTPHOT "
    r"epoch=(?P<epoch>\d+) "
    r"band=(?P<band>[A-Za-z]) "
    r"MJD=(?P<mjd>[-+0-9.eE]+) "
    r"NEA=(?P<nea>[-+0-9.eE]+) "
    r"PSFSIG_ASEC=(?P<psfsig>[-+0-9.eE]+) "
    r"GALMAG_NEA=(?P<galmag>[-+0-9.eE]+) "
    r"ZPT=(?P<zpt>[-+0-9.eE]+) "
    r"GAIN=(?P<gain>[-+0-9.eE]+) "
    r"HOSTVAR_PE2=(?P<hostvar>[-+0-9.eE]+)$"
)


def numbers(line: str):
    return [
        float(x)
        for x in re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
            line,
        )
    ]


def parse_hostphot(path: Path):
    rows = []

    for raw in path.read_text().splitlines():
        m = HOSTPHOT_PATTERN.match(raw.strip())
        if m is None:
            continue

        rows.append(
            {
                "epoch": int(m.group("epoch")),
                "band": m.group("band"),
                "mjd": float(m.group("mjd")),
                "psfsig": float(m.group("psfsig")),
                "galmag_native": float(m.group("galmag")),
            }
        )

    if len(rows) != 102:
        raise RuntimeError(
            f"Expected 102 HOSTPHOT rows; found {len(rows)}"
        )

    return rows


def candidate_grid_lines(text: str):
    """
    Return diagnostic candidate lines to make parser failures easy
    to diagnose without silently guessing.
    """
    out = []

    for line in text.splitlines():
        u = line.upper()

        if (
            "PSF" in u
            or "GALMAG" in u
            or "GALFRAC" in u
            or re.search(r"(^|\s)[GRIZ](\s|:|=)", line, re.I)
        ):
            out.append(line)

    return out


def parse_native_grid(path: Path):
    """
    Parse the PSFSIG and per-band GALMAG arrays printed in DUMP_SNHOST.

    The pinned SNANA diagnostic print formats have varied over time,
    therefore detection is intentionally label-based rather than tied
    to one exact whitespace layout.
    """

    text = path.read_text()
    lines = text.splitlines()

    psf_grid = None
    band_grids = {}

    # ---------- PSF grid ----------
    for line in lines:
        u = line.upper()

        if "PSF" not in u:
            continue

        vals = numbers(line)

        # Ignore line-number / index-like candidates and require the
        # nine aperture values.
        if len(vals) >= 9:
            cand = np.asarray(vals[-9:], dtype=float)

            if (
                np.all(np.isfinite(cand))
                and np.all(np.diff(cand) > 0.0)
                and cand[0] < 0.1
                and cand[-1] > 1.0
            ):
                psf_grid = cand
                break

    # If parsing misses the PSF line, use the exact source-defined grid.
    # This is not an inferred grid: these constants were source-validated.
    if psf_grid is None:
        psf_grid = SOURCE_PSF_GRID.copy()

    # ---------- per-band GALMAG ----------
    #
    # Accept lines containing a band token plus nine numeric magnitudes.
    # Magnitudes for this host should be in a physically plausible range.
    for band in BANDS:
        patterns = [
            re.compile(
                rf"(^|[^A-Za-z0-9]){band}([^A-Za-z0-9]|$)",
                re.I,
            ),
            re.compile(
                rf"{band}_obs",
                re.I,
            ),
        ]

        candidates = []

        for line in lines:
            if not any(p.search(line) for p in patterns):
                continue

            vals = numbers(line)

            if len(vals) < 9:
                continue

            # Use the final nine numbers because some diagnostics prepend
            # an index or other scalar.
            cand = np.asarray(vals[-9:], dtype=float)

            if (
                np.all(np.isfinite(cand))
                and np.all((cand > 5.0) & (cand < 60.0))
            ):
                candidates.append((line, cand))

        if candidates:
            # Prefer a line explicitly mentioning MAG/GALMAG/obs.
            candidates.sort(
                key=lambda x: (
                    "MAG" not in x[0].upper()
                    and "_OBS" not in x[0].upper(),
                    len(x[0]),
                )
            )

            band_grids[band] = candidates[0][1]

    if len(band_grids) != 4:
        print("===== CANDIDATE HOST-BLOCK LINES =====")
        for line in candidate_grid_lines(text):
            print(line)

        missing = [
            b for b in BANDS
            if b not in band_grids
        ]

        raise RuntimeError(
            "Could not identify GALMAG grids for bands: "
            + ",".join(missing)
        )

    return psf_grid, band_grids


def snana_interp(psf: float, grid_x, grid_y):
    """
    Reproduce interp_GALMAG_HOSTLIB endpoint handling followed
    by ordinary linear interpolation.
    """

    xmin = float(grid_x[0])
    xmax = float(grid_x[-1])

    x = float(psf)

    if x < xmin:
        x = xmin + 0.0001

    if x > xmax:
        x = xmax - 0.0001

    # interp_1DFUN mode used by interp_GALMAG_HOSTLIB is linear
    # interpolation on the precomputed PSF/GALMAG grid.
    return float(
        np.interp(
            x,
            grid_x,
            grid_y,
        )
    )


def main():
    rows = parse_hostphot(HOSTPHOT)
    psf_grid, galmag_grid = parse_native_grid(HOST_BLOCK)

    print("===== NATIVE PSF GRID =====")
    for i, x in enumerate(psf_grid, start=1):
        print(f"{i:2d}  {x:.12g}")

    print()

    for band in BANDS:
        print(f"===== NATIVE {band}-BAND GALMAG GRID =====")
        for x, mag in zip(
            psf_grid,
            galmag_grid[band],
            strict=True,
        ):
            print(f"{x:.12g}  {mag:.12g}")
        print()

    comparison = []

    for row in rows:
        calc = snana_interp(
            row["psfsig"],
            psf_grid,
            galmag_grid[row["band"]],
        )

        native = row["galmag_native"]
        diff = calc - native

        comparison.append(
            {
                **row,
                "galmag_interp": calc,
                "absdiff": abs(diff),
                "reldiff": (
                    abs(diff) / abs(native)
                    if native != 0.0
                    else 0.0
                ),
            }
        )

    absdiff = np.asarray(
        [r["absdiff"] for r in comparison]
    )

    reldiff = np.asarray(
        [r["reldiff"] for r in comparison]
    )

    print("===== GALMAG_NEA INTERPOLATION VALIDATION =====")
    print(f"epochs                  : {len(comparison)}")
    print(f"max absolute difference : {absdiff.max():.12e}")
    print(f"mean absolute difference: {absdiff.mean():.12e}")
    print(f"max relative difference : {reldiff.max():.12e}")
    print(f"mean relative difference: {reldiff.mean():.12e}")

    worst = int(np.argmax(absdiff))
    w = comparison[worst]

    print()
    print("===== WORST EPOCH =====")
    print(f"epoch        : {w['epoch']}")
    print(f"band         : {w['band']}")
    print(f"MJD          : {w['mjd']}")
    print(f"PSFSIG       : {w['psfsig']:.12g}")
    print(f"native GALMAG: {w['galmag_native']:.12g}")
    print(f"interp GALMAG: {w['galmag_interp']:.12g}")
    print(f"absolute diff: {w['absdiff']:.12e}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        OUTDIR
        / "stage3d7_native_galmag_interpolation_comparison.csv"
    )

    fields = [
        "epoch",
        "band",
        "mjd",
        "psfsig",
        "galmag_native",
        "galmag_interp",
        "absdiff",
        "reldiff",
    ]

    with csv_path.open("w") as f:
        f.write(",".join(fields) + "\n")

        for row in comparison:
            f.write(
                ",".join(
                    str(row[k])
                    for k in fields
                )
                + "\n"
            )

    grid_path = (
        OUTDIR
        / "stage3d7_native_galmag_grid.txt"
    )

    with grid_path.open("w") as f:
        f.write(
            "index psfsig g r i z\n"
        )

        for i, x in enumerate(psf_grid):
            f.write(
                f"{i+1} "
                f"{x:.16e} "
                f"{galmag_grid['g'][i]:.16e} "
                f"{galmag_grid['r'][i]:.16e} "
                f"{galmag_grid['i'][i]:.16e} "
                f"{galmag_grid['z'][i]:.16e}\n"
            )

    # DUMP_SNHOST GALMAG values are printed at finite precision,
    # so interpolation cannot be expected to reproduce the hidden
    # full-precision internal GALMAG array to machine precision.
    #
    # Start with a deliberately strict 5e-4 mag gate. If it fails,
    # inspect the discrepancy before relaxing anything.
    passed = bool(absdiff.max() < 5.0e-4)

    summary_path = (
        OUTDIR
        / "stage3d7_native_galmag_interpolation_validation.txt"
    )

    summary_path.write_text(
        "\n".join(
            [
                "Stage 3D.7 native GALMAG interpolation validation",
                f"epochs={len(comparison)}",
                f"max_abs_mag={absdiff.max():.16e}",
                f"mean_abs_mag={absdiff.mean():.16e}",
                f"max_rel={reldiff.max():.16e}",
                f"pass={passed}",
                "",
            ]
        )
    )

    print()
    print(f"grid       : {grid_path}")
    print(f"comparison : {csv_path}")
    print(f"summary    : {summary_path}")

    if not passed:
        print()
        print("GALMAG INTERPOLATION: FAIL")
        raise SystemExit(1)

    print()
    print("GALMAG INTERPOLATION: PASS")
    print()
    print(
        "STAGE3D7 NATIVE GALMAG INTERPOLATION VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
