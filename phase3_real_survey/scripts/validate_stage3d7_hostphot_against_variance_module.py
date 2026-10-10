#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np


ROOT = Path("phase3_real_survey")

VARIANCE_PATH = (
    ROOT
    / "scripts"
    / "validate_stage3d_baseline_variance.py"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d7_production_semantics"
)

ORACLE_DIR = (
    RESULT_ROOT
    / "native_snana_oracle"
)

HOSTPHOT = (
    ORACLE_DIR
    / "des5yr_ia_host_oracle_hostphot_diag.txt"
)

GRID = (
    RESULT_ROOT
    / "stage3d7_native_galmag_grid.txt"
)

OUT = (
    RESULT_ROOT
    / "stage3d7_variance_module_regression.txt"
)

CSV = (
    RESULT_ROOT
    / "stage3d7_variance_module_regression.csv"
)

PIXSIZE = 0.270


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


variance = load_module(
    "stage3d_variance",
    VARIANCE_PATH,
)


PATTERN = re.compile(
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


def parse_oracle():
    rows = []

    for raw in HOSTPHOT.read_text().splitlines():
        m = PATTERN.match(raw.strip())

        if m is None:
            continue

        rows.append(
            {
                "epoch": int(m.group("epoch")),
                "band": m.group("band"),
                "mjd": float(m.group("mjd")),
                "nea": float(m.group("nea")),
                "psfsig_native": float(m.group("psfsig")),
                "galmag_native": float(m.group("galmag")),
                "zpt": float(m.group("zpt")),
                "gain": float(m.group("gain")),
                "hostvar_native": float(m.group("hostvar")),
            }
        )

    if len(rows) != 102:
        raise RuntimeError(
            f"Expected 102 native epochs; found {len(rows)}"
        )

    return rows


def parse_grid():
    data = np.genfromtxt(
        GRID,
        names=True,
        dtype=None,
        encoding=None,
    )

    psf_grid = np.asarray(
        data["psfsig"],
        dtype=float,
    )

    galmag_grid = {
        band: np.asarray(
            data[band],
            dtype=float,
        )
        for band in "griz"
    }

    return psf_grid, galmag_grid


def main():
    rows = parse_oracle()
    psf_grid, galmag_grid = parse_grid()

    comparisons = []

    for row in rows:
        psf_calc = (
            variance.psfsig_arcsec_from_nea(
                row["nea"],
                PIXSIZE,
            )
        )

        galmag_calc = (
            variance.interp_galmag_hostlib(
                psf_calc,
                psf_grid,
                galmag_grid[row["band"]],
            )
        )

        hostvar_calc = (
            variance.host_variance_pe2_from_galmag(
                galmag_nea=galmag_calc,
                zpt=row["zpt"],
                gain=row["gain"],
            )
        )

        psf_abs = abs(
            psf_calc
            - row["psfsig_native"]
        )

        galmag_abs = abs(
            galmag_calc
            - row["galmag_native"]
        )

        hostvar_rel = (
            abs(
                hostvar_calc
                - row["hostvar_native"]
            )
            / abs(row["hostvar_native"])
        )

        comparisons.append(
            {
                **row,
                "psfsig_calc": psf_calc,
                "psfsig_absdiff": psf_abs,
                "galmag_calc": galmag_calc,
                "galmag_absdiff": galmag_abs,
                "hostvar_calc": hostvar_calc,
                "hostvar_reldiff": hostvar_rel,
            }
        )

    max_psf = max(
        r["psfsig_absdiff"]
        for r in comparisons
    )

    max_galmag = max(
        r["galmag_absdiff"]
        for r in comparisons
    )

    max_hostvar_rel = max(
        r["hostvar_reldiff"]
        for r in comparisons
    )

    print(
        "===== STAGE 3D.7 VARIANCE MODULE REGRESSION ====="
    )
    print(f"epochs                 : {len(rows)}")
    print(
        f"max PSFSIG abs diff    : {max_psf:.12e}"
    )
    print(
        f"max GALMAG abs diff    : {max_galmag:.12e}"
    )
    print(
        f"max HOSTVAR rel diff   : {max_hostvar_rel:.12e}"
    )

    # PSF and HOSTVAR diagnostics are printed at high precision.
    # GALMAG grid came from DUMP_SNHOST at limited decimal precision.
    psf_pass = max_psf < 1.0e-10
    galmag_pass = max_galmag < 5.0e-4
    hostvar_pass = max_hostvar_rel < 5.0e-4

    print()
    print(
        "PSFSIG MODULE OPERATOR: "
        + ("PASS" if psf_pass else "FAIL")
    )
    print(
        "GALMAG MODULE OPERATOR: "
        + ("PASS" if galmag_pass else "FAIL")
    )
    print(
        "HOSTVAR MODULE OPERATOR: "
        + ("PASS" if hostvar_pass else "FAIL")
    )

    fields = [
        "epoch",
        "band",
        "mjd",
        "nea",
        "psfsig_native",
        "psfsig_calc",
        "psfsig_absdiff",
        "galmag_native",
        "galmag_calc",
        "galmag_absdiff",
        "zpt",
        "gain",
        "hostvar_native",
        "hostvar_calc",
        "hostvar_reldiff",
    ]

    with CSV.open("w") as f:
        f.write(",".join(fields) + "\n")

        for row in comparisons:
            f.write(
                ",".join(
                    str(row[k])
                    for k in fields
                )
                + "\n"
            )

    passed = (
        psf_pass
        and galmag_pass
        and hostvar_pass
    )

    OUT.write_text(
        "\n".join(
            [
                "Stage 3D.7 variance-module regression",
                f"epochs={len(rows)}",
                f"max_psfsig_abs={max_psf:.16e}",
                f"max_galmag_abs={max_galmag:.16e}",
                f"max_hostvar_rel={max_hostvar_rel:.16e}",
                f"psfsig_pass={psf_pass}",
                f"galmag_pass={galmag_pass}",
                f"hostvar_pass={hostvar_pass}",
                f"pass={passed}",
                "",
            ]
        )
    )

    print()
    print(f"comparison: {CSV}")
    print(f"summary   : {OUT}")

    if not passed:
        print()
        print(
            "STAGE3D7 VARIANCE MODULE REGRESSION: FAIL"
        )
        raise SystemExit(1)

    print()
    print(
        "STAGE3D7 VARIANCE MODULE REGRESSION: PASS"
    )


if __name__ == "__main__":
    main()
