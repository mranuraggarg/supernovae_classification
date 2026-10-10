#!/usr/bin/env python3

"""
Stage 3D.7 native-SNANA host-photon oracle validation.

Reads the epoch-level diagnostic rows emitted directly from the pinned
SNANA gen_fluxNoise_calc() implementation and independently recomputes:

    PSFSIG_ASEC = PIXSIZE * sqrt(NEA / (4*pi))

and

    HOSTVAR_PE2 = GAIN * 10**(0.4 * (ZPT - GALMAG_NEA))

This validates the exact epoch-level host-photon semantics extracted
from native SNANA before those semantics are wired into Stage 3D.6.

No Stage-1/Stage-2 outputs are read.
No classifier is run.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


ROOT = Path("phase3_real_survey")

ORACLE = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d7_production_semantics"
    / "native_snana_oracle"
    / "des5yr_ia_host_oracle_hostphot_diag.txt"
)

OUTDIR = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d7_production_semantics"
)

# Native DES SIMLIB header printed by SNANA:
#   SIMLIB pixel size: 0.270 asec
PIXSIZE = 0.270

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


def parse_oracle(path: Path):
    rows = []

    for raw in path.read_text().splitlines():
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
                "galmag_nea": float(m.group("galmag")),
                "zpt": float(m.group("zpt")),
                "gain": float(m.group("gain")),
                "hostvar_native": float(m.group("hostvar")),
            }
        )

    if not rows:
        raise RuntimeError(
            f"No STAGE3D7_HOSTPHOT rows found in {path}"
        )

    return rows


def main():
    rows = parse_oracle(ORACLE)

    if len(rows) != 102:
        raise RuntimeError(
            f"Expected 102 native epochs; found {len(rows)}"
        )

    psf_abs = []
    psf_rel = []
    host_abs = []
    host_rel = []

    output_rows = []

    for row in rows:
        psfsig_calc = PIXSIZE * math.sqrt(
            row["nea"] / (4.0 * math.pi)
        )

        hostvar_calc = (
            row["gain"]
            * 10.0
            ** (
                0.4
                * (
                    row["zpt"]
                    - row["galmag_nea"]
                )
            )
        )

        d_psf = psfsig_calc - row["psfsig_native"]
        d_host = hostvar_calc - row["hostvar_native"]

        r_psf = (
            abs(d_psf) / abs(row["psfsig_native"])
            if row["psfsig_native"] != 0.0
            else 0.0
        )

        r_host = (
            abs(d_host) / abs(row["hostvar_native"])
            if row["hostvar_native"] != 0.0
            else 0.0
        )

        psf_abs.append(abs(d_psf))
        psf_rel.append(r_psf)
        host_abs.append(abs(d_host))
        host_rel.append(r_host)

        output_rows.append(
            {
                **row,
                "psfsig_calc": psfsig_calc,
                "psfsig_absdiff": abs(d_psf),
                "psfsig_reldiff": r_psf,
                "hostvar_calc": hostvar_calc,
                "hostvar_absdiff": abs(d_host),
                "hostvar_reldiff": r_host,
            }
        )

    psf_abs = np.asarray(psf_abs)
    psf_rel = np.asarray(psf_rel)
    host_abs = np.asarray(host_abs)
    host_rel = np.asarray(host_rel)

    print("===== STAGE 3D.7 NATIVE HOST-PHOTON ORACLE =====")
    print(f"epochs: {len(rows)}")
    print(f"PIXSIZE: {PIXSIZE:.6f} arcsec/pixel")
    print()

    print("===== PSFSIG_ASEC REPRODUCTION =====")
    print(f"max absolute difference : {psf_abs.max():.12e}")
    print(f"max relative difference : {psf_rel.max():.12e}")
    print(f"mean relative difference: {psf_rel.mean():.12e}")
    print()

    print("===== HOSTVAR_PE2 REPRODUCTION =====")
    print(f"max absolute difference : {host_abs.max():.12e}")
    print(f"max relative difference : {host_rel.max():.12e}")
    print(f"mean relative difference: {host_rel.mean():.12e}")
    print()

    # Native values were printed with finite decimal precision,
    # therefore use tight but not machine-epsilon tolerances.
    psf_pass = bool(psf_rel.max() < 1.0e-10)
    host_pass = bool(host_rel.max() < 1.0e-9)

    print(
        "PSFSIG OPERATOR: "
        + ("PASS" if psf_pass else "FAIL")
    )

    print(
        "HOSTVAR OPERATOR: "
        + ("PASS" if host_pass else "FAIL")
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        OUTDIR
        / "stage3d7_native_hostphot_epoch_comparison.csv"
    )

    header = [
        "epoch",
        "band",
        "mjd",
        "nea",
        "psfsig_native",
        "psfsig_calc",
        "psfsig_absdiff",
        "psfsig_reldiff",
        "galmag_nea",
        "zpt",
        "gain",
        "hostvar_native",
        "hostvar_calc",
        "hostvar_absdiff",
        "hostvar_reldiff",
    ]

    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")

        for row in output_rows:
            f.write(
                ",".join(
                    str(row[k])
                    for k in header
                )
                + "\n"
            )

    summary_path = (
        OUTDIR
        / "stage3d7_native_hostphot_validation.txt"
    )

    summary_path.write_text(
        "\n".join(
            [
                "Stage 3D.7 native SNANA host-photon validation",
                f"epochs={len(rows)}",
                f"pixsize={PIXSIZE}",
                f"psfsig_max_abs={psf_abs.max():.16e}",
                f"psfsig_max_rel={psf_rel.max():.16e}",
                f"hostvar_max_abs={host_abs.max():.16e}",
                f"hostvar_max_rel={host_rel.max():.16e}",
                f"psfsig_pass={psf_pass}",
                f"hostvar_pass={host_pass}",
                "",
            ]
        )
    )

    print()
    print(f"epoch table: {csv_path}")
    print(f"summary    : {summary_path}")

    if not (psf_pass and host_pass):
        raise SystemExit(1)

    print()
    print("STAGE3D7 NATIVE HOST-PHOTON VALIDATION: PASS")


if __name__ == "__main__":
    main()
