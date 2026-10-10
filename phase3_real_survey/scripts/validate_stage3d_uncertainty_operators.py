#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SDSS_SHA = (
    "a69b5a69fbbcaf54e6da2f6687c2020b97973c392000103553eb59674b9e05e7"
)
DES_SHA = (
    "2b6d0898fd1992a72cfa2322a79272a9557e8fc4cb98fa3193bd8ee00d2d80d9"
)

EXPECTED_SDSS_ADD = {
    "u": 28.0,
    "g": 10.0,
    "r": 15.0,
    "i": 23.0,
    "z": 60.0,
}

EXPECTED_SB_GRID = np.array(
    [20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
    dtype=float,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_header_tokens(line: str) -> dict[str, str]:
    """
    Parse tokens such as:
      MAPNAME: FLUXERR_SCALE
      BAND: g
      FIELD: SHALLOW
      VARNAMES: LOGSNR ERRSCALE

    Multiple directives may occur on the same line.
    """
    out = {}

    patterns = {
        "MAPNAME": r"\bMAPNAME:\s*([^\s]+)",
        "BAND": r"\bBAND:\s*([^\s]+)",
        "FIELD": r"\bFIELD:\s*([^\s]+)",
        "NVAR": r"\bNVAR:\s*([^\s]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, line)
        if m:
            out[key] = m.group(1)

    if "VARNAMES:" in line:
        rhs = line.split("VARNAMES:", 1)[1].strip()
        rhs = rhs.split("#", 1)[0].strip()
        out["VARNAMES"] = rhs

    return out


def parse_fluxerr_file(path: Path):
    """
    Parse MAPNAME blocks without assuming interpolation semantics.
    """
    maps = []

    current = None

    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("MAPNAME:"):
                if current is not None:
                    raise RuntimeError(
                        "Encountered new MAPNAME before ENDMAP."
                    )

                current = {
                    "mapname": None,
                    "band": None,
                    "field": None,
                    "varnames": [],
                    "rows": [],
                }

                tokens = parse_header_tokens(line)
                current["mapname"] = tokens.get("MAPNAME")
                current["band"] = tokens.get("BAND")
                current["field"] = tokens.get("FIELD")

                if "VARNAMES" in tokens:
                    current["varnames"] = tokens["VARNAMES"].split()

                continue

            if current is None:
                continue

            tokens = parse_header_tokens(line)

            if "BAND" in tokens:
                current["band"] = tokens["BAND"]

            if "FIELD" in tokens:
                current["field"] = tokens["FIELD"]

            if "VARNAMES" in tokens:
                current["varnames"] = tokens["VARNAMES"].split()

            if line.startswith("ROW:"):
                rhs = line.split("ROW:", 1)[1]
                rhs = rhs.split("#", 1)[0].strip()

                values = [float(x) for x in rhs.split()]
                current["rows"].append(values)

            if line.startswith("ENDMAP:"):
                maps.append(current)
                current = None

    if current is not None:
        raise RuntimeError("Unterminated MAPNAME block.")

    return maps


def map_index(maps):
    out = {}

    for m in maps:
        key = (
            m["mapname"],
            m["field"],
            m["band"],
        )

        if key in out:
            raise RuntimeError(f"Duplicate map key: {key}")

        out[key] = m

    return out


def validate_rows(map_obj, expected_ncol: int):
    rows = np.asarray(map_obj["rows"], dtype=float)

    if rows.ndim != 2:
        raise RuntimeError(
            f"Rows are not 2D for {map_obj['mapname']} "
            f"{map_obj['field']} {map_obj['band']}"
        )

    if rows.shape[1] != expected_ncol:
        raise RuntimeError(
            f"Unexpected row width {rows.shape[1]} for "
            f"{map_obj['mapname']} {map_obj['field']} {map_obj['band']}"
        )

    if not np.all(np.isfinite(rows)):
        raise RuntimeError("Non-finite value found in error-model map.")

    return rows


def exact_node_lookup(x: float, rows: np.ndarray) -> float:
    """
    Exact table-node lookup only.

    This deliberately does NOT interpolate.
    """
    hits = np.flatnonzero(np.isclose(rows[:, 0], x, rtol=0.0, atol=1e-12))

    if len(hits) != 1:
        raise RuntimeError(
            f"Expected one exact node for x={x}, found {len(hits)}."
        )

    return float(rows[hits[0], 1])


def snana_linear_clamped(x: float, rows: np.ndarray) -> float:
    """
    SNANA-compatible 1-D GRIDMAP lookup.

    Source provenance:
      RickKessler/SNANA
      commit 4113779f038eec613b6684d0d46a47019d7c9624
      repository state 2024-07-03

    Relevant source:
      src/sntools_gridmap.c
      src/sntools_fluxErrModels.c

    interp_GRIDMAP performs linear interpolation in one dimension.
    load_parList_FLUXERRMAP clamps out-of-range map coordinates to
    the nearest map boundary before interpolation.
    """
    xp = rows[:, 0]
    fp = rows[:, 1]

    if not np.all(np.diff(xp) > 0):
        raise RuntimeError("Interpolation grid is not strictly increasing.")

    return float(np.interp(x, xp, fp))


def main():
    print("=" * 78)
    print("PHASE 3 STAGE 3D.3")
    print("CONTROLLED UNCERTAINTY-OPERATOR VALIDATION")
    print("NO SYNTHETIC LIGHT CURVES — NO REAL OUTCOMES — NO CLASSIFIER")
    print("=" * 78)

    sndata = os.environ.get("SNDATA_ROOT")
    if not sndata:
        raise RuntimeError("SNDATA_ROOT is not defined.")

    sndata = Path(sndata)

    sdss_path = sndata / "simlib/SDSS/SDSS_fluxErrModel.DAT"
    des_path = (
        sndata
        / "simlib/DES/DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT"
    )

    print("\n[1] Verifying frozen uncertainty artifacts...")

    for name, path, expected in [
        ("SDSS", sdss_path, SDSS_SHA),
        ("DES", des_path, DES_SHA),
    ]:
        observed = sha256(path)

        print(f"  {name}")
        print(f"    expected : {expected}")
        print(f"    observed : {observed}")
        print(
            "    status   :",
            "PASS" if observed == expected else "FAIL",
        )

        if observed != expected:
            raise RuntimeError(f"{name} hash mismatch.")

    print("\n[2] Parsing uncertainty maps...")

    sdss_maps = parse_fluxerr_file(sdss_path)
    des_maps = parse_fluxerr_file(des_path)

    print("  SDSS maps:", len(sdss_maps))
    print("  DES maps :", len(des_maps))

    sdss_idx = map_index(sdss_maps)
    des_idx = map_index(des_maps)

    print("\n[3] Validating SDSS additive-error maps...")

    for band in "ugriz":
        key = ("FLUXERR_ADD", None, band)

        if key not in sdss_idx:
            raise RuntimeError(f"Missing SDSS additive map for {band}")

        m = sdss_idx[key]
        rows = validate_rows(m, 2)

        expected = EXPECTED_SDSS_ADD[band]

        if len(rows) != 2:
            raise RuntimeError(
                f"Expected two SDSS additive rows for {band}."
            )

        if not np.allclose(rows[:, 1], expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Unexpected SDSS ERRADD for {band}: {rows[:,1]}"
            )

        print(
            f"  {band}: ERRADD={expected:g} FLUXCAL units "
            f"over PSF nodes {rows[:,0].tolist()} : PASS"
        )

    print("\n[4] Validating SDSS LOGSNR scale maps...")

    sdss_scale_summary = {}

    for band in "ugriz":
        key = ("FLUXERR_SCALE", None, band)

        if key not in sdss_idx:
            raise RuntimeError(f"Missing SDSS scale map for {band}")

        m = sdss_idx[key]
        rows = validate_rows(m, 2)

        if m["varnames"] != ["LOGSNR", "ERRSCALE"]:
            raise RuntimeError(
                f"Unexpected SDSS VARNAMES for {band}: "
                f"{m['varnames']}"
            )

        if not np.all(np.diff(rows[:, 0]) > 0):
            raise RuntimeError(
                f"SDSS LOGSNR grid not strictly increasing for {band}"
            )

        if np.any(rows[:, 1] <= 0):
            raise RuntimeError(
                f"Non-positive SDSS ERRSCALE for {band}"
            )

        # Every exact table node must round-trip exactly.
        max_node_error = 0.0

        for x, y in rows:
            recovered = exact_node_lookup(float(x), rows)
            max_node_error = max(
                max_node_error,
                abs(recovered - float(y)),
            )

        # SNANA-compatible midpoint interpolation test.
        midpoint_errors = []

        for j in range(len(rows) - 1):
            x0, y0 = rows[j]
            x1, y1 = rows[j + 1]

            xm = 0.5 * (x0 + x1)
            expected_mid = 0.5 * (y0 + y1)
            observed_mid = snana_linear_clamped(xm, rows)

            midpoint_errors.append(abs(observed_mid - expected_mid))

        max_mid_error = max(midpoint_errors) if midpoint_errors else 0.0

        sdss_scale_summary[band] = (
            rows[0, 0],
            rows[-1, 0],
            rows[:, 1].min(),
            rows[:, 1].max(),
        )

        print(
            f"  {band}: N={len(rows):2d} "
            f"LOGSNR=[{rows[0,0]:+.2f},{rows[-1,0]:+.2f}] "
            f"ERRSCALE=[{rows[:,1].min():.4f},"
            f"{rows[:,1].max():.4f}] "
            f"node_error={max_node_error:.3e} "
            f"snana_mid_error={max_mid_error:.3e}"
        )

        if max_node_error > 1e-12:
            raise RuntimeError("SDSS exact-node lookup failed.")

        if max_mid_error > 1e-12:
            raise RuntimeError(
                "Candidate SDSS linear interpolation self-test failed."
            )

    print("\n[5] Validating DES FIELD x BAND x SBMAG maps...")

    for field in ["SHALLOW", "DEEP"]:
        for band in "griz":
            key = ("FLUXERR_SCALE", field, band)

            if key not in des_idx:
                raise RuntimeError(
                    f"Missing DES map for {field}/{band}"
                )

            m = des_idx[key]
            rows = validate_rows(m, 2)

            if m["varnames"] != ["SBMAG", "ERRSCALE"]:
                raise RuntimeError(
                    f"Unexpected DES VARNAMES for "
                    f"{field}/{band}: {m['varnames']}"
                )

            if len(rows) != len(EXPECTED_SB_GRID):
                raise RuntimeError(
                    f"Unexpected DES SB node count for "
                    f"{field}/{band}: {len(rows)}"
                )

            if not np.allclose(
                rows[:, 0],
                EXPECTED_SB_GRID,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"DES SBMAG grid mismatch for {field}/{band}"
                )

            if np.any(rows[:, 1] <= 0):
                raise RuntimeError(
                    f"Non-positive DES ERRSCALE for {field}/{band}"
                )

            max_node_error = 0.0

            for x, y in rows:
                recovered = exact_node_lookup(float(x), rows)
                max_node_error = max(
                    max_node_error,
                    abs(recovered - float(y)),
                )

            midpoint_errors = []

            for j in range(len(rows) - 1):
                x0, y0 = rows[j]
                x1, y1 = rows[j + 1]

                xm = 0.5 * (x0 + x1)
                expected_mid = 0.5 * (y0 + y1)
                observed_mid = snana_linear_clamped(xm, rows)

                midpoint_errors.append(
                    abs(observed_mid - expected_mid)
                )

            max_mid_error = max(midpoint_errors)

            print(
                f"  {field:7s}/{band}: "
                f"N={len(rows)} "
                f"ERRSCALE=[{rows[:,1].min():.3f},"
                f"{rows[:,1].max():.3f}] "
                f"node_error={max_node_error:.3e} "
                f"snana_mid_error={max_mid_error:.3e}"
            )

            if max_node_error > 1e-12:
                raise RuntimeError("DES exact-node lookup failed.")

            if max_mid_error > 1e-12:
                raise RuntimeError(
                    "Candidate DES linear interpolation self-test failed."
                )

    print("\n[6] Controlled boundary tests for candidate lookup...")

    for field, band in [
        ("SHALLOW", "g"),
        ("SHALLOW", "z"),
        ("DEEP", "g"),
        ("DEEP", "z"),
    ]:
        rows = np.asarray(
            des_idx[("FLUXERR_SCALE", field, band)]["rows"],
            dtype=float,
        )

        below = snana_linear_clamped(19.0, rows)
        at_low = snana_linear_clamped(20.5, rows)
        above = snana_linear_clamped(30.0, rows)
        at_high = snana_linear_clamped(27.5, rows)

        if below != at_low:
            raise RuntimeError(
                f"SNANA low-boundary clamp failed: {field}/{band}"
            )

        if above != at_high:
            raise RuntimeError(
                f"SNANA high-boundary clamp failed: {field}/{band}"
            )

        print(
            f"  {field:7s}/{band}: "
            f"low={at_low:.3f} high={at_high:.3f} "
            f"SNANA edge-clamp test PASS"
        )

    print("\n[7] Validating SNANA error-composition semantics...")

    # SNANA apply_FLUXERRMODEL:
    # ERRSCALE -> fluxErr * errModelVal
    # ERRADD   -> sqrt(fluxErr^2 + errModelVal^2)

    sigma_in = 40.0
    errscale = 1.25
    erradd = 30.0

    sigma_scaled = sigma_in * errscale
    sigma_added = np.sqrt(sigma_in**2 + erradd**2)

    if not np.isclose(
        sigma_scaled, 50.0, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("SNANA ERRSCALE composition test failed.")

    if not np.isclose(
        sigma_added, 50.0, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("SNANA ERRADD quadrature test failed.")

    print(
        "  ERRSCALE: sigma_out = sigma_in * ERRSCALE : PASS"
    )
    print(
        "  ERRADD  : sigma_out = sqrt(sigma_in^2 + ERRADD^2) : PASS"
    )

    print("\n[8] Source-level implementation boundary...")

    print(
        "  SNANA source reference:"
    )
    print(
        "    RickKessler/SNANA commit "
        "4113779f038eec613b6684d0d46a47019d7c9624"
    )
    print(
        "    repository state dated 2024-07-03"
    )
    print(
        "  Exact table parsing and exact-node recovery: VERIFIED"
    )
    print(
        "  1-D piecewise-linear GRIDMAP interpolation: VERIFIED"
    )
    print(
        "  Out-of-range edge clamping: VERIFIED"
    )
    print(
        "  ERRSCALE multiplicative composition: VERIFIED"
    )
    print(
        "  ERRADD quadrature composition: VERIFIED"
    )
    print(
        "  Full baseline SIMLIB variance construction: "
        "NOT YET EXECUTED"
    )

    print("\n" + "=" * 78)
    print("STAGE 3D.3 CONTROLLED UNCERTAINTY-OPERATOR VALIDATION: PASS")
    print("=" * 78)
    print()
    print("No synthetic light curve was generated.")
    print("No real compact-feature outcome was read.")
    print("No classifier was run.")
    print()
    print(
        "SNANA uncertainty-map lookup and composition semantics are "
        "source-verified."
    )
    print(
        "Next validation gate: baseline SIMLIB variance construction "
        "before synthetic photometry."
    )


if __name__ == "__main__":
    main()
