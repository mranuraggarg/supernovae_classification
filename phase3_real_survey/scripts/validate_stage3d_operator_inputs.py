#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path("phase3_real_survey")

LOCK = ROOT / "preregistration" / "stage3c_execution_lock.json"

EXPECTED = {
    "sdss_simlib": {
        "path": "simlib/SDSS/SDSS_3year.SIMLIB",
        "sha256": "fad8558b04c2fc10e26d18f42d016ab52311078ea87d8cfb44f1c47e3fe7c6c9",
    },
    "sdss_fluxerr": {
        "path": "simlib/SDSS/SDSS_fluxErrModel.DAT",
        "sha256": "a69b5a69fbbcaf54e6da2f6687c2020b97973c392000103553eb59674b9e05e7",
    },
    "des_simlib": {
        "path": "simlib/DES/DES-SN5YR_DES.SIMLIB",
        "sha256": "f5575ecd5b50c526ff63b758c5f6b88a710a185040dfd3c6e5008ca7b67d92a0",
    },
    "des_fluxerr_sim": {
        "path": "simlib/DES/DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT",
        "sha256": "2b6d0898fd1992a72cfa2322a79272a9557e8fc4cb98fa3193bd8ee00d2d80d9",
    },
    "des_fluxerr_fake": {
        "path": "simlib/DES/DES-SN5YR_DES_FLUXERRMODEL_FAKE.DAT",
        "sha256": "1f1285b56d2d7edfe6ed4d293ee96d71acc05bbe03f6d52ccf8cba5d26538afd",
    },
    "des_dcr_chrom": {
        "path": "simlib/DES/DES-SN5YR_DES_DCR+CHROM.DAT",
        "sha256": "1044cd913bfe28fed0982cc9a3f01d491bb211238ac51023858695fb30cd53c8",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_simlib(path: Path):
    fields = set()
    libids = []
    n_s_rows = 0
    current_field = None

    field_re = re.compile(r"^\s*FIELD:\s*(\S+)")
    libid_re = re.compile(r"^\s*LIBID:\s*(\d+)")

    with path.open(errors="replace") as f:
        for line in f:
            m = libid_re.match(line)
            if m:
                libids.append(int(m.group(1)))

            m = field_re.match(line)
            if m:
                current_field = m.group(1)
                fields.add(current_field)

            if line.startswith("S:"):
                n_s_rows += 1

    return {
        "n_libids": len(libids),
        "unique_libids": len(set(libids)),
        "min_libid": min(libids) if libids else None,
        "max_libid": max(libids) if libids else None,
        "fields": sorted(fields),
        "n_observation_rows": n_s_rows,
    }


def parse_fluxerr_maps(path: Path):
    mapnames = []
    bands = set()
    fields = set()
    varnames = set()
    n_rows = 0

    with path.open(errors="replace") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("MAPNAME:"):
                mapnames.append(stripped.split(":", 1)[1].strip())

            if stripped.startswith("BAND:"):
                bands.add(stripped.split(":", 1)[1].strip().split()[0])

            if stripped.startswith("FIELD:"):
                fields.add(stripped.split(":", 1)[1].strip().split()[0])

            if "VARNAMES:" in stripped:
                rhs = stripped.split("VARNAMES:", 1)[1].strip()
                varnames.update(rhs.split())

            if stripped.startswith("ROW:"):
                n_rows += 1

    return {
        "mapnames": sorted(set(mapnames)),
        "bands": sorted(bands),
        "fields": sorted(fields),
        "varnames": sorted(varnames),
        "n_rows": n_rows,
    }


def main():
    print("=" * 78)
    print("PHASE 3 STAGE 3D")
    print("FROZEN FORWARD-OPERATOR INPUT VALIDATION")
    print("NO SIMULATION — NO REAL FEATURE OUTCOMES — NO CLASSIFIER")
    print("=" * 78)

    sndata = os.environ.get("SNDATA_ROOT")
    if not sndata:
        raise RuntimeError("SNDATA_ROOT is not defined.")

    sndata = Path(sndata)
    if not sndata.is_dir():
        raise RuntimeError(f"SNDATA_ROOT does not exist: {sndata}")

    print("\n[1] Reading frozen Stage 3C execution lock...")

    with LOCK.open() as f:
        lock = json.load(f)

    print("  status :", lock["status"])
    print(
        "  MC     :",
        lock["monte_carlo"]["realizations_per_condition"],
        "realizations per condition",
    )
    print(
        "  RNG    :",
        lock["monte_carlo"]["rng_algorithm"],
    )

    if lock["monte_carlo"]["realizations_per_condition"] != 500:
        raise RuntimeError("Frozen Monte Carlo count is not 500.")

    if lock["monte_carlo"]["rng_algorithm"] != "numpy.random.PCG64":
        raise RuntimeError("Frozen RNG algorithm mismatch.")

    print("\n[2] Verifying frozen Stage 3B/3C artifacts...")

    resolved = {}

    for name, item in EXPECTED.items():
        path = sndata / item["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing artifact: {path}")

        observed = sha256(path)
        status = "PASS" if observed == item["sha256"] else "FAIL"

        print(f"  {name}")
        print(f"    expected : {item['sha256']}")
        print(f"    observed : {observed}")
        print(f"    status   : {status}")

        if status != "PASS":
            raise RuntimeError(f"Hash mismatch for {name}")

        resolved[name] = path

    print("\n[3] Parsing SDSS SIMLIB structure...")

    sdss = parse_simlib(resolved["sdss_simlib"])

    for k, v in sdss.items():
        print(f"  {k}: {v}")

    if not {"82N", "82S"}.issubset(set(sdss["fields"])):
        raise RuntimeError("Expected SDSS strata 82N/82S not recovered.")

    if sdss["n_libids"] == 0 or sdss["n_observation_rows"] == 0:
        raise RuntimeError("SDSS SIMLIB parser recovered no usable content.")

    print("\n[4] Parsing DES SIMLIB structure...")

    des = parse_simlib(resolved["des_simlib"])

    for k, v in des.items():
        print(f"  {k}: {v}")

    if des["n_libids"] == 0 or des["n_observation_rows"] == 0:
        raise RuntimeError("DES SIMLIB parser recovered no usable content.")

    deep = {"X3", "C3"}
    shallow = {"S1", "S2", "C1", "C2", "X1", "X2", "E1", "E2"}

    found_fields = set(des["fields"])

    if not deep.intersection(found_fields):
        raise RuntimeError("No DES deep field recovered.")

    if not shallow.intersection(found_fields):
        raise RuntimeError("No DES shallow field recovered.")

    print("\n[5] Parsing SDSS nonlinear uncertainty model...")

    sdss_err = parse_fluxerr_maps(resolved["sdss_fluxerr"])

    for k, v in sdss_err.items():
        print(f"  {k}: {v}")

    if "FLUXERR_ADD" not in sdss_err["mapnames"]:
        raise RuntimeError("SDSS FLUXERR_ADD map missing.")

    if "FLUXERR_SCALE" not in sdss_err["mapnames"]:
        raise RuntimeError("SDSS FLUXERR_SCALE map missing.")

    if "LOGSNR" not in sdss_err["varnames"]:
        raise RuntimeError("SDSS LOGSNR dependence not recovered.")

    print("\n[6] Parsing DES simulation uncertainty model...")

    des_err = parse_fluxerr_maps(resolved["des_fluxerr_sim"])

    for k, v in des_err.items():
        print(f"  {k}: {v}")

    if "FLUXERR_SCALE" not in des_err["mapnames"]:
        raise RuntimeError("DES FLUXERR_SCALE map missing.")

    if "SBMAG" not in des_err["varnames"]:
        raise RuntimeError("DES SBMAG dependence not recovered.")

    if not {"SHALLOW", "DEEP"}.issubset(set(des_err["fields"])):
        raise RuntimeError("DES SHALLOW/DEEP error-model strata missing.")

    print("\n[7] Checking frozen host-SB grid...")

    expected_sb = [20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5]
    observed_sb = lock["host_surface_brightness"]["values"]

    print("  frozen :", observed_sb)

    if observed_sb != expected_sb:
        raise RuntimeError("Frozen host-SB grid mismatch.")

    print("\n" + "=" * 78)
    print("STAGE 3D INPUT VALIDATION: PASS")
    print("=" * 78)
    print()
    print("No synthetic light curves were generated.")
    print("No Stage-1/Stage-2 outcome was read.")
    print("No classifier was run.")


if __name__ == "__main__":
    main()
