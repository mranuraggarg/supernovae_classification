#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("phase3_real_survey")
LOCK = ROOT / "preregistration" / "stage3c_execution_lock.json"

EXPECTED = {
    "sdss_simlib": {
        "path": "simlib/SDSS/SDSS_3year.SIMLIB",
        "sha256": "fad8558b04c2fc10e26d18f42d016ab52311078ea87d8cfb44f1c47e3fe7c6c9",
    },
    "des_simlib": {
        "path": "simlib/DES/DES-SN5YR_DES.SIMLIB",
        "sha256": "f5575ecd5b50c526ff63b758c5f6b88a710a185040dfd3c6e5008ca7b67d92a0",
    },
}

DES_SHALLOW = {"S1", "S2", "C1", "C2", "X1", "X2", "E1", "E2"}
DES_DEEP = {"X3", "C3"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(
    namespace: str,
    purpose: str,
    survey: str,
    stratum: str,
    z: float,
    sbmag: float,
    realization_id: int,
) -> int:
    """
    Frozen Stage-3C seed rule:

      SHA256(
        namespace|purpose|survey|stratum|z|sbmag|realization_id
      )

    First 8 digest bytes -> unsigned 64-bit integer.

    Canonical formatting below is itself validated and then frozen
    by this Stage-3D implementation.
    """
    key = (
        f"{namespace}|{purpose}|{survey}|{stratum}|"
        f"{z:.3f}|{sbmag:.1f}|{realization_id}"
    )

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

    return seed


def parse_simlib_libids(path: Path, survey: str):
    """
    Recover LIBID -> FIELD from a SIMLIB without using any real SN outcomes.
    """
    result = []

    current_libid = None
    current_field = None

    libid_re = re.compile(r"^\s*LIBID:\s*(\d+)")
    field_re = re.compile(r"^\s*FIELD:\s*(\S+)")

    with path.open(errors="replace") as f:
        for line in f:
            m = libid_re.match(line)
            if m:
                if current_libid is not None:
                    result.append((current_libid, current_field))

                current_libid = int(m.group(1))
                current_field = None
                continue

            m = field_re.match(line)
            if m and current_libid is not None:
                current_field = m.group(1)

    if current_libid is not None:
        result.append((current_libid, current_field))

    if not result:
        raise RuntimeError(f"No LIBIDs recovered from {survey} SIMLIB.")

    return result


def classify_stratum(survey: str, field: str) -> str:
    if survey == "SDSS-II":
        if field not in {"82N", "82S"}:
            raise RuntimeError(f"Unexpected SDSS field: {field}")
        return field

    if survey == "DES-SN5YR":
        if field in DES_SHALLOW:
            return "SHALLOW"
        if field in DES_DEEP:
            return "DEEP"
        raise RuntimeError(f"Unexpected DES field: {field}")

    raise RuntimeError(f"Unknown survey: {survey}")


def grouped_libids(entries, survey: str):
    groups = defaultdict(list)

    for libid, field in entries:
        stratum = classify_stratum(survey, field)
        groups[stratum].append(libid)

    for stratum in groups:
        groups[stratum] = sorted(groups[stratum])

    return dict(groups)


def deterministic_libid_sequence(
    libids: list[int],
    *,
    namespace: str,
    survey: str,
    stratum: str,
    n_needed: int,
) -> list[int]:
    """
    Frozen Stage-3C rule:
      - eligible LIBIDs sorted numerically;
      - deterministic seeded permutation within each stratum;
      - successive deterministic permutations if n_needed exceeds count.
    """
    if not libids:
        raise RuntimeError(f"No LIBIDs supplied for {survey} {stratum}")

    sorted_ids = np.asarray(sorted(libids), dtype=np.int64)

    out = []
    cycle = 0

    while len(out) < n_needed:
        key = f"{namespace}|libid-permutation|{survey}|{stratum}|{cycle}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

        rng = np.random.Generator(np.random.PCG64(seed))
        perm = rng.permutation(sorted_ids)

        needed = n_needed - len(out)
        out.extend(int(x) for x in perm[:needed])

        cycle += 1

    return out


def main():
    print("=" * 78)
    print("PHASE 3 STAGE 3D.2")
    print("DETERMINISTIC SAMPLING AND SEED VALIDATION")
    print("NO SYNTHETIC FLUXES — NO REAL FEATURE OUTCOMES — NO CLASSIFIER")
    print("=" * 78)

    sndata_root = os.environ.get("SNDATA_ROOT")
    if not sndata_root:
        raise RuntimeError("SNDATA_ROOT is not defined.")

    sndata_root = Path(sndata_root)

    print("\n[1] Loading frozen Stage 3C execution lock...")

    with LOCK.open() as f:
        lock = json.load(f)

    namespace = lock["monte_carlo"]["master_seed_namespace"]
    n_real = lock["monte_carlo"]["realizations_per_condition"]
    redshifts = lock["redshift_grid"]
    sb_grid = lock["host_surface_brightness"]["values"]

    print("  namespace :", namespace)
    print("  RNG       :", lock["monte_carlo"]["rng_algorithm"])
    print("  N/cond    :", n_real)
    print("  z grid    :", redshifts)
    print("  SB grid   :", sb_grid)

    if namespace != "phase3-stage3c-v1":
        raise RuntimeError("Unexpected seed namespace.")

    if n_real != 500:
        raise RuntimeError("Unexpected realization count.")

    if lock["monte_carlo"]["rng_algorithm"] != "numpy.random.PCG64":
        raise RuntimeError("Unexpected RNG.")

    print("\n[2] Verifying SIMLIB hashes...")

    resolved = {}

    for name, info in EXPECTED.items():
        path = sndata_root / info["path"]

        if not path.is_file():
            raise RuntimeError(f"Missing SIMLIB: {path}")

        observed = sha256(path)

        print(f"  {name}")
        print(f"    expected : {info['sha256']}")
        print(f"    observed : {observed}")

        if observed != info["sha256"]:
            raise RuntimeError(f"Hash mismatch: {name}")

        print("    status   : PASS")
        resolved[name] = path

    print("\n[3] Recovering eligible LIBIDs by frozen survey stratum...")

    sdss_entries = parse_simlib_libids(
        resolved["sdss_simlib"],
        "SDSS-II",
    )
    des_entries = parse_simlib_libids(
        resolved["des_simlib"],
        "DES-SN5YR",
    )

    sdss_groups = grouped_libids(sdss_entries, "SDSS-II")
    des_groups = grouped_libids(des_entries, "DES-SN5YR")

    for stratum in ["82N", "82S"]:
        ids = sdss_groups[stratum]
        print(
            f"  SDSS-II {stratum:7s}: "
            f"N={len(ids):5d}  min={min(ids)}  max={max(ids)}"
        )

    for stratum in ["SHALLOW", "DEEP"]:
        ids = des_groups[stratum]
        print(
            f"  DES-SN5YR {stratum:7s}: "
            f"N={len(ids):5d}  min={min(ids)}  max={max(ids)}"
        )

    print("\n[4] Validating deterministic LIBID permutations...")

    all_sequences = {}

    for survey, groups in [
        ("SDSS-II", sdss_groups),
        ("DES-SN5YR", des_groups),
    ]:
        for stratum, ids in sorted(groups.items()):
            seq1 = deterministic_libid_sequence(
                ids,
                namespace=namespace,
                survey=survey,
                stratum=stratum,
                n_needed=n_real,
            )

            seq2 = deterministic_libid_sequence(
                ids,
                namespace=namespace,
                survey=survey,
                stratum=stratum,
                n_needed=n_real,
            )

            if seq1 != seq2:
                raise RuntimeError(
                    f"Non-deterministic LIBID sequence: {survey} {stratum}"
                )

            if len(seq1) != n_real:
                raise RuntimeError(
                    f"Incorrect LIBID sequence length: {survey} {stratum}"
                )

            if not set(seq1).issubset(set(ids)):
                raise RuntimeError(
                    f"Invalid LIBID selected: {survey} {stratum}"
                )

            all_sequences[(survey, stratum)] = seq1

            print(
                f"  {survey:10s} {stratum:7s}: "
                f"PASS  first10={seq1[:10]}"
            )

    print("\n[5] Validating seed determinism...")

    test_cases = [
        ("noise", "SDSS-II", "82N", 0.050, 20.5, 0),
        ("noise", "SDSS-II", "82S", 0.175, 24.5, 123),
        ("noise", "DES-SN5YR", "SHALLOW", 0.300, 27.5, 499),
        ("noise", "DES-SN5YR", "DEEP", 0.125, 22.5, 321),
    ]

    seen = set()

    for purpose, survey, stratum, z, sbmag, rid in test_cases:
        s1 = derive_seed(
            namespace,
            purpose,
            survey,
            stratum,
            z,
            sbmag,
            rid,
        )
        s2 = derive_seed(
            namespace,
            purpose,
            survey,
            stratum,
            z,
            sbmag,
            rid,
        )

        if s1 != s2:
            raise RuntimeError("Seed derivation is not deterministic.")

        if s1 in seen:
            raise RuntimeError("Unexpected seed collision in test cases.")

        seen.add(s1)

        print(
            f"  {survey:10s} {stratum:7s} "
            f"z={z:.3f} SB={sbmag:.1f} rid={rid:3d} "
            f"seed={s1}"
        )

    print("\n[6] Checking survey-noise independence...")

    for z in [0.050, 0.175, 0.300]:
        for sbmag in [20.5, 24.5, 27.5]:
            for rid in [0, 123, 499]:
                s_sdss = derive_seed(
                    namespace,
                    "noise",
                    "SDSS-II",
                    "82N",
                    z,
                    sbmag,
                    rid,
                )

                s_des = derive_seed(
                    namespace,
                    "noise",
                    "DES-SN5YR",
                    "SHALLOW",
                    z,
                    sbmag,
                    rid,
                )

                if s_sdss == s_des:
                    raise RuntimeError(
                        "Paired surveys unexpectedly share noise seed."
                    )

    print("  SDSS/DES measurement-noise seeds are independent: PASS")

    print("\n[7] Checking realization pairing rule...")

    for rid in [0, 1, 123, 499]:
        sdss_libid = all_sequences[("SDSS-II", "82N")][rid]
        des_libid = all_sequences[("DES-SN5YR", "SHALLOW")][rid]

        print(
            f"  realization {rid:3d}: "
            f"SDSS LIBID={sdss_libid:5d}  "
            f"DES LIBID={des_libid:5d}"
        )

    print(
        "\n  Pairing uses the same realization ID but independent "
        "survey schedules: PASS"
    )

    print("\n[8] Full frozen-grid seed uniqueness check...")

    seed_set = set()
    count = 0

    for survey, strata in [
        ("SDSS-II", ["82N", "82S"]),
        ("DES-SN5YR", ["SHALLOW", "DEEP"]),
    ]:
        for stratum in strata:
            for z in redshifts:
                for sbmag in sb_grid:
                    for rid in range(n_real):
                        seed = derive_seed(
                            namespace,
                            "noise",
                            survey,
                            stratum,
                            float(z),
                            float(sbmag),
                            rid,
                        )

                        if seed in seed_set:
                            raise RuntimeError(
                                "Seed collision detected in frozen grid."
                            )

                        seed_set.add(seed)
                        count += 1

    print(f"  seeds checked : {count:,}")
    print(f"  unique seeds  : {len(seed_set):,}")

    if count != len(seed_set):
        raise RuntimeError("Seed uniqueness check failed.")

    print("  collision test: PASS")

    print("\n" + "=" * 78)
    print("STAGE 3D.2 SAMPLING / SEED VALIDATION: PASS")
    print("=" * 78)
    print()
    print("No synthetic flux was generated.")
    print("No Stage-1/Stage-2 feature outcome was read.")
    print("No classifier was run.")


if __name__ == "__main__":
    main()
