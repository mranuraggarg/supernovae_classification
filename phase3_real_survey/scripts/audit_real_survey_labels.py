#!/usr/bin/env python3

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"
DES_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "des"

ZMIN = 0.05
ZMAX = 0.30


def decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def find_first(root, names):
    matches = []
    for name in names:
        matches.extend(
            p for p in root.rglob(name)
            if p.is_file()
        )
    if not matches:
        return None
    return sorted(set(matches), key=lambda p: (len(str(p)), str(p)))[0]


def first_table(path):
    hdul = fits.open(path, memmap=True)
    for hdu in hdul:
        if (
            hasattr(hdu, "columns")
            and hdu.columns is not None
            and hdu.data is not None
        ):
            return hdul, hdu
    hdul.close()
    raise RuntimeError(f"No table HDU in {path}")


def resolve(columns, names):
    lookup = {c.upper(): c for c in columns}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    return None


def get_redshift(row, columns):
    for name in [
        "REDSHIFT_HELIO",
        "REDSHIFT_FINAL",
        "REDSHIFT_SPEC",
        "HOSTGAL_SPECZ",
        "REDSHIFT",
    ]:
        col = resolve(columns, [name])
        if col is None:
            continue
        try:
            z = float(row[col])
        except Exception:
            continue
        if np.isfinite(z) and z > 0:
            return z, col
    return None, None


def audit(name, path):
    hdul, hdu = first_table(path)

    try:
        data = hdu.data
        cols = list(hdu.columns.names or [])

        type_col = resolve(cols, ["SNTYPE", "TYPE"])

        if type_col is None:
            raise RuntimeError(f"{name}: no SNTYPE/TYPE column")

        total = Counter()
        in_support = Counter()
        z_columns = Counter()
        examples = defaultdict(list)

        for row in data:
            raw = decode(row[type_col])
            total[raw] += 1

            z, zcol = get_redshift(row, cols)

            if zcol is not None:
                z_columns[zcol] += 1

            if z is not None and ZMIN <= z <= ZMAX:
                in_support[raw] += 1

                if len(examples[raw]) < 5:
                    examples[raw].append(z)

        print()
        print("=" * 78)
        print(name)
        print("=" * 78)
        print(f"HEAD rows: {len(data):,}")
        print(f"Type column: {type_col}")
        print()
        print("Redshift columns actually used:")
        for col, n in z_columns.most_common():
            print(f"  {col:24s} {n:,}")

        print()
        print("SNTYPE / TYPE values")
        print("raw_value        total      z=0.05..0.30     example_z")
        print("-" * 70)

        def sort_key(item):
            value = item[0]
            try:
                return (0, float(value))
            except Exception:
                return (1, value)

        for raw, n in sorted(total.items(), key=sort_key):
            nz = in_support.get(raw, 0)
            ex = ", ".join(f"{x:.4f}" for x in examples.get(raw, []))
            print(f"{raw:14s} {n:9,d} {nz:16,d}     {ex}")

        print()
        print("No class interpretation has been applied.")
        print("No PHOT table or compact feature was inspected.")

    finally:
        hdul.close()


def main():
    sdss = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_HEAD.FITS",
            "SDSS_allCandidates+BOSS_HEAD.FITS.gz",
        ],
    )

    des = find_first(
        DES_ROOT,
        [
            "DES-SN5YR_DES_HEAD.FITS.gz",
            "DES-SN5YR_DES_HEAD.FITS",
        ],
    )

    if sdss is None or des is None:
        raise RuntimeError("Could not locate both HEAD files")

    audit("SDSS-II", sdss)
    audit("DES-SN5YR", des)


if __name__ == "__main__":
    main()
