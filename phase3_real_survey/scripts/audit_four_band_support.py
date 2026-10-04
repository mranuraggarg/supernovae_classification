#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"
DES_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "des"

Z_MIN = 0.05
Z_MAX = 0.30


SDSS_SECURE_TYPES = {
    118: ("Ia", "Ia"),
    120: ("Ia", "Ia"),
    111: ("CC", "Ib"),
    115: ("CC", "Ib"),
    112: ("CC", "Ic"),
    113: ("CC", "II"),
    117: ("CC", "II"),
}

DES_SECURE_TYPES = {
    1: ("Ia", "Ia"),
    23: ("CC", "IIb"),
    29: ("CC", "II"),
    32: ("CC", "Ib"),
    33: ("CC", "Ic"),
    39: ("CC", "Ibc"),
}


def decode(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


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


def resolve(columns: list[str], names: list[str]) -> str | None:
    lookup = {c.upper(): c for c in columns}

    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]

    return None


def first_table(path: Path):
    hdul = fits.open(path, memmap=True)

    for hdu in hdul:
        if (
            hasattr(hdu, "columns")
            and hdu.columns is not None
            and hdu.data is not None
        ):
            return hdul, hdu

    hdul.close()
    raise RuntimeError(f"No table found in {path}")


def secure_class(survey: str, raw: Any):
    try:
        code = int(float(decode(raw)))
    except Exception:
        return None, None

    if survey == "SDSS-II":
        return SDSS_SECURE_TYPES.get(code, (None, None))

    if survey == "DES-SN5YR":
        return DES_SECURE_TYPES.get(code, (None, None))

    return None, None


def redshift(row, columns):
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
            return z

    return None


def audit_survey(
    survey: str,
    head_path: Path,
    phot_path: Path,
):
    head_hdul, head_hdu = first_table(head_path)
    phot_hdul, phot_hdu = first_table(phot_path)

    try:
        head = head_hdu.data
        phot = phot_hdu.data

        hcols = list(head_hdu.columns.names or [])
        pcols = list(phot_hdu.columns.names or [])

        type_col = resolve(hcols, ["SNTYPE", "TYPE"])
        pmin_col = resolve(hcols, ["PTROBS_MIN"])
        pmax_col = resolve(hcols, ["PTROBS_MAX"])

        mjd_col = resolve(pcols, ["MJD"])
        band_col = resolve(pcols, ["FLT", "BAND", "FILTER"])
        flux_col = resolve(pcols, ["FLUXCAL"])
        ferr_col = resolve(pcols, ["FLUXCALERR", "FLUXCAL_ERR"])

        required = [
            type_col,
            pmin_col,
            pmax_col,
            mjd_col,
            band_col,
            flux_col,
            ferr_col,
        ]

        if any(x is None for x in required):
            raise RuntimeError(f"{survey}: required column missing")

        counts = Counter()
        subtype_counts = Counter()

        for hrow in head:
            cls, subtype = secure_class(
                survey,
                hrow[type_col],
            )

            if cls is None:
                continue

            z = redshift(hrow, hcols)

            if z is None or not (Z_MIN <= z <= Z_MAX):
                continue

            counts[f"secure_z_{cls}"] += 1
            subtype_counts[subtype] += 1

            pmin = int(hrow[pmin_col])
            pmax = int(hrow[pmax_col])

            prows = phot[pmin - 1:pmax]

            mjd = np.asarray(prows[mjd_col], dtype=float)
            flux = np.asarray(prows[flux_col], dtype=float)
            ferr = np.asarray(prows[ferr_col], dtype=float)
            band = np.asarray(
                [decode(v).lower() for v in prows[band_col]],
                dtype=object,
            )

            structural = (
                np.isfinite(mjd)
                & np.isfinite(flux)
                & np.isfinite(ferr)
                & (ferr > 0)
                & np.isin(
                    band,
                    np.array(["g", "r", "i", "z"], dtype=object),
                )
            )

            mjd = mjd[structural]
            flux = flux[structural]
            ferr = ferr[structural]
            band = band[structural]

            raw_support = {
                b: bool(np.any(band == b))
                for b in "griz"
            }

            active = (
                (flux > 0)
                & (ferr > 0)
                & ((flux / ferr) >= 3.0)
            )

            active_support = {
                b: bool(
                    np.any(
                        active
                        & (band == b)
                    )
                )
                for b in "griz"
            }

            all_raw = all(raw_support.values())
            all_active = all(active_support.values())

            if all_raw:
                counts[f"all_raw_griz_{cls}"] += 1

            if all_active:
                counts[f"all_active_griz_{cls}"] += 1

            if all_raw and not all_active:
                counts[f"fallback_required_{cls}"] += 1

            missing_active = tuple(
                b
                for b in "griz"
                if not active_support[b]
            )

            if missing_active:
                counts[
                    f"missing_active_{cls}_{''.join(missing_active)}"
                ] += 1

        print()
        print("=" * 78)
        print(survey)
        print("=" * 78)

        print(
            f"Secure in z support Ia : "
            f"{counts.get('secure_z_Ia', 0):,}"
        )
        print(
            f"Secure in z support CC : "
            f"{counts.get('secure_z_CC', 0):,}"
        )

        print()
        print("All raw griz present:")
        print(
            f"  Ia: {counts.get('all_raw_griz_Ia', 0):,}"
        )
        print(
            f"  CC: {counts.get('all_raw_griz_CC', 0):,}"
        )

        print()
        print("All four griz have >=1 active epoch (flux>0, S/N>=3):")
        print(
            f"  Ia: {counts.get('all_active_griz_Ia', 0):,}"
        )
        print(
            f"  CC: {counts.get('all_active_griz_CC', 0):,}"
        )

        print()
        print("Would require at least one band fallback:")
        print(
            f"  Ia: {counts.get('fallback_required_Ia', 0):,}"
        )
        print(
            f"  CC: {counts.get('fallback_required_CC', 0):,}"
        )

        print()
        print("Most common missing-active-band patterns:")

        for key, n in counts.most_common():
            if key.startswith("missing_active_"):
                print(f"  {key:35s} {n:,}")

        return counts

    finally:
        head_hdul.close()
        phot_hdul.close()


def main():
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

    audit_survey(
        "SDSS-II",
        sdss_head,
        sdss_phot,
    )

    audit_survey(
        "DES-SN5YR",
        des_head,
        des_phot,
    )


if __name__ == "__main__":
    main()
