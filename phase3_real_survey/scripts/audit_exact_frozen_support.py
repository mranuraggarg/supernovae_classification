#!/usr/bin/env python3

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]
SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"
DES_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "des"

ZMIN = 0.05
ZMAX = 0.30
SNR = 3.0
IGNORE_TOL = 0.003

SDSS_TYPES = {
    118: ("Ia", "Ia"),
    120: ("Ia", "Ia"),
    111: ("CC", "Ib"),
    115: ("CC", "Ib"),
    112: ("CC", "Ic"),
    113: ("CC", "II"),
    117: ("CC", "II"),
}

DES_TYPES = {
    1: ("Ia", "Ia"),
    23: ("CC", "IIb"),
    29: ("CC", "II"),
    32: ("CC", "Ib"),
    33: ("CC", "Ic"),
    39: ("CC", "Ibc"),
}

IGNORE_RE = re.compile(
    r"^\s*IGNORE:\s+(\S+)\s+([0-9.]+)\s+([A-Za-z])(?:\s+.*)?$"
)


def decode(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def resolve(cols, names):
    lookup = {c.upper(): c for c in cols}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    return None


def find_first(root: Path, names):
    found = []
    for name in names:
        found += [p for p in root.rglob(name) if p.is_file()]
    if not found:
        return None
    return sorted(set(found), key=lambda p: (len(str(p)), str(p)))[0]


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
    raise RuntimeError(f"No table in {path}")


def get_z(row, cols):
    for name in [
        "REDSHIFT_HELIO",
        "REDSHIFT_FINAL",
        "REDSHIFT_SPEC",
        "HOSTGAL_SPECZ",
        "REDSHIFT",
    ]:
        col = resolve(cols, [name])
        if col is None:
            continue
        try:
            z = float(row[col])
        except Exception:
            continue
        if np.isfinite(z) and z > 0:
            return z
    return None


def top3_positive(flux):
    x = np.asarray(flux, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return 0.0
    x = np.sort(x)[::-1]
    return float(np.mean(x[:min(3, len(x))]))


def frozen_support(times, bands, flux, ferr):
    """
    Exact support logic from phase2_tier4_make_variants.py.

    Assumes band identifiers have ALREADY been transformed according
    to the scenario being tested.
    """
    rows = [
        (float(t), str(b), float(f), float(e))
        for t, b, f, e in zip(times, bands, flux, ferr)
        if b in ("g", "r", "i", "z")
    ]

    if not rows:
        return False, {}

    rows.sort(key=lambda x: x[0])

    active = [
        row for row in rows
        if (
            row[2] > 0
            and row[3] > 0
            and row[2] / row[3] >= SNR
        )
    ]

    active_group = active if active else rows

    t0 = active_group[0][0]
    t1 = active_group[-1][0]

    representatives = {}
    fallback = {}

    for band in "griz":
        band_group = [
            row for row in active_group
            if row[1] == band
        ]

        if not band_group:
            fallback[band] = True
            band_group = [
                row for row in rows
                if (
                    row[1] == band
                    and t0 <= row[0] <= t1
                )
            ]
        else:
            fallback[band] = False

        if not band_group:
            representatives[band] = 0.0
        else:
            representatives[band] = top3_positive(
                [row[2] for row in band_group]
            )

    survive = all(
        representatives[b] > 0
        for b in "griz"
    )

    return survive, {
        "fallback": fallback,
        "representatives": representatives,
        "global_fallback": len(active) == 0,
    }


def build_ignore_map(head, hcols, phot, pcols):
    ignore_file = next(
        p for p in SDSS_ROOT.rglob("SDSS_allCandidates+BOSS.IGNORE")
        if p.is_file()
    )

    entries = []

    for line in ignore_file.read_text(errors="replace").splitlines():
        m = IGNORE_RE.match(line)
        if m:
            cid, mjd, filt = m.groups()
            entries.append((cid, float(mjd), filt))

    idcol = resolve(hcols, ["SNID", "CID"])
    pmincol = resolve(hcols, ["PTROBS_MIN"])
    pmaxcol = resolve(hcols, ["PTROBS_MAX"])
    mjdcol = resolve(pcols, ["MJD"])
    bandcol = resolve(pcols, ["FLT", "BAND", "FILTER"])

    lookup = {
        decode(row[idcol]): i
        for i, row in enumerate(head)
    }

    out = {}

    for cid, target_mjd, filt in entries:
        if cid not in lookup:
            continue

        row = head[lookup[cid]]
        lo = int(row[pmincol])
        hi = int(row[pmaxcol])
        prows = phot[lo - 1:hi]

        mjd = np.asarray(prows[mjdcol], dtype=float)
        band = np.asarray([decode(v) for v in prows[bandcol]], dtype=object)

        delta = np.abs(mjd - target_mjd)

        # IGNORE file uses lowercase filters.
        idx = np.where(
            (np.char.lower(band.astype(str)) == filt.lower())
            & np.isfinite(mjd)
            & (delta <= IGNORE_TOL)
        )[0]

        if len(idx) != 1:
            raise RuntimeError(
                f"IGNORE mapping failure CID={cid} MJD={target_mjd} "
                f"filter={filt}: matches={len(idx)}"
            )

        out.setdefault(cid, set()).add(int(idx[0]))

    return out


def audit(survey, head_path, phot_path):
    hh, hdu_h = first_table(head_path)
    ph, hdu_p = first_table(phot_path)

    try:
        head = hdu_h.data
        phot = hdu_p.data
        hcols = list(hdu_h.columns.names or [])
        pcols = list(hdu_p.columns.names or [])

        idcol = resolve(hcols, ["SNID", "CID"])
        typecol = resolve(hcols, ["SNTYPE", "TYPE"])
        pmincol = resolve(hcols, ["PTROBS_MIN"])
        pmaxcol = resolve(hcols, ["PTROBS_MAX"])

        mjdcol = resolve(pcols, ["MJD"])
        bandcol = resolve(pcols, ["FLT", "BAND", "FILTER"])
        fluxcol = resolve(pcols, ["FLUXCAL"])
        errcol = resolve(pcols, ["FLUXCALERR", "FLUXCAL_ERR"])

        type_map = SDSS_TYPES if survey == "SDSS-II" else DES_TYPES

        ignore_map = (
            build_ignore_map(head, hcols, phot, pcols)
            if survey == "SDSS-II"
            else {}
        )

        counts = {
            "exact": Counter(),
            "casefold": Counter(),
        }

        fallback_counts = {
            "exact": Counter(),
            "casefold": Counter(),
        }

        raw_band_tokens = Counter()

        for hrow in head:
            try:
                code = int(float(decode(hrow[typecol])))
            except Exception:
                continue

            if code not in type_map:
                continue

            cls, subtype = type_map[code]

            z = get_z(hrow, hcols)
            if z is None or not (ZMIN <= z <= ZMAX):
                continue

            lo = int(hrow[pmincol])
            hi = int(hrow[pmaxcol])
            prows = phot[lo - 1:hi]

            mjd = np.asarray(prows[mjdcol], dtype=float)
            flux = np.asarray(prows[fluxcol], dtype=float)
            ferr = np.asarray(prows[errcol], dtype=float)
            rawbands = np.asarray(
                [decode(v) for v in prows[bandcol]],
                dtype=object,
            )

            for token in rawbands:
                raw_band_tokens[str(token)] += 1

            cid = decode(hrow[idcol]) if idcol is not None else ""

            keep = np.ones(len(prows), dtype=bool)

            for j in ignore_map.get(cid, set()):
                keep[j] = False

            structural = (
                keep
                & np.isfinite(mjd)
                & np.isfinite(flux)
                & np.isfinite(ferr)
                & (ferr > 0)
            )

            mjd2 = mjd[structural]
            flux2 = flux[structural]
            ferr2 = ferr[structural]
            raw2 = rawbands[structural]

            scenarios = {
                # Literal frozen builder semantics.
                "exact": raw2.copy(),

                # Current Phase-3 behavior.
                "casefold": np.asarray(
                    [str(v).lower() for v in raw2],
                    dtype=object,
                ),
            }

            for scenario, bands in scenarios.items():
                survive, diag = frozen_support(
                    mjd2,
                    bands,
                    flux2,
                    ferr2,
                )

                counts[scenario][f"in_z_{cls}"] += 1

                if survive:
                    counts[scenario][f"survive_{cls}"] += 1

                    if any(diag["fallback"].values()):
                        fallback_counts[scenario][f"fallback_{cls}"] += 1

                    for b in "griz":
                        if diag["fallback"][b]:
                            fallback_counts[scenario][f"fallback_{cls}_{b}"] += 1

        print()
        print("=" * 78)
        print(survey)
        print("=" * 78)

        print("\nRaw filter tokens:")
        for token, n in raw_band_tokens.most_common():
            print(f"  {token!r:8s} {n:,}")

        for scenario in ("exact", "casefold"):
            print()
            print(f"[{scenario}]")

            print(
                f"  Ia in z support : "
                f"{counts[scenario].get('in_z_Ia', 0):,}"
            )
            print(
                f"  Ia survive      : "
                f"{counts[scenario].get('survive_Ia', 0):,}"
            )
            print(
                f"  CC in z support : "
                f"{counts[scenario].get('in_z_CC', 0):,}"
            )
            print(
                f"  CC survive      : "
                f"{counts[scenario].get('survive_CC', 0):,}"
            )

            print(
                f"  surviving Ia using >=1 band fallback: "
                f"{fallback_counts[scenario].get('fallback_Ia', 0):,}"
            )
            print(
                f"  surviving CC using >=1 band fallback: "
                f"{fallback_counts[scenario].get('fallback_CC', 0):,}"
            )

            print("  fallback by band:")
            for cls in ("Ia", "CC"):
                vals = " ".join(
                    f"{b}={fallback_counts[scenario].get(f'fallback_{cls}_{b}', 0)}"
                    for b in "griz"
                )
                print(f"    {cls}: {vals}")

        return counts

    finally:
        hh.close()
        ph.close()


def main():
    sdss_head = find_first(
        SDSS_ROOT,
        ["SDSS_allCandidates+BOSS_HEAD.FITS",
         "SDSS_allCandidates+BOSS_HEAD.FITS.gz"],
    )
    sdss_phot = find_first(
        SDSS_ROOT,
        ["SDSS_allCandidates+BOSS_PHOT.FITS",
         "SDSS_allCandidates+BOSS_PHOT.FITS.gz"],
    )
    des_head = find_first(
        DES_ROOT,
        ["DES-SN5YR_DES_HEAD.FITS.gz",
         "DES-SN5YR_DES_HEAD.FITS"],
    )
    des_phot = find_first(
        DES_ROOT,
        ["DES-SN5YR_DES_PHOT.FITS.gz",
         "DES-SN5YR_DES_PHOT.FITS"],
    )

    audit("SDSS-II", sdss_head, sdss_phot)
    audit("DES-SN5YR", des_head, des_phot)


if __name__ == "__main__":
    main()
