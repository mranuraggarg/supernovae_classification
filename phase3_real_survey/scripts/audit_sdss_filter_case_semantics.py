#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]
SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"

ZMIN = 0.05
ZMAX = 0.30

SDSS_SECURE_TYPES = {
    118: ("Ia", "Ia"),
    120: ("Ia", "Ia"),
    111: ("CC", "Ib"),
    115: ("CC", "Ib"),
    112: ("CC", "Ic"),
    113: ("CC", "II"),
    117: ("CC", "II"),
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


def resolve(cols, names):
    lookup = {c.upper(): c for c in cols}
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
    raise RuntimeError(f"No table HDU in {path}")


def get_redshift(row, cols):
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


def print_documentation():
    print("=" * 78)
    print("1. OFFICIAL RELEASE DOCUMENTATION")
    print("=" * 78)

    candidates = []

    for p in SDSS_ROOT.rglob("*"):
        if not p.is_file():
            continue

        name = p.name.lower()

        if (
            "readme02" in name
            or "filter" in name
            or "readme" in name
        ):
            candidates.append(p)

    keywords = [
        "ugrizugriz",
        "ugriz",
        "ccd",
        "filter",
        "data fit",
        "simulation",
        "transmission",
        "kcor",
        "upper",
        "lower",
    ]

    hits = 0

    for path in sorted(
        candidates,
        key=lambda p: str(p),
    ):
        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except Exception:
            continue

        relevant = []

        for i, line in enumerate(lines, start=1):
            low = line.lower()

            if any(k in low for k in keywords):
                relevant.append((i, line))

        if not relevant:
            continue

        print()
        print(f"FILE: {path}")

        for lineno, line in relevant[:120]:
            print(f"  {lineno:4d}: {line}")

        hits += len(relevant)

    print()
    print(f"Documentation hits printed/scanned: {hits}")


def main():
    print_documentation()

    head_path = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_HEAD.FITS",
            "SDSS_allCandidates+BOSS_HEAD.FITS.gz",
        ],
    )

    phot_path = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_PHOT.FITS",
            "SDSS_allCandidates+BOSS_PHOT.FITS.gz",
        ],
    )

    if head_path is None or phot_path is None:
        raise RuntimeError("Could not locate SDSS HEAD/PHOT files.")

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

        required = [
            idcol,
            typecol,
            pmincol,
            pmaxcol,
            mjdcol,
            bandcol,
            fluxcol,
            errcol,
        ]

        if any(x is None for x in required):
            raise RuntimeError("Required SDSS columns missing.")

        print()
        print("=" * 78)
        print("2. RAW PHOT FILTER TOKENS")
        print("=" * 78)

        raw_tokens = Counter(
            decode(v)
            for v in phot[bandcol]
        )

        for token, n in raw_tokens.most_common():
            print(f"  {token!r:8s} {n:,}")

        print()
        print("=" * 78)
        print("3. SECURE COMMON-z SAMPLE: CASE USAGE")
        print("=" * 78)

        objects_by_class = Counter()
        token_objects = Counter()
        case_pattern = Counter()
        per_band_pattern = defaultdict(Counter)

        duplicate_epoch_pairs = Counter()
        near_duplicate_examples = []

        for hrow in head:
            try:
                code = int(float(decode(hrow[typecol])))
            except Exception:
                continue

            if code not in SDSS_SECURE_TYPES:
                continue

            cls, subtype = SDSS_SECURE_TYPES[code]

            z = get_redshift(hrow, hcols)

            if z is None or not (ZMIN <= z <= ZMAX):
                continue

            objects_by_class[cls] += 1

            lo = int(hrow[pmincol])
            hi = int(hrow[pmaxcol])

            prows = phot[lo - 1:hi]

            mjd = np.asarray(
                prows[mjdcol],
                dtype=float,
            )

            flux = np.asarray(
                prows[fluxcol],
                dtype=float,
            )

            ferr = np.asarray(
                prows[errcol],
                dtype=float,
            )

            tokens = np.asarray(
                [decode(v) for v in prows[bandcol]],
                dtype=object,
            )

            structural = (
                np.isfinite(mjd)
                & np.isfinite(flux)
                & np.isfinite(ferr)
                & (ferr > 0)
            )

            mjd = mjd[structural]
            tokens = tokens[structural]

            unique_tokens = set(tokens.tolist())

            for token in unique_tokens:
                token_objects[token] += 1

            lower_present = any(
                t in unique_tokens
                for t in "ugriz"
            )

            upper_present = any(
                t in unique_tokens
                for t in "UGRIZ"
            )

            if lower_present and upper_present:
                case_pattern[f"{cls}:mixed"] += 1
            elif lower_present:
                case_pattern[f"{cls}:lower_only"] += 1
            elif upper_present:
                case_pattern[f"{cls}:upper_only"] += 1
            else:
                case_pattern[f"{cls}:neither"] += 1

            for base in "ugriz":
                low = base
                up = base.upper()

                has_low = low in unique_tokens
                has_up = up in unique_tokens

                if has_low and has_up:
                    patt = "both"
                elif has_low:
                    patt = "lower_only"
                elif has_up:
                    patt = "upper_only"
                else:
                    patt = "absent"

                per_band_pattern[
                    f"{cls}:{base}"
                ][patt] += 1

            # ---------------------------------------------------------
            # Check whether upper/lower variants occur at identical or
            # nearly identical MJDs within the same object.
            # ---------------------------------------------------------
            cid = decode(hrow[idcol])

            for base in "ugriz":
                low_idx = np.where(
                    tokens == base
                )[0]

                up_idx = np.where(
                    tokens == base.upper()
                )[0]

                if len(low_idx) == 0 or len(up_idx) == 0:
                    continue

                low_mjd = mjd[low_idx]
                up_mjd = mjd[up_idx]

                for lm in low_mjd:
                    delta = np.abs(
                        up_mjd - lm
                    )

                    dmin = float(
                        np.min(delta)
                    )

                    if dmin < 1e-6:
                        duplicate_epoch_pairs[
                            f"{base}:same_MJD"
                        ] += 1

                    elif dmin < 0.01:
                        duplicate_epoch_pairs[
                            f"{base}:within_0.01d"
                        ] += 1

                        if len(near_duplicate_examples) < 30:
                            nearest = float(
                                up_mjd[
                                    np.argmin(delta)
                                ]
                            )

                            near_duplicate_examples.append(
                                (
                                    cid,
                                    base,
                                    float(lm),
                                    nearest,
                                    dmin,
                                )
                            )

        print(
            f"Secure common-z objects: "
            f"Ia={objects_by_class['Ia']}, "
            f"CC={objects_by_class['CC']}"
        )

        print()
        print("Objects containing each exact token:")

        for token in list("ugrizUGRIZ"):
            print(
                f"  {token!r}: "
                f"{token_objects.get(token, 0):,}"
            )

        print()
        print("Overall case pattern:")

        for cls in ("Ia", "CC"):
            for patt in (
                "lower_only",
                "upper_only",
                "mixed",
                "neither",
            ):
                print(
                    f"  {cls:2s} {patt:10s}: "
                    f"{case_pattern.get(f'{cls}:{patt}', 0):,}"
                )

        print()
        print("=" * 78)
        print("4. PER-BAND LOWER/UPPER PRESENCE")
        print("=" * 78)

        for cls in ("Ia", "CC"):
            print()
            print(cls)

            for base in "ugriz":
                c = per_band_pattern[
                    f"{cls}:{base}"
                ]

                print(
                    f"  {base}: "
                    f"lower_only={c.get('lower_only', 0):3d}  "
                    f"upper_only={c.get('upper_only', 0):3d}  "
                    f"both={c.get('both', 0):3d}  "
                    f"absent={c.get('absent', 0):3d}"
                )

        print()
        print("=" * 78)
        print("5. LOWER/UPPER MJD COINCIDENCE")
        print("=" * 78)

        if duplicate_epoch_pairs:
            for key, value in sorted(
                duplicate_epoch_pairs.items()
            ):
                print(
                    f"  {key:22s} {value:,}"
                )
        else:
            print(
                "  No same-object lower/upper near-epoch pairs found."
            )

        if near_duplicate_examples:
            print()
            print("Examples within 0.01 day:")

            for (
                cid,
                band,
                low_mjd,
                up_mjd,
                delta,
            ) in near_duplicate_examples:
                print(
                    f"  CID={cid:>6s} "
                    f"{band}/{band.upper()} "
                    f"{low_mjd:.6f} vs {up_mjd:.6f} "
                    f"delta={delta:.6f} d"
                )

        print()
        print("=" * 78)
        print("AUDIT COMPLETE")
        print("=" * 78)

        print(
            "\nNo feature distributions, classifier outputs, or "
            "cross-survey statistics were calculated."
        )

    finally:
        hh.close()
        ph.close()


if __name__ == "__main__":
    main()
