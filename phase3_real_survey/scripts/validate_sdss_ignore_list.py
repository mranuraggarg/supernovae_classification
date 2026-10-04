#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"

IGNORE_FILE = (
    SDSS_ROOT
    / "SDSS_dataRelease-snana"
    / "SDSS_allCandidates+BOSS"
    / "SDSS_allCandidates+BOSS.IGNORE"
)

HEAD_FILE = (
    SDSS_ROOT
    / "SDSS_allCandidates+BOSS"
    / "SDSS_allCandidates+BOSS"
    / "SDSS_allCandidates+BOSS_HEAD.FITS"
)

PHOT_FILE = (
    SDSS_ROOT
    / "SDSS_allCandidates+BOSS"
    / "SDSS_allCandidates+BOSS"
    / "SDSS_allCandidates+BOSS_PHOT.FITS"
)

# The release notes mention MJD precision/truncation history.
# This tolerance is used only for provenance matching, not science.
MJD_TOL = 0.0030


def decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def resolve(cols, names):
    lookup = {c.upper(): c for c in cols}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    return None


entries = []

pattern = re.compile(
    r"^\s*IGNORE:\s+(\S+)\s+([0-9.]+)\s+([A-Za-z])(?:\s+.*)?$"
)

for lineno, line in enumerate(
    IGNORE_FILE.read_text(errors="replace").splitlines(),
    start=1,
):
    m = pattern.match(line)
    if not m:
        continue

    cid, mjd, filt = m.groups()

    entries.append(
        {
            "line": lineno,
            "cid": cid,
            "mjd": float(mjd),
            "filter": filt.lower(),
            "text": line.strip(),
        }
    )

print("=" * 78)
print("SDSS OFFICIAL IGNORE-LIST VALIDATION")
print("=" * 78)

print(f"\nIgnore file: {IGNORE_FILE}")
print(f"Parsed IGNORE entries: {len(entries)}")

with fits.open(HEAD_FILE, memmap=True) as h:
    head = h[1].data
    hcols = list(h[1].columns.names)

with fits.open(PHOT_FILE, memmap=True) as h:
    phot = h[1].data
    pcols = list(h[1].columns.names)

id_col = resolve(hcols, ["SNID", "CID"])
pmin_col = resolve(hcols, ["PTROBS_MIN"])
pmax_col = resolve(hcols, ["PTROBS_MAX"])

mjd_col = resolve(pcols, ["MJD"])
band_col = resolve(pcols, ["FLT", "BAND", "FILTER"])

if None in (id_col, pmin_col, pmax_col, mjd_col, band_col):
    raise RuntimeError("Required HEAD/PHOT columns not found.")

head_index = {
    decode(row[id_col]): i
    for i, row in enumerate(head)
}

matched = []
unmatched = []
ambiguous = []

for entry in entries:
    cid = entry["cid"]

    if cid not in head_index:
        unmatched.append((entry, "CID not found"))
        continue

    row = head[head_index[cid]]

    pmin = int(row[pmin_col])
    pmax = int(row[pmax_col])

    prows = phot[pmin - 1:pmax]

    mjd = np.asarray(prows[mjd_col], dtype=float)

    band = np.asarray(
        [decode(v).lower() for v in prows[band_col]],
        dtype=object,
    )

    delta = np.abs(mjd - entry["mjd"])

    mask = (
        (band == entry["filter"])
        & np.isfinite(mjd)
        & (delta <= MJD_TOL)
    )

    idx = np.where(mask)[0]

    if len(idx) == 1:
        matched.append(
            (
                entry,
                float(mjd[idx[0]]),
                float(delta[idx[0]]),
            )
        )

    elif len(idx) == 0:
        # Report nearest same-filter row to diagnose precision mismatch.
        same_band = np.where(
            band == entry["filter"]
        )[0]

        if len(same_band):
            nearest = same_band[
                np.argmin(delta[same_band])
            ]

            reason = (
                f"nearest MJD={mjd[nearest]:.6f}, "
                f"delta={delta[nearest]:.6f}"
            )
        else:
            reason = "no epoch in requested filter"

        unmatched.append(
            (
                entry,
                reason,
            )
        )

    else:
        ambiguous.append(
            (
                entry,
                [
                    (float(mjd[i]), float(delta[i]))
                    for i in idx
                ],
            )
        )


print()
print(f"Matched uniquely : {len(matched)}")
print(f"Unmatched        : {len(unmatched)}")
print(f"Ambiguous        : {len(ambiguous)}")

if matched:
    max_delta = max(
        delta
        for _, _, delta in matched
    )

    print(
        f"Maximum matched |delta MJD|: "
        f"{max_delta:.6f} days"
    )


if unmatched:
    print("\nUNMATCHED ENTRIES")
    for entry, reason in unmatched[:30]:
        print(
            f"  line {entry['line']:3d}: "
            f"{entry['cid']} {entry['mjd']:.6f} "
            f"{entry['filter']} -> {reason}"
        )


if ambiguous:
    print("\nAMBIGUOUS ENTRIES")
    for entry, matches in ambiguous[:30]:
        print(
            f"  line {entry['line']:3d}: "
            f"{entry['cid']} {entry['mjd']:.6f} "
            f"{entry['filter']} -> {matches}"
        )


print("\n" + "=" * 78)

# Unmatched IGNORE entries are acceptable only when their CID is
# completely absent from the selected final HEAD release.
unmatched_present_cid = [
    (entry, reason)
    for entry, reason in unmatched
    if entry["cid"] in head_index
]

unmatched_absent_cid = [
    (entry, reason)
    for entry, reason in unmatched
    if entry["cid"] not in head_index
]

if len(unmatched_present_cid) == 0 and len(ambiguous) == 0:
    print("IGNORE-LIST VALIDATION GATE: PASS")
    print(
        "All applicable official IGNORE entries map uniquely "
        "to released SDSS PHOT measurements."
    )
    print(
        f"Non-applicable entries from CIDs absent in the selected release: "
        f"{len(unmatched_absent_cid)}"
    )
else:
    print("IGNORE-LIST VALIDATION GATE: REVIEW REQUIRED")
    print(
        "At least one IGNORE entry for a CID present in the selected release "
        "is unmatched or ambiguous."
    )

    if unmatched_present_cid:
        print("\nUNMATCHED ENTRIES FOR PRESENT CIDs")
        for entry, reason in unmatched_present_cid:
            print(
                f"  line {entry['line']:3d}: "
                f"{entry['cid']} {entry['mjd']:.6f} "
                f"{entry['filter']} -> {reason}"
            )

print("=" * 78)
