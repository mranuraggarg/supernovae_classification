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


pattern = re.compile(
    r"^\s*IGNORE:\s+(\S+)\s+([0-9.]+)\s+([A-Za-z])(?:\s+.*)?$"
)

entries = []

for lineno, line in enumerate(
    IGNORE_FILE.read_text(errors="replace").splitlines(),
    start=1,
):
    m = pattern.match(line)

    if m:
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


head_index = {
    decode(row[id_col]): i
    for i, row in enumerate(head)
}


matched_exactish = []
missing_cid = []
nearest_rows = []


for entry in entries:
    cid = entry["cid"]
    filt = entry["filter"]

    if cid not in head_index:
        missing_cid.append(entry)
        continue

    row = head[head_index[cid]]

    pmin = int(row[pmin_col])
    pmax = int(row[pmax_col])

    prows = phot[pmin - 1:pmax]

    mjd = np.asarray(
        prows[mjd_col],
        dtype=float,
    )

    band = np.asarray(
        [decode(v).lower() for v in prows[band_col]],
        dtype=object,
    )

    same = np.where(
        band == filt
    )[0]

    if len(same) == 0:
        nearest_rows.append(
            {
                **entry,
                "nearest_mjd": np.nan,
                "delta": np.nan,
                "note": "no same-filter epoch",
            }
        )
        continue

    deltas = np.abs(
        mjd[same] - entry["mjd"]
    )

    j = same[
        np.argmin(deltas)
    ]

    nearest_rows.append(
        {
            **entry,
            "nearest_mjd": float(mjd[j]),
            "delta": float(
                abs(mjd[j] - entry["mjd"])
            ),
            "note": "",
        }
    )


print("=" * 78)
print("SDSS IGNORE-LIST MISMATCH AUDIT")
print("=" * 78)

print()
print(f"Parsed entries: {len(entries)}")
print(f"Missing CID entries: {len(missing_cid)}")
print(f"Entries with a same-filter nearest epoch: {sum(np.isfinite(x['delta']) for x in nearest_rows)}")

griz = [
    row
    for row in nearest_rows
    if (
        row["filter"] in {"g", "r", "i", "z"}
        and np.isfinite(row["delta"])
    )
]

if griz:
    deltas = np.array(
        [row["delta"] for row in griz],
        dtype=float,
    )

    print()
    print("Nearest same-filter MJD offsets for griz entries:")
    print(f"  minimum : {deltas.min():.6f} d")
    print(f"  median  : {np.median(deltas):.6f} d")
    print(f"  90th pct: {np.quantile(deltas, 0.90):.6f} d")
    print(f"  95th pct: {np.quantile(deltas, 0.95):.6f} d")
    print(f"  maximum : {deltas.max():.6f} d")

    print()
    print("Largest griz offsets:")

    for row in sorted(
        griz,
        key=lambda x: x["delta"],
        reverse=True,
    )[:20]:

        print(
            f"  line {row['line']:3d}: "
            f"CID={row['cid']:>6s} "
            f"{row['filter']} "
            f"ignore={row['mjd']:.6f} "
            f"nearest={row['nearest_mjd']:.6f} "
            f"delta={row['delta']:.6f}"
        )


print()
print("Missing CIDs:")

for row in missing_cid:
    print(
        f"  line {row['line']:3d}: "
        f"CID={row['cid']} "
        f"MJD={row['mjd']:.6f} "
        f"filter={row['filter']}"
    )


print()
print("CID 19318 presence checks:")

for target in ["19318"]:
    print(
        f"  HEAD contains {target}: "
        f"{target in head_index}"
    )


print()
print("Done. No ignore entry has been applied.")
