#!/usr/bin/env python3

"""
Phase 3 primary-cohort validation.

This script validates ONLY population/cohort selection.

It DOES NOT:
- calculate any compact feature;
- summarize feature distributions;
- compare SDSS and DES feature values;
- run a classifier.

Frozen primary support interpretation:
- 0.05 <= z_helio <= 0.30
- secure transient-spectroscopic normal Ia or ordinary CC
- documented structural epoch validity
- SDSS official IGNORE entries removed
- literal lowercase g,r,i,z
- >=1 S/N-active epoch in every one of g,r,i,z
- active := flux > 0, flux_err > 0, flux/flux_err >= 3

Expected primary cohort:
SDSS-II   : 305 Ia / 61 CC
DES-SN5YR : 134 Ia / 51 CC
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = (
    REPO
    / "phase3_real_survey"
    / "data"
    / "raw"
    / "sdss"
)

DES_ROOT = (
    REPO
    / "phase3_real_survey"
    / "data"
    / "raw"
    / "des"
)

OUTDIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "primary_cohort_validation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

JSON_OUT = (
    OUTDIR
    / "primary_cohort_validation.json"
)

TXT_OUT = (
    OUTDIR
    / "primary_cohort_validation.txt"
)

Z_MIN = 0.05
Z_MAX = 0.30
SNR_THRESHOLD = 3.0
IGNORE_TOL_DAYS = 0.003


EXPECTED = {
    "SDSS-II": {
        "Ia": 305,
        "CC": 61,
    },
    "DES-SN5YR": {
        "Ia": 134,
        "CC": 51,
    },
}


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


IGNORE_RE = re.compile(
    r"^\s*IGNORE:\s+"
    r"(\S+)\s+"
    r"([0-9.]+)\s+"
    r"([A-Za-z])"
    r"(?:\s+.*)?$"
)


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        ).strip()

    return str(value).strip()


def resolve(
    columns: list[str],
    candidates: list[str],
) -> str | None:

    lookup = {
        name.upper(): name
        for name in columns
    }

    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[
                candidate.upper()
            ]

    return None


def find_first(
    root: Path,
    names: list[str],
) -> Path | None:

    found = []

    for name in names:
        found.extend(
            p
            for p in root.rglob(name)
            if p.is_file()
        )

    if not found:
        return None

    return sorted(
        set(found),
        key=lambda p: (
            len(str(p)),
            str(p),
        ),
    )[0]


def first_table(path: Path):
    hdul = fits.open(
        path,
        memmap=True,
    )

    for hdu in hdul:
        if (
            hasattr(
                hdu,
                "columns",
            )
            and hdu.columns is not None
            and hdu.data is not None
        ):
            return hdul, hdu

    hdul.close()

    raise RuntimeError(
        f"No FITS table found in {path}"
    )


def get_redshift(
    row,
    columns: list[str],
) -> tuple[
    float | None,
    str | None,
]:

    for name in [
        "REDSHIFT_HELIO",
        "REDSHIFT_FINAL",
        "REDSHIFT_SPEC",
        "HOSTGAL_SPECZ",
        "REDSHIFT",
    ]:

        column = resolve(
            columns,
            [name],
        )

        if column is None:
            continue

        try:
            z = float(
                row[column]
            )
        except Exception:
            continue

        if (
            np.isfinite(z)
            and z > 0
        ):
            return (
                z,
                column,
            )

    return (
        None,
        None,
    )


def secure_class(
    survey: str,
    value: Any,
):

    try:
        code = int(
            float(
                decode(value)
            )
        )
    except Exception:
        return None

    mapping = (
        SDSS_SECURE_TYPES
        if survey == "SDSS-II"
        else DES_SECURE_TYPES
    )

    return mapping.get(code)


def locate_sdss_ignore() -> Path:

    matches = [
        path
        for path in SDSS_ROOT.rglob(
            "SDSS_allCandidates+BOSS.IGNORE"
        )
        if path.is_file()
    ]

    if not matches:
        raise RuntimeError(
            "Official SDSS IGNORE file not found."
        )

    return sorted(
        matches,
        key=lambda p: (
            len(str(p)),
            str(p),
        ),
    )[0]


def parse_ignore(
    path: Path,
):

    entries = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines(),
        start=1,
    ):

        match = IGNORE_RE.match(
            line
        )

        if not match:
            continue

        cid, mjd, filt = (
            match.groups()
        )

        entries.append(
            {
                "line": line_number,
                "cid": cid,
                "mjd": float(mjd),
                "filter": filt.lower(),
            }
        )

    return entries


def build_ignore_map(
    head,
    hcols,
    phot,
    pcols,
):

    entries = parse_ignore(
        locate_sdss_ignore()
    )

    idcol = resolve(
        hcols,
        ["SNID", "CID"],
    )

    pmincol = resolve(
        hcols,
        ["PTROBS_MIN"],
    )

    pmaxcol = resolve(
        hcols,
        ["PTROBS_MAX"],
    )

    mjdcol = resolve(
        pcols,
        ["MJD"],
    )

    bandcol = resolve(
        pcols,
        [
            "FLT",
            "BAND",
            "FILTER",
        ],
    )

    required = [
        idcol,
        pmincol,
        pmaxcol,
        mjdcol,
        bandcol,
    ]

    if any(
        value is None
        for value in required
    ):
        raise RuntimeError(
            "Cannot resolve columns needed "
            "for SDSS IGNORE mapping."
        )

    head_lookup = {
        decode(row[idcol]): index
        for index, row
        in enumerate(head)
    }

    ignore_map = {}

    audit = Counter()

    maximum_delta = 0.0

    for entry in entries:

        audit["parsed"] += 1

        cid = entry["cid"]

        if cid not in head_lookup:
            audit[
                "absent_release_cid"
            ] += 1
            continue

        hrow = head[
            head_lookup[cid]
        ]

        pmin = int(
            hrow[pmincol]
        )

        pmax = int(
            hrow[pmaxcol]
        )

        rows = phot[
            pmin - 1:pmax
        ]

        mjd = np.asarray(
            rows[mjdcol],
            dtype=float,
        )

        band = np.asarray(
            [
                decode(v)
                for v in rows[
                    bandcol
                ]
            ],
            dtype=object,
        )

        delta = np.abs(
            mjd
            - entry["mjd"]
        )

        matches = np.where(
            (
                np.char.lower(
                    band.astype(str)
                )
                == entry["filter"]
            )
            & np.isfinite(mjd)
            & (
                delta
                <= IGNORE_TOL_DAYS
            )
        )[0]

        if len(matches) == 1:

            index = int(
                matches[0]
            )

            ignore_map.setdefault(
                cid,
                set(),
            ).add(
                index
            )

            audit[
                "applied"
            ] += 1

            maximum_delta = max(
                maximum_delta,
                float(
                    delta[index]
                ),
            )

        elif len(matches) == 0:

            audit[
                "unmatched_present"
            ] += 1

        else:

            audit[
                "ambiguous"
            ] += 1

    if (
        audit[
            "unmatched_present"
        ]
        != 0
        or audit[
            "ambiguous"
        ]
        != 0
    ):
        raise RuntimeError(
            "SDSS IGNORE mapping does not "
            "match validated state."
        )

    return (
        ignore_map,
        {
            "parsed":
                audit["parsed"],

            "applied":
                audit["applied"],

            "absent_release_cid":
                audit[
                    "absent_release_cid"
                ],

            "unmatched_present":
                audit[
                    "unmatched_present"
                ],

            "ambiguous":
                audit[
                    "ambiguous"
                ],

            "maximum_delta_days":
                maximum_delta,
        },
    )


def validate_survey(
    survey: str,
    head_path: Path,
    phot_path: Path,
):

    hh, hdu_h = first_table(
        head_path
    )

    ph, hdu_p = first_table(
        phot_path
    )

    try:

        head = hdu_h.data
        phot = hdu_p.data

        hcols = list(
            hdu_h.columns.names
            or []
        )

        pcols = list(
            hdu_p.columns.names
            or []
        )

        idcol = resolve(
            hcols,
            ["SNID", "CID"],
        )

        typecol = resolve(
            hcols,
            ["SNTYPE", "TYPE"],
        )

        pmincol = resolve(
            hcols,
            ["PTROBS_MIN"],
        )

        pmaxcol = resolve(
            hcols,
            ["PTROBS_MAX"],
        )

        mjdcol = resolve(
            pcols,
            ["MJD"],
        )

        bandcol = resolve(
            pcols,
            [
                "FLT",
                "BAND",
                "FILTER",
            ],
        )

        fluxcol = resolve(
            pcols,
            ["FLUXCAL"],
        )

        errcol = resolve(
            pcols,
            [
                "FLUXCALERR",
                "FLUXCAL_ERR",
            ],
        )

        required = {
            "type": typecol,
            "PTROBS_MIN": pmincol,
            "PTROBS_MAX": pmaxcol,
            "MJD": mjdcol,
            "band": bandcol,
            "FLUXCAL": fluxcol,
            "FLUXCALERR": errcol,
        }

        missing = [
            name
            for name, value
            in required.items()
            if value is None
        ]

        if missing:
            raise RuntimeError(
                f"{survey}: missing columns "
                f"{missing}"
            )

        ignore_map = {}
        ignore_audit = None

        if survey == "SDSS-II":

            (
                ignore_map,
                ignore_audit,
            ) = build_ignore_map(
                head,
                hcols,
                phot,
                pcols,
            )

        counts = Counter()
        subtype_counts = Counter()
        redshift_sources = Counter()
        missing_patterns = Counter()

        accepted_ids = []

        for hrow in head:

            counts[
                "head_total"
            ] += 1

            class_info = secure_class(
                survey,
                hrow[typecol],
            )

            if class_info is None:
                continue

            cls, subtype = (
                class_info
            )

            counts[
                f"secure_{cls}"
            ] += 1

            z, z_source = (
                get_redshift(
                    hrow,
                    hcols,
                )
            )

            if z is None:
                counts[
                    "missing_redshift"
                ] += 1
                continue

            redshift_sources[
                z_source
            ] += 1

            if not (
                Z_MIN
                <= z
                <= Z_MAX
            ):
                counts[
                    f"outside_z_{cls}"
                ] += 1
                continue

            counts[
                f"in_z_{cls}"
            ] += 1

            pmin = int(
                hrow[pmincol]
            )

            pmax = int(
                hrow[pmaxcol]
            )

            if (
                pmin < 1
                or pmax < pmin
                or pmax > len(phot)
            ):
                counts[
                    "invalid_pointer"
                ] += 1
                continue

            rows = phot[
                pmin - 1:pmax
            ]

            mjd = np.asarray(
                rows[mjdcol],
                dtype=float,
            )

            flux = np.asarray(
                rows[fluxcol],
                dtype=float,
            )

            ferr = np.asarray(
                rows[errcol],
                dtype=float,
            )

            bands = np.asarray(
                [
                    decode(v)
                    for v in rows[
                        bandcol
                    ]
                ],
                dtype=object,
            )

            cid = (
                decode(
                    hrow[idcol]
                )
                if idcol is not None
                else ""
            )

            keep = np.ones(
                len(rows),
                dtype=bool,
            )

            if (
                survey == "SDSS-II"
            ):
                for local_index in (
                    ignore_map.get(
                        cid,
                        set(),
                    )
                ):
                    keep[
                        local_index
                    ] = False

            # Frozen structural validity.
            structural = (
                keep
                & np.isfinite(mjd)
                & np.isfinite(flux)
                & np.isfinite(ferr)
                & (ferr > 0)
            )

            mjd = mjd[
                structural
            ]

            flux = flux[
                structural
            ]

            ferr = ferr[
                structural
            ]

            bands = bands[
                structural
            ]

            # IMPORTANT:
            # literal lowercase only.
            #
            # No case folding and no
            # UGRIZ -> ugriz remapping
            # in the primary cohort.
            active = (
                (flux > 0)
                & (ferr > 0)
                & (
                    flux / ferr
                    >= SNR_THRESHOLD
                )
            )

            support = {
                band: bool(
                    np.any(
                        active
                        & (
                            bands
                            == band
                        )
                    )
                )
                for band
                in "griz"
            }

            missing = "".join(
                band
                for band in "griz"
                if not support[
                    band
                ]
            )

            if missing:

                counts[
                    f"failed_support_{cls}"
                ] += 1

                missing_patterns[
                    f"{cls}:{missing}"
                ] += 1

                continue

            counts[
                f"accepted_{cls}"
            ] += 1

            subtype_counts[
                subtype
            ] += 1

            accepted_ids.append(
                cid
            )

        result = {
            "survey":
                survey,

            "expected":
                EXPECTED[survey],

            "observed": {
                "Ia":
                    counts.get(
                        "accepted_Ia",
                        0,
                    ),

                "CC":
                    counts.get(
                        "accepted_CC",
                        0,
                    ),
            },

            "counts":
                dict(counts),

            "accepted_subtypes":
                dict(
                    subtype_counts
                ),

            "redshift_sources":
                dict(
                    redshift_sources
                ),

            "support_failure_patterns":
                dict(
                    missing_patterns
                ),

            "sdss_ignore":
                ignore_audit,

            "accepted_object_count":
                len(
                    accepted_ids
                ),
        }

        result[
            "pass"
        ] = (
            result[
                "observed"
            ]
            == result[
                "expected"
            ]
        )

        return result

    finally:

        hh.close()
        ph.close()


def format_result(
    result,
) -> list[str]:

    lines = []

    survey = result[
        "survey"
    ]

    observed = result[
        "observed"
    ]

    expected = result[
        "expected"
    ]

    counts = result[
        "counts"
    ]

    lines.append(
        survey
    )

    lines.append(
        "-" * len(survey)
    )

    lines.append(
        "Secure objects within z support:"
    )

    lines.append(
        f"  Ia : "
        f"{counts.get('in_z_Ia', 0):,}"
    )

    lines.append(
        f"  CC : "
        f"{counts.get('in_z_CC', 0):,}"
    )

    lines.append("")

    lines.append(
        "Primary four-band survivors:"
    )

    lines.append(
        f"  Ia : "
        f"{observed['Ia']:,} "
        f"(expected {expected['Ia']:,})"
    )

    lines.append(
        f"  CC : "
        f"{observed['CC']:,} "
        f"(expected {expected['CC']:,})"
    )

    lines.append("")

    lines.append(
        "Support failures:"
    )

    lines.append(
        f"  Ia : "
        f"{counts.get('failed_support_Ia', 0):,}"
    )

    lines.append(
        f"  CC : "
        f"{counts.get('failed_support_CC', 0):,}"
    )

    if result[
        "sdss_ignore"
    ] is not None:

        ig = result[
            "sdss_ignore"
        ]

        lines.append("")

        lines.append(
            "SDSS official IGNORE:"
        )

        lines.append(
            f"  parsed             : "
            f"{ig['parsed']}"
        )

        lines.append(
            f"  applied            : "
            f"{ig['applied']}"
        )

        lines.append(
            f"  absent-release CID : "
            f"{ig['absent_release_cid']}"
        )

        lines.append(
            f"  ambiguous          : "
            f"{ig['ambiguous']}"
        )

        lines.append(
            f"  unmatched-present  : "
            f"{ig['unmatched_present']}"
        )

        lines.append(
            f"  maximum MJD delta  : "
            f"{ig['maximum_delta_days']:.6f} d"
        )

    lines.append("")

    lines.append(
        "Gate: "
        + (
            "PASS"
            if result["pass"]
            else "FAIL"
        )
    )

    return lines


def main():

    print("=" * 78)
    print(
        "PHASE 3 PRIMARY-COHORT VALIDATION"
    )
    print(
        "COUNTS AND SUPPORT ONLY - "
        "NO COMPACT FEATURES"
    )
    print("=" * 78)

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

    paths = {
        "SDSS HEAD":
            sdss_head,

        "SDSS PHOT":
            sdss_phot,

        "DES HEAD":
            des_head,

        "DES PHOT":
            des_phot,
    }

    print()
    print("[1] Located products")

    for label, path in paths.items():

        if path is None:
            raise RuntimeError(
                f"{label} not found."
            )

        print(
            f"  {label:10s}: "
            f"{path}"
        )

    print()
    print(
        "[2] Applying frozen primary "
        "population/support rule..."
    )

    sdss = validate_survey(
        "SDSS-II",
        sdss_head,
        sdss_phot,
    )

    des = validate_survey(
        "DES-SN5YR",
        des_head,
        des_phot,
    )

    print()

    for line in format_result(
        sdss
    ):
        print(line)

    print()

    for line in format_result(
        des
    ):
        print(line)

    overall_pass = (
        sdss["pass"]
        and des["pass"]
    )

    payload = {
        "stage":
            "primary_cohort_validation",

        "feature_values_computed":
            False,

        "cross_survey_feature_statistics":
            False,

        "primary_rule": {
            "z_min":
                Z_MIN,

            "z_max":
                Z_MAX,

            "snr_threshold":
                SNR_THRESHOLD,

            "bands":
                [
                    "g",
                    "r",
                    "i",
                    "z",
                ],

            "sdss_filter_case":
                "literal lowercase only",

            "require_active_support_each_band":
                True,

            "sdss_ignore_applied":
                True,
        },

        "sdss":
            sdss,

        "des":
            des,

        "overall_pass":
            overall_pass,
    }

    JSON_OUT.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "=" * 78,
        "PHASE 3 PRIMARY-COHORT VALIDATION",
        "COUNTS AND SUPPORT ONLY - NO COMPACT FEATURES",
        "=" * 78,
        "",
    ]

    report_lines.extend(
        format_result(
            sdss
        )
    )

    report_lines.append("")

    report_lines.extend(
        format_result(
            des
        )
    )

    report_lines.extend(
        [
            "",
            "=" * 78,
            (
                "PRIMARY-COHORT VALIDATION GATE: PASS"
                if overall_pass
                else
                "PRIMARY-COHORT VALIDATION GATE: FAIL"
            ),
            "=" * 78,
            "",
            "No compact-feature values were computed.",
            "No SDSS-DES feature outcome was inspected.",
            "",
        ]
    )

    TXT_OUT.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)

    if overall_pass:

        print(
            "PRIMARY-COHORT VALIDATION GATE: PASS"
        )

        print(
            "Frozen primary selection reproduces "
            "the registered cohort exactly."
        )

    else:

        print(
            "PRIMARY-COHORT VALIDATION GATE: FAIL"
        )

        print(
            "Do not generate confirmatory "
            "real-survey features."
        )

    print("=" * 78)

    print()
    print(
        f"JSON   : {JSON_OUT}"
    )

    print(
        f"Report : {TXT_OUT}"
    )

    return (
        0
        if overall_pass
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
