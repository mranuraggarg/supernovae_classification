#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------
# Phase 3 pre-outcome validation:
# validate/freeze release-specific PHOT-row quality handling.
#
# THIS SCRIPT DOES NOT:
#   - calculate any of the 16 compact features
#   - calculate S/N-active windows
#   - inspect class-conditioned outcomes
#   - compare SDSS and DES compact-feature distributions
#   - train or evaluate a classifier
#   - tune any mask to improve survey agreement
#
# It DOES:
#   - identify real photometry rows vs SNANA separator/sentinel rows
#   - verify finite epoch/flux/error values
#   - verify positive flux uncertainties
#   - inspect PHOTFLAG values without assuming undocumented meanings
#   - search local release documentation for flag definitions
#   - freeze the minimal label-blind structural mask
#
# Frozen principle:
#   retain legitimate negative forced-photometry fluxes.
#   Do NOT impose flux > 0 or S/N cuts at the quality-mask stage.
# ---------------------------------------------------------------------


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"
DES_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "des"

OUTDIR = REPO / "phase3_real_survey" / "results" / "quality_mask_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

JSON_REPORT = OUTDIR / "quality_mask_validation.json"
MD_REPORT = OUTDIR / "quality_mask_validation.md"
CSV_REGISTRY = OUTDIR / "frozen_quality_mask_registry.csv"


def decode(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def find_first(root: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []

    for pattern in patterns:
        candidates.extend(
            p for p in root.rglob(pattern)
            if p.is_file()
        )

    if not candidates:
        return None

    return sorted(
        set(candidates),
        key=lambda p: (len(str(p)), str(p))
    )[0]


def resolve_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {c.upper(): c for c in columns}

    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]

    return None


def load_table(path: Path):
    hdul = fits.open(path, memmap=True)

    for hdu in hdul:
        if (
            hasattr(hdu, "columns")
            and hdu.columns is not None
            and hdu.data is not None
        ):
            return hdul, hdu

    hdul.close()
    raise RuntimeError(f"No binary table found in {path}")


def find_docs(root: Path) -> list[Path]:
    patterns = [
        "*.README",
        "README*",
        "*.readme",
        "*.txt",
        "*.md",
        "*.DOC",
        "*.doc",
    ]

    found: list[Path] = []

    for pattern in patterns:
        found.extend(
            p for p in root.rglob(pattern)
            if p.is_file() and p.stat().st_size < 5_000_000
        )

    return sorted(set(found))


def search_docs(
    docs: list[Path],
    terms: list[str],
) -> list[dict[str, Any]]:

    pattern = re.compile(
        "|".join(re.escape(term) for term in terms),
        flags=re.IGNORECASE,
    )

    hits: list[dict[str, Any]] = []

    for path in docs:
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            continue

        lines = text.splitlines()

        for index, line in enumerate(lines, start=1):
            if pattern.search(line):
                hits.append(
                    {
                        "path": str(path),
                        "line": index,
                        "text": line.strip(),
                    }
                )

    return hits


def integer_flag_summary(values: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values)

    try:
        vals = vals.astype(np.int64)
    except Exception:
        return {
            "available": False,
            "reason": "Could not convert flag column to integer.",
        }

    counter = Counter(int(v) for v in vals)

    # Count occurrence of each bit position, without assigning meaning.
    bit_counts = {}

    if counter:
        max_value = max(counter)

        max_bit = max_value.bit_length()

        for bit in range(max_bit):
            mask = 1 << bit
            count = int(np.sum((vals & mask) != 0))
            if count:
                bit_counts[str(bit)] = {
                    "mask_decimal": mask,
                    "rows_set": count,
                }

    return {
        "available": True,
        "unique_value_count": len(counter),
        "most_common_values": [
            {
                "value": value,
                "rows": count,
            }
            for value, count in counter.most_common(25)
        ],
        "bit_occurrence": bit_counts,
    }


def inspect_phot(
    survey: str,
    path: Path,
) -> dict[str, Any]:

    hdul, hdu = load_table(path)

    try:
        columns = list(hdu.columns.names or [])
        data = hdu.data

        time_col = resolve_column(
            columns,
            ["MJD"],
        )

        band_col = resolve_column(
            columns,
            ["FLT", "BAND", "FILTER"],
        )

        flux_col = resolve_column(
            columns,
            ["FLUXCAL"],
        )

        err_col = resolve_column(
            columns,
            ["FLUXCALERR", "FLUXCAL_ERR"],
        )

        flag_col = resolve_column(
            columns,
            ["PHOTFLAG", "PHOT_FLAG"],
        )

        photprob_col = resolve_column(
            columns,
            ["PHOTPROB", "PHOT_PROB"],
        )

        required = {
            "time": time_col,
            "band": band_col,
            "flux": flux_col,
            "flux_error": err_col,
        }

        missing = [
            key
            for key, value in required.items()
            if value is None
        ]

        if missing:
            raise RuntimeError(
                f"{survey}: missing required columns: {missing}"
            )

        mjd = np.asarray(data[time_col], dtype=float)
        flux = np.asarray(data[flux_col], dtype=float)
        ferr = np.asarray(data[err_col], dtype=float)

        bands = np.asarray(
            [decode(v) for v in data[band_col]],
            dtype=object,
        )

        canonical_griz = np.isin(
            np.char.lower(bands.astype(str)),
            np.array(["g", "r", "i", "z"]),
        )

        separator_band = bands == "-"

        finite_time = np.isfinite(mjd)
        finite_flux = np.isfinite(flux)
        finite_err = np.isfinite(ferr)

        positive_err = ferr > 0

        structural_valid = (
            canonical_griz
            & finite_time
            & finite_flux
            & finite_err
            & positive_err
        )

        negative_flux = flux < 0
        zero_flux = flux == 0
        positive_flux = flux > 0

        result: dict[str, Any] = {
            "survey": survey,
            "path": str(path),
            "rows_total": int(len(data)),
            "columns": columns,
            "resolved_columns": {
                "time": time_col,
                "band": band_col,
                "flux": flux_col,
                "flux_error": err_col,
                "photflag": flag_col,
                "photprob": photprob_col,
            },
            "band_counts": dict(
                sorted(
                    Counter(bands.tolist()).items(),
                    key=lambda kv: str(kv[0])
                )
            ),
            "row_audit": {
                "canonical_griz_rows": int(canonical_griz.sum()),
                "separator_dash_rows": int(separator_band.sum()),
                "finite_time_rows": int(finite_time.sum()),
                "finite_flux_rows": int(finite_flux.sum()),
                "finite_error_rows": int(finite_err.sum()),
                "positive_error_rows": int(positive_err.sum()),
                "nonpositive_error_rows": int((~positive_err).sum()),
                "structural_valid_rows": int(structural_valid.sum()),
            },
            "flux_signs_all_rows": {
                "negative": int(negative_flux.sum()),
                "zero": int(zero_flux.sum()),
                "positive": int(positive_flux.sum()),
            },
            "flux_signs_structural_valid_rows": {
                "negative": int((negative_flux & structural_valid).sum()),
                "zero": int((zero_flux & structural_valid).sum()),
                "positive": int((positive_flux & structural_valid).sum()),
            },
        }

        # Verify whether non-positive uncertainty rows are purely separators.
        nonpositive_err = ~positive_err

        result["separator_consistency"] = {
            "nonpositive_error_and_dash_band": int(
                (nonpositive_err & separator_band).sum()
            ),
            "nonpositive_error_not_dash_band": int(
                (nonpositive_err & ~separator_band).sum()
            ),
            "dash_band_with_positive_error": int(
                (separator_band & positive_err).sum()
            ),
        }

        # PHOTFLAG is inspected only descriptively.
        if flag_col is not None:
            result["photflag_summary"] = integer_flag_summary(
                np.asarray(data[flag_col])
            )
        else:
            result["photflag_summary"] = {
                "available": False,
                "reason": "No PHOTFLAG column found.",
            }

        if photprob_col is not None:
            photprob = np.asarray(
                data[photprob_col],
                dtype=float,
            )

            result["photprob_summary"] = {
                "finite_rows": int(np.isfinite(photprob).sum()),
                "min": (
                    float(np.nanmin(photprob))
                    if np.isfinite(photprob).any()
                    else None
                ),
                "max": (
                    float(np.nanmax(photprob))
                    if np.isfinite(photprob).any()
                    else None
                ),
            }

        # No statistics involving S/N or compact features are calculated.
        return result

    finally:
        hdul.close()


def write_registry(
    sdss_result: dict[str, Any],
    des_result: dict[str, Any],
) -> None:

    rows = []

    for result in (sdss_result, des_result):
        survey = result["survey"]

        rows.extend(
            [
                {
                    "survey": survey,
                    "rule_order": 1,
                    "rule": "Keep only canonical griz PHOT rows",
                    "status": "FROZEN",
                    "reason": (
                        "Frozen feature representation uses griz only; "
                        "separator '-' rows and SDSS u/U/G/R/I/Z auxiliary "
                        "tokens are not compact-feature input."
                    ),
                },
                {
                    "survey": survey,
                    "rule_order": 2,
                    "rule": "Require finite MJD",
                    "status": "FROZEN",
                    "reason": "Structural measurement validity.",
                },
                {
                    "survey": survey,
                    "rule_order": 3,
                    "rule": "Require finite FLUXCAL",
                    "status": "FROZEN",
                    "reason": "Structural measurement validity.",
                },
                {
                    "survey": survey,
                    "rule_order": 4,
                    "rule": "Require finite FLUXCALERR",
                    "status": "FROZEN",
                    "reason": "Structural measurement validity.",
                },
                {
                    "survey": survey,
                    "rule_order": 5,
                    "rule": "Require FLUXCALERR > 0",
                    "status": "FROZEN",
                    "reason": (
                        "Non-positive uncertainty rows are SNANA structural/"
                        "separator rows unless audit shows otherwise."
                    ),
                },
                {
                    "survey": survey,
                    "rule_order": 6,
                    "rule": "Retain negative and zero FLUXCAL measurements",
                    "status": "FROZEN",
                    "reason": (
                        "Both releases are forced-photometry products. "
                        "Flux sign is not an epoch-quality criterion."
                    ),
                },
                {
                    "survey": survey,
                    "rule_order": 7,
                    "rule": "No S/N threshold at quality-mask stage",
                    "status": "FROZEN",
                    "reason": (
                        "The S/N >= 3 rule belongs to the already frozen "
                        "compact-feature builder, not release-quality filtering."
                    ),
                },
                {
                    "survey": survey,
                    "rule_order": 8,
                    "rule": "No detection/PHOTPROB threshold",
                    "status": "FROZEN",
                    "reason": (
                        "No threshold may be introduced unless release "
                        "documentation explicitly marks measurements invalid."
                    ),
                },
                {
                    "survey": survey,
                    "rule_order": 9,
                    "rule": "Do not reject PHOTFLAG values by empirical frequency",
                    "status": "FROZEN_PENDING_DOCUMENTED_INVALID_BITS",
                    "reason": (
                        "Bit frequencies alone do not establish quality semantics. "
                        "Only release-documented invalid-image/measurement bits "
                        "may be excluded."
                    ),
                },
            ]
        )

    with CSV_REGISTRY.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "survey",
                "rule_order",
                "rule",
                "status",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:

    print("=" * 78)
    print("PHASE 3 QUALITY-MASK VALIDATION")
    print("NO COMPACT FEATURES OR CLASS-CONDITIONED RESULTS")
    print("=" * 78)

    sdss_phot = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_PHOT.FITS",
            "SDSS_allCandidates+BOSS_PHOT.FITS.gz",
        ],
    )

    des_phot = find_first(
        DES_ROOT,
        [
            "DES-SN5YR_DES_PHOT.FITS.gz",
            "DES-SN5YR_DES_PHOT.FITS",
        ],
    )

    if sdss_phot is None:
        print("ERROR: SDSS PHOT file not found.")
        return 1

    if des_phot is None:
        print("ERROR: DES PHOT file not found.")
        return 1

    print("\n[1] Located photometry products")
    print(f"  SDSS: {sdss_phot}")
    print(f"  DES : {des_phot}")

    print("\n[2] Inspecting structural row validity...")

    sdss = inspect_phot(
        "SDSS-II",
        sdss_phot,
    )

    des = inspect_phot(
        "DES-SN5YR",
        des_phot,
    )

    for result in (sdss, des):
        audit = result["row_audit"]
        sep = result["separator_consistency"]

        print(f"\n  {result['survey']}")
        print(f"    total rows                  : {result['rows_total']:,}")
        print(f"    canonical griz rows         : {audit['canonical_griz_rows']:,}")
        print(f"    separator '-' rows          : {audit['separator_dash_rows']:,}")
        print(f"    non-positive error rows     : {audit['nonpositive_error_rows']:,}")
        print(f"    structurally valid rows     : {audit['structural_valid_rows']:,}")

        print("    separator consistency:")
        print(
            "      nonpositive error + '-'  : "
            f"{sep['nonpositive_error_and_dash_band']:,}"
        )
        print(
            "      nonpositive error not '-' : "
            f"{sep['nonpositive_error_not_dash_band']:,}"
        )
        print(
            "      '-' with positive error   : "
            f"{sep['dash_band_with_positive_error']:,}"
        )

        signs = result["flux_signs_structural_valid_rows"]

        print("    valid-row flux signs:")
        print(f"      negative                  : {signs['negative']:,}")
        print(f"      zero                      : {signs['zero']:,}")
        print(f"      positive                  : {signs['positive']:,}")

    print("\n[3] Searching local release documentation for flag semantics...")

    sdss_docs = find_docs(SDSS_ROOT)
    des_docs = find_docs(DES_ROOT)

    search_terms = [
        "PHOTFLAG",
        "PHOTPROB",
        "flag",
        "bad image",
        "bad phot",
        "reject",
        "invalid",
        "quality",
    ]

    sdss_hits = search_docs(
        sdss_docs,
        search_terms,
    )

    des_hits = search_docs(
        des_docs,
        search_terms,
    )

    print(f"  SDSS documentation files scanned : {len(sdss_docs)}")
    print(f"  SDSS relevant text hits          : {len(sdss_hits)}")
    print(f"  DES documentation files scanned  : {len(des_docs)}")
    print(f"  DES relevant text hits           : {len(des_hits)}")

    print("\n[4] PHOTFLAG descriptive audit...")

    for result in (sdss, des):
        summary = result["photflag_summary"]

        print(f"\n  {result['survey']}")

        if not summary.get("available"):
            print("    PHOTFLAG unavailable.")
            continue

        print(
            f"    unique PHOTFLAG values: "
            f"{summary['unique_value_count']}"
        )

        print("    most common values:")

        for entry in summary["most_common_values"][:10]:
            print(
                f"      {entry['value']:>10d} : "
                f"{entry['rows']:,}"
            )

        print(
            "    NOTE: no PHOTFLAG value is rejected here "
            "without documented semantics."
        )

    print("\n[5] Freezing structural quality mask...")

    write_registry(sdss, des)

    # Gate-critical structural test:
    # every non-positive FLUXCALERR row should be a '-' separator.
    structural_ok = True

    for result in (sdss, des):
        sep = result["separator_consistency"]

        if sep["nonpositive_error_not_dash_band"] != 0:
            structural_ok = False

    # Documentation evidence is retained, but lack of PHOTFLAG bit semantics
    # does not justify inventing a mask.
    report = {
        "purpose": (
            "Freeze label-blind PHOT-row structural quality handling "
            "before any compact-feature comparison."
        ),
        "sdss": sdss,
        "des": des,
        "documentation_search": {
            "sdss_hits": sdss_hits,
            "des_hits": des_hits,
        },
        "frozen_mask": {
            "include_bands": ["g", "r", "i", "z"],
            "require_finite_mjd": True,
            "require_finite_flux": True,
            "require_finite_flux_error": True,
            "require_positive_flux_error": True,
            "retain_negative_flux": True,
            "retain_zero_flux": True,
            "apply_sn_threshold_here": False,
            "apply_detection_threshold": False,
            "empirical_photflag_filter": False,
            "photflag_policy": (
                "Only release-documented invalid measurement/image bits "
                "may be excluded. No bit may be removed because of its "
                "effect on cross-survey feature alignment."
            ),
        },
        "gate": "PASS" if structural_ok else "REVIEW_REQUIRED",
    }

    JSON_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    md: list[str] = [
        "# Phase 3 Quality-Mask Validation",
        "",
        "**No compact features, survey-domain distances, or classifier "
        "outcomes were calculated.**",
        "",
        f"## Gate: {'PASS' if structural_ok else 'REVIEW REQUIRED'}",
        "",
        "## Frozen structural mask",
        "",
        "For both surveys:",
        "",
        "1. Keep canonical `g,r,i,z` photometry rows only.",
        "2. Require finite MJD.",
        "3. Require finite FLUXCAL.",
        "4. Require finite FLUXCALERR.",
        "5. Require `FLUXCALERR > 0`.",
        "6. Retain legitimate negative and zero FLUXCAL measurements.",
        "7. Apply no S/N cut at this stage.",
        "8. Apply no detection/PHOTPROB threshold.",
        "9. Apply no empirical PHOTFLAG rejection.",
        "",
        "The existing frozen builder will later apply its registered "
        "`flux > 0`, `flux_err > 0`, and `S/N >= 3` active-observation "
        "logic. That is a feature-extraction rule, not a release-quality mask.",
        "",
    ]

    for result in (sdss, des):
        audit = result["row_audit"]
        sep = result["separator_consistency"]
        signs = result["flux_signs_structural_valid_rows"]

        md.extend(
            [
                f"## {result['survey']}",
                "",
                f"- Total PHOT rows: {result['rows_total']:,}",
                f"- Canonical griz rows: {audit['canonical_griz_rows']:,}",
                f"- Separator '-' rows: {audit['separator_dash_rows']:,}",
                f"- Structural-valid rows: {audit['structural_valid_rows']:,}",
                f"- Non-positive-error rows outside '-' separators: "
                f"{sep['nonpositive_error_not_dash_band']:,}",
                f"- Valid negative-flux rows retained: {signs['negative']:,}",
                "",
            ]
        )

    md.extend(
        [
            "## PHOTFLAG rule",
            "",
            "PHOTFLAG values were enumerated descriptively only.",
            "",
            "No PHOTFLAG bit is excluded unless the selected release "
            "documentation explicitly identifies it as an invalid "
            "measurement/image condition.",
            "",
            "Observed flag frequency is not evidence of invalidity.",
            "",
            "## Next step",
            "",
            "If this gate passes, the next pre-outcome task is cadence-template "
            "injection using released unlabeled observing schedules.",
            "",
        ]
    )

    MD_REPORT.write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)

    if structural_ok:
        print("QUALITY-MASK VALIDATION GATE: PASS")
        print(
            "Structural mask frozen. Negative forced fluxes remain retained."
        )
    else:
        print("QUALITY-MASK VALIDATION GATE: REVIEW REQUIRED")
        print(
            "At least one real-band row has non-positive FLUXCALERR."
        )

    print("=" * 78)

    print(f"\nMarkdown report : {MD_REPORT}")
    print(f"JSON report     : {JSON_REPORT}")
    print(f"Mask registry   : {CSV_REGISTRY}")

    if not structural_ok:
        print(
            "\nSTOP: inspect the anomalous real-band non-positive-error rows "
            "before proceeding."
        )
        return 1

    print(
        "\nNext permitted step: cadence-template injection using "
        "unlabeled released schedules."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
