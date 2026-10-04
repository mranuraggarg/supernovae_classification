#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from astropy.io import fits
except ImportError:
    print("ERROR: astropy is required in the active environment.", file=sys.stderr)
    print("Install it into astro-ml before running this validator.", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------
# Phase-3 SDSS-II / DES-SN5YR input validation
#
# IMPORTANT:
#   This script does NOT:
#   - compute compact features
#   - compare cross-survey feature distributions
#   - inspect classifier performance
#   - tune any threshold
#   - construct any normalization
#
# It validates only:
#   - product identity/checksums
#   - FITS accessibility
#   - required schema
#   - band tokens
#   - presence of negative forced-photometry values
#   - metadata needed for later frozen-builder execution
# ---------------------------------------------------------------------


REPO = Path(__file__).resolve().parents[2]

SDSS_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "sdss"
DES_ROOT = REPO / "phase3_real_survey" / "data" / "raw" / "des"

REPORT_DIR = REPO / "phase3_real_survey" / "results" / "input_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

JSON_REPORT = REPORT_DIR / "input_validation_report.json"
MD_REPORT = REPORT_DIR / "input_validation_report.md"


EXPECTED = {
    "sdss_archive": {
        "name": "SDSS_dataRelease-snana.tar.gz",
        "sha256": "b0861ce0d8bd5ab138bc2e9f41dd5321912d76cb43c718789a7bc637d56e3dbd",
    },
    "sdss_nested": {
        "name": "SDSS_allCandidates+BOSS.tar.gz",
        "sha256": "033992473b0dace08233bf9b4ef395a2267148f64ee021cc725a475d70d34d42",
    },
    "des_archive": {
        "name": "DES-SN5YR-1.2.zip",
        "md5": "9019a6ddc569553bc323e9e1b68a55bf",
    },
    "des_head": {
        "name": "DES-SN5YR_DES_HEAD.FITS.gz",
        "sha256": "08c1fb41dfe8b4e8a969f205144f94c9f6838a1a895a0fedc75b837fb87fd00a",
    },
    "des_phot": {
        "name": "DES-SN5YR_DES_PHOT.FITS.gz",
        "sha256": "10248f0364e90f88626ef19cf5ecf052500bbcbfd27f14f8b94cf8ffbc7dd01c",
    },
    "des_readme": {
        "name": "DES-SN5YR_DES.README",
        "sha256": "75cd01a0786cbe983b71b154565c8158fcb97e4b7113633457b77350eb845544",
    },
}


OBS_REQUIRED_ALIASES = {
    "time": ["MJD"],
    "band": ["FLT", "BAND", "FILTER"],
    "flux": ["FLUXCAL"],
    "flux_error": ["FLUXCALERR", "FLUXCAL_ERR"],
}

OBS_EXPECTED_ALIASES = {
    "quality_flag": ["PHOTFLAG", "PHOT_FLAG"],
    "field": ["FIELD"],
    "zeropoint": ["ZEROPT", "ZPT"],
    "gain": ["GAIN"],
    "psf": ["PSF_SIG1", "PSF_SIG", "PSF"],
    "sky": ["SKY_SIG", "SKYSIG"],
    "read_noise": ["RDNOISE", "READNOISE", "READ_NOISE"],
}

HEAD_EXPECTED_ALIASES = {
    "class_label": ["SNTYPE", "TYPE"],
    "heliocentric_redshift": ["REDSHIFT_HELIO", "REDSHIFT_FINAL", "REDSHIFT"],
    "mw_extinction": ["MWEBV"],
    "nobs": ["NOBS"],
}


def hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_all(root: Path, filename: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob(filename) if p.is_file())


def find_first(root: Path, patterns: list[str]) -> Path | None:
    if not root.exists():
        return None

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(p for p in root.rglob(pattern) if p.is_file())

    if not candidates:
        return None

    # Prefer exact released-product names where possible.
    return sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))[0]


def decode_value(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def resolve_alias(columns: list[str], aliases: list[str]) -> str | None:
    cmap = {c.upper(): c for c in columns}
    for alias in aliases:
        if alias.upper() in cmap:
            return cmap[alias.upper()]
    return None


def find_table_hdu(path: Path, preferred_tokens: list[str]):
    hdul = fits.open(path, memmap=True)

    # Prefer table HDUs whose name contains a requested token.
    for hdu in hdul:
        if not hasattr(hdu, "columns") or hdu.columns is None:
            continue
        name = (getattr(hdu, "name", "") or "").upper()
        if any(token.upper() in name for token in preferred_tokens):
            return hdul, hdu

    # Otherwise return the first binary table.
    for hdu in hdul:
        if hasattr(hdu, "columns") and hdu.columns is not None:
            return hdul, hdu

    hdul.close()
    raise RuntimeError(f"No FITS table HDU found in {path}")


def summarize_columns(path: Path, table_kind: str) -> dict[str, Any]:
    preferred = ["PHOT"] if table_kind == "phot" else ["HEAD"]
    hdul, hdu = find_table_hdu(path, preferred)

    try:
        cols = list(hdu.columns.names or [])
        nrows = len(hdu.data) if hdu.data is not None else 0

        alias_groups = (
            {**OBS_REQUIRED_ALIASES, **OBS_EXPECTED_ALIASES}
            if table_kind == "phot"
            else HEAD_EXPECTED_ALIASES
        )

        resolved = {
            key: resolve_alias(cols, aliases)
            for key, aliases in alias_groups.items()
        }

        return {
            "path": str(path),
            "hdu_name": getattr(hdu, "name", ""),
            "rows": int(nrows),
            "columns": cols,
            "resolved": resolved,
        }
    finally:
        hdul.close()


def analyze_phot_schema(path: Path, survey: str) -> dict[str, Any]:
    hdul, hdu = find_table_hdu(path, ["PHOT"])

    try:
        cols = list(hdu.columns.names or [])
        data = hdu.data

        resolved_required = {
            key: resolve_alias(cols, aliases)
            for key, aliases in OBS_REQUIRED_ALIASES.items()
        }

        resolved_expected = {
            key: resolve_alias(cols, aliases)
            for key, aliases in OBS_EXPECTED_ALIASES.items()
        }

        missing_required = [
            key for key, col in resolved_required.items() if col is None
        ]

        result: dict[str, Any] = {
            "path": str(path),
            "rows": int(len(data)),
            "required_columns": resolved_required,
            "expected_metadata_columns": resolved_expected,
            "missing_required": missing_required,
            "builder_schema_compatible": len(missing_required) == 0,
        }

        band_col = resolved_required["band"]
        flux_col = resolved_required["flux"]
        err_col = resolved_required["flux_error"]
        time_col = resolved_required["time"]

        if band_col is not None:
            raw_bands = data[band_col]
            unique_bands = sorted({decode_value(v) for v in raw_bands})
            result["band_tokens"] = unique_bands

            canonical_griz = {"g", "r", "i", "z"}

            direct_griz = set(unique_bands) & canonical_griz
            prefixed_griz = {
                token
                for token in unique_bands
                if token.lower() in {"des-g", "des-r", "des-i", "des-z"}
            }

            if canonical_griz.issubset(set(unique_bands)):
                result["band_token_status"] = "DIRECT_GRIZ"
            elif len(prefixed_griz) == 4:
                result["band_token_status"] = "REQUIRES_PREDECLARED_TOKEN_MAPPING"
            else:
                result["band_token_status"] = "CHECK_REQUIRED"

        if flux_col is not None:
            flux = data[flux_col]
            finite_mask = ~getattr(flux, "mask", False)

            # FITS columns generally return ndarray; keep this simple and robust.
            try:
                n_negative = int((flux < 0).sum())
                n_zero = int((flux == 0).sum())
                n_positive = int((flux > 0).sum())
            except Exception:
                n_negative = n_zero = n_positive = -1

            result["flux_sign_counts"] = {
                "negative": n_negative,
                "zero": n_zero,
                "positive": n_positive,
            }

        if err_col is not None:
            ferr = data[err_col]
            try:
                result["nonpositive_flux_error_rows"] = int((ferr <= 0).sum())
            except Exception:
                result["nonpositive_flux_error_rows"] = -1

        if time_col is not None:
            mjd = data[time_col]
            try:
                result["mjd_min"] = float(mjd.min())
                result["mjd_max"] = float(mjd.max())
            except Exception:
                pass

        # We deliberately do NOT calculate S/N distributions, active windows,
        # compact features, or survey-to-survey statistics here.

        return result
    finally:
        hdul.close()


def validate_hash(
    root: Path,
    filename: str,
    algo: str,
    expected_digest: str,
) -> dict[str, Any]:
    matches = find_all(root, filename)

    if not matches:
        return {
            "filename": filename,
            "status": "NOT_FOUND",
            "expected": expected_digest,
        }

    records = []
    for path in matches:
        digest = hash_file(path, algo)
        records.append(
            {
                "path": str(path),
                "digest": digest,
                "matches_expected": digest.lower() == expected_digest.lower(),
            }
        )

    overall = (
        "PASS"
        if any(r["matches_expected"] for r in records)
        else "HASH_MISMATCH"
    )

    return {
        "filename": filename,
        "algorithm": algo,
        "expected": expected_digest,
        "status": overall,
        "matches": records,
    }


def markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Check | Result |", "|---|---|"]
    for a, b in rows:
        lines.append(f"| {a} | {b} |")
    return "\n".join(lines)


def main() -> int:
    report: dict[str, Any] = {
        "purpose": "Phase-3 input validation only; no compact-feature comparison",
        "repo": str(REPO),
        "sdss_root": str(SDSS_ROOT),
        "des_root": str(DES_ROOT),
        "checksums": {},
        "sdss": {},
        "des": {},
        "final_gate": None,
    }

    print("=" * 72)
    print("PHASE 3 INPUT VALIDATION")
    print("SDSS-II final SMP <-> DES-SN5YR SMP v1.2")
    print("=" * 72)

    # --------------------------------------------------------------
    # Checksums
    # --------------------------------------------------------------

    print("\n[1] Checking immutable product identities...")

    report["checksums"]["sdss_archive"] = validate_hash(
        SDSS_ROOT,
        EXPECTED["sdss_archive"]["name"],
        "sha256",
        EXPECTED["sdss_archive"]["sha256"],
    )

    report["checksums"]["sdss_nested"] = validate_hash(
        SDSS_ROOT,
        EXPECTED["sdss_nested"]["name"],
        "sha256",
        EXPECTED["sdss_nested"]["sha256"],
    )

    report["checksums"]["des_archive"] = validate_hash(
        DES_ROOT,
        EXPECTED["des_archive"]["name"],
        "md5",
        EXPECTED["des_archive"]["md5"],
    )

    report["checksums"]["des_head"] = validate_hash(
        DES_ROOT,
        EXPECTED["des_head"]["name"],
        "sha256",
        EXPECTED["des_head"]["sha256"],
    )

    report["checksums"]["des_phot"] = validate_hash(
        DES_ROOT,
        EXPECTED["des_phot"]["name"],
        "sha256",
        EXPECTED["des_phot"]["sha256"],
    )

    report["checksums"]["des_readme"] = validate_hash(
        DES_ROOT,
        EXPECTED["des_readme"]["name"],
        "sha256",
        EXPECTED["des_readme"]["sha256"],
    )

    for key, rec in report["checksums"].items():
        print(f"  {key:18s}: {rec['status']}")

    # --------------------------------------------------------------
    # Locate FITS products
    # --------------------------------------------------------------

    print("\n[2] Locating FITS products...")

    sdss_head = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_HEAD.FITS.gz",
            "SDSS_allCandidates+BOSS_HEAD.FITS",
        ],
    )

    sdss_phot = find_first(
        SDSS_ROOT,
        [
            "SDSS_allCandidates+BOSS_PHOT.FITS.gz",
            "SDSS_allCandidates+BOSS_PHOT.FITS",
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
        "SDSS HEAD": sdss_head,
        "SDSS PHOT": sdss_phot,
        "DES HEAD": des_head,
        "DES PHOT": des_phot,
    }

    for label, path in paths.items():
        print(f"  {label:10s}: {path if path else 'NOT FOUND'}")

    missing_fits = [label for label, path in paths.items() if path is None]

    if missing_fits:
        report["final_gate"] = "FAIL"
        report["missing_fits_products"] = missing_fits

        JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nFAIL: missing FITS products:")
        for item in missing_fits:
            print(f"  - {item}")
        print(f"\nReport written to: {JSON_REPORT}")
        return 1

    # --------------------------------------------------------------
    # Schema checks
    # --------------------------------------------------------------

    print("\n[3] Checking HEAD schemas...")

    report["sdss"]["head"] = summarize_columns(sdss_head, "head")
    report["des"]["head"] = summarize_columns(des_head, "head")

    for survey in ("sdss", "des"):
        h = report[survey]["head"]
        print(f"\n  {survey.upper()} HEAD rows: {h['rows']}")
        for logical, actual in h["resolved"].items():
            print(f"    {logical:24s}: {actual or 'NOT FOUND'}")

    print("\n[4] Checking PHOT schemas...")

    report["sdss"]["phot"] = analyze_phot_schema(sdss_phot, "SDSS")
    report["des"]["phot"] = analyze_phot_schema(des_phot, "DES")

    for survey in ("sdss", "des"):
        p = report[survey]["phot"]

        print(f"\n  {survey.upper()} PHOT rows: {p['rows']:,}")
        print(
            "    builder schema compatible:",
            p["builder_schema_compatible"],
        )

        print("    required columns:")
        for logical, actual in p["required_columns"].items():
            print(f"      {logical:14s}: {actual or 'NOT FOUND'}")

        print("    band tokens:", ", ".join(p.get("band_tokens", [])))
        print(
            "    band token status:",
            p.get("band_token_status", "UNKNOWN"),
        )

        signs = p.get("flux_sign_counts", {})
        if signs:
            print(
                "    flux signs:",
                f"negative={signs.get('negative'):,}",
                f"zero={signs.get('zero'):,}",
                f"positive={signs.get('positive'):,}",
            )

        print(
            "    nonpositive flux errors:",
            p.get("nonpositive_flux_error_rows", "UNKNOWN"),
        )

    # --------------------------------------------------------------
    # Builder-compatibility gate
    # --------------------------------------------------------------

    print("\n[5] Builder compatibility gate...")

    sdss_ok = report["sdss"]["phot"]["builder_schema_compatible"]
    des_ok = report["des"]["phot"]["builder_schema_compatible"]

    sdss_band_status = report["sdss"]["phot"].get(
        "band_token_status", "CHECK_REQUIRED"
    )
    des_band_status = report["des"]["phot"].get(
        "band_token_status", "CHECK_REQUIRED"
    )

    direct_or_mappable = {
        "DIRECT_GRIZ",
        "REQUIRES_PREDECLARED_TOKEN_MAPPING",
    }

    schema_gate = (
        sdss_ok
        and des_ok
        and sdss_band_status in direct_or_mappable
        and des_band_status in direct_or_mappable
    )

    if schema_gate:
        report["final_gate"] = "PASS"
    else:
        report["final_gate"] = "REVIEW_REQUIRED"

    # --------------------------------------------------------------
    # Markdown summary
    # --------------------------------------------------------------

    md_lines = [
        "# Phase 3 Input Validation",
        "",
        "This report validates survey products and schemas only.",
        "",
        "**No compact-feature distributions, cross-survey distances, "
        "or classifier outputs were calculated.**",
        "",
        "## Final gate",
        "",
        f"**{report['final_gate']}**",
        "",
        "## Product checks",
        "",
    ]

    product_rows: list[tuple[str, str]] = []
    for key, rec in report["checksums"].items():
        product_rows.append((key, rec["status"]))
    md_lines.append(markdown_table(product_rows))

    md_lines.extend(
        [
            "",
            "## Observation schema",
            "",
        ]
    )

    schema_rows = [
        (
            "SDSS builder schema",
            str(report["sdss"]["phot"]["builder_schema_compatible"]),
        ),
        (
            "DES builder schema",
            str(report["des"]["phot"]["builder_schema_compatible"]),
        ),
        (
            "SDSS bands",
            ", ".join(report["sdss"]["phot"].get("band_tokens", [])),
        ),
        (
            "DES bands",
            ", ".join(report["des"]["phot"].get("band_tokens", [])),
        ),
        (
            "SDSS band-token status",
            report["sdss"]["phot"].get("band_token_status", "UNKNOWN"),
        ),
        (
            "DES band-token status",
            report["des"]["phot"].get("band_token_status", "UNKNOWN"),
        ),
    ]

    md_lines.append(markdown_table(schema_rows))

    md_lines.extend(
        [
            "",
            "## Important boundary",
            "",
            "- Negative forced-photometry values were counted only to verify "
            "that the released measurement character is preserved.",
            "- No S/N-active windows were calculated.",
            "- No compact features were calculated.",
            "- No SDSS-versus-DES feature statistic was calculated.",
            "- No class-conditioned outcome was inspected.",
            "",
        ]
    )

    MD_REPORT.write_text("\n".join(md_lines), encoding="utf-8")
    JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"FINAL INPUT-VALIDATION GATE: {report['final_gate']}")
    print("=" * 72)

    print(f"\nJSON report: {JSON_REPORT}")
    print(f"Markdown report: {MD_REPORT}")

    if report["final_gate"] == "PASS":
        print(
            "\nNext permitted step: validate quality masks and reproduce the "
            "registered synthetic passband/cadence predictions."
        )
        return 0

    print(
        "\nDo not proceed to compact-feature comparison until the flagged "
        "schema/band-token issue is resolved."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
