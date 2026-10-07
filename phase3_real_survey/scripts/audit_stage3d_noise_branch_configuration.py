#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path("phase3_real_survey")
OUTDIR = ROOT / "results" / "stage3d_operator_validation"

SNANA_COMMIT = "4113779f038eec613b6684d0d46a47019d7c9624"

DES_INPUT = (
    ROOT
    / "data/raw/des/DES-SN5YR-1.2/7_PIPPIN_FILES/"
      "base_files/sim/sim_des5yr_survey.input"
)

EXPECTED = {
    "sdss_simlib": {
        "rel": "simlib/SDSS/SDSS_3year.SIMLIB",
        "sha256":
        "fad8558b04c2fc10e26d18f42d016ab52311078ea87d8cfb44f1c47e3fe7c6c9",
    },
    "sdss_fluxerr": {
        "rel": "simlib/SDSS/SDSS_fluxErrModel.DAT",
        "sha256":
        "a69b5a69fbbcaf54e6da2f6687c2020b97973c392000103553eb59674b9e05e7",
    },
    "des_simlib": {
        "rel": "simlib/DES/DES-SN5YR_DES.SIMLIB",
        "sha256":
        "f5575ecd5b50c526ff63b758c5f6b88a710a185040dfd3c6e5008ca7b67d92a0",
    },
    "des_fluxerr": {
        "rel": "simlib/DES/DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT",
        "sha256":
        "2b6d0898fd1992a72cfa2322a79272a9557e8fc4cb98fa3193bd8ee00d2d80d9",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_key(path: Path, key: str) -> int:
    text = path.read_text(errors="replace")
    return len(re.findall(rf"\b{re.escape(key)}\s*:", text))


def extract_scalar(path: Path, key: str):
    text = path.read_text(errors="replace")

    m = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*:\s*([^\s#]+)",
        text,
    )

    return m.group(1) if m else None


def contains_active_key(path: Path, key: str) -> bool:
    """
    True only for a non-commented configuration line.
    """
    pat = re.compile(
        rf"^\s*{re.escape(key)}\s*:",
        re.MULTILINE,
    )

    return bool(pat.search(path.read_text(errors="replace")))


def main():
    print("=" * 78)
    print("PHASE 3 STAGE 3D.5")
    print("SURVEY NOISE-BRANCH CONFIGURATION AUDIT")
    print("NO PHOTOMETRIC REALIZATIONS — NO FEATURE OUTCOMES — NO CLASSIFIER")
    print("=" * 78)

    sndata = os.environ.get("SNDATA_ROOT")

    if not sndata:
        raise RuntimeError("SNDATA_ROOT is not defined.")

    sndata = Path(sndata)

    resolved = {}

    print("\n[1] Verifying frozen survey artifacts...")

    for name, info in EXPECTED.items():
        path = sndata / info["rel"]

        if not path.is_file():
            raise RuntimeError(f"Missing artifact: {path}")

        observed = sha256(path)

        print(f"  {name}")
        print(f"    expected : {info['sha256']}")
        print(f"    observed : {observed}")

        if observed != info["sha256"]:
            raise RuntimeError(f"Hash mismatch for {name}")

        print("    status   : PASS")

        resolved[name] = path

    print("\n[2] Source-level SNANA defaults...")

    defaults = {
        "SMEARFLAG_FLUX": 1,
        "SMEARFLAG_ZEROPT": 0,
        "SMEARFLAG_HOSTGAL": 0,
    }

    print("  SNANA commit        :", SNANA_COMMIT)

    for key, value in defaults.items():
        print(f"  {key:20s}: {value}")

    print("\n[3] DES nominal configuration...")

    if not DES_INPUT.is_file():
        raise RuntimeError(
            f"DES nominal simulation input missing: {DES_INPUT}"
        )

    des_simlib_cfg = extract_scalar(DES_INPUT, "SIMLIB_FILE")
    des_fluxerr_cfg = extract_scalar(DES_INPUT, "FLUXERRMODEL_FILE")
    des_zp_cfg = extract_scalar(DES_INPUT, "SMEARFLAG_ZEROPT")

    print("  SIMLIB_FILE        :", des_simlib_cfg)
    print("  FLUXERRMODEL_FILE  :", des_fluxerr_cfg)
    print("  SMEARFLAG_ZEROPT   :", des_zp_cfg)

    if des_zp_cfg != "3":
        raise RuntimeError(
            "Nominal DES configuration does not have "
            "SMEARFLAG_ZEROPT=3."
        )

    des_hostnoise = contains_active_key(
        DES_INPUT,
        "HOSTNOISE_FILE",
    )

    print("  HOSTNOISE_FILE active:", des_hostnoise)

    if des_hostnoise:
        raise RuntimeError(
            "Unexpected active DES HOSTNOISE_FILE with "
            "FLUXERRMODEL_FILE."
        )

    print("\n[4] SIMLIB template-noise state...")

    template_state = {}

    for survey, path in [
        ("SDSS-II", resolved["sdss_simlib"]),
        ("DES-SN5YR", resolved["des_simlib"]),
    ]:
        counts = {
            key: count_key(path, key)
            for key in [
                "TEMPLATE_ZPT",
                "TEMPLATE_SKYSIG",
                "TEMPLATE_CCDSIG",
                "TEMPLATE_READNOISE",
            ]
        }

        template_enabled = (
            counts["TEMPLATE_ZPT"] > 0
            or counts["TEMPLATE_SKYSIG"] > 0
            or counts["TEMPLATE_CCDSIG"] > 0
            or counts["TEMPLATE_READNOISE"] > 0
        )

        template_state[survey] = {
            "counts": counts,
            "enabled_by_simlib": template_enabled,
        }

        print(f"\n  {survey}")

        for key, n in counts.items():
            print(f"    {key:22s}: {n}")

        print(
            "    template noise enabled:",
            template_enabled,
        )

    print("\n[5] Host-noise branch interpretation...")

    print(
        "  Legacy HOSTNOISE_FILE image-noise branch: "
        "OFF for nominal DES input."
    )

    print(
        "  DES host-SB uncertainty scaling is supplied by "
        "DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT."
    )

    print(
        "  DES host photon Poisson noise is separately ON because "
        "the nominal survey input sets HOSTLIB_MSKOPT=258 and "
        "bit 2 enables Poisson noise."
    )

    print("\n[6] SDSS zeropoint branch provenance...")

    print(
        "  No exact nominal simulation input tied to "
        "SDSS_3year.SIMLIB has yet been identified that explicitly "
        "sets SMEARFLAG_ZEROPT."
    )

    print(
        "  Therefore the source-defined default is the only directly "
        "pinned setting for this operator unless stronger provenance "
        "is found."
    )

    sdss_zp = defaults["SMEARFLAG_ZEROPT"]

    print("  frozen SDSS SMEARFLAG_ZEROPT candidate:", sdss_zp)

    print("\n[7] Constructing configuration lock...")

    lock = {
        "stage": "3D.5",
        "status": "FROZEN_NOISE_BRANCH_CONFIGURATION",
        "snana_source_commit": SNANA_COMMIT,
        "scientific_boundary": {
            "real_feature_outcomes_read": False,
            "classifier_results_read": False,
            "parameters_tuned_to_stage1_or_stage2": False,
        },
        "SDSS-II": {
            "simlib": "SDSS_3year.SIMLIB",
            "measurement_smearing": True,
            "zeropoint_smearing": {
                "enabled": False,
                "smearflag": 0,
                "basis":
                    "SNANA pinned-source default; no explicit "
                    "SDSS_3year nominal override identified",
            },
            "template_noise": {
                "enabled":
                    template_state["SDSS-II"]["enabled_by_simlib"],
                "basis": "SIMLIB TEMPLATE_* directives",
            },
            "legacy_hostnoise_file": False,
            "host_photon_noise": {
                "enabled": False,
                "basis":
                    "SNANA default SMEARFLAG_HOSTGAL=0 and no "
                    "host-library activation frozen for Stage 3C",
            },
            "fluxerr_model":
                "SDSS_fluxErrModel.DAT",
        },
        "DES-SN5YR": {
            "simlib": "DES-SN5YR_DES.SIMLIB",
            "measurement_smearing": True,
            "zeropoint_smearing": {
                "enabled": True,
                "smearflag": 3,
                "apply_to_true_error": True,
                "report_in_flux_error": True,
                "basis":
                    "nominal DES-SN5YR sim_des5yr_survey.input",
            },
            "template_noise": {
                "enabled":
                    template_state["DES-SN5YR"]["enabled_by_simlib"],
                "basis": "SIMLIB TEMPLATE_* directives",
            },
            "legacy_hostnoise_file": False,
            "host_photon_noise": {
                "enabled": True,
                "basis":
                    "nominal DES-SN5YR sim_des5yr_survey.input "
                    "sets HOSTLIB_MSKOPT=258; bit 2 enables "
                    "host-galaxy Poisson noise",
            },
            "fluxerr_model":
                "DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT",
        },
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)

    out = OUTDIR / "stage3d5_noise_branch_lock.json"

    out.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n"
    )

    print("  wrote:", out)

    print("\n" + "=" * 78)
    print("STAGE 3D.5 NOISE-BRANCH CONFIGURATION: PASS")
    print("=" * 78)

    print()
    print("No stochastic photometry was generated.")
    print("No Stage-1/Stage-2 feature outcome was read.")
    print("No classifier was run.")


if __name__ == "__main__":
    main()
