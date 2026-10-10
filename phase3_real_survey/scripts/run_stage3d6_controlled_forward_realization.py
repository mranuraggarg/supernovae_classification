#!/usr/bin/env python3

"""
Phase 3 Stage 3D.6
Controlled end-to-end forward-observation realization.

VALIDATION ONLY — NOT PRODUCTION MONTE CARLO.

Connects:
  Hsiao07
    -> frozen survey responses
    -> deterministic SIMLIB selection
    -> diagnostic absolute FLUXCAL scale
    -> validated baseline variance
    -> validated survey uncertainty maps
    -> deterministic PCG64 stochastic realization
    -> frozen compact feature extractor

No Stage-1/Stage-2 outcomes are read.
No classifier is run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import numpy as np


ROOT = Path("phase3_real_survey")

OUTDIR = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d6_controlled_realization"
)

CONTROL_LOCK = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d6_controlled_realization_lock.json"
)

STAGE3C_LOCK = (
    ROOT
    / "preregistration"
    / "stage3c_execution_lock.json"
)

NOISE_LOCK = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d5_noise_branch_lock.json"
)

SCRIPTS = ROOT / "scripts"

REDSHIFT = 0.150
DES_SBMAG = 24.5
REALIZATION_ID = 0

PHASE_MIN = -20
PHASE_MAX = 85

BANDS = "griz"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampling = load_module(
    "stage3d_sampling",
    SCRIPTS / "validate_stage3d_sampling_and_seeds.py",
)

variance = load_module(
    "stage3d_variance",
    SCRIPTS / "validate_stage3d_baseline_variance.py",
)

uncertainty = load_module(
    "stage3d_uncertainty",
    SCRIPTS / "validate_stage3d_uncertainty_operators.py",
)

passband = load_module(
    "stage3d_passband",
    SCRIPTS / "reproduce_passband_predictions.py",
)

features_mod = load_module(
    "stage3d_features",
    SCRIPTS / "build_real_survey_features.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def parse_simlib(path: Path):
    """
    Parse complete SIMLIB LIBIDs needed by controlled Stage 3D.6.

    Returned observations retain the exact epoch-level instrumental
    quantities needed by the validated Stage-3D.4 variance operator.
    """

    global_settings = variance.read_global_simlib_settings(path)

    libids = {}

    current = None

    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.strip()

            m = re.match(r"^LIBID:\s*(\d+)", line)
            if m:
                current = {
                    "libid": int(m.group(1)),
                    "field": None,
                    "pixsize": None,
                    "template_zpt": None,
                    "template_skysig": None,
                    "obs": [],
                }

                libids[current["libid"]] = current
                continue

            if current is None:
                continue

            if line.startswith("FIELD:"):
                parts = line.split()
                if len(parts) >= 2:
                    current["field"] = parts[1]
                continue

            m_pix = re.search(
                r"\bPIXSIZE:\s*([0-9.eE+-]+)",
                line,
            )

            if m_pix:
                current["pixsize"] = float(m_pix.group(1))

            if line.startswith("TEMPLATE_ZPT:"):
                current["template_zpt"] = [
                    float(x)
                    for x in line.split(":", 1)[1].split()
                ]
                continue

            if line.startswith("TEMPLATE_SKYSIG:"):
                current["template_skysig"] = [
                    float(x)
                    for x in line.split(":", 1)[1].split()
                ]
                continue

            if line.startswith("S:"):
                p = line.split()

                if len(p) < 12:
                    raise RuntimeError(
                        f"Unexpected SIMLIB row:\n{line}"
                    )

                current["obs"].append(
                    {
                        "mjd": float(p[1]),
                        "idexpt": p[2],
                        "band": p[3],
                        "gain": float(p[4]),
                        "readnoise": float(p[5]),
                        "skysig": float(p[6]),
                        "psf1": float(p[7]),
                        "psf2": float(p[8]),
                        "psfratio": float(p[9]),
                        "zpt": float(p[10]),
                        "zpterr": float(p[11]),
                    }
                )

    for item in libids.values():
        if item["pixsize"] is None:
            item["pixsize"] = global_settings["pixsize_global"]

        item["psf_unit"] = global_settings["psf_unit"]

    return libids


def phase_flux_proxy(
    hsiao,
    response,
    *,
    phase_rest: float,
    redshift: float,
) -> float:
    """
    Interpolate the frozen daily Hsiao synthetic-photometry sequence
    to arbitrary rest-frame phase.
    """

    phases = np.arange(
        PHASE_MIN,
        PHASE_MAX + 1,
        dtype=float,
    )

    values = []

    rwave, rtrans = response

    for p in phases:
        wave, flux = hsiao[int(p)]

        values.append(
            passband.synthetic_flux_proxy(
                wave,
                flux,
                rwave,
                rtrans,
                redshift,
            )
        )

    values = np.asarray(values, dtype=float)

    if phase_rest < PHASE_MIN or phase_rest > PHASE_MAX:
        return 0.0

    return float(
        np.interp(
            phase_rest,
            phases,
            values,
        )
    )


def build_response_sets():
    """
    Resolve the already checksummed frozen passband artifacts.
    """

    sdss_kcor = passband.locate_frozen_artifact(
        "sdss_kcor"
    )

    hsiao_path = passband.locate_frozen_artifact(
        "hsiao"
    )

    sdss = {
        b: passband.load_sdss_response(
            sdss_kcor,
            b,
        )
        for b in BANDS
    }

    des = {
        b: passband.load_des_response(
            passband.locate_frozen_artifact(
                f"des_{b}"
            )
        )
        for b in BANDS
    }

    hsiao = passband.load_hsiao(
        hsiao_path
    )

    return hsiao, sdss, des, sdss_kcor, hsiao_path


def diagnostic_scale(hsiao, sdss_responses):
    """
    One global Stage-3D.6 diagnostic scale:

      SDSS-r dense noiseless peak at z=0.150 -> FLUXCAL 1000.

    The resulting multiplier is applied unchanged to all bands and
    both surveys.
    """

    vals = []

    for phase in range(
        PHASE_MIN,
        PHASE_MAX + 1,
    ):
        wave, flux = hsiao[phase]
        rwave, rtrans = sdss_responses["r"]

        vals.append(
            passband.synthetic_flux_proxy(
                wave,
                flux,
                rwave,
                rtrans,
                REDSHIFT,
            )
        )

    peak = float(np.nanmax(vals))

    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError(
            "Invalid SDSS-r Hsiao diagnostic peak."
        )

    return 1000.0 / peak, peak


def map_lookup(index, key, x):
    m = index[key]

    rows = uncertainty.validate_rows(
        m,
        2,
    )

    return uncertainty.snana_linear_clamped(
        x,
        rows,
    )


def apply_sdss_uncertainty(
    base_fluxerr: float,
    fluxcal_true: float,
    band: str,
    sdss_maps,
):
    """
    Validated SDSS FLUXERR_ADD + LOGSNR-dependent FLUXERR_SCALE.
    """

    # FLUXERR_ADD map coordinate is PSF but frozen values are
    # constant across the tabulated PSF range.
    add_map = sdss_maps[
        ("FLUXERR_ADD", None, band)
    ]

    add_rows = uncertainty.validate_rows(
        add_map,
        2,
    )

    erradd = float(add_rows[0, 1])

    err_after_add = math.sqrt(
        base_fluxerr**2
        + erradd**2
    )

    if err_after_add <= 0:
        raise RuntimeError(
            "Non-positive SDSS uncertainty."
        )

    snr = max(
        abs(fluxcal_true) / err_after_add,
        1.0e-30,
    )

    logsnr = math.log10(snr)

    errscale = map_lookup(
        sdss_maps,
        ("FLUXERR_SCALE", None, band),
        logsnr,
    )

    return (
        err_after_add * errscale,
        erradd,
        errscale,
        logsnr,
    )


def apply_des_uncertainty(
    base_fluxerr: float,
    *,
    field: str,
    band: str,
    des_maps,
):
    stratum = sampling.classify_stratum(
        "DES-SN5YR",
        field,
    )

    errscale = map_lookup(
        des_maps,
        ("FLUXERR_SCALE", stratum, band),
        DES_SBMAG,
    )

    return base_fluxerr * errscale, errscale


def template_value(
    values,
    band: str,
):
    if values is None:
        return None

    index = {
        "g": 0,
        "r": 1,
        "i": 2,
        "z": 3,
    }[band]

    if index >= len(values):
        raise RuntimeError(
            f"Template array missing {band}"
        )

    return float(values[index])


def generate_branch(
    *,
    survey: str,
    stratum: str,
    simlib_entry,
    hsiao,
    responses,
    flux_scale: float,
    sdss_maps,
    des_maps,
):
    observations = [
        x for x in simlib_entry["obs"]
        if x["band"] in BANDS
    ]

    if not observations:
        raise RuntimeError(
            f"No griz observations in {survey} "
            f"LIBID {simlib_entry['libid']}"
        )

    mjds = np.asarray(
        [x["mjd"] for x in observations],
        dtype=float,
    )

    # Controlled Stage-3D.6-only rule.
    peak_mjd = 0.5 * (
        float(np.min(mjds))
        + float(np.max(mjds))
    )

    seed = sampling.derive_seed(
        "phase3-stage3c-v1",
        "measurement-noise",
        survey,
        stratum,
        REDSHIFT,
        DES_SBMAG,
        REALIZATION_ID,
    )

    rng = np.random.Generator(
        np.random.PCG64(seed)
    )

    rows = []

    for obs in observations:
        band = obs["band"]

        phase_rest = (
            obs["mjd"] - peak_mjd
        ) / (1.0 + REDSHIFT)

        proxy = phase_flux_proxy(
            hsiao,
            responses[band],
            phase_rest=phase_rest,
            redshift=REDSHIFT,
        )

        flux_true = (
            proxy * flux_scale
            if np.isfinite(proxy)
            else 0.0
        )

        psf1, psf2, psfratio, nea = (
            variance.convert_psf_to_internal(
                obs["psf1"],
                obs["psf2"],
                obs["psfratio"],
                simlib_entry["pixsize"],
                simlib_entry["psf_unit"],
            )
        )

        include_zp = (
            survey == "DES-SN5YR"
        )

        include_template = (
            survey == "DES-SN5YR"
            and simlib_entry[
                "template_skysig"
            ] is not None
            and simlib_entry[
                "template_zpt"
            ] is not None
        )

        template_skysig = 0.0
        template_zpt = None

        if include_template:
            template_skysig = template_value(
                simlib_entry[
                    "template_skysig"
                ],
                band,
            )

            template_zpt = template_value(
                simlib_entry[
                    "template_zpt"
                ],
                band,
            )

        base = variance.baseline_variance(
            fluxcal=max(
                flux_true,
                0.0,
            ),
            zpt=obs["zpt"],
            gain=obs["gain"],
            skysig=obs["skysig"],
            readnoise=obs["readnoise"],
            nea=nea,
            zpterr=obs["zpterr"],
            include_zp=include_zp,
            template_skysig=template_skysig,
            template_readnoise=0.0,
            template_zpt=template_zpt,
            include_template=include_template,

            # Explicit Stage-3D.6 limitation.
            host_flux_pe=0.0,
        )

        ferr = base["fluxcalerr_in"]

        diagnostic = {}

        if survey == "SDSS-II":
            (
                ferr,
                erradd,
                errscale,
                logsnr,
            ) = apply_sdss_uncertainty(
                ferr,
                flux_true,
                band,
                sdss_maps,
            )

            diagnostic.update(
                {
                    "erradd": erradd,
                    "errscale": errscale,
                    "logsnr": logsnr,
                }
            )

        else:
            ferr, errscale = (
                apply_des_uncertainty(
                    ferr,
                    field=simlib_entry["field"],
                    band=band,
                    des_maps=des_maps,
                )
            )

            diagnostic.update(
                {
                    "errscale": errscale,
                }
            )

        if not np.isfinite(ferr) or ferr <= 0:
            raise RuntimeError(
                "Invalid final FLUXCALERR."
            )

        flux_obs = float(
            flux_true
            + rng.normal(
                loc=0.0,
                scale=ferr,
            )
        )

        rows.append(
            {
                "mjd": obs["mjd"],
                "band": band,
                "phase_rest": phase_rest,
                "flux_true": flux_true,
                "fluxcal": flux_obs,
                "fluxcalerr": ferr,
                "field": simlib_entry["field"],
                "libid": simlib_entry["libid"],
                "gain": obs["gain"],
                "skysig": obs["skysig"],
                "readnoise": obs["readnoise"],
                "psf1_internal": psf1,
                "psf2_internal": psf2,
                "psfratio": psfratio,
                "nea": nea,
                "zpt": obs["zpt"],
                "zpterr": obs["zpterr"],
                "baseline_fluxcalerr":
                    base["fluxcalerr_in"],
                "template_variance_pe2":
                    base["template_variance_pe2"],
                **diagnostic,
            }
        )

    feature_values, feature_audit = (
        features_mod.frozen_features(
            np.asarray(
                [r["mjd"] for r in rows]
            ),
            np.asarray(
                [r["band"] for r in rows],
                dtype=object,
            ),
            np.asarray(
                [r["fluxcal"] for r in rows]
            ),
            np.asarray(
                [r["fluxcalerr"] for r in rows]
            ),
        )
    )

    return {
        "survey": survey,
        "stratum": stratum,
        "field": simlib_entry["field"],
        "libid": simlib_entry["libid"],
        "peak_mjd": peak_mjd,
        "seed": seed,
        "n_epochs": len(rows),
        "epochs": rows,
        "features": feature_values,
        "feature_audit": feature_audit,
    }


def main():
    print("=" * 78)
    print("PHASE 3 STAGE 3D.6")
    print("CONTROLLED END-TO-END FORWARD-OBSERVATION REALIZATION")
    print("VALIDATION ONLY — NOT PRODUCTION MONTE CARLO")
    print("=" * 78)

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for p in [
        CONTROL_LOCK,
        STAGE3C_LOCK,
        NOISE_LOCK,
    ]:
        require_file(p)

    print("\n[1] Frozen control artifacts...")
    print("  control lock :", CONTROL_LOCK)
    print("  Stage 3C     :", STAGE3C_LOCK)
    print("  Stage 3D.5   :", NOISE_LOCK)

    sndata = os.environ.get(
        "SNDATA_ROOT"
    )

    if not sndata:
        raise RuntimeError(
            "SNDATA_ROOT is not defined."
        )

    sndata = Path(sndata)

    sdss_simlib_path = (
        sndata
        / "simlib/SDSS/SDSS_3year.SIMLIB"
    )

    des_simlib_path = (
        sndata
        / "simlib/DES/DES-SN5YR_DES.SIMLIB"
    )

    sdss_fluxerr_path = (
        sndata
        / "simlib/SDSS/SDSS_fluxErrModel.DAT"
    )

    des_fluxerr_path = (
        sndata
        / "simlib/DES/"
          "DES-SN5YR_DES_FLUXERRMODEL_SIM.DAT"
    )

    for p in [
        sdss_simlib_path,
        des_simlib_path,
        sdss_fluxerr_path,
        des_fluxerr_path,
    ]:
        require_file(p)

    print("\n[2] Loading frozen passbands and Hsiao07...")

    (
        hsiao,
        sdss_responses,
        des_responses,
        sdss_kcor,
        hsiao_path,
    ) = build_response_sets()

    print("  Hsiao :", hsiao_path)
    print("  SDSS  :", sdss_kcor)
    print("  bands :", BANDS)

    flux_scale, raw_r_peak = diagnostic_scale(
        hsiao,
        sdss_responses,
    )

    print("\n[3] Controlled diagnostic normalization...")
    print(
        f"  raw SDSS-r Hsiao peak : "
        f"{raw_r_peak:.12e}"
    )
    print(
        f"  global multiplier     : "
        f"{flux_scale:.12e}"
    )
    print(
        "  target SDSS-r peak    : "
        "1000 FLUXCAL"
    )

    print("\n[4] Parsing SIMLIBs...")

    sdss_simlib = parse_simlib(
        sdss_simlib_path
    )

    des_simlib = parse_simlib(
        des_simlib_path
    )

    sdss_entries = (
        sampling.parse_simlib_libids(
            sdss_simlib_path,
            "SDSS-II",
        )
    )

    des_entries = (
        sampling.parse_simlib_libids(
            des_simlib_path,
            "DES-SN5YR",
        )
    )

    sdss_groups = sampling.grouped_libids(
        sdss_entries,
        "SDSS-II",
    )

    des_groups = sampling.grouped_libids(
        des_entries,
        "DES-SN5YR",
    )

    print("\n[5] Deterministic LIBID selection...")

    selected = {}
    selection_audit = {}

    def first_four_band_complete(
        *,
        survey: str,
        stratum: str,
        groups: dict,
        simlib_data: dict,
    ) -> int:

        seq = sampling.deterministic_libid_sequence(
            groups[stratum],
            namespace="phase3-stage3c-v1",
            survey=survey,
            stratum=stratum,
            n_needed=20,
        )

        skipped = []

        for rank, libid in enumerate(seq):

            entry = simlib_data[libid]

            bands = sorted(
                {
                    obs["band"]
                    for obs in entry["obs"]
                    if obs["band"] in BANDS
                }
            )

            if set(bands) == set(BANDS):

                selection_audit[
                    (survey, stratum)
                ] = {
                    "selected_libid": libid,
                    "selected_rank": rank,
                    "selected_bands": bands,
                    "skipped": skipped,
                    "eligibility_rule": (
                        "first four-band-complete LIBID in the frozen "
                        "deterministic sequence; Stage-3D.6 validation only"
                    ),
                }

                return libid

            skipped.append(
                {
                    "rank": rank,
                    "libid": libid,
                    "bands": bands,
                    "reason": (
                        "missing one or more of g,r,i,z"
                    ),
                }
            )

        raise RuntimeError(
            "No four-band-complete LIBID found in first "
            f"20 deterministic candidates for "
            f"{survey} {stratum}."
        )

    for stratum in ["82N", "82S"]:

        selected[
            ("SDSS-II", stratum)
        ] = first_four_band_complete(
            survey="SDSS-II",
            stratum=stratum,
            groups=sdss_groups,
            simlib_data=sdss_simlib,
        )

    for stratum in ["SHALLOW", "DEEP"]:

        selected[
            ("DES-SN5YR", stratum)
        ] = first_four_band_complete(
            survey="DES-SN5YR",
            stratum=stratum,
            groups=des_groups,
            simlib_data=des_simlib,
        )

    for key, libid in selected.items():

        audit = selection_audit[key]

        print(
            f"  {key[0]:10s} "
            f"{key[1]:8s} -> LIBID {libid} "
            f"(rank {audit['selected_rank']}, "
            f"skipped {len(audit['skipped'])})"
        )

    print("\n[6] Loading validated uncertainty maps...")

    sdss_maps = uncertainty.map_index(
        uncertainty.parse_fluxerr_file(
            sdss_fluxerr_path
        )
    )

    des_maps = uncertainty.map_index(
        uncertainty.parse_fluxerr_file(
            des_fluxerr_path
        )
    )

    print("  SDSS maps : PASS")
    print("  DES maps  : PASS")

    print("\n[7] Executing four controlled branches...")

    results = []

    for stratum in ["82N", "82S"]:
        libid = selected[
            ("SDSS-II", stratum)
        ]

        result = generate_branch(
            survey="SDSS-II",
            stratum=stratum,
            simlib_entry=sdss_simlib[
                libid
            ],
            hsiao=hsiao,
            responses=sdss_responses,
            flux_scale=flux_scale,
            sdss_maps=sdss_maps,
            des_maps=des_maps,
        )

        results.append(result)

        print(
            f"  SDSS-II {stratum}: "
            f"LIBID={libid} "
            f"N={result['n_epochs']} "
            f"features="
            f"{'PASS' if result['features'] else 'FAIL'}"
        )

    for stratum in [
        "SHALLOW",
        "DEEP",
    ]:
        libid = selected[
            ("DES-SN5YR", stratum)
        ]

        result = generate_branch(
            survey="DES-SN5YR",
            stratum=stratum,
            simlib_entry=des_simlib[
                libid
            ],
            hsiao=hsiao,
            responses=des_responses,
            flux_scale=flux_scale,
            sdss_maps=sdss_maps,
            des_maps=des_maps,
        )

        results.append(result)

        print(
            f"  DES-SN5YR {stratum}: "
            f"LIBID={libid} "
            f"N={result['n_epochs']} "
            f"features="
            f"{'PASS' if result['features'] else 'FAIL'}"
        )

    payload = {
        "stage": "3D.6",
        "status":
            "CONTROLLED_VALIDATION_ONLY",
        "redshift": REDSHIFT,
        "des_sbmag": DES_SBMAG,
        "realization_id":
            REALIZATION_ID,
        "normalization": {
            "sdss_r_target_peak_fluxcal":
                1000.0,
            "global_multiplier":
                flux_scale,
        },
        "limitations": {
            "production_amplitude_not_frozen":
                True,
            "production_peak_mjd_rule_not_frozen":
                True,
            "des_host_photon_poisson":
                False,
        },
        "libid_selection_audit": {
            f"{survey}|{stratum}": audit
            for (survey, stratum), audit
            in selection_audit.items()
        },
        "branches": results,
    }

    out = OUTDIR / "stage3d6_result.json"

    out.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    print("\n[8] Output...")
    print("  wrote :", out)
    print("  SHA256:", sha256(out))

    failed = [
        r
        for r in results
        if r["features"] is None
    ]

    if failed:
        print("\n" + "=" * 78)
        print(
            "STAGE 3D.6 CONTROLLED REALIZATION: FAIL"
        )
        print("=" * 78)

        for r in failed:
            print(
                r["survey"],
                r["stratum"],
                r["feature_audit"],
            )

        raise SystemExit(1)

    print("\n" + "=" * 78)
    print(
        "STAGE 3D.6 CONTROLLED REALIZATION: PASS"
    )
    print("=" * 78)

    print(
        "\nThis PASS validates integration only."
    )
    print(
        "It does NOT authorize the production "
        "Stage-3C Monte Carlo."
    )


if __name__ == "__main__":
    main()
