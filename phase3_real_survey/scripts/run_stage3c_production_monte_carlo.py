#!/usr/bin/env python3

"""
Phase 3 Stage 3C production Monte Carlo.

PRODUCTION FORWARD-MODEL EXECUTION ONLY.

Uses the frozen, authorized Stage-3C design:
  Hsiao07
    -> frozen survey responses
    -> deterministic SIMLIB selection
    -> frozen common FLUXCAL normalization
    -> frozen paired peak-anchor nuisance
    -> validated baseline variance
    -> validated survey uncertainty maps
    -> deterministic PCG64 stochastic realization
    -> frozen compact feature extractor

No Stage-1/Stage-2 outcomes are read.
No classifier is run.
No operator parameter may be tuned from production outputs.
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
    / "stage3c_production_monte_carlo"
)

PREPRODUCTION_FREEZE = (
    ROOT
    / "preregistration"
    / "stage3c_preproduction_freeze_2026-10-10.json"
)

PRODUCTION_AUTHORIZATION = (
    ROOT
    / "preregistration"
    / "stage3c_production_authorization_2026-10-10.json"
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

NORMALIZATION_REFERENCE_REDSHIFT = 0.150

REDSHIFT_GRID = np.array(
    [0.050, 0.075, 0.100, 0.125, 0.150, 0.175,
     0.200, 0.225, 0.250, 0.275, 0.300],
    dtype=float,
)

DES_SBMAG_GRID = np.array(
    [20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5],
    dtype=float,
)

N_REALIZATIONS = 500
REALIZATION_IDS = range(N_REALIZATIONS)

# Stage-3D.7 native-SNANA validated controlled DES host.
# This grid is host-specific and is used only for the controlled
# validation realization, not as a general production HOSTLIB model.
DES_HOST_GALMAG_GRID = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d7_production_semantics"
    / "stage3d7_native_galmag_grid.txt"
)


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
                NORMALIZATION_REFERENCE_REDSHIFT,
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


def load_des_host_galmag_grid():
    """
    Load the Stage-3D.7 native-SNANA validated GALMAG-vs-PSF grid
    for the single controlled DES host.

    The artifact contains:
        index psfsig g r i z

    It is intentionally a validation fixture, not a general HOSTLIB
    replacement.
    """

    require_file(DES_HOST_GALMAG_GRID)

    data = np.genfromtxt(
        DES_HOST_GALMAG_GRID,
        names=True,
        dtype=None,
        encoding=None,
    )

    psf_grid = np.asarray(
        data["psfsig"],
        dtype=float,
    )

    galmag_grid = {
        band: np.asarray(
            data[band],
            dtype=float,
        )
        for band in BANDS
    }

    if psf_grid.ndim != 1 or psf_grid.size < 2:
        raise RuntimeError(
            "Invalid controlled DES host PSF grid."
        )

    if not np.all(np.diff(psf_grid) > 0.0):
        raise RuntimeError(
            "Controlled DES host PSF grid is not increasing."
        )

    for band in BANDS:
        if galmag_grid[band].shape != psf_grid.shape:
            raise RuntimeError(
                f"Controlled DES GALMAG grid mismatch for {band}."
            )

    return psf_grid, galmag_grid


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
    des_sbmag: float,
    des_maps,):
    stratum = sampling.classify_stratum(
        "DES-SN5YR",
        field,
    )

    errscale = map_lookup(
        des_maps,
        ("FLUXERR_SCALE", stratum, band),
        des_sbmag,
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
    redshift: float,
    des_sbmag: float,
    realization_id: int,
    peak_mjd: float,
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

    seed = sampling.derive_seed(
        "phase3-stage3c-v1",
        "measurement-noise",
        survey,
        stratum,
        redshift,
        des_sbmag,
        realization_id,
    )

    rng = np.random.Generator(
        np.random.PCG64(seed)
    )

    rows = []

    des_host_psf_grid = None
    des_host_galmag_grid = None

    if survey == "DES-SN5YR":
        (
            des_host_psf_grid,
            des_host_galmag_grid,
        ) = load_des_host_galmag_grid()

    for obs in observations:
        band = obs["band"]

        phase_rest = (
            obs["mjd"] - peak_mjd
        ) / (1.0 + redshift)

        proxy = phase_flux_proxy(
            hsiao,
            responses[band],
            phase_rest=phase_rest,
            redshift=redshift,
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

        host_flux_pe = 0.0
        galmag_nea = None
        psfsig_arcsec = None

        if survey == "DES-SN5YR":
            psfsig_arcsec = (
                variance.psfsig_arcsec_from_nea(
                    nea,
                    simlib_entry["pixsize"],
                )
            )

            galmag_nea = (
                variance.interp_galmag_hostlib(
                    psfsig_arcsec,
                    des_host_psf_grid,
                    des_host_galmag_grid[band],
                )
            )

            host_flux_pe = (
                variance.host_variance_pe2_from_galmag(
                    galmag_nea=galmag_nea,
                    zpt=obs["zpt"],
                    gain=obs["gain"],
                )
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

            # Stage-3D.7 validated native-SNANA host-photon term.
            # SDSS remains zero in this controlled realization.
            host_flux_pe=host_flux_pe,
        )

        ferr = base["fluxcalerr_in"]

        diagnostic = {}

        if survey == "DES-SN5YR":
            diagnostic.update(
                {
                    "psfsig_arcsec": psfsig_arcsec,
                    "galmag_nea": galmag_nea,
                    "host_variance_pe2": host_flux_pe,
                }
            )

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
                    des_sbmag=des_sbmag,
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



def anchor_eligible_libids(libids, simlib):
    """Apply the frozen Stage 3C peak-anchor eligibility amendment."""
    eligible = []

    for libid in sorted(int(x) for x in libids):
        try:
            candidate_peak_anchors(simlib[libid])
        except RuntimeError:
            continue
        eligible.append(libid)

    return eligible


def derive_peak_anchor_seed(redshift: float, realization_id: int) -> int:
    """
    Frozen Stage-3C intrinsic peak-placement seed.

    Deliberately excludes survey, stratum, and SBMAG so that the same
    intrinsic latent placement variable is paired across survey branches.
    """
    key = (
        f"phase3-stage3c-v1|peak_anchor|"
        f"{redshift:.3f}|{int(realization_id)}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def peak_anchor_quantile(redshift: float, realization_id: int) -> float:
    seed = derive_peak_anchor_seed(redshift, realization_id)
    rng = np.random.Generator(np.random.PCG64(seed))
    return float(rng.random())


def candidate_peak_anchors(simlib_entry) -> np.ndarray:
    """
    Ordered candidate peak MJDs using the frozen cadence-injection rule:
    split seasons at gaps >45 d; within each season retain positive gaps
    <=30 d; place candidates at fractions 0.25, 0.50, 0.75 of each gap.
    """
    times = np.array(
        sorted(
            float(obs["mjd"])
            for obs in simlib_entry["obs"]
            if obs["band"] in BANDS
        ),
        dtype=float,
    )

    if times.size < 2:
        raise RuntimeError("SIMLIB entry has fewer than two usable epochs")

    breaks = np.where(np.diff(times) > 45.0)[0] + 1
    seasons = np.split(times, breaks)

    candidates = []

    for season_times in seasons:
        if season_times.size < 2:
            continue

        gaps = np.diff(season_times)
        valid = (
            (gaps > 0.0)
            & (gaps <= 30.0)
        )

        left = season_times[:-1][valid]
        gap = gaps[valid]

        for left_mjd, gap_days in zip(left, gap):
            for fraction in (0.25, 0.50, 0.75):
                candidates.append(
                    float(left_mjd + gap_days * fraction)
                )

    if not candidates:
        raise RuntimeError(
            "SIMLIB entry has no eligible frozen peak-anchor candidates"
        )

    return np.asarray(candidates, dtype=float)


def map_peak_anchor_quantile(simlib_entry, quantile: float) -> tuple[float, int, int]:
    candidates = candidate_peak_anchors(simlib_entry)

    n_candidates = int(candidates.size)
    index = min(
        int(np.floor(float(quantile) * n_candidates)),
        n_candidates - 1,
    )

    return float(candidates[index]), index, n_candidates


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Frozen Stage-3C production Monte Carlo"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate production setup without generating realizations.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(
            "Stop after writing this many new branch records. "
            "For smoke testing only."
        ),
    )
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTDIR / "stage3c_production_branches.jsonl"
    manifest_path = OUTDIR / "stage3c_production_manifest.json"

    amendment_path = (
        ROOT
        / "preregistration"
        / "stage3c_peak_anchor_eligibility_amendment_2026-10-10.json"
    )

    reauthorization_path = (
        ROOT
        / "preregistration"
        / "stage3c_peak_anchor_eligibility_reauthorization_2026-10-10.json"
    )

    required_provenance = [
        PREPRODUCTION_FREEZE,
        PRODUCTION_AUTHORIZATION,
        amendment_path,
        reauthorization_path,
        STAGE3C_LOCK,
        NOISE_LOCK,
    ]

    for path in required_provenance:
        if not path.is_file():
            raise RuntimeError(
                f"Required production provenance artifact missing: {path}"
            )

    prefreeze = json.loads(PREPRODUCTION_FREEZE.read_text())
    authorization = json.loads(PRODUCTION_AUTHORIZATION.read_text())
    amendment = json.loads(amendment_path.read_text())
    reauthorization = json.loads(reauthorization_path.read_text())

    if prefreeze["amplitude_normalization"]["status"] != "FROZEN":
        raise RuntimeError("Amplitude normalization is not frozen")

    if prefreeze["peak_epoch_rule"]["status"] != "FROZEN":
        raise RuntimeError("Peak-epoch rule is not frozen")

    if authorization.get("production_authorized") is not True:
        raise RuntimeError("Original Stage-3C production authorization missing")

    if reauthorization.get("production_authorized") is not True:
        raise RuntimeError(
            "Peak-anchor eligibility amendment is not reauthorized"
        )

    amendment_sha = hashlib.sha256(
        amendment_path.read_bytes()
    ).hexdigest()

    registered_amendment_sha = (
        reauthorization["amendment"]["sha256"]
    )

    if amendment_sha != registered_amendment_sha:
        raise RuntimeError(
            "Peak-anchor eligibility amendment hash mismatch"
        )

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

    sdss_maps = uncertainty.map_index(
        uncertainty.parse_fluxerr_file(sdss_fluxerr_path)
    )

    des_maps = uncertainty.map_index(
        uncertainty.parse_fluxerr_file(des_fluxerr_path)
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

    print("\n[3] Frozen Stage-3C amplitude normalization...")
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

    print("\n[5] Frozen deterministic LIBID sequences...")

    selected = {}
    selection_audit = {}

    # ------------------------------------------------------------------
    # Frozen amended anchor-eligible pools
    # ------------------------------------------------------------------

    sdss_eligible = {
        stratum: anchor_eligible_libids(
            sdss_groups[stratum],
            sdss_simlib,
        )
        for stratum in ("82N", "82S")
    }

    des_eligible = {
        stratum: anchor_eligible_libids(
            des_groups[stratum],
            des_simlib,
        )
        for stratum in ("SHALLOW", "DEEP")
    }

    expected_pool_hashes = {
        ("SDSS-II", "82N"):
            "8039aaf6f2c0618be14072b13a9d3d6ef946efd3797c6ec587d122c09b12c5d0",
        ("SDSS-II", "82S"):
            "bf701210223499194b75656fa504b76cad38f99e64779992a87f5340422a44dc",
        ("DES-SN5YR", "SHALLOW"):
            "b6338c2bc076e5c357cdd772274c65f08fed5babd201191bd2f3c564dab9c855",
        ("DES-SN5YR", "DEEP"):
            "15caef11db419d496ec8e8a457d4a4ee170ac696c2584feca8b1b420b0e016d7",
    }

    pools = {
        ("SDSS-II", "82N"): sdss_eligible["82N"],
        ("SDSS-II", "82S"): sdss_eligible["82S"],
        ("DES-SN5YR", "SHALLOW"): des_eligible["SHALLOW"],
        ("DES-SN5YR", "DEEP"): des_eligible["DEEP"],
    }

    for key, pool in pools.items():
        payload = json.dumps(
            [int(x) for x in pool],
            separators=(",", ":"),
        ).encode("utf-8")

        actual = hashlib.sha256(payload).hexdigest()
        expected = expected_pool_hashes[key]

        if actual != expected:
            raise RuntimeError(
                f"Eligible-pool hash mismatch for {key}: "
                f"{actual} != {expected}"
            )

    # ------------------------------------------------------------------
    # Frozen deterministic 500-realization SIMLIB sequences
    # ------------------------------------------------------------------

    sequences = {}

    for survey, stratum in (
        ("SDSS-II", "82N"),
        ("SDSS-II", "82S"),
        ("DES-SN5YR", "SHALLOW"),
        ("DES-SN5YR", "DEEP"),
    ):
        sequences[(survey, stratum)] = (
            sampling.deterministic_libid_sequence(
                pools[(survey, stratum)],
                namespace="phase3-stage3c-v1",
                survey=survey,
                stratum=stratum,
                n_needed=N_REALIZATIONS,
            )
        )

        if len(sequences[(survey, stratum)]) != N_REALIZATIONS:
            raise RuntimeError(
                f"Wrong deterministic sequence length for "
                f"{survey}/{stratum}"
            )

    # ------------------------------------------------------------------
    # Complete peak-anchor feasibility check before production
    # ------------------------------------------------------------------

    for survey, stratum in (
        ("SDSS-II", "82N"),
        ("SDSS-II", "82S"),
        ("DES-SN5YR", "SHALLOW"),
        ("DES-SN5YR", "DEEP"),
    ):
        simlib = (
            sdss_simlib
            if survey == "SDSS-II"
            else des_simlib
        )

        for libid in sequences[(survey, stratum)]:
            anchors = candidate_peak_anchors(
                simlib[int(libid)]
            )

            if anchors.size < 1:
                raise RuntimeError(
                    f"Zero-anchor production LIBID: "
                    f"{survey}/{stratum}/{libid}"
                )

    feature_names = [
        "z_peak_flux",
        "r_mean_flux",
        "peak_color_g_minus_r",
        "i_peak_flux",
        "peak_color_r_minus_i",
        "peak_color_i_minus_z",
        "g_mean_flux",
        "r_peak_flux",
        "z_std_flux",
        "i_amplitude",
        "i_std_flux",
        "time_span",
        "z_time_of_peak",
        "i_time_of_peak",
        "r_time_of_peak",
        "r_std_flux",
    ]

    expected_total = (
        len(REDSHIFT_GRID)
        * len(DES_SBMAG_GRID)
        * N_REALIZATIONS
        * 4
    )

    if expected_total != 176000:
        raise RuntimeError(
            f"Unexpected production branch count: {expected_total}"
        )

    def script_sha256():
        return hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()

    manifest = {
        "stage": "Stage 3C production Monte Carlo",
        "status": "PREFLIGHT_PASS",
        "production_authorized": True,
        "classifier_run": False,
        "stage1_stage2_outcomes_read": False,
        "expected_branch_records": expected_total,
        "redshift_grid": [
            float(x) for x in REDSHIFT_GRID
        ],
        "sbmag_grid": [
            float(x) for x in DES_SBMAG_GRID
        ],
        "realizations_per_condition": N_REALIZATIONS,
        "survey_strata": {
            "SDSS-II": ["82N", "82S"],
            "DES-SN5YR": ["SHALLOW", "DEEP"],
        },
        "feature_names": feature_names,
        "normalization_reference_redshift":
            NORMALIZATION_REFERENCE_REDSHIFT,
        "amendment_sha256": amendment_sha,
        "reauthorization_sha256": hashlib.sha256(
            reauthorization_path.read_bytes()
        ).hexdigest(),
        "runner_sha256": script_sha256(),
        "output_file": str(output_path),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print("===== STAGE 3C PRODUCTION PREFLIGHT =====")
    print("Expected branch records:", expected_total)
    print("Eligible pools:")
    for key in (
        ("SDSS-II", "82N"),
        ("SDSS-II", "82S"),
        ("DES-SN5YR", "SHALLOW"),
        ("DES-SN5YR", "DEEP"),
    ):
        print(
            f"  {key[0]:10s} {key[1]:8s}: "
            f"{len(pools[key])}"
        )
    print("Production authorization: PASS")
    print("Amended eligibility: PASS")
    print("Peak-anchor feasibility: PASS")
    print("Classifier: NOT RUN")

    if args.preflight_only:
        print("STAGE 3C PREFLIGHT ONLY: PASS")
        return

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    completed = set()

    if output_path.is_file():
        with output_path.open("r") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)

                key = (
                    row["survey"],
                    row["stratum"],
                    f'{float(row["redshift"]):.3f}',
                    f'{float(row["sbmag"]):.1f}',
                    int(row["realization_id"]),
                )

                if key in completed:
                    raise RuntimeError(
                        f"Duplicate existing production key "
                        f"at line {line_number}: {key}"
                    )

                completed.add(key)

        print(
            "Existing completed branch records:",
            len(completed),
        )

    new_records = 0
    total_seen = 0

    branch_specs = (
        ("SDSS-II", "82N"),
        ("SDSS-II", "82S"),
        ("DES-SN5YR", "SHALLOW"),
        ("DES-SN5YR", "DEEP"),
    )

    manifest["status"] = "IN_PROGRESS"
    manifest["existing_records_at_start"] = len(completed)

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    with output_path.open("a", buffering=1) as fout:

        for redshift_value in REDSHIFT_GRID:
            z = float(redshift_value)

            for sbmag_value in DES_SBMAG_GRID:
                sbmag = float(sbmag_value)

                for realization_id in REALIZATION_IDS:

                    latent_q = peak_anchor_quantile(
                        z,
                        realization_id,
                    )

                    for survey, stratum in branch_specs:

                        total_seen += 1

                        key = (
                            survey,
                            stratum,
                            f"{z:.3f}",
                            f"{sbmag:.1f}",
                            int(realization_id),
                        )

                        if key in completed:
                            continue

                        libid = int(
                            sequences[(survey, stratum)][
                                realization_id
                            ]
                        )

                        if survey == "SDSS-II":
                            simlib_entry = sdss_simlib[libid]
                            responses = sdss_responses
                        else:
                            simlib_entry = des_simlib[libid]
                            responses = des_responses

                        (
                            peak_mjd,
                            peak_anchor_index,
                            n_peak_anchor_candidates,
                        ) = map_peak_anchor_quantile(
                            simlib_entry,
                            latent_q,
                        )

                        result = generate_branch(
                            survey=survey,
                            stratum=stratum,
                            simlib_entry=simlib_entry,
                            hsiao=hsiao,
                            responses=responses,
                            flux_scale=flux_scale,
                            redshift=z,
                            des_sbmag=sbmag,
                            realization_id=realization_id,
                            peak_mjd=peak_mjd,
                            sdss_maps=sdss_maps,
                            des_maps=des_maps,
                        )

                        if result["features"] is None:
                            feature_status = "rejected"
                            feature_rejection_reason = (
                                result["feature_audit"].get(
                                    "reason",
                                    "unknown",
                                )
                            )
                            features = None
                        else:
                            feature_status = "accepted"
                            feature_rejection_reason = None
                            features = {}

                            for name in feature_names:
                                value = result["features"][name]

                                if value is None:
                                    features[name] = None
                                else:
                                    value = float(value)
                                    features[name] = (
                                        value
                                        if np.isfinite(value)
                                        else None
                                    )

                        row = {
                            "survey": survey,
                            "stratum": stratum,
                            "redshift": z,
                            "sbmag": sbmag,
                            "realization_id":
                                int(realization_id),
                            "libid": int(result["libid"]),
                            "field": result["field"],
                            "peak_anchor_quantile":
                                float(latent_q),
                            "peak_anchor_index":
                                int(peak_anchor_index),
                            "n_peak_anchor_candidates":
                                int(n_peak_anchor_candidates),
                            "peak_mjd":
                                float(result["peak_mjd"]),
                            "measurement_noise_seed":
                                int(result["seed"]),
                            "n_epochs":
                                int(result["n_epochs"]),
                            "feature_status":
                                feature_status,
                            "feature_rejection_reason":
                                feature_rejection_reason,
                            "feature_audit":
                                result["feature_audit"],
                            "features": features,
                        }

                        fout.write(
                            json.dumps(
                                row,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                        completed.add(key)
                        new_records += 1

                        if (
                            new_records % 1000 == 0
                            or new_records == 1
                        ):
                            print(
                                "new_records=",
                                new_records,
                                " total_completed=",
                                len(completed),
                                "/",
                                expected_total,
                                " current=",
                                key,
                            )

                        if (
                            args.max_records is not None
                            and new_records >= args.max_records
                        ):
                            manifest["status"] = (
                                "SMOKE_TEST_STOP"
                            )
                            manifest[
                                "completed_branch_records"
                            ] = len(completed)

                            manifest_path.write_text(
                                json.dumps(
                                    manifest,
                                    indent=2,
                                    sort_keys=True,
                                )
                                + "\n"
                            )

                            print(
                                "Stopped after --max-records",
                                args.max_records,
                            )
                            return

    if len(completed) != expected_total:
        raise RuntimeError(
            "Production loop ended with unexpected "
            f"record count: {len(completed)} "
            f"!= {expected_total}"
        )

    output_sha = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    manifest["status"] = "COMPLETE"
    manifest["completed_branch_records"] = len(completed)
    manifest["production_output_sha256"] = output_sha

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    manifest_sha = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    (OUTDIR / "stage3c_production_manifest.json.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n"
    )

    (OUTDIR / "stage3c_production_branches.jsonl.sha256").write_text(
        f"{output_sha}  {output_path.name}\n"
    )

    print()
    print("===== STAGE 3C PRODUCTION COMPLETE =====")
    print("Branch records:", len(completed))
    print("Output SHA256:", output_sha)
    print("Manifest SHA256:", manifest_sha)
    print("Classifier: NOT RUN")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
