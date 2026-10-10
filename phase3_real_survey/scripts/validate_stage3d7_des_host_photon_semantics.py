#!/usr/bin/env python3

"""
Phase 3 Stage 3D.7
DES host-photon / SBMAG source-semantics validator.

Purpose
-------
Validate the source-faithful downstream SNANA chain

    GALMAG(PSF grid)
      -> SBMAG
      -> GALMAG over epoch NEA
      -> host photon variance
      -> baseline FLUXCAL uncertainty
      -> DES SBMAG-dependent FLUXERRMODEL

This validator intentionally starts from a supplied GALMAG grid.

It does NOT yet independently reproduce GEN_SNHOST_GALMAG's numerical
Sersic/Gauss2d_Overlap integration. That upstream operator is validated
separately before production integration.

No Stage-1/Stage-2 results are read.
No classifier results are read.
No parameters are tuned to feature outcomes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("phase3_real_survey")
SCRIPTS = ROOT / "scripts"

OUTDIR = (
    ROOT
    / "results"
    / "stage3d_operator_validation"
    / "stage3d7_production_semantics"
)

ZEROPOINT_FLUXCAL = 27.5
TWOPI = 2.0 * math.pi

# Exact fixed PSF-sigma grid from SNANA sntools_host.c.
# Units: arcsec.
PSF_GRID_ARCSEC = np.asarray(
    [
        0.03 / 2.35,
        0.07 / 2.35,
        0.10 / 2.35,
        0.20 / 2.35,
        0.40 / 2.35,
        0.80 / 2.35,
        1.30 / 2.35,
        2.10 / 2.35,
    ],
    dtype=float,
)

# Final point is PSFMAX_SNANA/2.35 and is supplied from the command
# line because we want to avoid silently assuming its value.


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


variance = load_module(
    "stage3d_variance",
    SCRIPTS / "validate_stage3d_baseline_variance.py",
)

uncertainty = load_module(
    "stage3d_uncertainty",
    SCRIPTS / "validate_stage3d_uncertainty_operators.py",
)

sampling = load_module(
    "stage3d_sampling",
    SCRIPTS / "validate_stage3d_sampling_and_seeds.py",
)


def source_sersic_bn_high_n(n: float) -> float:
    """
    Exact SNANA get_Sersic_bn branch for n > 0.36.

    Source:
      sntools_host.c:get_Sersic_bn()
    """

    if not n > 0.36:
        raise ValueError(
            "Source-exact low-n branch requires SNANA's frozen "
            "SERSIC_TABLE.grid_n/grid_bn table. Do not approximate it."
        )

    n2 = n * n
    n3 = n2 * n
    n4 = n2 * n2

    bn = 0.0
    bn += 2.0 * n - 1.0 / 3.0
    bn += 4.0 / (405.0 * n)
    bn += 46.0 / (25515.0 * n2)
    bn += 131.0 / (1148175.0 * n3)
    bn += 2194697.0 / (30690717750.0 * n4)

    return bn


def make_psf_grid(psfmax_snana: float) -> np.ndarray:
    return np.concatenate(
        [
            PSF_GRID_ARCSEC,
            np.asarray(
                [psfmax_snana / 2.35],
                dtype=float,
            ),
        ]
    )


def snana_interp_galmag(
    *,
    psfsig_arcsec: float,
    galmag_grid: np.ndarray,
    psf_grid: np.ndarray,
) -> float:
    """
    Reproduce interp_GALMAG_HOSTLIB boundary semantics.

    SNANA first moves out-of-range PSF values just inside the fixed
    interpolation interval, then performs 1-D interpolation.
    """

    if galmag_grid.shape != psf_grid.shape:
        raise ValueError(
            "GALMAG grid and PSF grid must have identical lengths."
        )

    pmin = float(psf_grid[0])
    pmax = float(psf_grid[-1])

    p = float(psfsig_arcsec)

    if p < pmin:
        p = pmin + 0.0001

    if p > pmax:
        p = pmax - 0.0001

    return float(
        np.interp(
            p,
            psf_grid,
            galmag_grid,
        )
    )


def source_sbmag(
    *,
    galmag_grid: np.ndarray,
    psf_grid: np.ndarray,
    hostlib_sbradius: float = 1.2,
) -> tuple[float, float, float]:
    """
    Exact downstream SBMAG construction from GEN_SNHOST_GALMAG.

      psfsig = HOSTLIB_SBRADIUS / 2
      AREA   = pi * (4 * psfsig^2)
      SB_MAG = GALMAG(psfsig) + 2.5 log10(AREA)
      SB_FLUXCAL = 10^[-0.4*(SB_MAG-27.5)]
    """

    psfsig = hostlib_sbradius / 2.0

    area = math.pi * (
        4.0 * psfsig * psfsig
    )

    galmag = snana_interp_galmag(
        psfsig_arcsec=psfsig,
        galmag_grid=galmag_grid,
        psf_grid=psf_grid,
    )

    sbmag = galmag + 2.5 * math.log10(area)

    sb_fluxcal = 10.0 ** (
        -0.4 * (
            sbmag - ZEROPOINT_FLUXCAL
        )
    )

    return sbmag, sb_fluxcal, galmag


def epoch_host_photon_variance(
    *,
    galmag_grid: np.ndarray,
    psf_grid: np.ndarray,
    nea_pixels: float,
    pixsize_arcsec: float,
    zpt: float,
    gain: float,
) -> dict[str, float]:
    """
    Exact SNANA gen_fluxNoise_calc host-photo-stat path.

      area_bg       = NEA
      psfsig_arcsec = pixsize * sqrt(NEA/(2*TWOPI))
      galmag        = interp_GALMAG_HOSTLIB(...)
      fluxgal_pe    = gain * 10^[0.4*(zpt-galmag)]

    fluxgal_pe is directly stored as SQSIG_HOST_PHOT.
    """

    psfsig_arcsec = (
        pixsize_arcsec
        * math.sqrt(
            nea_pixels / (2.0 * TWOPI)
        )
    )

    galmag_nea = snana_interp_galmag(
        psfsig_arcsec=psfsig_arcsec,
        galmag_grid=galmag_grid,
        psf_grid=psf_grid,
    )

    fluxgal_pe = (
        gain
        * 10.0 ** (
            0.4 * (
                zpt - galmag_nea
            )
        )
    )

    return {
        "nea_pixels": float(nea_pixels),
        "psfsig_arcsec": float(psfsig_arcsec),
        "galmag_nea": float(galmag_nea),
        "fluxgal_pe": float(fluxgal_pe),
        "sqsig_host_phot": float(fluxgal_pe),
    }


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


def parse_galmag_grid(text: str) -> np.ndarray:
    values = np.asarray(
        [
            float(x)
            for x in text.split(",")
            if x.strip()
        ],
        dtype=float,
    )

    if values.size != 9:
        raise ValueError(
            "Need exactly 9 GALMAG values corresponding to "
            "SNANA's nine nonzero PSF-grid points."
        )

    return values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--galmag-grid",
        required=True,
        help=(
            "Comma-separated nine GALMAG values corresponding "
            "to SNANA PSF bins 1..9."
        ),
    )

    parser.add_argument(
        "--psfmax-snana",
        required=True,
        type=float,
        help="Frozen SNANA PSFMAX_SNANA value.",
    )

    parser.add_argument(
        "--band",
        required=True,
        choices=list("griz"),
    )

    parser.add_argument(
        "--field",
        required=True,
    )

    parser.add_argument(
        "--fluxcal",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--zpt",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--gain",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--skysig",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--readnoise",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--nea",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--pixsize",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--zpterr",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--template-skysig",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--template-zpt",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--des-fluxerr-file",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    galmag_grid = parse_galmag_grid(
        args.galmag_grid
    )

    psf_grid = make_psf_grid(
        args.psfmax_snana
    )

    if psf_grid.size != galmag_grid.size:
        raise RuntimeError(
            "Internal PSF/GALMAG grid length mismatch."
        )

    sbmag, sb_fluxcal, sb_galmag = (
        source_sbmag(
            galmag_grid=galmag_grid,
            psf_grid=psf_grid,
        )
    )

    host = epoch_host_photon_variance(
        galmag_grid=galmag_grid,
        psf_grid=psf_grid,
        nea_pixels=args.nea,
        pixsize_arcsec=args.pixsize,
        zpt=args.zpt,
        gain=args.gain,
    )

    include_template = (
        args.template_zpt is not None
        and args.template_skysig > 0.0
    )

    base_without_host = variance.baseline_variance(
        fluxcal=max(args.fluxcal, 0.0),
        zpt=args.zpt,
        gain=args.gain,
        skysig=args.skysig,
        readnoise=args.readnoise,
        nea=args.nea,
        zpterr=args.zpterr,
        include_zp=True,
        template_skysig=args.template_skysig,
        template_readnoise=0.0,
        template_zpt=args.template_zpt,
        include_template=include_template,
        host_flux_pe=0.0,
    )

    base_with_host = variance.baseline_variance(
        fluxcal=max(args.fluxcal, 0.0),
        zpt=args.zpt,
        gain=args.gain,
        skysig=args.skysig,
        readnoise=args.readnoise,
        nea=args.nea,
        zpterr=args.zpterr,
        include_zp=True,
        template_skysig=args.template_skysig,
        template_readnoise=0.0,
        template_zpt=args.template_zpt,
        include_template=include_template,
        host_flux_pe=host["fluxgal_pe"],
    )

    expected_delta = host["fluxgal_pe"]

    measured_delta = (
        base_with_host["baseline_variance_pe2"]
        - base_without_host["baseline_variance_pe2"]
    )

    if not np.isclose(
        measured_delta,
        expected_delta,
        rtol=1.0e-12,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Host variance did not enter baseline variance "
            "with SNANA source semantics."
        )

    des_maps = uncertainty.map_index(
        uncertainty.parse_fluxerr_file(
            args.des_fluxerr_file
        )
    )

    stratum = sampling.classify_stratum(
        "DES-SN5YR",
        args.field,
    )

    errscale = map_lookup(
        des_maps,
        (
            "FLUXERR_SCALE",
            stratum,
            args.band,
        ),
        sbmag,
    )

    fluxcalerr_pre_map = float(
        base_with_host["fluxcalerr_in"]
    )

    fluxcalerr_post_map = (
        fluxcalerr_pre_map
        * errscale
    )

    result = {
        "stage": "3D.7",
        "status": "PASS",
        "scientific_boundary": {
            "classifier_results_read": False,
            "stage1_stage2_results_read": False,
            "parameters_tuned_to_outcomes": False,
        },
        "inputs": {
            "band": args.band,
            "field": args.field,
            "des_stratum": stratum,
            "fluxcal_true": args.fluxcal,
            "zpt": args.zpt,
            "gain": args.gain,
            "skysig": args.skysig,
            "readnoise": args.readnoise,
            "nea_pixels": args.nea,
            "pixsize_arcsec": args.pixsize,
            "zpterr": args.zpterr,
            "galmag_grid": galmag_grid.tolist(),
            "psf_grid_arcsec": psf_grid.tolist(),
        },
        "sbmag_semantics": {
            "hostlib_sbradius_arcsec": 1.2,
            "sb_psfsig_arcsec": 0.6,
            "galmag_at_sb_psf": sb_galmag,
            "sbmag": sbmag,
            "sb_fluxcal": sb_fluxcal,
        },
        "epoch_host_photon": host,
        "baseline_without_host": base_without_host,
        "baseline_with_host": base_with_host,
        "host_variance_delta_check": {
            "expected_pe2": expected_delta,
            "measured_pe2": measured_delta,
            "pass": True,
        },
        "des_fluxerrmodel": {
            "errscale": float(errscale),
            "fluxcalerr_pre_map": fluxcalerr_pre_map,
            "fluxcalerr_post_map": fluxcalerr_post_map,
        },
        "scope": {
            "upstream_galfrac_reproduced": False,
            "galmag_grid_treated_as_source_input": True,
            "reason": (
                "GEN_SNHOST_GALMAG Sersic integration-table and "
                "Gauss2d_Overlap construction remain separate "
                "source-validation targets."
            ),
        },
    }

    out = (
        OUTDIR
        / "des_host_photon_semantics_validation.json"
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("===== STAGE 3D.7 DES HOST-PHOTON VALIDATION =====")
    print()
    print(f"Field / stratum : {args.field} / {stratum}")
    print(f"Band            : {args.band}")
    print()
    print(f"SBMAG           : {sbmag:.8f}")
    print(f"SB_FLUXCAL      : {sb_fluxcal:.8f}")
    print()
    print(
        "epoch PSF sigma : "
        f"{host['psfsig_arcsec']:.8f} arcsec"
    )
    print(
        "GALMAG_NEA      : "
        f"{host['galmag_nea']:.8f}"
    )
    print(
        "SQSIG_HOST_PHOT : "
        f"{host['sqsig_host_phot']:.8f} pe^2"
    )
    print()
    print(
        "FLUXCALERR pre  : "
        f"{fluxcalerr_pre_map:.8f}"
    )
    print(
        "DES ERRSCALE    : "
        f"{errscale:.8f}"
    )
    print(
        "FLUXCALERR post : "
        f"{fluxcalerr_post_map:.8f}"
    )
    print()
    print(
        "Host variance insertion: PASS"
    )
    print(
        "Downstream DES host-photon semantics: PASS"
    )
    print()
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
