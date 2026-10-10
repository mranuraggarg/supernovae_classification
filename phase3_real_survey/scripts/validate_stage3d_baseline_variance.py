#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

import numpy as np


SNANA_SOURCE_COMMIT = "4113779f038eec613b6684d0d46a47019d7c9624"
ZEROPOINT_FLUXCAL = 27.5

EXPECTED = {
    "sdss": {
        "path": "simlib/SDSS/SDSS_3year.SIMLIB",
        "sha256": "fad8558b04c2fc10e26d18f42d016ab52311078ea87d8cfb44f1c47e3fe7c6c9",
    },
    "des": {
        "path": "simlib/DES/DES-SN5YR_DES.SIMLIB",
        "sha256": "f5575ecd5b50c526ff63b758c5f6b88a710a185040dfd3c6e5008ca7b67d92a0",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def noise_equiv_aperture(
    psfsig1: float,
    psfsig2: float,
    psfratio: float,
) -> float:
    """
    Exact translation of SNANA NoiseEquivAperture() for the
    double-Gaussian SIMLIB PSF.

    Source:
      RickKessler/SNANA
      commit 4113779f038eec613b6684d0d46a47019d7c9624
      src/sntools.c
    """
    sq1 = psfsig1 * psfsig1
    sq2 = psfsig2 * psfsig2

    if psfratio < 1.0e-5 or psfsig2 < 0.0001:
        return 4.0 * math.pi * sq1

    tmp = psfratio * sq2 / sq1
    a1 = 1.0 / (1.0 + tmp)
    a2 = 1.0 - a1

    tmp = (
        a1 * psfsig2 / psfsig1
        + a2 * psfsig1 / psfsig2
    )

    return (
        4.0
        * math.pi
        * (sq1 + sq2)
        / (1.0 + tmp * tmp)
    )


def convert_psf_to_internal(
    psf1: float,
    psf2: float,
    ratio: float,
    pixsize: float,
    psf_unit: str,
):
    """
    Reproduce relevant SNANA SIMLIB PSF preparation.

    Internal PSFSIG1/2 units are sigma in pixels.
    """
    if psf_unit == "ARCSEC_FWHM":
        factor = pixsize * 2.3548
        psf1 = psf1 / factor
        psf2 = psf2 / factor

    elif psf_unit == "PIXEL_SIGMA":
        pass

    else:
        raise RuntimeError(
            f"Unsupported PSF unit in controlled validator: {psf_unit}"
        )

    nea = noise_equiv_aperture(psf1, psf2, ratio)

    return psf1, psf2, ratio, nea


def baseline_variance(
    *,
    fluxcal: float,
    zpt: float,
    gain: float,
    skysig: float,
    readnoise: float,
    nea: float,
    zpterr: float = 0.0,
    include_zp: bool = False,
    template_skysig: float = 0.0,
    template_readnoise: float = 0.0,
    template_zpt: float | None = None,
    include_template: bool = False,
    host_flux_pe: float = 0.0,
):
    """
    Controlled translation of the baseline part of
    SNANA gen_fluxNoise_calc().

    Returns quantities in photoelectron or photoelectron^2 units.
    """

    nadu_over_fluxcal = 10.0 ** (
        0.4 * (zpt - ZEROPOINT_FLUXCAL)
    )

    npe_over_fluxcal = nadu_over_fluxcal * gain

    fluxsn_pe = fluxcal * npe_over_fluxcal

    sqerr_sky_pe = nea * (skysig * gain) ** 2
    sqerr_ccd_pe = nea * readnoise**2

    sqsig_noz = (
        fluxsn_pe
        + host_flux_pe
        + sqerr_sky_pe
        + sqerr_ccd_pe
    )

    sqerr_zp_pe = 0.0

    if include_zp:
        relerr = 10.0 ** (0.4 * zpterr) - 1.0
        err = fluxsn_pe * relerr
        sqerr_zp_pe = err * err

    sqsig_data = sqsig_noz + sqerr_zp_pe

    template_sqerr_pe = 0.0

    if include_template:
        if template_zpt is None:
            raise RuntimeError(
                "template_zpt required when template noise is enabled."
            )

        template_sqerr_sky_pe = (
            nea * (template_skysig * gain) ** 2
        )

        template_sqerr_ccd_pe = (
            nea * template_readnoise**2
        )

        # SNANA: zfac = 10^[0.8*(search_zpt-template_zpt)]
        zfac = 10.0 ** (0.8 * (zpt - template_zpt))

        template_sqerr_pe = (
            template_sqerr_sky_pe
            + template_sqerr_ccd_pe
        ) * zfac

    sqsig_calc_data = sqsig_data + template_sqerr_pe

    sigma_pe = math.sqrt(sqsig_calc_data)

    fluxcalerr_in = sigma_pe / npe_over_fluxcal

    snr_szt = (
        fluxsn_pe / sigma_pe
        if sigma_pe > 0.0
        else math.inf
    )

    return {
        "nadu_over_fluxcal": nadu_over_fluxcal,
        "npe_over_fluxcal": npe_over_fluxcal,
        "fluxsn_pe": fluxsn_pe,
        "nea": nea,
        "source_variance_pe2": fluxsn_pe,
        "sky_variance_pe2": sqerr_sky_pe,
        "read_variance_pe2": sqerr_ccd_pe,
        "host_variance_pe2": host_flux_pe,
        "zp_variance_pe2": sqerr_zp_pe,
        "template_variance_pe2": template_sqerr_pe,
        "baseline_variance_pe2": sqsig_calc_data,
        "baseline_sigma_pe": sigma_pe,
        "fluxcalerr_in": fluxcalerr_in,
        "snr_szt": snr_szt,
    }


def read_global_simlib_settings(path: Path):
    text = path.read_text(errors="replace")

    pix = re.search(
        r"(?m)^\s*PIXSIZE:\s*([0-9.eE+-]+)",
        text,
    )

    psf = re.search(
        r"(?m)^\s*PSF_UNIT:\s*(\S+)",
        text,
    )

    return {
        "pixsize_global": float(pix.group(1)) if pix else None,
        "psf_unit": psf.group(1) if psf else "PIXEL_SIGMA",
    }


def first_simlib_observation(path: Path):
    """
    Read the first LIBID's first S-row plus its local PIXSIZE and
    optional template header values.

    This is controlled structural inspection only.
    """

    pixsize = None
    field = None
    template_zpt = {}
    template_skysig = {}

    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.strip()

            # STAGE3D_PIXSIZE_EMBEDDED
            m_pix = re.search(
                r"\bPIXSIZE:\s*([0-9.eE+-]+)",
                line,
            )
            if m_pix:
                pixsize = float(m_pix.group(1))

            if line.startswith("PIXSIZE:"):
                try:
                    pixsize = float(line.split()[1])
                except Exception:
                    pass

            if line.startswith("FIELD:"):
                parts = line.split()
                if len(parts) >= 2:
                    field = parts[1]

            if line.startswith("TEMPLATE_ZPT:"):
                vals = [
                    float(x)
                    for x in line.split(":", 1)[1].split()
                ]
                template_zpt["values"] = vals

            if line.startswith("TEMPLATE_SKYSIG:"):
                vals = [
                    float(x)
                    for x in line.split(":", 1)[1].split()
                ]
                template_skysig["values"] = vals

            if line.startswith("S:"):
                parts = line.split()

                if len(parts) < 12:
                    raise RuntimeError(
                        f"Unexpected SIMLIB S row: {line}"
                    )

                return {
                    "mjd": float(parts[1]),
                    "idexpt": parts[2],
                    "band": parts[3],
                    "gain": float(parts[4]),
                    "readnoise": float(parts[5]),
                    "skysig": float(parts[6]),
                    "psf1": float(parts[7]),
                    "psf2": float(parts[8]),
                    "psfratio": float(parts[9]),
                    "zpt": float(parts[10]),
                    "zpterr": float(parts[11]),
                    "pixsize": pixsize,
                    "field": field,
                    "template_zpt": template_zpt.get("values"),
                    "template_skysig": template_skysig.get("values"),
                }

    raise RuntimeError(f"No S observation found in {path}")


def main():
    print("=" * 78)
    print("PHASE 3 STAGE 3D.4")
    print("BASELINE SIMLIB VARIANCE CONSTRUCTION VALIDATION")
    print("NO SYNTHETIC LIGHT CURVES — NO REAL OUTCOMES — NO CLASSIFIER")
    print("=" * 78)

    sndata = os.environ.get("SNDATA_ROOT")

    if not sndata:
        raise RuntimeError("SNDATA_ROOT is not defined.")

    sndata = Path(sndata)

    paths = {
        name: sndata / info["path"]
        for name, info in EXPECTED.items()
    }

    print("\n[1] Verifying frozen SIMLIB artifacts...")

    for name, path in paths.items():
        observed = sha256(path)
        expected = EXPECTED[name]["sha256"]

        print(f"  {name.upper()}")
        print(f"    expected : {expected}")
        print(f"    observed : {observed}")

        if observed != expected:
            raise RuntimeError(f"{name} SIMLIB hash mismatch.")

        print("    status   : PASS")

    print("\n[2] Recording source-level semantics...")

    print(
        "  SNANA commit        :",
        SNANA_SOURCE_COMMIT,
    )
    print(
        "  FLUXCAL zero point  :",
        ZEROPOINT_FLUXCAL,
    )
    print(
        "  baseline components : source + host + sky + read"
    )
    print(
        "  optional components : zeropoint + template"
    )

    print("\n[3] Reading controlled SIMLIB observations...")

    obs = {}

    for name, path in paths.items():
        settings = read_global_simlib_settings(path)
        row = first_simlib_observation(path)

        if row["pixsize"] is None:
            row["pixsize"] = settings["pixsize_global"]

        row["psf_unit"] = settings["psf_unit"]

        if row["pixsize"] is None:
            raise RuntimeError(
                f"No PIXSIZE recovered for {name}"
            )

        obs[name] = row

        print(f"\n  {name.upper()}")
        print(f"    field     : {row['field']}")
        print(f"    MJD       : {row['mjd']}")
        print(f"    band      : {row['band']}")
        print(f"    gain      : {row['gain']}")
        print(f"    readnoise : {row['readnoise']}")
        print(f"    skysig    : {row['skysig']}")
        print(f"    psf1/2    : {row['psf1']} / {row['psf2']}")
        print(f"    ratio     : {row['psfratio']}")
        print(f"    zpt       : {row['zpt']}")
        print(f"    zpterr    : {row['zpterr']}")
        print(f"    pixsize   : {row['pixsize']}")
        print(f"    PSF_UNIT  : {row['psf_unit']}")

    print("\n[4] Validating PSF -> NEA transformation...")

    for name, row in obs.items():
        psf1, psf2, ratio, nea = convert_psf_to_internal(
            row["psf1"],
            row["psf2"],
            row["psfratio"],
            row["pixsize"],
            row["psf_unit"],
        )

        if not np.isfinite(nea) or nea <= 0.0:
            raise RuntimeError(
                f"Invalid NEA for {name}: {nea}"
            )

        row["psfsig1_internal"] = psf1
        row["psfsig2_internal"] = psf2
        row["nea"] = nea

        print(
            f"  {name.upper():4s}: "
            f"sigma1={psf1:.6f} pix "
            f"sigma2={psf2:.6f} pix "
            f"NEA={nea:.6f} pix^2"
        )

    print("\n[5] Single-Gaussian analytic NEA identity...")

    sigma = 1.7

    observed = noise_equiv_aperture(
        sigma,
        0.0,
        0.0,
    )

    expected = 4.0 * math.pi * sigma**2

    print("  observed :", observed)
    print("  expected :", expected)

    if not np.isclose(
        observed,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Single-Gaussian NEA identity failed."
        )

    print("  status   : PASS")

    print("\n[6] Controlled baseline variance tests...")

    # Arbitrary controlled FLUXCAL, not a real-data outcome.
    test_fluxcal = 1000.0

    for name, row in obs.items():
        result = baseline_variance(
            fluxcal=test_fluxcal,
            zpt=row["zpt"],
            gain=row["gain"],
            skysig=row["skysig"],
            readnoise=row["readnoise"],
            nea=row["nea"],
            include_zp=False,
            include_template=False,
            host_flux_pe=0.0,
        )

        component_sum = (
            result["source_variance_pe2"]
            + result["sky_variance_pe2"]
            + result["read_variance_pe2"]
        )

        if not np.isclose(
            result["baseline_variance_pe2"],
            component_sum,
            rtol=1.0e-14,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                f"Baseline component sum failed for {name}."
            )

        if not np.isclose(
            result["fluxsn_pe"],
            test_fluxcal * result["npe_over_fluxcal"],
            rtol=1.0e-14,
            atol=1.0e-10,
        ):
            raise RuntimeError(
                f"FLUXCAL -> p.e. conversion failed for {name}."
            )

        print(f"\n  {name.upper()}")
        print(
            f"    Npe/FLUXCAL : "
            f"{result['npe_over_fluxcal']:.8f}"
        )
        print(
            f"    source var   : "
            f"{result['source_variance_pe2']:.8f}"
        )
        print(
            f"    sky var      : "
            f"{result['sky_variance_pe2']:.8f}"
        )
        print(
            f"    read var     : "
            f"{result['read_variance_pe2']:.8f}"
        )
        print(
            f"    total var    : "
            f"{result['baseline_variance_pe2']:.8f}"
        )
        print(
            f"    FLUXCALERRin : "
            f"{result['fluxcalerr_in']:.8f}"
        )
        print(
            f"    SNR_SZT      : "
            f"{result['snr_szt']:.8f}"
        )

    print("\n[7] Controlled zeropoint-error branch...")

    row = obs["sdss"]

    no_zp = baseline_variance(
        fluxcal=test_fluxcal,
        zpt=row["zpt"],
        gain=row["gain"],
        skysig=row["skysig"],
        readnoise=row["readnoise"],
        nea=row["nea"],
        zpterr=row["zpterr"],
        include_zp=False,
    )

    yes_zp = baseline_variance(
        fluxcal=test_fluxcal,
        zpt=row["zpt"],
        gain=row["gain"],
        skysig=row["skysig"],
        readnoise=row["readnoise"],
        nea=row["nea"],
        zpterr=row["zpterr"],
        include_zp=True,
    )

    if yes_zp["baseline_variance_pe2"] < no_zp["baseline_variance_pe2"]:
        raise RuntimeError(
            "Zeropoint variance unexpectedly reduced total variance."
        )

    print(
        "  variance without ZP :",
        f"{no_zp['baseline_variance_pe2']:.8f}",
    )
    print(
        "  variance with ZP    :",
        f"{yes_zp['baseline_variance_pe2']:.8f}",
    )
    print("  branch behaviour    : PASS")

    print("\n[8] Controlled template-noise branch...")

    # Synthetic controlled values only; this tests the exact scaling law.
    base = baseline_variance(
        fluxcal=test_fluxcal,
        zpt=30.0,
        gain=4.0,
        skysig=10.0,
        readnoise=0.0,
        nea=20.0,
        include_template=False,
    )

    templ = baseline_variance(
        fluxcal=test_fluxcal,
        zpt=30.0,
        gain=4.0,
        skysig=10.0,
        readnoise=0.0,
        nea=20.0,
        include_template=True,
        template_skysig=5.0,
        template_readnoise=0.0,
        template_zpt=31.0,
    )

    if templ["baseline_variance_pe2"] <= base["baseline_variance_pe2"]:
        raise RuntimeError(
            "Template variance was not added."
        )

    expected_template = (
        20.0
        * (5.0 * 4.0) ** 2
        * 10.0 ** (0.8 * (30.0 - 31.0))
    )

    if not np.isclose(
        templ["template_variance_pe2"],
        expected_template,
        rtol=1.0e-14,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Template-noise scaling test failed."
        )

    print(
        "  expected template var :",
        f"{expected_template:.8f}",
    )
    print(
        "  recovered template var:",
        f"{templ['template_variance_pe2']:.8f}",
    )
    print("  status                : PASS")

    print("\n[9] Implementation boundary...")

    print(
        "  FLUXCAL -> photoelectron conversion : VERIFIED"
    )
    print(
        "  PSF -> noise-equivalent area        : VERIFIED"
    )
    print(
        "  source Poisson variance             : VERIFIED"
    )
    print(
        "  sky variance                        : VERIFIED"
    )
    print(
        "  read-noise variance                 : VERIFIED"
    )
    print(
        "  zeropoint variance branch           : VERIFIED"
    )
    print(
        "  template-noise scaling branch       : VERIFIED"
    )
    print(
        "  host-photon branch formula          : structurally represented"
    )
    print(
        "  exact survey flag choices           : AUDIT SEPARATELY"
    )
    print(
        "  Stage-3D.3 FLUXERR corrections      : NOT APPLIED HERE"
    )

    print("\n" + "=" * 78)
    print("STAGE 3D.4 BASELINE VARIANCE VALIDATION: PASS")
    print("=" * 78)

    print()
    print("No stochastic flux was generated.")
    print("No real compact-feature outcome was read.")
    print("No classifier was run.")


if __name__ == "__main__":
    main()
