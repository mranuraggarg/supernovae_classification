#!/usr/bin/env python3
"""Command-line runner for the frozen Stage 3C Step 5 analysis.

This runner must be executed only after researcher authorization to inspect the
production scientific outcomes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stage3c_step5_analysis import (
    AnalysisConfig,
    analyze_jsonl,
    sha256_file,
)
from stage3c_step5_timing import hsiao_r_peak_phase
from stage3c_step5_native import native_backend_provenance


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "phase3_real_survey"
SCRIPTS = ROOT / "scripts"
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "stage3c_production_monte_carlo"
    / "stage3c_production_branches.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "stage3c_step5_statistical_analysis"
    / "stage3c_step5_analysis.json"
)
EXPECTED_PRODUCTION_SHA256 = (
    "2bbfc6454ad7a1f6457442c30d47418ea2d71a8987eb73c9309fe07d021cada0"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_timing_reference_phases():
    passband = load_module(
        "stage3c_step5_passband",
        SCRIPTS / "reproduce_passband_predictions.py",
    )
    paths = {
        key: passband.locate_frozen_artifact(key)
        for key in ("sdss_kcor", "des_r", "hsiao")
    }
    hsiao = passband.load_hsiao(paths["hsiao"])
    responses = {
        "SDSS-II": passband.load_sdss_response(paths["sdss_kcor"], "r"),
        "DES-SN5YR": passband.load_des_response(paths["des_r"]),
    }
    references = {}
    for redshift in AnalysisConfig().redshifts:
        for survey, strata in (
            ("SDSS-II", ("82N", "82S")),
            ("DES-SN5YR", ("SHALLOW", "DEEP")),
        ):
            phase = hsiao_r_peak_phase(
                hsiao,
                responses[survey],
                float(redshift),
                passband.synthetic_flux_proxy,
            )
            for stratum in strata:
                references[(survey, stratum, round(float(redshift), 3))] = phase
    return references, {
        "hsiao_path": str(paths["hsiao"]),
        "hsiao_sha256": sha256_file(paths["hsiao"]),
        "sdss_kcor_path": str(paths["sdss_kcor"]),
        "sdss_kcor_sha256": sha256_file(paths["sdss_kcor"]),
        "des_r_response_path": str(paths["des_r"]),
        "des_r_response_sha256": sha256_file(paths["des_r"]),
    }


def code_provenance():
    paths = (
        SCRIPTS / "run_stage3c_step5_analysis.py",
        SCRIPTS / "stage3c_step5_analysis.py",
        SCRIPTS / "stage3c_step5_statistics.py",
        SCRIPTS / "stage3c_step5_native.py",
        SCRIPTS / "stage3c_step5_timing.py",
        SCRIPTS / "stage3c_step5_records.py",
        ROOT / "native" / "stage3c_bootstrap_timing.cpp",
        ROOT
        / "results"
        / "passband_validation"
        / "reproduced_passband_predictions.csv",
        ROOT
        / "preregistration"
        / "stage3c_step5_statistical_analysis_clarification_2026-10-12.md",
        ROOT
        / "preregistration"
        / "stage3c_step5_statistical_analysis_freeze_2026-10-12.json",
    )
    return {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in paths
    }


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen Stage 3C Step 5 statistical analysis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_PRODUCTION_SHA256,
    )
    parser.add_argument(
        "--timing-backend",
        choices=("native", "python"),
        default="native",
        help="Timing-bootstrap implementation (default: native C++ kernel).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.output.with_name(
        "stage3c_step5_analysis_manifest.json"
    )
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing analysis artifact: "
            f"{args.output} or {manifest_path}"
        )

    started = time.perf_counter()
    native_provenance = (
        native_backend_provenance()
        if args.timing_backend == "native"
        else None
    )
    references, reference_provenance = build_timing_reference_phases()
    provenance = code_provenance()
    result = analyze_jsonl(
        args.input,
        references,
        config=AnalysisConfig(timing_bootstrap_backend=args.timing_backend),
        expected_sha256=args.expected_sha256,
        code_provenance=provenance,
    )
    result["timing_reference_provenance"] = reference_provenance
    result["execution"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_interpretation_generated": False,
        "classifier_executed": False,
        "native_timing_backend": native_provenance,
    }
    atomic_write_json(args.output, result)

    manifest = {
        "schema": "stage3c-step5-analysis-manifest-v1",
        "analysis_output": str(args.output),
        "analysis_output_sha256": sha256_file(args.output),
        "input_path": str(args.input),
        "input_sha256": result["input_provenance"]["sha256"],
        "code_provenance": provenance,
        "scientific_interpretation_generated": False,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Analysis output: {args.output}")
    print(f"Manifest: {manifest_path}")
    print(f"Elapsed seconds: {result['execution']['elapsed_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
