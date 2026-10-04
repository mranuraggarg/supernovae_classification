#!/usr/bin/env python3

"""
Phase 3 pre-outcome feature-builder equivalence audit.

The authoritative Phase-2 builder is NOT imported as a module.
Instead, the exact required constants/functions are extracted from the
checksummed frozen source using Python AST and executed in isolation.

This avoids unrelated import-time dependencies while retaining direct
traceability to the registered source artifact.

NO REAL SDSS/DES FEATURE OUTCOMES ARE READ.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]

FROZEN_PATH = (
    REPO
    / "archive"
    / "phase2"
    / "scripts"
    / "phase2_tier4_make_variants.py"
)

PHASE3_PATH = (
    REPO
    / "phase3_real_survey"
    / "scripts"
    / "build_real_survey_features.py"
)

EXPECTED_FROZEN_SHA256 = (
    "7d2b1b7c23d685e652b4b7df383441537162f9dc3829f8edba380a8ddf2a4d76"
)

FEATURES = [
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

ATOL = 1e-12
RTOL = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_phase3(path: Path):
    spec = importlib.util.spec_from_file_location(
        "phase3_real_builder",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_frozen_builder_from_source(path: Path):
    """
    Extract only the registered Tier-4 feature-builder constants and
    functions from the frozen source.

    No top-level imports or unrelated Phase-2 code are executed.
    """

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    wanted_constants = {
        "COLOR_CLIP_RANGE",
        "SPCC_BANDS",
        "SNR_ACTIVE_THRESHOLD",
        "MIN_COLOR_FLUX_THRESHOLD",
    }

    wanted_functions = {
        "_positive_log10_1p",
        "_signed_log10_1p",
        "_clip_value",
        "_safe_color_from_peak_fluxes",
        "_compress_feature_row",
        "_representative_flux_for_color",
        "_build_compact_row_from_observations",
    }

    selected_nodes = []

    found_constants = set()
    found_functions = set()

    for node in tree.body:

        if isinstance(
            node,
            (ast.Assign, ast.AnnAssign),
        ):

            targets = []

            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]

            names = {
                target.id
                for target in targets
                if isinstance(
                    target,
                    ast.Name,
                )
            }

            if names & wanted_constants:
                selected_nodes.append(node)
                found_constants |= (
                    names
                    & wanted_constants
                )

        elif isinstance(
            node,
            ast.FunctionDef,
        ):

            if node.name in wanted_functions:
                selected_nodes.append(node)
                found_functions.add(
                    node.name
                )

    missing_constants = (
        wanted_constants
        - found_constants
    )

    missing_functions = (
        wanted_functions
        - found_functions
    )

    if missing_constants:
        raise RuntimeError(
            "Missing frozen constants: "
            + ", ".join(
                sorted(
                    missing_constants
                )
            )
        )

    if missing_functions:
        raise RuntimeError(
            "Missing frozen functions: "
            + ", ".join(
                sorted(
                    missing_functions
                )
            )
        )

    isolated_module = ast.Module(
        body=selected_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        isolated_module
    )

    namespace = {
        "math": math,
        "np": np,
        "Any": object,
    }

    code = compile(
        isolated_module,
        filename=str(path),
        mode="exec",
    )

    exec(
        code,
        namespace,
        namespace,
    )

    return namespace


def obs(
    time,
    band,
    flux,
    err,
):
    return {
        "time": float(time),
        "band": str(band),
        "flux": float(flux),
        "flux_err": float(err),
    }


def evaluate_old(
    old,
    observations,
):

    fn = old[
        "_build_compact_row_from_observations"
    ]

    row = fn(
        snid=1,
        label_name="Ia",
        sim_z=0.15,
        observations=[
            dict(x)
            for x in observations
        ],
    )

    if row is None:
        return None

    return {
        feature: float(
            row[feature]
        )
        for feature in FEATURES
    }


def evaluate_new(
    new,
    observations,
):

    mjd = np.asarray(
        [
            x["time"]
            for x in observations
        ],
        dtype=float,
    )

    band = np.asarray(
        [
            x["band"]
            for x in observations
        ],
        dtype=object,
    )

    flux = np.asarray(
        [
            x["flux"]
            for x in observations
        ],
        dtype=float,
    )

    ferr = np.asarray(
        [
            x["flux_err"]
            for x in observations
        ],
        dtype=float,
    )

    row, diagnostic = (
        new.frozen_features(
            mjd,
            band,
            flux,
            ferr,
        )
    )

    return row


def compare_case(
    name,
    old,
    new,
    observations,
):

    a = evaluate_old(
        old,
        observations,
    )

    b = evaluate_new(
        new,
        observations,
    )

    if (
        (a is None)
        != (b is None)
    ):
        return {
            "case": name,
            "status": "FAIL",
            "reason": (
                "acceptance mismatch: "
                f"frozen="
                f"{'ACCEPT' if a is not None else 'REJECT'}, "
                f"phase3="
                f"{'ACCEPT' if b is not None else 'REJECT'}"
            ),
        }

    if a is None:
        return {
            "case": name,
            "status": "PASS",
            "reason": "both reject",
        }

    differences = []

    for feature in FEATURES:

        av = float(
            a[feature]
        )

        bv = float(
            b[feature]
        )

        if not math.isclose(
            av,
            bv,
            rel_tol=RTOL,
            abs_tol=ATOL,
        ):
            differences.append(
                (
                    feature,
                    av,
                    bv,
                    abs(
                        av - bv
                    ),
                )
            )

    if differences:
        return {
            "case": name,
            "status": "FAIL",
            "reason": "feature mismatch",
            "differences": differences,
        }

    return {
        "case": name,
        "status": "PASS",
        "reason": (
            "all 16 features identical "
            "within tolerance"
        ),
    }


def deterministic_cases():

    cases = {}

    rows = []

    for band, scale in zip(
        "griz",
        [
            100.0,
            130.0,
            150.0,
            120.0,
        ],
    ):
        rows.extend(
            [
                obs(
                    10.0,
                    band,
                    0.20 * scale,
                    2.0,
                ),
                obs(
                    20.0,
                    band,
                    0.70 * scale,
                    2.0,
                ),
                obs(
                    30.0,
                    band,
                    1.00 * scale,
                    2.0,
                ),
                obs(
                    40.0,
                    band,
                    0.55 * scale,
                    2.0,
                ),
            ]
        )

    cases[
        "all_active"
    ] = rows

    rows = []

    for band in "gri":
        rows.extend(
            [
                obs(
                    0,
                    band,
                    20,
                    2,
                ),
                obs(
                    10,
                    band,
                    40,
                    2,
                ),
                obs(
                    20,
                    band,
                    25,
                    2,
                ),
            ]
        )

    rows.extend(
        [
            obs(
                0,
                "z",
                1.5,
                1.0,
            ),
            obs(
                10,
                "z",
                2.5,
                1.0,
            ),
            obs(
                20,
                "z",
                1.0,
                1.0,
            ),
        ]
    )

    cases[
        "z_band_fallback"
    ] = rows

    rows = [
        obs(0, "g", 30, 2),
        obs(10, "g", 50, 2),
        obs(20, "g", 20, 2),

        obs(0, "r", 2.0, 1),
        obs(10, "r", 2.5, 1),
        obs(20, "r", 1.5, 1),

        obs(0, "i", 40, 2),
        obs(10, "i", 60, 2),
        obs(20, "i", 30, 2),

        obs(0, "z", 1.0, 1),
        obs(10, "z", 2.0, 1),
        obs(20, "z", 1.5, 1),
    ]

    cases[
        "multiple_band_fallback"
    ] = rows

    rows = []

    for j, band in enumerate(
        "griz"
    ):
        rows.extend(
            [
                obs(
                    0,
                    band,
                    1.0 + j,
                    1.0,
                ),
                obs(
                    10,
                    band,
                    2.0 + j,
                    1.5,
                ),
                obs(
                    20,
                    band,
                    1.5 + j,
                    1.0,
                ),
            ]
        )

    cases[
        "global_fallback"
    ] = rows

    rows = []

    for j, band in enumerate(
        "griz"
    ):
        rows.extend(
            [
                obs(
                    0,
                    band,
                    -8.0 - j,
                    2.0,
                ),
                obs(
                    10,
                    band,
                    25.0 + j,
                    2.0,
                ),
                obs(
                    20,
                    band,
                    45.0 + j,
                    2.0,
                ),
                obs(
                    30,
                    band,
                    -2.0,
                    2.0,
                ),
            ]
        )

    cases[
        "negative_forced_flux"
    ] = rows

    rows = []

    for j, band in enumerate(
        "griz"
    ):
        rows.extend(
            [
                obs(
                    30,
                    band,
                    20 + j,
                    2,
                ),
                obs(
                    0,
                    band,
                    10 + j,
                    2,
                ),
                obs(
                    20,
                    band,
                    50 + j,
                    2,
                ),
                obs(
                    10,
                    band,
                    30 + j,
                    2,
                ),
            ]
        )

    cases[
        "unsorted_input"
    ] = rows

    cases[
        "single_epoch_each_band"
    ] = [
        obs(
            1,
            "g",
            20,
            2,
        ),
        obs(
            2,
            "r",
            30,
            2,
        ),
        obs(
            3,
            "i",
            40,
            2,
        ),
        obs(
            4,
            "z",
            50,
            2,
        ),
    ]

    cases[
        "colour_clipping"
    ] = [
        obs(
            0,
            "g",
            1000000,
            1,
        ),
        obs(
            0,
            "r",
            1000,
            1,
        ),
        obs(
            0,
            "i",
            1,
            0.1,
        ),
        obs(
            0,
            "z",
            0.001,
            0.0001,
        ),
    ]

    cases[
        "missing_z"
    ] = [
        obs(
            0,
            "g",
            20,
            2,
        ),
        obs(
            10,
            "g",
            30,
            2,
        ),
        obs(
            0,
            "r",
            30,
            2,
        ),
        obs(
            10,
            "r",
            40,
            2,
        ),
        obs(
            0,
            "i",
            40,
            2,
        ),
        obs(
            10,
            "i",
            50,
            2,
        ),
    ]

    cases[
        "nonpositive_z_representative"
    ] = [
        obs(
            0,
            "g",
            20,
            2,
        ),
        obs(
            10,
            "g",
            30,
            2,
        ),
        obs(
            0,
            "r",
            30,
            2,
        ),
        obs(
            10,
            "r",
            40,
            2,
        ),
        obs(
            0,
            "i",
            40,
            2,
        ),
        obs(
            10,
            "i",
            50,
            2,
        ),
        obs(
            0,
            "z",
            -4,
            2,
        ),
        obs(
            10,
            "z",
            -1,
            2,
        ),
    ]

    return cases


def random_cases(
    n=1000,
    seed=20261004,
):

    rng = np.random.default_rng(
        seed
    )

    for case_index in range(
        n
    ):

        rows = []

        n_epochs = int(
            rng.integers(
                3,
                14,
            )
        )

        base_times = np.sort(
            rng.uniform(
                0.0,
                120.0,
                size=n_epochs,
            )
        )

        for band in "griz":

            if rng.random() < 0.03:
                continue

            band_scale = float(
                rng.uniform(
                    10.0,
                    300.0,
                )
            )

            phase_peak = float(
                rng.uniform(
                    20.0,
                    80.0,
                )
            )

            width = float(
                rng.uniform(
                    10.0,
                    40.0,
                )
            )

            for t in base_times:

                signal = (
                    band_scale
                    * math.exp(
                        -0.5
                        * (
                            (
                                t
                                - phase_peak
                            )
                            / width
                        )
                        ** 2
                    )
                )

                err = float(
                    rng.uniform(
                        1.0,
                        15.0,
                    )
                )

                measured = float(
                    signal
                    + rng.normal(
                        0.0,
                        err,
                    )
                )

                rows.append(
                    obs(
                        t
                        + rng.normal(
                            0.0,
                            0.25,
                        ),
                        band,
                        measured,
                        err,
                    )
                )

        rng.shuffle(
            rows
        )

        yield (
            f"random_{case_index:04d}",
            rows,
        )


def main():

    print("=" * 78)
    print(
        "PHASE 3 FEATURE-BUILDER "
        "EQUIVALENCE AUDIT"
    )
    print(
        "SYNTHETIC OBSERVATIONS ONLY - "
        "NO REAL SURVEY OUTCOMES"
    )
    print("=" * 78)

    print()
    print(
        "[1] Verifying registered frozen builder..."
    )

    frozen_hash = sha256(
        FROZEN_PATH
    )

    print(
        "  expected SHA-256 : "
        f"{EXPECTED_FROZEN_SHA256}"
    )

    print(
        "  observed SHA-256 : "
        f"{frozen_hash}"
    )

    if (
        frozen_hash
        != EXPECTED_FROZEN_SHA256
    ):
        print()
        print(
            "FEATURE-BUILDER EQUIVALENCE "
            "GATE: FAIL"
        )
        print(
            "Frozen Tier-4 builder hash does "
            "not match preregistration."
        )
        return 1

    print(
        "  hash             : PASS"
    )

    print()
    print(
        "[2] Extracting frozen builder "
        "directly from checksummed source..."
    )

    old = (
        load_frozen_builder_from_source(
            FROZEN_PATH
        )
    )

    print(
        "  frozen extraction: PASS"
    )

    print()
    print(
        "[3] Loading Phase-3 implementation..."
    )

    new = load_phase3(
        PHASE3_PATH
    )

    if not hasattr(
        new,
        "frozen_features",
    ):
        raise RuntimeError(
            "Phase-3 frozen_features "
            "function not found."
        )

    print(
        "  Phase-3 import   : PASS"
    )

    print()
    print(
        "[4] Running deterministic edge cases..."
    )

    results = []

    for (
        name,
        observations,
    ) in deterministic_cases().items():

        result = compare_case(
            name,
            old,
            new,
            observations,
        )

        results.append(
            result
        )

        print(
            f"  {name:32s} "
            f"{result['status']:4s}  "
            f"{result['reason']}"
        )

        if result.get(
            "differences"
        ):
            for (
                feature,
                a,
                b,
                delta,
            ) in result[
                "differences"
            ]:
                print(
                    f"      "
                    f"{feature:28s} "
                    f"old={a:.17g} "
                    f"new={b:.17g} "
                    f"|delta|={delta:.3e}"
                )

    print()
    print(
        "[5] Running deterministic randomized corpus..."
    )

    random_failures = []

    n_random = 1000

    for (
        name,
        observations,
    ) in random_cases(
        n=n_random
    ):

        result = compare_case(
            name,
            old,
            new,
            observations,
        )

        if (
            result[
                "status"
            ]
            != "PASS"
        ):

            random_failures.append(
                result
            )

            if (
                len(
                    random_failures
                )
                <= 10
            ):
                print(
                    f"  {name}: "
                    f"{result['reason']}"
                )

                for (
                    feature,
                    a,
                    b,
                    delta,
                ) in result.get(
                    "differences",
                    [],
                )[:10]:
                    print(
                        f"      "
                        f"{feature:28s} "
                        f"old={a:.17g} "
                        f"new={b:.17g} "
                        f"|delta|={delta:.3e}"
                    )

    print(
        f"  randomized cases : "
        f"{n_random:,}"
    )

    print(
        f"  failures         : "
        f"{len(random_failures):,}"
    )

    deterministic_failures = [
        result
        for result in results
        if (
            result["status"]
            != "PASS"
        )
    ]

    print()
    print("=" * 78)

    if (
        not deterministic_failures
        and not random_failures
    ):
        print(
            "FEATURE-BUILDER EQUIVALENCE "
            "GATE: PASS"
        )

        print(
            "Phase-3 implementation reproduces "
            "the registered Tier-4 operator on "
            "all synthetic validation cases."
        )

        status = 0

    else:

        print(
            "FEATURE-BUILDER EQUIVALENCE "
            "GATE: FAIL"
        )

        print(
            "Deterministic failures: "
            f"{len(deterministic_failures)}"
        )

        print(
            "Randomized failures   : "
            f"{len(random_failures)}"
        )

        print(
            "Do not proceed to real-survey "
            "feature outcomes until the "
            "implementation difference is resolved."
        )

        status = 1

    print("=" * 78)

    return status


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
