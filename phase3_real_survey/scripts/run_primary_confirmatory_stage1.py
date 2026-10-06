#!/usr/bin/env python3

"""
Phase 3 preregistered primary confirmatory analysis — Stage 1.

THIS IS THE FIRST EMPIRICAL OUTCOME REVEAL.

Permitted here:
- only the five preregistered primary features;
- fixed preregistered redshift bins;
- normal-Ia primary analysis;
- DES shallow/deep and SDSS north/south strata;
- registered quantiles;
- object-level bootstrap uncertainty;
- cadence-spread summaries for r_time_of_peak.

Explicitly NOT done here:
- no other compact features;
- no classifier;
- no SHAP/permutation importance;
- no feature ranking;
- no exploratory correlations;
- no tuning;
- no physics correction fitted from outcomes;
- no secondary/diagnostic metrics.

The frozen synthetic passband operator is deliberately NOT applied in Stage 1.
Stage 1 reveals only the registered raw empirical effects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]

DATA_DIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "real_feature_tables"
)

OUT_DIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "primary_confirmatory_stage1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SDSS_PATH = DATA_DIR / "sdss_compact_features.csv"
DES_PATH = DATA_DIR / "des_compact_features.csv"

EXPECTED_SHA256 = {
    SDSS_PATH:
        "bdb0653f0638d3998f77c2b1889aad8bfe438fc5c849e5312da525aec7335541",
    DES_PATH:
        "8963b9e1fb4fe0b30cec58599ae01c53ecf7088a3ddd555b5aed73098d39f669",
}

PRIMARY_PASSBAND = [
    "peak_color_r_minus_i",
    "peak_color_i_minus_z",
    "i_peak_flux",
    "z_peak_flux",
]

PRIMARY_TIMING = "r_time_of_peak"

PRIMARY_FEATURES = (
    PRIMARY_PASSBAND
    + [PRIMARY_TIMING]
)

Z_EDGES = np.array(
    [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ],
    dtype=float,
)

Z_LABELS = [
    "[0.05,0.10)",
    "[0.10,0.15)",
    "[0.15,0.20)",
    "[0.20,0.25)",
    "[0.25,0.30]",
]

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261006


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def resolve_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:

    lookup = {
        str(c).lower(): str(c)
        for c in frame.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[
                candidate.lower()
            ]

    raise RuntimeError(
        "Required column not found. Tried: "
        + ", ".join(candidates)
    )


def fixed_z_bin(z: float) -> str | None:

    if not np.isfinite(z):
        return None

    if z < 0.05 or z > 0.30:
        return None

    if z == 0.30:
        return Z_LABELS[-1]

    index = int(
        np.searchsorted(
            Z_EDGES,
            z,
            side="right",
        )
        - 1
    )

    if index < 0 or index >= len(Z_LABELS):
        return None

    return Z_LABELS[index]


def q10(values):
    return float(
        np.quantile(
            values,
            0.10,
        )
    )


def median(values):
    return float(
        np.median(values)
    )


def q90(values):
    return float(
        np.quantile(
            values,
            0.90,
        )
    )


def mad(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    med = np.median(values)

    return float(
        np.median(
            np.abs(
                values - med
            )
        )
    )


def iqr(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.quantile(
            values,
            0.75,
        )
        - np.quantile(
            values,
            0.25,
        )
    )


def bootstrap_median_difference(
    sdss_values,
    des_values,
    rng,
    n_boot=BOOTSTRAP_REPS,
):

    a = np.asarray(
        sdss_values,
        dtype=float,
    )

    b = np.asarray(
        des_values,
        dtype=float,
    )

    if (
        len(a) < 2
        or len(b) < 2
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    observed = (
        np.median(b)
        - np.median(a)
    )

    sims = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        aa = rng.choice(
            a,
            size=len(a),
            replace=True,
        )

        bb = rng.choice(
            b,
            size=len(b),
            replace=True,
        )

        sims[i] = (
            np.median(bb)
            - np.median(aa)
        )

    lower, upper = np.quantile(
        sims,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(observed),
        float(lower),
        float(upper),
    )


def bootstrap_spread_difference(
    sdss_values,
    des_values,
    statistic,
    rng,
    n_boot=BOOTSTRAP_REPS,
):

    a = np.asarray(
        sdss_values,
        dtype=float,
    )

    b = np.asarray(
        des_values,
        dtype=float,
    )

    if (
        len(a) < 3
        or len(b) < 3
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    observed = (
        statistic(b)
        - statistic(a)
    )

    sims = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        aa = rng.choice(
            a,
            size=len(a),
            replace=True,
        )

        bb = rng.choice(
            b,
            size=len(b),
            replace=True,
        )

        sims[i] = (
            statistic(bb)
            - statistic(aa)
        )

    lower, upper = np.quantile(
        sims,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(observed),
        float(lower),
        float(upper),
    )


def prepare_frame(
    path: Path,
    survey_name: str,
):

    df = pd.read_csv(
        path
    )

    class_col = resolve_column(
        df,
        [
            "class",
            "label_name",
        ],
    )

    z_col = resolve_column(
        df,
        [
            "z",
            "z_helio",
            "redshift_helio",
            "sim_z",
        ],
    )

    field_col = resolve_column(
        df,
        [
            "field",
            "field_name",
            "survey_field",
        ],
    )

    # Both locked tables explicitly contain field_stratum.
    # DES: SHALLOW/DEEP
    # SDSS: 82N/82S
    stratum_col = resolve_column(
        df,
        [
            "field_stratum",
        ],
    )

    missing_features = [
        feature
        for feature
        in PRIMARY_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"{survey_name}: missing primary features: "
            + ", ".join(
                missing_features
            )
        )

    # Primary normal-Ia confirmatory population only.
    ia = df.loc[
        df[class_col].astype(str)
        == "Ia"
    ].copy()

    ia["_z"] = pd.to_numeric(
        ia[z_col],
        errors="raise",
    )

    ia["_z_bin"] = [
        fixed_z_bin(
            float(z)
        )
        for z in ia["_z"]
    ]

    ia["_field"] = (
        ia[field_col]
        .astype(str)
    )

    ia["_stratum"] = (
        ia[stratum_col]
        .astype(str)
    )

    # Keep aliases used by the preregistered reporting code.
    if survey_name == "DES-SN5YR":
        ia["_depth"] = ia["_stratum"]
    else:
        ia["_depth"] = "NA"

    if ia["_z_bin"].isna().any():
        raise RuntimeError(
            f"{survey_name}: Ia object outside "
            "the frozen redshift support."
        )

    return ia


def main():

    print("=" * 78)
    print(
        "PHASE 3 PRIMARY CONFIRMATORY ANALYSIS — STAGE 1"
    )
    print(
        "FIRST EMPIRICAL OUTCOME REVEAL"
    )
    print(
        "ONLY FIVE PREREGISTERED PRIMARY FEATURES"
    )
    print("=" * 78)

    print()
    print(
        "[1] Verifying locked empirical tables..."
    )

    for path, expected in EXPECTED_SHA256.items():

        observed = sha256(
            path
        )

        print(
            f"  {path.name}"
        )

        print(
            f"    expected : {expected}"
        )

        print(
            f"    observed : {observed}"
        )

        if observed != expected:
            raise RuntimeError(
                f"Hash mismatch for {path}"
            )

        print(
            "    status   : PASS"
        )

    print()
    print(
        "[2] Loading only preregistered "
        "primary features and metadata..."
    )

    sdss = prepare_frame(
        SDSS_PATH,
        "SDSS-II",
    )

    des = prepare_frame(
        DES_PATH,
        "DES-SN5YR",
    )

    print(
        f"  SDSS normal Ia : {len(sdss)}"
    )

    print(
        f"  DES normal Ia  : {len(des)}"
    )

    if len(sdss) != 305:
        raise RuntimeError(
            "SDSS Ia primary count changed."
        )

    if len(des) != 134:
        raise RuntimeError(
            "DES Ia primary count changed."
        )

    print()
    print(
        "[3] Registered passband-feature "
        "quantiles and DES-SDSS differences..."
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    passband_rows = []

    for feature in PRIMARY_PASSBAND:

        print()
        print(
            f"  {feature}"
        )

        for z_bin in Z_LABELS:

            a = sdss.loc[
                sdss["_z_bin"]
                == z_bin,
                feature,
            ].to_numpy(
                dtype=float
            )

            b = des.loc[
                des["_z_bin"]
                == z_bin,
                feature,
            ].to_numpy(
                dtype=float
            )

            if (
                len(a) == 0
                or len(b) == 0
            ):
                row = {
                    "feature":
                        feature,

                    "z_bin":
                        z_bin,

                    "sdss_n":
                        len(a),

                    "des_n":
                        len(b),
                }

                passband_rows.append(
                    row
                )

                print(
                    f"    {z_bin}: "
                    f"insufficient support "
                    f"(SDSS={len(a)}, DES={len(b)})"
                )

                continue

            (
                delta_med,
                ci_low,
                ci_high,
            ) = bootstrap_median_difference(
                a,
                b,
                rng,
            )

            row = {
                "feature":
                    feature,

                "z_bin":
                    z_bin,

                "sdss_n":
                    int(
                        len(a)
                    ),

                "des_n":
                    int(
                        len(b)
                    ),

                "sdss_q10":
                    q10(a),

                "sdss_q50":
                    median(a),

                "sdss_q90":
                    q90(a),

                "des_q10":
                    q10(b),

                "des_q50":
                    median(b),

                "des_q90":
                    q90(b),

                "delta_q10_DES_minus_SDSS":
                    q10(b)
                    - q10(a),

                "delta_q50_DES_minus_SDSS":
                    delta_med,

                "delta_q90_DES_minus_SDSS":
                    q90(b)
                    - q90(a),

                "delta_q50_ci95_low":
                    ci_low,

                "delta_q50_ci95_high":
                    ci_high,
            }

            passband_rows.append(
                row
            )

            print(
                f"    {z_bin}  "
                f"N={len(a)}/{len(b)}  "
                f"Delta50={delta_med:+.6f}  "
                f"95% CI "
                f"[{ci_low:+.6f}, "
                f"{ci_high:+.6f}]"
            )

    passband = pd.DataFrame(
        passband_rows
    )

    print()
    print(
        "[4] Registered survey-stratum summaries..."
    )

    stratum_rows = []

    # DES shallow/deep.
    for feature in PRIMARY_FEATURES:

        for z_bin in Z_LABELS:

            for depth in sorted(
                des["_depth"].unique()
            ):

                values = des.loc[
                    (
                        des["_z_bin"]
                        == z_bin
                    )
                    & (
                        des["_depth"]
                        == depth
                    ),
                    feature,
                ].to_numpy(
                    dtype=float
                )

                if len(values) == 0:
                    continue

                stratum_rows.append(
                    {
                        "survey":
                            "DES-SN5YR",

                        "stratum_type":
                            "depth",

                        "stratum":
                            depth,

                        "z_bin":
                            z_bin,

                        "feature":
                            feature,

                        "n":
                            int(
                                len(values)
                            ),

                        "q10":
                            q10(values),

                        "median":
                            median(values),

                        "q90":
                            q90(values),

                        "mad":
                            mad(values),

                        "iqr":
                            iqr(values),
                    }
                )

    # SDSS north/south.
    for feature in PRIMARY_FEATURES:

        for z_bin in Z_LABELS:

            for field in sorted(
                sdss["_field"].unique()
            ):

                values = sdss.loc[
                    (
                        sdss["_z_bin"]
                        == z_bin
                    )
                    & (
                        sdss["_field"]
                        == field
                    ),
                    feature,
                ].to_numpy(
                    dtype=float
                )

                if len(values) == 0:
                    continue

                stratum_rows.append(
                    {
                        "survey":
                            "SDSS-II",

                        "stratum_type":
                            "field",

                        "stratum":
                            field,

                        "z_bin":
                            z_bin,

                        "feature":
                            feature,

                        "n":
                            int(
                                len(values)
                            ),

                        "q10":
                            q10(values),

                        "median":
                            median(values),

                        "q90":
                            q90(values),

                        "mad":
                            mad(values),

                        "iqr":
                            iqr(values),
                    }
                )

    strata = pd.DataFrame(
        stratum_rows
    )

    print(
        f"  stratum rows written: "
        f"{len(strata)}"
    )

    print()
    print(
        "[5] Registered r_time_of_peak "
        "spread comparison..."
    )

    timing_rows = []

    for z_bin in Z_LABELS:

        a = sdss.loc[
            sdss["_z_bin"]
            == z_bin,
            PRIMARY_TIMING,
        ].to_numpy(
            dtype=float
        )

        b = des.loc[
            des["_z_bin"]
            == z_bin,
            PRIMARY_TIMING,
        ].to_numpy(
            dtype=float
        )

        if (
            len(a) < 3
            or len(b) < 3
        ):
            continue

        (
            delta_mad,
            mad_low,
            mad_high,
        ) = bootstrap_spread_difference(
            a,
            b,
            mad,
            rng,
        )

        (
            delta_iqr,
            iqr_low,
            iqr_high,
        ) = bootstrap_spread_difference(
            a,
            b,
            iqr,
            rng,
        )

        timing_rows.append(
            {
                "z_bin":
                    z_bin,

                "sdss_n":
                    int(
                        len(a)
                    ),

                "des_n":
                    int(
                        len(b)
                    ),

                "sdss_q10":
                    q10(a),

                "sdss_median":
                    median(a),

                "sdss_q90":
                    q90(a),

                "sdss_mad":
                    mad(a),

                "sdss_iqr":
                    iqr(a),

                "des_q10":
                    q10(b),

                "des_median":
                    median(b),

                "des_q90":
                    q90(b),

                "des_mad":
                    mad(b),

                "des_iqr":
                    iqr(b),

                "delta_mad_DES_minus_SDSS":
                    delta_mad,

                "delta_mad_ci95_low":
                    mad_low,

                "delta_mad_ci95_high":
                    mad_high,

                "delta_iqr_DES_minus_SDSS":
                    delta_iqr,

                "delta_iqr_ci95_low":
                    iqr_low,

                "delta_iqr_ci95_high":
                    iqr_high,
            }
        )

        print(
            f"  {z_bin} "
            f"N={len(a)}/{len(b)} "
            f"DeltaMAD={delta_mad:+.6f} "
            f"DeltaIQR={delta_iqr:+.6f}"
        )

    timing = pd.DataFrame(
        timing_rows
    )

    passband_path = (
        OUT_DIR
        / "primary_passband_raw.csv"
    )

    strata_path = (
        OUT_DIR
        / "primary_strata_raw.csv"
    )

    timing_path = (
        OUT_DIR
        / "primary_r_time_of_peak_raw.csv"
    )

    passband.to_csv(
        passband_path,
        index=False,
    )

    strata.to_csv(
        strata_path,
        index=False,
    )

    timing.to_csv(
        timing_path,
        index=False,
    )

    manifest = {
        "stage":
            "preregistered_primary_confirmatory_stage1",

        "first_outcome_reveal":
            True,

        "primary_features":
            PRIMARY_FEATURES,

        "passband_features":
            PRIMARY_PASSBAND,

        "timing_feature":
            PRIMARY_TIMING,

        "redshift_bins":
            Z_LABELS,

        "bootstrap_reps":
            BOOTSTRAP_REPS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "normal_Ia_only_primary":
            True,

        "sdss_Ia_n":
            int(
                len(sdss)
            ),

        "des_Ia_n":
            int(
                len(des)
            ),

        "exploratory_features_inspected":
            False,

        "classifier_run":
            False,

        "synthetic_operator_applied":
            False,
    }

    manifest_path = (
        OUT_DIR
        / "stage1_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "PRIMARY CONFIRMATORY STAGE 1: COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        f"Passband table : {passband_path}"
    )

    print(
        f"Strata table   : {strata_path}"
    )

    print(
        f"Timing table   : {timing_path}"
    )

    print(
        f"Manifest       : {manifest_path}"
    )

    print()
    print(
        "STOP HERE."
    )

    print(
        "Do not inspect secondary features or run "
        "the classifier."
    )


if __name__ == "__main__":
    main()
