#!/usr/bin/env python3

"""
Phase 3 preregistered primary confirmatory analysis — Stage 2.

PURPOSE
-------
Compare the already-frozen Stage-1 real-data effects with the already-frozen
physics predictions.

PASSBAND PRIMARY FEATURES
-------------------------
peak_color_r_minus_i
peak_color_i_minus_z
i_peak_flux
z_peak_flux

For colours:
    x_SDSS_to_DES = x_SDSS + Delta_colour(z)

For peak-flux features:
    x = log10(1 + F)
    F = 10**x - 1
    F_SDSS_to_DES = F * 10**(-0.4 * Delta_mag(z))
    x_SDSS_to_DES = log10(1 + F_SDSS_to_DES)

No coefficient is fitted from SDSS/DES outcomes.

CADENCE PRIMARY FEATURE
-----------------------
r_time_of_peak

The real-data spread differences are compared with the already-frozen
fixed-template cadence-injection predictions. No empirical timing correction
is fitted in this stage.

NOT PERMITTED HERE
------------------
- no secondary features
- no classifier
- no SHAP/permutation importance
- no outcome-selected transformations
- no retuning of passband/cadence operators
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


REPO = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "real_feature_tables"
)

STAGE1_DIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "primary_confirmatory_stage1"
)

PASSBAND_DIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "passband_validation"
)

CADENCE_DIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "cadence_template_injection"
)

OUT_DIR = (
    REPO
    / "phase3_real_survey"
    / "results"
    / "primary_confirmatory_stage2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SDSS_PATH = FEATURE_DIR / "sdss_compact_features.csv"
DES_PATH = FEATURE_DIR / "des_compact_features.csv"

PASSBAND_PATH = (
    PASSBAND_DIR
    / "reproduced_passband_predictions.csv"
)

CADENCE_SUMMARY_PATH = (
    CADENCE_DIR
    / "cadence_injection_summary.csv"
)

CADENCE_FIELD_PATH = (
    CADENCE_DIR
    / "cadence_injection_by_field.csv"
)

STAGE1_PASSBAND_PATH = (
    STAGE1_DIR
    / "primary_passband_raw.csv"
)

STAGE1_TIMING_PATH = (
    STAGE1_DIR
    / "primary_r_time_of_peak_raw.csv"
)


EXPECTED_FEATURE_HASHES = {
    SDSS_PATH:
        "bdb0653f0638d3998f77c2b1889aad8bfe438fc5c849e5312da525aec7335541",

    DES_PATH:
        "8963b9e1fb4fe0b30cec58599ae01c53ecf7088a3ddd555b5aed73098d39f669",
}


PASSBAND_FEATURES = {
    "peak_color_r_minus_i": {
        "prediction_column":
            "dtop3_r-i",

        "kind":
            "colour",
    },

    "peak_color_i_minus_z": {
        "prediction_column":
            "dtop3_i-z",

        "kind":
            "colour",
    },

    "i_peak_flux": {
        "prediction_column":
            "dpeak_i",

        "kind":
            "compressed_flux",
    },

    "z_peak_flux": {
        "prediction_column":
            "dpeak_z",

        "kind":
            "compressed_flux",
    },
}


Z_BINS = [
    (0.05, 0.10, "[0.05,0.10)"),
    (0.10, 0.15, "[0.10,0.15)"),
    (0.15, 0.20, "[0.15,0.20)"),
    (0.20, 0.25, "[0.20,0.25)"),
    (0.25, 0.30, "[0.25,0.30]"),
]

Z_BIN_CENTRES = {
    "[0.05,0.10)": 0.075,
    "[0.10,0.15)": 0.125,
    "[0.15,0.20)": 0.175,
    "[0.20,0.25)": 0.225,
    "[0.25,0.30]": 0.275,
}


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def fixed_z_bin(
    z: float,
) -> str | None:

    if not np.isfinite(z):
        return None

    for low, high, label in Z_BINS:

        if label == "[0.25,0.30]":

            if low <= z <= high:
                return label

        elif low <= z < high:
            return label

    return None


def interpolate_prediction(
    z,
    prediction_grid,
    column,
):

    return np.interp(
        np.asarray(
            z,
            dtype=float,
        ),
        prediction_grid["z"].to_numpy(
            dtype=float
        ),
        prediction_grid[column].to_numpy(
            dtype=float
        ),
    )


def apply_colour_operator(
    values,
    delta,
):

    return (
        np.asarray(
            values,
            dtype=float,
        )
        + np.asarray(
            delta,
            dtype=float,
        )
    )


def apply_compressed_flux_operator(
    values,
    delta_mag,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    dm = np.asarray(
        delta_mag,
        dtype=float,
    )

    # Frozen feature:
    # x = log10(1 + max(F,0)).
    #
    # Primary cohort peak fluxes are non-negative.
    flux = (
        np.power(
            10.0,
            x,
        )
        - 1.0
    )

    flux = np.maximum(
        flux,
        0.0,
    )

    flux_ratio = np.power(
        10.0,
        -0.4 * dm,
    )

    transformed_flux = (
        flux
        * flux_ratio
    )

    return np.log10(
        1.0
        + transformed_flux
    )


def median(
    values,
):

    return float(
        np.median(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def mad(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    med = np.median(
        x
    )

    return float(
        np.median(
            np.abs(
                x - med
            )
        )
    )


def iqr(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.quantile(
            x,
            0.75,
        )
        - np.quantile(
            x,
            0.25,
        )
    )


def prepare_ia(
    frame,
):

    ia = frame.loc[
        frame["class"]
        .astype(str)
        == "Ia"
    ].copy()

    ia["_z"] = pd.to_numeric(
        ia["z_helio"],
        errors="raise",
    )

    ia["_z_bin"] = [
        fixed_z_bin(
            float(z)
        )
        for z in ia["_z"]
    ]

    ia["_stratum"] = (
        ia["field_stratum"]
        .astype(str)
    )

    if ia["_z_bin"].isna().any():
        raise RuntimeError(
            "Object outside frozen "
            "redshift support."
        )

    return ia


def passband_comparison(
    feature,
    kind,
    prediction_column,
    sdss_group,
    des_group,
    prediction_grid,
    comparison_name,
    sdss_stratum,
    des_stratum,
    z_bin,
):

    if (
        len(sdss_group) == 0
        or len(des_group) == 0
    ):
        return None

    sdss_values = (
        sdss_group[
            feature
        ].to_numpy(
            dtype=float
        )
    )

    des_values = (
        des_group[
            feature
        ].to_numpy(
            dtype=float
        )
    )

    delta = interpolate_prediction(
        sdss_group["_z"],
        prediction_grid,
        prediction_column,
    )

    if kind == "colour":

        transformed = (
            apply_colour_operator(
                sdss_values,
                delta,
            )
        )

    elif kind == "compressed_flux":

        transformed = (
            apply_compressed_flux_operator(
                sdss_values,
                delta,
            )
        )

    else:

        raise RuntimeError(
            f"Unknown operator kind: "
            f"{kind}"
        )

    raw_delta = (
        median(
            des_values
        )
        - median(
            sdss_values
        )
    )

    residual_delta = (
        median(
            des_values
        )
        - median(
            transformed
        )
    )

    before_w = float(
        wasserstein_distance(
            sdss_values,
            des_values,
        )
    )

    after_w = float(
        wasserstein_distance(
            transformed,
            des_values,
        )
    )

    if abs(
        raw_delta
    ) > 1e-12:

        fraction_removed = (
            raw_delta
            - residual_delta
        ) / raw_delta

    else:

        fraction_removed = (
            np.nan
        )

    return {
        "feature":
            feature,

        "z_bin":
            z_bin,

        "comparison":
            comparison_name,

        "sdss_stratum":
            sdss_stratum,

        "des_stratum":
            des_stratum,

        "sdss_n":
            int(
                len(
                    sdss_values
                )
            ),

        "des_n":
            int(
                len(
                    des_values
                )
            ),

        "sdss_raw_median":
            median(
                sdss_values
            ),

        "sdss_transformed_median":
            median(
                transformed
            ),

        "des_observed_median":
            median(
                des_values
            ),

        "raw_delta_DES_minus_SDSS":
            raw_delta,

        "residual_delta_after_operator":
            residual_delta,

        "median_operator_shift":
            (
                median(
                    transformed
                )
                - median(
                    sdss_values
                )
            ),

        "wasserstein_before":
            before_w,

        "wasserstein_after":
            after_w,

        "wasserstein_change_after_minus_before":
            (
                after_w
                - before_w
            ),

        "fraction_signed_median_displacement_removed":
            fraction_removed,

        "mean_frozen_prediction_on_SDSS_objects":
            float(
                np.mean(
                    delta
                )
            ),

        "median_frozen_prediction_on_SDSS_objects":
            median(
                delta
            ),
    }


def main():

    print(
        "=" * 78
    )

    print(
        "PHASE 3 PRIMARY CONFIRMATORY "
        "ANALYSIS — STAGE 2"
    )

    print(
        "FROZEN PHYSICS OPERATORS ONLY"
    )

    print(
        "=" * 78
    )

    print()
    print(
        "[1] Verifying locked real "
        "feature tables..."
    )

    for path, expected in (
        EXPECTED_FEATURE_HASHES.items()
    ):

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
                f"Hash mismatch: {path}"
            )

        print(
            "    status   : PASS"
        )

    print()
    print(
        "[2] Loading frozen predictions "
        "and Stage-1 outcomes..."
    )

    prediction = pd.read_csv(
        PASSBAND_PATH
    )

    if not bool(
        prediction["pass"]
        .astype(bool)
        .all()
    ):

        raise RuntimeError(
            "Frozen passband reproduction "
            "contains a failed row."
        )

    expected_z = np.arange(
        0.05,
        0.3000001,
        0.025,
    )

    actual_z = prediction[
        "z"
    ].to_numpy(
        dtype=float
    )

    if not np.allclose(
        actual_z,
        expected_z,
        rtol=0.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            "Frozen passband redshift grid "
            "does not match registration."
        )

    stage1_passband = (
        pd.read_csv(
            STAGE1_PASSBAND_PATH
        )
    )

    stage1_timing = (
        pd.read_csv(
            STAGE1_TIMING_PATH
        )
    )

    cadence_summary = (
        pd.read_csv(
            CADENCE_SUMMARY_PATH
        )
    )

    cadence_field = (
        pd.read_csv(
            CADENCE_FIELD_PATH
        )
    )

    print(
        "  frozen passband grid : PASS"
    )

    print(
        f"  Stage-1 passband rows: "
        f"{len(stage1_passband)}"
    )

    print(
        f"  Stage-1 timing rows  : "
        f"{len(stage1_timing)}"
    )

    print()
    print(
        "[3] Applying fixed passband "
        "operator to normal-Ia SDSS..."
    )

    sdss = prepare_ia(
        pd.read_csv(
            SDSS_PATH
        )
    )

    des = prepare_ia(
        pd.read_csv(
            DES_PATH
        )
    )

    if len(sdss) != 305:
        raise RuntimeError(
            "SDSS Ia count changed."
        )

    if len(des) != 134:
        raise RuntimeError(
            "DES Ia count changed."
        )

    passband_rows = []

    for (
        feature,
        specification,
    ) in PASSBAND_FEATURES.items():

        print()
        print(
            f"  {feature}"
        )

        for _, _, z_bin in Z_BINS:

            sdss_bin = sdss.loc[
                sdss["_z_bin"]
                == z_bin
            ]

            des_bin = des.loc[
                des["_z_bin"]
                == z_bin
            ]

            # Primary pooled comparison.
            row = passband_comparison(
                feature=
                    feature,

                kind=
                    specification[
                        "kind"
                    ],

                prediction_column=
                    specification[
                        "prediction_column"
                    ],

                sdss_group=
                    sdss_bin,

                des_group=
                    des_bin,

                prediction_grid=
                    prediction,

                comparison_name=
                    "POOLED",

                sdss_stratum=
                    "ALL",

                des_stratum=
                    "ALL",

                z_bin=
                    z_bin,
            )

            if row is not None:

                passband_rows.append(
                    row
                )

                print(
                    f"    {z_bin} "
                    f"raw={row['raw_delta_DES_minus_SDSS']:+.6f} "
                    f"residual={row['residual_delta_after_operator']:+.6f} "
                    f"W={row['wasserstein_before']:.6f}"
                    f"->{row['wasserstein_after']:.6f}"
                )

            # Registered field/depth sensitivity:
            # each SDSS stripe against each DES depth class.
            for sdss_stratum in (
                "82N",
                "82S",
            ):

                for des_stratum in (
                    "SHALLOW",
                    "DEEP",
                ):

                    a = sdss_bin.loc[
                        sdss_bin[
                            "_stratum"
                        ]
                        == sdss_stratum
                    ]

                    b = des_bin.loc[
                        des_bin[
                            "_stratum"
                        ]
                        == des_stratum
                    ]

                    row = passband_comparison(
                        feature=
                            feature,

                        kind=
                            specification[
                                "kind"
                            ],

                        prediction_column=
                            specification[
                                "prediction_column"
                            ],

                        sdss_group=
                            a,

                        des_group=
                            b,

                        prediction_grid=
                            prediction,

                        comparison_name=
                            "STRATIFIED",

                        sdss_stratum=
                            sdss_stratum,

                        des_stratum=
                            des_stratum,

                        z_bin=
                            z_bin,
                    )

                    if row is not None:

                        passband_rows.append(
                            row
                        )

    passband_metrics = pd.DataFrame(
        passband_rows
    )

    print()
    print(
        "[4] Verifying Stage-1 pooled "
        "raw medians are reproduced..."
    )

    pooled = (
        passband_metrics.loc[
            passband_metrics[
                "comparison"
            ]
            == "POOLED"
        ]
        .copy()
    )

    check = stage1_passband.merge(
        pooled[
            [
                "feature",
                "z_bin",
                "raw_delta_DES_minus_SDSS",
            ]
        ],
        on=[
            "feature",
            "z_bin",
        ],
        how="left",
        validate="one_to_one",
    )

    max_difference = float(
        np.max(
            np.abs(
                check[
                    "delta_q50_DES_minus_SDSS"
                ].to_numpy(
                    dtype=float
                )
                - check[
                    "raw_delta_DES_minus_SDSS"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    print(
        f"  maximum absolute difference: "
        f"{max_difference:.3e}"
    )

    if max_difference > 1e-12:

        raise RuntimeError(
            "Stage-2 raw medians do not "
            "reproduce frozen Stage-1."
        )

    print(
        "  Stage-1 reproduction: PASS"
    )

    print()
    print(
        "[5] Comparing real r_time_of_peak "
        "spread with frozen cadence model..."
    )

    cadence_rows = []

    for _, stage1_row in (
        stage1_timing.iterrows()
    ):

        z_bin = str(
            stage1_row[
                "z_bin"
            ]
        )

        z_center = (
            Z_BIN_CENTRES[
                z_bin
            ]
        )

        sdss_model = (
            cadence_summary.loc[
                (
                    cadence_summary[
                        "survey"
                    ]
                    == "SDSS-II"
                )
                & np.isclose(
                    cadence_summary[
                        "redshift"
                    ],
                    z_center,
                )
                & (
                    cadence_summary[
                        "stratum"
                    ]
                    == "ALL"
                )
            ]
        )

        des_model = (
            cadence_summary.loc[
                (
                    cadence_summary[
                        "survey"
                    ]
                    == "DES-SN5YR"
                )
                & np.isclose(
                    cadence_summary[
                        "redshift"
                    ],
                    z_center,
                )
                & (
                    cadence_summary[
                        "stratum"
                    ]
                    == "ALL"
                )
            ]
        )

        if (
            len(sdss_model) != 1
            or len(des_model) != 1
        ):

            raise RuntimeError(
                f"Missing cadence model "
                f"at z={z_center}"
            )

        s = sdss_model.iloc[0]
        d = des_model.iloc[0]

        predicted_delta_mad = (
            float(
                d[
                    "mad_error_days"
                ]
            )
            - float(
                s[
                    "mad_error_days"
                ]
            )
        )

        predicted_delta_iqr = (
            float(
                d[
                    "iqr_error_days"
                ]
            )
            - float(
                s[
                    "iqr_error_days"
                ]
            )
        )

        cadence_rows.append(
            {
                "z_bin":
                    z_bin,

                "z_center":
                    z_center,

                "real_delta_mad":
                    float(
                        stage1_row[
                            "delta_mad_DES_minus_SDSS"
                        ]
                    ),

                "real_delta_mad_ci95_low":
                    float(
                        stage1_row[
                            "delta_mad_ci95_low"
                        ]
                    ),

                "real_delta_mad_ci95_high":
                    float(
                        stage1_row[
                            "delta_mad_ci95_high"
                        ]
                    ),

                "predicted_delta_mad":
                    predicted_delta_mad,

                "real_delta_iqr":
                    float(
                        stage1_row[
                            "delta_iqr_DES_minus_SDSS"
                        ]
                    ),

                "real_delta_iqr_ci95_low":
                    float(
                        stage1_row[
                            "delta_iqr_ci95_low"
                        ]
                    ),

                "real_delta_iqr_ci95_high":
                    float(
                        stage1_row[
                            "delta_iqr_ci95_high"
                        ]
                    ),

                "predicted_delta_iqr":
                    predicted_delta_iqr,

                "predicted_DES_mad":
                    float(
                        d[
                            "mad_error_days"
                        ]
                    ),

                "predicted_SDSS_mad":
                    float(
                        s[
                            "mad_error_days"
                        ]
                    ),

                "predicted_DES_iqr":
                    float(
                        d[
                            "iqr_error_days"
                        ]
                    ),

                "predicted_SDSS_iqr":
                    float(
                        s[
                            "iqr_error_days"
                        ]
                    ),
            }
        )

        print(
            f"  {z_bin} "
            f"MAD real/model="
            f"{float(stage1_row['delta_mad_DES_minus_SDSS']):+.3f}/"
            f"{predicted_delta_mad:+.3f} "
            f"IQR real/model="
            f"{float(stage1_row['delta_iqr_DES_minus_SDSS']):+.3f}/"
            f"{predicted_delta_iqr:+.3f}"
        )

    cadence_compare = pd.DataFrame(
        cadence_rows
    )

    print()
    print(
        "[6] Recording frozen DES "
        "shallow/deep cadence predictions..."
    )

    cadence_strata_rows = []

    for z_bin, z_center in (
        Z_BIN_CENTRES.items()
    ):

        sdss_all = (
            cadence_summary.loc[
                (
                    cadence_summary[
                        "survey"
                    ]
                    == "SDSS-II"
                )
                & np.isclose(
                    cadence_summary[
                        "redshift"
                    ],
                    z_center,
                )
                & (
                    cadence_summary[
                        "stratum"
                    ]
                    == "ALL"
                )
            ]
            .iloc[0]
        )

        for des_depth in (
            "SHALLOW",
            "DEEP",
        ):

            des_depth_row = (
                cadence_field.loc[
                    (
                        cadence_field[
                            "survey"
                        ]
                        == "DES-SN5YR"
                    )
                    & np.isclose(
                        cadence_field[
                            "redshift"
                        ],
                        z_center,
                    )
                    & (
                        cadence_field[
                            "stratum"
                        ]
                        == des_depth
                    )
                ]
            )

            if len(
                des_depth_row
            ) != 1:

                raise RuntimeError(
                    f"Missing {des_depth} "
                    f"cadence row at "
                    f"z={z_center}"
                )

            d = (
                des_depth_row.iloc[0]
            )

            cadence_strata_rows.append(
                {
                    "z_bin":
                        z_bin,

                    "z_center":
                        z_center,

                    "des_depth":
                        des_depth,

                    "predicted_delta_mad_vs_SDSS_ALL":
                        float(
                            d[
                                "mad_error_days"
                            ]
                        )
                        - float(
                            sdss_all[
                                "mad_error_days"
                            ]
                        ),

                    "predicted_delta_iqr_vs_SDSS_ALL":
                        float(
                            d[
                                "iqr_error_days"
                            ]
                        )
                        - float(
                            sdss_all[
                                "iqr_error_days"
                            ]
                        ),

                    "des_mad_error_days":
                        float(
                            d[
                                "mad_error_days"
                            ]
                        ),

                    "des_iqr_error_days":
                        float(
                            d[
                                "iqr_error_days"
                            ]
                        ),

                    "sdss_all_mad_error_days":
                        float(
                            sdss_all[
                                "mad_error_days"
                            ]
                        ),

                    "sdss_all_iqr_error_days":
                        float(
                            sdss_all[
                                "iqr_error_days"
                            ]
                        ),
                }
            )

    cadence_strata = pd.DataFrame(
        cadence_strata_rows
    )

    passband_out = (
        OUT_DIR
        / "passband_operator_metrics.csv"
    )

    cadence_out = (
        OUT_DIR
        / "cadence_model_comparison.csv"
    )

    cadence_strata_out = (
        OUT_DIR
        / "cadence_depth_predictions.csv"
    )

    passband_metrics.to_csv(
        passband_out,
        index=False,
    )

    cadence_compare.to_csv(
        cadence_out,
        index=False,
    )

    cadence_strata.to_csv(
        cadence_strata_out,
        index=False,
    )

    manifest = {
        "stage":
            "preregistered_primary_confirmatory_stage2",

        "stage1_modified":
            False,

        "primary_passband_features":
            list(
                PASSBAND_FEATURES
            ),

        "primary_cadence_feature":
            "r_time_of_peak",

        "passband_operator":
            (
                "Frozen dense Hsiao DES-SDSS "
                "prediction interpolated at each "
                "SDSS object's redshift"
            ),

        "colour_operator":
            "x_DES_pred = x_SDSS + Delta_colour(z)",

        "compressed_flux_operator":
            (
                "F=10**x-1; "
                "F_DES_pred=F*10**(-0.4*Delta_mag(z)); "
                "x_DES_pred=log10(1+F_DES_pred)"
            ),

        "cadence_operator":
            (
                "Previously frozen object-level "
                "schedule injection; no fitted "
                "real-data timing correction"
            ),

        "secondary_features_inspected":
            False,

        "classifier_run":
            False,

        "outcome_fitted_coefficients":
            False,

        "stage1_reproduction_max_abs_difference":
            max_difference,

        "input_sha256": {
            str(
                SDSS_PATH.relative_to(
                    REPO
                )
            ):
                sha256(
                    SDSS_PATH
                ),

            str(
                DES_PATH.relative_to(
                    REPO
                )
            ):
                sha256(
                    DES_PATH
                ),

            str(
                PASSBAND_PATH.relative_to(
                    REPO
                )
            ):
                sha256(
                    PASSBAND_PATH
                ),

            str(
                CADENCE_SUMMARY_PATH.relative_to(
                    REPO
                )
            ):
                sha256(
                    CADENCE_SUMMARY_PATH
                ),

            str(
                CADENCE_FIELD_PATH.relative_to(
                    REPO
                )
            ):
                sha256(
                    CADENCE_FIELD_PATH
                ),

            str(
                STAGE1_PASSBAND_PATH.relative_to(
                    REPO
                )
            ):
                sha256(
                    STAGE1_PASSBAND_PATH
                ),

            str(
                STAGE1_TIMING_PATH.relative_to(
                    REPO
                )
            ):
                sha256(
                    STAGE1_TIMING_PATH
                ),
        },
    }

    manifest_path = (
        OUT_DIR
        / "stage2_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 78
    )

    print(
        "PRIMARY CONFIRMATORY STAGE 2: "
        "COMPLETE"
    )

    print(
        "=" * 78
    )

    print()
    print(
        f"Passband metrics : "
        f"{passband_out}"
    )

    print(
        f"Cadence comparison: "
        f"{cadence_out}"
    )

    print(
        f"Cadence depth     : "
        f"{cadence_strata_out}"
    )

    print(
        f"Manifest          : "
        f"{manifest_path}"
    )

    print()
    print(
        "STOP HERE."
    )

    print(
        "Do not inspect secondary features "
        "or run the classifier."
    )


if __name__ == "__main__":
    main()
