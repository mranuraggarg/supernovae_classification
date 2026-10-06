#!/usr/bin/env python3

"""
Phase 3 Stage 3A
Epoch-level measurement/noise metadata audit.

NO CLASS LABELS.
NO COMPACT FEATURES.
NO CROSS-SURVEY CLASSIFIER RESULTS.

Purpose:
- inspect released per-epoch instrumental metadata;
- establish which variables are usable for the preregistered
  noise/depth forward operator;
- retain only structurally valid griz observations;
- characterize survey/band/field-stratum measurement conditions.

This script does NOT fit any correction to SDSS-DES feature differences.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path("phase3_real_survey")

SDSS = (
    ROOT
    / "data/raw/sdss/SDSS_allCandidates+BOSS/"
      "SDSS_allCandidates+BOSS/"
      "SDSS_allCandidates+BOSS_PHOT.FITS"
)

DES = (
    ROOT
    / "data/raw/des/DES-SN5YR-1.2/0_DATA/"
      "DES-SN5YR_DES/"
      "DES-SN5YR_DES_PHOT.FITS.gz"
)

OUT = (
    ROOT
    / "results/epoch_measurement_model_audit"
)

OUT.mkdir(parents=True, exist_ok=True)


VARIABLES = [
    "FLUXCALERR",
    "PSF_SIG1",
    "PSF_SIG2",
    "PSF_RATIO",
    "SKY_SIG",
    "SKY_SIG_T",
    "RDNOISE",
    "ZEROPT",
    "ZEROPT_ERR",
    "GAIN",
]


def decode(x):
    if isinstance(x, bytes):
        return x.decode(errors="replace").strip()
    return str(x).strip()


def quantiles(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "q01": np.nan,
            "q10": np.nan,
            "q50": np.nan,
            "q90": np.nan,
            "q99": np.nan,
        }

    q = np.quantile(
        x,
        [0.01, 0.10, 0.50, 0.90, 0.99],
    )

    return {
        "q01": float(q[0]),
        "q10": float(q[1]),
        "q50": float(q[2]),
        "q90": float(q[3]),
        "q99": float(q[4]),
    }


def des_stratum(field):
    return (
        "DEEP"
        if field in {"C3", "X3"}
        else "SHALLOW"
    )


def load(path, survey):

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        cols = set(data.names)

        band_col = "FLT" if survey == "SDSS-II" else "BAND"

        frame = pd.DataFrame({
            "band": [
                decode(v)
                for v in data[band_col]
            ],
            "field": [
                decode(v)
                for v in data["FIELD"]
            ],
            "flux": np.asarray(
                data["FLUXCAL"],
                dtype=float,
            ),
            "fluxerr": np.asarray(
                data["FLUXCALERR"],
                dtype=float,
            ),
        })

        for name in VARIABLES:
            if name in cols:
                frame[name] = np.asarray(
                    data[name],
                    dtype=float,
                )

    # Structural quality mask already frozen:
    # canonical griz, finite flux/error, error > 0.
    frame["band"] = frame["band"].str.lower()

    mask = (
        frame["band"].isin(["g", "r", "i", "z"])
        & np.isfinite(frame["flux"])
        & np.isfinite(frame["fluxerr"])
        & (frame["fluxerr"] > 0)
    )

    frame = frame.loc[mask].copy()

    frame["survey"] = survey

    if survey == "DES-SN5YR":
        frame["stratum"] = [
            des_stratum(x)
            for x in frame["field"]
        ]
    else:
        frame["stratum"] = frame["field"]

    return frame


def summarize(frame):

    rows = []

    groups = [
        ("ALL", "ALL", frame),
    ]

    for band, g in frame.groupby("band"):
        groups.append(
            (band, "ALL", g)
        )

    for (band, stratum), g in frame.groupby(
        ["band", "stratum"]
    ):
        groups.append(
            (band, stratum, g)
        )

    for band, stratum, g in groups:

        for var in VARIABLES:

            if var not in g.columns:
                continue

            x = g[var].to_numpy(dtype=float)

            finite = np.isfinite(x)

            usable = x[finite]

            q = quantiles(usable)

            rows.append({
                "survey":
                    g["survey"].iloc[0],

                "band":
                    band,

                "stratum":
                    stratum,

                "variable":
                    var,

                "n_rows":
                    int(len(g)),

                "n_finite":
                    int(finite.sum()),

                "finite_fraction":
                    float(finite.mean()),

                "n_positive":
                    int(
                        np.sum(
                            usable > 0
                        )
                    ),

                "positive_fraction_of_finite":
                    (
                        float(
                            np.mean(
                                usable > 0
                            )
                        )
                        if len(usable)
                        else np.nan
                    ),

                "n_unique":
                    int(
                        len(
                            np.unique(
                                usable
                            )
                        )
                    ),

                **q,
            })

    return pd.DataFrame(rows)


def relationship_audit(frame):

    rows = []

    predictors = [
        "PSF_SIG1",
        "PSF_SIG2",
        "PSF_RATIO",
        "SKY_SIG",
        "SKY_SIG_T",
        "RDNOISE",
        "ZEROPT",
        "ZEROPT_ERR",
        "GAIN",
    ]

    for (band, stratum), g in frame.groupby(
        ["band", "stratum"]
    ):

        y = g["FLUXCALERR"].to_numpy(
            dtype=float
        )

        for var in predictors:

            if var not in g.columns:
                continue

            x = g[var].to_numpy(
                dtype=float
            )

            good = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            if good.sum() < 3:
                continue

            xv = x[good]
            yv = y[good]

            if (
                np.std(xv) == 0
                or np.std(yv) == 0
            ):
                pearson = np.nan
            else:
                pearson = float(
                    np.corrcoef(
                        xv,
                        yv,
                    )[0, 1]
                )

            # Rank correlation reveals monotonic
            # but nonlinear structure without
            # imposing a functional form.
            rank_x = pd.Series(xv).rank().to_numpy()
            rank_y = pd.Series(yv).rank().to_numpy()

            if (
                np.std(rank_x) == 0
                or np.std(rank_y) == 0
            ):
                spearman = np.nan
            else:
                spearman = float(
                    np.corrcoef(
                        rank_x,
                        rank_y,
                    )[0, 1]
                )

            rows.append({
                "survey":
                    g["survey"].iloc[0],

                "band":
                    band,

                "stratum":
                    stratum,

                "predictor":
                    var,

                "n":
                    int(good.sum()),

                "pearson_r":
                    pearson,

                "spearman_r":
                    spearman,
            })

    return pd.DataFrame(rows)


def main():

    print("=" * 78)
    print("PHASE 3 STAGE 3A")
    print("EPOCH-LEVEL MEASUREMENT / NOISE AUDIT")
    print("NO LABELS OR COMPACT-FEATURE OUTCOMES")
    print("=" * 78)

    print()
    print("[1] Loading structurally valid released epochs...")

    sdss = load(
        SDSS,
        "SDSS-II",
    )

    des = load(
        DES,
        "DES-SN5YR",
    )

    print(
        f"  SDSS valid griz epochs : {len(sdss):,}"
    )

    print(
        f"  DES valid griz epochs  : {len(des):,}"
    )

    print()
    print("[2] Auditing variable usability...")

    summary = pd.concat(
        [
            summarize(sdss),
            summarize(des),
        ],
        ignore_index=True,
    )

    print()
    print("Minimum finite fractions:")

    for survey in [
        "SDSS-II",
        "DES-SN5YR",
    ]:

        s = summary[
            summary["survey"] == survey
        ]

        print()
        print(survey)

        for var in VARIABLES:

            v = s[
                s["variable"] == var
            ]

            if len(v) == 0:
                print(
                    f"  {var:12s}: unavailable"
                )
                continue

            print(
                f"  {var:12s}: "
                f"{v['finite_fraction'].min():.6f}"
            )

    print()
    print(
        "[3] Auditing relation between released "
        "FLUXCALERR and instrumental metadata..."
    )

    relationships = pd.concat(
        [
            relationship_audit(sdss),
            relationship_audit(des),
        ],
        ignore_index=True,
    )

    print()
    print(
        "Strongest absolute rank relationships "
        "with FLUXCALERR:"
    )

    for survey in [
        "SDSS-II",
        "DES-SN5YR",
    ]:

        s = relationships[
            relationships["survey"]
            == survey
        ].copy()

        s["abs_spearman"] = (
            s["spearman_r"].abs()
        )

        s = s.sort_values(
            "abs_spearman",
            ascending=False,
        )

        print()
        print(survey)

        print(
            s[
                [
                    "band",
                    "stratum",
                    "predictor",
                    "n",
                    "spearman_r",
                ]
            ]
            .head(12)
            .to_string(
                index=False
            )
        )

    summary_path = (
        OUT
        / "epoch_metadata_summary.csv"
    )

    relationship_path = (
        OUT
        / "fluxerr_metadata_relationships.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    relationships.to_csv(
        relationship_path,
        index=False,
    )

    manifest = {
        "stage":
            "stage3a_epoch_measurement_model_audit",

        "class_labels_read":
            False,

        "compact_features_read":
            False,

        "cross_survey_feature_outcomes_read":
            False,

        "structural_mask":
            (
                "canonical griz, finite FLUXCAL, "
                "finite FLUXCALERR, FLUXCALERR > 0"
            ),

        "variables_audited":
            VARIABLES,

        "des_depth_mapping": {
            "DEEP": ["C3", "X3"],
            "SHALLOW": [
                "C1",
                "C2",
                "E1",
                "E2",
                "S1",
                "S2",
                "X1",
                "X2",
            ],
        },

        "purpose":
            (
                "Freeze the usable released "
                "instrumental/noise metadata before "
                "constructing the Stage-3 forward "
                "measurement operator."
            ),
    }

    manifest_path = (
        OUT
        / "stage3a_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print()
    print("=" * 78)
    print("STAGE 3A AUDIT: COMPLETE")
    print("=" * 78)

    print(
        f"Summary       : {summary_path}"
    )

    print(
        f"Relationships : {relationship_path}"
    )

    print(
        f"Manifest      : {manifest_path}"
    )

    print()
    print(
        "STOP HERE. Do not fit a noise model yet."
    )


if __name__ == "__main__":
    main()
