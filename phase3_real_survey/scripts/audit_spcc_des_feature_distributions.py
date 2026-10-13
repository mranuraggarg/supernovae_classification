#!/usr/bin/env python3
"""Stage 6: SPCC-DES feature-distribution audit.

Read-only scientific diagnostic.

- Uses the historical 16-feature SPCC training population.
- Uses the frozen 185-object DES cohort.
- Does not inspect DES classification labels.
- Does not load or execute the classifier.
- Does not modify feature extraction or preprocessing.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(
    "/Volumes/AstroSSD/share/github/"
    "supernovae_classification"
)

BACKUP = Path(
    "/Volumes/AstroSSD/share/github/"
    "supernovae_classification.backup.20260510-200046"
)

SPCC_PATH = (
    BACKUP / "data/processed/spcc_features_tier1.csv"
)

DES_PATH = (
    ROOT / "phase3_real_survey/results/"
    "historical_tier1_des/des_historical_tier1_16.csv"
)

AUDIT_PATH = (
    ROOT / "phase3_real_survey/results/"
    "historical_tier1_des/extraction_audit.json"
)

OUTPUT_DIR = (
    ROOT / "phase3_real_survey/results/"
    "historical_tier1_des/stage6_distribution_audit"
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


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def main():
    spcc = pd.read_csv(SPCC_PATH)
    des = pd.read_csv(DES_PATH)

    with open(AUDIT_PATH) as f:
        extraction_audit = json.load(f)

    assert extraction_audit["features"] == FEATURES
    assert len(des) == 185

    assert all(feature in spcc.columns for feature in FEATURES)
    assert list(des.columns) == ["object_id"] + FEATURES

    # Historical SPCC binary labels.
    #
    # The original split is classwise, with NumPy default_rng(42).
    # We deliberately reconstruct the partition rather than
    # drawing a new random split.

    # Original SPCC multiclass label encoding:
    # Ia = 1; all other supernova types are non-Ia.
    # Convert to the historical binary classification target.
    assert "label_id" in spcc.columns
    assert "label_name" in spcc.columns

    expected_mapping = {
        1: "Ia",
        2: "II",
        3: "Ibc",
        21: "IIn",
        22: "IIP",
        23: "IIL",
        32: "Ib",
        33: "Ic",
    }

    observed_mapping = (
        spcc[["label_id", "label_name"]]
        .drop_duplicates()
        .set_index("label_id")["label_name"]
        .to_dict()
    )

    assert observed_mapping == expected_mapping, (
        f"Unexpected SPCC label mapping: {observed_mapping}"
    )

    labels = (
        spcc["label_id"].to_numpy() == 1
    ).astype(np.int8)

    assert len(spcc) == 21319
    assert int(labels.sum()) == 5088
    assert int((labels == 0).sum()) == 16231

    rng = np.random.default_rng(42)

    trainval_indices = []
    test_indices = []

    for label in (0, 1):
        indices = np.flatnonzero(labels == label)

        rng.shuffle(indices)

        n_test = round(0.20 * len(indices))

        test_indices.extend(indices[:n_test])
        trainval_indices.extend(indices[n_test:])

    trainval_indices = np.array(
        sorted(trainval_indices),
        dtype=int,
    )

    test_indices = np.array(
        sorted(test_indices),
        dtype=int,
    )

    assert len(trainval_indices) == 17055
    assert len(test_indices) == 4264

    train = spcc.iloc[trainval_indices]

    x_spcc = train[FEATURES].to_numpy(dtype=np.float64)
    x_des = des[FEATURES].to_numpy(dtype=np.float64)

    assert np.isfinite(x_spcc).all()
    assert np.isfinite(x_des).all()

    rows = []

    outside_matrix = np.zeros(
        (len(des), len(FEATURES)),
        dtype=bool,
    )

    for j, feature in enumerate(FEATURES):
        a = x_spcc[:, j]
        b = x_des[:, j]

        sa = summary(a)
        sb = summary(b)

        iqr = sa["q75"] - sa["q25"]

        if iqr > 0:
            median_shift_iqr = (
                (sb["median"] - sa["median"]) / iqr
            )
        else:
            median_shift_iqr = np.nan

        outside = (
            (b < sa["min"]) |
            (b > sa["max"])
        )

        outside_matrix[:, j] = outside

        ks = ks_2samp(
            a,
            b,
            alternative="two-sided",
            method="asymp",
        )

        rows.append({
            "feature": feature,
            "spcc_min": sa["min"],
            "spcc_q25": sa["q25"],
            "spcc_median": sa["median"],
            "spcc_q75": sa["q75"],
            "spcc_max": sa["max"],
            "des_min": sb["min"],
            "des_q25": sb["q25"],
            "des_median": sb["median"],
            "des_q75": sb["q75"],
            "des_max": sb["max"],
            "median_shift_spcc_iqr": median_shift_iqr,
            "des_outside_spcc_range_count": int(outside.sum()),
            "des_outside_spcc_range_pct": (
                100.0 * outside.mean()
            ),
            "ks_statistic": float(ks.statistic),
        })

    results = pd.DataFrame(rows)

    object_outside = outside_matrix.any(axis=1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        OUTPUT_DIR / "feature_distribution_comparison.csv"
    )

    results.to_csv(
        table_path,
        index=False,
        float_format="%.12g",
    )

    report = {
        "stage": "Stage 6 SPCC-DES distribution audit",
        "spcc_reference_population": "historical train+validation",
        "spcc_reference_count": len(train),
        "spcc_heldout_test_excluded": len(test_indices),
        "des_count": len(des),
        "feature_count": len(FEATURES),
        "des_labels_used": False,
        "classifier_executed": False,
        "objects_outside_any_training_feature_range": int(
            object_outside.sum()
        ),
        "objects_outside_any_training_feature_range_pct": (
            100.0 * object_outside.mean()
        ),
        "inputs_sha256": {
            "spcc": sha256(SPCC_PATH),
            "des": sha256(DES_PATH),
            "des_extraction_audit": sha256(AUDIT_PATH),
        },
    }

    report_path = OUTPUT_DIR / "distribution_audit.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("STAGE 6: SPCC-DES FEATURE-DISTRIBUTION AUDIT")
    print("=" * 70)

    print("SPCC training+validation:", len(train))
    print("SPCC held-out test excluded:", len(test_indices))
    print("DES objects:", len(des))
    print("Features:", len(FEATURES))

    print()
    print(
        "DES objects outside at least one SPCC training range:",
        int(object_outside.sum()),
        f"({100 * object_outside.mean():.2f}%)",
    )

    print()
    print("FEATURE DISTRIBUTION DIAGNOSTICS")
    print("-" * 70)

    display = results[
        [
            "feature",
            "median_shift_spcc_iqr",
            "des_outside_spcc_range_pct",
            "ks_statistic",
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Feature comparison:", table_path)
    print("Audit report:", report_path)

    print()
    print("STAGE 6 COMPUTATION: PASS")
    print("Scientific interpretation: PENDING")
    print("Classifier predictions: NOT RUN")


if __name__ == "__main__":
    main()
