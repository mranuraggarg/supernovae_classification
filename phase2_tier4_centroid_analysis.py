#!/usr/bin/env python3
"""
Phase 2 Tier 4: Class-conditional centroid analysis.

This script compares whether SPCC and PLAsTiCC occupy the same compact
16-feature space class-conditionally.

It measures:
- SPCC Ia centroid
- PLAsTiCC Ia centroid
- SPCC non-Ia centroid
- PLAsTiCC non-Ia centroid

Main diagnostic:
If the SPCC Ia centroid and PLAsTiCC Ia centroid are far apart relative to
within-survey Ia/non-Ia separation, then SPCC -> PLAsTiCC transfer is
fundamentally limited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


COMPACT_FEATURES = [
    "r_mean_flux",
    "g_mean_flux",
    "z_peak_flux",
    "i_peak_flux",
    "peak_color_g_minus_r",
    "peak_color_r_minus_i",
    "peak_color_i_minus_z",
    "i_std_flux",
    "z_std_flux",
    "r_std_flux",
    "r_time_of_peak",
    "i_time_of_peak",
    "z_time_of_peak",
    "time_span",
    "r_peak_flux",
    "i_amplitude",
]


DEFAULT_SPCC_PATH = "data/processed/phase2_tier1_compact_baseline.csv"
DEFAULT_PLASTICC_PATH = "data/PLAsTiCC/features/train_compact_features.csv"
DEFAULT_RESULTS_DIR = "results/phase2_tier4_centroid_analysis"
DEFAULT_PLOTS_DIR = "plots/phase2_tier4_centroid_analysis"


def find_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "target",
        "true_target",
        "class",
        "label",
        "label_name",
        "label_id",
        "y",
        "is_ia",
        "is_Ia",
        "binary_target",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find label column. Expected one of: "
        + ", ".join(candidates)
        + f". Available columns: {list(df.columns)}"
    )


def make_binary_labels(series: pd.Series) -> pd.Series:
    """
    Convert labels to binary:
    Ia     -> 1
    non-Ia -> 0

    Handles common cases:
    - already binary 0/1
    - SPCC-style class 1 for Ia
    - PLAsTiCC-style target 90 for Ia
    - string labels containing Ia
    """

    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        unique_values = set(pd.Series(s.dropna().unique()).astype(int).tolist())

        if unique_values <= {0, 1}:
            return s.astype(int)

        if 90 in unique_values:
            return (s.astype(int) == 90).astype(int)

        if 1 in unique_values:
            return (s.astype(int) == 1).astype(int)

        raise ValueError(
            f"Numeric label column found, but Ia class could not be inferred. "
            f"Unique values: {sorted(unique_values)}"
        )

    s_lower = s.astype(str).str.lower()

    ia_mask = (
        s_lower.eq("ia")
        | s_lower.eq("type ia")
        | s_lower.eq("snia")
        | s_lower.eq("type_ia")
        | s_lower.eq("sn ia")
        | s_lower.str.contains("type ia", regex=False)
        | s_lower.str.contains("type_ia", regex=False)
    )

    return ia_mask.astype(int)


def validate_features(df: pd.DataFrame, dataset_name: str) -> None:
    missing = [f for f in COMPACT_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing compact features: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def clean_feature_table(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    validate_features(df, dataset_name)

    label_col = find_label_column(df)

    out = df[COMPACT_FEATURES].copy()
    out["is_ia"] = make_binary_labels(df[label_col])

    before = len(out)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=COMPACT_FEATURES + ["is_ia"])

    after = len(out)

    print(
        f"{dataset_name}: label_col={label_col}, "
        f"rows_before={before}, rows_after={after}, "
        f"Ia={int(out['is_ia'].sum())}, non-Ia={int((out['is_ia'] == 0).sum())}"
    )

    if out["is_ia"].nunique() < 2:
        raise ValueError(f"{dataset_name} does not contain both Ia and non-Ia labels.")

    return out


def centroid(df: pd.DataFrame, class_value: int) -> np.ndarray:
    subset = df[df["is_ia"] == class_value]
    return subset[COMPACT_FEATURES].mean(axis=0).to_numpy()


def centroid_frame(named_centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []

    for name, values in named_centroids.items():
        row = {"centroid": name}
        row.update(dict(zip(COMPACT_FEATURES, values)))
        rows.append(row)

    return pd.DataFrame(rows)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(pairwise_distances([a], [b], metric="cosine")[0, 0])


def build_distance_table(named_centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    comparisons = [
        (
            "cross_survey_ia_shift",
            "SPCC_Ia",
            "PLAsTiCC_Ia",
            "Distance between Ia centroids across surveys",
        ),
        (
            "cross_survey_non_ia_shift",
            "SPCC_non_Ia",
            "PLAsTiCC_non_Ia",
            "Distance between non-Ia centroids across surveys",
        ),
        (
            "spcc_class_separation",
            "SPCC_Ia",
            "SPCC_non_Ia",
            "Within-SPCC Ia vs non-Ia separation",
        ),
        (
            "plasticc_class_separation",
            "PLAsTiCC_Ia",
            "PLAsTiCC_non_Ia",
            "Within-PLAsTiCC Ia vs non-Ia separation",
        ),
        (
            "spcc_ia_to_plasticc_non_ia",
            "SPCC_Ia",
            "PLAsTiCC_non_Ia",
            "SPCC Ia centroid vs PLAsTiCC non-Ia centroid",
        ),
        (
            "plasticc_ia_to_spcc_non_ia",
            "PLAsTiCC_Ia",
            "SPCC_non_Ia",
            "PLAsTiCC Ia centroid vs SPCC non-Ia centroid",
        ),
    ]

    rows = []

    for name, left, right, interpretation in comparisons:
        a = named_centroids[left]
        b = named_centroids[right]

        rows.append(
            {
                "comparison": name,
                "left": left,
                "right": right,
                "euclidean_distance": euclidean(a, b),
                "cosine_distance": cosine_distance(a, b),
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def build_feature_shift_table(named_centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    spcc_ia = named_centroids["SPCC_Ia"]
    plasticc_ia = named_centroids["PLAsTiCC_Ia"]
    spcc_non_ia = named_centroids["SPCC_non_Ia"]
    plasticc_non_ia = named_centroids["PLAsTiCC_non_Ia"]

    rows = []

    for idx, feature in enumerate(COMPACT_FEATURES):
        ia_shift = plasticc_ia[idx] - spcc_ia[idx]
        non_ia_shift = plasticc_non_ia[idx] - spcc_non_ia[idx]

        rows.append(
            {
                "feature": feature,
                "spcc_ia_centroid": spcc_ia[idx],
                "plasticc_ia_centroid": plasticc_ia[idx],
                "ia_shift_plasticc_minus_spcc": ia_shift,
                "abs_ia_shift": abs(ia_shift),
                "spcc_non_ia_centroid": spcc_non_ia[idx],
                "plasticc_non_ia_centroid": plasticc_non_ia[idx],
                "non_ia_shift_plasticc_minus_spcc": non_ia_shift,
                "abs_non_ia_shift": abs(non_ia_shift),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_ia_shift", ascending=False)


def interpret_results(distance_df: pd.DataFrame) -> dict:
    d = distance_df.set_index("comparison")

    ia_shift = float(d.loc["cross_survey_ia_shift", "euclidean_distance"])
    non_ia_shift = float(d.loc["cross_survey_non_ia_shift", "euclidean_distance"])
    spcc_sep = float(d.loc["spcc_class_separation", "euclidean_distance"])
    plasticc_sep = float(d.loc["plasticc_class_separation", "euclidean_distance"])

    ia_shift_vs_spcc_sep = ia_shift / spcc_sep if spcc_sep > 0 else np.nan
    ia_shift_vs_plasticc_sep = ia_shift / plasticc_sep if plasticc_sep > 0 else np.nan

    if ia_shift >= spcc_sep or ia_shift >= plasticc_sep:
        conclusion = (
            "Ia centroids are not aligned across surveys. "
            "SPCC -> PLAsTiCC transfer is fundamentally limited in the compact feature space."
        )
    elif non_ia_shift > ia_shift:
        conclusion = (
            "Ia centroids are relatively better aligned than non-Ia centroids. "
            "The main transfer problem is likely contaminant-class mismatch."
        )
    else:
        conclusion = (
            "Ia centroid shift is smaller than within-survey class separation. "
            "Transfer limitation may be driven by boundary calibration, feature distribution width, "
            "or non-Ia mixture differences rather than Ia centroid mismatch alone."
        )

    return {
        "cross_survey_ia_shift": ia_shift,
        "cross_survey_non_ia_shift": non_ia_shift,
        "spcc_class_separation": spcc_sep,
        "plasticc_class_separation": plasticc_sep,
        "ia_shift_vs_spcc_class_separation": ia_shift_vs_spcc_sep,
        "ia_shift_vs_plasticc_class_separation": ia_shift_vs_plasticc_sep,
        "conclusion": conclusion,
    }


def plot_feature_shifts(feature_shift_df: pd.DataFrame, plots_dir: Path) -> None:
    top = feature_shift_df.sort_values("abs_ia_shift", ascending=True)

    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["ia_shift_plasticc_minus_spcc"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("PLAsTiCC Ia centroid - SPCC Ia centroid")
    plt.ylabel("Compact feature")
    plt.title("Class-conditional Ia centroid shift by feature")
    plt.tight_layout()
    plt.savefig(plots_dir / "centroid_shift_barplot.png", dpi=200)
    plt.close()


def plot_centroid_pca(named_centroids: dict[str, np.ndarray], plots_dir: Path) -> None:
    names = list(named_centroids.keys())
    matrix = np.vstack([named_centroids[name] for name in names])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)

    plt.figure(figsize=(7, 6))

    for name, xy in zip(names, coords):
        plt.scatter(xy[0], xy[1], s=100)
        plt.text(xy[0], xy[1], f"  {name}", va="center")

    plt.axhline(0, linewidth=1, linestyle="--")
    plt.axvline(0, linewidth=1, linestyle="--")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.title("PCA projection of class-conditional centroids")
    plt.tight_layout()
    plt.savefig(plots_dir / "centroid_pca.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Class-conditional centroid analysis for SPCC vs PLAsTiCC."
    )

    parser.add_argument("--spcc", default=DEFAULT_SPCC_PATH)
    parser.add_argument("--plasticc", default=DEFAULT_PLASTICC_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)

    args = parser.parse_args()

    spcc_path = Path(args.spcc)
    plasticc_path = Path(args.plasticc)
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading SPCC: {spcc_path}")
    print(f"Reading PLAsTiCC: {plasticc_path}")

    spcc_raw = pd.read_csv(spcc_path)
    plasticc_raw = pd.read_csv(plasticc_path)

    spcc = clean_feature_table(spcc_raw, "SPCC")
    plasticc = clean_feature_table(plasticc_raw, "PLAsTiCC")

    scaler = StandardScaler()
    scaler.fit(spcc[COMPACT_FEATURES])

    spcc_scaled = pd.DataFrame(
        scaler.transform(spcc[COMPACT_FEATURES]),
        columns=COMPACT_FEATURES,
        index=spcc.index,
    )
    spcc_scaled["is_ia"] = spcc["is_ia"].to_numpy()

    plasticc_scaled = pd.DataFrame(
        scaler.transform(plasticc[COMPACT_FEATURES]),
        columns=COMPACT_FEATURES,
        index=plasticc.index,
    )
    plasticc_scaled["is_ia"] = plasticc["is_ia"].to_numpy()

    named_centroids = {
        "SPCC_Ia": centroid(spcc_scaled, 1),
        "PLAsTiCC_Ia": centroid(plasticc_scaled, 1),
        "SPCC_non_Ia": centroid(spcc_scaled, 0),
        "PLAsTiCC_non_Ia": centroid(plasticc_scaled, 0),
    }

    centroids_df = centroid_frame(named_centroids)
    distance_df = build_distance_table(named_centroids)
    feature_shift_df = build_feature_shift_table(named_centroids)
    summary = interpret_results(distance_df)

    centroids_df.to_csv(results_dir / "centroids_standardized.csv", index=False)
    distance_df.to_csv(results_dir / "centroid_distances.csv", index=False)
    feature_shift_df.to_csv(results_dir / "feature_centroid_shifts.csv", index=False)

    with open(results_dir / "centroid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_feature_shifts(feature_shift_df, plots_dir)
    plot_centroid_pca(named_centroids, plots_dir)

    print("\nCentroid distance summary:")
    print(distance_df[["comparison", "euclidean_distance", "cosine_distance"]])

    print("\nInterpretation:")
    print(summary["conclusion"])

    print(f"\nSaved results to: {results_dir}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
