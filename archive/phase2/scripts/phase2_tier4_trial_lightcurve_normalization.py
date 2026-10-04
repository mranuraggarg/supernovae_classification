#!/usr/bin/env python3
"""
Phase 2 Tier 4 trial:
Light-curve-level normalization before compact-feature extraction.

Goal:
Test whether SPCC -> PLAsTiCC transfer improves when both surveys are
normalized at the raw light-curve level before compact features are built.

Important: this script currently only writes normalized raw PLAsTiCC light-curve tables. It does not yet rebuild compact feature tables or connect those tables to the Tier-4 domain-swap scripts. Until the compact-feature builder is wired in below, running phase2_tier4_trial_domain_swap.py will still evaluate the existing trial feature tables rather than these normalized raw files.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


OUT_DIR = Path("results/phase2_tier4_trial_lc_norm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUT_DIR = Path("data/phase2_tier4_trial_lc_norm")
FEATURE_OUT_DIR.mkdir(parents=True, exist_ok=True)


NORMALIZATION_MODES = [
    "none",
    "event_peak",
    "band_peak",
    "event_p95",
]


def safe_scale(x: float, eps: float = 1e-6) -> float:
    if not np.isfinite(x) or abs(x) < eps:
        return 1.0
    return float(x)


def normalize_lightcurve(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Expected columns:
        object_id or snid
        band
        mjd
        flux
        fluxerr

    Applies normalization before compact-feature extraction.
    """

    df = df.copy()

    if mode == "none":
        df["flux_norm"] = df["flux"]
        df["fluxerr_norm"] = df["fluxerr"]
        return df

    if mode == "event_peak":
        scale = safe_scale(np.nanmax(np.abs(df["flux"].values)))
        df["flux_norm"] = df["flux"] / scale
        df["fluxerr_norm"] = df["fluxerr"] / scale
        return df

    if mode == "event_p95":
        scale = safe_scale(np.nanpercentile(np.abs(df["flux"].values), 95))
        df["flux_norm"] = df["flux"] / scale
        df["fluxerr_norm"] = df["fluxerr"] / scale
        return df

    if mode == "band_peak":
        df["flux_norm"] = np.nan
        df["fluxerr_norm"] = np.nan

        for band, sub in df.groupby("band"):
            scale = safe_scale(np.nanmax(np.abs(sub["flux"].values)))
            idx = sub.index
            df.loc[idx, "flux_norm"] = df.loc[idx, "flux"] / scale
            df.loc[idx, "fluxerr_norm"] = df.loc[idx, "fluxerr"] / scale

        return df

    raise ValueError(f"Unknown normalization mode: {mode}")


def normalize_by_object(raw_df: pd.DataFrame, mode: str, object_col: str) -> pd.DataFrame:
    parts = []

    for _, obj_df in raw_df.groupby(object_col):
        parts.append(normalize_lightcurve(obj_df, mode))

    return pd.concat(parts, ignore_index=True)


def build_trial_features(mode: str):
    """
    You need to connect this section to your current feature builder.

    Replace the placeholder paths and builder calls with the actual functions
    from phase2_tier4_make_variants.py.
    """

    # Example raw paths. Adjust if your repo uses different names.
    spcc_raw_path = Path("data/spcc/raw_observations.csv")
    plasticc_raw_path = Path("data/PLAsTiCC/training_set.csv")
    plasticc_meta_path = Path("data/PLAsTiCC/training_set_metadata.csv")

    # Load PLAsTiCC raw table.
    plasticc = pd.read_csv(plasticc_raw_path)

    # Standardize PLAsTiCC column names if needed.
    plasticc = plasticc.rename(
        columns={
            "object_id": "object_id",
            "passband": "band",
            "mjd": "mjd",
            "flux": "flux",
            "flux_err": "fluxerr",
        }
    )

    plasticc_norm = normalize_by_object(
        plasticc,
        mode=mode,
        object_col="object_id",
    )

    plasticc_norm["flux_original"] = plasticc_norm["flux"]
    plasticc_norm["fluxerr_original"] = plasticc_norm["fluxerr"]
    plasticc_norm["flux"] = plasticc_norm["flux_norm"]
    plasticc_norm["fluxerr"] = plasticc_norm["fluxerr_norm"]

    norm_raw_path = FEATURE_OUT_DIR / f"plasticc_raw_normalized_{mode}.csv"
    plasticc_norm.to_csv(norm_raw_path, index=False)

    print(f"[OK] Wrote normalized PLAsTiCC raw table: {norm_raw_path}")
    print(
        "[WARN] This mode has only produced a normalized raw PLAsTiCC table. "
        "It has NOT yet produced a compact-feature table for domain-swap testing."
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Hook this into your existing compact-feature builder.
    #
    # Example placeholder:
    #
    # from phase2_tier4_make_variants import build_plasticc_compact_features
    #
    # out_path = FEATURE_OUT_DIR / f"plasticc_compact_{mode}.csv"
    # build_plasticc_compact_features(
    #     training_set_path=norm_raw_path,
    #     metadata_path=plasticc_meta_path,
    #     output_path=out_path,
    # )
    #
    # return out_path
    # ------------------------------------------------------------------

    return norm_raw_path


def main():
    manifest = {}

    for mode in NORMALIZATION_MODES:
        print(f"\n=== Running light-curve normalization mode: {mode} ===")
        out_path = build_trial_features(mode)
        manifest[mode] = str(out_path)

    manifest_path = OUT_DIR / "lightcurve_normalization_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Wrote manifest: {manifest_path}")
    print(
        "\n[IMPORTANT] The light-curve-normalization trial is not complete until "
        "these normalized raw tables are passed through the same compact-feature "
        "builder used by phase2_tier4_make_variants.py."
    )


if __name__ == "__main__":
    main()