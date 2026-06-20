"""Generate paper-ready Phase 2 Section 5 and 6 material from final outputs."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2_tier2_common import COMPACT_FEATURES, FROZEN_BASELINE_METRICS
from phase2_tier3_model_compare import build_train_test_split, train_and_evaluate_model


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "phase2_tier4"
PLOTS_DIR = ROOT / "plots" / "phase2_tier4"

SPCC_COMPACT_PATH = ROOT / "data" / "spcc" / "features" / "compact_features.csv"
PLASTICC_ORIGINAL_PATH = ROOT / "data" / "PLAsTiCC" / "features" / "compact_features.csv"
PLASTICC_PARITY_PATH = ROOT / "data" / "PLAsTiCC" / "features" / "train_compact_features.csv"

TIER2_MINIMAL_PATH = ROOT / "results" / "phase2_tier2" / "minimal_core_metrics.csv"
TIER3_MODEL_COMPARE_PATH = ROOT / "results" / "phase2_tier3" / "model_compare_metrics.csv"
TIER3_CV_PATH = ROOT / "results" / "phase2_tier3" / "cv_stability_metrics.csv"
TIER3_NOISE_PATH = ROOT / "results" / "phase2_tier3" / "noise_test_metrics.csv"
TIER4_SHIFT_PATH = ROOT / "results" / "phase2_tier4" / "shift_test_metrics.csv"

PAPER_MD_PATH = RESULTS_DIR / "paper_sections_5_6.md"
SUMMARY_CSV_PATH = RESULTS_DIR / "paper_results_summary.csv"
REDUCED_CSV_PATH = RESULTS_DIR / "paper_reduced_training_metrics.csv"

FEATURE_FAMILIES = [
    ("z_peak_flux", "Brightness", "z-band peak flux"),
    ("peak_color_r_minus_i", "Color", "peak r-i color"),
    ("time_span", "Time", "time span"),
    ("z_std_flux", "Variability", "z-band flux scatter"),
]

FEATURE_LABELS = {
    "z_peak_flux": "z peak flux",
    "r_mean_flux": "r mean flux",
    "peak_color_g_minus_r": "g-r color",
    "i_peak_flux": "i peak flux",
    "peak_color_r_minus_i": "r-i color",
    "peak_color_i_minus_z": "i-z color",
    "g_mean_flux": "g mean flux",
    "r_peak_flux": "r peak flux",
    "z_std_flux": "z flux std",
    "i_amplitude": "i amplitude",
    "i_std_flux": "i flux std",
    "time_span": "time span",
    "z_time_of_peak": "z peak time",
    "i_time_of_peak": "i peak time",
    "r_time_of_peak": "r peak time",
    "r_std_flux": "r flux std",
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key == "label_name":
                    row[key] = value
                elif key in {"snid", "label_id"}:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def format_metric(value: float) -> str:
    return f"{value:.6f}"


def class_stats(rows: list[dict[str, Any]], feature_name: str) -> dict[str, float]:
    ia = np.array([float(row[feature_name]) for row in rows if row["label_name"] == "Ia"], dtype=float)
    non = np.array([float(row[feature_name]) for row in rows if row["label_name"] != "Ia"], dtype=float)
    pooled = math.sqrt(
        (((len(ia) - 1) * float(np.var(ia, ddof=1))) + ((len(non) - 1) * float(np.var(non, ddof=1))))
        / max(len(ia) + len(non) - 2, 1)
    )
    effect_size = (float(np.mean(ia)) - float(np.mean(non))) / pooled if pooled else 0.0
    return {
        "ia_median": float(np.median(ia)),
        "non_median": float(np.median(non)),
        "ia_mean": float(np.mean(ia)),
        "non_mean": float(np.mean(non)),
        "effect_size": effect_size,
    }


def top_correlations(rows: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    corr = frame[COMPACT_FEATURES].corr()
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(COMPACT_FEATURES):
        for right in COMPACT_FEATURES[i + 1 :]:
            value = float(corr.loc[left, right])
            pairs.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    pairs.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return pairs[:limit]


def plot_feature_distributions(rows: list[dict[str, Any]]) -> Path:
    frame = pd.DataFrame(rows)
    frame["class_label"] = np.where(frame["label_name"] == "Ia", "Ia", "non-Ia")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (feature_name, family_name, title_label) in zip(axes.flatten(), FEATURE_FAMILIES):
        grouped = [
            frame.loc[frame["class_label"] == "Ia", feature_name].to_numpy(),
            frame.loc[frame["class_label"] == "non-Ia", feature_name].to_numpy(),
        ]
        violin = axis.violinplot(grouped, positions=[1, 2], showmeans=False, showmedians=True, widths=0.8)
        for body, color in zip(violin["bodies"], ["#2563eb", "#b91c1c"]):
            body.set_facecolor(color)
            body.set_alpha(0.45)
        violin["cmedians"].set_color("#111827")
        axis.boxplot(
            grouped,
            positions=[1, 2],
            widths=0.22,
            patch_artist=True,
            boxprops={"facecolor": "#ffffff", "alpha": 0.85},
            medianprops={"color": "#111827", "linewidth": 1.3},
        )
        axis.set_xticks([1, 2], ["Ia", "non-Ia"])
        axis.set_title(f"{family_name}: {title_label}")
        axis.grid(alpha=0.18, axis="y")
    fig.suptitle("Feature distributions by class on the SPCC compact set", fontsize=14)
    fig.tight_layout()
    output_path = PLOTS_DIR / "paper_feature_distribution_vs_class.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(rows: list[dict[str, Any]]) -> Path:
    frame = pd.DataFrame(rows)
    corr = frame[COMPACT_FEATURES].corr().to_numpy()
    labels = [FEATURE_LABELS.get(name, name) for name in COMPACT_FEATURES]

    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=70, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    axis.set_title("Compact-feature correlation matrix")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    output_path = PLOTS_DIR / "paper_feature_correlation_heatmap.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_shift_sensitivity(shift_frame: pd.DataFrame) -> Path:
    ordered = shift_frame.set_index("shift_name").loc[["noise", "short_span", "flux_scale", "no_z", "no_i"]].reset_index()
    labels = ["noise", "short span", "flux scale", "no z", "no i"]

    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    bars = axis.bar(labels, ordered["f1"], color=["#15803d", "#65a30d", "#ca8a04", "#dc2626", "#991b1b"])
    axis.set_ylim(0.0, 0.9)
    axis.set_ylabel("F1")
    axis.set_title("Tier-4 shift sensitivity")
    axis.grid(alpha=0.2, axis="y")
    for bar, f1_drop in zip(bars, ordered["f1_drop"]):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"drop {f1_drop:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path = PLOTS_DIR / "paper_shift_sensitivity.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_reduced_training_rows() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label, path in [("original_compact", PLASTICC_ORIGINAL_PATH), ("parity_safe_compact", PLASTICC_PARITY_PATH)]:
        rows = load_feature_rows(path)
        train_x, train_y, test_x, test_y = build_train_test_split(rows, COMPACT_FEATURES)
        result = train_and_evaluate_model("xgboost", train_x, train_y, test_x, test_y, COMPACT_FEATURES)
        metrics = result["metrics"]
        records.append(
            {
                "dataset_label": label,
                "train_row_count": len(rows),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
            }
        )
    return pd.DataFrame(records).sort_values("train_row_count", ascending=False).reset_index(drop=True)


def plot_reduced_training(reduced_frame: pd.DataFrame) -> Path:
    ordered = reduced_frame.sort_values("train_row_count")
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(ordered["train_row_count"], ordered["f1"], marker="o", color="#1d4ed8", label="F1")
    axis.plot(ordered["train_row_count"], ordered["roc_auc"], marker="s", color="#047857", label="ROC-AUC")
    axis.plot(ordered["train_row_count"], ordered["pr_auc"], marker="^", color="#b45309", label="PR-AUC")
    axis.set_xlabel("Training-row count")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Reduced training-set sensitivity")
    axis.grid(alpha=0.2)
    for _, row in ordered.iterrows():
        axis.text(row["train_row_count"], row["f1"] + 0.015, str(int(row["train_row_count"])), ha="center", fontsize=8)
    axis.legend(frameon=False)
    fig.tight_layout()
    output_path = PLOTS_DIR / "paper_reduced_training_set.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def best_model_row(model_compare: pd.DataFrame) -> pd.Series:
    return model_compare.sort_values(["pr_auc", "f1"], ascending=False).iloc[0]


def best_cv_row(cv_frame: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        cv_frame.groupby("protocol")[["f1", "roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "protocol",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
    ]
    best = grouped.sort_values("f1_mean", ascending=False).iloc[0]
    return {name: best[name] for name in grouped.columns}


def choose_noise_row(noise_frame: pd.DataFrame) -> pd.Series:
    return noise_frame.loc[noise_frame["scenario_name"] == "flux_noise_0p25sigma"].iloc[0]


def choose_minimal_row(minimal_frame: pd.DataFrame) -> pd.Series:
    return minimal_frame.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]


def choose_shift_row(shift_frame: pd.DataFrame) -> pd.Series:
    return shift_frame.sort_values("f1_drop", ascending=False).iloc[0]


def build_summary_rows(
    model_row: pd.Series,
    cv_row: dict[str, Any],
    noise_row: pd.Series,
    minimal_row: pd.Series,
    shift_row: pd.Series,
    reduced_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    reduced_best = reduced_frame.sort_values("train_row_count", ascending=False).iloc[0]
    reduced_small = reduced_frame.sort_values("train_row_count").iloc[0]
    reduced_delta_f1 = float(reduced_small["f1"] - reduced_best["f1"])
    reduced_delta_roc = float(reduced_small["roc_auc"] - reduced_best["roc_auc"])
    reduced_delta_pr = float(reduced_small["pr_auc"] - reduced_best["pr_auc"])

    return [
        {
            "test": "baseline",
            "f1": float(FROZEN_BASELINE_METRICS["f1"]),
            "roc_auc": float(FROZEN_BASELINE_METRICS["roc_auc"]),
            "pr_auc": float(FROZEN_BASELINE_METRICS["pr_auc"]),
            "comment": "Tier-1 frozen 16-feature compact baseline.",
        },
        {
            "test": "model compare",
            "f1": float(model_row["f1"]),
            "roc_auc": float(model_row["roc_auc"]),
            "pr_auc": float(model_row["pr_auc"]),
            "comment": f"Best Tier-3 model: {model_row['model_label']}.",
        },
        {
            "test": "CV",
            "f1": float(cv_row["f1_mean"]),
            "roc_auc": float(cv_row["roc_auc_mean"]),
            "pr_auc": float(cv_row["pr_auc_mean"]),
            "comment": f"Best resampling protocol: {cv_row['protocol']} (mean over runs).",
        },
        {
            "test": "noise",
            "f1": float(noise_row["f1"]),
            "roc_auc": float(noise_row["roc_auc"]),
            "pr_auc": float(noise_row["pr_auc"]),
            "comment": "Tier-3 moderate flux-noise perturbation (+0.25 sigma).",
        },
        {
            "test": "reduced train",
            "f1": float(reduced_small["f1"]),
            "roc_auc": float(reduced_small["roc_auc"]),
            "pr_auc": float(reduced_small["pr_auc"]),
            "comment": (
                f"Parity-safe PLAsTiCC table ({int(reduced_small['train_row_count'])} rows); "
                f"vs {int(reduced_best['train_row_count'])} rows: delta F1 {reduced_delta_f1:+.3f}, "
                f"delta ROC {reduced_delta_roc:+.3f}, delta PR {reduced_delta_pr:+.3f}."
            ),
        },
        {
            "test": "minimal features",
            "f1": float(minimal_row["f1"]),
            "roc_auc": float(minimal_row["roc_auc"]),
            "pr_auc": float(minimal_row["pr_auc"]),
            "comment": f"Best reduced subset: {minimal_row['subset_name']} ({int(minimal_row['feature_count'])} features).",
        },
        {
            "test": "shift",
            "f1": float(shift_row["f1"]),
            "roc_auc": float(shift_row["roc_auc"]),
            "pr_auc": float(shift_row["pr_auc"]),
            "comment": f"Hardest Tier-4 shift: {shift_row['shift_name']} (F1 drop {float(shift_row['f1_drop']):.3f}).",
        },
    ]


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(SUMMARY_CSV_PATH, index=False)


def write_reduced_csv(reduced_frame: pd.DataFrame) -> None:
    reduced_frame.to_csv(REDUCED_CSV_PATH, index=False)


def write_markdown(
    *,
    distribution_plot: Path,
    correlation_plot: Path,
    shift_plot: Path,
    reduced_plot: Path,
    representative_stats: dict[str, dict[str, float]],
    correlations: list[dict[str, Any]],
    reduced_frame: pd.DataFrame,
    summary_rows: list[dict[str, Any]],
) -> None:
    reduced_best = reduced_frame.sort_values("train_row_count", ascending=False).iloc[0]
    reduced_small = reduced_frame.sort_values("train_row_count").iloc[0]
    reduced_delta_f1 = float(reduced_small["f1"] - reduced_best["f1"])
    reduced_delta_roc = float(reduced_small["roc_auc"] - reduced_best["roc_auc"])
    reduced_delta_pr = float(reduced_small["pr_auc"] - reduced_best["pr_auc"])

    lines = [
        "# Paper Sections 5 and 6 Draft",
        "",
        "## 5 Do these features correspond to real light-curve physics?",
        "",
        "The Tier-3 and Tier-4 results support a physical interpretation rather than a purely statistical one. "
        "Class separation appears directly in feature distributions, the feature families correlate in physically sensible ways, "
        "and performance under controlled shifts degrades gradually instead of collapsing immediately.",
        "",
        "### 5.1 Feature distribution vs class",
        "",
        f"![Feature distribution vs class]({distribution_plot})",
        "",
        "Representative SPCC class splits:",
    ]

    for feature_name, family_name, title_label in FEATURE_FAMILIES:
        stats = representative_stats[feature_name]
        lines.append(
            f"- **{family_name} ({title_label})**: Ia median {stats['ia_median']:.3f}, "
            f"non-Ia median {stats['non_median']:.3f}, Cohen d {stats['effect_size']:.3f}."
        )

    lines.extend(
        [
            "",
            "These distributions show that the compact features are not random latent coordinates. "
            "Brightness and variability proxies shift upward for Ia events, peak-color features show a visibly different class locus, "
            "and the temporal coverage differs by class rather than collapsing onto one common distribution.",
            "",
            "### 5.2 Correlation between features",
            "",
            f"![Feature correlation heatmap]({correlation_plot})",
            "",
            "Strongest correlations in the SPCC compact table:",
        ]
    )

    for row in correlations:
        lines.append(
            f"- `{row['feature_a']}` vs `{row['feature_b']}`: Pearson r = {row['correlation']:.3f}."
        )

    lines.extend(
        [
            "",
            "The correlation structure is also physically plausible. Flux-amplitude and flux-scatter features cluster strongly, "
            "same-band mean and peak brightness move together, and neighboring-band brightness features remain highly coupled. "
            "This is the pattern expected when the compact table is preserving light-curve scale and shape information instead of arbitrary dataset idiosyncrasies.",
            "",
            "### 5.3 Physical interpretation",
            "",
            f"- **Brightness**: `z_peak_flux` and `r_mean_flux` remain among the strongest Tier-2/Tier-3 signals. "
            f"`z_peak_flux` separates Ia from non-Ia with Cohen d {representative_stats['z_peak_flux']['effect_size']:.3f}, "
            "consistent with class-dependent luminosity scale near peak.",
            f"- **Color**: `peak_color_r_minus_i` is one of the most stable high-importance features across gain, permutation, SHAP, and ablation analyses. "
            f"Ia events show a higher median `r-i` proxy ({representative_stats['peak_color_r_minus_i']['ia_median']:.3f} vs {representative_stats['peak_color_r_minus_i']['non_median']:.3f}), "
            "which is consistent with color acting as a coarse spectral-slope indicator.",
            f"- **Time**: `time_span` carries a smaller but still real class effect. "
            f"The Ia median span is {representative_stats['time_span']['ia_median']:.3f}, compared with {representative_stats['time_span']['non_median']:.3f} for non-Ia, "
            "showing that temporal coverage contributes useful discriminative structure instead of pure sampling noise.",
            f"- **Variability**: `z_std_flux` and `i_std_flux` stay important after multiple checks, "
            f"with `z_std_flux` showing Cohen d {representative_stats['z_std_flux']['effect_size']:.3f}. "
            "That supports the idea that the compact table is retaining information about rise/decline strength and band-wise curve structure.",
            "",
            "### 5.4 Sensitivity to shift",
            "",
            f"![Shift sensitivity]({shift_plot})",
            "",
            "Tier-4 shift results show a graded rather than catastrophic failure mode. "
            "Noise and shortened time span produce only small drops, flux scaling produces a moderate drop, and removing full bands hurts much more strongly. "
            "This hierarchy is physically sensible: the model tolerates mild perturbations but depends strongly on cross-band structure.",
            "",
            "### 5.5 Reduced training set test",
            "",
            f"![Reduced training set sensitivity]({reduced_plot})",
            "",
            f"The PLAsTiCC compact training table shrinks from {int(reduced_best['train_row_count'])} rows in the original extraction to "
            f"{int(reduced_small['train_row_count'])} rows in the parity-safe extraction. "
            f"Across that reduction, F1 changes from {reduced_best['f1']:.6f} to {reduced_small['f1']:.6f} "
            f"(delta {reduced_delta_f1:+.6f}), while ROC-AUC changes by {reduced_delta_roc:+.6f} "
            f"and PR-AUC changes by {reduced_delta_pr:+.6f}.",
            "",
            "The main Tier-4 interpretation is therefore not a collapse but a broad stability result: "
            "scores stay in the same range even after the usable training set is reduced. "
            "That is more consistent with real retained signal than with a fragile shortcut that disappears as soon as the sample changes.",
            "",
            "## 6 Results summary",
            "",
            "| Test | F1 | ROC | PR | Comment |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )

    for row in summary_rows:
        lines.append(
            f"| {row['test']} | {row['f1']:.6f} | {row['roc_auc']:.6f} | {row['pr_auc']:.6f} | {row['comment']} |"
        )

    lines.extend(
        [
            "",
            "Overall, the final Phase-2 picture is consistent across tiers: the compact features are competitive with the full baseline, "
            "remain stable under resampling, retain useful performance under noise and reduced feature subsets, and degrade in a structured way under domain shift. "
            "That combination supports the claim that the retained features correspond to real light-curve physics, even though Tier-4 still shows that survey transfer is not fully solved.",
            "",
        ]
    )

    PAPER_MD_PATH.write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()

    spcc_rows = load_feature_rows(SPCC_COMPACT_PATH)
    tier2_minimal = load_csv(TIER2_MINIMAL_PATH)
    tier3_models = load_csv(TIER3_MODEL_COMPARE_PATH)
    tier3_cv = load_csv(TIER3_CV_PATH)
    tier3_noise = load_csv(TIER3_NOISE_PATH)
    tier4_shift = load_csv(TIER4_SHIFT_PATH)

    distribution_plot = plot_feature_distributions(spcc_rows)
    correlation_plot = plot_correlation_heatmap(spcc_rows)
    shift_plot = plot_shift_sensitivity(tier4_shift)
    reduced_frame = evaluate_reduced_training_rows()
    reduced_plot = plot_reduced_training(reduced_frame)

    representative_stats = {feature_name: class_stats(spcc_rows, feature_name) for feature_name, _, _ in FEATURE_FAMILIES}
    correlations = top_correlations(spcc_rows)

    model_row = best_model_row(tier3_models)
    cv_row = best_cv_row(tier3_cv)
    noise_row = choose_noise_row(tier3_noise)
    minimal_row = choose_minimal_row(tier2_minimal)
    shift_row = choose_shift_row(tier4_shift)
    summary_rows = build_summary_rows(model_row, cv_row, noise_row, minimal_row, shift_row, reduced_frame)

    write_summary_csv(summary_rows)
    write_reduced_csv(reduced_frame)
    write_markdown(
        distribution_plot=distribution_plot,
        correlation_plot=correlation_plot,
        shift_plot=shift_plot,
        reduced_plot=reduced_plot,
        representative_stats=representative_stats,
        correlations=correlations,
        reduced_frame=reduced_frame,
        summary_rows=summary_rows,
    )

    print(
        {
            "paper_markdown": str(PAPER_MD_PATH),
            "summary_csv": str(SUMMARY_CSV_PATH),
            "reduced_csv": str(REDUCED_CSV_PATH),
        }
    )


if __name__ == "__main__":
    main()
