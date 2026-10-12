#!/usr/bin/env python3
"""Create Stage 3C scientific tables and figures from frozen results only.

This reporter does not open production Monte Carlo records and does not
recompute bootstrap statistics. It summarizes the immutable analysis JSON.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "phase3_real_survey"
RESULT_DIR = ROOT / "results" / "stage3c_step5_statistical_analysis"
ANALYSIS_PATH = RESULT_DIR / "stage3c_step5_analysis.json"
MANIFEST_PATH = RESULT_DIR / "stage3c_step5_analysis_manifest.json"
OUTPUT_DIR = ROOT / "docs" / "analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

FEATURES = (
    "peak_color_r_minus_i",
    "peak_color_i_minus_z",
    "i_peak_flux",
    "z_peak_flux",
    "r_time_of_peak",
    "time_span",
)
FEATURE_LABELS = {
    "peak_color_r_minus_i": "Peak colour proxy r-i (mag)",
    "peak_color_i_minus_z": "Peak colour proxy i-z (mag)",
    "i_peak_flux": "i peak flux [log10(1+FLUXCAL)]",
    "z_peak_flux": "z peak flux [log10(1+FLUXCAL)]",
    "r_time_of_peak": "r time of peak (days)",
    "time_span": "Active time span (days)",
}
COMPARISON_STYLES = {
    "DES-SN5YR/SHALLOW - SDSS-II/82N": ("#2166ac", "SHALLOW - 82N"),
    "DES-SN5YR/SHALLOW - SDSS-II/82S": ("#67a9cf", "SHALLOW - 82S"),
    "DES-SN5YR/DEEP - SDSS-II/82N": ("#b2182b", "DEEP - 82N"),
    "DES-SN5YR/DEEP - SDSS-II/82S": ("#ef8a62", "DEEP - 82S"),
}
BRANCH_STYLES = {
    ("SDSS-II", "82N"): "#2166ac",
    ("SDSS-II", "82S"): "#67a9cf",
    ("DES-SN5YR", "SHALLOW"): "#b2182b",
    ("DES-SN5YR", "DEEP"): "#ef8a62",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_results():
    manifest = json.loads(MANIFEST_PATH.read_text())
    actual = sha256(ANALYSIS_PATH)
    expected = manifest["analysis_output_sha256"]
    if actual != expected:
        raise RuntimeError(
            f"Analysis checksum mismatch: expected {expected}, found {actual}"
        )
    return json.loads(ANALYSIS_PATH.read_text()), manifest, actual


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


def save_figure(fig, stem):
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def index_results(result):
    feature_rows = {
        (
            row["comparison"], row["feature"],
            row["sbmag_mag_per_arcsec2"], row["redshift"],
        ): row
        for row in result["primary_and_supportive_feature_statistics"]
    }
    feature_bands = {}
    for band in result["feature_simultaneous_confidence_bands"]:
        for node in band["nodes"]:
            feature_bands[(
                band["comparison"], band["feature"],
                band["sbmag_mag_per_arcsec2"], node["redshift"],
            )] = node

    timing_rows = {
        (
            row["comparison"], row["metric"],
            row["sbmag_mag_per_arcsec2"], row["redshift"],
        ): row
        for row in result["timing_comparisons"]
    }
    timing_bands = {}
    for band in result["timing_simultaneous_confidence_bands"]:
        for node in band["nodes"]:
            timing_bands[(
                band["comparison"], band["metric"],
                band["sbmag_mag_per_arcsec2"], node["redshift"],
            )] = node
    return feature_rows, feature_bands, timing_rows, timing_bands


def registered_prediction(result, feature):
    nodes = result["physical_prediction_metadata"][
        "synthetic_passband_predictions"
    ]["nodes"]
    if feature not in ("peak_color_r_minus_i", "peak_color_i_minus_z"):
        return None
    return {
        node["redshift"]: node[feature]["des_minus_sdss"]
        for node in nodes
    }


def plot_feature(result, feature_rows, feature_bands, feature):
    redshifts = result["configuration"]["redshifts"]
    sbmags = result["configuration"]["sbmags"]
    prediction = registered_prediction(result, feature)
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.2), sharex=True, sharey=True)
    offsets = np.linspace(-0.0017, 0.0017, len(COMPARISON_STYLES))

    for axis, sbmag in zip(axes.flat, sbmags):
        for offset, (comparison, (color, label)) in zip(
            offsets, COMPARISON_STYLES.items()
        ):
            rows = [
                feature_rows[(comparison, feature, sbmag, redshift)]
                for redshift in redshifts
            ]
            point = np.array([
                row["median_difference_des_minus_sdss"] for row in rows
            ])
            lower = np.array([
                row["median_difference_bootstrap"]["lower"] for row in rows
            ])
            upper = np.array([
                row["median_difference_bootstrap"]["upper"] for row in rows
            ])
            nodes = [
                feature_bands[(comparison, feature, sbmag, redshift)]
                for redshift in redshifts
            ]
            simultaneous_lower = np.array([node["lower"] for node in nodes])
            simultaneous_upper = np.array([node["upper"] for node in nodes])
            shifted_z = np.asarray(redshifts) + offset
            axis.fill_between(
                redshifts,
                simultaneous_lower,
                simultaneous_upper,
                color=color,
                alpha=0.055,
                linewidth=0,
            )
            axis.plot(
                redshifts, simultaneous_lower,
                color=color, alpha=0.35, linewidth=0.65, linestyle=":",
            )
            axis.plot(
                redshifts, simultaneous_upper,
                color=color, alpha=0.35, linewidth=0.65, linestyle=":",
            )
            axis.errorbar(
                shifted_z,
                point,
                yerr=np.vstack((point - lower, upper - point)),
                color=color,
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                elinewidth=0.65,
                capsize=1.2,
                label=label,
            )

        if prediction is not None:
            axis.plot(
                redshifts,
                [prediction[z] for z in redshifts],
                color="#111111",
                linewidth=1.4,
                linestyle="--",
                label="Registered passband-only prediction",
            )
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set_title(f"SBMAG = {sbmag:.1f} mag arcsec$^{{-2}}$")
        axis.grid(color="#d9d9d9", linewidth=0.45, alpha=0.65)
        axis.set_xlim(0.043, 0.307)

    for axis in axes[-1, :]:
        axis.set_xlabel("Redshift")
    for axis in axes[:, 0]:
        axis.set_ylabel("Median difference: DES - SDSS")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(), loc="upper center", ncol=len(unique),
        frameon=False, bbox_to_anchor=(0.5, 0.965),
    )
    title = FEATURE_LABELS[feature]
    subtitle = "Pointwise 95% intervals; dotted envelopes are simultaneous 11-redshift bands"
    if feature in ("i_peak_flux", "z_peak_flux"):
        subtitle += ". Synthetic magnitude terms are not overlaid because units are incompatible"
    fig.suptitle(f"{title}\n{subtitle}", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, f"feature_{feature}")


def plot_timing(result, timing_rows, timing_bands, metric, title):
    redshifts = result["configuration"]["redshifts"]
    sbmags = result["configuration"]["sbmags"]
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.2), sharex=True, sharey=True)
    offsets = np.linspace(-0.0017, 0.0017, len(COMPARISON_STYLES))

    for axis, sbmag in zip(axes.flat, sbmags):
        for offset, (comparison, (color, label)) in zip(
            offsets, COMPARISON_STYLES.items()
        ):
            rows = [
                timing_rows[(comparison, metric, sbmag, redshift)]
                for redshift in redshifts
            ]
            point = np.array([row["difference_des_minus_sdss"] for row in rows])
            lower = np.array([row["bootstrap"]["lower"] for row in rows])
            upper = np.array([row["bootstrap"]["upper"] for row in rows])
            nodes = [
                timing_bands[(comparison, metric, sbmag, redshift)]
                for redshift in redshifts
            ]
            simultaneous_lower = np.array([node["lower"] for node in nodes])
            simultaneous_upper = np.array([node["upper"] for node in nodes])
            shifted_z = np.asarray(redshifts) + offset
            axis.fill_between(
                redshifts,
                simultaneous_lower,
                simultaneous_upper,
                color=color,
                alpha=0.055,
                linewidth=0,
            )
            axis.plot(
                redshifts, simultaneous_lower,
                color=color, alpha=0.35, linewidth=0.65, linestyle=":",
            )
            axis.plot(
                redshifts, simultaneous_upper,
                color=color, alpha=0.35, linewidth=0.65, linestyle=":",
            )
            axis.errorbar(
                shifted_z,
                point,
                yerr=np.vstack((point - lower, upper - point)),
                color=color,
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                elinewidth=0.65,
                capsize=1.2,
                label=label,
            )
        axis.axhline(0, color="#555555", linewidth=0.7)
        axis.set_title(f"SBMAG = {sbmag:.1f} mag arcsec$^{{-2}}$")
        axis.grid(color="#d9d9d9", linewidth=0.45, alpha=0.65)
        axis.set_xlim(0.043, 0.307)

    for axis in axes[-1, :]:
        axis.set_xlabel("Redshift")
    for axis in axes[:, 0]:
        axis.set_ylabel("DES - SDSS (days)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle(
        f"{title}\nPointwise 95% intervals; dotted envelopes: simultaneous 11-redshift bands",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return save_figure(fig, f"timing_{metric}")


def plot_data_quality(result):
    redshifts = result["configuration"]["redshifts"]
    sbmags = result["configuration"]["sbmags"]
    acceptance = {
        (row["survey"], row["stratum"], row["redshift"],
         row["sbmag_mag_per_arcsec2"]): row["acceptance_fraction"]
        for row in result["acceptance_statistics"]
    }
    fig = plt.figure(figsize=(14.5, 15.0))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.82))
    heat_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    rejection_axis = fig.add_subplot(grid[2, 0])
    availability_axis = fig.add_subplot(grid[2, 1])

    image = None
    for axis, branch in zip(heat_axes, BRANCH_STYLES):
        matrix = np.array([
            [acceptance[(branch[0], branch[1], z, sb)] for sb in sbmags]
            for z in redshifts
        ])
        image = axis.imshow(
            matrix, origin="lower", aspect="auto", vmin=0.95, vmax=1.0,
            cmap="viridis",
        )
        for row_i in range(matrix.shape[0]):
            for col_i in range(matrix.shape[1]):
                value = matrix[row_i, col_i]
                text_color = "white" if value < 0.976 else "black"
                axis.text(
                    col_i, row_i, f"{100*value:.1f}",
                    ha="center", va="center", fontsize=6.2,
                    color=text_color,
                )
        axis.set_title(f"{branch[0]} / {branch[1]}")
        axis.set_xticks(range(len(sbmags)), [f"{value:.1f}" for value in sbmags], rotation=45)
        axis.set_yticks(range(len(redshifts)), [f"{value:.3f}" for value in redshifts])
        axis.set_xlabel("SBMAG (mag arcsec$^{-2}$)")
        axis.set_ylabel("Redshift")
    colorbar_axis = fig.add_axes((0.925, 0.39, 0.015, 0.37))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Acceptance fraction")

    reasons = sorted(
        result["rejection_reason_counts"].items(),
        key=lambda item: item[1],
    )
    rejection_axis.barh(
        [name.replace("missing usable ", "missing ") for name, _ in reasons],
        [count for _, count in reasons],
        color="#b2182b",
    )
    rejection_axis.set_title("Rejection mechanisms (all rejected records)")
    rejection_axis.set_xlabel("Records")
    rejection_axis.grid(axis="x", color="#d9d9d9", linewidth=0.5)

    availability = defaultdict(lambda: [0, 0])
    for row in result["feature_availability_statistics"]:
        key = (row["survey"], row["stratum"], row["feature"])
        availability[key][0] += row["feature_available_realizations"]
        availability[key][1] += row["accepted_realizations"]
    x = np.arange(len(FEATURES))
    width = 0.18
    for index, (branch, color) in enumerate(BRANCH_STYLES.items()):
        values = [
            100 * availability[(branch[0], branch[1], feature)][0]
            / availability[(branch[0], branch[1], feature)][1]
            for feature in FEATURES
        ]
        availability_axis.bar(
            x + (index - 1.5) * width,
            values,
            width,
            color=color,
            label=f"{branch[0]} / {branch[1]}",
        )
    availability_axis.set_xticks(
        x,
        [
            "r-i", "i-z", "i peak", "z peak", "r peak time", "time span"
        ],
        rotation=35,
        ha="right",
    )
    availability_axis.set_ylim(99.5, 100.05)
    availability_axis.set_ylabel("Available among accepted records (%)")
    availability_axis.set_title("Feature availability")
    availability_axis.legend(frameon=False, fontsize=7, loc="lower left")
    availability_axis.grid(axis="y", color="#d9d9d9", linewidth=0.5)

    fig.suptitle(
        "Stage 3C data quality: acceptance, rejection, and feature availability",
        y=0.995,
    )
    fig.subplots_adjust(
        top=0.95, bottom=0.06, left=0.09, right=0.89,
        hspace=0.42, wspace=0.28,
    )
    return save_figure(fig, "data_quality")


def sign_counts(rows, band_index, kind):
    point_negative = point_positive = point_zero = 0
    sim_negative = sim_positive = sim_zero = 0
    for row in rows:
        bootstrap = (
            row["median_difference_bootstrap"]
            if kind == "feature" else row["bootstrap"]
        )
        if bootstrap["upper"] < 0:
            point_negative += 1
        elif bootstrap["lower"] > 0:
            point_positive += 1
        else:
            point_zero += 1
        key = (
            row["comparison"],
            row["feature"] if kind == "feature" else row["metric"],
            row["sbmag_mag_per_arcsec2"],
            row["redshift"],
        )
        node = band_index[key]
        if node["upper"] < 0:
            sim_negative += 1
        elif node["lower"] > 0:
            sim_positive += 1
        else:
            sim_zero += 1
    return (
        f"{point_negative}/{point_zero}/{point_positive}",
        f"{sim_negative}/{sim_zero}/{sim_positive}",
    )


def fmt_range(values, digits=4):
    return f"{min(values):.{digits}f} to {max(values):.{digits}f}"


def write_summary_tables(result, feature_bands, timing_bands):
    lines = [
        "# Stage 3C Summary Tables",
        "",
        "All ranges and counts summarize the retained strata without pooling their values.",
        "Sign-count columns are `negative / includes zero / positive` for the 88",
        "redshift-by-SBMAG cells within one survey comparison.",
        "",
        "## Data Integrity",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Analysis SHA-256 | `{sha256(ANALYSIS_PATH)}` |",
        f"| Complete design records | {result['grid_validation']['total_records']:,} |",
        f"| Bootstrap replicates | {result['configuration']['n_bootstrap']:,} |",
        f"| Undefined-statistic diagnostics | {len(result['undefined_statistic_diagnostics'])} |",
        "",
        "## Acceptance by Branch",
        "",
        "| Branch | Accepted | Rejected | Cell-fraction range |",
        "|---|---:|---:|---:|",
    ]
    grouped_acceptance = defaultdict(list)
    for row in result["acceptance_statistics"]:
        grouped_acceptance[(row["survey"], row["stratum"])].append(row)
    for branch, rows in grouped_acceptance.items():
        accepted = sum(row["accepted_realizations"] for row in rows)
        rejected = sum(row["rejected_realizations"] for row in rows)
        fractions = [row["acceptance_fraction"] for row in rows]
        lines.append(
            f"| {branch[0]} / {branch[1]} | {accepted:,} | {rejected:,} | "
            f"{100*min(fractions):.1f}% to {100*max(fractions):.1f}% |"
        )

    lines.extend([
        "",
        "## Rejection Mechanisms",
        "",
        "| Mechanism | Records | Fraction of rejections |",
        "|---|---:|---:|",
    ])
    total_rejections = sum(result["rejection_reason_counts"].values())
    for reason, count in sorted(
        result["rejection_reason_counts"].items(),
        key=lambda item: -item[1],
    ):
        lines.append(
            f"| {reason} | {count:,} | {100*count/total_rejections:.2f}% |"
        )

    lines.extend([
        "",
        "## Feature Results by Registered Comparison",
        "",
        "| Feature | Comparison | Median range | q10 range | q90 range | W1 range | Pointwise sign counts | Simultaneous sign counts |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    grouped_features = defaultdict(list)
    for row in result["primary_and_supportive_feature_statistics"]:
        grouped_features[(row["feature"], row["comparison"])].append(row)
    for feature in FEATURES:
        for comparison in COMPARISON_STYLES:
            rows = grouped_features[(feature, comparison)]
            point_counts, simultaneous_counts = sign_counts(
                rows, feature_bands, "feature"
            )
            lines.append(
                f"| `{feature}` | {COMPARISON_STYLES[comparison][1]} | "
                f"{fmt_range([row['median_difference_des_minus_sdss'] for row in rows])} | "
                f"{fmt_range([row['q10_difference_des_minus_sdss'] for row in rows])} | "
                f"{fmt_range([row['q90_difference_des_minus_sdss'] for row in rows])} | "
                f"{fmt_range([row['wasserstein_1'] for row in rows])} | "
                f"{point_counts} | {simultaneous_counts} |"
            )

    lines.extend([
        "",
        "## Key Redshift Envelopes",
        "",
        "Each envelope is the minimum and maximum among the 32 unpooled",
        "comparison-by-SBMAG coordinates at that redshift.",
        "",
        "| Feature | z | Median envelope | Pointwise negative / zero / positive |",
        "|---|---:|---:|---:|",
    ])
    rows_by_feature_redshift = defaultdict(list)
    for row in result["primary_and_supportive_feature_statistics"]:
        rows_by_feature_redshift[(row["feature"], row["redshift"])].append(row)
    for feature in FEATURES:
        for redshift in result["configuration"]["redshifts"]:
            rows = rows_by_feature_redshift[(feature, redshift)]
            negative = sum(
                row["median_difference_bootstrap"]["upper"] < 0
                for row in rows
            )
            positive = sum(
                row["median_difference_bootstrap"]["lower"] > 0
                for row in rows
            )
            zero = len(rows) - negative - positive
            lines.append(
                f"| `{feature}` | {redshift:.3f} | "
                f"{fmt_range([row['median_difference_des_minus_sdss'] for row in rows])} | "
                f"{negative} / {zero} / {positive} |"
            )

    lines.extend([
        "",
        "## Physical Timing Comparisons",
        "",
        "| Metric | Comparison | Difference range (days) | Pointwise sign counts | Simultaneous sign counts |",
        "|---|---|---:|---:|---:|",
    ])
    grouped_timing = defaultdict(list)
    for row in result["timing_comparisons"]:
        grouped_timing[(row["metric"], row["comparison"])].append(row)
    for metric in (
        "signed_median_days", "median_absolute_error_days", "mad_days",
        "iqr_days", "q10_days", "q90_days",
    ):
        for comparison in COMPARISON_STYLES:
            rows = grouped_timing[(metric, comparison)]
            point_counts, simultaneous_counts = sign_counts(
                rows, timing_bands, "timing"
            )
            lines.append(
                f"| `{metric}` | {COMPARISON_STYLES[comparison][1]} | "
                f"{fmt_range([row['difference_des_minus_sdss'] for row in rows])} | "
                f"{point_counts} | {simultaneous_counts} |"
            )

    lines.extend([
        "",
        "## Availability",
        "",
        "Every registered feature is available for every accepted record in",
        "all 352 branch/redshift/SBMAG cells. Missingness occurs at whole-record",
        "rejection, not as feature-specific missing values among accepted records.",
        "",
    ])
    path = OUTPUT_DIR / "stage3c_summary_tables.md"
    path.write_text("\n".join(lines))
    return path


def main():
    result, manifest, analysis_sha = load_verified_results()
    if result["undefined_statistic_diagnostics"]:
        raise RuntimeError("Frozen result contains undefined-statistic diagnostics.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    feature_rows, feature_bands, timing_rows, timing_bands = index_results(result)

    created = []
    created.extend(plot_data_quality(result))
    for feature in FEATURES:
        created.extend(plot_feature(result, feature_rows, feature_bands, feature))
    created.extend(plot_timing(
        result, timing_rows, timing_bands, "mad_days",
        "Physical r-band timing-error MAD difference",
    ))
    created.extend(plot_timing(
        result, timing_rows, timing_bands, "iqr_days",
        "Physical r-band timing-error IQR difference",
    ))
    created.append(write_summary_tables(result, feature_bands, timing_bands))

    print(f"Verified analysis SHA-256: {analysis_sha}")
    print(f"Manifest schema: {manifest['schema']}")
    for path in created:
        print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
