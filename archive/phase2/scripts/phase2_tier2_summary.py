"""Aggregate Phase 2 Tier 2 experiment outputs into one report."""

from __future__ import annotations

import os
from typing import Any

from phase2_tier2_common import (
    PLOTS_DIR,
    RESULTS_DIR,
    baseline_reference_payload,
    create_context,
    maybe_import_matplotlib,
    normalize_path,
    parse_numeric_rows,
    read_csv_rows,
    safe_label_from_loss,
    write_json,
)


FEATURE_ABLATION_CSV_PATH = f"{RESULTS_DIR}/feature_ablation_metrics.csv"
BLOCK_ABLATION_CSV_PATH = f"{RESULTS_DIR}/block_ablation_metrics.csv"
SUBSET_GROWTH_CSV_PATH = f"{RESULTS_DIR}/subset_growth_metrics.csv"
MINIMAL_CORE_CSV_PATH = f"{RESULTS_DIR}/minimal_core_metrics.csv"

MASTER_SUMMARY_PATH = f"{RESULTS_DIR}/phase2_tier2_master_summary.json"
REPORT_PATH = f"{RESULTS_DIR}/phase2_tier2_report.md"

FEATURE_PLOT_PATH = f"{PLOTS_DIR}/feature_ablation_delta_f1.png"
BLOCK_PLOT_PATH = f"{PLOTS_DIR}/block_ablation_delta_f1.png"
SUBSET_PLOT_PATH = f"{PLOTS_DIR}/subset_growth_f1.png"
SUBSET_PAPER_PLOT_PATH = f"{PLOTS_DIR}/subset_growth_paper_ready.png"
CORE_PLOT_PATH = f"{PLOTS_DIR}/minimal_core_tradeoff.png"


def load_required_rows(path: str, label: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label} output: {path}")
    return parse_numeric_rows(read_csv_rows(path))


def assign_labels(rows: list[dict[str, Any]], delta_f1_key: str = "delta_f1", delta_pr_auc_key: str = "delta_pr_auc") -> None:
    losses_f1 = [max(-float(row[delta_f1_key]), 0.0) for row in rows]
    losses_pr_auc = [max(-float(row[delta_pr_auc_key]), 0.0) for row in rows]
    max_loss_f1 = max(losses_f1, default=0.0)
    max_loss_pr_auc = max(losses_pr_auc, default=0.0)
    for row in rows:
        loss_f1 = max(-float(row[delta_f1_key]), 0.0)
        loss_pr_auc = max(-float(row[delta_pr_auc_key]), 0.0)
        row["label"] = safe_label_from_loss(loss_f1, loss_pr_auc, max_loss_f1, max_loss_pr_auc)


def plot_feature_ablation(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    sorted_rows = sorted(rows, key=lambda row: row["delta_f1"])
    labels = [row["feature_removed"] for row in sorted_rows]
    deltas = [row["delta_f1"] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, deltas, color="#b33f62")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Phase 2 Tier 2 Feature Ablation (delta F1 vs compact baseline)")
    ax.set_ylabel("delta F1")
    ax.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    fig.savefig(FEATURE_PLOT_PATH, dpi=200)
    plt.close(fig)


def plot_block_ablation(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    sorted_rows = sorted(rows, key=lambda row: row["delta_f1"])
    labels = [row["block_removed"] for row in sorted_rows]
    deltas = [row["delta_f1"] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, deltas, color="#d95d39")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Phase 2 Tier 2 Block Ablation (delta F1 vs compact baseline)")
    ax.set_ylabel("delta F1")
    fig.tight_layout()
    fig.savefig(BLOCK_PLOT_PATH, dpi=200)
    plt.close(fig)


def plot_subset_growth(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    x_values = [row["feature_count"] for row in rows]
    y_values = [row["f1"] for row in rows]
    labels = [row["subset_name"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_values, y_values, marker="o", color="#2a6f97")
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title("Phase 2 Tier 2 Subset Growth")
    ax.set_xlabel("feature count")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(SUBSET_PLOT_PATH, dpi=200)
    plt.close(fig)


def plot_subset_growth_paper_ready(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    stage_order = [
        "brightness_only",
        "brightness_plus_color",
        "brightness_plus_color_plus_variability",
        "full_compact",
    ]
    stage_labels = [
        "Brightness only",
        "Brightness + color",
        "Brightness + color + variability",
        "Full compact model",
    ]
    row_by_name = {row["subset_name"]: row for row in rows}
    filtered_rows = [row_by_name[name] for name in stage_order if name in row_by_name]
    if len(filtered_rows) < 2:
        return

    x_values = list(range(1, len(filtered_rows) + 1))
    f1_values = [row["f1"] for row in filtered_rows]
    pr_auc_values = [row["pr_auc"] for row in filtered_rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x_values, f1_values, marker="o", linewidth=2.2, color="#1d4e89", label="F1")
    ax.plot(x_values, pr_auc_values, marker="s", linewidth=2.2, color="#c84c09", label="PR-AUC")
    ax.set_xticks(x_values)
    ax.set_xticklabels(stage_labels, rotation=15, ha="right")
    ax.set_ylim(
        min(min(f1_values), min(pr_auc_values)) - 0.02,
        max(max(f1_values), max(pr_auc_values)) + 0.02,
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Subset stage")
    ax.set_title("Classification performance as compact feature information is added")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SUBSET_PAPER_PLOT_PATH, dpi=250)
    plt.close(fig)


def plot_minimal_core(rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    x_values = [row["feature_count"] for row in rows]
    y_values = [row["f1"] for row in rows]
    labels = [row["subset_name"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, s=100, color="#3a7d44")
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title("Phase 2 Tier 2 Minimal Core Tradeoff")
    ax.set_xlabel("feature count")
    ax.set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(CORE_PLOT_PATH, dpi=200)
    plt.close(fig)


def best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(rows, key=lambda row: row[key])


def write_report(
    context_payload: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    subset_rows: list[dict[str, Any]],
    core_rows: list[dict[str, Any]],
) -> None:
    strongest_feature_losses = sorted(feature_rows, key=lambda row: row["delta_f1"])[:5]
    weakest_feature_losses = sorted(feature_rows, key=lambda row: row["delta_f1"], reverse=True)[:5]
    strongest_block_loss = min(block_rows, key=lambda row: row["delta_f1"])
    best_core = best_row(core_rows, "f1")
    best_growth = best_row(subset_rows, "f1")

    lines = [
        "# Phase 2 Tier 2 Report",
        "",
        "## Baseline reference",
        f"- Source CSV: `{context_payload['source_csv']}`",
        f"- Compact feature count: {context_payload['compact_feature_count']}",
        f"- Frozen baseline F1: {context_payload['baseline_metrics']['f1']:.6f}",
        f"- Frozen baseline ROC-AUC: {context_payload['baseline_metrics']['roc_auc']:.6f}",
        f"- Frozen baseline PR-AUC: {context_payload['baseline_metrics']['pr_auc']:.6f}",
        "",
        "## Feature ablation highlights",
    ]
    for row in strongest_feature_losses:
        lines.append(
            f"- {row['feature_removed']} ({row['feature_group']}): delta F1 {row['delta_f1']:+.6f}, "
            f"delta PR-AUC {row['delta_pr_auc']:+.6f}, label {row['label']}."
        )
    lines.extend(["", "## Near-redundant feature candidates"])
    for row in weakest_feature_losses:
        lines.append(
            f"- {row['feature_removed']} ({row['feature_group']}): delta F1 {row['delta_f1']:+.6f}, "
            f"delta PR-AUC {row['delta_pr_auc']:+.6f}, label {row['label']}."
        )
    lines.extend(
        [
            "",
            "## Block ablation highlights",
            f"- Largest loss: removing {strongest_block_loss['block_removed']} produced delta F1 {strongest_block_loss['delta_f1']:+.6f} "
            f"and delta PR-AUC {strongest_block_loss['delta_pr_auc']:+.6f}, labeled {strongest_block_loss['label']}.",
            "",
            "## Subset growth",
            f"- Best subset-growth result: {best_growth['subset_name']} at F1 {best_growth['f1']:.6f} "
            f"(delta F1 {best_growth['delta_f1']:+.6f}).",
            "",
            "## Minimal core",
            f"- Best reduced core: {best_core['subset_name']} with {best_core['feature_count']} features, "
            f"F1 {best_core['f1']:.6f}, PR-AUC {best_core['pr_auc']:.6f}, delta F1 {best_core['delta_f1']:+.6f}.",
            f"- Feature list: {best_core['feature_list']}.",
            "",
            "## Generated artifacts",
            f"- Feature ablation plot: `{normalize_path(FEATURE_PLOT_PATH)}`",
            f"- Block ablation plot: `{normalize_path(BLOCK_PLOT_PATH)}`",
            f"- Subset growth plot: `{normalize_path(SUBSET_PLOT_PATH)}`",
            f"- Paper-ready subset growth plot: `{normalize_path(SUBSET_PAPER_PLOT_PATH)}`",
            f"- Minimal core plot: `{normalize_path(CORE_PLOT_PATH)}`",
        ]
    )

    with open(REPORT_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    context = create_context()
    feature_rows = load_required_rows(FEATURE_ABLATION_CSV_PATH, "feature ablation")
    block_rows = load_required_rows(BLOCK_ABLATION_CSV_PATH, "block ablation")
    subset_rows = load_required_rows(SUBSET_GROWTH_CSV_PATH, "subset growth")
    core_rows = load_required_rows(MINIMAL_CORE_CSV_PATH, "minimal core")

    assign_labels(feature_rows)
    assign_labels(block_rows)

    plot_feature_ablation(feature_rows)
    plot_block_ablation(block_rows)
    plot_subset_growth(subset_rows)
    plot_subset_growth_paper_ready(subset_rows)
    plot_minimal_core(core_rows)

    summary_payload = {
        "baseline_reference": baseline_reference_payload(context),
        "feature_ablation": feature_rows,
        "block_ablation": block_rows,
        "subset_growth": subset_rows,
        "minimal_core": core_rows,
        "plots": {
            "feature_ablation_delta_f1": FEATURE_PLOT_PATH,
            "block_ablation_delta_f1": BLOCK_PLOT_PATH,
            "subset_growth_f1": SUBSET_PLOT_PATH,
            "subset_growth_paper_ready": SUBSET_PAPER_PLOT_PATH,
            "minimal_core_tradeoff": CORE_PLOT_PATH,
        },
    }
    write_json(MASTER_SUMMARY_PATH, summary_payload)
    write_report(summary_payload["baseline_reference"], feature_rows, block_rows, subset_rows, core_rows)
    print(f"Wrote {MASTER_SUMMARY_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
