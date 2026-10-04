#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from phase2_tier4_windowed_plasticc import (
    build_windowed_plasticc_tables,
)


WINDOWS = [60, 75, 90, 105, 120, 180]

PLASTICC_FEATURE_FILE = Path(
    "data/PLAsTiCC/features/train_compact_features.csv"
)

WINDOW_FEATURE_DIR = Path(
    "data/PLAsTiCC/features/windowed"
)

RESULT_ROOT = Path(
    "results/phase2_tier4_window_sweep"
)

RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str]) -> None:
    print()
    print("=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)

    subprocess.run(
        cmd,
        check=True,
    )


def extract_domain_swap_metrics(
    metrics_csv: Path,
) -> dict:

    df = pd.read_csv(metrics_csv)

    mask = (
        (df["train_domain"] == "spcc")
        & (df["test_domain"] == "plasticc")
    )

    row = df.loc[mask].iloc[0]

    return {
        "f1": float(row["f1"]),
        "pr_auc": float(row["pr_auc"]),
        "roc_auc": float(row["roc_auc"]),
    }


def main():

    print()
    print("Building windowed feature tables...")
    print()

    build_windowed_plasticc_tables()

    summary_rows = []

    for window in WINDOWS:

        print()
        print("#" * 80)
        print(f"WINDOW = +/- {window} days")
        print("#" * 80)

        source_file = (
            WINDOW_FEATURE_DIR
            / f"compact_features_window_detected_max_flux_pm{window}.csv"
        )

        if not source_file.exists():
            raise FileNotFoundError(source_file)

        shutil.copy2(
            source_file,
            PLASTICC_FEATURE_FILE,
        )

        run_command(
            ["python", "phase2_tier4_plasticc_audit.py"]
        )

        run_command(
            ["python", "phase2_tier4_domain_swap.py"]
        )

        window_dir = (
            RESULT_ROOT
            / f"pm{window}"
        )

        window_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for file in Path("results/phase2_tier4").glob(
            "plasticc_audit*"
        ):
            shutil.copy2(
                file,
                window_dir / file.name,
            )

        for file in Path("results/phase2_tier4").glob(
            "domain_swap*"
        ):
            shutil.copy2(
                file,
                window_dir / file.name,
            )

        metrics = extract_domain_swap_metrics(
            Path(
                "results/phase2_tier4/domain_swap_metrics.csv"
            )
        )

        metrics["window_days"] = window

        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        "window_days"
    )

    summary_csv = (
        RESULT_ROOT
        / "window_sweep_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    best_row = summary_df.loc[
        summary_df["f1"].idxmax()
    ]

    report_path = (
        RESULT_ROOT
        / "window_sweep_report.md"
    )

    lines = [
        "# Phase 2 Tier 4 Window Sweep",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Best Window",
        "",
        f"Window: ±{int(best_row['window_days'])} days",
        "",
        f"F1: {best_row['f1']:.4f}",
        f"PR-AUC: {best_row['pr_auc']:.4f}",
        f"ROC-AUC: {best_row['roc_auc']:.4f}",
        "",
    ]

    report_path.write_text(
        "\n".join(lines)
    )

    print()
    print("Results written:")
    print(summary_csv)
    print(report_path)


if __name__ == "__main__":
    main()
