"""Audit SPCC versus PLAsTiCC compact-feature compatibility for Tier 4."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter

from phase2_tier2_common import COMPACT_FEATURES
from phase2_tier4_common import PLASTICC_TEST_COMPACT_CSV_PATH, PLASTICC_TRAIN_COMPACT_CSV_PATH, RESULTS_DIR, SPCC_COMPACT_CSV_PATH, write_markdown


JSON_PATH = f"{RESULTS_DIR}/plasticc_audit.json"
REPORT_PATH = f"{RESULTS_DIR}/plasticc_audit_report.md"


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    size = len(ordered)
    def pick(p: float) -> float:
        return ordered[int((size - 1) * p)]
    return {
        "min": ordered[0],
        "p50": pick(0.50),
        "p95": pick(0.95),
        "max": ordered[-1],
    }


def summarize_dataset(rows: list[dict[str, str]]) -> dict[str, object]:
    time_consistency_violations = sum(
        1
        for row in rows
        if any(float(row[name]) > float(row["time_span"]) + 1e-9 for name in ("r_time_of_peak", "i_time_of_peak", "z_time_of_peak"))
    )
    return {
        "row_count": len(rows),
        "label_counts": dict(Counter(row["label_name"] for row in rows)),
        "time_consistency_violations": time_consistency_violations,
        "feature_quantiles": {
            feature_name: quantiles([float(row[feature_name]) for row in rows])
            for feature_name in COMPACT_FEATURES
        },
    }


def mismatch_flags(spcc_summary: dict[str, object], plasticc_summary: dict[str, object]) -> list[dict[str, object]]:
    flags = []
    spcc_quantiles = spcc_summary["feature_quantiles"]
    plasticc_quantiles = plasticc_summary["feature_quantiles"]
    for feature_name in COMPACT_FEATURES:
        spcc_p50 = float(spcc_quantiles[feature_name]["p50"])
        plasticc_p50 = float(plasticc_quantiles[feature_name]["p50"])
        scale_ratio = (plasticc_p50 + 1e-9) / (spcc_p50 + 1e-9)
        if scale_ratio > 3.0 or scale_ratio < (1.0 / 3.0):
            flags.append(
                {
                    "feature": feature_name,
                    "spcc_p50": spcc_p50,
                    "plasticc_p50": plasticc_p50,
                    "median_ratio": scale_ratio,
                }
            )
    flags.sort(key=lambda row: abs(row["median_ratio"] - 1.0), reverse=True)
    return flags


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(SPCC_COMPACT_CSV_PATH):
        raise FileNotFoundError(f"Missing SPCC compact features: {SPCC_COMPACT_CSV_PATH}")
    plasticc_path = PLASTICC_TEST_COMPACT_CSV_PATH if os.path.exists(PLASTICC_TEST_COMPACT_CSV_PATH) else PLASTICC_TRAIN_COMPACT_CSV_PATH
    if not os.path.exists(plasticc_path):
        raise FileNotFoundError(
            f"Missing PLAsTiCC compact features: expected {PLASTICC_TRAIN_COMPACT_CSV_PATH} or {PLASTICC_TEST_COMPACT_CSV_PATH}"
        )

    spcc_rows = load_rows(SPCC_COMPACT_CSV_PATH)
    plasticc_rows = load_rows(plasticc_path)
    spcc_summary = summarize_dataset(spcc_rows)
    plasticc_summary = summarize_dataset(plasticc_rows)
    flags = mismatch_flags(spcc_summary, plasticc_summary)

    payload = {
        "spcc": spcc_summary,
        "plasticc": plasticc_summary,
        "mismatch_flags": flags,
    }
    with open(JSON_PATH, "w") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Phase 2 Tier 4 PLAsTiCC Audit",
        "",
        "This audit checks whether the current PLAsTiCC compact-feature export is on a comparable scale to SPCC before interpreting cross-survey results as scientific.",
        "",
        f"SPCC rows: {spcc_summary['row_count']}",
        f"PLAsTiCC rows: {plasticc_summary['row_count']}",
        f"SPCC labels: {spcc_summary['label_counts']}",
        f"PLAsTiCC labels: {plasticc_summary['label_counts']}",
        f"SPCC time consistency violations: {spcc_summary['time_consistency_violations']}",
        f"PLAsTiCC time consistency violations: {plasticc_summary['time_consistency_violations']}",
        "",
        "## Largest median-scale mismatches",
        "",
        "| feature | SPCC p50 | PLAsTiCC p50 | ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in flags[:12]:
        lines.append(
            f"| {row['feature']} | {row['spcc_p50']:.6f} | {row['plasticc_p50']:.6f} | {row['median_ratio']:.3f} |"
        )
    write_markdown(REPORT_PATH, lines)

    print(json.dumps({"json_path": JSON_PATH, "report_path": REPORT_PATH, "flag_count": len(flags)}, indent=2))


if __name__ == "__main__":
    main()
