"""Audit event-level rejection stages for the Tier-4 compact feature builder.

The audit intentionally replays the selection logic used by
``phase2_tier4_make_variants.py`` while recording the first rejection stage for
each event.  It does not infer rejection rates from final feature tables.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from feature_pipeline.cleaning.spcc_clean import clean_event
from feature_pipeline.loaders.spcc_raw import iter_spcc_files, load_spcc_raw_event
from feature_pipeline.policies import KEY_TYPES


RESULTS_DIR = "results/phase2_tier4"
CSV_PATH = os.path.join(RESULTS_DIR, "event_rejection_audit.csv")
JSON_PATH = os.path.join(RESULTS_DIR, "event_rejection_audit.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "event_rejection_audit_report.md")

SPCC_RAW_GLOB = "data/spcc/raw/DES_*.DAT"
PLASTICC_LIGHTCURVE_PATH = "data/PLAsTiCC/training_set.csv"
PLASTICC_METADATA_PATH = "data/PLAsTiCC/training_set_metadata.csv"
PLASTICC_PASSBANDS = {"g": 1, "r": 2, "i": 3, "z": 4}
SPCC_BANDS = ("g", "r", "i", "z")
SNR_ACTIVE_THRESHOLD = 3.0
MIN_COLOR_FLUX_THRESHOLD = 0.0


ACCEPTED_STAGE = "accepted"
STAGE_ORDER = [
    "cleaning_rejection",
    "class_exclusion",
    "compact_feature_rejection_no_griz_observations",
    "missing_band_rejection",
    "positive_flux_support_rejection",
    ACCEPTED_STAGE,
]


@dataclass(frozen=True)
class AuditRecord:
    dataset: str
    object_id: int | str
    label_name: str
    class_group: str
    stage: str
    accepted: bool
    cleaning_reasons: tuple[str, ...]
    missing_bands: tuple[str, ...]
    nonpositive_support_bands: tuple[str, ...]
    raw_observation_count: int
    selected_observation_count: int
    active_observation_count: int
    event_span: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "object_id": self.object_id,
            "label_name": self.label_name,
            "class_group": self.class_group,
            "stage": self.stage,
            "accepted": self.accepted,
            "cleaning_reasons": list(self.cleaning_reasons),
            "missing_bands": list(self.missing_bands),
            "nonpositive_support_bands": list(self.nonpositive_support_bands),
            "raw_observation_count": self.raw_observation_count,
            "selected_observation_count": self.selected_observation_count,
            "active_observation_count": self.active_observation_count,
            "event_span": self.event_span,
        }


def _class_group(label_name: str | None) -> str:
    if label_name == "Ia":
        return "Ia"
    if label_name and label_name not in {"missing", "unknown"}:
        return "non-Ia"
    return "unknown"


def _normalize_observations(
    observations: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    if mode != "none":
        raise ValueError("This audit replays the retained Tier-4 pipeline with normalization_mode='none'.")
    return [dict(obs) for obs in observations]


def _representative_flux_for_color(fluxes: list[float]) -> float:
    positive_fluxes = sorted(
        [float(value) for value in fluxes if float(value) > MIN_COLOR_FLUX_THRESHOLD],
        reverse=True,
    )
    if not positive_fluxes:
        return 0.0
    top_k = positive_fluxes[: min(3, len(positive_fluxes))]
    return float(sum(top_k) / len(top_k))


def _stage_for_observations(observations: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    filtered = [obs for obs in observations if obs["band"] in SPCC_BANDS]
    if not filtered:
        return (
            "compact_feature_rejection_no_griz_observations",
            {
                "selected_observation_count": 0,
                "active_observation_count": 0,
                "event_span": None,
                "missing_bands": list(SPCC_BANDS),
                "nonpositive_support_bands": [],
            },
        )

    filtered.sort(key=lambda obs: float(obs["time"]))
    active = [
        obs
        for obs in filtered
        if float(obs["flux"]) > 0.0
        and float(obs["flux_err"]) > 0.0
        and float(obs["flux"]) / float(obs["flux_err"]) >= SNR_ACTIVE_THRESHOLD
    ]
    active_group = active if active else filtered
    event_start_time = float(active_group[0]["time"])
    event_end_time = float(active_group[-1]["time"])
    event_span = max(event_end_time - event_start_time, 0.0)

    missing_bands: list[str] = []
    nonpositive_support_bands: list[str] = []
    representative_fluxes: dict[str, float] = {}

    for band_name in SPCC_BANDS:
        band_group = [obs for obs in active_group if obs["band"] == band_name]
        if not band_group:
            band_group = [
                obs
                for obs in filtered
                if obs["band"] == band_name
                and event_start_time <= float(obs["time"]) <= event_end_time
            ]
        if not band_group:
            missing_bands.append(band_name)
            representative_fluxes[band_name] = 0.0
            continue

        fluxes = [float(obs["flux"]) for obs in band_group]
        representative_fluxes[band_name] = _representative_flux_for_color(fluxes)
        if representative_fluxes[band_name] <= MIN_COLOR_FLUX_THRESHOLD:
            nonpositive_support_bands.append(band_name)

    diagnostics = {
        "selected_observation_count": len(filtered),
        "active_observation_count": len(active),
        "event_span": event_span,
        "missing_bands": missing_bands,
        "nonpositive_support_bands": nonpositive_support_bands,
        "representative_fluxes": representative_fluxes,
    }
    if missing_bands:
        return "missing_band_rejection", diagnostics
    if nonpositive_support_bands:
        return "positive_flux_support_rejection", diagnostics
    return ACCEPTED_STAGE, diagnostics


def audit_spcc(normalization_mode: str = "none") -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for path in iter_spcc_files(SPCC_RAW_GLOB):
        raw_event = load_spcc_raw_event(path)
        label_name = raw_event.sim_type or "missing"
        object_id: int | str = raw_event.snid if raw_event.snid is not None else os.path.basename(path)
        raw_count = len(raw_event.observations)
        cleaning_result = clean_event(raw_event, min_observations_per_event=1)

        if not cleaning_result.accepted or cleaning_result.event is None:
            records.append(
                AuditRecord(
                    dataset="SPCC",
                    object_id=object_id,
                    label_name=label_name,
                    class_group=_class_group(label_name),
                    stage="cleaning_rejection",
                    accepted=False,
                    cleaning_reasons=tuple(cleaning_result.reasons),
                    missing_bands=(),
                    nonpositive_support_bands=(),
                    raw_observation_count=raw_count,
                    selected_observation_count=0,
                    active_observation_count=0,
                    event_span=None,
                )
            )
            continue

        event = cleaning_result.event
        label_name = str(event.sim_type)
        if label_name not in KEY_TYPES:
            records.append(
                AuditRecord(
                    dataset="SPCC",
                    object_id=int(event.snid),
                    label_name=label_name,
                    class_group=_class_group(label_name),
                    stage="class_exclusion",
                    accepted=False,
                    cleaning_reasons=(),
                    missing_bands=(),
                    nonpositive_support_bands=(),
                    raw_observation_count=raw_count,
                    selected_observation_count=len(event.observations),
                    active_observation_count=0,
                    event_span=None,
                )
            )
            continue

        observations = [
            {
                "time": obs.mjd,
                "band": obs.band,
                "flux": obs.flux,
                "flux_err": obs.flux_err,
            }
            for obs in event.observations
        ]
        observations = _normalize_observations(observations, normalization_mode)
        stage, diagnostics = _stage_for_observations(observations)
        records.append(
            AuditRecord(
                dataset="SPCC",
                object_id=int(event.snid),
                label_name=label_name,
                class_group=_class_group(label_name),
                stage=stage,
                accepted=stage == ACCEPTED_STAGE,
                cleaning_reasons=(),
                missing_bands=tuple(diagnostics["missing_bands"]),
                nonpositive_support_bands=tuple(diagnostics["nonpositive_support_bands"]),
                raw_observation_count=raw_count,
                selected_observation_count=int(diagnostics["selected_observation_count"]),
                active_observation_count=int(diagnostics["active_observation_count"]),
                event_span=diagnostics["event_span"],
            )
        )
    return records


def _plasticc_label(target: int) -> str:
    return "Ia" if int(target) == 90 else "non-Ia"


def _plasticc_redshift(meta: dict[str, Any]) -> float:
    specz = meta.get("hostgal_specz")
    photoz = meta.get("hostgal_photoz")
    if specz is not None and not (isinstance(specz, float) and math.isnan(specz)) and float(specz) > 0:
        return float(specz)
    return float(photoz)


def audit_plasticc(normalization_mode: str = "none") -> list[AuditRecord]:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return []

    if not os.path.exists(PLASTICC_LIGHTCURVE_PATH) or not os.path.exists(PLASTICC_METADATA_PATH):
        return []

    lightcurves = pd.read_csv(PLASTICC_LIGHTCURVE_PATH)
    metadata_frame = pd.read_csv(PLASTICC_METADATA_PATH)
    metadata = metadata_frame[["object_id", "hostgal_photoz", "hostgal_specz", "target"]]
    meta_lookup = metadata.set_index("object_id").to_dict("index")
    passband_lookup = {band_id: name for name, band_id in PLASTICC_PASSBANDS.items()}

    records: list[AuditRecord] = []
    for object_id, group in lightcurves.groupby("object_id", sort=True):
        if object_id not in meta_lookup:
            continue
        meta = meta_lookup[object_id]
        label_name = _plasticc_label(int(meta["target"]))
        observations = [
            {
                "time": float(row["mjd"]),
                "band": passband_lookup[int(row["passband"])],
                "flux": float(row["flux"]),
                "flux_err": float(row["flux_err"]),
            }
            for _, row in group.iterrows()
            if int(row["passband"]) in passband_lookup
        ]
        observations = _normalize_observations(observations, normalization_mode)
        stage, diagnostics = _stage_for_observations(observations)
        records.append(
            AuditRecord(
                dataset="PLAsTiCC training",
                object_id=int(object_id),
                label_name=label_name,
                class_group=_class_group(label_name),
                stage=stage,
                accepted=stage == ACCEPTED_STAGE,
                cleaning_reasons=(),
                missing_bands=tuple(diagnostics["missing_bands"]),
                nonpositive_support_bands=tuple(diagnostics["nonpositive_support_bands"]),
                raw_observation_count=len(group),
                selected_observation_count=int(diagnostics["selected_observation_count"]),
                active_observation_count=int(diagnostics["active_observation_count"]),
                event_span=diagnostics["event_span"],
            )
        )
        _ = _plasticc_redshift(meta)
    return records


def _pct(part: int, whole: int) -> float:
    return float(100.0 * part / whole) if whole else 0.0


def summarize_records(records: list[AuditRecord]) -> dict[str, Any]:
    by_dataset: dict[str, list[AuditRecord]] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset].append(record)

    datasets: dict[str, Any] = {}
    for dataset, dataset_records in by_dataset.items():
        total = len(dataset_records)
        accepted = sum(1 for record in dataset_records if record.accepted)
        stage_counts = Counter(record.stage for record in dataset_records)
        class_counts: dict[str, Any] = {}
        for class_group in ["Ia", "non-Ia", "unknown"]:
            class_records = [record for record in dataset_records if record.class_group == class_group]
            class_total = len(class_records)
            class_accepted = sum(1 for record in class_records if record.accepted)
            class_counts[class_group] = {
                "total": class_total,
                "accepted": class_accepted,
                "rejected": class_total - class_accepted,
                "accepted_percent": _pct(class_accepted, class_total),
                "rejected_percent": _pct(class_total - class_accepted, class_total),
                "stage_counts": dict(Counter(record.stage for record in class_records)),
            }

        subtype_counts: dict[str, Any] = {}
        for label_name in sorted({record.label_name for record in dataset_records}):
            label_records = [record for record in dataset_records if record.label_name == label_name]
            label_total = len(label_records)
            label_accepted = sum(1 for record in label_records if record.accepted)
            subtype_counts[label_name] = {
                "total": label_total,
                "accepted": label_accepted,
                "rejected": label_total - label_accepted,
                "rejected_percent": _pct(label_total - label_accepted, label_total),
                "stage_counts": dict(Counter(record.stage for record in label_records)),
            }

        datasets[dataset] = {
            "total_events": total,
            "accepted_events": accepted,
            "rejected_events": total - accepted,
            "rejection_percent": _pct(total - accepted, total),
            "stage_counts": {stage: int(stage_counts.get(stage, 0)) for stage in STAGE_ORDER},
            "class_counts": class_counts,
            "subtype_counts": subtype_counts,
        }
    return {
        "audit_name": "phase2_tier4_event_rejection_audit",
        "selection_logic": {
            "cleaning": "SPCC clean_event(..., min_observations_per_event=1) removes events with invalid required metadata or insufficient finite observations.",
            "class_exclusion": f"SPCC class labels are checked against KEY_TYPES={KEY_TYPES}; no exclusion is expected when all labels are known.",
            "active_window": f"Use observations with F > 0, sigma_F > 0, and F/sigma_F >= {SNR_ACTIVE_THRESHOLD}; if none exist, use all selected g/r/i/z observations.",
            "strongest_positive_observations": "For each band, sort positive fluxes in descending order and average the first min(3, n_positive) values.",
            "color_support": "Reject compact-feature rows when any of g/r/i/z lacks a strictly positive representative flux.",
        },
        "outputs": {
            "csv": CSV_PATH,
            "json": JSON_PATH,
            "report": REPORT_PATH,
        },
        "datasets": datasets,
    }


def write_csv_summary(summary: dict[str, Any]) -> None:
    fieldnames = [
        "dataset",
        "class_group",
        "stage",
        "events",
        "total_events",
        "percent_of_dataset",
        "accepted_events",
        "rejected_events",
        "rejection_percent",
    ]
    rows: list[dict[str, Any]] = []
    for dataset, dataset_summary in summary["datasets"].items():
        total = int(dataset_summary["total_events"])
        for stage in STAGE_ORDER:
            events = int(dataset_summary["stage_counts"].get(stage, 0))
            rows.append(
                {
                    "dataset": dataset,
                    "class_group": "all",
                    "stage": stage,
                    "events": events,
                    "total_events": total,
                    "percent_of_dataset": _pct(events, total),
                    "accepted_events": dataset_summary["accepted_events"],
                    "rejected_events": dataset_summary["rejected_events"],
                    "rejection_percent": dataset_summary["rejection_percent"],
                }
            )
        for class_group, class_summary in dataset_summary["class_counts"].items():
            class_total = int(class_summary["total"])
            if class_total == 0:
                continue
            for stage in STAGE_ORDER:
                events = int(class_summary["stage_counts"].get(stage, 0))
                rows.append(
                    {
                        "dataset": dataset,
                        "class_group": class_group,
                        "stage": stage,
                        "events": events,
                        "total_events": class_total,
                        "percent_of_dataset": _pct(events, class_total),
                        "accepted_events": class_summary["accepted"],
                        "rejected_events": class_summary["rejected"],
                        "rejection_percent": class_summary["rejected_percent"],
                    }
                )
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stage_table_lines(dataset_summary: dict[str, Any]) -> list[str]:
    total = int(dataset_summary["total_events"])
    lines = [
        "| Stage | Events | Percent |",
        "| --- | ---: | ---: |",
    ]
    for stage in STAGE_ORDER:
        count = int(dataset_summary["stage_counts"].get(stage, 0))
        lines.append(f"| {stage} | {count} | {_pct(count, total):.2f}% |")
    return lines


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 Tier 4 Event-Rejection Audit",
        "",
        "This audit reconstructs the event-selection pipeline stage by stage. It does not infer rejection percentages from final compact-feature row counts.",
        "",
        "## Strongest Positive Observations",
        "",
        "For band $b$, let $P_b = \\{F_{b,j}: F_{b,j} > 0\\}$ be the positive flux measurements available after the active-window and band-fallback rule. Sort them as $F_{b,(1)} \\ge F_{b,(2)} \\ge \\cdots \\ge F_{b,(n_b)}$ and set $k_b = \\min(3,n_b)$. The representative flux used for color construction is",
        "",
        "$$",
        "F^{(b)}_{\\rm rep} = \\frac{1}{k_b}\\sum_{j=1}^{k_b} F_{b,(j)}, \\quad n_b > 0.",
        "$$",
        "",
        "If $n_b=0$ for any of $g,r,i,z$, the event is rejected at the positive-flux support stage.",
        "",
    ]
    for dataset, dataset_summary in summary["datasets"].items():
        lines += [
            f"## {dataset}",
            "",
            f"- Total events entering audit: {dataset_summary['total_events']}",
            f"- Accepted events: {dataset_summary['accepted_events']}",
            f"- Rejected events: {dataset_summary['rejected_events']} ({dataset_summary['rejection_percent']:.2f}%)",
            "",
            "### Rejection Stage Counts",
            "",
            *_stage_table_lines(dataset_summary),
            "",
            "### Ia and non-Ia Counts",
            "",
            "| Class | Total | Accepted | Rejected | Rejected percent |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for class_group in ["Ia", "non-Ia", "unknown"]:
            class_summary = dataset_summary["class_counts"][class_group]
            if class_summary["total"] == 0:
                continue
            lines.append(
                f"| {class_group} | {class_summary['total']} | {class_summary['accepted']} | "
                f"{class_summary['rejected']} | {class_summary['rejected_percent']:.2f}% |"
            )
        lines += [
            "",
            "### Counts by Label",
            "",
            "| Label | Total | Accepted | Rejected | Rejected percent | Dominant rejection stage |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for label_name, subtype_summary in dataset_summary["subtype_counts"].items():
            rejected_stage_counts = {
                stage: count
                for stage, count in subtype_summary["stage_counts"].items()
                if stage != ACCEPTED_STAGE and count
            }
            dominant = max(rejected_stage_counts, key=rejected_stage_counts.get) if rejected_stage_counts else "none"
            lines.append(
                f"| {label_name} | {subtype_summary['total']} | {subtype_summary['accepted']} | "
                f"{subtype_summary['rejected']} | {subtype_summary['rejected_percent']:.2f}% | {dominant} |"
            )
        lines.append("")
    lines += [
        "## Output Files",
        "",
        f"- CSV summary: `{CSV_PATH}`",
        f"- JSON summary: `{JSON_PATH}`",
        f"- Markdown report: `{REPORT_PATH}`",
        "",
    ]
    with open(REPORT_PATH, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = audit_spcc() + audit_plasticc()
    summary = summarize_records(records)
    write_csv_summary(summary)
    with open(JSON_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)
    write_report(summary)
    print(
        json.dumps(
            {
                "records": len(records),
                "csv": CSV_PATH,
                "json": JSON_PATH,
                "report": REPORT_PATH,
                "datasets": list(summary["datasets"].keys()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
