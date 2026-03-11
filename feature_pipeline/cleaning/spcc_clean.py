"""Cleaning and summary helpers for raw SPCC events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import isfinite

from feature_pipeline.schemas import SPCCRawEvent


@dataclass(frozen=True)
class CleaningResult:
    event: SPCCRawEvent | None
    accepted: bool
    reasons: list[str]
    summary: dict


def _is_valid_hostz(hostz: tuple[float, float] | None) -> bool:
    if hostz is None:
        return True
    return all(isfinite(value) for value in hostz)


def summarize_event(event: SPCCRawEvent) -> dict:
    bands = [obs.band for obs in event.observations]
    mjds = [obs.mjd for obs in event.observations]
    return {
        "snid": event.snid,
        "observation_count": len(event.observations),
        "band_counts": dict(Counter(bands)),
        "time_span": 0.0 if not mjds else max(mjds) - min(mjds),
        "has_hostz": event.hostz is not None,
        "sim_type": event.sim_type,
    }


def clean_event(event: SPCCRawEvent, *, min_observations_per_event: int = 1) -> CleaningResult:
    reasons: list[str] = []

    cleaned_observations = sorted(event.observations, key=lambda obs: (obs.mjd, obs.band))
    cleaned_event = replace(event, observations=cleaned_observations)

    if cleaned_event.snid is None:
        reasons.append("missing_snid")
    if cleaned_event.sim_type is None:
        reasons.append("missing_sim_type")
    if cleaned_event.sim_z is None or not isfinite(cleaned_event.sim_z):
        reasons.append("invalid_sim_z")
    if cleaned_event.ra is None or not isfinite(cleaned_event.ra):
        reasons.append("invalid_ra")
    if cleaned_event.decl is None or not isfinite(cleaned_event.decl):
        reasons.append("invalid_decl")
    if cleaned_event.mwebv is None or not isfinite(cleaned_event.mwebv):
        reasons.append("invalid_mwebv")
    if not _is_valid_hostz(cleaned_event.hostz):
        reasons.append("invalid_hostz")
    if len(cleaned_event.observations) < min_observations_per_event:
        reasons.append("insufficient_observations")

    valid_observations = []
    dropped_observations = 0
    for obs in cleaned_event.observations:
        if not (isfinite(obs.mjd) and isfinite(obs.flux) and isfinite(obs.flux_err)):
            dropped_observations += 1
            continue
        valid_observations.append(obs)

    if dropped_observations:
        cleaned_event = replace(cleaned_event, observations=valid_observations)
    if len(cleaned_event.observations) < min_observations_per_event:
        reasons.append("insufficient_observations_after_numeric_filter")

    summary = summarize_event(cleaned_event)
    summary["dropped_observations"] = dropped_observations
    summary["accepted"] = not reasons
    summary["rejection_reasons"] = reasons

    return CleaningResult(
        event=cleaned_event if not reasons else None,
        accepted=not reasons,
        reasons=reasons,
        summary=summary,
    )


def summarize_cleaning_results(results: list[CleaningResult]) -> dict:
    accepted = [result for result in results if result.accepted]
    rejected = [result for result in results if not result.accepted]
    rejection_counts = Counter(reason for result in rejected for reason in result.reasons)
    class_counts = Counter(result.summary["sim_type"] for result in accepted)

    return {
        "total_events": len(results),
        "accepted_events": len(accepted),
        "rejected_events": len(rejected),
        "accepted_class_counts": dict(class_counts),
        "rejection_reason_counts": dict(rejection_counts),
        "accepted_summaries": [result.summary for result in accepted],
        "rejected_summaries": [result.summary for result in rejected],
    }
