"""Raw SPCC light-curve loading utilities."""

from __future__ import annotations

import glob
from dataclasses import dataclass

from feature_pipeline.config import DEFAULT_NORMALIZATION
from feature_pipeline.schemas import SPCCRawEvent, SPCCRawObservation

DEFAULT_SPCC_RAW_GLOB = "data/SIMGEN_PUBLIC_DES/DES_*.DAT"


@dataclass(frozen=True)
class NormalizedSPCCObservation:
    time: float
    band: str
    flux: float
    flux_err: float


def iter_spcc_files(input_glob: str = DEFAULT_SPCC_RAW_GLOB) -> list[str]:
    return sorted(glob.glob(input_glob))


def load_spcc_raw_event(filename: str) -> SPCCRawEvent:
    survey = snid = ra = decl = mwebv = hostid = hostz = spec = sim_type = sim_z = sn_type = None
    observations: list[SPCCRawObservation] = []

    with open(filename, "r") as handle:
        for line in handle:
            parts = line.split(":")
            if not parts:
                continue
            key = parts[0]
            if key == "SURVEY":
                survey = parts[1].strip()
            elif key == "SNID":
                snid = int(parts[1].strip())
            elif key == "SNTYPE":
                sn_type = int(parts[1].strip())
            elif key == "RA":
                ra = float(parts[1].split("deg")[0].strip()) / DEFAULT_NORMALIZATION.position_norm
            elif key == "DECL":
                decl = float(parts[1].split("deg")[0].strip()) / DEFAULT_NORMALIZATION.position_norm
            elif key == "MWEBV":
                mwebv = float(parts[1].split("MW")[0].strip())
            elif key == "HOST_GALAXY_GALID":
                hostid = int(parts[1].strip())
            elif key == "HOST_GALAXY_PHOTO-Z":
                hostz = (
                    float(parts[1].split("+-")[0].strip()),
                    float(parts[1].split("+-")[1].strip()),
                )
            elif key == "REDSHIFT_SPEC":
                spec = (
                    float(parts[1].split("+-")[0].strip()),
                    float(parts[1].split("+-")[1].strip()),
                )
            elif key == "SIM_COMMENT":
                sim_type = parts[1].split("SN Type =")[1].split(",")[0].strip()
            elif key == "SIM_REDSHIFT":
                sim_z = float(parts[1])
            elif key == "OBS":
                fields = parts[1].split()
                observations.append(
                    SPCCRawObservation(
                        mjd=float(fields[0]),
                        band=fields[1],
                        flux=float(fields[3]) / DEFAULT_NORMALIZATION.flux_norm,
                        flux_err=float(fields[4]) / DEFAULT_NORMALIZATION.flux_norm,
                    )
                )

    return SPCCRawEvent(
        survey=survey,
        snid=snid,
        sn_type=sn_type,
        sim_type=sim_type,
        sim_z=sim_z,
        ra=ra,
        decl=decl,
        mwebv=mwebv,
        hostid=hostid,
        hostz=hostz,
        spec=spec,
        observations=observations,
    )


def normalize_event_observations(event: SPCCRawEvent) -> list[NormalizedSPCCObservation]:
    if not event.observations:
        return []
    first_obs = event.observations[0].mjd
    return [
        NormalizedSPCCObservation(
            time=(obs.mjd - first_obs) / DEFAULT_NORMALIZATION.time_norm,
            band=obs.band,
            flux=obs.flux,
            flux_err=obs.flux_err,
        )
        for obs in event.observations
    ]
