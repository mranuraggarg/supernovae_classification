"""Internal schemas for the owned SPCC preprocessing pipeline."""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class SPCCRawObservation:
    mjd: float
    band: str
    flux: float
    flux_err: float


@dataclass(frozen=True)
class SPCCRawEvent:
    survey: Optional[str]
    snid: Optional[int]
    sn_type: Optional[int]
    sim_type: Optional[str]
    sim_z: Optional[float]
    ra: Optional[float]
    decl: Optional[float]
    mwebv: Optional[float]
    hostid: Optional[int]
    hostz: Optional[tuple[float, float]]
    spec: Optional[tuple[float, float]]
    observations: list[SPCCRawObservation]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["observation_count"] = len(self.observations)
        payload["bands"] = sorted({obs.band for obs in self.observations})
        return payload
