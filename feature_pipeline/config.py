"""Shared configuration for the Phase 2 SPCC preprocessing pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SPCCNormalizationConfig:
    flux_norm: float = 1.0
    time_norm: float = 1.0
    position_norm: float = 1.0


@dataclass(frozen=True)
class SPCCGroupingConfig:
    grouping: float = 1.0


DEFAULT_NORMALIZATION = SPCCNormalizationConfig()
DEFAULT_GROUPING = SPCCGroupingConfig()
