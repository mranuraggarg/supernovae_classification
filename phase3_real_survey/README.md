# Phase 3 — Real-Survey Physics-First Transfer

## Scientific objective

Investigate whether documented differences between two real observing systems
(SDSS-II SN and DES-SN5YR) explain part of the failure of a compact,
interpretable supernova classifier to transfer between surveys.

The experiment is physics-first.

XGBoost and the frozen compact 16-feature representation are downstream
diagnostics, not the primary subject of the study.

## Selected surveys

- SDSS-II Supernova Survey final three-season SMP release
- DES-SN5YR SMP v1.2

## Status

Survey-pair provenance gate: GO

Next phase:
pre-register physical predictions before inspecting cross-survey compact-feature
distributions.

## Directory structure

- `docs/provenance/` — survey/instrument provenance and audit records
- `preregistration/` — frozen hypotheses, predictions and planned measurements
- `scripts/` — Phase-3 analysis code only
- `results/` — generated numerical results
- `plots/` — generated figures
- `data/` — documentation/manifests only; large survey data are not committed

## Scientific rule

No target-survey feature distributions or classifier results should be used to
formulate the pre-registered physical predictions.
