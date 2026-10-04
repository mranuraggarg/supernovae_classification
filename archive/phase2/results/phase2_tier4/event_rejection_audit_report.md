# Phase 2 Tier 4 Event-Rejection Audit

This audit reconstructs the event-selection pipeline stage by stage. It does not infer rejection percentages from final compact-feature row counts.

## Strongest Positive Observations

For band $b$, let $P_b = \{F_{b,j}: F_{b,j} > 0\}$ be the positive flux measurements available after the active-window and band-fallback rule. Sort them as $F_{b,(1)} \ge F_{b,(2)} \ge \cdots \ge F_{b,(n_b)}$ and set $k_b = \min(3,n_b)$. The representative flux used for color construction is

$$
F^{(b)}_{\rm rep} = \frac{1}{k_b}\sum_{j=1}^{k_b} F_{b,(j)}, \quad n_b > 0.
$$

If $n_b=0$ for any of $g,r,i,z$, the event is rejected at the positive-flux support stage.

## SPCC

- Total events entering audit: 21319
- Accepted events: 20992
- Rejected events: 327 (1.53%)

### Rejection Stage Counts

| Stage | Events | Percent |
| --- | ---: | ---: |
| cleaning_rejection | 0 | 0.00% |
| class_exclusion | 0 | 0.00% |
| compact_feature_rejection_no_griz_observations | 0 | 0.00% |
| missing_band_rejection | 86 | 0.40% |
| positive_flux_support_rejection | 241 | 1.13% |
| accepted | 20992 | 98.47% |

### Ia and non-Ia Counts

| Class | Total | Accepted | Rejected | Rejected percent |
| --- | ---: | ---: | ---: | ---: |
| Ia | 5088 | 4975 | 113 | 2.22% |
| non-Ia | 16231 | 16017 | 214 | 1.32% |

### Counts by Label

| Label | Total | Accepted | Rejected | Rejected percent | Dominant rejection stage |
| --- | ---: | ---: | ---: | ---: | --- |
| II | 12027 | 11916 | 111 | 0.92% | positive_flux_support_rejection |
| IIL | 425 | 408 | 17 | 4.00% | missing_band_rejection |
| IIP | 189 | 185 | 4 | 2.12% | positive_flux_support_rejection |
| IIn | 789 | 786 | 3 | 0.38% | missing_band_rejection |
| Ia | 5088 | 4975 | 113 | 2.22% | positive_flux_support_rejection |
| Ib | 1438 | 1391 | 47 | 3.27% | positive_flux_support_rejection |
| Ibc | 259 | 253 | 6 | 2.32% | positive_flux_support_rejection |
| Ic | 1104 | 1078 | 26 | 2.36% | positive_flux_support_rejection |

## PLAsTiCC training

- Total events entering audit: 7848
- Accepted events: 6235
- Rejected events: 1613 (20.55%)

### Rejection Stage Counts

| Stage | Events | Percent |
| --- | ---: | ---: |
| cleaning_rejection | 0 | 0.00% |
| class_exclusion | 0 | 0.00% |
| compact_feature_rejection_no_griz_observations | 0 | 0.00% |
| missing_band_rejection | 1340 | 17.07% |
| positive_flux_support_rejection | 273 | 3.48% |
| accepted | 6235 | 79.45% |

### Ia and non-Ia Counts

| Class | Total | Accepted | Rejected | Rejected percent |
| --- | ---: | ---: | ---: | ---: |
| Ia | 2313 | 1852 | 461 | 19.93% |
| non-Ia | 5535 | 4383 | 1152 | 20.81% |

### Counts by Label

| Label | Total | Accepted | Rejected | Rejected percent | Dominant rejection stage |
| --- | ---: | ---: | ---: | ---: | --- |
| Ia | 2313 | 1852 | 461 | 19.93% | missing_band_rejection |
| non-Ia | 5535 | 4383 | 1152 | 20.81% | missing_band_rejection |

## Output Files

- CSV summary: `results/phase2_tier4/event_rejection_audit.csv`
- JSON summary: `results/phase2_tier4/event_rejection_audit.json`
- Markdown report: `results/phase2_tier4/event_rejection_audit_report.md`
