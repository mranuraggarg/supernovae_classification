# Phase 2 Tier 4 Cross-Survey Comparability Note

## Purpose

Phase 2 Tier 4 asks whether the compact feature representation is still useful when the data source changes.

The central challenge is that SPCC and PLAsTiCC are not naturally identical surveys. If the two datasets are compared using different feature-construction rules, then poor transfer may reflect pipeline mismatch rather than genuine domain shift.

The main Tier 4 engineering work therefore focused on making SPCC and PLAsTiCC as comparable as possible before interpreting the domain-generalization results scientifically.

## Why The Earlier Setup Was Not Enough

The original Tier 1 compact table was optimized around SPCC-specific feature extraction using reconstructed bandwise light curves. PLAsTiCC, by contrast, entered Tier 4 through raw CSV observations.

That created a major asymmetry:

- SPCC features came from reconstructed light curves.
- PLAsTiCC features came from direct observed points.
- Color and timing features were therefore not defined under fully matching rules.

In early Tier 4 experiments this mismatch produced very poor SPCC to PLAsTiCC transfer and made it unclear whether the failure was scientific or merely procedural.

## Final Tier 4 Comparability Strategy

The final Tier 4 setup does not reuse the old SPCC compact baseline table as the cross-survey reference representation.

Instead, both surveys are rebuilt under one shared raw-observation compact-feature pipeline inside [phase2_tier4_make_variants.py](/Users/anuraggarg/work/supernovae_classification/phase2_tier4_make_variants.py).

The key design choice is simple:

- use raw observations for both SPCC and PLAsTiCC,
- apply the same compact-feature formulas to both,
- apply the same event-window logic to both,
- then evaluate domain generalization.

This makes the comparison much more defensible scientifically.

## Step 1: Rebuild SPCC Compact Features From Raw SPCC Observations

SPCC compact features are now rebuilt directly from raw DES `.DAT` files rather than copied from the older reconstructed Tier 1 compact table.

In practice this means:

1. Raw SPCC events are loaded from `data/spcc/raw/DES_*.DAT`.
2. Each event is cleaned using the existing SPCC cleaning rules.
3. Observations are sorted by time.
4. Compact features are computed from the observed light-curve points themselves.

This change matters because it removes the previous asymmetry in which SPCC used reconstructed curves but PLAsTiCC used raw observations.

## Step 2: Build PLAsTiCC Compact Features Under The Same Rules

PLAsTiCC compact features are now built from raw PLAsTiCC observations using the same compact-feature formulas used for SPCC.

The builder reads:

- `data/PLAsTiCC/training_set.csv`
- `data/PLAsTiCC/training_set_metadata.csv`

and converts each object into the same 16-feature compact representation used by Tier 4.

This gives SPCC and PLAsTiCC a common feature schema with comparable meaning.

## Step 3: Use The Same Active-Window Rule In Both Surveys

One of the most important Tier 4 comparability changes is the definition of the event window.

For both SPCC and PLAsTiCC, the pipeline now defines an active subset of observations using an observation-level signal-to-noise rule:

\[
\frac{F}{\sigma_F} \ge 3
\]

using only observations with positive flux and positive flux uncertainty.

If such active observations exist, they define the event window. If not, the full observation set is used as fallback.

This active-window rule is applied identically in both surveys.

Why this helps:

- it reduces dependence on long low-information tails,
- it makes event duration more comparable across surveys,
- it avoids one survey receiving a broader temporal window than the other simply because of cadence or background noise.

## Step 4: Compute The Same Core Summary Statistics In Both Surveys

Once the event window is fixed, the same compact statistics are extracted for each band:

- peak flux,
- mean flux,
- standard deviation of flux,
- time of peak,
- and for the `i` band, amplitude.

Global temporal coverage is summarized by:

- `time_span`

defined as the event-window duration.

Band peak times are measured relative to the common event start:

\[
t^{(b)}_{\mathrm{peak}} = t^{(b)}_{\max} - t_{\mathrm{start}}
\]

and clipped into the valid event interval.

This is why the later PLAsTiCC audit reports show zero time-consistency violations.

## Step 5: Compress Features In The Same Way

The compact features remain numerically compressed before training, again using one shared rule for both surveys:

- positive flux-like quantities use `log10(1 + value)` after non-negative clipping,
- signed mean-flux quantities use signed `log10(1 + |value|)`,
- color features are clipped to a fixed bounded range.

This shared compression rule keeps feature scales manageable while preserving directionality.

## Step 6: Color Features Are Built Under Shared Raw-Observation Logic

Color mismatch was one of the hardest remaining Tier 4 problems.

The final retained solution is:

- do not use the old SPCC reconstructed color logic,
- do not use the stricter same-epoch interpolation version that was later tested and rolled back,
- instead compute color from a representative positive flux in each band.

For each band, the representative flux is defined as the mean of the strongest positive observations in that band:

\[
F^{(b)}_{\mathrm{rep}} = \mathrm{mean}(\text{top positive flux observations in band } b)
\]

and the color proxies are then defined by the same magnitude-style flux-ratio rule:

\[
g-r = -2.5 \log_{10}\left(\frac{F^{(g)}_{\mathrm{rep}}}{F^{(r)}_{\mathrm{rep}}}\right)
\]

\[
r-i = -2.5 \log_{10}\left(\frac{F^{(r)}_{\mathrm{rep}}}{F^{(i)}_{\mathrm{rep}}}\right)
\]

\[
i-z = -2.5 \log_{10}\left(\frac{F^{(i)}_{\mathrm{rep}}}{F^{(z)}_{\mathrm{rep}}}\right)
\]

Only events with usable positive support across all four `g/r/i/z` bands are kept in this parity-safe compact set.

This shared representative-flux rule was chosen because it improved SPCC–PLAsTiCC comparability while keeping a much larger usable sample than the stricter same-epoch interpolation design.

## Step 7: Why The Same-Epoch Color Upgrade Was Not Kept

Scientifically, that idea was attractive because it reduced cadence mismatch further.

However, in practice it created several problems that made the strict rule unsuitable as the main Tier-4 pipeline.

During testing, the usable PLAsTiCC sample size decreased strongly when the same-epoch requirement was enforced.

Approximate training row counts were:

- original compact extraction: ~7800 events  
- first parity upgrade: ~6200 events  
- strict same-epoch upgrade: ~2700 events  

This large reduction indicates that the stricter rule was too restrictive for realistic cadence differences between SPCC and PLAsTiCC.

In addition, the strict setup produced unstable behaviour in shift tests, especially when `i` or `z` band information was removed, suggesting that the feature set became overly dependent on tightly synchronized color measurements.

Because of this, the improved transfer observed under the strict rule could reflect both better feature parity and strong sample selection.

For this reason, the strict same-epoch interpolation design is treated as a sensitivity test, while the representative-flux color definition is retained as the main Tier-4 pipeline.


## Step 8: Why This Final Setup Is Better

The final Tier 4 setup is stronger than the original one for three reasons:

1. SPCC and PLAsTiCC are now both built from raw observations rather than from mismatched source representations.
2. Time-window and time-of-peak definitions are now consistent across surveys.
3. Color features are now generated using one shared survey-agnostic rule.

This does not make the two surveys perfectly identical, but it makes the comparison much more scientifically meaningful.

Distribution-shift tests show that the compact representation remains stable under noise and reduced time span, shows moderate degradation under flux scaling, and larger but non-catastrophic degradation when entire bands are removed (largest drop for missing i band). These results indicate partial robustness rather than full survey invariance.

## What The Results Mean Under This Setup

Under the final retained setup, Tier 4 no longer asks:

"Does a model trained on SPCC transfer to PLAsTiCC despite a mismatched feature-construction pipeline?"

Instead it asks a better question:

"Does the compact representation generalize across surveys once both surveys are mapped into a shared raw-observation compact feature space?"


That is a substantially stronger scientific test.


In the final Tier 4 evaluation, direct SPCC→PLAsTiCC transfer still shows a measurable performance drop relative to within-survey evaluation, indicating that the compact representation is not fully survey-invariant.

However, the window-harmonization study demonstrated that transfer performance improves when PLAsTiCC compact features are computed over transient-centered windows comparable to the characteristic duration scale of SPCC events. This result suggests that part of the observed cross-survey gap originates from differences in the effective temporal domain used during feature construction rather than from classifier failure alone.

Mixed-domain training further improves transfer performance, while distribution-shift tests continue to show moderate but non-catastrophic degradation. Taken together, these results support the interpretation that the compact feature representation captures physically meaningful structure, but that survey-dependent feature distributions remain an important limitation.

## Class-Conditional Centroid Analysis

To determine whether the remaining transfer limitation originated from classifier behaviour or from the feature representation itself, a class-conditional centroid analysis was performed using the final compact 16-feature space.

The analysis measured four standardized centroids:

- SPCC Ia,
- PLAsTiCC Ia,
- SPCC non-Ia,
- PLAsTiCC non-Ia.

The resulting distances were:

| Quantity | Distance |
|----------|----------|
| SPCC Ia → PLAsTiCC Ia | 6.125 |
| SPCC non-Ia → PLAsTiCC non-Ia | 7.175 |
| SPCC Ia → SPCC non-Ia | 1.710 |
| PLAsTiCC Ia → PLAsTiCC non-Ia | 2.448 |

A useful comparison is the ratio between the cross-survey Ia shift and the intrinsic class separation inside each survey:

| Metric | Value |
|---------|-------|
| Ia shift / SPCC class separation | 3.58× |
| Ia shift / PLAsTiCC class separation | 2.50× |

This result is important because the same astrophysical class should ideally occupy a similar region of feature space across surveys if direct transfer is to succeed.

Instead, the SPCC Ia and PLAsTiCC Ia centroids are separated by a distance substantially larger than the Ia/non-Ia separation within either survey.

This indicates that the dominant limitation is not merely a classifier-boundary problem. The compact feature representation itself is not fully survey invariant.

The centroid result therefore strengthens the interpretation of the window-harmonization study. Event-window harmonization improves transfer because it reduces one source of mismatch, but it does not completely align the feature distributions of the two surveys.

The Tier-4 conclusion is therefore not that compact features fail. Rather, the compact representation remains physically meaningful and useful within a survey, while direct SPCC→PLAsTiCC transfer remains limited by residual domain shift in the compact feature space.


## Revised Tier-4 Scientific Interpretation

The original Tier-4 question was whether a compact physically interpretable model trained on SPCC could transfer directly to PLAsTiCC.

The combined evidence from feature-definition audits, window-harmonization experiments, distribution-shift tests, mixed-domain training, and centroid analysis suggests a more nuanced conclusion.

The compact feature representation captures physically meaningful transient structure and remains reasonably robust under a range of perturbations. However, the representation is not fully survey invariant.

After harmonizing feature definitions and temporal windows, substantial class-conditional separation between SPCC and PLAsTiCC still remains. The observed transfer gap therefore reflects genuine survey-domain differences rather than a simple preprocessing artifact.

A more defensible interpretation is that compact physically interpretable features provide a useful common representation across surveys, but successful deployment on unseen survey domains may require feature-space harmonization, domain adaptation, or restriction to a smaller set of survey-stable features.

This distinction is scientifically important because it separates the value of interpretability from the stronger claim of survey universality.

## Remaining Limitation

Even after the comparability upgrades, a residual parity mismatch can remain, most notably in `peak_color_r_minus_i` and occasionally in other color features. The final retained Tier-4 pipeline therefore represents a substantially improved cross-survey comparison, but not a claim of perfect survey equivalence.

This means the Tier 4 result should still be interpreted as:

- scientifically meaningful,
- much more reliable than the earlier mismatch-dominated version,
- but not a claim of perfect survey equivalence.

That is an acceptable and honest limitation for the paper.

## Tier-4 Window-Harmonization Investigation

After the feature-definition audit, a second Tier-4 question emerged.

Even when SPCC and PLAsTiCC used the same compact-feature formulas, the two surveys were still being summarized over very different effective temporal windows.

The audit showed that SPCC compact features were typically derived from transient-scale event durations, whereas PLAsTiCC compact features often reflected much longer survey histories.

This raised the possibility that identical feature names were still encoding different physical time domains.

To test this hypothesis, PLAsTiCC compact features were rebuilt using transient-centered windows defined around the strongest detected flux measurement.

A science-constrained sweep was then performed using:

- ±60 days,
- ±75 days,
- ±90 days,
- ±105 days,
- ±120 days,
- ±180 days.

These values were not chosen through model tuning. Instead they were selected to bracket the characteristic duration scales observed in SPCC.

The intent was to determine whether cross-survey transfer improves when PLAsTiCC compact features are computed over temporal windows comparable to those represented in SPCC.

## Why The ±60 Day Window Was Retained

The window sweep produced two particularly strong transfer regimes.

The ±60 day and ±105 day windows both improved SPCC→PLAsTiCC transfer relative to the non-windowed baseline.

However, the ±60 day window was retained as the primary Tier-4 configuration.

The reason is scientific rather than purely performance-based.

The median SPCC event duration is approximately 67 days, making the ±60 day window the closest representation of the central tendency of the training survey.

By contrast, the ±105 day window more closely reflects the upper tail of the SPCC duration distribution and therefore assumes access to a substantially larger fraction of the transient evolution.

For a realistic transfer-learning scenario, a newly released survey is unlikely to have complete transient histories available immediately. A compact representation based on the core transient behaviour is therefore more appropriate than one requiring near-complete light-curve evolution.

The retained interpretation is therefore:

- ±60 days represents the characteristic SPCC transient scale,
- ±105 days serves as a sensitivity analysis demonstrating that the result is not unique to a single window size,
- and improved transfer under both windows supports the broader conclusion that event-window harmonization is an important contributor to cross-survey comparability.

This distinction is important because the goal of Tier-4 is not to optimize a benchmark metric, but to determine whether physically interpretable compact features can transfer to previously unseen survey data under scientifically defensible assumptions.

## Measurement Harmonization Versus Real Domain Shift

An important Tier-4 principle is that not every survey difference should be treated in the same way.

Some differences are removable representation mismatches introduced by preprocessing choices. Others are genuine domain differences produced by the surveys themselves.

This distinction is central to the scientific meaning of Tier 4.

### Removable representation mismatch

These are differences that should be harmonized before cross-survey comparison because they are analogous to measuring the same quantity in different units.

Examples include:

- different feature-construction rules,
- different time-origin conventions,
- different compression or normalization conventions,
- different calibration representations,
- different preprocessing logic for colors or event windows.

If these are left uncorrected, the classifier can fail for procedural reasons rather than scientific ones. In that sense, representation mismatch behaves like a base-measurement error that propagates into the downstream classification model.

The Tier-4 comparability work is therefore intended to reduce this removable mismatch.

### Genuine domain difference

These are differences that should remain in the experiment because they are the real content of cross-survey generalization.

Examples include:

- survey cadence,
- missing bands,
- noise level,
- photometric depth,
- class composition,
- redshift distribution,
- instrument throughput differences that remain after consistent feature definition.

These are not pipeline bugs. They are the actual observational conditions under which the compact representation must generalize.

### Practical Tier-4 rule

The correct strategy is therefore:

- harmonize differences caused by the feature pipeline,
- preserve differences caused by the surveys themselves.

This is why Tier 4 uses shared feature semantics across surveys while still allowing realized feature values to differ from one survey to another.

For example, `r_peak_flux` should mean the same mathematical quantity in both surveys, even though its observed distribution may still differ because the instruments and observing conditions differ. The definition should be invariant; the measured values need not be.

This framework gives Tier 4 a defensible scientific interpretation:

- poor transfer after harmonization suggests genuine domain shift,
- poor transfer before harmonization may only reflect pipeline mismatch.

## Practical Summary

The final retained Tier 4 comparability workflow is:

1. rebuild SPCC compact features from raw SPCC observations,
2. build PLAsTiCC compact features from raw PLAsTiCC observations,
3. use the same observation-level active-window rule in both surveys,
4. use the same bandwise summary features in both surveys,
5. use the same compression rules in both surveys,
6. define colors using the same representative positive-flux rule in both surveys,
7. run domain generalization on this shared compact feature space.

This is the version that should be described as the final Tier 4 methodology.
