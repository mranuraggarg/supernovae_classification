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

In the final Tier 4 evaluation, direct SPCC→PLAsTiCC transfer shows a clear performance drop, while mixed-domain training improves cross-survey performance, and distribution-shift tests produce moderate but not catastrophic degradation. This combination of results supports the interpretation that the compact representation captures physically meaningful structure but is not fully survey-independent.

## Remaining Limitation

Even after the comparability upgrades, a residual parity mismatch can remain, most notably in `peak_color_r_minus_i` and occasionally in other color features. The final retained Tier-4 pipeline therefore represents a substantially improved cross-survey comparison, but not a claim of perfect survey equivalence.

This means the Tier 4 result should still be interpreted as:

- scientifically meaningful,
- much more reliable than the earlier mismatch-dominated version,
- but not a claim of perfect survey equivalence.

That is an acceptable and honest limitation for the paper.

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
