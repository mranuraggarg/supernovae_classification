# Stage 3C Step 5 Analysis Validation

Date: 2026-10-12

Status: implementation and artificial-data validation complete; production
scientific analysis not executed.

## Scope

The integrated engine implements the frozen Stage 3C Step 5 methodology in
`stage3c_step5_statistical_analysis_clarification_2026-10-12.md`. It does not
train a classifier, modify the Monte Carlo, pool registered comparisons, tune
an estimator, or generate a scientific interpretation.

## Implemented Analysis

- Exact four-branch, 11-redshift, eight-SBMAG, 500-realization grid validation.
- Duplicate, missing, and unexpected realization-ID detection, including
  rejected records in the required grid.
- Separate record-acceptance and per-feature finite-availability masks.
- Four separately retained DES-minus-SDSS comparisons.
- Five primary features and supportive `time_span`.
- Marginal median, 0.1-quantile, and 0.9-quantile differences plus uncorrected
  Wasserstein-1 distance.
- Frozen 5,000-replicate PCG64 shared-realization bootstrap.
- Pointwise percentile intervals and aligned 11-redshift simultaneous bands.
- Explicit counts and unavailable results for undefined bootstrap replicates.
- Survey-specific Hsiao r-band physical timing errors and all frozen timing
  summaries, with MAD as primary dispersion and IQR as corroborative.
- Frozen passband-prediction nodes and provenance. Observed-minus-predicted
  residuals are enabled only for the mathematically compatible colour proxies.
- Atomic JSON output, output manifest, input/code provenance, explicit units,
  and refusal to overwrite an existing result.

## Native Timing Kernel

The six timing summaries have both a NumPy reference implementation and a
C++17 implementation. The command-line runner defaults to the native kernel.
The native implementation changes only execution speed: bootstrap indices,
masks, estimators, intervals, and simultaneous-band calculations remain in the
frozen Python orchestration. Compiler, source, and generated-library checksums
are recorded in the execution metadata.

## Automated Validation

Command:

```bash
/opt/miniconda3/envs/astro-ml/bin/python -m unittest discover \
  -s phase3_real_survey/tests -p 'test_stage3c_step5*.py' -v
```

Result: 33 tests passed.

Coverage includes all four comparisons, complete and invalid grids, duplicate
and unexpected IDs, rejected records, accepted records with unavailable
features, feature-specific masks, marginal rather than paired median
differences, shared paired resampling, deterministic seeds, pointwise
intervals, simultaneous bands, undefined replicates, physical timing errors,
MAD/IQR comparisons, two-pass JSONL equivalence, strict JSON serialization,
output metadata, record-order independence, and direct native-versus-reference
numerical agreement. The frozen 176,000-record grid validator is exercised with
a generated full grid.

Additional checks passed:

- Python syntax compilation and module imports.
- Command-line help and frozen timing-reference reconstruction (44 branch and
  redshift references).
- C++17 compilation with Apple clang 21.0.0.
- Repository whitespace check with `git diff --check`.

## Full-Dimension Artificial Benchmark

Command:

```bash
/usr/bin/time -l /opt/miniconda3/envs/astro-ml/bin/python \
  phase3_real_survey/scripts/benchmark_stage3c_step5_artificial.py \
  --timing-backend native
```

The benchmark uses the production dimensions and 5,000 bootstrap replicates,
but constructs artificial arrays and never opens the production JSONL.

- Analysis-engine elapsed time: 75.10 seconds.
- Wall time including interpreter startup: 75.97 seconds.
- Maximum resident set reported by `/usr/bin/time`: 586,727,424 bytes.
- macOS peak-memory-footprint metric: 182,584,040 bytes.
- Expected output cardinalities obtained: 2,112 feature rows and 2,112 timing
  comparison rows.
- Undefined artificial statistics: zero for this fixture.

The NumPy reference benchmark took 357.15 seconds on the same machine before
the native timing kernel was enabled. A realistic production estimate is about
two to four minutes and below 0.8 GB resident memory, allowing for two streaming
JSONL passes, hashing, response loading, and output serialization. The Python
fallback is expected to take roughly six to nine minutes.

## Production Input Check

Only filesystem metadata and SHA-256 were inspected. The production records
were not parsed by the analysis engine.

- Size: 217,820,289 bytes.
- SHA-256: `2bbfc6454ad7a1f6457442c30d47418ea2d71a8987eb73c9309fe07d021cada0`.
- Expected checksum match: yes.

## Files

Created:

- `phase3_real_survey/scripts/stage3c_step5_analysis.py`
- `phase3_real_survey/scripts/run_stage3c_step5_analysis.py`
- `phase3_real_survey/scripts/stage3c_step5_native.py`
- `phase3_real_survey/native/stage3c_bootstrap_timing.cpp`
- `phase3_real_survey/scripts/benchmark_stage3c_step5_artificial.py`
- `phase3_real_survey/tests/test_stage3c_step5_integration.py`
- `phase3_real_survey/schemas/stage3c_step5_analysis_output.schema.json`
- `phase3_real_survey/docs/validation/stage3c_step5_analysis_validation.md`

Modified:

- `phase3_real_survey/scripts/stage3c_step5_records.py`
- `phase3_real_survey/scripts/stage3c_step5_statistics.py`
- `phase3_real_survey/tests/test_stage3c_step5_core.py`
- `.gitignore`

The production simulation runner and all frozen preregistration and production
files remain unchanged.

## Remaining Limitations

- Native execution requires a C++17 compiler; the validated NumPy backend is
  available when native compilation is unsuitable.
- Under the frozen rule, any undefined replicate makes its pointwise interval
  and associated simultaneous band unavailable. The engine reports rather than
  repairs this condition.
- Synthetic peak-flux predictions are AB-magnitude differences, whereas the
  frozen extracted peak features are `log10(1+FLUXCAL)`. The engine records the
  prediction but does not create an unsupported residual or normalization.
- The JSON Schema fixes the top-level contract and principal containers; the
  integration tests enforce the detailed row semantics.

## Authorized Production Command

Do not run until the researcher authorizes scientific execution.

```bash
/opt/miniconda3/envs/astro-ml/bin/python \
  phase3_real_survey/scripts/run_stage3c_step5_analysis.py \
  --input phase3_real_survey/results/stage3c_production_monte_carlo/stage3c_production_branches.jsonl \
  --expected-sha256 2bbfc6454ad7a1f6457442c30d47418ea2d71a8987eb73c9309fe07d021cada0 \
  --output phase3_real_survey/results/stage3c_step5_statistical_analysis/stage3c_step5_analysis.json \
  --timing-backend native
```

The engine is ready for researcher-authorized scientific execution.
