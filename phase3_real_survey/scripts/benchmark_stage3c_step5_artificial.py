#!/usr/bin/env python3
"""Full-dimension artificial benchmark for the Stage 3C analysis engine.

This script constructs compact in-memory arrays with the frozen production
dimensions. It never opens the production Monte Carlo JSONL.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from stage3c_step5_analysis import AnalysisConfig, Stage3CData, analyze_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timing-backend",
        choices=("native", "python"),
        default="native",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = AnalysisConfig(timing_bootstrap_backend=args.timing_backend)
    references = {
        (survey, stratum, round(float(redshift), 3)): 0
        for survey, stratum in config.branches
        for redshift in config.redshifts
    }
    data = Stage3CData(config, references)
    data.seen[:] = True
    realization_ids = np.arange(config.n_realizations)

    for branch_i in range(len(config.branches)):
        for z_i, _ in enumerate(config.redshifts):
            for sb_i, _ in enumerate(config.sbmags):
                accepted = (
                    realization_ids + branch_i + z_i + sb_i
                ) % 37 != 0
                data.accepted[branch_i, z_i, sb_i] = accepted
                for feature_i in range(len(config.features)):
                    available = accepted & (
                        (realization_ids + feature_i) % (41 + feature_i) != 0
                    )
                    values = (
                        branch_i * 0.25
                        + z_i * 0.01
                        + sb_i * 0.001
                        + realization_ids * 0.0001
                        + feature_i * 0.00001
                    )
                    data.features[
                        branch_i, z_i, sb_i, :, feature_i
                    ] = values
                    data.feature_available[
                        branch_i, z_i, sb_i, :, feature_i
                    ] = available

                timing_available = data.feature_available[
                    branch_i,
                    z_i,
                    sb_i,
                    :,
                    data.feature_index["r_time_of_peak"],
                ]
                timing = (realization_ids % 19 - 9) * (
                    1.0 + branch_i * 0.1
                )
                data.timing_error[branch_i, z_i, sb_i] = timing
                data.timing_available[
                    branch_i, z_i, sb_i
                ] = timing_available

    started = time.perf_counter()
    result = analyze_data(
        data,
        input_provenance={"kind": "full-dimension-artificial-benchmark"},
        code_provenance={"benchmark": "artificial"},
    )
    elapsed = time.perf_counter() - started
    print(f"timing_backend={args.timing_backend}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(
        "feature_rows="
        f"{len(result['primary_and_supportive_feature_statistics'])}"
    )
    print(
        "timing_comparisons="
        f"{len(result['timing_comparisons'])}"
    )
    print(
        "undefined_diagnostics="
        f"{len(result['undefined_statistic_diagnostics'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
