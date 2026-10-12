"""Artificial-data integration tests for the production Stage 3C engine.

No production Monte Carlo record is opened by this suite.
"""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage3c_step5_analysis import (  # noqa: E402
    AnalysisConfig,
    COMPARISONS,
    analyze_jsonl,
    analyze_records,
    comparison_name,
    sha256_file,
)
from stage3c_step5_records import (  # noqa: E402
    BRANCHES,
    REDSHIFTS,
    REGISTERED_FEATURES,
    SBMAGS,
    validate_complete_grid,
    validate_grid,
)
from stage3c_step5_statistics import (  # noqa: E402
    bootstrap_frequency_counts,
    bootstrap_timing_statistics,
    bootstrap_weighted_quantile,
    paired_bootstrap_indices,
)


def artificial_config(redshifts=(0.05, 0.075), sbmags=(20.5,),
                      n_realizations=4, n_bootstrap=64):
    return AnalysisConfig(
        redshifts=tuple(redshifts),
        sbmags=tuple(sbmags),
        n_realizations=n_realizations,
        n_bootstrap=n_bootstrap,
    )


def timing_references(config, phase=0):
    return {
        (survey, stratum, round(float(z), 3)): phase
        for survey, stratum in BRANCHES
        for z in config.redshifts
    }


def make_records(config, *, rejected=None, unavailable=None):
    rejected = set(rejected or ())
    unavailable = set(unavailable or ())
    branch_offsets = {
        ("SDSS-II", "82N"): 0.0,
        ("SDSS-II", "82S"): 10.0,
        ("DES-SN5YR", "SHALLOW"): 20.0,
        ("DES-SN5YR", "DEEP"): 30.0,
    }
    timing_values = {
        ("SDSS-II", "82N"): [0.0, 0.0, 0.0, 0.0],
        ("SDSS-II", "82S"): [1.0, 1.0, 1.0, 1.0],
        ("DES-SN5YR", "SHALLOW"): [0.0, 2.0, 4.0, 6.0],
        ("DES-SN5YR", "DEEP"): [0.0, 4.0, 8.0, 12.0],
    }
    records = []
    for branch in BRANCHES:
        for z in config.redshifts:
            for sbmag in config.sbmags:
                for realization_id in range(config.n_realizations):
                    key = (
                        branch[0], branch[1], round(float(z), 3),
                        round(float(sbmag), 1), realization_id,
                    )
                    common = {
                        "survey": branch[0],
                        "stratum": branch[1],
                        "redshift": float(z),
                        "sbmag": float(sbmag),
                        "realization_id": realization_id,
                        "peak_mjd": 59000.0,
                        "feature_audit": {"first_active_mjd": 59000.0},
                    }
                    if key in rejected:
                        records.append({
                            **common,
                            "feature_status": "rejected",
                            "feature_rejection_reason": "artificial_rejection",
                            "features": None,
                        })
                        continue

                    values = {
                        feature: branch_offsets[branch] + realization_id
                        for feature in REGISTERED_FEATURES
                    }
                    values["r_time_of_peak"] = timing_values[branch][realization_id]

                    # A deliberately asymmetric fixture distinguishes the
                    # difference of marginal medians from paired differences.
                    if z == config.redshifts[0] and sbmag == config.sbmags[0]:
                        if branch == ("SDSS-II", "82N"):
                            values["i_peak_flux"] = [0.0, 0.0, 0.0, 100.0][realization_id]
                        elif branch == ("DES-SN5YR", "SHALLOW"):
                            values["i_peak_flux"] = [0.0, 1.0, 2.0, 3.0][realization_id]

                    for survey, stratum, uz, usb, uid, feature in unavailable:
                        if key == (survey, stratum, uz, usb, uid):
                            values[feature] = None

                    records.append({
                        **common,
                        "feature_status": "accepted",
                        "feature_rejection_reason": None,
                        "features": values,
                    })
    return records


def select(rows, **criteria):
    matches = [
        row for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row, found {len(matches)}: {criteria}")
    return matches[0]


class TestGridIntegration(unittest.TestCase):

    def setUp(self):
        self.config = artificial_config()
        self.records = make_records(self.config)

    def test_complete_artificial_grid_and_all_comparisons(self):
        summary = validate_grid(
            self.records,
            branches=self.config.branches,
            redshifts=self.config.redshifts,
            sbmags=self.config.sbmags,
            n_realizations=self.config.n_realizations,
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["total_records"], 32)

        result = analyze_records(
            self.records,
            timing_references(self.config),
            config=self.config,
        )
        names = {
            row["comparison"]
            for row in result["primary_and_supportive_feature_statistics"]
        }
        self.assertEqual(
            names,
            {comparison_name(value) for value in COMPARISONS},
        )

    def test_missing_duplicate_and_unexpected_ids_fail(self):
        with self.assertRaises(ValueError):
            validate_grid(
                self.records[:-1],
                redshifts=self.config.redshifts,
                sbmags=self.config.sbmags,
                n_realizations=self.config.n_realizations,
            )

        with self.assertRaises(ValueError):
            validate_grid(
                self.records + [dict(self.records[0])],
                redshifts=self.config.redshifts,
                sbmags=self.config.sbmags,
                n_realizations=self.config.n_realizations,
            )

        altered = list(self.records)
        replacement = dict(altered[3])
        replacement["realization_id"] = self.config.n_realizations
        altered[3] = replacement
        with self.assertRaises(ValueError):
            validate_grid(
                altered,
                redshifts=self.config.redshifts,
                sbmags=self.config.sbmags,
                n_realizations=self.config.n_realizations,
            )

    def test_frozen_full_grid_validator(self):
        def records():
            for survey, stratum in BRANCHES:
                for redshift in REDSHIFTS:
                    for sbmag in SBMAGS:
                        for realization_id in range(500):
                            yield {
                                "survey": survey,
                                "stratum": stratum,
                                "redshift": redshift,
                                "sbmag": sbmag,
                                "realization_id": realization_id,
                                "peak_mjd": 59000.0,
                                "feature_status": "rejected",
                                "feature_rejection_reason": "artificial",
                                "feature_audit": {},
                                "features": None,
                            }

        self.assertEqual(validate_complete_grid(records()), 176000)


class TestAnalysisIntegration(unittest.TestCase):

    def setUp(self):
        self.config = artificial_config()

    def analyze(self, records):
        return analyze_records(
            records,
            timing_references(self.config),
            config=self.config,
            input_provenance={"kind": "artificial"},
            code_provenance={"test": "artificial"},
        )

    def test_marginal_median_and_timing_dispersion(self):
        result = self.analyze(make_records(self.config))
        comparison = comparison_name(COMPARISONS[0])
        row = select(
            result["primary_and_supportive_feature_statistics"],
            comparison=comparison,
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
            feature="i_peak_flux",
        )
        self.assertEqual(row["median_difference_des_minus_sdss"], 1.5)
        self.assertNotEqual(
            row["median_difference_des_minus_sdss"],
            np.median(np.array([0.0, 1.0, 2.0, 3.0]) - np.array([0.0, 0.0, 0.0, 100.0])),
        )

        timing = select(
            result["timing_comparisons"],
            comparison=comparison,
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
            metric="mad_days",
        )
        self.assertEqual(timing["difference_des_minus_sdss"], 2.0)
        iqr = select(
            result["timing_comparisons"],
            comparison=comparison,
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
            metric="iqr_days",
        )
        self.assertEqual(iqr["difference_des_minus_sdss"], 3.0)

    def test_rejection_and_feature_availability_are_separate(self):
        rejected = {("DES-SN5YR", "DEEP", 0.05, 20.5, 3)}
        unavailable = {
            ("DES-SN5YR", "SHALLOW", 0.05, 20.5, 2, "z_peak_flux")
        }
        result = self.analyze(
            make_records(
                self.config,
                rejected=rejected,
                unavailable=unavailable,
            )
        )
        acceptance = select(
            result["acceptance_statistics"],
            survey="DES-SN5YR",
            stratum="DEEP",
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
        )
        self.assertEqual(acceptance["accepted_realizations"], 3)
        availability = select(
            result["feature_availability_statistics"],
            survey="DES-SN5YR",
            stratum="SHALLOW",
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
            feature="z_peak_flux",
        )
        self.assertEqual(availability["accepted_realizations"], 4)
        self.assertEqual(availability["feature_available_realizations"], 3)
        self.assertEqual(availability["feature_unavailable_among_accepted"], 1)

    def test_native_timing_kernel_matches_python_reference(self):
        values = np.array([-4.0, -1.5, 0.0, 2.0, 9.0, np.nan])
        available = np.array([True, True, False, True, True, False])
        indices = paired_bootstrap_indices(
            0.125,
            24.5,
            n_realizations=values.size,
            n_bootstrap=257,
        )
        reference = bootstrap_timing_statistics(
            values, available, indices, backend="python"
        )
        native = bootstrap_timing_statistics(
            values, available, indices, backend="native"
        )
        self.assertEqual(reference.keys(), native.keys())
        for metric in reference:
            np.testing.assert_allclose(
                native[metric], reference[metric], rtol=0.0, atol=1e-12,
                equal_nan=True,
            )

        unavailable = np.zeros(values.size, dtype=bool)
        native_undefined = bootstrap_timing_statistics(
            values, unavailable, indices, backend="native"
        )
        self.assertTrue(all(
            np.all(np.isnan(replicates))
            for replicates in native_undefined.values()
        ))

    def test_integrated_native_backend_matches_reference_analysis(self):
        records = make_records(self.config)
        reference = self.analyze(records)
        native_config = artificial_config()
        native_config = AnalysisConfig(
            redshifts=native_config.redshifts,
            sbmags=native_config.sbmags,
            n_realizations=native_config.n_realizations,
            n_bootstrap=native_config.n_bootstrap,
            timing_bootstrap_backend="native",
        )
        native = analyze_records(
            records,
            timing_references(native_config),
            config=native_config,
            input_provenance={"kind": "artificial"},
            code_provenance={"test": "artificial"},
        )
        self.assertEqual(
            native["primary_and_supportive_feature_statistics"],
            reference["primary_and_supportive_feature_statistics"],
        )
        for actual, expected in zip(
            native["timing_comparisons"], reference["timing_comparisons"]
        ):
            self.assertEqual(actual.keys(), expected.keys())
            self.assertEqual(actual["comparison"], expected["comparison"])
            self.assertEqual(actual["metric"], expected["metric"])
            self.assertAlmostEqual(
                actual["difference_des_minus_sdss"],
                expected["difference_des_minus_sdss"],
                places=12,
            )
            for bound in ("lower", "upper"):
                self.assertAlmostEqual(
                    actual["bootstrap"][bound],
                    expected["bootstrap"][bound],
                    places=12,
                )

    def test_determinism_and_input_order_independence(self):
        records = make_records(self.config)
        shuffled = list(records)
        random.Random(441).shuffle(shuffled)
        first = self.analyze(records)
        second = self.analyze(shuffled)
        self.assertEqual(first, second)

    def test_jsonl_two_pass_analysis_matches_in_memory_engine(self):
        records = make_records(self.config)
        expected = self.analyze(records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artificial.jsonl"
            with path.open("w") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            actual = analyze_jsonl(
                path,
                timing_references(self.config),
                config=self.config,
                expected_sha256=sha256_file(path),
                code_provenance={"test": "artificial"},
            )
        self.assertEqual(
            actual["primary_and_supportive_feature_statistics"],
            expected["primary_and_supportive_feature_statistics"],
        )
        self.assertTrue(actual["input_provenance"]["sha256_matches_expected"])

    def test_undefined_replicates_are_reported_and_bands_unavailable(self):
        config = artificial_config(n_realizations=2, n_bootstrap=128)
        unavailable = set()
        for z in config.redshifts:
            unavailable.add(("DES-SN5YR", "SHALLOW", z, 20.5, 1, "i_peak_flux"))
            unavailable.add(("SDSS-II", "82N", z, 20.5, 0, "i_peak_flux"))
        result = analyze_records(
            make_records(config, unavailable=unavailable),
            timing_references(config),
            config=config,
        )
        comparison = comparison_name(COMPARISONS[0])
        row = select(
            result["primary_and_supportive_feature_statistics"],
            comparison=comparison,
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
            feature="i_peak_flux",
        )
        self.assertGreater(
            row["median_difference_bootstrap"]["undefined_replicates"], 0
        )
        self.assertFalse(row["median_difference_bootstrap"]["available"])
        band = select(
            result["feature_simultaneous_confidence_bands"],
            comparison=comparison,
            sbmag_mag_per_arcsec2=20.5,
            feature="i_peak_flux",
        )
        self.assertFalse(band["available"])
        self.assertTrue(result["undefined_statistic_diagnostics"])

    def test_schema_metadata_and_simultaneous_band(self):
        result = self.analyze(make_records(self.config))
        schema = json.loads(
            (ROOT / "schemas" / "stage3c_step5_analysis_output.schema.json").read_text()
        )
        self.assertEqual(result["schema"], "stage3c-step5-analysis-v1")
        json.dumps(result, allow_nan=False)
        self.assertTrue(set(schema["required"]).issubset(result))
        self.assertFalse(result["reproducibility"]["missing_features_imputed"])
        predictions = result["physical_prediction_metadata"]
        self.assertEqual(
            len(predictions["synthetic_passband_predictions"]["nodes"]), 11
        )
        color_row = select(
            result["primary_and_supportive_feature_statistics"],
            comparison=comparison_name(COMPARISONS[0]),
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
            feature="peak_color_r_minus_i",
        )
        self.assertTrue(
            color_row["registered_passband_prediction"]["residual_available"]
        )
        flux_row = select(
            result["primary_and_supportive_feature_statistics"],
            comparison=comparison_name(COMPARISONS[0]),
            redshift=0.05,
            sbmag_mag_per_arcsec2=20.5,
            feature="i_peak_flux",
        )
        self.assertFalse(
            flux_row["registered_passband_prediction"]["residual_available"]
        )
        band = select(
            result["feature_simultaneous_confidence_bands"],
            comparison=comparison_name(COMPARISONS[0]),
            sbmag_mag_per_arcsec2=20.5,
            feature="peak_color_i_minus_z",
        )
        self.assertTrue(band["available"])
        self.assertEqual(len(band["nodes"]), len(self.config.redshifts))

    def test_weighted_bootstrap_matches_explicit_resampling(self):
        values = np.array([1.0, 2.0, np.nan, 4.0])
        available = np.array([True, True, False, True])
        indices = paired_bootstrap_indices(
            0.05, 20.5, n_realizations=4, n_bootstrap=50
        )
        counts = bootstrap_frequency_counts(indices, 4)
        actual = bootstrap_weighted_quantile(values, available, counts, 0.5)
        expected = []
        for draw in indices:
            selected = values[draw][available[draw]]
            expected.append(np.nan if selected.size == 0 else np.median(selected))
        np.testing.assert_allclose(actual, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
