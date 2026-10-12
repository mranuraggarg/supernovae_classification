"""Regression tests for frozen Stage 3C Step 5 statistical and timing logic.

These tests use artificial data only.
No production Monte Carlo records are accessed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage3c_step5_statistics import (
    validated_sample,
    median_difference,
    quantile_difference,
    median_absolute_deviation,
    interquartile_range,
    wasserstein_one,
    stage3c_bootstrap_seed,
    paired_bootstrap_indices,
    masked_median,
    bootstrap_median_differences,
    pointwise_bootstrap_interval,
    simultaneous_redshift_band,
)

from stage3c_step5_timing import (
    hsiao_r_peak_phase,
    r_peak_timing_error,
)


class TestStatisticalCore(unittest.TestCase):

    def test_marginal_median_difference(self):
        des = np.array([0., 10., 20.])
        sdss = np.array([0., 0., 20.])

        expected = np.median(des) - np.median(sdss)

        self.assertEqual(median_difference(des, sdss), expected)

        # This fixture distinguishes marginal-median difference
        # from the median of realization-wise differences.
        self.assertNotEqual(
            expected,
            np.median(des - sdss),
        )

    def test_quantile_difference(self):
        des = np.array([2., 4., 6., 8.])
        sdss = np.array([1., 2., 3., 4.])

        self.assertAlmostEqual(
            quantile_difference(des, sdss, 0.5),
            2.5,
        )

    def test_dispersion(self):
        x = np.array([0., 1., 2., 3., 4.])

        self.assertEqual(
            median_absolute_deviation(x), 1.0
        )

        self.assertEqual(
            interquartile_range(x), 2.0
        )

    def test_wasserstein(self):
        a = np.array([0., 1., 2.])
        b = np.array([1., 2., 3.])

        self.assertAlmostEqual(
            wasserstein_one(a, b), 1.0
        )

    def test_invalid_sample(self):
        with self.assertRaises(ValueError):
            validated_sample([])

        with self.assertRaises(ValueError):
            validated_sample([1., np.nan])

    def test_bootstrap_determinism(self):
        a = paired_bootstrap_indices(
            0.150, 23.5,
            n_realizations=500,
            n_bootstrap=100,
        )

        b = paired_bootstrap_indices(
            0.150, 23.5,
            n_realizations=500,
            n_bootstrap=100,
        )

        np.testing.assert_array_equal(a, b)

        self.assertEqual(a.shape, (100, 500))

        self.assertTrue(np.all(a >= 0))
        self.assertTrue(np.all(a < 500))

    def test_bootstrap_seed_changes_by_cell(self):
        self.assertNotEqual(
            stage3c_bootstrap_seed(0.150, 23.5),
            stage3c_bootstrap_seed(0.175, 23.5),
        )

        self.assertNotEqual(
            stage3c_bootstrap_seed(0.150, 23.5),
            stage3c_bootstrap_seed(0.150, 24.5),
        )

    def test_asymmetric_rejection(self):
        des = np.array([10., 20., 30., 40.])
        sdss = np.array([1., 2., 3., 4.])

        des_mask = np.array([True, True, False, True])
        sdss_mask = np.array([True, False, True, True])

        indices = np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ])

        result = bootstrap_median_differences(
            des,
            des_mask,
            sdss,
            sdss_mask,
            indices,
        )

        expected = (
            np.median(des[des_mask])
            - np.median(sdss[sdss_mask])
        )

        self.assertEqual(result.shape, (2,))
        np.testing.assert_allclose(
            result,
            np.array([expected, expected]),
        )

    def test_masked_median(self):
        values = np.array([1., 100., 3.])
        accepted = np.array([True, False, True])

        self.assertEqual(
            masked_median(values, accepted), 2.0
        )

    def test_pointwise_interval(self):
        replicates = np.arange(100., dtype=float)

        interval = pointwise_bootstrap_interval(
            replicates
        )

        self.assertIsNotNone(interval)

        self.assertAlmostEqual(
            interval[0],
            np.quantile(replicates, 0.025),
        )

        self.assertAlmostEqual(
            interval[1],
            np.quantile(replicates, 0.975),
        )

    def test_undefined_pointwise_interval(self):
        replicates = np.array([1., np.nan, 3.])

        self.assertIsNone(
            pointwise_bootstrap_interval(replicates)
        )

    def test_simultaneous_band(self):
        theta = np.array([1., 2., 3.])

        boot = np.array([
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9],
            [1.2, 2.2, 3.2],
        ])

        band = simultaneous_redshift_band(
            theta, boot
        )

        self.assertIsNotNone(band)

    def test_undefined_simultaneous_band(self):
        theta = np.array([1., 2.])

        boot = np.array([
            [1.1, 2.1],
            [np.nan, 2.0],
        ])

        self.assertIsNone(
            simultaneous_redshift_band(theta, boot)
        )


class TestPhysicalTiming(unittest.TestCase):

    @staticmethod
    def artificial_model():
        wave = np.array([4000., 5000., 6000.])

        hsiao = {}

        for phase in range(-20, 86):
            amplitude = max(
                100.0 - (phase - 5) ** 2,
                0.0,
            )

            hsiao[phase] = (
                wave,
                np.full(3, amplitude),
            )

        response = (
            wave,
            np.array([0., 1., 0.]),
        )

        return hsiao, response

    @staticmethod
    def artificial_proxy(
        wave, flux, response_wave, response, redshift
    ):
        return float(np.max(flux))

    def test_known_peak_phase(self):
        hsiao, response = self.artificial_model()

        phase = hsiao_r_peak_phase(
            hsiao,
            response,
            0.15,
            self.artificial_proxy,
        )

        self.assertEqual(phase, 5)

    def test_earliest_peak_tie(self):
        hsiao, response = self.artificial_model()

        wave = hsiao[5][0]

        hsiao[6] = (
            wave,
            np.full(3, 100.0),
        )

        phase = hsiao_r_peak_phase(
            hsiao,
            response,
            0.15,
            self.artificial_proxy,
        )

        self.assertEqual(phase, 5)

    def test_zero_timing_error(self):
        error = r_peak_timing_error(
            first_active_mjd=58990.,
            recovered_r_time_of_peak=16.,
            peak_mjd=59000.,
            redshift=0.2,
            physical_peak_phase=5,
        )

        self.assertAlmostEqual(error, 0.0)

    def test_positive_timing_error(self):
        error = r_peak_timing_error(
            first_active_mjd=58990.,
            recovered_r_time_of_peak=19.,
            peak_mjd=59000.,
            redshift=0.2,
            physical_peak_phase=5,
        )

        self.assertAlmostEqual(error, 3.0)

    def test_invalid_redshift(self):
        hsiao, response = self.artificial_model()

        with self.assertRaises(ValueError):
            hsiao_r_peak_phase(
                hsiao,
                response,
                -0.1,
                self.artificial_proxy,
            )


if __name__ == "__main__":
    unittest.main()
