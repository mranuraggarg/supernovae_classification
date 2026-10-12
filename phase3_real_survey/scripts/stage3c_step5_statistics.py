"""Stage 3C Step 5 statistical core.

Implements descriptive statistics from the frozen
Stage 3C statistical-analysis clarification.

No production data are loaded by this module.
"""

import numpy as np
from scipy.stats import wasserstein_distance


def validated_sample(values):
    """Return a nonempty, finite, one-dimensional sample."""
    x = np.asarray(values, dtype=float)

    if x.ndim != 1:
        raise ValueError("Sample must be one-dimensional.")

    if x.size == 0:
        raise ValueError("Sample must not be empty.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Sample contains nonfinite values.")

    return x


def median_difference(des, sdss):
    """Difference of marginal medians: DES minus SDSS."""
    d = validated_sample(des)
    s = validated_sample(sdss)
    return float(np.median(d) - np.median(s))


def quantile_difference(des, sdss, q):
    """Difference of marginal quantiles: DES minus SDSS."""
    if not 0 <= q <= 1:
        raise ValueError("Quantile must lie between 0 and 1.")

    d = validated_sample(des)
    s = validated_sample(sdss)

    return float(np.quantile(d, q) - np.quantile(s, q))


def median_absolute_deviation(values):
    """Unscaled median absolute deviation about the median."""
    x = validated_sample(values)
    return float(np.median(np.abs(x - np.median(x))))


def interquartile_range(values):
    """75th percentile minus 25th percentile."""
    x = validated_sample(values)
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))


def wasserstein_one(des, sdss):
    """Uncorrected first Wasserstein distance."""
    d = validated_sample(des)
    s = validated_sample(sdss)
    return float(wasserstein_distance(d, s))


def stage3c_bootstrap_seed(redshift, sbmag):
    """Frozen SHA-256-derived seed for one redshift/SBMAG cell."""
    import hashlib

    seed_string = (
        f"stage3c-step5-bootstrap-v1|20261006|"
        f"z={redshift:.3f}|sbmag={sbmag:.1f}"
    )
    digest = hashlib.sha256(seed_string.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def paired_bootstrap_indices(redshift, sbmag, n_realizations=500,
                             n_bootstrap=5000):
    """Generate shared realization-ID bootstrap indices.

    Each row contains one resample of the original realization IDs.
    The same index matrix must be used for every survey branch.
    """
    if n_realizations <= 0 or n_bootstrap <= 0:
        raise ValueError("Bootstrap dimensions must be positive.")

    seed = stage3c_bootstrap_seed(redshift, sbmag)
    rng = np.random.Generator(np.random.PCG64(seed))

    return rng.integers(
        0,
        n_realizations,
        size=(n_bootstrap, n_realizations),
    )


def masked_median(values, accepted):
    """Median of accepted values; rejected values contribute no feature."""
    x = np.asarray(values, dtype=float)
    mask = np.asarray(accepted, dtype=bool)

    if x.ndim != 1 or mask.ndim != 1 or x.shape != mask.shape:
        raise ValueError("Values and acceptance masks must be aligned.")

    selected = x[mask]

    if selected.size == 0:
        raise ValueError("No accepted realizations.")

    return float(np.median(validated_sample(selected)))


def bootstrap_median_differences(
    des_values,
    des_accepted,
    sdss_values,
    sdss_accepted,
    indices,
):
    """Paired-bootstrap distribution of marginal median differences.

    Rejected realizations remain in the resampling frame.
    Each survey's acceptance mask is applied after resampling.
    """
    d = np.asarray(des_values, dtype=float)
    s = np.asarray(sdss_values, dtype=float)
    dm = np.asarray(des_accepted, dtype=bool)
    sm = np.asarray(sdss_accepted, dtype=bool)
    idx = np.asarray(indices)

    if d.ndim != 1 or s.shape != d.shape:
        raise ValueError("Survey arrays must be aligned 1D arrays.")

    if dm.shape != d.shape or sm.shape != d.shape:
        raise ValueError("Acceptance masks must match survey arrays.")

    if idx.ndim != 2 or idx.shape[1] != d.size:
        raise ValueError("Invalid bootstrap index matrix.")

    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("Bootstrap indices must be integers.")

    if np.any(idx < 0) or np.any(idx >= d.size):
        raise ValueError("Bootstrap index out of range.")

    # Only accepted feature values must be finite.
    validated_sample(d[dm])
    validated_sample(s[sm])

    result = np.empty(idx.shape[0], dtype=float)

    for k, draw in enumerate(idx):
        d_selected = d[draw][dm[draw]]
        s_selected = s[draw][sm[draw]]

        if d_selected.size == 0 or s_selected.size == 0:
            result[k] = np.nan
            continue

        result[k] = np.median(d_selected) - np.median(s_selected)

    return result


def pointwise_bootstrap_interval(replicates, confidence=0.95):
    """Percentile bootstrap interval; undefined replicates are not dropped."""
    x = np.asarray(replicates, dtype=float)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("Replicates must be a nonempty 1D array.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between 0 and 1.")

    if not np.all(np.isfinite(x)):
        return None

    alpha = 1.0 - confidence

    return (
        float(np.quantile(x, alpha / 2.0)),
        float(np.quantile(x, 1.0 - alpha / 2.0)),
    )


def simultaneous_redshift_band(point_estimates, bootstrap_matrix,
                               confidence=0.95):
    """Unstandardized max-absolute-deviation simultaneous confidence band.

    bootstrap_matrix shape: (n_bootstrap, n_redshifts).
    Each row represents one replicate across the complete redshift grid.
    """
    theta = np.asarray(point_estimates, dtype=float)
    boot = np.asarray(bootstrap_matrix, dtype=float)

    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("Point estimates must be nonempty and 1D.")

    if boot.ndim != 2 or boot.shape[1] != theta.size:
        raise ValueError("Bootstrap matrix has incompatible dimensions.")

    if boot.shape[0] == 0:
        raise ValueError("Bootstrap matrix must contain replicates.")

    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between 0 and 1.")

    if not np.all(np.isfinite(theta)):
        return None

    if not np.all(np.isfinite(boot)):
        return None

    maximum_deviations = np.max(
        np.abs(boot - theta[np.newaxis, :]),
        axis=1,
    )

    critical_value = float(np.quantile(maximum_deviations, confidence))

    lower = theta - critical_value
    upper = theta + critical_value

    return {
        "critical_value": critical_value,
        "lower": lower,
        "upper": upper,
    }
