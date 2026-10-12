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


def bootstrap_frequency_counts(indices, n_realizations):
    """Convert bootstrap index rows to compact realization multiplicities."""
    idx = np.asarray(indices)

    if idx.ndim != 2:
        raise ValueError("Bootstrap indices must be two-dimensional.")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("Bootstrap indices must be integers.")
    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive.")
    if np.any(idx < 0) or np.any(idx >= n_realizations):
        raise ValueError("Bootstrap index out of range.")

    dtype = np.uint16 if idx.shape[1] <= np.iinfo(np.uint16).max else np.uint32
    counts = np.zeros((idx.shape[0], n_realizations), dtype=dtype)
    rows = np.arange(idx.shape[0])

    for column in range(idx.shape[1]):
        np.add.at(counts, (rows, idx[:, column]), 1)

    return counts


def bootstrap_weighted_quantile(values, available, counts, q):
    """Quantiles of bootstrap resamples represented by multiplicity rows.

    The interpolation is identical to NumPy's default linear quantile on the
    explicitly expanded resample, while unavailable realizations retain zero
    weight. A row with no available draw is returned as NaN.
    """
    if not 0 <= q <= 1:
        raise ValueError("Quantile must lie between 0 and 1.")

    x = np.asarray(values, dtype=float)
    mask = np.asarray(available, dtype=bool)
    weights = np.asarray(counts)

    if x.ndim != 1 or mask.shape != x.shape:
        raise ValueError("Values and availability mask must be aligned.")
    if weights.ndim != 2 or weights.shape[1] != x.size:
        raise ValueError("Bootstrap counts have incompatible dimensions.")
    if np.any(weights < 0):
        raise ValueError("Bootstrap counts must be nonnegative.")
    if np.any(mask & ~np.isfinite(x)):
        raise ValueError("Available values must be finite.")

    result = np.full(weights.shape[0], np.nan, dtype=float)
    available_indices = np.flatnonzero(mask)

    if available_indices.size == 0:
        return result

    order = available_indices[np.argsort(x[available_indices], kind="stable")]
    sorted_values = x[order]
    sorted_counts = weights[:, order].astype(np.int64, copy=False)
    cumulative = np.cumsum(sorted_counts, axis=1)
    totals = cumulative[:, -1]
    defined = totals > 0

    if not np.any(defined):
        return result

    positions = (totals[defined] - 1) * float(q)
    lower_rank = np.floor(positions).astype(np.int64)
    upper_rank = np.ceil(positions).astype(np.int64)
    fraction = positions - lower_rank
    selected_cumulative = cumulative[defined]

    lower_index = np.argmax(
        selected_cumulative > lower_rank[:, np.newaxis],
        axis=1,
    )
    upper_index = np.argmax(
        selected_cumulative > upper_rank[:, np.newaxis],
        axis=1,
    )

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    result[defined] = lower_value + fraction * (upper_value - lower_value)
    return result


def bootstrap_marginal_quantile_differences(
    des_values,
    des_available,
    sdss_values,
    sdss_available,
    counts,
    q=0.5,
):
    """Paired-ID bootstrap of a difference between marginal quantiles."""
    des = bootstrap_weighted_quantile(
        des_values, des_available, counts, q
    )
    sdss = bootstrap_weighted_quantile(
        sdss_values, sdss_available, counts, q
    )
    return des - sdss


def _bootstrap_timing_statistics_python(values, available, indices):
    """Bootstrap all frozen timing summaries for one survey cell.

    The explicit sampled array is used only for timing because MAD is centered
    on a replicate-specific median. At production dimensions this remains a
    bounded temporary array of 5000 x 500 values.
    """
    x = np.asarray(values, dtype=float)
    mask = np.asarray(available, dtype=bool)
    idx = np.asarray(indices)

    if x.ndim != 1 or mask.shape != x.shape:
        raise ValueError("Values and availability mask must be aligned.")
    if idx.ndim != 2 or idx.shape[1] != x.size:
        raise ValueError("Bootstrap indices have incompatible dimensions.")
    if np.any(mask & ~np.isfinite(x)):
        raise ValueError("Available timing errors must be finite.")

    sampled = x[idx].astype(float, copy=True)
    sampled[~mask[idx]] = np.nan
    defined = np.any(np.isfinite(sampled), axis=1)

    output = {
        name: np.full(idx.shape[0], np.nan, dtype=float)
        for name in (
            "signed_median_days",
            "median_absolute_error_days",
            "mad_days",
            "iqr_days",
            "q10_days",
            "q90_days",
        )
    }

    if not np.any(defined):
        return output

    active = sampled[defined]
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(active, axis=1)
        q10 = np.nanquantile(active, 0.10, axis=1)
        q25 = np.nanquantile(active, 0.25, axis=1)
        q75 = np.nanquantile(active, 0.75, axis=1)
        q90 = np.nanquantile(active, 0.90, axis=1)
        median_absolute = np.nanmedian(np.abs(active), axis=1)
        mad = np.nanmedian(
            np.abs(active - median[:, np.newaxis]),
            axis=1,
        )

    output["signed_median_days"][defined] = median
    output["median_absolute_error_days"][defined] = median_absolute
    output["mad_days"][defined] = mad
    output["iqr_days"][defined] = q75 - q25
    output["q10_days"][defined] = q10
    output["q90_days"][defined] = q90
    return output


def bootstrap_timing_statistics(values, available, indices, backend="python"):
    """Bootstrap frozen timing summaries with a validated backend.

    ``python`` is the readable reference implementation. ``native`` invokes
    the optional C++ kernel and fails if it cannot be built or loaded.
    """
    if backend not in {"python", "native"}:
        raise ValueError("backend must be 'python' or 'native'.")

    if backend == "python":
        return _bootstrap_timing_statistics_python(values, available, indices)

    from stage3c_step5_native import bootstrap_timing_statistics_native

    return bootstrap_timing_statistics_native(values, available, indices)
