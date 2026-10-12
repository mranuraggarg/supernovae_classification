"""Stage 3C Step 5B: frozen physical timing-reference calculations.

No production Monte Carlo results are loaded.
"""

import numpy as np


PHASE_MIN = -20
PHASE_MAX = 85


def hsiao_r_peak_phase(
    hsiao,
    response,
    redshift,
    synthetic_flux_proxy,
):
    """Find the earliest maximum of the noiseless r-band flux.

    Parameters
    ----------
    hsiao : dict
        Mapping integer rest-frame phase to (wavelength, flux).
    response : tuple
        Observed-frame (wavelength, transmission) arrays.
    redshift : float
        Cosmological redshift.
    synthetic_flux_proxy : callable
        Frozen synthetic-photometry function.

    Returns
    -------
    int
        Rest-frame phase of maximum positive finite r-band flux.
    """
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("Redshift must be finite and nonnegative.")

    response_wave, response_trans = response

    phases = np.arange(PHASE_MIN, PHASE_MAX + 1)

    values = np.empty(phases.size, dtype=float)

    for index, phase in enumerate(phases):
        wave, flux = hsiao[int(phase)]

        values[index] = synthetic_flux_proxy(
            wave,
            flux,
            response_wave,
            response_trans,
            redshift,
        )

    valid = np.isfinite(values) & (values > 0)

    if not np.any(valid):
        raise ValueError("No finite positive synthetic r-band flux.")

    valid_phases = phases[valid]
    valid_values = values[valid]

    # np.argmax returns the first maximum, preserving the
    # frozen earliest-phase tie-breaking convention.
    return int(valid_phases[np.argmax(valid_values)])


def r_peak_timing_error(
    first_active_mjd,
    recovered_r_time_of_peak,
    peak_mjd,
    redshift,
    physical_peak_phase,
):
    """Recovered minus physical observer-frame r-band peak MJD."""
    parameters = np.asarray(
        [
            first_active_mjd,
            recovered_r_time_of_peak,
            peak_mjd,
            redshift,
            physical_peak_phase,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(parameters)):
        raise ValueError("Timing inputs must be finite.")

    if redshift < 0:
        raise ValueError("Redshift must be nonnegative.")

    recovered_peak_mjd = (
        first_active_mjd + recovered_r_time_of_peak
    )

    physical_peak_mjd = (
        peak_mjd + (1.0 + redshift) * physical_peak_phase
    )

    return float(recovered_peak_mjd - physical_peak_mjd)
