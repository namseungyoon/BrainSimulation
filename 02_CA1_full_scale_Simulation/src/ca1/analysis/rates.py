"""Population firing-rate statistics.

All functions accept ``spikes: dict[str, list[np.ndarray]]`` where each value
is a list of per-cell spike-time arrays in **seconds**, already cropped for the
startup transient (done upstream in SimulatorBackend.simulate).

References
----------
* Golomb & Rinzel (1994) -- chi synchrony statistic.
* Shinomoto et al. (2009) -- CV_ISI as irregularity measure (AI regime ~1).
"""

from __future__ import annotations

import numpy as np

Spikes = dict[str, list[np.ndarray]]


# ---------------------------------------------------------------------------
# mean firing rates
# ---------------------------------------------------------------------------

def mean_rates(
    spikes: Spikes,
    duration_s: float,
    active_only: bool = False,
) -> dict[str, float]:
    """Mean firing rate per cell type in Hz.

    Parameters
    ----------
    spikes:
        Per-type list of per-cell spike-time arrays (seconds, already cropped).
    duration_s:
        Analysis window length in seconds (denominator for rate calculation).
        Must be the *actual* elapsed time after cropping -- not the total
        simulation duration.  Bug #3 in RECOVERY_PLAN: dividing by nominal
        duration inflates rates ~5x.
    active_only:
        If True, exclude silent cells (zero spikes) from the per-type average
        so the rate reflects the active sub-population only.

    Returns
    -------
    dict mapping cell-type name -> mean rate in Hz.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")

    rates: dict[str, float] = {}
    for cell_type, cell_trains in spikes.items():
        if not cell_trains:
            rates[cell_type] = 0.0
            continue
        counts = np.array([float(len(t)) for t in cell_trains])
        if active_only:
            active = counts[counts > 0]
            if active.size == 0:
                rates[cell_type] = 0.0
            else:
                rates[cell_type] = float(active.mean()) / duration_s
        else:
            rates[cell_type] = float(counts.mean()) / duration_s
    return rates


# ---------------------------------------------------------------------------
# irregularity: coefficient of variation of ISI
# ---------------------------------------------------------------------------

def cv_isi(spikes: Spikes) -> dict[str, float]:
    """Per-type mean CV of inter-spike intervals.

    A Poisson / AI (asynchronous-irregular) regime has CV ~ 1.  Regular
    oscillators have CV < 1; bursters CV > 1.  Only cells with >= 2 spikes
    contribute; types with no qualifying cells return NaN.

    Returns
    -------
    dict mapping cell-type name -> mean CV_ISI (nan if insufficient data).
    """
    result: dict[str, float] = {}
    for cell_type, cell_trains in spikes.items():
        cvs: list[float] = []
        for train in cell_trains:
            if len(train) < 2:
                continue
            isis = np.diff(np.sort(train))
            mu = isis.mean()
            if mu == 0.0:
                continue
            cvs.append(float(isis.std() / mu))
        result[cell_type] = float(np.mean(cvs)) if cvs else float("nan")
    return result


# ---------------------------------------------------------------------------
# Fano factor (spike-count variability)
# ---------------------------------------------------------------------------

def fano_factor(
    spikes: Spikes,
    duration_s: float,
    bin_s: float = 0.01,
) -> dict[str, float]:
    """Fano factor of spike counts per cell type.

    Computed by binning each cell's spike train into windows of ``bin_s``
    seconds, then returning ``var(counts) / mean(counts)`` averaged over
    cells.  Cells with mean count == 0 in all bins are excluded.

    Parameters
    ----------
    spikes:
        Per-type per-cell spike-time arrays (s).
    duration_s:
        Analysis window duration (s).
    bin_s:
        Bin width in seconds.

    Returns
    -------
    dict mapping cell-type name -> Fano factor (nan if insufficient data).
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if bin_s <= 0:
        raise ValueError(f"bin_s must be positive, got {bin_s}")

    n_bins = max(1, int(round(duration_s / bin_s)))
    edges = np.linspace(0.0, duration_s, n_bins + 1)

    result: dict[str, float] = {}
    for cell_type, cell_trains in spikes.items():
        fanos: list[float] = []
        for train in cell_trains:
            counts, _ = np.histogram(train, bins=edges)
            mu = counts.mean()
            if mu == 0.0:
                continue
            fanos.append(float(counts.var() / mu))
        result[cell_type] = float(np.mean(fanos)) if fanos else float("nan")
    return result


# ---------------------------------------------------------------------------
# population synchrony: Golomb-Rinzel chi^2 statistic
# ---------------------------------------------------------------------------

def population_synchrony_chi(
    spikes: Spikes,
    duration_s: float,
    dt_s: float = 0.001,
) -> float:
    """Golomb-Rinzel chi^2 population synchrony over all cell types pooled.

    Chi^2 = Var(population_mean_activity) / mean(per_cell_variance).

    The population mean activity is the mean spike-count per time bin averaged
    over all cells; chi^2 > 1 indicates synchrony, chi^2 ~ 0 indicates
    asynchrony.

    Parameters
    ----------
    spikes:
        All cell types; cells are pooled across types.
    duration_s:
        Analysis window duration (s).
    dt_s:
        Bin width in seconds.

    Returns
    -------
    chi^2 scalar (float).  Returns 0.0 if fewer than 2 cells.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")

    n_bins = max(1, int(round(duration_s / dt_s)))
    edges = np.linspace(0.0, duration_s, n_bins + 1)

    all_counts: list[np.ndarray] = []
    for cell_trains in spikes.values():
        for train in cell_trains:
            counts, _ = np.histogram(train, bins=edges)
            all_counts.append(counts.astype(float))

    n_cells = len(all_counts)
    if n_cells < 2:
        return 0.0

    mat = np.stack(all_counts, axis=0)         # (n_cells, n_bins)
    pop_mean = mat.mean(axis=0)                # mean across cells per bin
    var_pop = float(pop_mean.var())            # var of population mean
    mean_var_cell = float(mat.var(axis=1).mean())  # mean of per-cell variances
    if mean_var_cell == 0.0:
        return 0.0
    return var_pop / mean_var_cell
