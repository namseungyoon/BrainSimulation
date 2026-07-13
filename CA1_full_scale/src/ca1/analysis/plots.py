"""Matplotlib visualisation helpers for CA1 simulation results.

All functions are Agg-safe: they never call ``plt.show()`` and always accept an
optional ``ax`` argument so callers can compose multi-panel figures.  Import
sets the backend to ``Agg`` at module load time to avoid display-server
dependency in headless environments (the backend is only overridden if it has
not already been set by the calling script).

Typical usage
-------------
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> from ca1.analysis.plots import raster, psd_plot, phase_plot
>>> fig, axes = plt.subplots(1, 3, figsize=(15, 4))
>>> raster(result, ax=axes[0])
>>> psd_plot(freqs, power, ax=axes[1])
>>> phase_plot({"pyramidal": (195.0, 0.42, 0.001)}, ax=axes[2])
>>> fig.savefig("summary.png", dpi=150)
"""

from __future__ import annotations

from typing import Optional

import matplotlib
# Switch to Agg only if no interactive backend has been requested yet.
if matplotlib.get_backend().lower() in ("", "agg", "module://matplotlib_inline.backend_inline"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from ca1.types import SimResult

# Fixed colour cycle for reproducible per-type colouring
_TYPE_COLORS = list(mcolors.TABLEAU_COLORS.values())


def _get_color(cell_type: str, palette: dict[str, str]) -> str:
    """Return a consistent colour for a cell-type name."""
    if cell_type not in palette:
        palette[cell_type] = _TYPE_COLORS[len(palette) % len(_TYPE_COLORS)]
    return palette[cell_type]


# ---------------------------------------------------------------------------
# raster plot
# ---------------------------------------------------------------------------

def raster(
    result: SimResult,
    ax: Optional[plt.Axes] = None,
    max_cells_per_type: int = 200,
    dot_size: float = 0.5,
    alpha: float = 0.6,
) -> plt.Axes:
    """Spike raster plot, one row band per cell type.

    Parameters
    ----------
    result:
        SimResult; uses ``result.spikes`` and ``result.meta``.
    ax:
        Matplotlib axes to draw into.  Created if None.
    max_cells_per_type:
        Sub-sample to at most this many cells per type to keep the figure fast.
    dot_size:
        Marker size in points.
    alpha:
        Marker transparency.

    Returns
    -------
    ax with the raster drawn.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    palette: dict[str, str] = {}
    y_offset = 0
    y_ticks: list[float] = []
    y_labels: list[str] = []

    for cell_type, cell_trains in sorted(result.spikes.items()):
        n_cells = len(cell_trains)
        if n_cells == 0:
            continue
        color = _get_color(cell_type, palette)
        # Sub-sample cells deterministically
        indices = np.arange(n_cells)
        if n_cells > max_cells_per_type:
            rng = np.random.default_rng(seed=abs(hash(cell_type)) % (2 ** 31))
            indices = rng.choice(n_cells, size=max_cells_per_type, replace=False)
            indices.sort()

        band_start = y_offset
        for local_i, cell_idx in enumerate(indices):
            train = cell_trains[cell_idx]
            if len(train) == 0:
                continue
            y_vals = np.full(len(train), y_offset + local_i)
            ax.scatter(train, y_vals, s=dot_size, c=color, alpha=alpha,
                       linewidths=0, rasterized=True)

        band_end = y_offset + len(indices)
        y_ticks.append(0.5 * (band_start + band_end))
        y_labels.append(f"{cell_type} ({n_cells})")
        y_offset = band_end + max(1, int(0.05 * len(indices)))  # small gap

    duration_s = result.meta.duration_s - result.meta.crop_first_ms * 1e-3
    ax.set_xlim(0.0, max(duration_s, 0.001))
    ax.set_ylim(-1, y_offset + 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cell type (N)")
    ax.set_title(f"Spike raster  scale={result.meta.scale:.3g}  "
                 f"backend={result.meta.backend}")
    return ax


# ---------------------------------------------------------------------------
# PSD plot
# ---------------------------------------------------------------------------

def psd_plot(
    freqs: np.ndarray,
    power: np.ndarray,
    ax: Optional[plt.Axes] = None,
    theta_band: tuple[float, float] = (5.0, 10.0),
    gamma_band: tuple[float, float] = (30.0, 80.0),
    f_max: float = 120.0,
    log_scale: bool = True,
) -> plt.Axes:
    """Power spectral density with theta/gamma band shading.

    Parameters
    ----------
    freqs, power:
        Outputs of ``ca1.analysis.spectral.welch_psd``.
    ax:
        Target axes; created if None.
    theta_band, gamma_band:
        Frequency bands to shade.
    f_max:
        Upper frequency limit for display.
    log_scale:
        If True, use log-scale on y-axis.

    Returns
    -------
    ax with the PSD drawn.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    mask = freqs <= f_max
    f_plot = freqs[mask]
    p_plot = power[mask]

    ax.plot(f_plot, p_plot, color="steelblue", linewidth=1.2)
    ax.axvspan(*theta_band, alpha=0.15, color="orange", label="Theta (5-10 Hz)")
    ax.axvspan(*gamma_band, alpha=0.10, color="green", label="Gamma (30-80 Hz)")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title("LFP Proxy Power Spectral Density")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, f_max)
    return ax


# ---------------------------------------------------------------------------
# phase preference polar plot
# ---------------------------------------------------------------------------

def phase_plot(
    phase_by_type: dict[str, tuple[float, float, float]],
    ax: Optional[plt.Axes] = None,
    band_label: str = "theta",
) -> plt.Axes:
    """Polar plot of mean spike-phase preference per cell type.

    Parameters
    ----------
    phase_by_type:
        Mapping from cell-type name to
        ``(mean_phase_deg, vector_strength, rayleigh_p)``
        as returned by ``ca1.analysis.spectral.phase_preference``.
    ax:
        Polar axes; created if None.  **Must be a polar axes.**
    band_label:
        String label for the oscillation band (used in title).

    Returns
    -------
    ax with arrows drawn from origin to each cell-type's mean phase vector.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")

    palette: dict[str, str] = {}
    for cell_type, (mean_deg, vs, p_val) in sorted(phase_by_type.items()):
        if np.isnan(mean_deg):
            continue
        color = _get_color(cell_type, palette)
        # Convert to radians; 0 deg = trough -> right (+x) on the polar plot
        theta_rad = np.deg2rad(mean_deg)
        ax.annotate(
            "",
            xy=(theta_rad, vs),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        )
        ax.plot(theta_rad, vs, "o", color=color, markersize=6,
                label=f"{cell_type} {mean_deg:.0f}° R={vs:.2f}")

    ax.set_theta_zero_location("E")   # 0 deg = rightward = trough
    ax.set_theta_direction(1)         # counter-clockwise = increasing phase
    ax.set_rlim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(f"Phase preference ({band_label})\n0° = trough", pad=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.05), fontsize=7)
    return ax
