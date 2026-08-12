"""Spectral analysis for the CA1 theta/gamma network.

All functions operate on SI units (seconds, Hz) unless noted.

Key invariants
--------------
* Theta band: 5-10 Hz  (Bezaire 2016 target ~7.8 Hz)
* Gamma band: 25-80 Hz (Bezaire 2016 target ~71 Hz)
* 0-degree phase convention: LFP trough (most negative amplitude).
* Inhibitory currents dominate the LFP proxy at the pyramidal soma; the SDF of
  pyramidal cells is a reasonable proxy when a real LFP is unavailable.
* Tort modulation index: Tort et al. (2010) PAC measure.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import scipy.signal as ss

from ca1.types import SimResult
from ca1.validation.targets import (
    CFC_MIN_WINDOW_S,
    CFC_MIN_Z_SCORE as _CFC_MIN_Z_SCORE,
    CFC_N_SURROGATES,
    GAMMA_BAND,
    SPECTRAL_PEAK_MIN_PROMINENCE_RATIO,
    THETA_BAND,
)

Spikes = dict[str, list[np.ndarray]]

# Pyramidal cell-type name used as LFP fallback (matches connectivity.json key)
_PYRAMIDAL_KEY = "pyramidal"

# Public aliases keep the analysis thresholds visible beside their outputs.
PEAK_PROMINENCE_MIN_RATIO = SPECTRAL_PEAK_MIN_PROMINENCE_RATIO
CFC_MIN_Z_SCORE = _CFC_MIN_Z_SCORE


# ---------------------------------------------------------------------------
# Spike-density function (SDF)
# ---------------------------------------------------------------------------

def spike_density(
    spikes: Spikes,
    duration_s: float,
    dt_s: float = 0.001,
    sigma_ms: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-kernel spike density function pooled over all cells.

    Pooling is population-level: every cell contributes equally, so the SDF
    represents the mean instantaneous rate across the population (Hz).

    Parameters
    ----------
    spikes:
        Per-type per-cell spike-time arrays (s).
    duration_s:
        Analysis window duration (s).
    dt_s:
        Time-step for the output grid (s).
    sigma_ms:
        Gaussian kernel width (ms).

    Returns
    -------
    t: np.ndarray  -- time axis (s), length N
    sdf: np.ndarray -- spike density (Hz), length N
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if sigma_ms <= 0:
        raise ValueError(f"sigma_ms must be positive, got {sigma_ms}")

    n_bins = int(round(duration_s / dt_s))
    t = np.arange(n_bins) * dt_s + dt_s * 0.5  # bin centres

    # Collect all spike times from all cells / types
    all_trains: list[np.ndarray] = []
    n_cells = 0
    for cell_trains in spikes.values():
        for train in cell_trains:
            all_trains.append(train)
            n_cells += 1

    if n_cells == 0 or not all_trains:
        return t, np.zeros(n_bins)

    # Bin all spikes into a single histogram (total spike count per bin)
    edges = np.linspace(0.0, duration_s, n_bins + 1)
    pooled, _ = np.histogram(np.concatenate(all_trains), bins=edges)
    pooled = pooled.astype(float)

    # Normalise to per-cell rate (Hz): divide by n_cells and by dt_s
    rate = pooled / (n_cells * dt_s)

    # Gaussian smoothing: sigma in samples
    sigma_s = (sigma_ms * 1e-3) / dt_s
    # Use a truncated Gaussian kernel of half-width 4*sigma
    half_w = int(math.ceil(4.0 * sigma_s))
    k_x = np.arange(-half_w, half_w + 1, dtype=float)
    kernel = np.exp(-0.5 * (k_x / sigma_s) ** 2)
    kernel /= kernel.sum()
    sdf = np.convolve(rate, kernel, mode="same")

    return t, sdf


# ---------------------------------------------------------------------------
# LFP proxy
# ---------------------------------------------------------------------------

def lfp_proxy(result: SimResult) -> Tuple[np.ndarray, float]:
    """Return a LFP proxy time series and its sample rate (Hz).

    Strategy (in order of preference):
    1. Use ``result.lfp`` if present (real LFP from backend).
    2. Compute the pyramidal-cell SDF and return it as the LFP analog.
       Pyramidal inhibitory inputs dominate the extracellular field in CA1.

    Returns
    -------
    lfp: np.ndarray  -- LFP proxy time series
    fs: float        -- sample rate in Hz
    """
    if result.lfp is not None and result.lfp_dt_s is not None:
        return result.lfp, 1.0 / result.lfp_dt_s

    # Fallback: pyramidal SDF
    duration_s = result.meta.duration_s - result.meta.crop_first_ms * 1e-3
    dt_s = result.meta.dt_s if result.meta.dt_s > 0 else 0.001

    pyr_spikes: Spikes = {}
    for key in result.spikes:
        if _PYRAMIDAL_KEY in key.lower():
            pyr_spikes[key] = result.spikes[key]
    if not pyr_spikes:
        raise ValueError(
            "Pyramidal spikes are required for pyramidal_spike_density LFP proxy"
        )

    _, sdf = spike_density(pyr_spikes, duration_s=duration_s, dt_s=dt_s)
    return sdf, 1.0 / dt_s


# ---------------------------------------------------------------------------
# Power spectral density (Welch)
# ---------------------------------------------------------------------------

def welch_psd(
    sig: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD with sensible defaults for LFP-length signals.

    nperseg is chosen as the smallest power-of-two that gives frequency
    resolution <= 0.5 Hz (i.e., >= 2 s segments), capped at half the signal.

    Returns
    -------
    freqs: np.ndarray  -- frequency axis (Hz)
    power: np.ndarray  -- one-sided PSD
    """
    n = len(sig)
    if n < 4:
        raise ValueError(f"Signal too short for Welch PSD: {n} samples")

    # Target resolution: 0.5 Hz -> need nperseg >= fs/0.5 = 2*fs
    target_nperseg = int(2.0 * fs)
    # Round up to power of two for FFT efficiency
    p = 1
    while p < target_nperseg:
        p <<= 1
    nperseg = min(p, n // 2)
    nperseg = max(nperseg, 4)  # guarantee enough samples

    freqs, power = ss.welch(sig, fs=fs, nperseg=nperseg, scaling="density")
    return freqs, power


# ---------------------------------------------------------------------------
# Band power + peak frequency
# ---------------------------------------------------------------------------

def band_power_peak(
    freqs: np.ndarray,
    power: np.ndarray,
    band: tuple[float, float],
    *,
    return_prominence: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, float, bool]:
    """Dominant frequency, peak power, and integrated band power.

    Parameters
    ----------
    freqs, power:
        Outputs of ``welch_psd``.
    band:
        (low_hz, high_hz) frequency band.

    Returns
    -------
    By default the historical three-value tuple is returned.  With
    ``return_prominence=True``, two audit fields are appended:
    ``peak/background`` from a robust log-log aperiodic fit, and whether the
    argmax is the band's lowest resolvable bin.  Acceptance requires a ratio
    of at least ``PEAK_PROMINENCE_MIN_RATIO`` and rejects that lower edge.
    """
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        base = (float("nan"), float("nan"), 0.0)
        return (*base, float("nan"), False) if return_prominence else base
    f_band = freqs[mask]
    p_band = power[mask]
    idx_peak = int(np.argmax(p_band))
    peak_freq = float(f_band[idx_peak])
    peak_power = float(p_band[idx_peak])
    band_power = float(np.trapz(p_band, f_band))
    if not return_prominence:
        return peak_freq, peak_power, band_power

    is_low_edge = idx_peak == 0
    background = _aperiodic_background_at_peak(
        freqs, power, band=band, peak_freq=peak_freq
    )
    prominence = (
        peak_power / background
        if np.isfinite(background) and background > 0.0
        else float("nan")
    )
    return peak_freq, peak_power, band_power, float(prominence), is_low_edge


def _aperiodic_background_at_peak(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    band: tuple[float, float],
    peak_freq: float,
) -> float:
    """Estimate local 1/f power with an iteratively robust log-log fit."""
    lo, hi = band
    positive_freqs = freqs[freqs > 0.0]
    if positive_freqs.size == 0:
        return float("nan")
    context = (
        (freqs >= max(float(positive_freqs[0]), lo * 0.5))
        & (freqs <= min(float(freqs[-1]), hi * 1.5))
        & (freqs > 0.0)
        & np.isfinite(power)
        & (power > 0.0)
    )
    if freqs.size > 1:
        resolution = float(np.median(np.diff(freqs)))
    else:
        resolution = 0.0
    # Remove the candidate and its leakage skirt before estimating the floor.
    context &= np.abs(freqs - peak_freq) > max(1.0, 3.0 * resolution)
    x = np.log(freqs[context])
    y = np.log(power[context])
    if x.size < 3:
        return float("nan")

    keep = np.ones(x.size, dtype=bool)
    coeff = np.polyfit(x, y, 1)
    for _ in range(3):
        coeff = np.polyfit(x[keep], y[keep], 1)
        residual = y - np.polyval(coeff, x)
        centre = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - centre)))
        if mad <= np.finfo(float).eps:
            break
        # Suppress narrow positive peaks while retaining the aperiodic floor.
        updated = residual <= centre + 2.5 * 1.4826 * mad
        if np.count_nonzero(updated) < 3 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(np.exp(np.polyval(coeff, math.log(peak_freq))))


# ---------------------------------------------------------------------------
# Phase preference (spike-triggered LFP phase)
# ---------------------------------------------------------------------------

def phase_preference(
    spike_times: np.ndarray,
    lfp: np.ndarray,
    fs: float,
    band: tuple[float, float] = (5, 10),
) -> Tuple[float, float, float]:
    """Mean phase, vector strength, and Rayleigh p-value of spike-triggered LFP phase.

    Convention: 0 degrees = LFP trough (most negative amplitude), consistent
    with Klausberger et al. / Bezaire et al. phase reporting.

    Parameters
    ----------
    spike_times:
        Spike times in seconds relative to the analysis window start.
    lfp:
        LFP proxy time series (length = len(lfp), sampled at ``fs`` Hz).
    fs:
        Sample rate of ``lfp`` in Hz.
    band:
        Band-pass filter band in Hz (default theta 5-10 Hz).

    Returns
    -------
    mean_phase_deg: float      -- circular mean phase in degrees [0, 360)
    vector_strength: float     -- resultant vector length R in [0, 1]
    rayleigh_p: float          -- Rayleigh test p-value (small = non-uniform)
    """
    if len(spike_times) == 0:
        return float("nan"), 0.0, 1.0
    if len(lfp) < 4:
        return float("nan"), 0.0, 1.0

    # Band-pass filter
    lo, hi = band
    nyq = fs / 2.0
    lo_norm = lo / nyq
    hi_norm = min(hi / nyq, 0.999)
    sos = ss.butter(4, [lo_norm, hi_norm], btype="bandpass", output="sos")
    filtered = ss.sosfiltfilt(sos, lfp)

    # Analytic signal -> instantaneous phase
    analytic = ss.hilbert(filtered)
    inst_phase = np.angle(analytic)  # radians, [-pi, pi]

    # Invert: convention 0 = trough means we add pi to the analytic phase
    # (analytic phase 0 = peak; trough = pi or -pi)
    inst_phase_trough = inst_phase + math.pi  # now 0 = trough

    # Sample LFP phase at each spike time
    spike_idx = np.round(spike_times * fs).astype(int)
    valid = (spike_idx >= 0) & (spike_idx < len(inst_phase_trough))
    if not np.any(valid):
        return float("nan"), 0.0, 1.0

    phases = inst_phase_trough[spike_idx[valid]]  # radians, 0=trough convention

    # Circular statistics
    sin_mean = float(np.sin(phases).mean())
    cos_mean = float(np.cos(phases).mean())
    vector_strength = float(math.sqrt(sin_mean ** 2 + cos_mean ** 2))
    mean_phase_rad = math.atan2(sin_mean, cos_mean)
    mean_phase_deg = float(math.degrees(mean_phase_rad) % 360.0)

    # Rayleigh test: z = n * R^2, p approximated
    n = int(np.sum(valid))
    z = n * vector_strength ** 2
    # Rayleigh p-value approximation (Zar 1999)
    rayleigh_p = float(math.exp(-z) * (1.0 + (2.0 * z - z ** 2) / (4.0 * n)
                                        - (24.0 * z - 132.0 * z ** 2 + 76.0 * z ** 3 - 9.0 * z ** 4)
                                        / (288.0 * n ** 2)))
    rayleigh_p = max(0.0, min(1.0, rayleigh_p))

    return mean_phase_deg, vector_strength, rayleigh_p


# ---------------------------------------------------------------------------
# Theta-gamma cross-frequency coupling (Tort MI)
# ---------------------------------------------------------------------------

def theta_gamma_cfc(
    lfp: np.ndarray,
    fs: float,
    *,
    n_surrogates: int = CFC_N_SURROGATES,
    random_seed: int = 0,
) -> tuple[float, float, float]:
    """Tort modulation index for theta-phase / gamma-amplitude coupling.

    Reference: Tort et al. (2010) J Neurophysiol.

    The MI is computed by:
    1. Band-pass the LFP using the canonical theta and gamma target bands.
    2. Extract theta phase and gamma amplitude envelope.
    3. Bin the amplitude by theta phase (18 bins of 20 degrees each).
    4. MI = (KL divergence from uniform) / log(N_bins).

    Returns
    -------
    mi, surrogate_p, surrogate_z
        Raw modulation index, one-sided permutation p-value, and z-score
        against block-permuted gamma-amplitude surrogates.
    """
    if len(lfp) < int(math.ceil(CFC_MIN_WINDOW_S * fs)):
        return float("nan"), float("nan"), float("nan")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be >= 1")

    nyq = fs / 2.0

    def _bandpass(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
        lo_n = lo / nyq
        hi_n = min(hi / nyq, 0.999)
        sos = ss.butter(4, [lo_n, hi_n], btype="bandpass", output="sos")
        return ss.sosfiltfilt(sos, sig)

    theta_filt = _bandpass(lfp, *THETA_BAND)
    gamma_filt = _bandpass(lfp, *GAMMA_BAND)

    theta_phase = np.angle(ss.hilbert(theta_filt))         # [-pi, pi]
    gamma_amp = np.abs(ss.hilbert(gamma_filt))

    observed_mi = _tort_mi(theta_phase, gamma_amp)

    # Permuting short contiguous blocks preserves local envelope smoothness but
    # breaks its alignment to theta phase.  Quarter-cycle blocks avoid the MI
    # invariance that circularly shifting a periodic envelope would create.
    theta_centre_hz = 0.5 * (THETA_BAND[0] + THETA_BAND[1])
    block_size = max(1, int(round(fs / (4.0 * theta_centre_hz))))
    blocks = [gamma_amp[start:start + block_size]
              for start in range(0, gamma_amp.size, block_size)]
    rng = np.random.default_rng(random_seed)
    surrogate_mi = np.empty(n_surrogates, dtype=float)
    for idx in range(n_surrogates):
        order = rng.permutation(len(blocks))
        shuffled_amp = np.concatenate([blocks[j] for j in order])
        surrogate_mi[idx] = _tort_mi(theta_phase, shuffled_amp)

    p_value = float(
        (1 + np.count_nonzero(surrogate_mi >= observed_mi))
        / (n_surrogates + 1)
    )
    surrogate_std = float(np.std(surrogate_mi, ddof=1))
    if surrogate_std > 0.0:
        z_score = float((observed_mi - np.mean(surrogate_mi)) / surrogate_std)
    elif observed_mi > float(np.mean(surrogate_mi)):
        z_score = float("inf")
    else:
        z_score = 0.0
    return observed_mi, p_value, z_score


def _tort_mi(theta_phase: np.ndarray, gamma_amp: np.ndarray) -> float:
    """Compute Tort MI for already-extracted phase and amplitude series."""
    n_bins = 18
    bin_idx = np.floor((theta_phase + math.pi) * n_bins / (2.0 * math.pi)).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    totals = np.bincount(bin_idx, weights=gamma_amp, minlength=n_bins)
    amp_by_phase = np.divide(
        totals,
        counts,
        out=np.zeros(n_bins, dtype=float),
        where=counts > 0,
    )

    total = amp_by_phase.sum()
    if total == 0.0:
        return 0.0

    # Normalise to probability distribution
    p = amp_by_phase / total

    # KL divergence from uniform (log base e)
    uniform = 1.0 / n_bins
    # Clip to avoid log(0)
    p_safe = np.where(p > 0, p, 1e-300)
    kl = float(np.sum(p_safe * np.log(p_safe / uniform)))

    # Normalise to [0, 1]
    mi = kl / math.log(n_bins)
    return float(np.clip(mi, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Self-check (run with: python -m ca1.analysis.spectral)
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Verify band_power_peak recovers ~8 Hz from a synthetic modulated signal."""
    rng = np.random.default_rng(0)
    fs = 1000.0          # Hz
    duration = 10.0      # s
    t = np.arange(int(duration * fs)) / fs
    # 8 Hz modulation + noise
    sig = np.sin(2.0 * math.pi * 8.0 * t) + 0.1 * rng.standard_normal(len(t))
    freqs, power = welch_psd(sig, fs)
    peak_f, peak_p, bp = band_power_peak(freqs, power, band=(5.0, 12.0))
    assert 7.0 <= peak_f <= 9.0, f"Self-check FAILED: peak at {peak_f:.2f} Hz, expected ~8 Hz"
    print(f"[spectral self-check] PASS  peak={peak_f:.2f} Hz  band_power={bp:.4f}")


if __name__ == "__main__":
    _self_check()
