"""Tests for ca1.analysis.spectral.

Skips gracefully if numpy / scipy are not installed.

Tests
-----
1. Synthetic spike ensemble modulated at 7.8 Hz -> spike_density -> welch_psd
   -> band_power_peak detects the peak within +/- 1 Hz.
2. phase_preference returns low Rayleigh p-value for a phase-locked spike train.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")

# Now safe to import the spectral module
spectral = pytest.importorskip("ca1.analysis.spectral")
types_mod = pytest.importorskip("ca1.types")


# ---------------------------------------------------------------------------
# Helpers: synthetic spike data
# ---------------------------------------------------------------------------

def _theta_modulated_spikes(
    n_cells: int = 200,
    duration_s: float = 10.0,
    theta_hz: float = 7.8,
    mean_rate_hz: float = 5.0,
    rng: "np.random.Generator | None" = None,
) -> dict[str, list["np.ndarray"]]:
    """Generate inhomogeneous Poisson spike trains modulated at theta_hz.

    The instantaneous firing rate is:
        lambda(t) = mean_rate_hz * (1 + cos(2*pi*theta_hz*t))

    This gives a spike density with a clear peak at theta_hz.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    dt = 0.001  # 1 ms bins
    t = np.arange(0, duration_s, dt)
    rate = mean_rate_hz * (1.0 + np.cos(2.0 * math.pi * theta_hz * t))
    prob = rate * dt  # approximate Bernoulli probability

    spikes: list[np.ndarray] = []
    for _ in range(n_cells):
        fired = t[rng.random(len(t)) < prob]
        spikes.append(fired)

    return {"pyramidalcell": spikes}


def _phase_locked_spikes(
    n_spikes: int = 200,
    theta_hz: float = 7.8,
    preferred_phase_deg: float = 90.0,
    jitter_deg: float = 20.0,
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Return spike times concentrated at preferred_phase_deg on theta cycle."""
    if rng is None:
        rng = np.random.default_rng(123)

    period_s = 1.0 / theta_hz
    phases = rng.normal(
        loc=math.radians(preferred_phase_deg),
        scale=math.radians(jitter_deg),
        size=n_spikes,
    )
    # Distribute cycles over 0..20 s
    cycle_idx = rng.integers(0, int(20.0 * theta_hz), size=n_spikes)
    spike_times = cycle_idx * period_s + (phases % (2.0 * math.pi)) / (2.0 * math.pi * theta_hz)
    return np.sort(spike_times)


def _fake_lfp_theta(
    duration_s: float = 10.0,
    fs: float = 1000.0,
    theta_hz: float = 7.8,
) -> np.ndarray:
    """Sinusoidal LFP proxy at theta_hz for phase_preference tests."""
    t = np.arange(0, duration_s, 1.0 / fs)
    return np.sin(2.0 * math.pi * theta_hz * t)


# ---------------------------------------------------------------------------
# spike_density
# ---------------------------------------------------------------------------

class TestSpikeDensity:
    def test_returns_two_arrays(self) -> None:
        spikes = _theta_modulated_spikes(n_cells=10, duration_s=2.0)
        t, dens = spectral.spike_density(spikes, duration_s=2.0)
        assert t.ndim == 1
        assert dens.ndim == 1
        assert t.shape == dens.shape

    def test_length_matches_duration(self) -> None:
        duration_s = 5.0
        dt_s = 0.001
        spikes = _theta_modulated_spikes(n_cells=10, duration_s=duration_s)
        t, dens = spectral.spike_density(spikes, duration_s=duration_s, dt_s=dt_s)
        expected_len = int(duration_s / dt_s)
        # Allow +-2 samples due to rounding
        assert abs(len(t) - expected_len) <= 2

    def test_nonnegative(self) -> None:
        spikes = _theta_modulated_spikes(n_cells=10, duration_s=2.0)
        _, dens = spectral.spike_density(spikes, duration_s=2.0)
        assert np.all(dens >= 0)


class TestLfpProxy:
    def test_refuses_all_cell_fallback_when_pyramidal_spikes_are_missing(self) -> None:
        meta = types_mod.SimMeta(
            duration_s=1.0,
            dt_s=0.001,
            n_cells_per_type={"PV_Basket": 1},
            scale=1.0,
            seed=1,
            backend="test",
            config_name="missing_pyramidal_lfp_proxy",
            crop_first_ms=0.0,
            lfp_proxy="pyramidal_spike_density",
        )
        result = types_mod.SimResult(
            spikes={"PV_Basket": [np.array([0.1, 0.2], dtype=float)]},
            meta=meta,
            lfp=None,
            lfp_dt_s=None,
        )

        with pytest.raises(ValueError, match="Pyramidal"):
            _ = spectral.lfp_proxy(result)


# ---------------------------------------------------------------------------
# welch_psd
# ---------------------------------------------------------------------------

class TestWelchPsd:
    def test_returns_freqs_and_power(self) -> None:
        spikes = _theta_modulated_spikes(duration_s=5.0)
        _, dens = spectral.spike_density(spikes, duration_s=5.0, dt_s=0.001)
        fs = 1.0 / 0.001
        freqs, power = spectral.welch_psd(dens, fs=fs)
        assert freqs.ndim == 1
        assert power.ndim == 1
        assert len(freqs) == len(power)
        assert np.all(power >= 0)

    def test_freqs_positive(self) -> None:
        sig = np.sin(2 * math.pi * 7.8 * np.arange(0, 5.0, 0.001))
        freqs, _ = spectral.welch_psd(sig, fs=1000.0)
        assert freqs[0] >= 0.0


# ---------------------------------------------------------------------------
# band_power_peak: detect 7.8 Hz within +/- 1 Hz
# ---------------------------------------------------------------------------

class TestBandPowerPeak:
    def test_detects_theta_peak(self) -> None:
        """Synthetic 7.8 Hz modulation -> detected peak within +/- 1 Hz."""
        duration_s = 10.0
        dt_s = 0.001
        fs = 1.0 / dt_s

        spikes = _theta_modulated_spikes(
            n_cells=300, duration_s=duration_s, theta_hz=7.8
        )
        _, dens = spectral.spike_density(spikes, duration_s=duration_s, dt_s=dt_s)
        freqs, power = spectral.welch_psd(dens, fs=fs)

        peak_freq, peak_power, band_power = spectral.band_power_peak(
            freqs, power, band=(4.0, 12.0)
        )

        assert abs(peak_freq - 7.8) <= 1.0, (
            f"Detected theta peak at {peak_freq:.2f} Hz; expected 7.8 +/- 1 Hz"
        )

    def test_returns_three_floats(self) -> None:
        sig = np.sin(2 * math.pi * 7.8 * np.arange(0, 5.0, 0.001))
        freqs, power = spectral.welch_psd(sig, fs=1000.0)
        result = spectral.band_power_peak(freqs, power, band=(4.0, 12.0))
        assert len(result) == 3

    def test_band_power_positive(self) -> None:
        sig = np.sin(2 * math.pi * 7.8 * np.arange(0, 5.0, 0.001))
        freqs, power = spectral.welch_psd(sig, fs=1000.0)
        _, _, band_power = spectral.band_power_peak(freqs, power, band=(4.0, 12.0))
        assert band_power > 0.0

    def test_rejects_monotonic_band_edge_argmax_as_peak(self) -> None:
        freqs = np.arange(1.0, 101.0, 0.25)
        power = 1.0 / freqs

        peak_freq, _, _, prominence, is_low_edge = spectral.band_power_peak(
            freqs,
            power,
            band=(5.0, 10.0),
            return_prominence=True,
        )

        assert peak_freq == 5.0
        assert is_low_edge
        assert prominence < spectral.PEAK_PROMINENCE_MIN_RATIO


class TestThetaGammaCfc:
    @staticmethod
    def _lfp(*, coupled: bool) -> tuple[np.ndarray, float]:
        fs = 500.0
        t = np.arange(0.0, 20.0, 1.0 / fs)
        theta = np.sin(2.0 * math.pi * 7.8 * t)
        if coupled:
            gamma_amplitude = 0.25 * (
                1.0 + 0.9 * np.cos(2.0 * math.pi * 7.8 * t)
            )
        else:
            gamma_amplitude = np.full_like(t, 0.25)
        gamma = gamma_amplitude * np.sin(2.0 * math.pi * 71.0 * t)
        return 2.0 * theta + gamma, fs

    def test_phase_locked_gamma_is_surrogate_significant(self) -> None:
        lfp, fs = self._lfp(coupled=True)

        mi, p_value, z_score = spectral.theta_gamma_cfc(
            lfp, fs, n_surrogates=99, random_seed=7
        )

        assert mi > 0.0
        assert p_value <= 0.05
        assert z_score >= spectral.CFC_MIN_Z_SCORE

    def test_independent_gamma_amplitude_is_not_surrogate_significant(self) -> None:
        lfp, fs = self._lfp(coupled=False)

        _, p_value, z_score = spectral.theta_gamma_cfc(
            lfp, fs, n_surrogates=99, random_seed=7
        )

        assert p_value > 0.05
        assert z_score < spectral.CFC_MIN_Z_SCORE


# ---------------------------------------------------------------------------
# phase_preference: Rayleigh test for phase-locked spikes
# ---------------------------------------------------------------------------

class TestPhasePreference:
    def test_phase_locked_low_rayleigh_p(self) -> None:
        """Strongly phase-locked train -> Rayleigh p < 0.05."""
        spike_times = _phase_locked_spikes(
            n_spikes=500,
            theta_hz=7.8,
            preferred_phase_deg=90.0,
            jitter_deg=15.0,
        )
        fs = 1000.0
        lfp = _fake_lfp_theta(duration_s=float(spike_times.max()) + 1.0, fs=fs)

        mean_phase_deg, resultant_len, p_val = spectral.phase_preference(
            spike_times, lfp, fs=fs, band=(5.0, 10.0)
        )

        assert p_val < 0.05, (
            f"Rayleigh p={p_val:.4f} >= 0.05 for a strongly phase-locked train; "
            "phase_preference may not be computing the circular statistic correctly"
        )
        # Resultant length R in [0,1]; should be high for locked spikes
        assert resultant_len > 0.2, (
            f"Resultant vector length R={resultant_len:.3f} too low for locked spikes"
        )

    def test_returns_three_values(self) -> None:
        spike_times = np.array([0.1, 0.228, 0.356])  # tiny set
        fs = 1000.0
        lfp = _fake_lfp_theta(duration_s=1.0, fs=fs)
        result = spectral.phase_preference(spike_times, lfp, fs=fs, band=(5.0, 10.0))
        assert len(result) == 3

    def test_mean_phase_in_range(self) -> None:
        spike_times = _phase_locked_spikes(n_spikes=200, theta_hz=7.8)
        fs = 1000.0
        lfp = _fake_lfp_theta(duration_s=float(spike_times.max()) + 1.0, fs=fs)
        mean_phase_deg, _, _ = spectral.phase_preference(
            spike_times, lfp, fs=fs, band=(5.0, 10.0)
        )
        assert -180.0 <= mean_phase_deg <= 360.0
