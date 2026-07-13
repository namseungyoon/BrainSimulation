"""Per-criterion acceptance checks for a CA1 SimResult.

Each ``check_*`` function returns a list of ``CheckResult`` objects, one per
tested quantity.  All checks are pure functions over a ``SimResult``; they
import the analysis functions lazily so the module can be imported even when
``ca1.analysis`` is not yet fully implemented.

Paper reference: Bezaire et al. (2016) eLife 5:e18566.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ca1.types import CheckResult, MetricValue, SimResult
from ca1.sim.modeldb_positions import electrode_roi_mask, filter_spikes_to_roi
from ca1.validation.targets import (
    CV_ISI_RANGE,
    GAMMA_BAND,
    GAMMA_PEAK_HZ,
    GAMMA_PEAK_TOLERANCE_HZ,
    CFC_MIN_Z_SCORE,
    CFC_MIN_WINDOW_S,
    CFC_N_SURROGATES,
    CFC_SURROGATE_ALPHA,
    MODEL_MODULATION,
    MODEL_PHASE_DEG,
    MODEL_RATES_HZ,
    MODULATION_ABS_TOL,
    MODULATION_REL_TOL,
    PHASE_TOLERANCE_DEG,
    RAYLEIGH_ALPHA,
    RATE_REL_TOL,
    RISING_GROUP,
    SPECTRAL_PEAK_MIN_PROMINENCE_RATIO,
    THETA_BAND,
    THETA_PEAK_HZ,
    TROUGH_GROUP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Spikes = dict[str, list[npt.NDArray[np.float64]]]
_LFP_PROXY_SPIKE_DENSITY = "pyramidal_spike_density"


class RatesModule(Protocol):
    def mean_rates(
        self,
        spikes: Spikes,
        duration_s: float,
        active_only: bool = False,
    ) -> dict[str, float]:
        ...

    def cv_isi(self, spikes: Spikes) -> dict[str, float]:
        ...


class SpectralModule(Protocol):
    def lfp_proxy(
        self,
        result: SimResult,
    ) -> tuple[npt.NDArray[np.float64] | None, float]:
        ...

    def welch_psd(
        self,
        sig: npt.NDArray[np.float64],
        fs: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        ...

    def band_power_peak(
        self,
        freqs: npt.NDArray[np.float64],
        power: npt.NDArray[np.float64],
        band: tuple[float, float],
        *,
        return_prominence: bool = False,
    ) -> tuple[float, ...]:
        ...

    def theta_gamma_cfc(
        self,
        lfp: npt.NDArray[np.float64],
        fs: float,
        *,
        n_surrogates: int,
        random_seed: int,
    ) -> tuple[float, float, float]:
        ...

    def phase_preference(
        self,
        spike_times: npt.NDArray[np.float64],
        lfp: npt.NDArray[np.float64],
        fs: float,
        band: tuple[float, float],
    ) -> tuple[float, float, float]:
        ...


def _angular_distance(a_deg: float, b_deg: float) -> float:
    """Smallest unsigned circular distance between two angles (degrees)."""
    diff = abs(a_deg - b_deg) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _load_rates() -> RatesModule | None:
    """Lazy import of ca1.analysis.rates (may not exist yet)."""
    try:
        import ca1.analysis.rates as _r  # type: ignore[import]
        return _r
    except ImportError:
        return None


def _load_spectral() -> SpectralModule | None:
    """Lazy import of ca1.analysis.spectral (may not exist yet)."""
    try:
        import ca1.analysis.spectral as _s  # type: ignore[import]
        return _s
    except ImportError:
        return None


def _analysis_duration_s(result: SimResult) -> float:
    return result.meta.duration_s - result.meta.crop_first_ms * 1e-3


# ---------------------------------------------------------------------------
# First-order checks (rates + audit-only ISI/sparseness statistics)
# ---------------------------------------------------------------------------

def check_first_order(result: SimResult) -> list[CheckResult]:
    """Check per-type mean rates plus provenance-visible plausibility warnings.

    Criteria
    --------
    1. Per-type mean rate within RATE_REL_TOL (30%) of MODEL_RATES_HZ target.
    2. CV(ISI) and Pyramidal active fraction are audit-only plausibility checks.
       They are not Table 5 hard gates; keeping them as visible warnings prevents
       unsupported local policy from masquerading as paper-faithful acceptance.

    Returns one required CheckResult per cell type (rate), one warning-only CV
    check per active type, and one warning-only active-fraction check for Pyramidal.
    """
    rates_mod = _load_rates()
    checks: list[CheckResult] = []

    duration_s = result.meta.duration_s - result.meta.crop_first_ms * 1e-3
    if duration_s <= 0.0:
        duration_s = result.meta.duration_s

    if rates_mod is None:
        return [
            CheckResult(
                name="first_order/unavailable",
                passed=False,
                required=True,
                detail=(
                    "ca1.analysis.rates not available; refusing local fallback "
                    "because rate/CV analysis provenance would be unauditable"
                ),
            )
        ]

    analysis_spikes = _analysis_spikes(result)
    scope_by_type = _analysis_scope_by_type(result)

    # --- compute mean rates ---
    mean_r: dict[str, float] = rates_mod.mean_rates(
        analysis_spikes, duration_s, active_only=False
    )
    cv_r: dict[str, float] = rates_mod.cv_isi(analysis_spikes)

    # --- rate checks ---
    for ct, target in MODEL_RATES_HZ.items():
        measured = mean_r.get(ct, None)
        if measured is None:
            checks.append(CheckResult(
                name=f"rate/{ct}",
                passed=False,
                required=True,
                detail=f"cell type '{ct}' not found in spikes",
                metrics={"target_hz": target},
            ))
            continue
        lo = target * (1.0 - RATE_REL_TOL)
        hi = target * (1.0 + RATE_REL_TOL)
        passed = lo <= measured <= hi
        checks.append(CheckResult(
            name=f"rate/{ct}",
            passed=passed,
            required=True,
            detail=(
                f"measured={measured:.2f} Hz, target={target:.2f} Hz "
                f"(tol={RATE_REL_TOL*100:.0f}%), band=[{lo:.2f},{hi:.2f}]"
            ),
            metrics={"measured_hz": measured, "target_hz": target,
                     "lo_hz": lo, "hi_hz": hi, **scope_by_type.get(ct, {})},
        ))

    # --- CV(ISI) checks ---
    cv_lo, cv_hi = CV_ISI_RANGE
    for ct in MODEL_RATES_HZ:
        cv = cv_r.get(ct, float("nan"))
        if math.isnan(cv):
            checks.append(CheckResult(
                name=f"cv_isi/{ct}",
                passed=False,
                required=False,
                detail=(
                    "CV(ISI) unavailable: insufficient spikes/data for "
                    f"cell type '{ct}'; not a Table 5 hard gate"
                ),
                metrics={"cv_lo": cv_lo, "cv_hi": cv_hi},
            ))
            continue
        passed = cv_lo <= cv <= cv_hi
        checks.append(CheckResult(
            name=f"cv_isi/{ct}",
            passed=passed,
            required=False,
            detail=(
                f"CV(ISI)={cv:.3f}, expected [{cv_lo},{cv_hi}]; "
                "not a Table 5 hard gate"
            ),
            metrics={"cv_isi": cv, "cv_lo": cv_lo, "cv_hi": cv_hi},
        ))

    # --- Pyramidal sparseness check ---
    pyr_cells = analysis_spikes.get("Pyramidal", [])
    if pyr_cells:
        n_total = len(pyr_cells)
        n_active = sum(1 for a in pyr_cells if a.size > 0)
        frac_active = n_active / n_total if n_total > 0 else 0.0
        # Local plausibility policy only: Table 5 reports rate, modulation, p-value,
        # and phase, but not active-cell fraction.
        passed = frac_active < 0.50
        checks.append(CheckResult(
            name="pyramidal_sparseness",
            passed=passed,
            required=False,
            detail=(
                f"{n_active}/{n_total} pyramidal cells active "
                f"({frac_active*100:.1f}%); local plausibility threshold < 50%, "
                "not a Table 5 hard gate"
            ),
            metrics={"frac_active": frac_active, "n_active": n_active,
                     "n_total": n_total, **scope_by_type.get("Pyramidal", {})},
        ))

    return checks


# ---------------------------------------------------------------------------
# Oscillation checks (LFP spectral + CFC)
# ---------------------------------------------------------------------------

def check_oscillation(result: SimResult) -> list[CheckResult]:
    """Check theta peak, theta>gamma dominance, gamma peak, and CFC.

    Criteria (all from paper Bezaire 2016, Figure 4 / p.6-7)
    ---------------------------------------------------------
    Peaks must clear a local aperiodic-background prominence ratio and cannot
    be the lowest resolvable band bin.  CFC must be surrogate-significant.
    """
    spectral_mod = _load_spectral()
    checks: list[CheckResult] = []

    if spectral_mod is None:
        checks.append(CheckResult(
            name="oscillation/unavailable",
            passed=False,
            required=True,
            detail=(
                "ca1.analysis.spectral not available; refusing to skip "
                "oscillation evidence"
            ),
        ))
        return checks

    if _analysis_duration_s(result) <= 0.0:
        checks.append(CheckResult(
            name="oscillation/no_lfp",
            passed=False,
            required=True,
            detail="analysis window is empty after cropping; oscillation evidence unavailable",
        ))
        return checks

    lfp_evidence = _lfp_for_spectral_evidence(result, spectral_mod)
    if lfp_evidence is None:
        checks.append(CheckResult(
            name="oscillation/no_lfp",
            passed=False,
            required=True,
            detail=(
                "LFP proxy unavailable or metadata does not explicitly allow "
                "spike-density fallback; refusing hidden spectral fallback"
            ),
        ))
        return checks
    lfp_array, fs = lfp_evidence
    if lfp_array is None or lfp_array.size < 2:
        checks.append(CheckResult(
            name="oscillation/no_lfp",
            passed=False,
            required=True,
            detail="LFP proxy unavailable or too short; oscillation evidence unavailable",
        ))
        return checks

    freqs, power = spectral_mod.welch_psd(lfp_array, fs)

    # 1. Theta peak
    theta_peak_hz, theta_peak_power, theta_band_power, theta_prominence, theta_low_edge = (
        spectral_mod.band_power_peak(
            freqs, power, THETA_BAND, return_prominence=True
        )
    )
    peak_err = abs(theta_peak_hz - THETA_PEAK_HZ)
    theta_prominent = (
        np.isfinite(theta_prominence)
        and theta_prominence >= SPECTRAL_PEAK_MIN_PROMINENCE_RATIO
        and not theta_low_edge
    )
    theta_ok = (
        THETA_BAND[0] <= theta_peak_hz <= THETA_BAND[1]
        and peak_err <= 1.5
        and theta_prominent
    )
    checks.append(CheckResult(
        name="oscillation/theta_peak",
        passed=theta_ok,
        required=True,
        detail=(
            f"theta peak={theta_peak_hz:.2f} Hz "
            f"(target={THETA_PEAK_HZ} Hz, |err|={peak_err:.2f} Hz <= 1.5); "
            f"prominence={theta_prominence:.3g}x "
            f"(>={SPECTRAL_PEAK_MIN_PROMINENCE_RATIO}x), "
            f"low-band-edge={theta_low_edge}"
        ),
        metrics={"theta_peak_hz": theta_peak_hz, "target_hz": THETA_PEAK_HZ,
                 "peak_err_hz": peak_err, "theta_band_power": theta_band_power,
                 "theta_peak_power": theta_peak_power,
                 "peak_prominence_ratio": theta_prominence,
                 "min_prominence_ratio": SPECTRAL_PEAK_MIN_PROMINENCE_RATIO,
                 "is_low_band_edge": theta_low_edge},
    ))

    gamma_peak_hz, gamma_peak_power, gamma_band_power, gamma_prominence, gamma_low_edge = (
        spectral_mod.band_power_peak(
            freqs, power, GAMMA_BAND, return_prominence=True
        )
    )
    gamma_peak_err = abs(gamma_peak_hz - GAMMA_PEAK_HZ)
    gamma_prominent = (
        np.isfinite(gamma_prominence)
        and gamma_prominence >= SPECTRAL_PEAK_MIN_PROMINENCE_RATIO
        and not gamma_low_edge
    )
    gamma_ok = (
        GAMMA_BAND[0] <= gamma_peak_hz <= GAMMA_BAND[1]
        and gamma_peak_err <= GAMMA_PEAK_TOLERANCE_HZ
        and gamma_prominent
    )

    # ModelDB's bezaire_modeldb/customout/Theta_Power_PS_Old.m:62-72 defines
    # each band's power as its maximum spectral value, not its unequal-band integral.
    # Theta > gamma dominance is meaningful only for validated peaks.
    dominance_ok = theta_ok and gamma_ok and theta_peak_power > gamma_peak_power
    checks.append(CheckResult(
        name="oscillation/theta_dominates_gamma",
        passed=dominance_ok,
        required=True,
        detail=(
            f"max-in-band criterion: theta peak power={theta_peak_power:.4g}, "
            f"gamma peak power={gamma_peak_power:.4g}; "
            f"integrated band power (context only): theta={theta_band_power:.4g}, "
            f"gamma={gamma_band_power:.4g}"
        ),
        metrics={"dominance_criterion": "max_in_band_power",
                 "theta_peak_power": theta_peak_power,
                 "gamma_peak_power": gamma_peak_power,
                 "theta_band_power": theta_band_power,
                 "gamma_band_power": gamma_band_power},
    ))

    # 3. Gamma peak
    checks.append(CheckResult(
        name="oscillation/gamma_peak",
        passed=gamma_ok,
        required=True,
        detail=(
            f"gamma peak={gamma_peak_hz:.2f} Hz "
            f"(paper={GAMMA_PEAK_HZ} Hz, |err|={gamma_peak_err:.2f} Hz "
            f"<={GAMMA_PEAK_TOLERANCE_HZ}); "
            f"prominence={gamma_prominence:.3g}x "
            f"(>={SPECTRAL_PEAK_MIN_PROMINENCE_RATIO}x), "
            f"low-band-edge={gamma_low_edge}"
        ),
        metrics={"gamma_peak_hz": gamma_peak_hz, "target_hz": GAMMA_PEAK_HZ,
                 "peak_err_hz": gamma_peak_err,
                 "gamma_peak_power": gamma_peak_power,
                 "peak_prominence_ratio": gamma_prominence,
                 "min_prominence_ratio": SPECTRAL_PEAK_MIN_PROMINENCE_RATIO,
                 "is_low_band_edge": gamma_low_edge},
    ))

    # 4. Theta-gamma CFC
    cfc_mi, cfc_p, cfc_z = spectral_mod.theta_gamma_cfc(
        lfp_array, fs, n_surrogates=CFC_N_SURROGATES, random_seed=result.meta.seed
    )
    cfc_significant = (
        np.isfinite(cfc_mi)
        and ((cfc_p <= CFC_SURROGATE_ALPHA) or (cfc_z >= CFC_MIN_Z_SCORE))
    )
    cfc_ok = theta_ok and gamma_ok and cfc_significant
    checks.append(CheckResult(
        name="oscillation/theta_gamma_cfc",
        passed=cfc_ok,
        required=True,
        detail=(
            f"CFC MI={cfc_mi:.4g}, surrogate p={cfc_p:.4g} "
            f"(<={CFC_SURROGATE_ALPHA}) or z={cfc_z:.3g} (>={CFC_MIN_Z_SCORE}); "
            f"requires validated theta/gamma peaks and >={CFC_MIN_WINDOW_S:g} s"
        ),
        metrics={"cfc_mi": cfc_mi, "cfc_surrogate_p": cfc_p,
                 "cfc_surrogate_z": cfc_z,
                 "cfc_n_surrogates": CFC_N_SURROGATES,
                 "cfc_min_window_s": CFC_MIN_WINDOW_S},
    ))

    return checks


# ---------------------------------------------------------------------------
# Phase-preference checks
# ---------------------------------------------------------------------------

def check_phase(result: SimResult) -> list[CheckResult]:
    """Check per-type mean theta phase preference and trough/rising group ordering.

    Criteria
    --------
    1. Per-type circular distance |mean_phase - target| <= PHASE_TOLERANCE_DEG (45 deg).
    2. Rayleigh test p < RAYLEIGH_ALPHA (0.05) for significant phase-locking.
    3. Trough-group mean phases are all within 90 deg of 0/360 (near trough).
    4. Rising-group mean phases are all within 90 deg of 180 (away from trough).
    5. Trough-group centroid < rising-group centroid in circular distance from trough.
    """
    spectral_mod = _load_spectral()
    checks: list[CheckResult] = []

    if spectral_mod is None:
        checks.append(CheckResult(
            name="phase/unavailable",
            passed=False,
            required=True,
            detail="ca1.analysis.spectral not available; phase evidence unavailable",
        ))
        return checks

    if _analysis_duration_s(result) <= 0.0:
        checks.append(CheckResult(
            name="phase/no_lfp",
            passed=False,
            required=True,
            detail="analysis window is empty after cropping",
        ))
        return checks

    lfp_evidence = _lfp_for_spectral_evidence(result, spectral_mod)
    if lfp_evidence is None:
        checks.append(CheckResult(
            name="phase/no_lfp",
            passed=False,
            required=True,
            detail=(
                "LFP proxy unavailable or metadata does not explicitly allow "
                "spike-density fallback; refusing hidden spectral fallback"
            ),
        ))
        return checks
    lfp_array, fs = lfp_evidence
    if lfp_array is None or lfp_array.size < 2:
        checks.append(CheckResult(
            name="phase/no_lfp",
            passed=False,
            required=True,
            detail="LFP proxy unavailable; phase evidence unavailable",
        ))
        return checks

    # Gather per-type mean phase and Rayleigh stats
    measured_phases: dict[str, float] = {}
    modulation_depths: dict[str, float] = {}
    rayleigh_ps: dict[str, float] = {}

    for ct in MODEL_PHASE_DEG:
        spike_times_list = _analysis_spikes(result).get(ct, [])
        if not spike_times_list:
            continue
        # Flatten spike times for the whole population
        import numpy as _np  # local import; numpy always available
        all_spikes = _np.concatenate(
            [a for a in spike_times_list if len(a) > 0]
        ) if any(len(a) > 0 for a in spike_times_list) else _np.array([])
        if all_spikes.size < 5:
            continue  # too few spikes for meaningful phase
        mean_phase_deg, vector_strength, rayleigh_p = spectral_mod.phase_preference(
            all_spikes, lfp_array, fs, band=THETA_BAND
        )
        measured_phases[ct] = mean_phase_deg
        modulation_depths[ct] = vector_strength
        rayleigh_ps[ct] = rayleigh_p

    # 1 & 2. Per-type phase distance + Rayleigh
    for ct, target_deg in MODEL_PHASE_DEG.items():
        if ct not in measured_phases:
            checks.append(CheckResult(
                name=f"phase/{ct}",
                passed=False,
                required=True,
                detail=f"no spikes or LFP data for '{ct}'",
                metrics={"target_deg": target_deg},
            ))
            checks.append(CheckResult(
                name=f"modulation/{ct}",
                passed=False,
                required=True,
                detail=f"no spikes or LFP data for '{ct}'",
                metrics={"target_modulation": MODEL_MODULATION[ct]},
            ))
            continue

        dist = _angular_distance(measured_phases[ct], target_deg)
        phase_ok = dist <= PHASE_TOLERANCE_DEG
        p_val = rayleigh_ps[ct]
        rayleigh_ok = p_val < RAYLEIGH_ALPHA
        passed = phase_ok and rayleigh_ok
        checks.append(CheckResult(
            name=f"phase/{ct}",
            passed=passed,
            required=True,
            detail=(
                f"mean_phase={measured_phases[ct]:.1f} deg, "
                f"target={target_deg:.1f} deg, "
                f"dist={dist:.1f} deg (<={PHASE_TOLERANCE_DEG}), "
                f"Rayleigh p={p_val:.3g} (<{RAYLEIGH_ALPHA})"
            ),
            metrics={"mean_phase_deg": measured_phases[ct],
                     "target_deg": target_deg,
                     "circular_dist_deg": dist,
                     "rayleigh_p": p_val,
                     **_analysis_scope_by_type(result).get(ct, {})},
        ))

        modulation = modulation_depths[ct]
        target_modulation = MODEL_MODULATION[ct]
        tolerance = max(
            MODULATION_ABS_TOL, MODULATION_REL_TOL * target_modulation
        )
        modulation_lo = max(0.0, target_modulation - tolerance)
        modulation_hi = min(1.0, target_modulation + tolerance)
        modulation_ok = modulation_lo <= modulation <= modulation_hi
        checks.append(CheckResult(
            name=f"modulation/{ct}",
            passed=modulation_ok,
            required=True,
            detail=(
                f"theta vector strength={modulation:.3f}, "
                f"Table 5 target={target_modulation:.3f}, "
                f"tolerance band=[{modulation_lo:.3f},{modulation_hi:.3f}]"
            ),
            metrics={"modulation_depth": modulation,
                     "target_modulation": target_modulation,
                     "modulation_lo": modulation_lo,
                     "modulation_hi": modulation_hi,
                     **_analysis_scope_by_type(result).get(ct, {})},
        ))

    # 3 & 4. Group-level trough/rising ordering
    trough_phases = [measured_phases[ct] for ct in TROUGH_GROUP
                     if ct in measured_phases]
    rising_phases = [measured_phases[ct] for ct in RISING_GROUP
                     if ct in measured_phases]

    # "near trough" = circular distance from 0 deg <= 90 deg
    if trough_phases:
        trough_ok = all(_angular_distance(p, 0.0) <= 90.0 for p in trough_phases)
        trough_centroid = _circular_mean_deg(trough_phases)
        checks.append(CheckResult(
            name="phase/trough_group_near_zero",
            passed=trough_ok,
            required=True,
            detail=(
                f"trough-group phases {[f'{p:.1f}' for p in trough_phases]} deg; "
                f"all within 90 deg of trough? {trough_ok}; "
                f"centroid={trough_centroid:.1f} deg"
            ),
            metrics={"trough_phases": trough_phases,
                     "trough_centroid_deg": trough_centroid},
        ))

    if rising_phases:
        # "away from trough" = circular distance from 0 deg > 90 deg
        rising_ok = all(_angular_distance(p, 0.0) > 90.0 for p in rising_phases)
        rising_centroid = _circular_mean_deg(rising_phases)
        checks.append(CheckResult(
            name="phase/rising_group_away_from_trough",
            passed=rising_ok,
            required=True,
            detail=(
                f"rising-group phases {[f'{p:.1f}' for p in rising_phases]} deg; "
                f"all > 90 deg from trough? {rising_ok}; "
                f"centroid={rising_centroid:.1f} deg"
            ),
            metrics={"rising_phases": rising_phases,
                     "rising_centroid_deg": rising_centroid},
        ))

    # 5. Trough-group centroid closer to 0 than rising-group centroid
    if trough_phases and rising_phases:
        trough_centroid = _circular_mean_deg(trough_phases)
        rising_centroid = _circular_mean_deg(rising_phases)
        d_trough = _angular_distance(trough_centroid, 0.0)
        d_rising = _angular_distance(rising_centroid, 0.0)
        ordering_ok = d_trough < d_rising
        checks.append(CheckResult(
            name="phase/group_ordering",
            passed=ordering_ok,
            required=True,
            detail=(
                f"trough-group centroid dist-from-zero={d_trough:.1f} deg, "
                f"rising-group dist-from-zero={d_rising:.1f} deg; "
                f"trough closer? {ordering_ok}"
            ),
            metrics={"trough_centroid_dist": d_trough,
                     "rising_centroid_dist": d_rising},
        ))

    return checks


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _lfp_for_spectral_evidence(
    result: SimResult,
    spectral_mod: SpectralModule,
) -> tuple[npt.NDArray[np.float64] | None, float] | None:
    if result.lfp is not None and result.lfp_dt_s is not None:
        return result.lfp, 1.0 / result.lfp_dt_s

    if result.meta.lfp_proxy.strip() != _LFP_PROXY_SPIKE_DENSITY:
        return None

    return spectral_mod.lfp_proxy(result)


def _analysis_spikes(result: SimResult) -> Spikes:
    if result.analysis_roi is None or result.cell_positions_um is None:
        return result.spikes
    return filter_spikes_to_roi(
        result.spikes,
        dict(result.cell_positions_um),
        result.analysis_roi,
    )


def _analysis_scope_by_type(result: SimResult) -> dict[str, dict[str, MetricValue]]:
    if result.analysis_roi is None or result.cell_positions_um is None:
        return {
            cell_type: {"analysis_scope": "all_cells", "roi_cells": len(trains)}
            for cell_type, trains in result.spikes.items()
        }
    metrics: dict[str, dict[str, MetricValue]] = {}
    for cell_type, trains in result.spikes.items():
        positions = result.cell_positions_um.get(cell_type)
        if positions is None:
            metrics[cell_type] = {
                "analysis_scope": "all_cells",
                "roi_cells": len(trains),
            }
            continue
        mask = electrode_roi_mask(positions, result.analysis_roi)
        metrics[cell_type] = {
            "analysis_scope": "electrode_roi",
            "roi_cells": int(np.count_nonzero(mask)),
        }
    return metrics


def _circular_mean_deg(angles_deg: list[float]) -> float:
    """Mean direction of a list of angles (degrees), returned in [0, 360)."""
    import math as _math
    if not angles_deg:
        return float("nan")
    sin_sum = sum(_math.sin(_math.radians(a)) for a in angles_deg)
    cos_sum = sum(_math.cos(_math.radians(a)) for a in angles_deg)
    mean_rad = _math.atan2(sin_sum, cos_sum)
    return _math.degrees(mean_rad) % 360.0
