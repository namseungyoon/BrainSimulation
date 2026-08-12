from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from ca1.types import ElectrodeRoi, SimMeta, SimResult
from ca1.validation import acceptance
from ca1.validation.harness import validate
from ca1.validation.targets import MODEL_PHASE_DEG, MODEL_RATES_HZ


def _lfp_result(lfp: npt.NDArray[np.float64], fs: float) -> SimResult:
    duration_s = len(lfp) / fs
    return SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=SimMeta(
            duration_s=duration_s,
            dt_s=1.0 / fs,
            n_cells_per_type={"Pyramidal": 1},
            scale=1.0,
            seed=1,
            backend="test",
            config_name="synthetic_lfp",
            crop_first_ms=0.0,
            lfp_proxy="modeldb_n_pole_reduced_domain_lfp",
        ),
        lfp=lfp,
        lfp_dt_s=1.0 / fs,
    )


def _check_by_name(checks: list, name: str):
    return next(check for check in checks if check.name == name)


def test_aperiodic_one_over_f_lfp_fails_peak_gates() -> None:
    fs = 500.0
    duration_s = 20.0
    n = int(fs * duration_s)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    rng = np.random.default_rng(2026)
    spectrum = np.zeros(freqs.size, dtype=complex)
    spectrum[1:] = (
        rng.normal(size=freqs.size - 1)
        + 1j * rng.normal(size=freqs.size - 1)
    ) / np.sqrt(freqs[1:])
    lfp = np.fft.irfft(spectrum, n=n)

    checks = acceptance.check_oscillation(_lfp_result(lfp, fs))

    theta = _check_by_name(checks, "oscillation/theta_peak")
    gamma = _check_by_name(checks, "oscillation/gamma_peak")
    assert not theta.passed
    assert not gamma.passed
    assert not _check_by_name(checks, "oscillation/theta_dominates_gamma").passed
    assert not _check_by_name(checks, "oscillation/theta_gamma_cfc").passed
    assert (
        theta.metrics["peak_prominence_ratio"]
        < theta.metrics["min_prominence_ratio"]
        or theta.metrics["is_low_band_edge"]
    )


@pytest.mark.parametrize("coupled", [True, False])
def test_theta_and_gamma_peaks_are_genuine_for_synthetic_oscillations(
    coupled: bool,
) -> None:
    fs = 500.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    theta = 2.0 * np.sin(2.0 * np.pi * 7.8 * t)
    gamma_amplitude = (
        0.25 * (1.0 + 0.9 * np.cos(2.0 * np.pi * 7.8 * t))
        if coupled
        else np.full_like(t, 0.25)
    )
    gamma = gamma_amplitude * np.sin(2.0 * np.pi * 71.0 * t)

    checks = acceptance.check_oscillation(_lfp_result(theta + gamma, fs))

    assert _check_by_name(checks, "oscillation/theta_peak").passed
    assert _check_by_name(checks, "oscillation/gamma_peak").passed
    assert _check_by_name(checks, "oscillation/theta_dominates_gamma").passed
    assert _check_by_name(checks, "oscillation/theta_gamma_cfc").passed is coupled


def test_narrow_theta_dominates_broad_weak_gamma_by_peak_power() -> None:
    fs = 500.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    lfp = np.sin(2.0 * np.pi * 7.8 * t)
    for gamma_hz in np.arange(25.5, 80.0, 0.5):
        lfp += 0.12 * np.sin(
            2.0 * np.pi * gamma_hz * t + 0.173 * gamma_hz
        )
    lfp += 0.25 * np.sin(2.0 * np.pi * 71.0 * t)

    checks = acceptance.check_oscillation(_lfp_result(lfp, fs))

    assert _check_by_name(checks, "oscillation/theta_peak").passed
    assert _check_by_name(checks, "oscillation/gamma_peak").passed
    dominance = _check_by_name(checks, "oscillation/theta_dominates_gamma")
    assert dominance.metrics["gamma_band_power"] > dominance.metrics["theta_band_power"]
    assert dominance.metrics["theta_peak_power"] > dominance.metrics["gamma_peak_power"]
    assert dominance.passed
    assert "max-in-band" in dominance.detail
    assert "integrated" in dominance.detail


def test_narrow_gamma_dominates_weak_theta_by_peak_power() -> None:
    fs = 500.0
    t = np.arange(0.0, 20.0, 1.0 / fs)
    theta = 0.25 * np.sin(2.0 * np.pi * 7.8 * t)
    gamma = np.sin(2.0 * np.pi * 71.0 * t)

    checks = acceptance.check_oscillation(_lfp_result(theta + gamma, fs))

    assert _check_by_name(checks, "oscillation/theta_peak").passed
    assert _check_by_name(checks, "oscillation/gamma_peak").passed
    dominance = _check_by_name(checks, "oscillation/theta_dominates_gamma")
    assert dominance.metrics["gamma_peak_power"] > dominance.metrics["theta_peak_power"]
    assert not dominance.passed


def test_modulation_depth_distinguishes_theta_locked_from_uniform_spikes() -> None:
    fs = 500.0
    duration_s = 20.0
    t = np.arange(0.0, duration_s, 1.0 / fs)
    lfp = np.sin(2.0 * np.pi * 7.8 * t)
    cycle_times = np.arange(0.25, duration_s - 0.25, 1.0 / 7.8)
    rng = np.random.default_rng(91)
    locked = cycle_times + rng.normal(0.0, 0.001, size=cycle_times.size)
    uniform = np.sort(rng.uniform(0.25, duration_s - 0.25, size=1000))

    def modulation_check(spikes: npt.NDArray[np.float64]):
        result = _lfp_result(lfp, fs)
        result.spikes = {"Pyramidal": [spikes]}
        return _check_by_name(
            acceptance.check_phase(result), "modulation/Pyramidal"
        )

    assert modulation_check(locked).passed
    assert not modulation_check(uniform).passed


def test_harness_demotes_phase_without_long_prominent_theta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ca1.validation.harness as harness
    from ca1.types import CheckResult

    phase_check = CheckResult("phase/Pyramidal", False, True, "synthetic")
    monkeypatch.setattr(harness, "check_first_order", lambda _result: [])
    monkeypatch.setattr(harness, "check_provenance", lambda _result, required: [])
    monkeypatch.setattr(harness, "check_phase", lambda _result: [phase_check])

    short = _lfp_result(np.zeros(250), 500.0)
    monkeypatch.setattr(
        harness,
        "check_oscillation",
        lambda _result: [CheckResult(
            "oscillation/theta_peak", True, True, "synthetic",
            {"theta_peak_hz": 7.8, "peak_prominence_ratio": 10.0},
        )],
    )
    short_report = harness.validate(short, tier="full")
    assert not _check_by_name(short_report.checks, "phase/Pyramidal").required

    long = _lfp_result(np.zeros(1000), 500.0)
    long_report = harness.validate(long, tier="full")
    assert _check_by_name(long_report.checks, "phase/Pyramidal").required

    monkeypatch.setattr(
        harness,
        "check_oscillation",
        lambda _result: [CheckResult(
            "oscillation/theta_peak", False, True, "synthetic",
            {"theta_peak_hz": 7.8, "peak_prominence_ratio": 1.0},
        )],
    )
    no_peak_report = harness.validate(long, tier="full")
    assert not _check_by_name(no_peak_report.checks, "phase/Pyramidal").required


def test_check_phase_uses_rayleigh_p_value_when_phase_preference_returns_vector_strength_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell_by_marker = {
        float(idx + 1): cell_type
        for idx, cell_type in enumerate(MODEL_PHASE_DEG)
    }
    spikes = {
        cell_type: [np.full(5, float(idx + 1))]
        for idx, cell_type in enumerate(MODEL_PHASE_DEG)
    }
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={cell_type: 1 for cell_type in MODEL_PHASE_DEG},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="phase_return_order",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_spike_density",
    )

    def fake_phase_preference(
        spike_times: npt.NDArray[np.float64],
        _lfp: npt.NDArray[np.float64],
        _fs: float,
        band: tuple[float, float],
    ) -> tuple[float, float, float]:
        del band
        marker = cast(float, spike_times.tolist()[0])
        cell_type = cell_by_marker[marker]
        return MODEL_PHASE_DEG[cell_type], 0.9, 1e-12

    def fake_lfp_proxy(_result: SimResult) -> tuple[npt.NDArray[np.float64], float]:
        return np.ones(100), 1000.0

    fake_spectral = SimpleNamespace(
        lfp_proxy=fake_lfp_proxy,
        phase_preference=fake_phase_preference,
    )
    monkeypatch.setattr(acceptance, "_load_spectral", lambda: fake_spectral)

    checks = acceptance.check_phase(SimResult(spikes=spikes, meta=meta))

    per_type_checks = [
        check for check in checks if check.name.startswith("phase/")
    ]
    assert per_type_checks
    assert all(check.passed for check in per_type_checks), (
        "check_phase treated vector strength as Rayleigh p-value; "
        "phase_preference returns (mean_phase, vector_strength, rayleigh_p)"
    )
    assert all(
        check.metrics.get("rayleigh_p") == 1e-12
        for check in per_type_checks
        if "rayleigh_p" in check.metrics
    )


def test_validate_reports_no_lfp_when_crop_consumes_duration() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="too_short",
        crop_first_ms=50.0,
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    assert not report.passed
    assert any(check.name == "oscillation/no_lfp" for check in report.checks)
    assert any(check.name == "phase/no_lfp" for check in report.checks)


def test_validate_surfaces_explicit_parameter_fallback_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="provenance",
        crop_first_ms=0.0,
        parameter_provenance={
            "neuron.Pyramidal": "analytic-fallback-after-failed-fit",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "neuron.Pyramidal" in provenance_checks[0].detail
    assert not report.passed


def test_validate_full_tier_fails_when_parameter_provenance_is_missing() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="missing_provenance",
        crop_first_ms=0.0,
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "cannot be audited" in provenance_checks[0].detail
    assert not report.passed


def test_validate_surfaces_declared_missing_parameter_fit_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="declared_missing_provenance",
        crop_first_ms=0.0,
        parameter_provenance={
            "aglif.Pyramidal": "missing-fit-provenance",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "aglif.Pyramidal" in provenance_checks[0].detail
    assert not report.passed


def test_validate_surfaces_placeholder_parameter_fit_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="placeholder_provenance",
        crop_first_ms=0.0,
        parameter_provenance={
            "neuron.Pyramidal": "placeholder",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "neuron.Pyramidal" in provenance_checks[0].detail
    assert not report.passed


def test_validate_surfaces_prototype_parameter_fit_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="prototype_provenance",
        crop_first_ms=0.0,
        parameter_provenance={
            "source_location_transfer.table": "prototype-source-location-transfer",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "source_location_transfer.table" in provenance_checks[0].detail
    assert not report.passed


def test_validate_surfaces_failed_parameter_fit_provenance_case_insensitively() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="failed_provenance",
        crop_first_ms=0.0,
        parameter_provenance={
            "aglif.Pyramidal": "fit-failed",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "aglif.Pyramidal" in provenance_checks[0].detail
    assert not report.passed


def test_validate_scaled_tier_hard_fails_failed_parameter_fit_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=0.1,
        seed=1,
        backend="test",
        config_name="scaled_failed_provenance",
        crop_first_ms=0.0,
        parameter_provenance={
            "aglif.Pyramidal": "failed-validation",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="scaled")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert not provenance_checks[0].required
    assert "aglif.Pyramidal" in provenance_checks[0].detail


def test_validate_scaled_tier_flags_attention_parameter_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=0.1,
        seed=1,
        backend="test",
        config_name="scaled_attention_provenance",
        crop_first_ms=0.0,
        parameter_provenance={
            "network.neuron_model": "aglif_dend_cond_beta",
            "aglif.Pyramidal": "nestgpu-fi-fit",
            "dendritic_transfer.Pyramidal": "neuron-epsp-location-compressed-fit",
            "source_location_transfer.table": (
                "diagnostic-noncanonical-source-location-transfer;mode=all_dend"
            ),
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="scaled")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert not provenance_checks[0].required
    assert "source_location_transfer.table" in provenance_checks[0].detail


def test_validate_full_tier_requires_dendritic_transfer_records_for_aglif_dend() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="missing_dendritic_transfer_records",
        crop_first_ms=0.0,
        parameter_provenance={
            "aglif.Pyramidal": "nestgpu-fi-fit",
            "network.neuron_model": "aglif_dend_cond_beta",
            "network.cell_types": "1",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "dendritic_transfer." in provenance_checks[0].detail
    assert not report.passed


def test_validate_surfaces_unvalidated_source_location_transfer_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="prototype_source_location",
        crop_first_ms=0.0,
        parameter_provenance={
            "source_location_transfer.table": (
                "unvalidated-prototype-source-location-transfer"
            ),
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "source_location_transfer.table" in provenance_checks[0].detail
    assert not report.passed


def test_validate_surfaces_prototype_source_location_transfer_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="prototype_source_location",
        crop_first_ms=0.0,
        parameter_provenance={
            "source_location_transfer.table": "prototype-source-location-transfer",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "source_location_transfer.table" in provenance_checks[0].detail
    assert not report.passed


@pytest.mark.parametrize(
    "fit_provenance",
    ["FAILED", "failed", "fit-failed", "FAILED ", "failed-validation"],
)
def test_validate_surfaces_failed_parameter_fit_variants(
    fit_provenance: str,
) -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="failed_fit",
        crop_first_ms=0.0,
        parameter_provenance={
            "neuron.Pyramidal": fit_provenance,
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "neuron.Pyramidal" in provenance_checks[0].detail
    assert not report.passed


def test_validate_aglif_dend_requires_dendritic_transfer_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="missing_dendritic_transfer",
        crop_first_ms=0.0,
        parameter_provenance={
            "network.neuron_model": "aglif_dend_cond_beta",
            "aglif.Pyramidal": "nestgpu-fi-fit",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    provenance_checks = [
        check for check in report.checks
        if check.name == "provenance/parameter_fits"
    ]
    assert len(provenance_checks) == 1
    assert not provenance_checks[0].passed
    assert provenance_checks[0].required
    assert "dendritic_transfer.Pyramidal" in provenance_checks[0].detail
    assert not report.passed


def test_validate_full_tier_fails_when_diagnostic_audit_metadata_is_missing() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="missing_diagnostic_audit",
        crop_first_ms=0.0,
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    diagnostic_checks = [
        check for check in report.checks
        if check.name == "provenance/diagnostic_runtime"
    ]
    assert len(diagnostic_checks) == 1
    assert not diagnostic_checks[0].passed
    assert diagnostic_checks[0].required
    assert "cannot be audited" in diagnostic_checks[0].detail
    assert not report.passed


def test_validate_full_tier_accepts_clean_diagnostic_audit_metadata() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="clean_diagnostic_audit",
        crop_first_ms=0.0,
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    diagnostic_checks = [
        check for check in report.checks
        if check.name == "provenance/diagnostic_runtime"
    ]
    assert len(diagnostic_checks) == 1
    assert diagnostic_checks[0].passed
    assert diagnostic_checks[0].required


def test_validate_full_tier_fails_when_lfp_proxy_claims_unstored_current() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="hidden_lfp_proxy_fallback",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_synaptic_current",
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
        lfp=None,
        lfp_dt_s=None,
    )

    report = validate(result, tier="full")

    lfp_checks = [
        check for check in report.checks
        if check.name == "provenance/lfp_proxy"
    ]
    assert len(lfp_checks) == 1
    assert not lfp_checks[0].passed
    assert lfp_checks[0].required
    assert "metadata claims pyramidal_synaptic_current" in lfp_checks[0].detail
    assert not report.passed


def test_validate_full_tier_rejects_reduced_current_lfp_proxy_for_final_phase_evidence() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="reduced_lfp_proxy",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_synaptic_current",
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
        lfp=np.sin(2.0 * np.pi * 7.8 * np.arange(0.0, 0.05, 0.001)),
        lfp_dt_s=0.001,
    )

    report = validate(result, tier="full")

    lfp_checks = [
        check for check in report.checks
        if check.name == "provenance/lfp_proxy"
    ]
    assert len(lfp_checks) == 1
    assert not lfp_checks[0].passed
    assert lfp_checks[0].required
    assert "diagnostic/scaled proxy" in lfp_checks[0].detail
    assert "modeldb_n_pole_reduced_domain_lfp" in lfp_checks[0].detail
    assert not report.passed


def test_validate_full_tier_rejects_n_pole_lfp_proxy_without_roi_context() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="n_pole_lfp_proxy",
        crop_first_ms=0.0,
        lfp_proxy="modeldb_n_pole_reduced_domain_lfp",
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
            "lfp.modeldb_n_pole_reduced_domain": "modeldb-n-pole-reduced-domain-lfp",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
        lfp=np.sin(2.0 * np.pi * 7.8 * np.arange(0.0, 0.05, 0.001)),
        lfp_dt_s=0.001,
    )

    report = validate(result, tier="full")

    lfp_checks = [
        check for check in report.checks
        if check.name == "provenance/lfp_proxy"
    ]
    assert len(lfp_checks) == 1
    assert not lfp_checks[0].passed
    assert lfp_checks[0].required
    assert "requires electrode ROI" in lfp_checks[0].detail


def test_validate_full_tier_accepts_explicit_n_pole_lfp_proxy_provenance() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="n_pole_lfp_proxy",
        crop_first_ms=0.0,
        lfp_proxy="modeldb_n_pole_reduced_domain_lfp",
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
            "lfp.modeldb_n_pole_reduced_domain": "modeldb-n-pole-reduced-domain-lfp",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
        lfp=np.sin(2.0 * np.pi * 7.8 * np.arange(0.0, 0.05, 0.001)),
        lfp_dt_s=0.001,
        cell_positions_um={"Pyramidal": np.array([[200.0, 100.0, 120.0]])},
        analysis_roi=ElectrodeRoi(
            center_um=(200.0, 100.0, 120.0),
            radius_um=1000.0,
            distance_mode="xyz",
        ),
    )

    report = validate(result, tier="full")

    lfp_checks = [
        check for check in report.checks
        if check.name == "provenance/lfp_proxy"
    ]
    assert len(lfp_checks) == 1
    assert lfp_checks[0].passed
    assert lfp_checks[0].required


def test_spectral_checks_refuse_hidden_spike_density_fallback_for_claimed_current() -> None:
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="hidden_spectral_fallback",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_synaptic_current",
    )
    result = SimResult(
        spikes={
            "Pyramidal": [
                np.array([0.10, 0.22, 0.34, 0.46, 0.58, 0.70], dtype=float)
            ]
        },
        meta=meta,
        lfp=None,
        lfp_dt_s=None,
    )

    oscillation_checks = acceptance.check_oscillation(result)
    phase_checks = acceptance.check_phase(result)

    assert oscillation_checks[0].name == "oscillation/no_lfp"
    assert phase_checks[0].name == "phase/no_lfp"
    assert "refusing hidden spectral fallback" in oscillation_checks[0].detail
    assert "refusing hidden spectral fallback" in phase_checks[0].detail


def test_validate_full_tier_fails_when_lfp_proxy_is_implicit_default() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="implicit_lfp_proxy_default",
        crop_first_ms=0.0,
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
        lfp=None,
        lfp_dt_s=None,
    )

    report = validate(result, tier="full")

    lfp_checks = [
        check for check in report.checks
        if check.name == "provenance/lfp_proxy"
    ]
    assert len(lfp_checks) == 1
    assert not lfp_checks[0].passed
    assert lfp_checks[0].required
    assert "metadata missing" in lfp_checks[0].detail
    assert not report.passed


def test_validate_full_tier_rejects_spike_density_lfp_proxy() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="visible_lfp_proxy",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_spike_density",
        parameter_provenance={
            "neuron.Pyramidal": "nest-validated",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
        lfp=None,
        lfp_dt_s=None,
    )

    report = validate(result, tier="full")

    lfp_checks = [
        check for check in report.checks
        if check.name == "provenance/lfp_proxy"
    ]
    assert len(lfp_checks) == 1
    assert not lfp_checks[0].passed
    assert lfp_checks[0].required
    assert "pyramidal_spike_density" in lfp_checks[0].detail
    assert not report.passed


def test_validate_fails_on_diagnostic_runtime_overrides() -> None:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="diagnostic",
        crop_first_ms=0.0,
        diagnostic_provenance={
            "env.CA1_AGLIF_DEND_GC_SCALE_BISTRATIFIED": "5",
        },
    )
    result = SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )

    report = validate(result, tier="full")

    diagnostic_checks = [
        check for check in report.checks
        if check.name == "provenance/diagnostic_runtime"
    ]
    assert len(diagnostic_checks) == 1
    assert not diagnostic_checks[0].passed
    assert diagnostic_checks[0].required
    assert "CA1_AGLIF_DEND_GC_SCALE_BISTRATIFIED" in diagnostic_checks[0].detail


def test_check_first_order_warns_when_cv_isi_is_unavailable_for_target_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={cell_type: 1 for cell_type in MODEL_RATES_HZ},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="missing_cv",
        crop_first_ms=0.0,
    )
    result = SimResult(
        spikes={
            cell_type: [np.array([0.1], dtype=float)]
            for cell_type in MODEL_RATES_HZ
        },
        meta=meta,
    )

    class FakeRates:
        def mean_rates(
            self,
            _spikes: acceptance.Spikes,
            _duration_s: float,
            active_only: bool = False,
        ) -> dict[str, float]:
            del active_only
            return dict(MODEL_RATES_HZ)

        def cv_isi(self, _spikes: acceptance.Spikes) -> dict[str, float]:
            return {}

    fake_rates = FakeRates()
    monkeypatch.setattr(acceptance, "_load_rates", lambda: fake_rates)

    checks = acceptance.check_first_order(result)

    cv_checks = [check for check in checks if check.name.startswith("cv_isi/")]
    assert len(cv_checks) == len(MODEL_RATES_HZ)
    assert all(not check.required for check in cv_checks)
    assert all(not check.passed for check in cv_checks)
    assert all("insufficient" in check.detail for check in cv_checks)
    assert all("not a Table 5 hard gate" in check.detail for check in cv_checks)


def test_check_first_order_warns_when_pyramidal_sparseness_exceeds_policy_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={cell_type: 1 for cell_type in MODEL_RATES_HZ},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="pyramidal_sparseness_policy",
        crop_first_ms=0.0,
    )
    result = SimResult(
        spikes={
            **{
                cell_type: [np.array([0.1, 0.2, 0.3], dtype=float)]
                for cell_type in MODEL_RATES_HZ
                if cell_type != "Pyramidal"
            },
            "Pyramidal": [
                np.array([0.1, 0.2, 0.3], dtype=float),
                np.array([0.15, 0.25, 0.35], dtype=float),
            ],
        },
        meta=meta,
    )

    class FakeRates:
        def mean_rates(
            self,
            _spikes: acceptance.Spikes,
            _duration_s: float,
            active_only: bool = False,
        ) -> dict[str, float]:
            del active_only
            return dict(MODEL_RATES_HZ)

        def cv_isi(self, _spikes: acceptance.Spikes) -> dict[str, float]:
            return {cell_type: 1.0 for cell_type in MODEL_RATES_HZ}

    fake_rates = FakeRates()
    monkeypatch.setattr(acceptance, "_load_rates", lambda: fake_rates)

    checks = acceptance.check_first_order(result)

    sparseness_checks = [
        check for check in checks if check.name == "pyramidal_sparseness"
    ]
    assert len(sparseness_checks) == 1
    assert not sparseness_checks[0].passed
    assert not sparseness_checks[0].required
    assert "not a Table 5 hard gate" in sparseness_checks[0].detail


def test_check_first_order_fails_loud_when_rates_module_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={cell_type: 1 for cell_type in MODEL_RATES_HZ},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="missing_rates_module",
        crop_first_ms=0.0,
    )
    result = SimResult(
        spikes={
            cell_type: [np.array([0.1, 0.2, 0.3], dtype=float)]
            for cell_type in MODEL_RATES_HZ
        },
        meta=meta,
    )
    monkeypatch.setattr(acceptance, "_load_rates", lambda: None)

    checks = acceptance.check_first_order(result)

    assert len(checks) == 1
    assert checks[0].name == "first_order/unavailable"
    assert checks[0].required
    assert not checks[0].passed
    assert "refusing local fallback" in checks[0].detail


def test_validate_full_tier_fails_loud_when_spectral_module_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={cell_type: 1 for cell_type in MODEL_RATES_HZ},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="missing_spectral_module",
        crop_first_ms=0.0,
        parameter_provenance={
            f"neuron.{cell_type}": "nest-validated"
            for cell_type in MODEL_RATES_HZ
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
        },
    )
    result = SimResult(
        spikes={
            cell_type: [np.array([0.1, 0.2, 0.3], dtype=float)]
            for cell_type in MODEL_RATES_HZ
        },
        meta=meta,
    )
    monkeypatch.setattr(acceptance, "_load_spectral", lambda: None)

    report = validate(result, tier="full")

    unavailable = {
        check.name: check
        for check in report.checks
        if check.name in {"oscillation/unavailable", "phase/unavailable"}
    }
    assert set(unavailable) == {"oscillation/unavailable", "phase/unavailable"}
    assert all(check.required for check in unavailable.values())
    assert all(not check.passed for check in unavailable.values())
    assert not report.passed
