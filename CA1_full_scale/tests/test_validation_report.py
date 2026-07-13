from __future__ import annotations

from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pytest

from ca1.types import SimMeta, SimResult
from ca1.validation import report


def _minimal_result() -> SimResult:
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="validation_report",
        crop_first_ms=0.0,
        lfp_proxy="modeldb_n_pole_reduced_domain_lfp",
    )
    return SimResult(
        spikes={"Pyramidal": [np.array([0.1, 0.2], dtype=float)]},
        meta=meta,
        lfp=np.sin(2.0 * np.pi * 7.8 * np.arange(0.0, 1.0, 0.001)),
        lfp_dt_s=0.001,
    )


def _no_lfp_result() -> SimResult:
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 1},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="validation_report_no_lfp",
        crop_first_ms=0.0,
        lfp_proxy="modeldb_n_pole_reduced_domain_lfp",
    )
    return SimResult(
        spikes={"Pyramidal": [np.array([0.1, 0.2], dtype=float)]},
        meta=meta,
        lfp=None,
        lfp_dt_s=None,
    )


def test_report_mean_rates_refuses_inline_fallback_when_rates_module_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_rates() -> NoReturn:
        raise ImportError("ca1.analysis.rates hidden for test")

    monkeypatch.setattr(report, "_rates_module", missing_rates)

    with pytest.raises(ImportError, match="ca1.analysis.rates"):
        _ = report.compare(None, _minimal_result())


def test_report_spectral_metrics_refuse_empty_fallback_when_spectral_module_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_spectral() -> NoReturn:
        raise ImportError("ca1.analysis.spectral hidden for test")

    monkeypatch.setattr(report, "_spectral_module", missing_spectral)

    with pytest.raises(ImportError, match="ca1.analysis.spectral"):
        _ = report.compare(None, _minimal_result())


def test_report_tables_do_not_use_spike_density_spectral_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hidden_spectral_fallback() -> NoReturn:
        raise AssertionError("report table must not call spectral fallback")

    def passing_validate(_result: SimResult) -> SimpleNamespace:
        return SimpleNamespace(tier="full", passed=True, checks=[])

    import ca1.validation.harness as harness

    monkeypatch.setattr(report, "_spectral_module", hidden_spectral_fallback)
    monkeypatch.setattr(harness, "validate", passing_validate)

    rendered = report.compare(None, _no_lfp_result())

    assert "| Theta peak (Hz) | — | — |" in rendered


def test_report_propagates_validation_errors_instead_of_rendering_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IntentionalValidationError(RuntimeError):
        pass

    def failing_validate(_result: SimResult) -> NoReturn:
        raise IntentionalValidationError("validation failure must stay detectable")

    import ca1.validation.harness as harness

    monkeypatch.setattr(harness, "validate", failing_validate)

    with pytest.raises(
        IntentionalValidationError,
        match="validation failure must stay detectable",
    ):
        _ = report.compare(None, _minimal_result())
