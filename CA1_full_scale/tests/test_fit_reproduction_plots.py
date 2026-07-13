from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ca1.analysis.fit_reproduction_data import fit_metrics, load_reproduction_dataset
from ca1.analysis.fit_reproduction_plots import create_fi_grid, save_figure
from ca1.analysis.fit_reproduction_response_plots import create_response_figure
from ca1.analysis.fit_reproduction_stats import count_stats_for_dataset
from ca1.analysis.fit_reproduction_traces import load_response_traces


def test_load_reproduction_dataset_estimates_aglif_passive_when_replay_is_absent(
    tmp_path: Path,
) -> None:
    # Given
    gt_path, aeif_path, aglif_path, _report_path, _trace_path = _write_minimal_inputs(tmp_path)

    # When
    dataset = load_reproduction_dataset(
        gt_path,
        aeif_path,
        aglif_path,
        cell_order=("Pyramidal",),
    )
    target = dataset.targets["Pyramidal"]
    aeif = dataset.fits["AEIF"]["Pyramidal"]
    aglif = dataset.fits["A-GLIF"]["Pyramidal"]

    # Then
    assert aeif.rates_hz is not None
    assert fit_metrics(target, aeif).rate_rmse_z == pytest.approx(0.6455, rel=1.0e-3)
    assert aglif.rates_hz is None
    assert aglif.passive is not None
    assert aglif.passive.rin_mohm == pytest.approx(100.0)
    assert aglif.passive.tau_ms == pytest.approx(10.0)


def test_load_reproduction_dataset_uses_aglif_replay_report_when_present(
    tmp_path: Path,
) -> None:
    # Given
    gt_path, aeif_path, aglif_path, report_path, _trace_path = _write_minimal_inputs(tmp_path)

    # When
    dataset = load_reproduction_dataset(
        gt_path,
        aeif_path,
        aglif_path,
        aglif_report_path=report_path,
        cell_order=("Pyramidal",),
    )
    aglif = dataset.fits["A-GLIF"]["Pyramidal"]

    # Then
    assert aglif.rates_hz is not None
    assert aglif.rates_hz.tolist() == [0.0, 9.0, 21.0]
    assert aglif.passed is True
    assert aglif.protocol == "test-replay"


def test_fi_grid_writes_nonempty_png(tmp_path: Path) -> None:
    # Given
    gt_path, aeif_path, aglif_path, report_path, _trace_path = _write_minimal_inputs(tmp_path)
    dataset = load_reproduction_dataset(
        gt_path,
        aeif_path,
        aglif_path,
        aglif_report_path=report_path,
        cell_order=("Pyramidal",),
    )

    # When
    fig = create_fi_grid(dataset)
    try:
        saved = save_figure(fig, tmp_path / "figure", ("png",), dpi=80)
    finally:
        plt.close(fig)

    # Then
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


def test_count_stats_quantify_spike_count_reproduction(tmp_path: Path) -> None:
    # Given
    gt_path, aeif_path, aglif_path, report_path, _trace_path = _write_minimal_inputs(tmp_path)
    dataset = load_reproduction_dataset(
        gt_path,
        aeif_path,
        aglif_path,
        aglif_report_path=report_path,
        cell_order=("Pyramidal",),
    )

    # When
    stats = count_stats_for_dataset(dataset)
    aeif = stats[("AEIF", "Pyramidal")]
    aglif = stats[("A-GLIF", "Pyramidal")]

    # Then
    assert aeif.count_rmse_z == pytest.approx(0.6455, rel=1.0e-3)
    assert aeif.max_abs_count_delta == pytest.approx(1.2)
    assert 0.0 <= aeif.chi_square_p <= 1.0
    assert aglif.count_rmse_z == pytest.approx(0.3227, rel=1.0e-3)
    assert aglif.max_abs_count_delta == pytest.approx(0.6)


def test_response_trace_report_writes_nonempty_png(tmp_path: Path) -> None:
    # Given
    _gt_path, _aeif_path, _aglif_path, _report_path, trace_path = _write_minimal_inputs(tmp_path)
    traces = load_response_traces(trace_path, cell_order=("Pyramidal",))

    # When
    fig = create_response_figure(traces)
    try:
        saved = save_figure(fig, tmp_path / "response", ("png",), dpi=80)
    finally:
        plt.close(fig)

    # Then
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0


def _write_minimal_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    gt_path = tmp_path / "gt.json"
    aeif_path = tmp_path / "aeif.json"
    aglif_path = tmp_path / "aglif.json"
    report_path = tmp_path / "aglif_report.json"
    trace_path = tmp_path / "trace_report.json"

    gt_path.write_text(json.dumps({"Pyramidal": _target_record()}), encoding="utf-8")
    aeif_path.write_text(json.dumps({"Pyramidal": _aeif_record()}), encoding="utf-8")
    aglif_path.write_text(json.dumps({"Pyramidal": _aglif_record()}), encoding="utf-8")
    report_path.write_text(json.dumps({"Pyramidal": _aglif_report()}), encoding="utf-8")
    trace_path.write_text(json.dumps({"Pyramidal": _trace_report()}), encoding="utf-8")
    return gt_path, aeif_path, aglif_path, report_path, trace_path


def _target_record() -> dict[str, object]:
    return {
        "currents_nA": [0.1, 0.2, 0.3],
        "rates_hz": [0.0, 10.0, 20.0],
        "rheobase_nA": 0.2,
        "E_L": -65.0,
        "Rin": 100.0,
        "tau_m": 10.0,
        "sag": 1.0,
        "sigma": {
            "rates_hz": [2.0, 2.0, 4.0],
            "Rin": 15.0,
            "tau_m": 1.5,
            "E_L": 2.0,
            "sag": 1.0,
        },
    }


def _aeif_record() -> dict[str, object]:
    return {
        "loss": 0.5,
        "validation": {
            "passed": True,
            "median_z": 0.2,
            "max_z": 0.5,
            "hard_fails": [],
            "nest_rates_hz": [0.0, 12.0, 18.0],
            "target_rates_hz": [0.0, 10.0, 20.0],
            "nest_passive": {"Rin": 90.0, "tau_m": 11.0, "E_L": -65.0, "sag": 1.2},
        },
    }


def _aglif_record() -> dict[str, object]:
    return {
        "V_th": -50.0,
        "E_L": -65.0,
        "C_m": 100.0,
        "tau_m": 10.0,
        "k_adap": 0.0,
        "k1": 0.01,
        "k2": 0.1,
        "A1": 0.0,
        "A2": 20.0,
        "I_e": 0.0,
        "V_peak": -45.0,
        "V_reset": -70.0,
        "t_ref": 2.0,
        "loss": 0.25,
    }


def _aglif_report() -> dict[str, object]:
    return {
        "rates_hz": [0.0, 9.0, 21.0],
        "count_window_ms": 600.0,
        "passive": {"Rin": 100.0, "tau_m": 10.0, "E_L": -65.0, "sag": 1.0},
        "passed": True,
        "median_z": 0.1,
        "max_z": 0.4,
        "hard_fails": [],
        "protocol": "test-replay",
    }


def _trace_report() -> dict[str, object]:
    return {
        "current_nA": 0.2,
        "current_ratio": 1.0,
        "time_ms": [0.0, 199.9, 200.0, 201.0, 202.0, 800.0, 900.0],
        "voltage_mV": {
            "GT": [-65.0, -65.0, -55.0, 25.0, -70.0, -64.0, -65.0],
            "AEIF": [-65.0, -65.0, -54.0, 20.0, -68.0, -64.5, -65.0],
            "A-GLIF": [-65.0, -65.0, -53.0, 18.0, -67.0, -64.2, -65.0],
        },
        "spike_times_ms": {
            "GT": [201.0],
            "AEIF": [201.0],
            "A-GLIF": [201.0],
        },
    }
