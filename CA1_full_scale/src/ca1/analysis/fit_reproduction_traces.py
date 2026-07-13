from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ca1.analysis.fit_reproduction_data import (
    json_field,
    json_mapping,
    json_number,
    load_targets,
    read_json_mapping,
)
from ca1.analysis.fit_reproduction_schema import (
    CELL_ORDER,
    CurveName,
    FloatArray,
    JsonValue,
    ResponseTrace,
    TargetCell,
)

_CURVES: tuple[CurveName, ...] = ("GT", "AEIF", "A-GLIF")
_DELAY_MS = 200.0
_DUR_MS = 600.0
_TSTOP_MS = 900.0
_E_REV = [0.0, 0.0, -60.0, -60.0, -90.0]
_TAU_RISE = [0.1, 0.8, 0.25, 1.0, 30.0]
_TAU_DECAY = [1.5, 5.0, 6.0, 15.0, 100.0]
_AGLIF_FIELDS = (
    "V_th",
    "E_L",
    "C_m",
    "tau_m",
    "k_adap",
    "k1",
    "k2",
    "A1",
    "A2",
    "I_e",
    "V_peak",
    "V_reset",
    "t_ref",
)
_AEIF_FIELDS = (
    "V_th",
    "Delta_T",
    "V_reset",
    "t_ref",
    "a",
    "b",
    "tau_w",
    "V_peak",
    "g_L",
    "C_m",
    "E_L",
    "I_e",
)


def load_response_traces(path: Path, cell_order: tuple[str, ...] = CELL_ORDER) -> dict[str, ResponseTrace]:
    raw = read_json_mapping(path)
    traces: dict[str, ResponseTrace] = {}
    for cell_name in cell_order:
        record = json_mapping(json_field(raw, cell_name), cell_name)
        time_ms = np.asarray(_number_list(json_field(record, "time_ms"), "time_ms"))
        voltages = _curve_arrays(json_mapping(json_field(record, "voltage_mV"), "voltage_mV"))
        spike_times = _curve_arrays(json_mapping(json_field(record, "spike_times_ms"), "spike_times_ms"))
        traces[cell_name] = ResponseTrace(
            cell_name=cell_name,
            current_nA=json_number(json_field(record, "current_nA"), "current_nA"),
            current_ratio=json_number(json_field(record, "current_ratio"), "current_ratio"),
            time_ms=time_ms,
            voltages_mV=voltages,
            spike_times_ms=spike_times,
        )
    return traces


def build_response_trace_report(
    gt_path: Path,
    aeif_path: Path,
    aglif_path: Path,
    cell_order: tuple[str, ...] = CELL_ORDER,
) -> dict[str, JsonValue]:
    targets = load_targets(gt_path, cell_order)
    aeif_raw = read_json_mapping(aeif_path)
    aglif_raw = read_json_mapping(aglif_path)
    from ca1.params import groundtruth as gt

    neuron_h = gt.neuron_session()
    current_by_cell = {
        cell_name: float(targets[cell_name].currents_nA[targets[cell_name].peak_index])
        for cell_name in cell_order
    }
    aglif = _aglif_traces(aglif_raw, targets, current_by_cell, cell_order)
    report: dict[str, JsonValue] = {}
    for cell_name in cell_order:
        target = targets[cell_name]
        current_nA = current_by_cell[cell_name]
        gt_time, gt_voltage = _gt_trace(neuron_h, cell_name, current_nA)
        aeif_time, aeif_voltage = _aeif_trace(
            json_mapping(json_field(aeif_raw, cell_name), f"AEIF:{cell_name}"),
            current_nA,
        )
        time_ms = _common_time(gt_time, aeif_time, aglif[cell_name][0])
        report[cell_name] = {
            "current_nA": current_nA,
            "current_ratio": current_nA / target.rheobase_nA,
            "time_ms": [float(value) for value in time_ms],
            "voltage_mV": {
                "GT": _interp_json(gt_time, gt_voltage, time_ms),
                "AEIF": _interp_json(aeif_time, aeif_voltage, time_ms),
                "A-GLIF": _interp_json(aglif[cell_name][0], aglif[cell_name][1], time_ms),
            },
            "spike_times_ms": {
                "GT": _spike_json(gt_time, gt_voltage),
                "AEIF": _spike_json(aeif_time, aeif_voltage),
                "A-GLIF": [float(value) for value in aglif[cell_name][2]],
            },
        }
    return report


def write_response_trace_report(report: dict[str, JsonValue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _gt_trace(h, cell_name: str, current_nA: float) -> tuple[FloatArray, FloatArray]:
    from ca1.params import groundtruth as gt

    template = gt.CELL_TEMPLATES[cell_name]
    h.load_file(str(gt._MODELDB / "cells" / f"class_{template}.hoc"))
    cell = getattr(h, template)(0, 0, 0)
    soma = gt._soma(cell)
    time_ms, voltage_mV = gt._run(h, soma, current_nA)
    return np.asarray(time_ms, dtype=float), np.asarray(voltage_mV, dtype=float)


def _aeif_trace(record: dict[str, JsonValue], current_nA: float) -> tuple[FloatArray, FloatArray]:
    from ca1.params.forward import nest_cpu_trace

    params = {key: json_number(json_field(record, key), key) for key in _AEIF_FIELDS}
    time_ms, voltage_mV = nest_cpu_trace(params, current_nA, resolution=0.1)
    return np.asarray(time_ms, dtype=float), np.asarray(voltage_mV, dtype=float)


def _aglif_traces(
    raw: dict[str, JsonValue],
    targets: dict[str, TargetCell],
    current_by_cell: dict[str, float],
    cell_order: tuple[str, ...],
) -> dict[str, tuple[FloatArray, FloatArray, FloatArray]]:
    import nestgpu as ngpu

    ngpu.SetKernelStatus("verbosity_level", 0)
    nodes = ngpu.Create("user_m1", len(cell_order), len(_E_REV))
    ngpu.SetStatus(nodes, {"E_rev": _E_REV, "tau_rise": _TAU_RISE, "tau_decay": _TAU_DECAY})
    for field in _AGLIF_FIELDS:
        values = [
            json_number(json_field(json_mapping(json_field(raw, cell), cell), field), field)
            for cell in cell_order
        ]
        ngpu.SetStatus(nodes, field, {"array": values})
    e_l = [targets[cell].passive.e_l_mv for cell in cell_order]
    ngpu.SetStatus(nodes, "V_m", {"array": e_l})
    ngpu.SetStatus(nodes, "I_adap", {"array": [0.0] * len(cell_order)})
    ngpu.SetStatus(nodes, "I_dep", {"array": [0.0] * len(cell_order)})
    ngpu.SetStatus(nodes, "refractory_step", {"array": [0.0] * len(cell_order)})
    ngpu.ActivateRecSpikeTimes(nodes, 4096)
    record = ngpu.CreateRecord("", ["V_m"] * len(cell_order), [nodes[i] for i in range(len(cell_order))], [0] * len(cell_order))
    ngpu.SetStatus(nodes, "I_e", {"array": [0.0] * len(cell_order)})
    ngpu.Simulate(_DELAY_MS)
    currents_pA = [current_by_cell[cell] * 1000.0 for cell in cell_order]
    ngpu.SetStatus(nodes, "I_e", {"array": currents_pA})
    ngpu.Simulate(_DUR_MS)
    ngpu.SetStatus(nodes, "I_e", {"array": [0.0] * len(cell_order)})
    ngpu.Simulate(_TSTOP_MS - _DELAY_MS - _DUR_MS)
    data = ngpu.GetRecordData(record)
    spike_times = ngpu.GetRecSpikeTimes(nodes)
    time_ms = np.asarray([row[0] for row in data], dtype=float)
    traces: dict[str, tuple[FloatArray, FloatArray, FloatArray]] = {}
    for idx, cell_name in enumerate(cell_order):
        voltage = np.asarray([row[idx + 1] for row in data], dtype=float)
        spikes = np.asarray([] if spike_times is None else spike_times[idx], dtype=float)
        traces[cell_name] = (time_ms, voltage, spikes)
    return traces


def _common_time(*times: FloatArray) -> FloatArray:
    start = max(float(time[0]) for time in times)
    stop = min(float(time[-1]) for time in times)
    return np.arange(start, stop + 0.05, 0.1)


def _interp_json(time_ms: FloatArray, voltage_mV: FloatArray, new_time: FloatArray) -> list[JsonValue]:
    return [float(value) for value in np.interp(new_time, time_ms, voltage_mV)]


def _spike_json(time_ms: FloatArray, voltage_mV: FloatArray) -> list[JsonValue]:
    mask = (time_ms >= _DELAY_MS) & (time_ms < _DELAY_MS + _DUR_MS)
    t_window = time_ms[mask]
    v_window = voltage_mV[mask]
    crossings = np.where((v_window[:-1] < 0.0) & (v_window[1:] >= 0.0))[0]
    return [float(t_window[idx]) for idx in crossings]


def _curve_arrays(record: dict[str, JsonValue]) -> dict[CurveName, FloatArray]:
    return {
        curve: np.asarray(_number_list(json_field(record, curve), curve), dtype=float)
        for curve in _CURVES
    }


def _number_list(value: JsonValue, label: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return [json_number(item, label) for item in value]
