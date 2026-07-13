#!/usr/bin/env python3
"""Single-GPU parity checks for user_m3; never imports or initializes MPI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ca1.config import build_network_spec
from ca1.sim.aglif_dend import (
    aglif_dend_compartments,
    aglif_dend_status,
    cck_user_m3_status,
)
from ca1.sim.gpu_backend import _required_dendritic_ports

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/full_scale_3dtopo.yaml"
FI = (0.09375, 0.125, 0.15625, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.75)
SOURCE = (11.6667, 15.0, 18.3333, 21.6667, 28.3333, 33.3333, 40.0, 48.3333, 50.0, 53.3333, 0.0, 0.0, 0.0, 0.0)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _guard() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES=1 exactly")
    mpi_size = int(os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    if mpi_size != 1:
        raise RuntimeError("user_m3 verification forbids MPI")


def _cell_status() -> dict[str, float]:
    status = aglif_dend_status("CCK_Basket")
    status.update(cck_user_m3_status(status["E_L"]))
    return status


def _configure(
    ngpu: Any, nodes: Any, spec: Any, *, source_refit_domains: bool = False
) -> None:
    status = _cell_status()
    receptors = spec.receptors_for_post("CCK_Basket")
    compartments = aglif_dend_compartments(
        receptors.names, "CCK_Basket",
        _required_dendritic_ports(spec, "CCK_Basket"),
        spec.source_location_transfer_table,
    )
    if source_refit_domains:
        compartments = [
            0.0 if name.startswith(("AMPA_fast", "AMPA_slow")) else domain
            for name, domain in zip(receptors.names, compartments, strict=True)
        ]
    ngpu.SetStatus(nodes, status)
    ngpu.SetStatus(nodes, {
        "E_rev": list(receptors.E_rev), "tau_rise": list(receptors.tau_rise),
        "tau_decay": list(receptors.tau_decay), "compartment": compartments,
    })
    ngpu.SetStatus(nodes, {
        "V_m": status["E_L"], "V_d": status["E_L"], "V_dist": status["E_L"],
        "I_adap": 0.0, "I_dep": 0.0, "h": status["h"], "refractory_step": 0.0,
    })


def _quantized_spike_generator_times(
    event_times_ms: np.ndarray, dt_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a Poisson realization to NEST-GPU's dt grid without losing events.

    Times are rounded to the nearest integer time slot, sorted, and duplicate
    slots are represented through ``spike_gen_mul``.  Slot zero is moved to the
    first future slot, matching the real afferent-injection path's strictly
    positive, sorted, distinct spike-generator grid.  Thus only the required
    dt-grid timing quantization occurs; event multiplicity is preserved exactly.
    """
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be positive and finite, got {dt_ms}")
    raw = np.asarray(event_times_ms, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("event_times_ms must be a finite nonnegative 1-D array")
    if raw.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    slots = np.maximum(1, np.rint(raw / dt_ms).astype(np.int64))
    unique_slots, multiplicities = np.unique(slots, return_counts=True)
    times = unique_slots.astype(np.float64) * dt_ms
    # Decimal dt values are not exactly representable as binary floats.  At
    # large timestamps, subtraction can therefore land one ulp below dt even
    # for adjacent integer slots.  Move only those representations upward by
    # the minimum float increment accepted by the strict NEST-GPU check.
    for index in range(1, times.size):
        if times[index] - times[index - 1] < dt_ms:
            times[index] = np.nextafter(times[index - 1] + dt_ms, np.inf)
    if int(multiplicities.sum()) != int(raw.size):
        raise RuntimeError("dt-grid quantization lost barrage events")
    if times.size > 1 and np.any(np.diff(unique_slots) < 1):
        raise RuntimeError("quantized barrage slots are not strictly increasing")
    if times.size > 1 and np.any(np.diff(times) < dt_ms):
        raise RuntimeError("quantized barrage times are closer than dt")
    return times, multiplicities.astype(np.int64, copy=False)


def _barrage_materials() -> tuple[Any, list[Any], dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    refit_script = _load("_user_m3_gpu_refit", ROOT / "scripts/cck_sca_refit.py")
    refit = json.loads((ROOT / "results/cck_sca_refit_candidate.json").read_text())
    rows = refit_script._barrage_rows(CONFIG)["CCK_Basket"]
    candidates = refit_script._barrage_candidate_rows(refit["excitatory_transfer"])
    return refit_script, rows, candidates


def validate_barrage_times(dt: float, conductance_scale: float) -> dict[str, Any]:
    """CPU-only proof that every injected barrage train meets the GPU constraint."""
    refit_script, rows, _candidates = _barrage_materials()
    reports: list[dict[str, Any]] = []
    for row in rows:
        schedule = refit_script.BARRAGE.poisson_schedule(
            row, 11000.0 - row.source.delay_ms, 20260712, 1
        )
        raw = schedule.event_times_ms
        times, multiplicities = _quantized_spike_generator_times(raw, dt)
        slots = np.rint(times / dt).astype(np.int64)
        ordered = bool(times.size < 2 or np.all(np.diff(slots) >= 1))
        spacing_ok = bool(
            times.size < 2
            or np.all(np.diff(times) >= dt)
        )
        events_preserved = int(multiplicities.sum()) == int(raw.size)
        if not ordered or not spacing_ok or not events_preserved:
            raise RuntimeError(f"invalid barrage spike-generator array for {row.row_id}")
        reports.append({
            "row": row.row_id,
            "raw_event_count": int(raw.size),
            "quantized_slot_count": int(times.size),
            "multiplicity_sum": int(multiplicities.sum()),
            "strictly_ordered_slots": ordered,
            "spacing_at_least_dt": spacing_ok,
            "events_preserved": events_preserved,
        })
    return {
        "cpu_only": True,
        "dt_ms": dt,
        "conductance_scale": conductance_scale,
        "quantization": (
            "nearest dt grid; slot zero -> first future slot; duplicate slots use "
            "spike_gen_mul; sub-ulp upward representation only when binary-float "
            "subtraction would otherwise be < dt"
        ),
        "rows": reports,
        "all_arrays_valid": True,
        "total_events": sum(item["raw_event_count"] for item in reports),
        "total_quantized_slots": sum(item["quantized_slot_count"] for item in reports),
    }


def current_ladder(ngpu: Any, dt: float, spec: Any) -> dict[str, Any]:
    receptors = spec.receptors_for_post("CCK_Basket")
    cells = []
    for current in FI:
        node = ngpu.Create("user_m3", 1, receptors.n_ports())
        _configure(ngpu, node, spec)
        ngpu.ActivateRecSpikeTimes(node, 4096)
        cells.append(node)
    record = ngpu.CreateRecord(
        "", [name for _ in cells for name in ("V_m", "h")],
        [int(node[0]) for node in cells for _ in range(2)], [0] * (2 * len(cells)),
    )
    ngpu.Simulate(200.0)
    for node, current in zip(cells, FI, strict=True):
        ngpu.SetStatus(node, {"I_e": current * 1000.0})
    ngpu.Simulate(1200.0)
    for node in cells:
        ngpu.SetStatus(node, {"I_e": 0.0})
    ngpu.Simulate(400.0)
    for node in cells:
        ngpu.SetStatus(node, {"I_e": 250.0})
    ngpu.Simulate(600.0)
    rates = []
    recovery = []
    for node in cells:
        raw = ngpu.GetRecSpikeTimes(node)
        spikes = np.asarray([] if raw is None or raw[0] is None else raw[0], dtype=float)
        rates.append(float(np.count_nonzero((spikes >= 800.0) & (spikes < 1400.0)) / 0.6))
        recovery.append(int(np.count_nonzero(spikes >= 1800.0)))
    passed = [abs(got - want) <= max(2.0, 0.2 * want) for got, want in zip(rates, SOURCE, strict=True)]
    if not all(passed) or recovery[FI.index(0.625)] <= 0:
        raise RuntimeError(f"GPU f-I/recovery gate failed: rates={rates}, recovery={recovery}")
    trace = np.asarray(ngpu.GetRecordData(record), dtype=float)
    return {"dt_ms": dt, "rates_hz": rates, "passed": passed,
            "blocked_cell_recovery_spikes": recovery[FI.index(0.625)],
            "record_rows": int(trace.shape[0])}


def barrage(ngpu: Any, dt: float, spec: Any, conductance_scale: float) -> dict[str, Any]:
    refit_script, rows, candidates = _barrage_materials()
    receptors = spec.receptors_for_post("CCK_Basket")
    cell = ngpu.Create("user_m3", 1, receptors.n_ports())
    # The validated barrage is Option-2 applied on top of the source-grounded
    # CCK intrinsic/transfer refit candidate, never a deployed configuration.
    _configure(ngpu, cell, spec, source_refit_domains=True)
    ngpu.ActivateRecSpikeTimes(cell, 4096)
    for row in rows:
        schedule = refit_script.BARRAGE.poisson_schedule(row, 11000.0 - row.source.delay_ms, 20260712, 1)
        source_times = schedule.event_times_ms
        if not source_times.size:
            continue
        times, counts = _quantized_spike_generator_times(source_times, dt)
        source = ngpu.Create("spike_generator", 1)
        ngpu.SetStatus(source, {"spike_times": times.tolist(), "spike_gen_mul": counts.tolist()})
        weight, _allocation = refit_script.BARRAGE._transfer_for_arm(row, "candidate_user_m2", candidates)
        ngpu.Connect(source, cell, {"rule": "all_to_all"}, {
            "weight": conductance_scale * weight * row.source.synapses_per_connection,
            "delay": row.source.delay_ms, "receptor": receptors.port_index(row.source.receptor),
        })
    record = ngpu.CreateRecord("", ["V_m", "h"], [int(cell[0]), int(cell[0])], [0, 0])
    ngpu.Simulate(11000.0)
    raw = ngpu.GetRecSpikeTimes(cell)
    spikes = np.asarray([] if raw is None or raw[0] is None else raw[0], dtype=float)
    trace = np.asarray(ngpu.GetRecordData(record), dtype=float)
    window = trace[:, 0] >= 1000.0
    rate = float(np.count_nonzero(spikes >= 1000.0) / 10.0)
    plateau = float(trace[window, 1].mean())
    expected = 46.3333333333 if conductance_scale == 0.5 else 0.0
    tolerance = max(2.0, 0.2 * expected)
    if abs(rate - expected) > tolerance or (expected == 0.0 and plateau <= -40.0):
        raise RuntimeError(f"GPU barrage gate failed: rate={rate}, plateau={plateau}")
    return {"dt_ms": dt, "conductance_scale": conductance_scale,
            "source_rate_hz": expected, "rate_hz": rate, "mean_v_mV": plateau,
            "mean_h": float(trace[window, 2].mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, required=True, choices=(0.05, 0.025))
    parser.add_argument("--mode", required=True, choices=("fi", "barrage", "barrage-times"))
    parser.add_argument("--conductance-scale", type=float, default=1.0,
                        choices=(0.5, 0.75, 1.0, 1.25))
    args = parser.parse_args()
    if args.mode == "barrage-times":
        print(json.dumps(validate_barrage_times(args.dt, args.conductance_scale), indent=2))
        return
    _guard()
    import nestgpu as ngpu  # CUDA initialization intentionally occurs only after guard
    ngpu.SetRandomSeed(20260712)
    ngpu.SetTimeResolution(args.dt)
    spec = build_network_spec(CONFIG, scale=1.0, seed=12345)
    result = (current_ladder(ngpu, args.dt, spec) if args.mode == "fi"
              else barrage(ngpu, args.dt, spec, args.conductance_scale))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
