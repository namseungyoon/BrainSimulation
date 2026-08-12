#!/usr/bin/env python3
"""Source-grounded CPU diagnosis of CCK_Basket/SCA over-firing.

This is a read-only analysis driver.  It reuses the paired unit-transfer,
full-converging barrage, and exact saved-event clamp implementations.  It does
not alter a model parameter or build/run a network.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ca1.config import build_network_spec
from ca1.extract.modeldb_tables import extract_connectivity
from ca1.params.groundtruth import CELL_TEMPLATES, cell_ground_truth, neuron_session, passive_features
from ca1.sim.aglif_dend import aglif_dend_status


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("CCK_Basket", "SCA")
SEEDS = (20260712, 20260713, 20260714)
DTS_MS = (0.05, 0.025)
CONFIG = ROOT / "configs" / "full_scale_3dtopo.yaml"
EDGES = ROOT / "results" / "edges_fullscale.h5"
RUN = ROOT / "results" / "fullscale_3dtopo_theta.h5"
OUTPUT = ROOT / "results" / "cck_sca_diagnosis.json"


def _load(name: str, filename: str) -> Any:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = _load("_cck_sca_paired", "paired_transfer_audit.py")
BARRAGE = _load("_cck_sca_barrage", "full_converging_barrage.py")
CLAMP = _load("_cck_sca_clamp", "exact_network_clamp_replay.py")


def _source_intrinsic_task(cell: str) -> tuple[str, dict[str, Any]]:
    h = neuron_session()
    return cell, cell_ground_truth(h, CELL_TEMPLATES[cell])


def _reduced_trace(
    cell: str,
    current_nA: float,
    dt_ms: float,
    duration_ms: float = 900.0,
    status_overrides: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = aglif_dend_status(cell)
    if status_overrides:
        s.update({key: float(value) for key, value in status_overrides.items()})
        if "g_c_scale" in status_overrides:
            membrane_conductance = float(s["C_m"]) / float(s["tau_m"])
            s["g_c"] = 2.0 * membrane_conductance * float(status_overrides["g_c_scale"])
        if "dist_coupling_ratio" in status_overrides or "g_c_scale" in status_overrides:
            ratio = float(status_overrides.get("dist_coupling_ratio", 0.25))
            s["g_c_dist"] = float(s["g_c"]) * ratio
    c = float(s["C_m"]); cd = c * float(s["dend_C_frac"]); cx = cd * float(s["dist_C_frac"])
    caps = np.asarray((c - cd, cd - cx, cx), dtype=float)
    state = np.asarray((float(s["E_L"]),) * 3 + (0.0, 0.0), dtype=float)
    n_steps = int(round(duration_ms / dt_ms)); times = np.arange(1, n_steps + 1) * dt_ms
    vm = np.empty(n_steps, dtype=float); spikes: list[float] = []
    refractory = 0; refr_steps = int(round(float(s["t_ref"]) / dt_ms))

    def derivative(x: np.ndarray, injected_pA: float, clamped: bool) -> np.ndarray:
        v0, v1, v2, ia, idep = x
        out = np.empty(5, dtype=float)
        out[0] = 0.0 if clamped else (
            -(caps[0] / float(s["tau_m"])) * (v0 - float(s["E_L"]))
            + float(s["g_c"]) * (v1 - v0) - ia + idep + float(s["I_e"]) + injected_pA
        ) / caps[0]
        out[1] = (
            -(caps[1] / float(s["tau_m"])) * float(s["dend_leak_scale"]) * (v1 - float(s["E_L"]))
            + float(s["g_c"]) * (v0 - v1) + float(s["g_c_dist"]) * (v2 - v1)
        ) / caps[1]
        out[2] = (
            -(caps[2] / float(s["tau_m"])) * float(s["dist_leak_scale"]) * (v2 - float(s["E_L"]))
            + float(s["g_c_dist"]) * (v1 - v2)
        ) / caps[2]
        out[3] = float(s["k_adap"]) * (v0 - float(s["E_L"])) - float(s["k2"]) * ia
        out[4] = -float(s["k1"]) * idep
        return out

    for i, now in enumerate(times):
        injected = current_nA * 1000.0 if 200.0 <= now < 800.0 else 0.0
        clamped = refractory > 0
        k1 = derivative(state, injected, clamped)
        k2 = derivative(state + 0.5 * dt_ms * k1, injected, clamped)
        k3 = derivative(state + 0.5 * dt_ms * k2, injected, clamped)
        k4 = derivative(state + dt_ms * k3, injected, clamped)
        state += dt_ms * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        if clamped:
            state[0] = float(s["V_reset"]); refractory -= 1
        elif state[0] >= float(s["V_th"]):
            spikes.append(now); state[0] = float(s["V_reset"])
            state[3] += float(s["A2"]); state[4] = float(s["A1"]); refractory = refr_steps
        vm[i] = state[0]
    return times, vm, np.asarray(spikes)


def _reduced_intrinsic(
    cell: str,
    currents: Sequence[float],
    dt_ms: float,
    status_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    t, v, _ = _reduced_trace(cell, -0.05, dt_ms, status_overrides=status_overrides)
    pas = passive_features(t, v, -0.05, 200.0, 800.0)
    rates = []
    for current in currents:
        _t, _v, spikes = _reduced_trace(
            cell, float(current), dt_ms, status_overrides=status_overrides
        )
        rates.append(float(np.sum((spikes >= 200.0) & (spikes < 800.0)) / 0.6))

    def fires(current: float) -> bool:
        _t, _v, spikes = _reduced_trace(
            cell, current, dt_ms, 600.0, status_overrides=status_overrides
        )
        return bool(np.any((spikes >= 200.0) & (spikes < 500.0)))

    lo, hi = 0.0, 0.8
    if fires(hi):
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if fires(mid): hi = mid
            else: lo = mid
    return {"dt_ms": dt_ms, "Rin_MOhm": float(pas["Rin"]), "tau_m_ms": float(pas["tau_m"]),
            "rheobase_nA": float(hi), "currents_nA": list(currents), "rates_hz": rates}


def intrinsic_audit() -> dict[str, Any]:
    with mp.get_context("spawn").Pool(len(TARGETS)) as pool:
        source = dict(pool.map(_source_intrinsic_task, TARGETS))
    cells = {}
    for cell in TARGETS:
        src = source[cell]
        reduced = [_reduced_intrinsic(cell, src["currents_nA"], dt) for dt in DTS_MS]
        primary = next(x for x in reduced if x["dt_ms"] == 0.025)
        cells[cell] = {"source_neuron": src, "deployed_user_m2": reduced,
            "percent_of_source": {
                "Rin": 100.0 * primary["Rin_MOhm"] / src["Rin"],
                "tau_m": 100.0 * primary["tau_m_ms"] / src["tau_m"],
                "rheobase": 100.0 * primary["rheobase_nA"] / src["rheobase_nA"],
            }}
    return {"protocol": "fresh source NEURON extraction and independent deployed user_m2 CPU RK4 replay", "cells": cells}


def _transfer_source_task(task: tuple[Any, float, int]) -> dict[str, Any]:
    row, dt, n_draws = task
    return PAIRED._run_source_row(row, dt, n_draws, PAIRED.LOCATION_SEED)


def _transfer_source_process(task: tuple[Any, float, int], queue: Any) -> None:
    try:
        queue.put((True, _transfer_source_task(task)))
    except BaseException as exc:
        queue.put((False, repr(exc)))


def transfer_audit(config: Path, n_draws: int = 32) -> dict[str, Any]:
    rows = PAIRED.configured_excitatory_rows(config, TARGETS)
    tasks = [(row, dt, n_draws) for row in rows for dt in DTS_MS]
    context = mp.get_context("spawn")
    captures = []
    for start in range(0, len(tasks), 4):
        queues = [context.Queue() for _ in tasks[start:start + 4]]
        processes = [context.Process(target=_transfer_source_process, args=(task, queue))
                     for task, queue in zip(tasks[start:start + 4], queues, strict=True)]
        for process in processes: process.start()
        batch = [queue.get() for queue in queues]
        for process in processes: process.join()
        for (ok, result), process in zip(batch, processes, strict=True):
            if not ok or process.exitcode != 0:
                raise RuntimeError(f"transfer source process failed: {result}")
            captures.append(result)
    records = []
    for (row, dt, _n_draws), capture in zip(tasks, captures, strict=True):
        source_summary = capture["source_summary"]
        # The HOC template's Vrest is an initialization constant, not the
        # deployed reduced cell's equilibrium.  Starting user_m2 at its own
        # immutable E_L prevents pre-event drift from contaminating the EPSP.
        reduced_rest = float(aglif_dend_status(row.post)["E_L"])
        deployed = PAIRED.run_user_m2_cpu(row, row.deployed_domain, dt, reduced_rest)
        peak = 100.0 * deployed.epsp_peak_mV / source_summary["epsp_peak_mV"]["median"]
        charge = 100.0 * abs(deployed.clamp_charge_nA_ms) / abs(source_summary["clamp_charge_nA_ms"]["median"])
        records.append({"row": f"{row.pre}->{row.post}", "dt_ms": dt, "contract": asdict(row),
            "source_summary": source_summary, "deployed_measurement": asdict(deployed),
            "reduced_initial_mV": reduced_rest,
            "peak_percent_of_source": peak, "charge_percent_of_source": charge,
            "over_transferred": bool(peak > 115.0 or charge > 110.0)})
    return {"n_location_draws": n_draws, "location_seed": PAIRED.LOCATION_SEED, "rows": records}


def _barrage_rows(config: Path, recurrent_proxy_hz: float) -> dict[str, list[Any]]:
    spec = build_network_spec(config, scale=1.0, seed=PAIRED.LOCATION_SEED)
    raw = extract_connectivity(index=spec.conndata_index, cellnumbers_index=spec.cellnumbers_index,
                               count_mode=spec.conndata_count_mode)
    result = {cell: [] for cell in TARGETS}
    for source in PAIRED.configured_excitatory_rows(config, TARGETS):
        key = f"{source.pre}_to_{source.post}"
        entry = raw["afferents" if source.kind == "aff" else "projections"][key]
        if source.kind == "aff":
            indegree = int(round(float(entry["synapses_per_cell"]) / source.synapses_per_connection)); rate = 0.65
        else:
            indegree = int(round(float(entry["indegree"]))); rate = recurrent_proxy_hz
        result[source.post].append(BARRAGE.BarrageRow(source, indegree, rate))
    for rows in result.values(): rows.sort(key=lambda x: x.source.pre)
    return result


def barrage_audit(config: Path, recurrent_proxy_hz: float, processes: int) -> dict[str, Any]:
    rows = _barrage_rows(config, recurrent_proxy_hz)
    tasks = [(list(rows[cell]), seed, dt, 1000.0, 10000.0) for cell in TARGETS for seed in SEEDS for dt in DTS_MS]
    context = mp.get_context("spawn")
    with context.Pool(min(processes, len(tasks)), maxtasksperchild=1) as pool:
        source = pool.map(BARRAGE._source_task, tasks, chunksize=1)
    reduced_tasks = [(list(rows[cell]), "deployed_user_m2", seed, dt, 1000.0, 10000.0, {}, None)
                     for cell in TARGETS for seed in SEEDS for dt in DTS_MS]
    with context.Pool(min(processes, len(reduced_tasks))) as pool:
        reduced = pool.map(BARRAGE._reduced_task, reduced_tasks, chunksize=1)
    results = [*source, *reduced]
    metadata = []
    for cell in TARGETS:
        for row in rows[cell]:
            item = asdict(row.source)
            item.update({"row": row.row_id, "indegree_true_connections": row.indegree_true,
                         "rate_hz_per_afferent": row.rate_hz_per_afferent,
                         "aggregate_event_rate_hz": row.indegree_true * row.rate_hz_per_afferent})
            metadata.append(item)
    return {"protocol": {"afferent_rate_hz": 0.65, "recurrent_pyramidal_proxy_hz": recurrent_proxy_hz,
            "proxy_rationale": "same preregistered held-out 1 Hz proxy as the prior PING-cell barrage",
            "transient_ms": 1000.0, "measure_ms": 10000.0, "seeds": list(SEEDS), "dt_ms": list(DTS_MS)},
        "rows": metadata, "results": [asdict(x) for x in results],
        "summary": BARRAGE._summaries(results)}


def _spatial_panel(run: h5py.File, target: str, count: int = 10) -> list[int]:
    positions = np.asarray(run["cell_positions"][target], dtype=float); ids = np.arange(len(positions))
    order = ids[np.lexsort((ids, positions[:, 2], positions[:, 1], positions[:, 0]))]
    return [int(block[len(block)//2]) for block in np.array_split(order, count)]


def clamp_audit(config: Path, edges_path: Path, run_path: Path) -> dict[str, Any]:
    CLAMP.TARGETS = TARGETS
    spec = build_network_spec(config, scale=1.0, seed=12345)
    with h5py.File(edges_path, "r") as edges, h5py.File(run_path, "r") as run:
        desc = CLAMP.projections(edges); selected = {t: _spatial_panel(run, t) for t in TARGETS}
        duration_s = float(run["meta"].attrs["duration_s"]); aff_dt = float(run["meta"].attrs["dt_s"]) * 1000.0
        recurrent, aff_counts, detail, summaries = CLAMP.reconstruct_step2(
            edges, run, desc, selected, duration_s=duration_s, seed=12345,
            afferent_rate_hz=0.65, network_spec=spec)
        inhibitors = sorted({d.pre for d in desc if d.post in TARGETS and d.pre not in CLAMP.EXCITATORY})
        arms = ("all", "no_inhibition", *(f"drop_{x}" for x in inhibitors))
        replay = []
        with tempfile.TemporaryDirectory(prefix="cck-sca-aff-") as tmp:
            store = {s: CLAMP.build_afferent_slot_store(Path(tmp), s, c, duration_ms=duration_s*1000,
                     dt_ms=aff_dt, seed=12345, rate_hz=0.65) for s, c in aff_counts.items()}
            for target in TARGETS:
                for target_id in selected[target]:
                    for dt in DTS_MS:
                        replay.extend(CLAMP.replay_target(edges, run, desc, spec, store, target, target_id,
                            dt_ms=dt, duration_ms=duration_s*1000, afferent_dt_ms=aff_dt, arms=arms))
        alternate_counts = {s: CLAMP._afferent_counts(s, len(c), 0.65, duration_s, 12346) for s, c in aff_counts.items()}
        sensitivity = []
        with tempfile.TemporaryDirectory(prefix="cck-sca-aff-alt-") as tmp:
            store = {s: CLAMP.build_afferent_slot_store(Path(tmp), s, c, duration_ms=duration_s*1000,
                     dt_ms=aff_dt, seed=12346, rate_hz=0.65) for s, c in alternate_counts.items()}
            for target in TARGETS:
                for target_id in selected[target]:
                    sensitivity.extend(CLAMP.replay_target(edges, run, desc, spec, store, target, target_id,
                        dt_ms=0.025, duration_ms=duration_s*1000, afferent_dt_ms=aff_dt, arms=arms))
        network_rates = {t: sum(run["spikes"][t][str(i)].shape[0] for i in range(len(run["spikes"][t]))) /
                         (len(run["spikes"][t]) * duration_s) for t in TARGETS}
        return {"protocol": {"duration_s": duration_s, "selected_ids": selected, "arms": arms,
                "saved_afferent_seed": 12345, "alternate_afferent_seed": 12346, "dt_ms": list(DTS_MS)},
            "incoming_projection_summaries": summaries, "selected_target_details": detail,
            "source_population_rates_hz": {**{k: float(v.sum())/(len(v)*duration_s) for k,v in aff_counts.items()},
                                            **{k: float(v.sum())/(len(v)*duration_s) for k,v in recurrent.items()}},
            "recorded_network_rates_hz": network_rates, "per_cell": replay,
            "summary": CLAMP._aggregate_replays(replay), "seed_sensitivity_summary": CLAMP._aggregate_replays(sensitivity)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG); parser.add_argument("--edges", type=Path, default=EDGES)
    parser.add_argument("--run", type=Path, default=RUN); parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--processes", type=int, default=4); parser.add_argument("--recurrent-proxy-hz", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="reuse completed phase checkpoints from --output")
    parser.add_argument("--force-phase", choices=("intrinsic", "excitatory_transfer", "excitation_only_barrage", "exact_clamp"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.output.read_text()) if args.resume and args.output.exists() else {
        "schema": "cck-sca-diagnosis/v1", "provenance": {"cpu_only": True, "gpu_used": False,
        "mpi_used": False, "network_built": False, "parameters_changed": False, "table5_rate_tuning": False,
        "config": str(args.config), "edges": str(args.edges), "run": str(args.run)}}
    phases = (("intrinsic", intrinsic_audit),
              ("excitatory_transfer", lambda: transfer_audit(args.config)),
              ("excitation_only_barrage", lambda: barrage_audit(args.config, args.recurrent_proxy_hz, args.processes)),
              ("exact_clamp", lambda: clamp_audit(args.config, args.edges, args.run)))
    if args.force_phase:
        report.pop(args.force_phase, None)
    for name, function in phases:
        if name not in report:
            report[name] = function()
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{name} complete", flush=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
