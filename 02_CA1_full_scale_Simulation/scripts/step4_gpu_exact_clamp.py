#!/usr/bin/env python3
# pyright: basic, reportPrivateUsage=false
"""Single-GPU exact-event clamp for the Step-4 backend check.

The event graph and trains are reconstructed by the Step-2/3 implementation in
``exact_network_clamp_replay.py``.  This script changes only the postsynaptic
execution engine: the same source times are installed on NEST-GPU
``spike_generator`` nodes and delivered to isolated deployed ``user_m2`` cells.

No MPI, network population, fitted value, or deployed synaptic value is used as
a free parameter here.  Connection weights are the persisted per-contact weight
times persisted contact multiplicity.  NEST-GPU's user_m2 kernel applies its own
per-port beta ``g0`` normalization, exactly as gpu_backend does.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import numpy.typing as npt

from ca1.config import build_network_spec
from ca1.sim.aglif_dend import aglif_dend_compartments, aglif_dend_status
from ca1.sim.gpu_backend import _required_dendritic_ports
from ca1.sim.nestgpu_api import NestGpuModule, nestgpu_module


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "full_scale_3dtopo.yaml"
DEFAULT_EDGES = ROOT / "results" / "edges_fullscale.h5"
DEFAULT_RUN = ROOT / "results" / "fullscale_3dtopo_theta.h5"
DEFAULT_CPU_REPORT = ROOT / "results" / "clamp_replay.json"
DEFAULT_OUTPUT = ROOT / "results" / "step4_gpu_exact_clamp.json"
TARGETS = ("PV_Basket", "Bistratified", "O_LM")
ARMS = ("all", "no_inhibition", "drop_CCK")
DT_MS = (0.1, 0.05, 0.025)
EXCITATORY = frozenset(("CA3", "ECIII", "Pyramidal"))


@dataclass(frozen=True)
class InjectedRow:
    target_type: str
    target_id: int
    projection: str
    pre: str
    port: int
    release_component: int
    delay_ms: float
    weight_nS_per_contact: float
    contacts: int
    source_times_ms: npt.NDArray[np.float64]
    source_ids: npt.NDArray[np.int64]

    @property
    def connection_weight_nS(self) -> float:
        return self.weight_nS_per_contact * self.contacts


def _load_replay_module() -> Any:
    """Load the authoritative reconstruction lazily (it builds a Cython helper)."""
    path = ROOT / "scripts" / "exact_network_clamp_replay.py"
    spec = importlib.util.spec_from_file_location("_step4_exact_cpu_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _h5_float(value: object) -> float:
    """Narrow an h5py scalar attribute for runtime and static type checkers."""
    return float(np.asarray(value).item())


def _enabled(arm: str, pre: str) -> bool:
    if pre in EXCITATORY:
        return True
    if arm == "no_inhibition":
        return False
    if arm == "drop_CCK" and pre == "CCK_Basket":
        return False
    return True


def _selected_targets(report: Mapping[str, Any], requested: Sequence[str]) -> dict[str, int]:
    selected = {
        target: int(report["protocol"]["target_selection"][target]["selected_ids"][0])
        for target in TARGETS
    }
    for item in requested:
        try:
            target, raw_id = item.split(":", maxsplit=1)
            target_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f"--target must be TYPE:ID, got {item!r}") from exc
        if target not in TARGETS:
            raise ValueError(f"unsupported target type {target!r}; choose from {TARGETS}")
        allowed = report["protocol"]["target_selection"][target]["selected_ids"]
        if target_id not in allowed:
            raise ValueError(f"{target}[{target_id}] was not selected by Step 2/3")
        selected[target] = target_id
    return selected


def _row_release_components(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Recover release-component boundaries from Step-3's reset edge offsets."""
    component_by_projection: dict[str, int] = defaultdict(lambda: -1)
    result: list[int] = []
    for row in rows:
        projection = str(row["projection"])
        if int(row.get("edge_start", 0)) == 0:
            component_by_projection[projection] += 1
        result.append(component_by_projection[projection])
    return result


def construct_rows(
    replay: Any,
    edges: h5py.File,
    run: h5py.File,
    descriptors: Sequence[Any],
    spec: Any,
    afferent_store: Mapping[str, tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    selected: Mapping[str, int],
    afferent_dt_ms: float,
) -> list[InjectedRow]:
    """Construct exact source schedules using the Step-3 source/time functions."""
    result: list[InjectedRow] = []
    for target in TARGETS:
        target_id = int(selected[target])
        raw_rows = replay._target_rows(edges, descriptors, target, target_id, spec)
        # _target_rows did not need edge_start after slicing; restore it from
        # descriptors solely to identify release-component boundaries.
        enriched: list[dict[str, Any]] = []
        cursor = 0
        for descriptor in descriptors:
            if descriptor.post != target:
                continue
            for port in descriptor.ports:
                row = dict(raw_rows[cursor])
                row["edge_start"] = int(port["edge_start"])
                enriched.append(row)
                cursor += 1
        if cursor != len(raw_rows):
            raise RuntimeError(f"row descriptor mismatch for {target}[{target_id}]")
        components = _row_release_components(enriched)
        for row, component in zip(enriched, components, strict=True):
            times: list[npt.NDArray[np.float64]] = []
            ids: list[npt.NDArray[np.int64]] = []
            for source_id in row["sources"]:
                source_times = replay._source_times_ms(
                    str(row["pre"]), int(source_id), run, afferent_store, afferent_dt_ms
                )
                if source_times.size:
                    times.append(np.asarray(source_times, dtype=np.float64))
                    ids.append(np.full(source_times.size, int(source_id), dtype=np.int64))
            result.append(InjectedRow(
                target_type=target,
                target_id=target_id,
                projection=str(row["projection"]),
                pre=str(row["pre"]),
                port=int(row["port"]),
                release_component=int(component),
                delay_ms=float(row["delay_ms"]),
                weight_nS_per_contact=float(row["weight_nS"]),
                contacts=int(row["contacts"]),
                source_times_ms=np.concatenate(times) if times else np.empty(0, dtype=np.float64),
                source_ids=np.concatenate(ids) if ids else np.empty(0, dtype=np.int64),
            ))
    return result


def arrival_manifest(
    rows: Sequence[InjectedRow], cpu_report: Mapping[str, Any], duration_ms: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prove scheduled-event parity with persisted Step-2 selected-target counts."""
    expected = {
        (str(item["target_type"]), int(item["target_id"]), str(item["projection"])):
            int(item["delivered_connection_events"])
        for item in cpu_report["step2"]["selected_target_details"]
    }
    manifest: list[dict[str, Any]] = []
    component_counts: dict[tuple[str, int, str, int], int] = defaultdict(int)
    for row in rows:
        scheduled = int(row.source_times_ms.size)
        arrivals = row.source_times_ms + row.delay_ms
        in_window = int(np.count_nonzero((arrivals >= 0.0) & (arrivals < duration_ms)))
        key = (row.target_type, row.target_id, row.projection, row.release_component)
        component_counts[key] += scheduled
        manifest.append({
            "target_type": row.target_type,
            "target_id": row.target_id,
            "projection": row.projection,
            "pre": row.pre,
            "release_component": row.release_component,
            "port": row.port,
            "delay_ms": row.delay_ms,
            "weight_nS_per_contact": row.weight_nS_per_contact,
            "contacts": row.contacts,
            "connection_weight_nS": row.connection_weight_nS,
            "scheduled_source_events": scheduled,
            "in_window_arrivals": in_window,
        })
    parity: list[dict[str, Any]] = []
    for (target, target_id, projection, component), observed in sorted(component_counts.items()):
        cpu_count = expected[(target, target_id, projection)]
        parity.append({
            "target_type": target,
            "target_id": target_id,
            "projection": projection,
            "release_component": component,
            "cpu_step2_count": cpu_count,
            "gpu_scheduled_count": observed,
            "equal": observed == cpu_count,
        })
    failed = [item for item in parity if not item["equal"]]
    if failed:
        raise RuntimeError(f"Step-2 event-count parity failed: {failed[:3]}")
    return manifest, parity


def _aggregate_train(
    times_ms: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Encode coincident source spikes through spike_generator's multiplier path."""
    if times_ms.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    unique, counts = np.unique(times_ms, return_counts=True)
    return unique.astype(np.float64), counts.astype(np.float64)


def _write_injection_schedule(rows: Sequence[InjectedRow], path: Path) -> None:
    """Persist source identity and emission time for an independently auditable replay."""
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum([row.source_times_ms.size for row in rows], dtype=np.int64, out=offsets[1:])
    source_times = np.concatenate([row.source_times_ms for row in rows]) if rows else np.empty(0)
    source_ids = np.concatenate([row.source_ids for row in rows]) if rows else np.empty(0, dtype=np.int64)
    metadata = [{
        "target_type": row.target_type, "target_id": row.target_id,
        "projection": row.projection, "pre": row.pre, "port": row.port,
        "release_component": row.release_component, "delay_ms": row.delay_ms,
        "weight_nS_per_contact": row.weight_nS_per_contact, "contacts": row.contacts,
        "connection_weight_nS": row.connection_weight_nS,
    } for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, offsets=offsets, source_times_ms=source_times, source_ids=source_ids,
        rows_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )


def _arm_port_arrival_counts(
    rows: Sequence[InjectedRow], duration_ms: float,
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        in_window = int(np.count_nonzero(
            (row.source_times_ms + row.delay_ms >= 0.0)
            & (row.source_times_ms + row.delay_ms < duration_ms)
        ))
        for arm in ARMS:
            if _enabled(arm, row.pre):
                item = counts[(row.target_type, row.target_id, arm, row.port)]
                item[0] += int(row.source_times_ms.size)
                item[1] += in_window
    return [{
        "target_type": target, "target_id": target_id, "arm": arm, "port": port,
        "scheduled_source_events": value[0], "in_window_arrivals": value[1],
    } for (target, target_id, arm, port), value in sorted(counts.items())]


def _create_cells(ngpu: NestGpuModule, spec: Any) -> tuple[Any, list[dict[str, Any]]]:
    cells: list[Any] = []
    layout: list[dict[str, Any]] = []
    for target in TARGETS:
        receptors = spec.receptors_for_post(target)
        nodes = ngpu.Create("user_m2", len(ARMS), receptors.n_ports())
        status = aglif_dend_status(target)
        compartments = aglif_dend_compartments(
            receptors.names, target, _required_dendritic_ports(spec, target),
            spec.source_location_transfer_table, spec.aglif_receive_domain_overrides,
        )
        ngpu.SetStatus(nodes, status)
        ngpu.SetStatus(nodes, {
            "E_rev": list(receptors.E_rev),
            "tau_rise": list(receptors.tau_rise),
            "tau_decay": list(receptors.tau_decay),
            "compartment": list(compartments),
        })
        e_l = float(status["E_L"])
        ngpu.SetStatus(nodes, {
            "V_m": e_l, "V_d": e_l, "V_dist": e_l,
            "I_adap": 0.0, "I_dep": 0.0, "refractory_step": 0.0,
        })
        ngpu.ActivateRecSpikeTimes(nodes, 4096)
        cells.append(nodes)
        for arm_idx, arm in enumerate(ARMS):
            layout.append({"target_type": target, "arm": arm, "node": nodes[arm_idx]})
    return cells, layout


def run_gpu(
    spec: Any, rows: Sequence[InjectedRow], selected: Mapping[str, int],
    *, dt_ms: float, duration_ms: float, trace_path: Path,
) -> list[dict[str, Any]]:
    import nestgpu as raw_ngpu  # noqa: PLC0415 -- CUDA import must remain lazy

    ngpu = nestgpu_module(raw_ngpu)
    if any(name in os.environ for name in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE")):
        size = int(os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
        if size != 1:
            raise RuntimeError("Step 4 is single-GPU only; MPI size must be 1")
    ngpu.SetRandomSeed(48271)
    ngpu.SetTimeResolution(dt_ms)
    cells, layout = _create_cells(ngpu, spec)
    cells_by_target: dict[str, Any] = dict(zip(TARGETS, cells, strict=True))

    for row in rows:
        times, multipliers = _aggregate_train(row.source_times_ms)
        if not times.size:
            continue
        source = ngpu.Create("spike_generator", 1)
        ngpu.SetStatus(source, {
            "spike_times": times.tolist(),
            "spike_gen_mul": multipliers.tolist(),
        })
        targets = cells_by_target[row.target_type]
        for arm_idx, arm in enumerate(ARMS):
            if not _enabled(arm, row.pre):
                continue
            ngpu.Connect(source, targets[arm_idx:arm_idx + 1], {"rule": "all_to_all"}, {
                "weight": row.connection_weight_nS,
                "delay": row.delay_ms,
                "receptor": row.port,
            })

    var_names: list[str] = []
    record_nodes: list[int] = []
    ports: list[int] = []
    columns: list[dict[str, Any]] = []
    for cell in layout:
        target = str(cell["target_type"])
        node = int(cell["node"])
        receptors = spec.receptors_for_post(target)
        compartments = aglif_dend_compartments(
            receptors.names, target, _required_dendritic_ports(spec, target),
            spec.source_location_transfer_table, spec.aglif_receive_domain_overrides,
        )
        for variable in ("V_m", "V_d", "V_dist", "I_adap", "I_dep"):
            var_names.append(variable); record_nodes.append(node); ports.append(0)
            columns.append({**cell, "node": node, "variable": variable, "port": None})
        for port in range(receptors.n_ports()):
            var_names.append("g"); record_nodes.append(node); ports.append(port)
            columns.append({**cell, "node": node, "variable": "g", "port": port,
                "receptor": receptors.names[port], "compartment": float(compartments[port]),
                "E_rev_mV": float(receptors.E_rev[port])})
    record = ngpu.CreateRecord("", var_names, record_nodes, ports)
    ngpu.SetRecordStride(record, 1)
    ngpu.Simulate(duration_ms)
    data = np.asarray(ngpu.GetRecordData(record), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(columns) + 1:
        raise RuntimeError(f"malformed GPU trace shape {data.shape}; expected {len(columns)+1} columns")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        time_ms=data[:, 0], values=data[:, 1:],
        columns_json=np.asarray(json.dumps(columns, separators=(",", ":"))),
    )

    recorded = [ngpu.GetRecSpikeTimes(nodes) for nodes in cells]
    results: list[dict[str, Any]] = []
    column_lookup = {(c["target_type"], c["arm"], c["variable"], c["port"]): i + 1 for i, c in enumerate(columns)}
    for target_idx, target in enumerate(TARGETS):
        for arm_idx, arm in enumerate(ARMS):
            raw = recorded[target_idx]
            spikes = np.asarray([] if raw is None or raw[arm_idx] is None else raw[arm_idx], dtype=float)
            voltage_summary = {}
            for variable in ("V_m", "V_d", "V_dist"):
                trace = data[:, column_lookup[(target, arm, variable, None)]]
                voltage_summary[variable] = {"mean_mV": float(trace.mean()), "min_mV": float(trace.min()), "max_mV": float(trace.max())}
            receptors = spec.receptors_for_post(target)
            conductance = []
            for port in range(receptors.n_ports()):
                trace = data[:, column_lookup[(target, arm, "g", port)]]
                conductance.append({"port": port, "receptor": receptors.names[port],
                    "mean_nS": float(trace.mean()), "max_nS": float(trace.max())})
            results.append({
                "target_type": target, "target_id": int(selected[target]), "arm": arm,
                "dt_ms": dt_ms, "duration_ms": duration_ms,
                "n_spikes": int(spikes.size), "rate_hz": float(spikes.size / (duration_ms / 1000.0)),
                "spike_times_ms": spikes.tolist(), "voltage": voltage_summary,
                "conductance_by_port": conductance,
            })
    return results


def _cpu_expected(cpu_report: Mapping[str, Any], selected: Mapping[str, int]) -> dict[tuple[str, str], bool]:
    arm_name = {"all": "all", "no_inhibition": "no_inhibition", "drop_CCK": "drop_CCK"}
    result: dict[tuple[str, str], bool] = {}
    for row in cpu_report["step3"]["per_cell"]:
        target = str(row["target_type"])
        if int(row["target_id"]) != int(selected.get(target, -1)) or float(row["dt_ms"]) != 0.025:
            continue
        arm = str(row["arm"])
        if arm in arm_name:
            result[(target, arm_name[arm])] = int(row["n_spikes"]) > 0
    return result


def _needs_window_check(primary: Sequence[Mapping[str, Any]], expected: Mapping[tuple[str, str], bool]) -> bool:
    return any((int(row["n_spikes"]) > 0) != expected.get((str(row["target_type"]), str(row["arm"])), int(row["n_spikes"]) > 0) for row in primary)


def _worker_command(args: argparse.Namespace, dt_ms: float, duration_ms: float, output: Path) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--dt-ms", str(dt_ms),
        "--duration-ms", str(duration_ms), "--config", str(args.config), "--edges", str(args.edges),
        "--run", str(args.run), "--cpu-report", str(args.cpu_report), "--output", str(output)]
    for target in args.target:
        command += ["--target", target]
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cpu-report", type=Path, default=DEFAULT_CPU_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", action="append", default=[], help="override with a Step-2 selected TYPE:ID")
    parser.add_argument("--preflight", action="store_true", help="reconstruct and check parity without importing NEST-GPU")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dt-ms", type=float, choices=DT_MS, default=0.1)
    parser.add_argument("--duration-ms", type=float, default=0.0)
    parser.add_argument("--check-window-ms", type=float, default=1000.0)
    return parser.parse_args()


def _single_run(args: argparse.Namespace) -> dict[str, Any]:
    replay = _load_replay_module()
    cpu_report = json.loads(args.cpu_report.read_text(encoding="utf-8"))
    selected = _selected_targets(cpu_report, args.target)
    spec = build_network_spec(args.config, scale=1.0, seed=12345)
    with h5py.File(args.edges, "r") as edges, h5py.File(args.run, "r") as run:
        descriptors = replay.projections(edges)
        source_duration_s = _h5_float(run["meta"].attrs["duration_s"])
        duration_ms = float(args.duration_ms or source_duration_s * 1000.0)
        afferent_dt_ms = _h5_float(run["meta"].attrs["dt_s"]) * 1000.0
        source_sizes = {aff.name.split("_to_", 1)[0]: int(aff.n_source) for aff in spec.afferents}
        sources = {d.pre for d in descriptors if d.kind == "afferent" and d.post in TARGETS}
        counts = {source: replay._afferent_counts(source, source_sizes[source], 0.65,
            source_duration_s, 12345) for source in sources}
        with tempfile.TemporaryDirectory(prefix="ca1-step4-afferent-") as tmp:
            store = {source: replay.build_afferent_slot_store(Path(tmp), source, count,
                duration_ms=source_duration_s * 1000.0,
                dt_ms=afferent_dt_ms, seed=12345, rate_hz=0.65) for source, count in counts.items()}
            rows = construct_rows(replay, edges, run, descriptors, spec, store, selected, afferent_dt_ms)
            manifest, parity = arrival_manifest(rows, cpu_report, source_duration_s * 1000.0)
            inhibitory_erev = {float(spec.receptors_for_post(row.target_type).E_rev[row.port]) for row in rows if row.pre not in EXCITATORY}
            if inhibitory_erev != {-60.0}:
                raise RuntimeError(f"deployed inhibitory E_rev mismatch: {sorted(inhibitory_erev)}")
            results: list[dict[str, Any]] = []
            schedule_path = args.output.with_name(args.output.stem + ".injection_schedule.npz")
            _write_injection_schedule(rows, schedule_path)
            trace_path = args.output.with_name(args.output.stem + ".traces.npz")
            if not args.preflight:
                results = run_gpu(spec, rows, selected, dt_ms=args.dt_ms, duration_ms=duration_ms, trace_path=trace_path)
    return {
        "protocol": {"backend": "NEST-GPU user_m2", "single_gpu": True, "mpi": False,
            "dt_ms": args.dt_ms, "duration_ms": duration_ms, "arms": list(ARMS),
            "selected_targets": selected, "immutable_deployed_parameters": True},
        "provenance": {"config": str(args.config), "edges": str(args.edges), "run": str(args.run),
            "cpu_report": str(args.cpu_report), "edge_sha256": cpu_report["provenance"]["edge_sha256"]},
        "arrival_manifest": manifest, "arrival_count_parity": parity,
        "arrival_count_parity_all": all(item["equal"] for item in parity),
        "arm_port_arrival_counts": _arm_port_arrival_counts(rows, duration_ms),
        "inhibitory_E_rev_mV": sorted(inhibitory_erev),
        "injection_schedule_npz": str(schedule_path),
        "trace_npz": None if args.preflight else str(trace_path), "results": results,
        "preflight_only": bool(args.preflight),
    }


def main() -> int:
    args = _parse_args()
    if args.worker or args.preflight:
        report = _single_run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}; arrival parity={report['arrival_count_parity_all']}")
        return 0

    # NEST-GPU has no safe in-process kernel reset.  Use one ordinary process
    # (never MPI) per time resolution, and run fine-dt checks only on a primary
    # GPU/CPU spike-conclusion disagreement.
    primary_path = args.output.with_suffix(".dt0.1.json")
    subprocess.run(_worker_command(args, 0.1, 0.0, primary_path), check=True)
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    cpu_report = json.loads(args.cpu_report.read_text(encoding="utf-8"))
    selected = _selected_targets(cpu_report, args.target)
    reports = [primary]
    if _needs_window_check(primary["results"], _cpu_expected(cpu_report, selected)):
        for dt_ms in (0.05, 0.025):
            path = args.output.with_suffix(f".dt{dt_ms:g}.json")
            subprocess.run(_worker_command(args, dt_ms, args.check_window_ms, path), check=True)
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    combined = {
        "protocol": {"single_gpu": True, "mpi": False, "conditional_fine_dt": True,
            "fine_dt_triggered": len(reports) > 1, "check_window_ms": args.check_window_ms},
        "runs": reports,
        "decision": "A silent + B firing + C rescue confirms H2/backend; disagreement localizes backend/port/normalization semantics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"wrote {args.output}; fine-dt checks triggered={len(reports) > 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
