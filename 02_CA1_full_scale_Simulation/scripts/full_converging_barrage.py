#!/usr/bin/env python3
"""Single-cell full-converging excitatory barrage audit (CPU only).

This is an emergent working-point companion to ``paired_transfer_audit.py``.
It reuses that audit's immutable ModelDB row contract, source-location filter,
cell templates, synapse kinetics, beta normalization, and user_m2 equations.
No network is built: every configured excitatory row entering one target cell
is represented by its full biological in-degree and an independent Poisson
source population.

The three arms are:

* source: native spiking ModelDB/NEURON multi-compartment cell;
* deployed: user_m2 CPU RK4 with checked-in transfer/passive parameters;
* candidate: user_m2 CPU RK4 with the charge-matched row transfer plus the
  passive overrides in ``results/dendrite_refit_candidate.json``.

The recurrent pyramidal rate is a declared held-out proxy and is never fitted
to Table 5 or to an output firing-rate target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ca1.config import build_network_spec
from ca1.extract.modeldb_tables import extract_connectivity
from ca1.params.groundtruth import _MODELDB, _soma, neuron_session
from ca1.sim.aglif_dend import aglif_dend_status


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "full_scale_3dtopo.yaml"
DEFAULT_TRANSFER = ROOT / "results" / "charge_matched_transfer_candidate.json"
DEFAULT_CANDIDATE = ROOT / "results" / "dendrite_refit_candidate.json"
DEFAULT_OUTPUT = ROOT / "results" / "full_converging_barrage.json"
DEFAULT_MARKDOWN = ROOT / "scratchpad" / "barrage_firing_result.md"
TARGET_CELLS = ("PV_Basket", "Bistratified", "O_LM")
DEFAULT_SEEDS = (20260712, 20260713, 20260714)
DEFAULT_DTS_MS = (0.05, 0.025)
EXTERNAL_RATE_HZ = 0.65
RECURRENT_PROXY_HZ = 1.0
Domain = str


def _load_paired_module() -> Any:
    path = ROOT / "scripts" / "paired_transfer_audit.py"
    spec = importlib.util.spec_from_file_location("_paired_transfer_for_barrage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = _load_paired_module()


@dataclass(frozen=True)
class BarrageRow:
    source: Any
    indegree_true: int
    rate_hz_per_afferent: float

    @property
    def row_id(self) -> str:
        return f"{self.source.pre}->{self.source.post}"


@dataclass(frozen=True)
class RowSchedule:
    row_id: str
    event_times_ms: np.ndarray
    location_indices: np.ndarray
    connection_event_counts: np.ndarray


@dataclass(frozen=True)
class ArmResult:
    cell: str
    arm: str
    seed: int
    dt_ms: float
    n_spikes: int
    rate_hz: float
    mean_v_mV: float
    max_v_mV: float
    threshold_mV: float
    threshold_status: str


def configured_rows(
    config: Path,
    recurrent_proxy_hz: float = RECURRENT_PROXY_HZ,
) -> dict[str, list[BarrageRow]]:
    """Build rows from the same selected conndata/NetworkSpec as paired audit.

    ``indegree_true`` here means the biological connection in-degree.  For an
    afferent row the selected conndata stores a contact budget, so it is divided
    by the immutable contacts-per-connection value.  Recurrent projections
    expose the biological in-degree directly.
    """
    if recurrent_proxy_hz <= 0.0:
        raise ValueError("recurrent proxy rate must be positive")
    spec = build_network_spec(config, scale=1.0, seed=PAIRED.LOCATION_SEED)
    raw = extract_connectivity(
        index=spec.conndata_index,
        cellnumbers_index=spec.cellnumbers_index,
        count_mode=spec.conndata_count_mode,
    )
    result: dict[str, list[BarrageRow]] = {cell: [] for cell in TARGET_CELLS}
    for source in PAIRED.configured_excitatory_rows(config):
        row_key = f"{source.pre}_to_{source.post}"
        if source.kind == "aff":
            entry = raw["afferents"][row_key]
            contacts = float(entry["synapses_per_cell"])
            indegree = int(round(contacts / source.synapses_per_connection))
            if not np.isclose(indegree * source.synapses_per_connection, contacts):
                raise ValueError(f"non-integral biological in-degree for {row_key}")
            rate = EXTERNAL_RATE_HZ
        else:
            entry = raw["projections"][row_key]
            indegree = int(round(float(entry["indegree"])))
            if not np.isclose(indegree, float(entry["indegree"])):
                raise ValueError(f"non-integral biological in-degree for {row_key}")
            rate = recurrent_proxy_hz
        result[source.post].append(BarrageRow(source, indegree, rate))
    for rows in result.values():
        rows.sort(key=lambda item: item.source.pre)
    return result


def _row_entropy(row_id: str) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def poisson_schedule(
    row: BarrageRow,
    duration_ms: float,
    seed: int,
    n_locations: int,
) -> RowSchedule:
    """Draw independent per-afferent Poisson trains and fixed contact sites.

    Conditional uniform event times are an exact homogeneous Poisson process.
    Each biological connection keeps one fixed, with-replacement location draw
    for every contact; all contacts on that connection share its spike train.
    """
    if duration_ms <= 0.0 or n_locations <= 0:
        raise ValueError("duration and location count must be positive")
    entropy = _row_entropy(row.row_id)
    spike_rng = np.random.default_rng(np.random.SeedSequence([seed, entropy, 0x5A17]))
    location_rng = np.random.default_rng(
        np.random.SeedSequence([seed, entropy, 0x10CA7E])
    )
    expected = row.rate_hz_per_afferent * duration_ms / 1000.0
    counts = spike_rng.poisson(expected, size=row.indegree_true).astype(np.int64)
    total = int(counts.sum())
    times = np.empty(total, dtype=float)
    cursor = 0
    for count in counts:
        next_cursor = cursor + int(count)
        if count:
            times[cursor:next_cursor] = np.sort(
                spike_rng.uniform(0.0, duration_ms, size=int(count))
            )
        cursor = next_cursor
    locations = location_rng.integers(
        0,
        n_locations,
        size=(row.indegree_true, row.source.synapses_per_connection),
        dtype=np.int64,
    )
    return RowSchedule(row.row_id, times, locations, counts)


def _connection_slices(counts: np.ndarray) -> Iterable[tuple[int, slice]]:
    cursor = 0
    for connection, count in enumerate(counts):
        next_cursor = cursor + int(count)
        yield connection, slice(cursor, next_cursor)
        cursor = next_cursor


def _source_threshold(cell_name: str) -> float:
    return -20.0 if cell_name == "O_LM" else -10.0


def run_neuron_source(
    rows: Sequence[BarrageRow],
    seed: int,
    dt_ms: float,
    transient_ms: float,
    measure_ms: float,
) -> ArmResult:
    """Replay the full barrage on one native, actively spiking ModelDB cell."""
    cell_name = rows[0].source.post
    if any(row.source.post != cell_name for row in rows):
        raise ValueError("all source rows must target one cell")
    h = neuron_session()
    template = rows[0].source.template
    h.load_file(str(_MODELDB / "cells" / f"class_{template}.hoc"))
    h.dt = dt_ms
    h.steps_per_ms = 1.0 / dt_ms
    h.cvode.active(0)
    cell = getattr(h, template)(0, 0, 0)
    soma = _soma(cell)
    tstop_ms = transient_ms + measure_ms

    keepalive: list[Any] = []
    scheduled: list[tuple[Any, np.ndarray]] = []
    for row in rows:
        segments = PAIRED.eligible_segments(h, cell, row.source)
        schedule = poisson_schedule(row, tstop_ms - row.source.delay_ms, seed, len(segments))
        # One linear point process per eligible segment is sufficient: events
        # from distinct fixed afferents sum exactly at a shared location.
        netcons: list[Any] = []
        per_location: list[list[np.ndarray]] = [[] for _ in segments]
        for connection, event_slice in _connection_slices(schedule.connection_event_counts):
            train = schedule.event_times_ms[event_slice] + row.source.delay_ms
            if not train.size:
                continue
            for location in schedule.location_indices[connection]:
                per_location[int(location)].append(train)
        for location, segment in enumerate(segments):
            if not per_location[location]:
                netcons.append(None)
                continue
            synapse = h.MyExp2Sid(segment.x, sec=segment.sec)
            synapse.tau1 = row.source.tau_rise_ms
            synapse.tau2 = row.source.tau_decay_ms
            synapse.e = row.source.e_rev_mV
            connection = h.NetCon(None, synapse)
            connection.weight[0] = row.source.source_gmax_nS / 1000.0
            arrivals = np.sort(np.concatenate(per_location[location]))
            keepalive.extend((synapse, connection))
            netcons.append(connection)
            scheduled.append((connection, arrivals))

    voltage = h.Vector()
    voltage.record(soma(0.5)._ref_v, dt_ms)
    time = h.Vector()
    time.record(h._ref_t, dt_ms)
    spikes = h.Vector()
    detector = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
    threshold = _source_threshold(cell_name)
    detector.threshold = threshold
    detector.record(spikes)
    keepalive.extend((voltage, time, spikes, detector))

    h.finitialize(float(cell.Vrest))
    for connection, arrivals in scheduled:
        for arrival in arrivals:
            connection.event(float(arrival))
    h.continuerun(tstop_ms)
    times = np.asarray(time, dtype=float)
    vm = np.asarray(voltage, dtype=float)
    spike_times = np.asarray(spikes, dtype=float)
    window = (times >= transient_ms) & (times < tstop_ms)
    measured_spikes = spike_times[
        (spike_times >= transient_ms) & (spike_times < tstop_ms)
    ]
    return ArmResult(
        cell=cell_name,
        arm="source_neuron",
        seed=seed,
        dt_ms=dt_ms,
        n_spikes=int(measured_spikes.size),
        rate_hz=float(measured_spikes.size / (measure_ms / 1000.0)),
        mean_v_mV=float(vm[window].mean()),
        max_v_mV=float(vm[window].max()),
        threshold_mV=threshold,
        threshold_status="suprathreshold/firing" if measured_spikes.size else "subthreshold/silent",
    )


def _passive_status(cell: str, overrides: Mapping[str, float] | None) -> dict[str, float]:
    status = aglif_dend_status(cell)
    if overrides:
        status.update({key: float(value) for key, value in overrides.items()})
        if "g_c_scale" in overrides:
            membrane_conductance = float(status["C_m"]) / float(status["tau_m"])
            status["g_c"] = 2.0 * membrane_conductance * float(overrides["g_c_scale"])
        if "dist_coupling_ratio" in overrides or "g_c_scale" in overrides:
            ratio = float(overrides.get("dist_coupling_ratio", 0.25))
            status["g_c_dist"] = float(status["g_c"]) * ratio
    return status


def _candidate_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{record['contract']['pre']}->{record['contract']['post']}": record
        for record in report["rows"]
    }


def _transfer_for_arm(
    row: BarrageRow,
    arm: str,
    candidate_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[float, np.ndarray]:
    if arm == "deployed_user_m2":
        allocation = np.asarray(
            [float(row.source.deployed_domain == domain) for domain in ("soma", "proximal", "distal")]
        )
        return float(row.source.deployed_gmax_nS), allocation
    if arm != "candidate_user_m2":
        raise ValueError(f"unknown reduced arm {arm!r}")
    record = candidate_rows[row.row_id]
    contract = record["contract"]
    if contract["source_gmax_nS"] != row.source.source_gmax_nS:
        raise ValueError(f"candidate/source contract mismatch for {row.row_id}")
    derivation = record["derivation"]
    allocation_map = derivation["graded_allocation"]
    allocation = np.asarray(
        [float(allocation_map[domain]) for domain in ("soma", "proximal", "distal")]
    )
    transferred = (
        float(row.source.source_gmax_nS)
        * float(derivation["constraint"]["total_transfer_scale"])
    )
    return transferred, allocation


def run_reduced(
    rows: Sequence[BarrageRow],
    arm: str,
    seed: int,
    dt_ms: float,
    transient_ms: float,
    measure_ms: float,
    candidate_rows: Mapping[str, Mapping[str, Any]],
    passive_overrides: Mapping[str, float] | None = None,
) -> ArmResult:
    """CPU RK4 replay of the checked-in user_m2 equations with real spikes."""
    cell = rows[0].source.post
    status = _passive_status(cell, passive_overrides)
    c_m = float(status["C_m"])
    c_dend = c_m * float(status["dend_C_frac"])
    c_dist = c_dend * float(status["dist_C_frac"])
    caps = np.asarray([c_m - c_dend, c_dend - c_dist, c_dist], dtype=float)
    e_l = float(status["E_L"])
    tstop_ms = transient_ms + measure_ms
    n_steps = int(round(tstop_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, tstop_ms):
        raise ValueError("duration must be an integer number of timesteps")

    # vm, vd, vdist, I_adap, I_dep, then (g, g1) for every row/domain.
    state = np.zeros(5 + 6 * len(rows), dtype=float)
    state[:3] = e_l
    event_counts: list[np.ndarray] = []
    amplitudes: list[np.ndarray] = []
    for row in rows:
        schedule = poisson_schedule(row, tstop_ms - row.source.delay_ms, seed, 1)
        arrivals = schedule.event_times_ms + row.source.delay_ms
        bins = np.ceil(arrivals / dt_ms - 1e-12).astype(np.int64)
        valid = bins[(bins >= 0) & (bins < n_steps)]
        event_counts.append(np.bincount(valid, minlength=n_steps))
        weight_nS, allocation = _transfer_for_arm(row, arm, candidate_rows)
        amplitudes.append(
            weight_nS
            * row.source.synapses_per_connection
            * PAIRED._beta_g0(row.source.tau_rise_ms, row.source.tau_decay_ms)
            * allocation
        )

    refractory = 0
    spike_count = 0
    measure_sum = 0.0
    measure_count = 0
    measure_max = -np.inf

    def derivative(values: np.ndarray, clamped_refractory: bool) -> np.ndarray:
        vm, vd, vx, i_adap, i_dep = values[:5]
        syn = np.zeros(3, dtype=float)
        result = np.zeros_like(values)
        for index, row in enumerate(rows):
            offset = 5 + 6 * index
            conductances = values[offset : offset + 6 : 2]
            syn += conductances * (float(row.source.e_rev_mV) - np.asarray([vm, vd, vx]))
            for domain in range(3):
                g_offset = offset + 2 * domain
                result[g_offset] = values[g_offset + 1] - values[g_offset] / row.source.tau_decay_ms
                result[g_offset + 1] = -values[g_offset + 1] / row.source.tau_rise_ms
        soma_num = (
            -(caps[0] / float(status["tau_m"])) * (vm - e_l)
            + float(status["g_c"]) * (vd - vm)
            - i_adap
            + i_dep
            + float(status["I_e"])
            + syn[0]
        )
        result[0] = 0.0 if clamped_refractory else soma_num / caps[0]
        result[1] = (
            -(caps[1] / float(status["tau_m"])) * float(status["dend_leak_scale"]) * (vd - e_l)
            + float(status["g_c"]) * (vm - vd)
            + float(status["g_c_dist"]) * (vx - vd)
            + syn[1]
        ) / caps[1]
        result[2] = (
            -(caps[2] / float(status["tau_m"])) * float(status["dist_leak_scale"]) * (vx - e_l)
            + float(status["g_c_dist"]) * (vd - vx)
            + syn[2]
        ) / caps[2]
        result[3] = float(status["k_adap"]) * (vm - e_l) - float(status["k2"]) * i_adap
        result[4] = -float(status["k1"]) * i_dep
        return result

    for step in range(n_steps):
        for index, counts in enumerate(event_counts):
            count = int(counts[step])
            if count:
                offset = 5 + 6 * index
                state[offset + 1 : offset + 6 : 2] += count * amplitudes[index]
        in_refractory = refractory > 0
        k1 = derivative(state, in_refractory)
        k2 = derivative(state + 0.5 * dt_ms * k1, in_refractory)
        k3 = derivative(state + 0.5 * dt_ms * k2, in_refractory)
        k4 = derivative(state + dt_ms * k3, in_refractory)
        state += dt_ms * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if in_refractory:
            state[0] = float(status["V_reset"])
            refractory -= 1
            spiked = False
        else:
            spiked = bool(state[0] >= float(status["V_th"]))
        now_ms = (step + 1) * dt_ms
        if spiked:
            if now_ms >= transient_ms:
                spike_count += 1
            state[0] = float(status["V_reset"])
            state[3] += float(status["A2"])
            state[4] = float(status["A1"])
            refractory = int(round(float(status["t_ref"]) / dt_ms))
        if now_ms >= transient_ms:
            measure_sum += float(state[0])
            measure_count += 1
            measure_max = max(measure_max, float(state[0]))

    return ArmResult(
        cell=cell,
        arm=arm,
        seed=seed,
        dt_ms=dt_ms,
        n_spikes=spike_count,
        rate_hz=float(spike_count / (measure_ms / 1000.0)),
        mean_v_mV=measure_sum / measure_count,
        max_v_mV=measure_max,
        threshold_mV=float(status["V_th"]),
        threshold_status="suprathreshold/firing" if spike_count else "subthreshold/silent",
    )


def _source_task(
    task: tuple[list[BarrageRow], int, float, float, float],
) -> ArmResult:
    return run_neuron_source(*task)


def _reduced_task(
    task: tuple[
        list[BarrageRow], str, int, float, float, float,
        Mapping[str, Mapping[str, Any]], Mapping[str, float] | None,
    ],
) -> ArmResult:
    return run_reduced(*task)


def _summaries(results: Sequence[ArmResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[ArmResult]] = {}
    for result in results:
        groups.setdefault((result.cell, result.arm, result.dt_ms), []).append(result)
    output: list[dict[str, Any]] = []
    for (cell, arm, dt_ms), values in sorted(groups.items()):
        rates = np.asarray([value.rate_hz for value in values], dtype=float)
        means = np.asarray([value.mean_v_mV for value in values], dtype=float)
        output.append({
            "cell": cell,
            "arm": arm,
            "dt_ms": dt_ms,
            "rate_mean_hz": float(rates.mean()),
            "rate_sd_hz": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
            "rate_min_hz": float(rates.min()),
            "rate_max_hz": float(rates.max()),
            "mean_v_mV": float(means.mean()),
            "n_seeds": len(values),
            "all_firing": all(value.n_spikes > 0 for value in values),
        })
    return output


def _gap_closed(summaries: Sequence[Mapping[str, Any]], dt_ms: float) -> list[dict[str, Any]]:
    lookup = {
        (str(row["cell"]), str(row["arm"]), float(row["dt_ms"])): float(row["rate_mean_hz"])
        for row in summaries
    }
    output: list[dict[str, Any]] = []
    for cell in TARGET_CELLS:
        source = lookup[(cell, "source_neuron", dt_ms)]
        deployed = lookup[(cell, "deployed_user_m2", dt_ms)]
        candidate = lookup[(cell, "candidate_user_m2", dt_ms)]
        old_gap = abs(source - deployed)
        new_gap = abs(source - candidate)
        gap_closed = None if old_gap == 0.0 else 100.0 * (old_gap - new_gap) / old_gap
        output.append({
            "cell": cell,
            "dt_ms": dt_ms,
            "source_rate_hz": source,
            "deployed_rate_hz": deployed,
            "candidate_rate_hz": candidate,
            "deployed_abs_gap_hz": old_gap,
            "candidate_abs_gap_hz": new_gap,
            "gap_closed_percent": gap_closed,
            "candidate_closer_to_source": new_gap < old_gap,
        })
    return output


def _dt_stability(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["cell"]), str(row["arm"]), float(row["dt_ms"])): float(row["rate_mean_hz"])
        for row in summaries
    }
    output: list[dict[str, Any]] = []
    for cell in TARGET_CELLS:
        for arm in ("source_neuron", "deployed_user_m2", "candidate_user_m2"):
            coarse = lookup[(cell, arm, 0.05)]
            fine = lookup[(cell, arm, 0.025)]
            output.append({
                "cell": cell,
                "arm": arm,
                "rate_dt_0p05_hz": coarse,
                "rate_dt_0p025_hz": fine,
                "absolute_difference_hz": abs(coarse - fine),
                "relative_difference_percent": (
                    None if fine == 0.0 else 100.0 * abs(coarse - fine) / abs(fine)
                ),
            })
    return output


def _row_metadata(rows_by_cell: Mapping[str, Sequence[BarrageRow]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for cell in TARGET_CELLS:
        for row in rows_by_cell[cell]:
            record = asdict(row.source)
            record.update({
                "row": row.row_id,
                "indegree_true_connections": row.indegree_true,
                "rate_hz_per_afferent": row.rate_hz_per_afferent,
                "aggregate_event_rate_hz": row.indegree_true * row.rate_hz_per_afferent,
            })
            metadata.append(record)
    return metadata


def run_audit(
    config: Path,
    transfer_path: Path,
    passive_path: Path,
    seeds: Sequence[int],
    dts_ms: Sequence[float],
    transient_ms: float,
    measure_ms: float,
    recurrent_proxy_hz: float,
    processes: int = 1,
) -> dict[str, Any]:
    if len(seeds) < 3:
        raise ValueError("seed sensitivity requires at least three seeds")
    if set(float(dt) for dt in dts_ms) != {0.05, 0.025}:
        raise ValueError("dt stability requires exactly 0.05 and 0.025 ms")
    rows_by_cell = configured_rows(config, recurrent_proxy_hz)
    candidate_rows = _candidate_rows(transfer_path)
    passive_report = json.loads(passive_path.read_text(encoding="utf-8"))
    tasks = [
        (list(rows_by_cell[cell]), int(seed), float(dt), transient_ms, measure_ms)
        for cell in TARGET_CELLS for seed in seeds for dt in dts_ms
    ]
    if processes == 1:
        source_results = [_source_task(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        # NEURON mechanism DLLs are process-global and cannot be loaded twice.
        # A fresh worker per source replay matches paired_transfer_audit's
        # source-batch isolation and also prevents retained HOC sections from
        # one template contaminating the next cell.
        with context.Pool(
            min(processes, len(tasks)),
            maxtasksperchild=1,
        ) as pool:
            source_results = pool.map(_source_task, tasks, chunksize=1)
    reduced_tasks: list[
        tuple[
            list[BarrageRow], str, int, float, float, float,
            Mapping[str, Mapping[str, Any]], Mapping[str, float] | None,
        ]
    ] = []
    for cell in TARGET_CELLS:
        fitted = passive_report["cells"][cell]["fitted_params"]
        for seed in seeds:
            for dt_ms in dts_ms:
                reduced_tasks.append((
                    list(rows_by_cell[cell]), "deployed_user_m2", int(seed), float(dt_ms),
                    transient_ms, measure_ms, candidate_rows, None,
                ))
                reduced_tasks.append((
                    list(rows_by_cell[cell]), "candidate_user_m2", int(seed), float(dt_ms),
                    transient_ms, measure_ms, candidate_rows, fitted,
                ))
    if processes == 1:
        reduced_results = [_reduced_task(task) for task in reduced_tasks]
    else:
        context = mp.get_context("spawn")
        with context.Pool(min(processes, len(reduced_tasks))) as pool:
            reduced_results = pool.map(_reduced_task, reduced_tasks, chunksize=1)
    results = [*source_results, *reduced_results]
    summaries = _summaries(results)
    return {
        "schema": "full-converging-barrage/v1",
        "provenance": {
            "method": "single-cell ModelDB NEURON vs user_m2 CPU RK4 replay",
            "paired_transfer_reuse": [
                "SourceRow/build_source_row", "eligible_segments", "neuron_session/cell templates",
                "MyExp2Sid kinetics and locations", "beta_g0", "user_m2 three-compartment equations",
            ],
            "network_built": False,
            "gpu_used": False,
            "mpi_used": False,
            "table5_rate_tuning": False,
            "source_gmax_kinetics_locations_contacts_immutable": True,
            "selected_connectivity": "full_scale_3dtopo conndata_430/per_cell",
            "candidate_transfer": str(transfer_path),
            "candidate_passives": str(passive_path),
        },
        "protocol": {
            "external_rate_hz_per_afferent": EXTERNAL_RATE_HZ,
            "recurrent_pyramidal_proxy_hz": recurrent_proxy_hz,
            "recurrent_proxy_rationale": (
                "preregistered round 1 Hz sparse-background proxy; held fixed across cells/arms, "
                "not selected from Table 5 and not adjusted after observing outputs"
            ),
            "transient_ms": transient_ms,
            "measure_ms": measure_ms,
            "seeds": list(seeds),
            "dt_ms": list(dts_ms),
        },
        "rows": _row_metadata(rows_by_cell),
        "results": [asdict(result) for result in results],
        "summary": summaries,
        "gap_closed_primary_dt": _gap_closed(summaries, 0.025),
        "dt_stability": _dt_stability(summaries),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    protocol = report["protocol"]
    summaries = report["summary"]
    primary = {row["cell"]: row for row in report["gap_closed_primary_dt"]}
    lookup = {
        (row["cell"], row["arm"], row["dt_ms"]): row for row in summaries
    }
    lines = [
        "# Full converging barrage firing result",
        "",
        "## Protocol",
        "",
        "This is a CPU-only, single-cell replay; no network, GPU, MPI, deployment, or rate fitting was used. "
        "It reuses `scripts/paired_transfer_audit.py` for the immutable source row contract, ModelDB cell/location machinery, synaptic beta normalization, and user_m2 equations.",
        "",
        f"External CA3/ECIII afferents fire independently at **{protocol['external_rate_hz_per_afferent']} Hz each**. "
        f"The recurrent pyramidal held-out proxy is **{protocol['recurrent_pyramidal_proxy_hz']} Hz per presynaptic cell**: "
        f"{protocol['recurrent_proxy_rationale']}. The analysis window is {protocol['measure_ms']/1000:g} s after a {protocol['transient_ms']/1000:g} s transient.",
        "",
        "## Immutable converging rows",
        "",
        "| row | in-degree (connections) | contacts/connection | source gmax/contact (nS) | deployed gmax/contact (nS) | receptor/domain | location rule | rate/afferent (Hz) |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in report["rows"]:
        conditions = ", ".join(row["distance_conditions"])
        lines.append(
            f"| {row['row']} | {row['indegree_true_connections']} | {row['synapses_per_connection']} | "
            f"{row['source_gmax_nS']:.6g} | {row['deployed_gmax_nS']:.6g} | "
            f"`{row['receptor']}` / {row['deployed_domain']} | `{row['section_list']}`; {conditions} | "
            f"{row['rate_hz_per_afferent']:.3g} |"
        )
    lines.extend([
        "",
        "## Primary result (dt 0.025 ms; mean ± sample SD across seeds)",
        "",
        "| cell | source NEURON (Hz) | deployed user_m2 (Hz) | candidate user_m2 (Hz) | candidate gap closed | candidate mean Vm (mV) | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for cell in TARGET_CELLS:
        source = lookup[(cell, "source_neuron", 0.025)]
        deployed = lookup[(cell, "deployed_user_m2", 0.025)]
        candidate = lookup[(cell, "candidate_user_m2", 0.025)]
        gap = primary[cell]["gap_closed_percent"]
        gap_text = "n/a" if gap is None else f"{gap:.1f}%"
        lines.append(
            f"| {cell} | {source['rate_mean_hz']:.3g} ± {source['rate_sd_hz']:.3g} | "
            f"{deployed['rate_mean_hz']:.3g} ± {deployed['rate_sd_hz']:.3g} | "
            f"{candidate['rate_mean_hz']:.3g} ± {candidate['rate_sd_hz']:.3g} | {gap_text} | "
            f"{candidate['mean_v_mV']:.3f} | {'firing' if candidate['all_firing'] else 'silent in ≥1 seed'} |"
        )
    lines.extend([
        "",
        "Gap closed is `100 × (|source-deployed| - |source-candidate|) / |source-deployed|`; negative values mean the candidate moved farther from source.",
        "",
        "Mean Vm is retained for all arms in the JSON summary. Source means include native action-potential waveforms; reduced means include reset/refractory samples, so they are descriptive within-arm diagnostics rather than a cross-model fit metric.",
        "",
        "## dt and seed stability",
        "",
        "| cell | arm | 0.05 ms (Hz) | 0.025 ms (Hz) | absolute Δ (Hz) | relative Δ |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in report["dt_stability"]:
        relative = row["relative_difference_percent"]
        relative_text = "n/a" if relative is None else f"{relative:.2f}%"
        lines.append(
            f"| {row['cell']} | {row['arm']} | {row['rate_dt_0p05_hz']:.3g} | "
            f"{row['rate_dt_0p025_hz']:.3g} | {row['absolute_difference_hz']:.3g} | {relative_text} |"
        )
    lines.extend([
        "",
        f"Seeds: {', '.join(str(seed) for seed in protocol['seeds'])}. Per-seed values and Vm/max-threshold diagnostics are retained in `results/full_converging_barrage.json`.",
        "",
        "## Verdict",
        "",
        "_Fill from the completed evidence run._",
        "",
        "## Smallest additional defensible lever if insufficient",
        "",
        "_Fill from the completed evidence run._",
        "",
        "## Verification",
        "",
        "_Fill after running the complete pytest suite._",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--transfer-candidate", type=Path, default=DEFAULT_TRANSFER)
    parser.add_argument("--passive-candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--transient-ms", type=float, default=1000.0)
    parser.add_argument("--measure-ms", type=float, default=10000.0)
    parser.add_argument("--recurrent-proxy-hz", type=float, default=RECURRENT_PROXY_HZ)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    if args.transient_ms < 0.0 or args.measure_ms < 10000.0:
        raise ValueError("transient must be non-negative and measurement must be >=10 s")
    report = run_audit(
        args.config.resolve(), args.transfer_candidate.resolve(), args.passive_candidate.resolve(),
        seeds, DEFAULT_DTS_MS, args.transient_ms, args.measure_ms,
        args.recurrent_proxy_hz, args.processes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
