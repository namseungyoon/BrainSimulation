#!/usr/bin/env python3
"""Paired source-NEURON/user_m2 inhibitory-transfer audit (CPU only).

Every configured inhibitory receptor row entering PV_Basket, Bistratified, or
O_LM is replayed as one biological connection event.  ModelDB conductance,
contacts, kinetics, section lists, reversals, and the deployed receptor-port
kinetics are read from the normal build path and are never fitted.  The only
optional fit is the reduced-model transfer scale/domain mapping, using somatic
IPSP peak and ideal somatic voltage-clamp charge at -55 mV.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from ca1.config import build_network_spec
from ca1.extract.modeldb_tables import extract_connectivity
from ca1.params.groundtruth import CELL_TEMPLATES, _MODELDB, _soma, neuron_session
from ca1.params.receptors import receptor_prefix
from ca1.sim.aglif_dend import aglif_dend_compartments, aglif_dend_status


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "full_scale_3dtopo.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "gaba_transfer_audit.json"
DEFAULT_CANDIDATE = ROOT / "results" / "gaba_transfer_candidate.json"
TARGETS = ("PV_Basket", "Bistratified", "O_LM")
EXCITATORY = frozenset(("Pyramidal", "CA3", "ECIII"))
HOLD_MV = -55.0
EVENT_MS = 300.0
POST_MS = 100.0
Domain = Literal["soma", "proximal", "distal"]
DOMAIN_CODE: dict[Domain, int] = {"soma": 0, "proximal": 1, "distal": 2}
_LOCATION_H: Any | None = None


@dataclass(frozen=True)
class InhibitoryRow:
    pre: str
    post: str
    template: str
    source_gmax_nS: float
    deployed_gmax_nS: float
    synapses_per_connection: int
    biological_indegree: float
    deployed_indegree: float
    source_tau_rise_ms: float
    source_tau_decay_ms: float
    source_e_rev_mV: float
    deployed_tau_rise_ms: float
    deployed_tau_decay_ms: float
    deployed_e_rev_mV: float
    delay_ms: float
    section_list: str
    distance_conditions: tuple[str, ...]
    source_location: str
    deployed_domain: Domain
    receptor_class: str
    deployed_receptor: str
    release_component: str
    transfer_scale: float

    @property
    def row_key(self) -> str:
        return f"{self.pre}->{self.post}|{self.deployed_receptor}"


@dataclass(frozen=True)
class Measurement:
    integral_g_nS_ms: float
    ipsp_peak_mV: float
    clamp_charge_nA_ms: float
    time_to_peak_ms: float
    baseline_mV: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_entries(pre: str, post: str) -> list[dict[str, Any]]:
    pre_hoc = CELL_TEMPLATES[pre]
    post_hoc = CELL_TEMPLATES[post]
    return [
        x for x in _load_json(ROOT / "src/ca1/params/syndata_120.json")["entries"]
        if x["presynaptic"] == pre_hoc and x["postsynaptic"] == post_hoc
    ]


def _required_ports(spec: Any, post: str) -> frozenset[str]:
    return frozenset(p.receptor for p in spec.projections if p.post == post)


def configured_rows(config: Path) -> list[InhibitoryRow]:
    spec = build_network_spec(config, scale=1.0, seed=12345)
    raw = extract_connectivity(
        index=spec.conndata_index,
        cellnumbers_index=spec.cellnumbers_index,
        count_mode=spec.conndata_count_mode,
    )
    table = _load_json(Path(spec.source_location_transfer_table))
    table_by_port = {
        (x["pre"], x["post"], x["port"]): x for x in table
        if str(x["receptor"]).startswith("GABA")
    }
    result: list[InhibitoryRow] = []
    for post in TARGETS:
        receptors = spec.receptors_for_post(post)
        compartments = aglif_dend_compartments(
            receptors.names, post, _required_ports(spec, post),
            spec.source_location_transfer_table, spec.aglif_receive_domain_overrides,
        )
        for projection in spec.projections:
            if projection.post != post or projection.pre in EXCITATORY:
                continue
            raw_row = raw["projections"][f"{projection.pre}_to_{post}"]
            port_index = receptors.names.index(projection.receptor)
            transfer = table_by_port[(projection.pre, post, projection.receptor)]
            entries = _source_entries(projection.pre, post)
            # One source entry corresponds to each compartment-aware primary port.
            # GABA_B, when configured, shares the ExpGABAab location entry.
            wanted_soma = str(transfer["aglif_compartment"]) == "soma"
            matching = [
                x for x in entries
                if ((str(x["section_list"]) == "soma_list") == wanted_soma)
            ]
            if projection.release_component == "GABA_B":
                matching = [x for x in entries if x["mechanism"] == "ExpGABAab"]
            if len(matching) != 1:
                raise ValueError(
                    f"cannot uniquely map {projection.pre}->{post}:{projection.receptor} "
                    f"to syndata row; matches={len(matching)}"
                )
            source = matching[0]
            params = source["parameters"]
            suffix = "_B" if projection.release_component == "GABA_B" else ""
            tr_key = f"tau_rise{suffix}"
            td_key = f"tau_decay{suffix}"
            er_key = f"e_rev{suffix}"
            result.append(InhibitoryRow(
                pre=projection.pre, post=post, template=CELL_TEMPLATES[post],
                source_gmax_nS=float(raw_row["weight_nS"]),
                deployed_gmax_nS=float(projection.weight_nS),
                synapses_per_connection=int(projection.synapses_per_connection),
                biological_indegree=float(projection.biological_indegree or projection.indegree),
                deployed_indegree=float(projection.indegree),
                source_tau_rise_ms=float(params[tr_key]),
                source_tau_decay_ms=float(params[td_key]),
                source_e_rev_mV=float(params[er_key]),
                deployed_tau_rise_ms=float(receptors.tau_rise[port_index]),
                deployed_tau_decay_ms=float(receptors.tau_decay[port_index]),
                deployed_e_rev_mV=float(receptors.E_rev[port_index]),
                delay_ms=float(projection.delay_ms),
                section_list=str(source["section_list"]),
                distance_conditions=tuple(str(x) for x in source["distance_conditions"]),
                source_location=str(transfer["loc"]),
                deployed_domain=("soma", "proximal", "distal")[int(compartments[port_index])],
                receptor_class=receptor_prefix(projection.receptor),
                deployed_receptor=projection.receptor,
                release_component=projection.release_component,
                transfer_scale=float(projection.weight_nS) / float(raw_row["weight_nS"]),
            ))
    return sorted(result, key=lambda x: (TARGETS.index(x.post), x.pre, x.deployed_receptor))


def _predicates(conditions: Iterable[str]) -> list[tuple[str, float]]:
    pattern = re.compile(r"distance\(x\)(?:\*1\.0)?\s*([<>])\s*(-?[0-9.]+)")
    output = []
    for condition in conditions:
        match = pattern.fullmatch(condition.replace(" ", ""))
        if match is None:
            raise ValueError(f"unsupported distance condition {condition!r}")
        output.append((match.group(1), float(match.group(2))))
    return output


def eligible_segments(h: Any, cell: Any, row: InhibitoryRow) -> list[Any]:
    soma = _soma(cell)
    h.distance(0.0, soma(0.5))
    predicates = _predicates(row.distance_conditions)
    output = []
    for section in getattr(cell, row.section_list):
        for segment in section:
            distance = float(h.distance(segment.x, sec=segment.sec))
            if all(distance > bound if op == ">" else distance < bound for op, bound in predicates):
                output.append(segment)
    # Some low-nseg source sections have no segment centre inside a short
    # perisomatic distance window even though a valid continuous section
    # interval exists.  The source placement rule is continuous in x, so use
    # deterministic 1/40-section candidate sites only for that edge case.
    if not output:
        for section in getattr(cell, row.section_list):
            for x in np.concatenate(([0.0], np.linspace(0.0005, 0.9995, 1000), [1.0])):
                distance = float(h.distance(float(x), sec=section))
                if all(distance > bound if op == ">" else distance < bound for op, bound in predicates):
                    output.append(section(float(x)))
    if not output:
        raise RuntimeError(f"no eligible segments for {row.row_key}")
    return output


def _baseline(values: np.ndarray, times: np.ndarray) -> float:
    mask = (times >= EVENT_MS - 25.0) & (times < EVENT_MS)
    return float(values[mask].mean())


def _measure(
    times: np.ndarray, voltage: np.ndarray, conductance: np.ndarray, clamp_i: np.ndarray
) -> Measurement:
    baseline_v = _baseline(voltage, times)
    baseline_i = _baseline(clamp_i, times)
    mask = (times >= EVENT_MS) & (times <= EVENT_MS + POST_MS)
    local_v = voltage[mask]
    local_t = times[mask]
    peak_index = int(np.argmin(local_v))
    return Measurement(
        integral_g_nS_ms=float(np.trapz(conductance[mask], local_t)),
        ipsp_peak_mV=float(baseline_v - local_v[peak_index]),
        clamp_charge_nA_ms=float(abs(np.trapz(clamp_i[mask] - baseline_i, local_t))),
        time_to_peak_ms=float(local_t[peak_index] - EVENT_MS),
        baseline_mV=baseline_v,
    )


def _calibrate_hold(h: Any, template: str, dt_ms: float) -> float:
    """Find a somatic DC current that settles the active source cell near -55 mV."""
    cell = getattr(h, template)(9000, 9000, 0)
    soma = _soma(cell)
    clamp = h.SEClamp(soma(0.5)); clamp.dur1 = EVENT_MS
    clamp.amp1 = HOLD_MV; clamp.rs = 1e-6
    current = h.Vector(); current.record(clamp._ref_i, dt_ms)
    h.finitialize(float(cell.Vrest)); h.continuerun(EVENT_MS - 25.0)
    values = np.asarray(current, dtype=float)
    return float(values[-int(round(25.0 / dt_ms)):].mean())


def _make_source_synapse(h: Any, segment: Any, row: InhibitoryRow) -> tuple[Any, Any, Any]:
    if row.release_component != "primary":
        raise NotImplementedError("configured GABA_B source rows require ExpGABAab replay")
    synapse = h.MyExp2Sid(segment.x, sec=segment.sec)
    synapse.tau1 = row.source_tau_rise_ms
    synapse.tau2 = row.source_tau_decay_ms
    synapse.e = row.source_e_rev_mV
    connection = h.NetCon(None, synapse)
    connection.weight[0] = row.source_gmax_nS / 1000.0
    conductance = h.Vector(); conductance.record(synapse._ref_g, h.dt)
    return synapse, connection, conductance


def run_neuron_source(
    row: InhibitoryRow, draws: np.ndarray, dt_ms: float
) -> tuple[list[Measurement], float]:
    h = neuron_session()
    h.load_file(str(_MODELDB / "cells" / f"class_{row.template}.hoc"))
    h.dt = dt_ms; h.steps_per_ms = 1.0 / dt_ms; h.cvode.active(0)
    hold_amp = _calibrate_hold(h, row.template, dt_ms)
    current_cells = [getattr(h, row.template)(i, i, 0) for i in range(len(draws))]
    control_cells = [getattr(h, row.template)(i + len(draws), i + len(draws), 0) for i in range(len(draws))]
    clamp_cells = [getattr(h, row.template)(i + 2*len(draws), i + 2*len(draws), 0) for i in range(len(draws))]
    current_v: list[Any] = []; control_v: list[Any] = []
    current_g: list[list[Any]] = []; clamp_i: list[Any] = []
    netcons: list[Any] = []; keepalive: list[Any] = []
    for cell, draw in zip(current_cells, draws, strict=True):
        soma = _soma(cell); hold = h.IClamp(soma(0.5))
        hold.delay = 0.0; hold.dur = EVENT_MS + POST_MS; hold.amp = hold_amp
        # Active CCK/SCA templates do not necessarily have a stable DC fixed
        # point at -55 mV.  Pre-hold the soma ideally, release at the event,
        # and subtract a same-cell-type no-synapse control trajectory.  The
        # differential voltage is the isolated current-clamp IPSP from an
        # exact -55 mV baseline, without changing any source mechanism.
        prehold = h.SEClamp(soma(0.5)); prehold.dur1 = EVENT_MS
        prehold.amp1 = HOLD_MV; prehold.rs = 1e-6
        v = h.Vector(); v.record(soma(0.5)._ref_v, dt_ms)
        segments = eligible_segments(h, cell, row); gs = []
        for index in draw:
            syn, nc, g = _make_source_synapse(h, segments[int(index)], row)
            keepalive.extend((syn, nc, g)); netcons.append(nc); gs.append(g)
        keepalive.extend((hold, prehold, v)); current_v.append(v); current_g.append(gs)
    for cell in control_cells:
        soma = _soma(cell); hold = h.IClamp(soma(0.5))
        hold.delay = 0.0; hold.dur = EVENT_MS + POST_MS; hold.amp = hold_amp
        prehold = h.SEClamp(soma(0.5)); prehold.dur1 = EVENT_MS
        prehold.amp1 = HOLD_MV; prehold.rs = 1e-6
        v = h.Vector(); v.record(soma(0.5)._ref_v, dt_ms)
        keepalive.extend((hold, prehold, v)); control_v.append(v)
    for cell, draw in zip(clamp_cells, draws, strict=True):
        soma = _soma(cell); clamp = h.SEClamp(soma(0.5))
        clamp.dur1 = EVENT_MS + POST_MS; clamp.amp1 = HOLD_MV; clamp.rs = 1e-6
        i = h.Vector(); i.record(clamp._ref_i, dt_ms)
        segments = eligible_segments(h, cell, row)
        for index in draw:
            syn, nc, g = _make_source_synapse(h, segments[int(index)], row)
            keepalive.extend((syn, nc, g)); netcons.append(nc)
        keepalive.extend((clamp, i)); clamp_i.append(i)
    time = h.Vector(); time.record(h._ref_t, dt_ms); keepalive.append(time)
    h.finitialize(float(current_cells[0].Vrest))
    for nc in netcons: nc.event(EVENT_MS)
    h.continuerun(EVENT_MS + POST_MS)
    times = np.asarray(time, dtype=float)
    output = []
    for v, control, gs, i in zip(current_v, control_v, current_g, clamp_i, strict=True):
        total_g = np.sum([np.asarray(g, dtype=float) * 1000.0 for g in gs], axis=0)
        isolated_voltage = HOLD_MV - (np.asarray(control, dtype=float) - np.asarray(v, dtype=float))
        output.append(_measure(times, isolated_voltage, total_g, np.asarray(i, dtype=float)))
    return output, hold_amp


def _source_process(row: InhibitoryRow, draws: np.ndarray, dt_ms: float, queue: Any) -> None:
    try:
        queue.put((True, run_neuron_source(row, draws, dt_ms)))
    except BaseException as exc:
        queue.put((False, repr(exc)))


def run_neuron_batched(
    row: InhibitoryRow, draws: np.ndarray, dt_ms: float, batch_size: int = 16
) -> tuple[list[Measurement], float]:
    context = mp.get_context("spawn"); groups = []
    for start in range(0, len(draws), batch_size):
        queue = context.Queue(); process = context.Process(
            target=_source_process, args=(row, draws[start:start + batch_size], dt_ms, queue)
        )
        process.start(); ok, value = queue.get(); process.join()
        if process.exitcode != 0 or not ok:
            raise RuntimeError(f"source batch failed for {row.row_key}: {value}")
        groups.append(value)
    return [x for measurements, _ in groups for x in measurements], float(groups[0][1])


def location_draws(row: InhibitoryRow, n: int, seed: int) -> np.ndarray:
    global _LOCATION_H
    if _LOCATION_H is None:
        _LOCATION_H = neuron_session()
    h = _LOCATION_H
    h.load_file(str(_MODELDB / "cells" / f"class_{row.template}.hoc"))
    cell = getattr(h, row.template)(0, 0, 0); count = len(eligible_segments(h, cell, row))
    entropy = sum(ord(x) for x in row.row_key)
    rng = np.random.default_rng(np.random.SeedSequence([seed, entropy, 0x6ABA]))
    return rng.integers(0, count, size=(n, row.synapses_per_connection), dtype=np.int64)


def _beta_g0(tau_rise: float, tau_decay: float) -> float:
    peak = tau_decay * tau_rise * np.log(tau_decay / tau_rise) / (tau_decay - tau_rise)
    denominator = np.exp(-peak / tau_decay) - np.exp(-peak / tau_rise)
    return (1.0 / tau_rise - 1.0 / tau_decay) / denominator


def run_reduced(
    row: InhibitoryRow, dt_ms: float, *, scale: float | None = None,
    domain: Domain | None = None, use_source_kinetics: bool = False,
) -> Measurement:
    status = aglif_dend_status(row.post)
    c = float(status["C_m"]); cd = c * float(status["dend_C_frac"])
    cx = cd * float(status["dist_C_frac"]); caps = np.asarray((c - cd, cd - cx, cx))
    tau = float(status["tau_m"]); e_l = float(status["E_L"])
    leaks = np.asarray((1.0, float(status["dend_leak_scale"]), float(status["dist_leak_scale"])))
    gc = float(status["g_c"]); gcd = float(status["g_c_dist"])
    if scale is None: scale = row.transfer_scale
    if domain is None: domain = row.deployed_domain
    dom = DOMAIN_CODE[domain]
    n_steps = int(round((EVENT_MS + POST_MS) / dt_ms)); event_step = int(round(EVENT_MS / dt_ms))
    # vm/vd/vdist, adaptation, depolarizing current, g, g1
    current = np.zeros(7); clamped = np.zeros(7)
    k2 = float(status["k2"]); steady_adap = 0.0 if k2 == 0 else float(status["k_adap"]) * (HOLD_MV - e_l) / k2
    # Exact held passive/adaptation equilibrium.  Dendritic leak multipliers
    # mean vd/vdist need not equal the held soma voltage.
    matrix = np.asarray((
        (caps[1] / tau * leaks[1] + gc + gcd, -gcd),
        (-gcd, caps[2] / tau * leaks[2] + gcd),
    ))
    rhs = np.asarray((caps[1] / tau * leaks[1] * e_l + gc * HOLD_MV, caps[2] / tau * leaks[2] * e_l))
    vd, vx = np.linalg.solve(matrix, rhs)
    current[:3] = (HOLD_MV, vd, vx); clamped[:3] = (HOLD_MV, vd, vx)
    current[3] = steady_adap; clamped[3] = steady_adap
    external_hold = caps[0] / tau * (HOLD_MV - e_l) - gc * (vd - HOLD_MV) + steady_adap - float(status["I_e"])
    # The initialized state is the exact held equilibrium, so the long
    # pre-event interval is filled analytically instead of integrating zeros.
    voltages = np.full(n_steps + 1, HOLD_MV); conductance = np.zeros(n_steps + 1); clamp_i = np.zeros(n_steps + 1)
    tr = row.source_tau_rise_ms if use_source_kinetics else row.deployed_tau_rise_ms
    td = row.source_tau_decay_ms if use_source_kinetics else row.deployed_tau_decay_ms
    e_rev = row.source_e_rev_mV if use_source_kinetics else row.deployed_e_rev_mV
    g0 = _beta_g0(tr, td)
    def derivative(x: np.ndarray, soma_clamped: bool) -> np.ndarray:
        v = x[:3]; g = x[5]
        syn = np.zeros(3); syn[dom] = g * (e_rev - v[dom])
        d = np.empty(7)
        soma_num = -caps[0] / tau * (v[0] - e_l) + gc * (v[1] - v[0]) - x[3] + x[4] + float(status["I_e"]) + external_hold + syn[0]
        d[0] = 0.0 if soma_clamped else soma_num / caps[0]
        d[1] = (-caps[1] / tau * leaks[1] * (v[1] - e_l) + gc * (v[0] - v[1]) + gcd * (v[2] - v[1]) + syn[1]) / caps[1]
        d[2] = (-caps[2] / tau * leaks[2] * (v[2] - e_l) + gcd * (v[1] - v[2]) + syn[2]) / caps[2]
        d[3] = float(status["k_adap"]) * (v[0] - e_l) - k2 * x[3]
        d[4] = -float(status["k1"]) * x[4]
        d[5] = x[6] - x[5] / td; d[6] = -x[6] / tr
        return d
    for step in range(event_step, n_steps):
        if step == event_step:
            jump = row.source_gmax_nS * row.synapses_per_connection * float(scale) * g0
            current[6] += jump; clamped[6] += jump
        conductance[step] = current[5]
        for state, is_clamped in ((current, False), (clamped, True)):
            k_1 = derivative(state, is_clamped); k_2 = derivative(state + dt_ms * k_1 / 2, is_clamped)
            k_3 = derivative(state + dt_ms * k_2 / 2, is_clamped); k_4 = derivative(state + dt_ms * k_3, is_clamped)
            state += dt_ms * (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
        clamped[0] = HOLD_MV
        soma_num = -caps[0] / tau * (clamped[0] - e_l) + gc * (clamped[1] - clamped[0]) - clamped[3] + clamped[4] + float(status["I_e"]) + external_hold
        if dom == 0: soma_num += clamped[5] * (e_rev - clamped[0])
        clamp_i[step + 1] = -soma_num / 1000.0
        conductance[step + 1] = current[5]; voltages[step + 1] = current[0]
    times = np.arange(n_steps + 1) * dt_ms
    return _measure(times, voltages, conductance, clamp_i)


def _summary(values: Sequence[Measurement]) -> dict[str, dict[str, float]]:
    output = {}
    for field in Measurement.__dataclass_fields__:
        a = np.asarray([getattr(x, field) for x in values], dtype=float)
        output[field] = {"median": float(np.median(a)), "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75)), "min": float(a.min()), "max": float(a.max())}
    return output


def _ratios(reduced: Measurement, source: Mapping[str, Mapping[str, float]]) -> tuple[float, float]:
    return (
        100.0 * reduced.ipsp_peak_mV / float(source["ipsp_peak_mV"]["median"]),
        100.0 * reduced.clamp_charge_nA_ms / float(source["clamp_charge_nA_ms"]["median"]),
    )


def _classify(peak: float, charge: float) -> str:
    if peak > 115.0 or charge > 110.0: return "over"
    if peak < 85.0 or charge < 90.0: return "under"
    return "faithful"


def fit_candidate(row: InhibitoryRow, source: Mapping[str, Mapping[str, float]], dt_ms: float) -> dict[str, Any]:
    target_peak = float(source["ipsp_peak_mV"]["median"])
    target_charge = float(source["clamp_charge_nA_ms"]["median"])
    options = []
    for domain in DOMAIN_CODE:
        unit = run_reduced(row, dt_ms, scale=1.0, domain=domain, use_source_kinetics=True)
        a = unit.ipsp_peak_mV / target_peak; b = unit.clamp_charge_nA_ms / target_charge
        linear_scale = (a + b) / (a*a + b*b)
        trial_scales = (linear_scale, 1.0/a, 1.0/b)
        trials = []
        for scale in trial_scales:
            scale = float(np.clip(scale, 0.002, 50.0))
            response = run_reduced(row, dt_ms, scale=scale, domain=domain, use_source_kinetics=True)
            peak, charge = _ratios(response, source)
            loss = (peak / 100.0 - 1) ** 2 + (charge / 100.0 - 1) ** 2
            trials.append({"domain": domain, "transfer_scale": scale, "transferred_gmax_nS": row.source_gmax_nS * scale, "loss": loss, "peak_percent": peak, "charge_percent": charge, "measurement": asdict(response)})
        options.append(min(trials, key=lambda x: x["loss"]))
    return min(options, key=lambda x: x["loss"])


def _one_row(row: InhibitoryRow, dt_ms: float, n_draws: int, seed: int, derive: bool) -> dict[str, Any]:
    draws = location_draws(row, n_draws, seed)
    source, hold_amp = run_neuron_batched(row, draws, dt_ms)
    source_summary = _summary(source); reduced = run_reduced(row, dt_ms)
    peak, charge = _ratios(reduced, source_summary)
    record = {"contract": asdict(row), "row_key": row.row_key, "dt_ms": dt_ms, "seed": seed, "n_location_draws": n_draws, "location_draw_indices": draws.tolist(), "source_hold_current_nA": hold_amp, "source": {"measurements": [asdict(x) for x in source], "summary": source_summary}, "deployed_reduced": asdict(reduced), "peak_percent_of_source": peak, "charge_percent_of_source": charge, "classification": _classify(peak, charge)}
    if derive: record["candidate"] = fit_candidate(row, source_summary, dt_ms)
    return record


def aggregate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in TARGETS:
        rows = [x for x in records if x["contract"]["post"] == target]
        if not rows:
            continue
        weights = np.asarray([float(x["contract"]["deployed_indegree"]) for x in rows])
        output.append({"target": target, "n_receptor_rows": len(rows), "indegree_weighted_peak_percent": float(np.average([x["peak_percent_of_source"] for x in rows], weights=weights)), "indegree_weighted_charge_percent": float(np.average([x["charge_percent_of_source"] for x in rows], weights=weights)), "over_rows": sum(x["classification"] == "over" for x in rows), "under_rows": sum(x["classification"] == "under" for x in rows), "faithful_rows": sum(x["classification"] == "faithful" for x in rows)})
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--only-row", action="append", default=[])
    parser.add_argument("--derive", action="store_true")
    parser.add_argument("--derive-from", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.derive_from:
        captures = [_load_json(path) for path in args.derive_from]
        records = [row for capture in captures for row in capture["rows"]]
        candidate_rows = []
        for index, record in enumerate(records, 1):
            row = InhibitoryRow(**record["contract"])
            print(f"[{index}/{len(records)}] fit {row.row_key}", flush=True)
            c = fit_candidate(row, record["source"]["summary"], float(record["dt_ms"]))
            candidate_rows.append({"row_key": row.row_key, "pre": row.pre, "post": row.post, "deployed_receptor": row.deployed_receptor, "source_gmax_nS": row.source_gmax_nS, "source_contacts": row.synapses_per_connection, "source_kinetics_ms": [row.source_tau_rise_ms, row.source_tau_decay_ms], "source_e_rev_mV": row.source_e_rev_mV, "transfer_scale": c["transfer_scale"], "transferred_gmax_nS": c["transferred_gmax_nS"], "domain": c["domain"], "allocation": {key: float(key == c["domain"]) for key in DOMAIN_CODE}, "peak_percent_of_source": c["peak_percent"], "charge_percent_of_source": c["charge_percent"], "source_response_gate_pass": 85 <= c["peak_percent"] <= 115 and 90 <= c["charge_percent"] <= 110})
        candidate = {"schema": "gaba-transfer-candidate/v1", "provenance": {"method": "paired-source-NEURON-vs-user_m2-IPSP-peak-and-voltage-clamp-charge", "fit_hold_mV": HOLD_MV, "fit_dt_ms": float(records[0]["dt_ms"]), "fit_seed": int(records[0]["seed"]), "not_rate_tuning": True, "deployed_unchanged": True, "reduced_mapping_uses_exact_source_pair_kinetics": True, "source_gmax_kinetics_locations_contacts_reversals_immutable": True, "source_captures": [str(x) for x in args.derive_from]}, "rows": candidate_rows}
        args.candidate.parent.mkdir(parents=True, exist_ok=True)
        args.candidate.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        return 0
    rows = configured_rows(args.config)
    if args.only_row:
        wanted = set(args.only_row); rows = [x for x in rows if x.row_key in wanted or f"{x.pre}->{x.post}" in wanted]
        if not rows: raise ValueError("--only-row matched no configured inhibitory rows")
    records = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row.row_key}", flush=True)
        records.append(_one_row(row, args.dt, args.draws, args.seed, args.derive))
    report = {"schema": "gaba-transfer-audit/v1", "protocol": {"config": str(args.config), "cpu_only": True, "hold_mV": HOLD_MV, "dt_ms": args.dt, "seed": args.seed, "n_location_draws": args.draws, "source_response_only": True, "table5_rate_tuning": False, "source_parameters_immutable": True}, "rows": records, "aggregate_by_target": aggregate(records)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.derive:
        candidate_rows = []
        for x in records:
            c = x["candidate"]; contract = x["contract"]
            candidate_rows.append({"row_key": x["row_key"], "pre": contract["pre"], "post": contract["post"], "deployed_receptor": contract["deployed_receptor"], "source_gmax_nS": contract["source_gmax_nS"], "source_contacts": contract["synapses_per_connection"], "source_kinetics_ms": [contract["source_tau_rise_ms"], contract["source_tau_decay_ms"]], "source_e_rev_mV": contract["source_e_rev_mV"], "transfer_scale": c["transfer_scale"], "transferred_gmax_nS": c["transferred_gmax_nS"], "domain": c["domain"], "allocation": {key: float(key == c["domain"]) for key in DOMAIN_CODE}, "peak_percent_of_source": c["peak_percent"], "charge_percent_of_source": c["charge_percent"], "source_response_gate_pass": 85 <= c["peak_percent"] <= 115 and 90 <= c["charge_percent"] <= 110})
        candidate = {"schema": "gaba-transfer-candidate/v1", "provenance": {"method": "paired-source-NEURON-vs-user_m2-IPSP-peak-and-voltage-clamp-charge", "fit_hold_mV": HOLD_MV, "fit_dt_ms": args.dt, "fit_seed": args.seed, "not_rate_tuning": True, "deployed_unchanged": True, "source_gmax_kinetics_locations_contacts_reversals_immutable": True}, "rows": candidate_rows}
        args.candidate.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
