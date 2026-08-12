#!/usr/bin/env python3
"""Paired ModelDB-to-``user_m2`` unit-transfer audit.

The harness deliberately tests one biological source event instead of a network
barrage.  It keeps the source ModelDB conductance and the *deployed*,
location-transferred NetworkSpec conductance separate, and can replay the latter
at any of the three user_m2 receptor domains.  The PV pyramidal-input probe is
the first configured use; the source-row and cell-template helpers are written
so subsequent working-point rows can share the same contract.

This file has no NEST/NEST-GPU import.  The reduced half is a CPU RK4 replay of
the checked-in user_m2 equations, which makes it usable on the CPU-only audit
host.
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
from ca1.sim.aglif_dend import aglif_dend_status, aglif_dend_compartments


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "full_scale_3dtopo.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "paired_transfer_audit_pyr_to_pv.json"
DEFAULT_CANDIDATE_OUTPUT = ROOT / "results" / "charge_matched_transfer_candidate.json"
POST = "PV_Basket"
PRE = "Pyramidal"
EVENT_MS = 100.0
POST_EVENT_MS = 50.0
N_DRAWS = 32
N_CONTACTS = 3
LOCATION_SEED = 12345
Domain = Literal["distal", "proximal", "soma"]
_DOMAIN_CODE: dict[Domain, int] = {"soma": 0, "proximal": 1, "distal": 2}


@dataclass(frozen=True)
class SourceRow:
    """The source and deployed forms of one biological connection row."""

    pre: str
    post: str
    template: str
    source_gmax_nS: float
    deployed_gmax_nS: float
    synapses_per_connection: int
    tau_rise_ms: float
    tau_decay_ms: float
    e_rev_mV: float
    delay_ms: float
    section_list: str
    distance_conditions: tuple[str, ...]
    deployed_domain: Domain
    receptor: str
    kind: Literal["rec", "aff"]

    @property
    def source_event_gmax_nS(self) -> float:
        return self.source_gmax_nS * self.synapses_per_connection

    @property
    def deployed_event_gmax_nS(self) -> float:
        return self.deployed_gmax_nS * self.synapses_per_connection


@dataclass(frozen=True)
class Measurement:
    """One location-draw response; conductance is in nS ms, charge in nA ms."""

    integral_g_nS_ms: float
    epsp_peak_mV: float
    voltage_area_mV_ms: float
    clamp_charge_nA_ms: float
    time_to_peak_ms: float


def _source_syndata_entry(template: str, pre: str) -> dict[str, Any]:
    """Look up source kinetics and location rule directly from syndata_120."""
    raw = json.loads((ROOT / "src/ca1/params/syndata_120.json").read_text())
    entries = raw["entries"]
    known_afferents = {
        "Pyramidal": "pyramidalcell",
        "CA3": "ca3cell",
        "ECIII": "eccell",
    }
    pre_hoc = known_afferents.get(pre, CELL_TEMPLATES.get(pre))
    if pre_hoc is None:
        raise ValueError(f"no HOC source template for {pre}")
    matches = [
        entry
        for entry in entries
        if entry["postsynaptic"] == template
        and entry["presynaptic"] == pre_hoc
        and entry["mechanism"] == "MyExp2Sid"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one MyExp2Sid syndata row for {pre_hoc}->{template}, found {len(matches)}"
        )
    return matches[0]


def build_source_row(config: Path, post: str = POST, pre: str = PRE) -> SourceRow:
    """Build the paired contract from config, conndata, and syndata.

    ``source_gmax_nS`` is read from untransferred ModelDB/conndata 430;
    ``deployed_gmax_nS`` is the already-transferred value in NetworkSpec.
    """
    spec = build_network_spec(config, scale=1.0, seed=12345)
    projection = next((item for item in spec.projections if item.pre == pre and item.post == post), None)
    afferent = next(
        (item for item in spec.afferents if item.name.split("_to_", 1)[0] == pre and item.post == post),
        None,
    )
    if (projection is None) == (afferent is None):
        raise ValueError(f"expected exactly one configured row for {pre}->{post}")
    raw = extract_connectivity(
        index=spec.conndata_index,
        cellnumbers_index=spec.cellnumbers_index,
        count_mode=spec.conndata_count_mode,
    )
    raw_key = f"{pre}_to_{post}"
    kind: Literal["rec", "aff"] = "rec" if projection is not None else "aff"
    raw_projection = raw["projections" if kind == "rec" else "afferents"][raw_key]
    deployed = projection if projection is not None else afferent
    assert deployed is not None
    template = CELL_TEMPLATES[post]
    source = _source_syndata_entry(template, pre)
    params = source["parameters"]
    domain_code = aglif_dend_compartments(
        (deployed.receptor,),
        post,
        frozenset((deployed.receptor,)),
        spec.source_location_transfer_table,
    )[0]
    domain_by_code: dict[int, Domain] = {0: "soma", 1: "proximal", 2: "distal"}
    return SourceRow(
        pre=pre,
        post=post,
        template=template,
        source_gmax_nS=float(raw_projection["weight_nS"]),
        deployed_gmax_nS=float(deployed.weight_nS),
        synapses_per_connection=int(deployed.synapses_per_connection),
        tau_rise_ms=float(params["tau_rise"]),
        tau_decay_ms=float(params["tau_decay"]),
        e_rev_mV=float(params["e_rev"]),
        delay_ms=float(deployed.delay_ms),
        section_list=str(source["section_list"]),
        distance_conditions=tuple(str(x) for x in source["distance_conditions"]),
        deployed_domain=domain_by_code[int(domain_code)],
        receptor=str(deployed.receptor),
        kind=kind,
    )


def _distance_predicates(conditions: Iterable[str]) -> list[tuple[str, float]]:
    predicates: list[tuple[str, float]] = []
    pattern = re.compile(r"distance\(x\)(?:\*1\.0)?\s*([<>])\s*([0-9.]+)")
    for condition in conditions:
        match = pattern.fullmatch(condition.replace(" ", ""))
        if match is None:
            raise ValueError(f"unsupported syndata distance condition: {condition!r}")
        predicates.append((match.group(1), float(match.group(2))))
    return predicates


def eligible_segments(h: Any, cell: Any, row: SourceRow) -> list[Any]:
    """Expose source HOC section lists and retain exactly syndata-eligible sites."""
    soma = _soma(cell)
    sections = getattr(cell, row.section_list)
    h.distance(0.0, soma(0.5))
    predicates = _distance_predicates(row.distance_conditions)
    eligible: list[Any] = []
    for section in sections:
        for segment in section:
            distance = float(h.distance(segment.x, sec=segment.sec))
            if all(
                distance > bound if operator == ">" else distance < bound
                for operator, bound in predicates
            ):
                eligible.append(segment)
    if not eligible:
        raise RuntimeError(f"no eligible {row.section_list} segments for {row}")
    return eligible


def location_draws(h: Any, row: SourceRow, seed: int, n_draws: int) -> np.ndarray:
    """Generate one seeded with-replacement triplet table, shared by source trials."""
    cell = getattr(h, row.template)(0, 0, 0)
    count = len(eligible_segments(h, cell, row))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x50_41_49_52]))
    return rng.integers(0, count, size=(n_draws, row.synapses_per_connection))


def _baseline(trace: np.ndarray, time_ms: np.ndarray) -> float:
    window = (time_ms >= EVENT_MS - 20.0) & (time_ms < EVENT_MS)
    return float(trace[window].mean())


def _measure_voltage(
    time_ms: np.ndarray,
    voltage_mV: np.ndarray,
    conductance_nS: np.ndarray,
) -> tuple[float, float, float, float]:
    baseline = _baseline(voltage_mV, time_ms)
    response = (time_ms >= EVENT_MS) & (time_ms <= EVENT_MS + POST_EVENT_MS)
    response_time = time_ms[response]
    response_voltage = voltage_mV[response] - baseline
    peak_index = int(np.argmax(response_voltage))
    return (
        float(np.trapz(conductance_nS[response], response_time)),
        float(response_voltage[peak_index]),
        float(np.trapz(response_voltage, response_time)),
        float(response_time[peak_index] - EVENT_MS),
    )


def _measure_clamp_charge(time_ms: np.ndarray, hold_current_nA: np.ndarray) -> float:
    """Return inward event charge as a positive magnitude, baseline-subtracted."""
    baseline = _baseline(hold_current_nA, time_ms)
    response = (time_ms >= EVENT_MS) & (time_ms <= EVENT_MS + POST_EVENT_MS)
    # A positive EPSC is conventionally reported as its inward charge magnitude.
    return float(-np.trapz(hold_current_nA[response] - baseline, time_ms[response]))


def run_neuron_source(
    h: Any,
    row: SourceRow,
    draws: np.ndarray,
    dt_ms: float,
) -> tuple[list[Measurement], float]:
    """Run paired current-clamp and ideal soma voltage-clamp ModelDB trials."""
    h.dt = dt_ms
    tstop = EVENT_MS + POST_EVENT_MS

    current_cells = [getattr(h, row.template)(idx, idx, 0) for idx in range(len(draws))]
    clamp_cells = [
        getattr(h, row.template)(idx + len(draws), idx + len(draws), 0) for idx in range(len(draws))
    ]
    rest_mV = float(current_cells[0].Vrest)
    current_v: list[Any] = []
    current_g: list[list[Any]] = []
    clamp_i: list[Any] = []
    keepalive: list[Any] = []
    netcons: list[Any] = []

    def add_contacts(cell: Any, draw: np.ndarray, weight_nS: float) -> list[Any]:
        segments = eligible_segments(h, cell, row)
        conductance_vectors: list[Any] = []
        for index in draw:
            segment = segments[int(index)]
            synapse = h.MyExp2Sid(segment.x, sec=segment.sec)
            synapse.tau1 = row.tau_rise_ms
            synapse.tau2 = row.tau_decay_ms
            synapse.e = row.e_rev_mV
            connection = h.NetCon(None, synapse)
            # ModelDB MyExp2Sid uses uS; the audited source contract is nS.
            connection.weight[0] = weight_nS / 1000.0
            vector = h.Vector()
            vector.record(synapse._ref_g, dt_ms)
            keepalive.extend((synapse, connection, vector))
            netcons.append(connection)
            conductance_vectors.append(vector)
        return conductance_vectors

    for cell, draw in zip(current_cells, draws, strict=True):
        soma = _soma(cell)
        no_hold = h.IClamp(soma(0.5))
        no_hold.delay, no_hold.dur, no_hold.amp = 0.0, tstop, 0.0
        vector = h.Vector()
        vector.record(soma(0.5)._ref_v, dt_ms)
        keepalive.extend((no_hold, vector))
        current_v.append(vector)
        current_g.append(add_contacts(cell, draw, row.source_gmax_nS))

    for cell, draw in zip(clamp_cells, draws, strict=True):
        soma = _soma(cell)
        clamp = h.SEClamp(soma(0.5))
        clamp.dur1, clamp.amp1, clamp.rs = tstop, rest_mV, 1e-6
        vector = h.Vector()
        vector.record(clamp._ref_i, dt_ms)
        keepalive.extend((clamp, vector, add_contacts(cell, draw, row.source_gmax_nS)))
        clamp_i.append(vector)

    time = h.Vector()
    time.record(h._ref_t, dt_ms)
    keepalive.append(time)
    h.finitialize(rest_mV)
    for connection in netcons:
        connection.event(EVENT_MS)
    h.continuerun(tstop)
    time_ms = np.asarray(time, dtype=float)
    measurements: list[Measurement] = []
    for voltage, conductances, hold in zip(current_v, current_g, clamp_i, strict=True):
        total_g_nS = np.sum([np.asarray(g, dtype=float) * 1000.0 for g in conductances], axis=0)
        integral_g, peak, area, t_peak = _measure_voltage(
            time_ms, np.asarray(voltage, dtype=float), total_g_nS
        )
        measurements.append(
            Measurement(
                integral_g_nS_ms=integral_g,
                epsp_peak_mV=peak,
                voltage_area_mV_ms=area,
                clamp_charge_nA_ms=_measure_clamp_charge(time_ms, np.asarray(hold, dtype=float)),
                time_to_peak_ms=t_peak,
            )
        )
    return measurements, rest_mV


def _neuron_source_batch(
    task: tuple[SourceRow, np.ndarray, float],
) -> tuple[list[Measurement], float]:
    """Fresh NEURON process for a bounded location batch.

    Templates retain section objects globally, so retaining all 64 current- and
    voltage-clamp PV cells in one process is needlessly memory hungry.  A fresh
    process per small batch is still a CPU-only experiment and preserves the
    exact precomputed location triplets.
    """
    row, draws, dt_ms = task
    h = neuron_session()
    h.load_file(str(_MODELDB / "cells" / f"class_{row.template}.hoc"))
    return run_neuron_source(h, row, draws, dt_ms)


def _neuron_source_process(
    row: SourceRow,
    draws: np.ndarray,
    dt_ms: float,
    queue: Any,
) -> None:
    """Process entry point kept separate so every batch has a virgin NEURON DLL."""
    try:
        queue.put((True, _neuron_source_batch((row, draws, dt_ms))))
    except BaseException as exc:  # propagate a useful source-side error to parent
        queue.put((False, repr(exc)))


def run_neuron_source_batched(
    row: SourceRow,
    draws: np.ndarray,
    dt_ms: float,
    batch_size: int = 32,
) -> tuple[list[Measurement], float]:
    """Execute bounded source batches in truly fresh interpreter processes."""
    context = mp.get_context("spawn")
    groups: list[tuple[list[Measurement], float]] = []
    for start in range(0, len(draws), batch_size):
        queue = context.Queue()
        process = context.Process(
            target=_neuron_source_process,
            args=(row, draws[start : start + batch_size], dt_ms, queue),
        )
        process.start()
        ok, result = queue.get()
        process.join()
        if process.exitcode != 0 or not ok:
            raise RuntimeError(f"NEURON source batch failed: {result}")
        groups.append(result)
    return [item for group, _rest in groups for item in group], groups[0][1]


def _beta_g0(tau_rise_ms: float, tau_decay_ms: float) -> float:
    peak = tau_decay_ms * tau_rise_ms * np.log(tau_decay_ms / tau_rise_ms)
    peak /= tau_decay_ms - tau_rise_ms
    denominator = np.exp(-peak / tau_decay_ms) - np.exp(-peak / tau_rise_ms)
    return (1.0 / tau_rise_ms - 1.0 / tau_decay_ms) / denominator


def run_user_m2_cpu(
    row: SourceRow,
    domain: Domain,
    dt_ms: float,
    source_rest_mV: float,
    *,
    transfer_scale: float | None = None,
    allocation: Mapping[Domain, float] | None = None,
    passive_overrides: Mapping[str, float] | None = None,
    max_transfer_scale: float = 1.0,
) -> Measurement:
    """Non-spiking RK4 replay with a deployed or graded receptor budget.

    ``allocation`` splits one receptor event across soma/proximal/distal.  Its
    entries are fractions of the total transfer scale, rather than additional
    conductances, so the optimization cannot increase the source biological
    gmax.  Omitted arguments retain the deployed one-port replay contract.
    """
    status = aglif_dend_status(row.post)
    if passive_overrides:
        status.update(passive_overrides)
        if "g_c_scale" in passive_overrides:
            membrane_conductance = float(status["C_m"]) / float(status["tau_m"])
            status["g_c"] = (
                2.0 * membrane_conductance * float(passive_overrides["g_c_scale"])
            )
        if "dist_coupling_ratio" in passive_overrides or "g_c_scale" in passive_overrides:
            ratio = float(passive_overrides.get("dist_coupling_ratio", 0.25))
            status["g_c_dist"] = float(status["g_c"]) * ratio
    c_m = float(status["C_m"])
    c_dend = c_m * float(status["dend_C_frac"])
    c_dist = c_dend * float(status["dist_C_frac"])
    c_prox, c_soma = c_dend - c_dist, c_m - c_dend
    e_l = float(status["E_L"])
    event_step = int(round(EVENT_MS / dt_ms))
    n_steps = int(round((EVENT_MS + POST_EVENT_MS) / dt_ms))
    if allocation is None:
        allocation = {key: 1.0 if key == domain else 0.0 for key in _DOMAIN_CODE}
    fractions = np.asarray([float(allocation.get(key, 0.0)) for key in _DOMAIN_CODE], dtype=float)
    if np.any(fractions < 0.0) or not np.isclose(float(fractions.sum()), 1.0):
        raise ValueError("allocation must be non-negative and sum to one")
    deployed_replay = transfer_scale is None
    if transfer_scale is None:
        transfer_scale = row.deployed_gmax_nS / row.source_gmax_nS
    # Treat floating-point roundoff at the source-budget boundary as exact,
    # while never allowing an actual source-gmax increase for a new candidate.
    if max_transfer_scale <= 0.0:
        raise ValueError("max_transfer_scale must be positive")
    if not deployed_replay and not 0.0 <= transfer_scale <= max_transfer_scale + 1e-6:
        raise ValueError(f"transfer_scale must be in [0, {max_transfer_scale:g}]")
    if not deployed_replay:
        transfer_scale = min(float(transfer_scale), max_transfer_scale)
    # vm, vd, vdist, i_adap, i_dep, then (g, g1) for soma/proximal/distal.
    state = np.zeros(11, dtype=float)
    state[:3] = source_rest_mV
    conductance = np.zeros(n_steps + 1, dtype=float)
    voltages = np.zeros(n_steps + 1, dtype=float)
    hold = np.zeros(n_steps + 1, dtype=float)
    voltages[0] = state[0]
    g0 = _beta_g0(row.tau_rise_ms, row.tau_decay_ms)

    def derivative(values: np.ndarray, clamped: bool) -> np.ndarray:
        vm, vd, vdist, i_adap, i_dep = values[:5]
        g_soma, g_prox, g_dist = values[5], values[7], values[9]
        soma_leak = -(c_soma / float(status["tau_m"])) * (vm - e_l)
        prox_leak = (
            -(c_prox / float(status["tau_m"])) * float(status["dend_leak_scale"]) * (vd - e_l)
        )
        dist_leak = (
            -(c_dist / float(status["tau_m"])) * float(status["dist_leak_scale"]) * (vdist - e_l)
        )
        syn_soma = g_soma * (row.e_rev_mV - vm)
        syn_prox = g_prox * (row.e_rev_mV - vd)
        syn_dist = g_dist * (row.e_rev_mV - vdist)
        result = np.empty(11, dtype=float)
        soma_numerator = (
            soma_leak
            + float(status["g_c"]) * (vd - vm)
            - i_adap
            + i_dep
            + float(status["I_e"])
            + syn_soma
        )
        result[0] = 0.0 if clamped else soma_numerator / c_soma
        result[1] = (
            prox_leak
            + float(status["g_c"]) * (vm - vd)
            + float(status["g_c_dist"]) * (vdist - vd)
            + syn_prox
        ) / c_prox
        result[2] = (dist_leak + float(status["g_c_dist"]) * (vd - vdist) + syn_dist) / c_dist
        result[3] = float(status["k_adap"]) * (vm - e_l) - float(status["k2"]) * i_adap
        result[4] = -float(status["k1"]) * i_dep
        for offset in (5, 7, 9):
            result[offset] = values[offset + 1] - values[offset] / row.tau_decay_ms
            result[offset + 1] = -values[offset + 1] / row.tau_rise_ms
        return result

    # Current-clamp trace and separate ideal somatic voltage-clamp charge trace.
    unclamped = state.copy()
    clamped = state.copy()
    for step in range(n_steps):
        if step == event_step:
            event_gmax = row.source_event_gmax_nS * transfer_scale * fractions
            unclamped[[6, 8, 10]] += event_gmax * g0
            clamped[[6, 8, 10]] += event_gmax * g0
        conductance[step] = unclamped[[5, 7, 9]].sum()
        k1 = derivative(unclamped, False)
        k2 = derivative(unclamped + 0.5 * dt_ms * k1, False)
        k3 = derivative(unclamped + 0.5 * dt_ms * k2, False)
        k4 = derivative(unclamped + dt_ms * k3, False)
        unclamped += dt_ms * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        q1 = derivative(clamped, True)
        q2 = derivative(clamped + 0.5 * dt_ms * q1, True)
        q3 = derivative(clamped + 0.5 * dt_ms * q2, True)
        q4 = derivative(clamped + dt_ms * q3, True)
        clamped += dt_ms * (q1 + 2.0 * q2 + 2.0 * q3 + q4) / 6.0
        clamped[0] = source_rest_mV
        # Positive numerator depolarizes the soma, so the clamp injects its negative.
        hold[step + 1] = (
            -(
                -(c_soma / float(status["tau_m"])) * (clamped[0] - e_l)
                + float(status["g_c"]) * (clamped[1] - clamped[0])
                - clamped[3]
                + clamped[4]
                + float(status["I_e"])
                + clamped[5] * (row.e_rev_mV - clamped[0])
            )
            / 1000.0
        )
        conductance[step + 1] = unclamped[[5, 7, 9]].sum()
        voltages[step + 1] = unclamped[0]
    time_ms = np.arange(n_steps + 1, dtype=float) * dt_ms
    integral_g, peak, area, t_peak = _measure_voltage(time_ms, voltages, conductance)
    return Measurement(
        integral_g_nS_ms=integral_g,
        epsp_peak_mV=peak,
        voltage_area_mV_ms=area,
        clamp_charge_nA_ms=_measure_clamp_charge(time_ms, hold),
        time_to_peak_ms=t_peak,
    )


def _summary(measurements: Sequence[Measurement]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for field in Measurement.__dataclass_fields__:
        values = np.asarray([getattr(item, field) for item in measurements], dtype=float)
        output[field] = {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def _within_15(value: float, target: float) -> bool:
    return abs(value - target) / abs(target) <= 0.15


def verdict(summary: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    source = summary["source_neuron"]
    distal = summary["deployed_distal"]
    alternatives = {key: summary[key] for key in ("proximal", "soma")}
    source_charge = abs(source["clamp_charge_nA_ms"]["median"])
    source_area = abs(source["voltage_area_mV_ms"]["median"])
    distal_ratio = min(
        abs(distal["clamp_charge_nA_ms"]["median"]) / source_charge,
        abs(distal["voltage_area_mV_ms"]["median"]) / source_area,
    )
    recovered: list[str] = []
    for name, values in alternatives.items():
        if _within_15(
            values["epsp_peak_mV"]["median"], source["epsp_peak_mV"]["median"]
        ) and _within_15(abs(values["clamp_charge_nA_ms"]["median"]), source_charge):
            recovered.append(name)
    deployed_matches = _within_15(
        distal["epsp_peak_mV"]["median"], source["epsp_peak_mV"]["median"]
    ) and _within_15(abs(distal["clamp_charge_nA_ms"]["median"]), source_charge)
    if deployed_matches:
        label = "REFUTE"
        reason = "deployed distal routing already matches source peak and clamp charge within 15%"
    elif distal_ratio < 0.70 and recovered:
        label = "CONFIRM"
        reason = f"deployed distal response is below 70% of source and {', '.join(recovered)} restores peak and charge within 15% at unchanged deployed conductance"
    else:
        label = "REFUTE"
        reason = "no soma/proximal condition restores both source peak and charge within 15% at unchanged conductance"
    return {
        "label": label,
        "reason": reason,
        "distal_min_area_or_charge_ratio": distal_ratio,
        "recovered_domains": recovered,
    }


def run_probe(config: Path, dt_ms: float, n_draws: int, seed: int) -> dict[str, Any]:
    row = build_source_row(config)
    h = neuron_session()
    h.load_file(str(_MODELDB / "cells" / f"class_{row.template}.hoc"))
    draws = location_draws(h, row, seed, n_draws)
    source, source_rest_mV = run_neuron_source_batched(row, draws, dt_ms)
    reduced = {
        "deployed_distal": [run_user_m2_cpu(row, row.deployed_domain, dt_ms, source_rest_mV)]
        * n_draws,
        "proximal": [run_user_m2_cpu(row, "proximal", dt_ms, source_rest_mV)] * n_draws,
        "soma": [run_user_m2_cpu(row, "soma", dt_ms, source_rest_mV)] * n_draws,
    }
    rows: dict[str, list[Measurement]] = {"source_neuron": source, **reduced}
    summaries = {name: _summary(values) for name, values in rows.items()}
    return {
        "contract": asdict(row),
        "dt_ms": dt_ms,
        "event_ms": EVENT_MS,
        "post_event_ms": POST_EVENT_MS,
        "n_location_draws": n_draws,
        "location_seed": seed,
        "source_rest_mV": source_rest_mV,
        "location_draw_indices": draws.tolist(),
        "measurements": {name: [asdict(item) for item in values] for name, values in rows.items()},
        "summary": summaries,
        "verdict": verdict(summaries),
    }


def _format_number(summary: dict[str, float]) -> str:
    return f"{summary['median']:.5g} [{summary['q25']:.5g}, {summary['q75']:.5g}]"


def markdown_table(report: dict[str, Any]) -> str:
    labels = {
        "source_neuron": "source NEURON",
        "deployed_distal": "deployed distal",
        "proximal": "proximal",
        "soma": "soma",
    }
    lines = [
        "| condition | integral-g (nS ms) | EPSP peak (mV) | voltage area (mV ms) | clamp charge (nA ms) | time-to-peak (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        summary = report["summary"][key]
        lines.append(
            f"| {label} | {_format_number(summary['integral_g_nS_ms'])} | "
            f"{_format_number(summary['epsp_peak_mV'])} | "
            f"{_format_number(summary['voltage_area_mV_ms'])} | "
            f"{_format_number(summary['clamp_charge_nA_ms'])} | "
            f"{_format_number(summary['time_to_peak_ms'])} |"
        )
    return "\n".join(lines)


def configured_excitatory_rows(
    config: Path,
    targets: Iterable[str] = ("PV_Basket", "Bistratified", "O_LM"),
) -> list[SourceRow]:
    """Return every configured AMPA row entering the requested targets.

    This is deliberately driven by the NetworkSpec, rather than by the transfer
    JSON, so an unconfigured historical table row cannot become a candidate.
    """
    spec = build_network_spec(config, scale=1.0, seed=LOCATION_SEED)
    keys: set[tuple[str, str]] = set()
    target_set = set(targets)
    for projection in spec.projections:
        if projection.post in target_set and projection.receptor.startswith("AMPA"):
            keys.add((projection.pre, projection.post))
    for afferent in spec.afferents:
        pre = afferent.name.split("_to_", 1)[0]
        if afferent.post in target_set and afferent.receptor.startswith("AMPA"):
            keys.add((pre, afferent.post))
    return [build_source_row(config, post=post, pre=pre) for pre, post in sorted(keys)]


def _ratios(candidate: Measurement, source: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        "peak": candidate.epsp_peak_mV / source["epsp_peak_mV"]["median"],
        "charge": abs(candidate.clamp_charge_nA_ms)
        / abs(source["clamp_charge_nA_ms"]["median"]),
    }


def _allocation_from_budget(budget: np.ndarray) -> tuple[float, dict[Domain, float]]:
    """Convert conductance shares into a transfer scale and domain fractions."""
    scale = float(budget.sum())
    if scale <= 0.0:
        return 0.0, {"soma": 1.0, "proximal": 0.0, "distal": 0.0}
    return scale, {
        domain: float(budget[index] / scale)
        for index, domain in enumerate(("soma", "proximal", "distal"))
    }


def derive_charge_matched_transfer(
    row: SourceRow,
    source_summary: Mapping[str, Mapping[str, float]],
    dt_ms: float,
    source_rest_mV: float,
    passive_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit a source-response candidate using only equal peak/charge loss.

    The fitted variables are the three non-negative source-gmax shares.  Their
    sum is constrained to <= 1, which makes the candidate a conductance
    reduction of the immutable ModelDB source value.  Firing is intentionally
    absent from this objective.
    """
    source_peak = source_summary["epsp_peak_mV"]["median"]
    source_charge = abs(source_summary["clamp_charge_nA_ms"]["median"])

    def measurement(budget: np.ndarray) -> Measurement:
        scale, allocation = _allocation_from_budget(budget)
        return run_user_m2_cpu(
            row,
            "soma",
            dt_ms,
            source_rest_mV,
            transfer_scale=scale,
            allocation=allocation,
            passive_overrides=passive_overrides,
        )

    def loss(budget: np.ndarray) -> float:
        response = measurement(budget)
        peak_ratio = response.epsp_peak_mV / source_peak
        charge_ratio = abs(response.clamp_charge_nA_ms) / source_charge
        return float((peak_ratio - 1.0) ** 2 + (charge_ratio - 1.0) ** 2)

    old_scale = row.deployed_gmax_nS / row.source_gmax_nS
    old_budget = np.zeros(3, dtype=float)
    old_budget[_DOMAIN_CODE[row.deployed_domain]] = old_scale
    # Preregistered cheap search: exact unit-share responses define the three
    # response columns, then every non-empty soma/prox/dist support is solved
    # by constrained linear least squares.  The winning candidate is replayed
    # by the full nonlinear RK4 model below.  This avoids making a generic
    # optimizer's iteration count a hidden experimental degree of freedom.
    columns = []
    for index in range(3):
        unit = np.zeros(3, dtype=float)
        unit[index] = 1.0
        response = measurement(unit)
        columns.append(
            [response.epsp_peak_mV / source_peak, abs(response.clamp_charge_nA_ms) / source_charge]
        )
    response_columns = np.asarray(columns, dtype=float).T
    linear_candidates: list[np.ndarray] = [np.minimum(old_budget, 1.0)]
    for support in range(1, 8):
        indices = [index for index in range(3) if support & (1 << index)]
        solution = np.linalg.lstsq(
            response_columns[:, indices], np.ones(2), rcond=None
        )[0]
        if np.any(solution < 0.0):
            continue
        candidate = np.zeros(3, dtype=float)
        candidate[indices] = solution
        if candidate.sum() > 1.0:
            candidate /= candidate.sum()
        linear_candidates.append(candidate)
    # The source-response linearization chooses feasible candidates, while the
    # final selection always uses the preregistered *nonlinear* equal-weight
    # peak/charge RK4 loss.  There are only eight candidates (old support plus
    # seven soma/prox/dist supports), so this remains cheap and deterministic.
    new_budget = min(linear_candidates, key=loss)
    new_scale, allocation = _allocation_from_budget(new_budget)
    old = run_user_m2_cpu(row, row.deployed_domain, dt_ms, source_rest_mV)
    new = measurement(new_budget)
    old_ratios = _ratios(old, source_summary)
    new_ratios = _ratios(new, source_summary)
    return {
        "objective": {
            "name": "equal_squared_relative_peak_and_clamp_charge_error",
            "formula": "(peak_ratio-1)^2 + (charge_ratio-1)^2",
            "firing_rate_in_objective": False,
            "loss": float(loss(new_budget)),
            "search": "seven non-empty domain supports in exact unit-response linearization; nonlinear RK4 equal-loss selection",
        },
        "constraint": {
            "source_gmax_fixed": True,
            "budget_parameterization": "sum(domain_source_gmax_shares) = transfer_scale <= 1",
            "total_transfer_scale": new_scale,
            "source_gmax_nS": row.source_gmax_nS,
            "transferred_gmax_nS": row.source_gmax_nS * new_scale,
        },
        "graded_allocation": allocation,
        "old_transfer": {
            "transfer_scale": old_scale,
            "allocation": {key: float(key == row.deployed_domain) for key in _DOMAIN_CODE},
            "measurement": asdict(old),
            "peak_percent_of_source": 100.0 * old_ratios["peak"],
            "charge_percent_of_source": 100.0 * old_ratios["charge"],
        },
        "new_charge_matched_transfer": {
            "measurement": asdict(new),
            "peak_percent_of_source": 100.0 * new_ratios["peak"],
            "charge_percent_of_source": 100.0 * new_ratios["charge"],
            "charge_at_least_90_percent": new_ratios["charge"] >= 0.90,
            "peak_within_15_percent": _within_15(new.epsp_peak_mV, source_peak),
        },
    }


def _run_charge_matched_row(row: SourceRow, dt_ms: float, n_draws: int, seed: int) -> dict[str, Any]:
    h = neuron_session()
    h.load_file(str(_MODELDB / "cells" / f"class_{row.template}.hoc"))
    draws = location_draws(h, row, seed, n_draws)
    source, source_rest_mV = run_neuron_source_batched(row, draws, dt_ms)
    source_summary = _summary(source)
    derivation = derive_charge_matched_transfer(row, source_summary, dt_ms, source_rest_mV)
    return {
        "contract": asdict(row),
        "dt_ms": dt_ms,
        "n_location_draws": n_draws,
        "location_seed": seed,
        "source_rest_mV": source_rest_mV,
        "location_draw_indices": draws.tolist(),
        "source_neuron": {
            "measurements": [asdict(item) for item in source],
            "summary": source_summary,
        },
        "derivation": derivation,
    }


def _run_source_row(row: SourceRow, dt_ms: float, n_draws: int, seed: int) -> dict[str, Any]:
    """Run only the expensive source side for an independent timestep check."""
    h = neuron_session()
    h.load_file(str(_MODELDB / "cells" / f"class_{row.template}.hoc"))
    draws = location_draws(h, row, seed, n_draws)
    source, source_rest_mV = run_neuron_source_batched(row, draws, dt_ms)
    return {
        "schema": "paired-source-response/v1",
        "contract": asdict(row),
        "dt_ms": dt_ms,
        "n_location_draws": n_draws,
        "location_seed": seed,
        "source_rest_mV": source_rest_mV,
        "measurements": [asdict(item) for item in source],
        "source_summary": _summary(source),
        "location_draw_indices": draws.tolist(),
    }


def derive_candidate_from_source_capture(capture_path: Path) -> dict[str, Any]:
    """Fit the CPU reduced model to an immutable source-NEURON capture."""
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if capture.get("schema") != "paired-source-response/v1":
        raise ValueError("--derive-source-capture requires a paired source-response capture")
    if float(capture["dt_ms"]) != 0.025:
        raise ValueError("charge-matched candidate derivation is preregistered at dt=0.025 ms")
    row = SourceRow(**capture["contract"])
    derivation = derive_charge_matched_transfer(
        row, capture["source_summary"], 0.025, float(capture["source_rest_mV"])
    )
    return {
        "schema": "charge-matched-transfer-candidate/v1",
        "provenance": {
            "method": "paired-source-NEURON-vs-user_m2-CPU-RK4",
            "source": "ModelDB conndata_430 + syndata_120",
            "fit_response": "somatic EPSP peak and ideal soma voltage-clamp charge",
            "not_rate_tuning": True,
            "firing_rate_in_objective": False,
            "deployed_table_unchanged": True,
            "source_capture": str(capture_path),
            "gpu_validation_next": "small recurrent loop then full-scale validation",
        },
        "primary_dt_ms": 0.025,
        "n_location_draws": capture["n_location_draws"],
        "rows": [{
            "contract": capture["contract"],
            "dt_ms": capture["dt_ms"],
            "n_location_draws": capture["n_location_draws"],
            "location_seed": capture["location_seed"],
            "source_rest_mV": capture["source_rest_mV"],
            "location_draw_indices": capture["location_draw_indices"],
            "source_neuron": {
                "measurements": capture["measurements"],
                "summary": capture["source_summary"],
            },
            "derivation": derivation,
        }],
        "dt_0p05_replay": [],
    }


def verify_charge_matched_dt_stability_from_capture(
    candidate_path: Path, source_capture_path: Path
) -> dict[str, Any]:
    """Check a candidate against a separately captured 0.05-ms source response."""
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    capture = json.loads(source_capture_path.read_text(encoding="utf-8"))
    if candidate.get("schema") != "charge-matched-transfer-candidate/v1":
        raise ValueError("--verify-candidate requires a charge-matched candidate report")
    if capture.get("schema") != "paired-source-response/v1" or float(capture["dt_ms"]) != 0.05:
        raise ValueError("--verify-source-capture requires a dt=0.05 paired source-response capture")
    if len(candidate["rows"]) != 1:
        raise ValueError("capture-backed stability verification expects one candidate row")
    record = candidate["rows"][0]
    if record["contract"]["pre"] != capture["contract"]["pre"] or record["contract"]["post"] != capture["contract"]["post"]:
        raise ValueError("candidate and source capture rows do not match")
    row = SourceRow(**capture["contract"])
    derived = record["derivation"]
    response = run_user_m2_cpu(
        row, "soma", 0.05, float(capture["source_rest_mV"]),
        transfer_scale=float(derived["constraint"]["total_transfer_scale"]),
        allocation=derived["graded_allocation"],
    )
    source_summary = capture["source_summary"]
    ratios = _ratios(response, source_summary)
    return {
        "schema": "charge-matched-transfer-stability/v1",
        "candidate_report": str(candidate_path),
        "source_capture": str(source_capture_path),
        "dt_ms": 0.05,
        "n_location_draws": capture["n_location_draws"],
        "rows": [{
            "pre": row.pre, "post": row.post, "dt_ms": 0.05,
            "source_summary": source_summary,
            "candidate_measurement": asdict(response),
            "peak_percent_of_source": 100.0 * ratios["peak"],
            "charge_percent_of_source": 100.0 * ratios["charge"],
            "peak_within_15_percent": _within_15(response.epsp_peak_mV, source_summary["epsp_peak_mV"]["median"]),
            "charge_at_least_90_percent": ratios["charge"] >= 0.90,
        }],
    }


def run_charge_matched_audit(
    config: Path,
    n_draws: int,
    seed: int,
    only_rows: Sequence[str] = (),
    include_dt_stability: bool = True,
) -> dict[str, Any]:
    """Derive 0.025-ms candidates and independently replay them at 0.05 ms."""
    rows = configured_excitatory_rows(config)
    if only_rows:
        wanted = set(only_rows)
        rows = [row for row in rows if f"{row.pre}->{row.post}" in wanted]
        missing = wanted - {f"{row.pre}->{row.post}" for row in rows}
        if missing:
            raise ValueError(f"--only-row is not a configured AMPA row: {sorted(missing)}")
    primary = [_run_charge_matched_row(row, 0.025, n_draws, seed) for row in rows]
    stability: list[dict[str, Any]] = []
    if not include_dt_stability:
        return {
            "schema": "charge-matched-transfer-candidate/v1",
            "provenance": {
                "method": "paired-source-NEURON-vs-user_m2-CPU-RK4",
                "source": "ModelDB conndata_430 + syndata_120",
                "fit_response": "somatic EPSP peak and ideal soma voltage-clamp charge",
                "not_rate_tuning": True,
                "firing_rate_in_objective": False,
                "deployed_table_unchanged": True,
                "gpu_validation_next": "small recurrent loop then full-scale validation",
            },
            "primary_dt_ms": 0.025,
            "n_location_draws": n_draws,
            "rows": primary,
            "dt_0p05_replay": stability,
        }
    for record, row in zip(primary, rows, strict=True):
        checked = _run_source_row(row, 0.05, n_draws, seed)
        # Do not refit at 0.05: this is an independent numerical replay of the
        # 0.025-ms candidate, against a separately sampled source response.
        candidate = record["derivation"]
        scale = float(candidate["constraint"]["total_transfer_scale"])
        allocation = candidate["graded_allocation"]
        source_summary = checked["source_summary"]
        response = run_user_m2_cpu(
            row, "soma", 0.05, float(checked["source_rest_mV"]),
            transfer_scale=scale, allocation=allocation,
        )
        ratios = _ratios(response, source_summary)
        stability.append({
            "pre": row.pre,
            "post": row.post,
            "dt_ms": 0.05,
            "source_summary": source_summary,
            "candidate_measurement": asdict(response),
            "peak_percent_of_source": 100.0 * ratios["peak"],
            "charge_percent_of_source": 100.0 * ratios["charge"],
            "peak_within_15_percent": _within_15(response.epsp_peak_mV, source_summary["epsp_peak_mV"]["median"]),
            "charge_at_least_90_percent": ratios["charge"] >= 0.90,
        })
    return {
        "schema": "charge-matched-transfer-candidate/v1",
        "provenance": {
            "method": "paired-source-NEURON-vs-user_m2-CPU-RK4",
            "source": "ModelDB conndata_430 + syndata_120",
            "fit_response": "somatic EPSP peak and ideal soma voltage-clamp charge",
            "not_rate_tuning": True,
            "firing_rate_in_objective": False,
            "deployed_table_unchanged": True,
            "gpu_validation_next": "small recurrent loop then full-scale validation",
        },
        "primary_dt_ms": 0.025,
        "n_location_draws": n_draws,
        "rows": primary,
        "dt_0p05_replay": stability,
    }


def verify_charge_matched_dt_stability(
    config: Path,
    candidate_path: Path,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Replay an existing 0.025-ms candidate at 0.05 ms without refitting."""
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("schema") != "charge-matched-transfer-candidate/v1":
        raise ValueError("--verify-candidate requires a charge-matched candidate report")
    stability: list[dict[str, Any]] = []
    for record in candidate["rows"]:
        contract = record["contract"]
        row = build_source_row(config, post=contract["post"], pre=contract["pre"])
        checked = _run_source_row(row, 0.05, n_draws, seed)
        derived = record["derivation"]
        response = run_user_m2_cpu(
            row,
            "soma",
            0.05,
            float(checked["source_rest_mV"]),
            transfer_scale=float(derived["constraint"]["total_transfer_scale"]),
            allocation=derived["graded_allocation"],
        )
        source_summary = checked["source_summary"]
        ratios = _ratios(response, source_summary)
        stability.append({
            "pre": row.pre,
            "post": row.post,
            "dt_ms": 0.05,
            "source_summary": source_summary,
            "candidate_measurement": asdict(response),
            "peak_percent_of_source": 100.0 * ratios["peak"],
            "charge_percent_of_source": 100.0 * ratios["charge"],
            "peak_within_15_percent": _within_15(response.epsp_peak_mV, source_summary["epsp_peak_mV"]["median"]),
            "charge_at_least_90_percent": ratios["charge"] >= 0.90,
        })
    return {
        "schema": "charge-matched-transfer-stability/v1",
        "candidate_report": str(candidate_path),
        "dt_ms": 0.05,
        "n_location_draws": n_draws,
        "rows": stability,
    }


def charge_matched_markdown_table(report: Mapping[str, Any]) -> str:
    stability = {(row["pre"], row["post"]): row for row in report["dt_0p05_replay"]}
    lines = [
        "| row | old peak % | old charge % | new peak % | new charge % | scale | soma / prox / dist | 0.05-ms new peak / charge % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["rows"]:
        contract = item["contract"]
        derived = item["derivation"]
        old = derived["old_transfer"]
        new = derived["new_charge_matched_transfer"]
        allocation = derived["graded_allocation"]
        checked = stability.get((contract["pre"], contract["post"]))
        stability_text = (
            f"{checked['peak_percent_of_source']:.1f} / {checked['charge_percent_of_source']:.1f}"
            if checked is not None
            else "pending"
        )
        lines.append(
            f"| {contract['pre']}→{contract['post']} | {old['peak_percent_of_source']:.1f} | "
            f"{old['charge_percent_of_source']:.1f} | {new['peak_percent_of_source']:.1f} | "
            f"{new['charge_percent_of_source']:.1f} | {derived['constraint']['total_transfer_scale']:.4f} | "
            f"{allocation['soma']:.3f} / {allocation['proximal']:.3f} / {allocation['distal']:.3f} | "
            f"{stability_text} |"
        )
    return "\n".join(lines)


def merge_charge_matched_candidate_reports(input_paths: Sequence[Path]) -> dict[str, Any]:
    """Merge independently completed row audits without refitting or rewriting a table."""
    if not input_paths:
        raise ValueError("at least one --merge-candidate input is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    candidates = [report for report in reports if report["schema"] == "charge-matched-transfer-candidate/v1"]
    checks = [report for report in reports if report["schema"] == "charge-matched-transfer-stability/v1"]
    if not candidates or len(candidates) + len(checks) != len(reports):
        raise ValueError("merge inputs must be candidate or dt-stability reports")
    reference = candidates[0]
    rows = [row for report in candidates for row in report["rows"]]
    stability = [row for report in candidates for row in report["dt_0p05_replay"]]
    stability.extend(row for report in checks for row in report["rows"])
    identities = [(row["contract"]["pre"], row["contract"]["post"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("candidate reports contain duplicate pre/post rows")
    stability_identities = [(row["pre"], row["post"]) for row in stability]
    if len(set(stability_identities)) != len(stability_identities):
        raise ValueError("candidate reports contain duplicate dt-stability rows")
    return {
        "schema": reference["schema"],
        "provenance": reference["provenance"],
        "primary_dt_ms": reference["primary_dt_ms"],
        "n_location_draws": reference["n_location_draws"],
        "rows": sorted(rows, key=lambda row: (row["contract"]["pre"], row["contract"]["post"])),
        "dt_0p05_replay": sorted(stability, key=lambda row: (row["pre"], row["post"])),
        "merged_from": [str(path) for path in input_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=LOCATION_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--derive-charge-matched",
        action="store_true",
        help="audit all configured AMPA rows for PV/Bistratified/O_LM and write candidate-only values",
    )
    parser.add_argument(
        "--skip-dt-stability",
        action="store_true",
        help="write only the 0.025-ms candidate; use --verify-candidate for a separate 0.05-ms replay",
    )
    parser.add_argument(
        "--verify-candidate",
        type=Path,
        metavar="CANDIDATE.json",
        help="independently replay a candidate at 0.05 ms without refitting",
    )
    parser.add_argument(
        "--capture-source",
        action="store_true",
        help="capture one source-NEURON row for later reduced-model derivation",
    )
    parser.add_argument(
        "--derive-source-capture",
        type=Path,
        metavar="SOURCE.json",
        help="derive one 0.025-ms candidate from an existing source capture",
    )
    parser.add_argument(
        "--verify-source-capture",
        type=Path,
        metavar="SOURCE_0p05.json",
        help="use an existing 0.05-ms source capture with --verify-candidate",
    )
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    parser.add_argument(
        "--only-row",
        action="append",
        default=[],
        metavar="PRE->POST",
        help="limit candidate derivation to a configured row (repeatable; audit debugging only)",
    )
    parser.add_argument(
        "--merge-candidate",
        action="append",
        type=Path,
        default=[],
        metavar="REPORT.json",
        help="merge independently completed candidate-only row reports; does not refit",
    )
    args = parser.parse_args()
    if args.dt <= 0.0 or args.draws <= 0:
        raise ValueError("--dt and --draws must be positive")
    if args.capture_source:
        if args.derive_charge_matched or args.merge_candidate or args.derive_source_capture:
            raise ValueError("--capture-source cannot be combined with derive or merge options")
        if len(args.only_row) != 1:
            raise ValueError("--capture-source requires exactly one --only-row PRE->POST")
        pre, separator, post = args.only_row[0].partition("->")
        if not separator:
            raise ValueError("--only-row must be PRE->POST")
        row = build_source_row(args.config.resolve(), post=post, pre=pre)
        report = _run_source_row(row, args.dt, args.draws, args.seed)
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote source capture {args.candidate_output}")
        return
    if args.derive_source_capture:
        if args.derive_charge_matched or args.merge_candidate or args.verify_candidate:
            raise ValueError("--derive-source-capture cannot be combined with derive, verify, or merge")
        report = derive_candidate_from_source_capture(args.derive_source_capture)
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(charge_matched_markdown_table(report))
        print(f"wrote candidate-only {args.candidate_output}")
        return
    if args.verify_candidate:
        if args.derive_charge_matched or args.merge_candidate:
            raise ValueError("--verify-candidate cannot be combined with derive or merge options")
        report = (
            verify_charge_matched_dt_stability_from_capture(
                args.verify_candidate, args.verify_source_capture
            )
            if args.verify_source_capture
            else verify_charge_matched_dt_stability(
                args.config.resolve(), args.verify_candidate, args.draws, args.seed
            )
        )
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote dt-stability report {args.candidate_output}")
        return
    if args.merge_candidate:
        if args.derive_charge_matched:
            raise ValueError("--merge-candidate cannot be combined with --derive-charge-matched")
        report = merge_charge_matched_candidate_reports(args.merge_candidate)
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(charge_matched_markdown_table(report))
        print(f"wrote merged candidate-only {args.candidate_output}")
        return
    if args.derive_charge_matched:
        report = run_charge_matched_audit(
            args.config.resolve(), args.draws, args.seed, args.only_row,
            include_dt_stability=not args.skip_dt_stability,
        )
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(charge_matched_markdown_table(report))
        print(f"wrote candidate-only {args.candidate_output}")
        return
    report = run_probe(args.config.resolve(), args.dt, args.draws, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(markdown_table(report))
    print()
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
