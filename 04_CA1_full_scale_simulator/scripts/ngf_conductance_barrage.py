#!/usr/bin/env python3
"""Compare an isolated ModelDB NGF cell and user_m2 under an ECIII barrage.

The diagnostic deliberately bypasses the recurrent CA1 network.  It builds the
ECIII -> Neurogliaform input from ``configs/full_scale.yaml`` and replays the
same seeded spike trains in NEURON and NEST-GPU.  The default reproduces the
ModelDB source statistics: 523 independent source cells, each represented by
one event with twice the native synaptic weight.  The old 1046-independent-
synapse representation remains available for comparison.  No fitted or
ground-truth parameter file is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ca1.config import build_network_spec
from ca1.extract.modeldb_tables import extract_connectivity
from ca1.params.groundtruth import _MODELDB, _soma, neuron_session
from ca1.sim.aglif_dend import aglif_dend_status


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "full_scale.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "ngf_conductance_barrage_synchronous.json"
POST = "Neurogliaform"
AFF_NAME = "ECIII_to_Neurogliaform"
SYNAPSE_MECHANISM = "MyExp2Sid"


@dataclass(frozen=True)
class Barrage:
    afferent_mode: str
    n_synapses: int
    n_source_trains: int
    synapses_per_source: int
    rate_hz_per_synapse: float
    weight_nS: float
    raw_modeldb_weight_nS: float
    location_transfer_scale: float
    tau_rise_ms: float
    tau_decay_ms: float
    e_rev_mV: float
    delay_ms: float
    transient_ms: float
    measure_ms: float
    dt_ms: float

    @property
    def tstop_ms(self) -> float:
        return self.transient_ms + self.measure_ms

    @property
    def source_event_weight_nS(self) -> float:
        """Peak conductance delivered by one source spike."""
        return self.weight_nS * self.synapses_per_source


def _source_barrage(
    config: Path,
    transient_ms: float,
    measure_ms: float,
    dt_ms: float,
    afferent_mode: str,
) -> Barrage:
    spec = build_network_spec(config, scale=1.0, seed=12345)
    aff = next(item for item in spec.afferents if item.name == AFF_NAME)
    receptors = spec.receptors_for_post(POST)
    port = receptors.port_index(aff.receptor)

    raw = extract_connectivity(
        index=430,
        cellnumbers_index=101,
        count_mode="per_cell",
    )
    raw_aff = raw["afferents"][AFF_NAME]  # type: ignore[index]
    raw_weight = float(raw_aff["weight_nS"])  # type: ignore[index]
    n_synapses = int(round(float(aff.synapses_per_cell)))
    synapses_per_connection = int(aff.synapses_per_connection)
    n_connections = int(round(n_synapses / synapses_per_connection))
    if n_synapses != 1046:
        raise ValueError(f"expected 1046 ECIII->NGF synapses, found {n_synapses}")
    if synapses_per_connection != 2 or n_connections != 523:
        raise ValueError(
            "expected 523 ECIII->NGF sources with 2 synapses each, found "
            f"{n_connections} sources with {synapses_per_connection} synapses each"
        )
    if afferent_mode == "synchronous-pairs":
        n_source_trains = n_connections
        synapses_per_source = synapses_per_connection
    elif afferent_mode == "independent-1046":
        n_source_trains = n_synapses
        synapses_per_source = 1
    else:
        raise ValueError(f"unknown afferent mode: {afferent_mode}")
    return Barrage(
        afferent_mode=afferent_mode,
        n_synapses=n_synapses,
        n_source_trains=n_source_trains,
        synapses_per_source=synapses_per_source,
        rate_hz_per_synapse=float(aff.rate_hz),
        weight_nS=float(aff.weight_nS),
        raw_modeldb_weight_nS=raw_weight,
        location_transfer_scale=float(aff.weight_nS) / raw_weight,
        tau_rise_ms=float(receptors.tau_rise[port]),
        tau_decay_ms=float(receptors.tau_decay[port]),
        e_rev_mV=float(receptors.E_rev[port]),
        delay_ms=float(aff.delay_ms),
        transient_ms=float(transient_ms),
        measure_ms=float(measure_ms),
        dt_ms=float(dt_ms),
    )


def poisson_trains(barrage: Barrage, seed: int) -> list[np.ndarray]:
    """Generate independent grid-aligned homogeneous Poisson source trains.

    Conditional on its Poisson event count, each train's times are uniform over
    the simulation window.  This is the standard construction of a homogeneous
    Poisson process.  Unique 0.1-ms slots make the trains directly replayable by
    both VecStim and NEST-GPU spike_generator.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xEC111]))
    n_slots = int(np.floor(barrage.tstop_ms / barrage.dt_ms))
    duration_s = barrage.tstop_ms / 1000.0
    trains: list[np.ndarray] = []
    for _ in range(barrage.n_source_trains):
        count = int(rng.poisson(barrage.rate_hz_per_synapse * duration_s))
        if count == 0:
            trains.append(np.empty(0, dtype=float))
            continue
        slots = np.sort(rng.choice(np.arange(1, n_slots), size=count, replace=False))
        trains.append(slots.astype(float) * barrage.dt_ms)
    return trains


def _neuron_seed_task(task: tuple[Barrage, int, tuple[float, ...]]) -> list[dict[str, Any]]:
    barrage, seed, scales = task
    h = neuron_session()
    h.load_file(str(_MODELDB / "cells" / "class_ngfcell.hoc"))
    h.dt = barrage.dt_ms

    trains = poisson_trains(barrage, seed)
    location_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x10_CA_7E]))

    cells = [h.ngfcell(scale_idx, scale_idx, 0) for scale_idx, _ in enumerate(scales)]
    somas = [_soma(cell) for cell in cells]
    segment_tables = [
        [segment for section in cell.dend for segment in section]
        for cell in cells
    ]
    n_locations = len(segment_tables[0])
    if n_locations == 0 or any(len(table) != n_locations for table in segment_tables):
        raise RuntimeError("NGF dendritic segment tables are empty or inconsistent")
    location_indices = location_rng.integers(0, n_locations, size=barrage.n_source_trains)

    # One schedule per independent source is shared across scale
    # conditions, so each scale receives exactly the same realization and
    # location draw.  This ModelDB build does not expose VecStim, therefore
    # source-less NetCons receive explicit absolute-time events after finitialize.
    synapses: list[Any] = []
    netcons: list[Any] = []
    netcons_by_train: list[list[Any]] = []
    for syn_idx, _times in enumerate(trains):
        location_idx = int(location_indices[syn_idx])
        train_connections: list[Any] = []
        for scale_idx, scale in enumerate(scales):
            segment = segment_tables[scale_idx][location_idx]
            synapse = h.MyExp2Sid(segment.x, sec=segment.sec)
            synapse.tau1 = barrage.tau_rise_ms
            synapse.tau2 = barrage.tau_decay_ms
            synapse.e = barrage.e_rev_mV
            connection = h.NetCon(None, synapse)
            # MyExp2Sid is a linear conductance synapse.  Therefore two
            # synchronous, co-located contacts with the same kinetics and
            # reversal sum exactly to one event at twice the native weight.
            connection.weight[0] = (
                barrage.source_event_weight_nS * float(scale) / 1000.0
            )
            synapses.append(synapse)
            netcons.append(connection)
            train_connections.append(connection)
        netcons_by_train.append(train_connections)

    spike_vectors: list[Any] = []
    detectors: list[Any] = []
    for soma in somas:
        spikes = h.Vector()
        detector = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        detector.threshold = -15.0  # ngfcell.connect_pre() uses the same threshold
        detector.record(spikes)
        spike_vectors.append(spikes)
        detectors.append(detector)

    h.finitialize(-65.0)
    for times, train_connections in zip(trains, netcons_by_train, strict=True):
        arrivals = times + barrage.delay_ms
        for arrival in arrivals[arrivals < barrage.tstop_ms]:
            for connection in train_connections:
                connection.event(float(arrival))
    h.continuerun(barrage.tstop_ms)
    window_s = barrage.measure_ms / 1000.0
    results: list[dict[str, Any]] = []
    for scale, spikes in zip(scales, spike_vectors, strict=True):
        spike_times = np.asarray(spikes, dtype=float)
        measured = spike_times[
            (spike_times >= barrage.transient_ms) & (spike_times < barrage.tstop_ms)
        ]
        results.append(
            {
                "backend": "NEURON ModelDB ngfcell",
                "afferent_mode": barrage.afferent_mode,
                "seed": seed,
                "scale": float(scale),
                "n_spikes": int(measured.size),
                "rate_hz": float(measured.size / window_s),
            }
        )
    return results


def run_neuron(
    barrage: Barrage,
    seeds: Sequence[int],
    scales: Sequence[float],
    processes: int,
) -> list[dict[str, Any]]:
    tasks = [(barrage, int(seed), tuple(scales)) for seed in seeds]
    if processes == 1:
        groups = [_neuron_seed_task(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with context.Pool(min(processes, len(tasks))) as pool:
            groups = pool.map(_neuron_seed_task, tasks)
    return [row for group in groups for row in group]


def run_reduced(
    barrage: Barrage,
    seeds: Sequence[int],
    scales: Sequence[float],
    gpu: int,
) -> list[dict[str, Any]]:
    # Must be set before the lazy nestgpu import.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import nestgpu as ngpu  # noqa: PLC0415

    ngpu.SetKernelStatus("verbosity_level", 0)
    ngpu.SetKernelStatus("rnd_seed", 48271)
    ngpu.SetTimeResolution(barrage.dt_ms)

    pairs = [(int(seed), float(scale)) for seed in seeds for scale in scales]
    cells = ngpu.Create("user_m2", len(pairs), 1)
    status = aglif_dend_status(POST)
    ngpu.SetStatus(cells, status)
    ngpu.SetStatus(
        cells,
        {
            "E_rev": [barrage.e_rev_mV],
            "tau_rise": [barrage.tau_rise_ms],
            "tau_decay": [barrage.tau_decay_ms],
            "compartment": [1.0],  # ECIII wide_dend_mean -> proximal domain
        },
    )
    e_l = float(status["E_L"])
    ngpu.SetStatus(
        cells,
        {
            "V_m": e_l,
            "V_d": e_l,
            "V_dist": e_l,
            "I_adap": 0.0,
            "I_dep": 0.0,
            "refractory_step": 0.0,
        },
    )
    ngpu.ActivateRecSpikeTimes(cells, 4096)

    source_count = len(seeds) * barrage.n_source_trains
    sources = ngpu.Create("spike_generator", source_count)
    for seed_idx, seed in enumerate(seeds):
        for syn_idx, times in enumerate(poisson_trains(barrage, int(seed))):
            if times.size == 0:
                continue
            source_idx = seed_idx * barrage.n_source_trains + syn_idx
            ngpu.SetStatus(
                sources[source_idx:source_idx + 1],
                {
                    "spike_times": times.tolist(),
                    "spike_gen_mul": [1.0] * int(times.size),
                },
            )

    for cell_idx, (seed, scale) in enumerate(pairs):
        seed_idx = list(seeds).index(seed)
        first = seed_idx * barrage.n_source_trains
        source_group = sources[first:first + barrage.n_source_trains]
        ngpu.Connect(
            source_group,
            cells[cell_idx:cell_idx + 1],
            {"rule": "all_to_all"},
            {
                "weight": barrage.source_event_weight_nS * scale,
                "delay": barrage.delay_ms,
                "receptor": 0,
            },
        )

    ngpu.Simulate(barrage.tstop_ms)
    recorded = ngpu.GetRecSpikeTimes(cells)
    window_s = barrage.measure_ms / 1000.0
    results: list[dict[str, Any]] = []
    for cell_idx, (seed, scale) in enumerate(pairs):
        raw_times = [] if recorded is None or recorded[cell_idx] is None else recorded[cell_idx]
        spike_times = np.asarray(raw_times, dtype=float)
        measured = spike_times[
            (spike_times >= barrage.transient_ms) & (spike_times < barrage.tstop_ms)
        ]
        results.append(
            {
                "backend": "NEST-GPU user_m2",
                "afferent_mode": barrage.afferent_mode,
                "seed": seed,
                "scale": scale,
                "n_spikes": int(measured.size),
                "rate_hz": float(measured.size / window_s),
            }
        )
    return results


def run_reduced_cpu_replay(
    barrage: Barrage,
    seeds: Sequence[int],
    scales: Sequence[float],
) -> list[dict[str, Any]]:
    """Replay the checked-in user_m2 equations when CUDA is unavailable.

    This is an explicit diagnostic fallback, not a substitute for the default
    NEST-GPU run.  It uses RK4 at the deployment 0.1-ms step and the same
    normalized beta-conductance equations as ``nest-gpu/src/user_m2*.{cu,h}``.
    """
    status = aglif_dend_status(POST)
    e_l = float(status["E_L"])
    dt = barrage.dt_ms
    n_steps = int(round(barrage.tstop_ms / dt))
    scale_array = np.asarray(scales, dtype=float)
    n_scales = scale_array.size

    tau_rise = barrage.tau_rise_ms
    tau_decay = barrage.tau_decay_ms
    peak_time = (
        tau_decay * tau_rise * np.log(tau_decay / tau_rise)
        / (tau_decay - tau_rise)
    )
    denom = np.exp(-peak_time / tau_decay) - np.exp(-peak_time / tau_rise)
    g0 = (1.0 / tau_rise - 1.0 / tau_decay) / denom

    c_m = float(status["C_m"])
    c_dend = c_m * float(status["dend_C_frac"])
    c_dist = c_dend * float(status["dist_C_frac"])
    c_prox = c_dend - c_dist
    c_soma = c_m - c_dend

    def derivative(state: np.ndarray, refractory: np.ndarray) -> np.ndarray:
        vm, vd, vdist, i_adap, i_dep, conductance, g1 = state.T
        voltage = np.where(
            refractory > 0,
            float(status["V_reset"]),
            np.minimum(vm, float(status["V_peak"])),
        )
        soma_leak = -(c_soma / float(status["tau_m"])) * (voltage - e_l)
        prox_leak = (
            -(c_prox / float(status["tau_m"]))
            * float(status["dend_leak_scale"])
            * (vd - e_l)
        )
        dist_leak = (
            -(c_dist / float(status["tau_m"]))
            * float(status["dist_leak_scale"])
            * (vdist - e_l)
        )
        output = np.empty_like(state)
        output[:, 0] = np.where(
            refractory > 0,
            0.0,
            (
                soma_leak
                + float(status["g_c"]) * (vd - voltage)
                - i_adap
                + i_dep
                + float(status["I_e"])
            )
            / c_soma,
        )
        output[:, 1] = (
            prox_leak
            + float(status["g_c"]) * (voltage - vd)
            + float(status["g_c_dist"]) * (vdist - vd)
            + conductance * (barrage.e_rev_mV - vd)
        ) / c_prox
        output[:, 2] = (
            dist_leak + float(status["g_c_dist"]) * (vd - vdist)
        ) / c_dist
        output[:, 3] = (
            float(status["k_adap"]) * (voltage - e_l)
            - float(status["k2"]) * i_adap
        )
        output[:, 4] = -float(status["k1"]) * i_dep
        output[:, 5] = g1 - conductance / tau_decay
        output[:, 6] = -g1 / tau_rise
        return output

    results: list[dict[str, Any]] = []
    for seed in seeds:
        event_counts = np.zeros(n_steps + 1, dtype=float)
        delay_steps = int(round(barrage.delay_ms / dt))
        for times in poisson_trains(barrage, int(seed)):
            indices = np.rint(times / dt).astype(int) + delay_steps
            indices = indices[indices <= n_steps]
            np.add.at(event_counts, indices, 1.0)

        # columns: V_m, V_d, V_dist, I_adap, I_dep, g, g1
        state = np.zeros((n_scales, 7), dtype=float)
        state[:, :3] = e_l
        refractory = np.zeros(n_scales, dtype=int)
        measured_spikes = np.zeros(n_scales, dtype=int)
        for step in range(n_steps):
            state[:, 6] += (
                event_counts[step]
                * barrage.source_event_weight_nS
                * scale_array
                * g0
            )
            k1 = derivative(state, refractory)
            k2 = derivative(state + 0.5 * dt * k1, refractory)
            k3 = derivative(state + 0.5 * dt * k2, refractory)
            k4 = derivative(state + dt * k3, refractory)
            state += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            active_refractory = refractory > 0
            state[active_refractory, 0] = float(status["V_reset"])
            refractory[active_refractory] -= 1
            firing = (~active_refractory) & (state[:, 0] >= float(status["V_th"]))
            spike_time = (step + 1) * dt
            if barrage.transient_ms <= spike_time < barrage.tstop_ms:
                measured_spikes += firing.astype(int)
            state[firing, 0] = float(status["V_reset"])
            state[firing, 3] += float(status["A2"])
            state[firing, 4] = float(status["A1"])
            refractory[firing] = int(round(float(status["t_ref"]) / dt))

        window_s = barrage.measure_ms / 1000.0
        for scale, count in zip(scales, measured_spikes, strict=True):
            results.append(
                {
                    "backend": "CPU RK4 replay of user_m2 equations",
                    "afferent_mode": barrage.afferent_mode,
                    "seed": int(seed),
                    "scale": float(scale),
                    "n_spikes": int(count),
                    "rate_hz": float(count / window_s),
                }
            )
    return results


def _summaries(rows: Sequence[dict[str, Any]], scales: Sequence[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for backend in sorted({str(row["backend"]) for row in rows}):
        by_scale: dict[str, Any] = {}
        means: list[float] = []
        for scale in scales:
            values = np.asarray(
                [row["rate_hz"] for row in rows if row["backend"] == backend and row["scale"] == scale],
                dtype=float,
            )
            mean = float(values.mean())
            means.append(mean)
            by_scale[f"{scale:g}"] = {
                "mean_hz": mean,
                "sd_hz": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "rates_hz": [float(value) for value in values],
            }
        slope = float(np.polyfit(np.asarray(scales), np.asarray(means), 1)[0])
        adjacent = [
            (means[idx + 1] - means[idx]) / (float(scales[idx + 1]) - float(scales[idx]))
            for idx in range(len(scales) - 1)
        ]
        summary[backend] = {
            "by_scale": by_scale,
            "linear_slope_hz_per_weight_scale": slope,
            "adjacent_slopes_hz_per_weight_scale": adjacent,
        }
    return summary


def _verdict(summary: dict[str, Any], reduced_backend: str) -> tuple[str, str]:
    neuron = summary["NEURON ModelDB ngfcell"]["by_scale"]["1"]["mean_hz"]
    reduced = summary[reduced_backend]["by_scale"]["1"]["mean_hz"]
    if reduced_backend != "NEST-GPU user_m2":
        return (
            "inconclusive (CUDA unavailable)",
            "NEURON was measured directly, but the reduced rate is a CPU replay of user_m2 equations and still requires the authorized GPU-2 confirmation run.",
        )
    if neuron >= 40.0 and reduced <= 30.0 and neuron - reduced >= 20.0:
        return (
            "intrinsic-gain mismatch",
            "NEURON is high while user_m2 remains low under identical input; a conductance/intrinsic re-fit is justified.",
        )
    if neuron <= 25.0 and reduced <= 25.0 and abs(neuron - reduced) <= 10.0:
        return (
            "both low / reduced response faithful",
            "The isolated cells agree at low rate; the 55.1 Hz NGF target must be recovered at network level, so intrinsic conductance re-fit is the wrong lever.",
        )
    return (
        "inconclusive",
        "The scale-1 rates do not satisfy either decisive pattern (NEURON-high/reduced-low or both-low-and-matched).",
    )


def _mode_description(barrage: dict[str, Any]) -> str:
    if barrage["afferent_mode"] == "synchronous-pairs":
        return (
            f"{barrage['n_source_trains']} independent ECIII source trains at "
            f"{barrage['rate_hz_per_synapse']} Hz, each delivering one doubled "
            f"{barrage['source_event_weight_nS']:.9g} nS event (two synchronous "
            f"{barrage['weight_nS']:.9g} nS contacts)"
        )
    return (
        f"{barrage['n_source_trains']} independent physical-synapse trains at "
        f"{barrage['rate_hz_per_synapse']} Hz, each delivering one "
        f"{barrage['weight_nS']:.9g} nS event"
    )


def _comparison(mode_results: dict[str, Any], scales: Sequence[float]) -> dict[str, Any]:
    if set(mode_results) != {"synchronous-pairs", "independent-1046"}:
        return {}
    output: dict[str, Any] = {}
    sync_summary = mode_results["synchronous-pairs"]["summary"]
    independent_summary = mode_results["independent-1046"]["summary"]
    for backend in sync_summary:
        if backend not in independent_summary:
            continue
        by_scale: dict[str, Any] = {}
        for scale in scales:
            key = f"{scale:g}"
            sync_rate = sync_summary[backend]["by_scale"][key]["mean_hz"]
            independent_rate = independent_summary[backend]["by_scale"][key]["mean_hz"]
            delta = sync_rate - independent_rate
            by_scale[key] = {
                "synchronous_mean_hz": sync_rate,
                "independent_mean_hz": independent_rate,
                "delta_hz_synchronous_minus_independent": delta,
                "percent_change": 100.0 * delta / independent_rate,
            }
        output[backend] = {"by_scale": by_scale}
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NGF afferent-statistics conductance barrage",
        "",
        "The correct ModelDB representation is `synchronous-pairs`: 523 independent source cells, each making two synchronous contacts. The implementation uses one event at twice the native weight. For a linear conductance synapse, two co-located events with identical kinetics and reversal potential sum exactly to that doubled event.",
        "",
    ]
    for mode, mode_result in report["mode_results"].items():
        barrage = mode_result["barrage"]
        summary = mode_result["summary"]
        reduced_backend = mode_result["reduced_backend"]
        lines.extend(
            [
                f"## {mode}",
                "",
                (
                    f"Input: {_mode_description(barrage)}; MyExp2Sid tau_rise/tau_decay "
                    f"{barrage['tau_rise_ms']}/{barrage['tau_decay_ms']} ms, E_rev "
                    f"{barrage['e_rev_mV']} mV. Rates use the final "
                    f"{barrage['measure_ms'] / 1000:g} s after a "
                    f"{barrage['transient_ms'] / 1000:g} s transient."
                ),
                "",
                f"| Weight scale | NEURON mean ± SD (Hz) | {reduced_backend} mean ± SD (Hz) |",
                "|---:|---:|---:|",
            ]
        )
        for scale in report["scales"]:
            key = f"{scale:g}"
            nrn = summary["NEURON ModelDB ngfcell"]["by_scale"][key]
            red = summary[reduced_backend]["by_scale"][key]
            lines.append(
                f"| {scale:g} | {nrn['mean_hz']:.3f} ± {nrn['sd_hz']:.3f} | "
                f"{red['mean_hz']:.3f} ± {red['sd_hz']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"Verdict: **{mode_result['verdict']['label']}** — {mode_result['verdict']['detail']}",
                "",
            ]
        )
    if report["comparison"]:
        lines.extend(["## Effect of corrected statistics", ""])
        for backend, comparison in report["comparison"].items():
            lines.extend(
                [
                    f"### {backend}",
                    "",
                    "| Weight scale | Synchronous pairs (Hz) | Independent 1046 (Hz) | Δ sync−independent (Hz) | Change |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for scale in report["scales"]:
                row = comparison["by_scale"][f"{scale:g}"]
                lines.append(
                    f"| {scale:g} | {row['synchronous_mean_hz']:.3f} | "
                    f"{row['independent_mean_hz']:.3f} | "
                    f"{row['delta_hz_synchronous_minus_independent']:+.3f} | "
                    f"{row['percent_change']:+.2f}% |"
                )
            lines.append("")
    reduced_note = (
        "The reduced result is a CPU RK4 replay because CUDA devices were unavailable in this sandbox; real NEST-GPU confirmation remains required."
        if report["gpu_confirmation_required"]
        else "The reduced result was run directly with NEST-GPU on GPU 2."
    )
    lines.extend(
        [
            f"Updated interpretation: {report['updated_interpretation']}",
            "",
            reduced_note,
            "",
            "GPU confirmation: `CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/ngf_conductance_barrage.py --afferent-mode synchronous-pairs`",
            "",
            "Caveat: NEURON distributes source events uniformly over ModelDB NGF dendritic segment candidates with replacement. user_m2 has no explicit morphology, so the same events are delivered to its proximal dendritic conductance port (compartment=1); its fitted dendritic coupling is otherwise unchanged.",
            "",
            "Provenance: `bezaire_modeldb/datasets/conndata_430.dat:42` (ECIII→ngfcell: 0.0035 uS, 523 connections, 2 synapses/connection), `bezaire_modeldb/datasets/syndata_120.dat` (kinetics/location), `src/ca1/sim/gpu_backend.py:367` and `src/ca1/sim/gpu_backend.py:390` (deployed indegree/weight mapping), plus `configs/full_scale.yaml` and `src/ca1/params/source_location_transfer_syndata120_budget_weighted.json` (deployed 3.466 nS effective native weight).",
        ]
    )
    return "\n".join(lines) + "\n"


def _print_report(report: dict[str, Any]) -> None:
    for mode, mode_result in report["mode_results"].items():
        barrage = mode_result["barrage"]
        reduced_backend = mode_result["reduced_backend"]
        print(f"MODE: {mode}; {_mode_description(barrage)}")
        print("scale  NEURON mean+-SD Hz   user_m2 mean+-SD Hz")
        for scale in report["scales"]:
            key = f"{scale:g}"
            neuron = mode_result["summary"]["NEURON ModelDB ngfcell"]["by_scale"][key]
            reduced = mode_result["summary"][reduced_backend]["by_scale"][key]
            print(
                f"{scale:>4g}   {neuron['mean_hz']:8.3f} +- {neuron['sd_hz']:6.3f}   "
                f"{reduced['mean_hz']:8.3f} +- {reduced['sd_hz']:6.3f}"
            )
        print(
            f"VERDICT: {mode_result['verdict']['label']} — "
            f"{mode_result['verdict']['detail']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701, 1702, 1703])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--transient-ms", type=float, default=1000.0)
    parser.add_argument("--measure-ms", type=float, default=5000.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--neuron-processes", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument(
        "--afferent-mode",
        choices=("synchronous-pairs", "independent-1046", "both"),
        default="synchronous-pairs",
        help=(
            "source statistics to replay; synchronous-pairs is the correct "
            "ModelDB representation and the default, while both is a reporting convenience"
        ),
    )
    parser.add_argument(
        "--reduced-backend",
        choices=("nestgpu", "user_m2_cpu_replay"),
        default="nestgpu",
        help="CPU replay is an explicitly non-decisive fallback when CUDA is unavailable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu != 2:
        raise ValueError("This workspace diagnostic is authorized for GPU 2 only")
    if args.scales != [1.0, 1.5, 2.0]:
        raise ValueError("this diagnostic uses the fixed weight-scale sweep 1.0, 1.5, 2.0")
    if len(args.seeds) < 3:
        raise ValueError("provide at least three seeds")
    if args.transient_ms < 1000.0 or args.measure_ms < 5000.0:
        raise ValueError("require at least 1 s transient and 5 s measurement")
    modes = (
        ["synchronous-pairs", "independent-1046"]
        if args.afferent_mode == "both"
        else [args.afferent_mode]
    )
    mode_results: dict[str, Any] = {}
    for mode in modes:
        barrage = _source_barrage(
            args.config.resolve(),
            args.transient_ms,
            args.measure_ms,
            args.dt_ms,
            mode,
        )
        neuron_rows = run_neuron(
            barrage, args.seeds, args.scales, max(1, args.neuron_processes)
        )
        if args.reduced_backend == "nestgpu":
            reduced_rows = run_reduced(barrage, args.seeds, args.scales, args.gpu)
            reduced_backend = "NEST-GPU user_m2"
        else:
            reduced_rows = run_reduced_cpu_replay(barrage, args.seeds, args.scales)
            reduced_backend = "CPU RK4 replay of user_m2 equations"
        rows = neuron_rows + reduced_rows
        summary = _summaries(rows, args.scales)
        verdict_label, verdict_detail = _verdict(summary, reduced_backend)
        barrage_dict = asdict(barrage)
        barrage_dict["source_event_weight_nS"] = barrage.source_event_weight_nS
        mode_results[mode] = {
            "barrage": barrage_dict,
            "source_spike_rate_hz_per_cell": (
                barrage.n_source_trains * barrage.rate_hz_per_synapse
            ),
            "physical_synaptic_event_rate_hz_per_cell": (
                barrage.n_source_trains
                * barrage.synapses_per_source
                * barrage.rate_hz_per_synapse
            ),
            "individual_runs": rows,
            "summary": summary,
            "reduced_backend": reduced_backend,
            "verdict": {"label": verdict_label, "detail": verdict_detail},
        }
    report = {
        "diagnostic": "isolated NGF ECIII afferent-statistics conductance barrage",
        "requested_afferent_mode": args.afferent_mode,
        "seeds": args.seeds,
        "scales": args.scales,
        "models": {
            "neuron": "ModelDB ngfcell + MyExp2Sid on uniformly sampled dendritic segments",
            "reduced": (
                "NEST-GPU user_m2, proximal dendritic port (compartment=1), GPU 2"
                if reduced_backend == "NEST-GPU user_m2"
                else "provisional CPU RK4 replay of checked-in user_m2 equations, proximal dendritic port (compartment=1); CUDA device unavailable"
            ),
        },
        "mode_results": mode_results,
        "comparison": _comparison(mode_results, args.scales),
        "updated_interpretation": (
            "The corrected scale-1 isolated NEURON reference remains high, so it still "
            "supports the conclusion that NGF is heavily suppressed in-network."
            if "synchronous-pairs" in mode_results
            and mode_results["synchronous-pairs"]["summary"]
            ["NEURON ModelDB ngfcell"]["by_scale"]["1"]["mean_hz"] >= 40.0
            else "The corrected isolated reference does not by itself support the prior in-network suppression interpretation."
        ),
        "gpu_confirmation_required": reduced_backend != "NEST-GPU user_m2",
        "gpu_confirmation_command": (
            "CUDA_VISIBLE_DEVICES=2 .venv/bin/python "
            "scripts/ngf_conductance_barrage.py --afferent-mode synchronous-pairs"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    _print_report(report)
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
