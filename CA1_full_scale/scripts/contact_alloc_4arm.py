#!/usr/bin/env python3
"""Decisive CPU-only four-arm contact-allocation replay.

This candidate diagnostic never builds or changes a deployed network.  It uses
the persisted biological source graph and recurrent spike trains, reconstructs
the historical CA3/ECIII source trains, and compares deployed port partitioning
with ModelDB's per-contact uniform draw over eligible synapse objects.  Native
NEURON arms use the checked-in ModelDB cell templates and source contracts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_network_clamp_replay as EXACT  # noqa: E402
import gaba_transfer_audit as GABA  # noqa: E402
import paired_transfer_audit as PAIRED  # noqa: E402

from ca1.config import build_network_spec  # noqa: E402
from ca1.params.groundtruth import _MODELDB, _soma, neuron_session  # noqa: E402


TARGETS = ("PV_Basket", "Bistratified", "O_LM")
EXCITATORY = frozenset(("Pyramidal", "CA3", "ECIII"))
CONTACT_SEEDS = (12345, 12346, 12347)
PRIMARY_DT_MS = 0.025
CHECK_DT_MS = 0.05
SAVED_AFFERENT_SEED = 12345
CONTROL_AFFERENT_SEED = 12346
DEFAULT_OUTPUT = ROOT / "results" / "contact_alloc_4arm.json"
DEFAULT_ELIGIBLE_OUTPUT = ROOT / "results" / "contact_alloc_eligible_segments.json"
DEFAULT_MARKDOWN = ROOT / "scratchpad" / "contact_alloc_4arm.md"
DOMAIN_NAMES = ("soma", "proximal", "distal")
RECRUIT_RATE_HZ = 1.0
RECRUIT_FRACTION = 0.5
PARTIAL_DEPOLARIZATION_MV = 1.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join(str(x) for x in (seed, *parts)).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(payload).digest()[:16], dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


def _spatial_panel(run: h5py.File, target: str, count: int) -> list[int]:
    positions = np.asarray(run["cell_positions"][target], dtype=float)
    ids = np.arange(len(positions), dtype=np.int64)
    order = ids[np.lexsort((ids, positions[:, 2], positions[:, 1], positions[:, 0]))]
    return [int(block[len(block) // 2]) for block in np.array_split(order, count) if len(block)]


def _contracts(config: Path) -> dict[tuple[str, str], list[Any]]:
    output: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in PAIRED.configured_excitatory_rows(config, TARGETS):
        output[(row.pre, row.post)].append(row)
    for row in GABA.configured_rows(config):
        output[(row.pre, row.post)].append(row)
    return dict(output)


def _site_records(h: Any, cell: Any, rows: Sequence[Any]) -> list[tuple[Any, Any]]:
    sites: list[tuple[Any, Any]] = []
    for row in rows:
        eligible = (
            PAIRED.eligible_segments(h, cell, row)
            if row.pre in EXCITATORY else GABA.eligible_segments(h, cell, row)
        )
        sites.extend((row, segment) for segment in eligible)
    if not sites:
        raise RuntimeError(f"no eligible sites for {rows[0].pre}->{rows[0].post}")
    return sites


def _eligible_worker(config: str, queue: Any) -> None:
    try:
        config_path = Path(config)
        contracts = _contracts(config_path)
        h = neuron_session()
        records = []
        loaded: set[str] = set()
        for target in TARGETS:
            rows_for_target = [rows for (pre, post), rows in contracts.items() if post == target]
            template = rows_for_target[0][0].template
            if template not in loaded:
                h.load_file(str(_MODELDB / "cells" / f"class_{template}.hoc"))
                loaded.add(template)
            cell = getattr(h, template)(0, 0, 0)
            for rows in sorted(rows_for_target, key=lambda x: x[0].pre):
                row_records = []
                for row in rows:
                    segments = (
                        PAIRED.eligible_segments(h, cell, row)
                        if row.pre in EXCITATORY else GABA.eligible_segments(h, cell, row)
                    )
                    row_records.append({
                        "section_list": row.section_list,
                        "distance_conditions": list(row.distance_conditions),
                        "source_kinetics_ms": [
                            float(row.tau_rise_ms), float(row.tau_decay_ms)
                        ] if row.pre in EXCITATORY else [
                            float(row.source_tau_rise_ms), float(row.source_tau_decay_ms)
                        ],
                        "source_e_rev_mV": float(row.e_rev_mV) if row.pre in EXCITATORY else float(row.source_e_rev_mV),
                        "source_gmax_nS": float(row.source_gmax_nS),
                        "contacts": int(row.synapses_per_connection),
                        "eligible_segment_count": len(segments),
                        "eligible_segments": [{
                            "section": str(segment.sec.name()), "x": float(segment.x),
                            "distance_um": float(h.distance(segment.x, sec=segment.sec)),
                        } for segment in segments],
                        "reduced_domain": str(row.deployed_domain),
                        "deployed_receptor": str(row.receptor) if row.pre in EXCITATORY else str(row.deployed_receptor),
                        "deployed_port_indegree": None if row.pre in EXCITATORY else float(row.deployed_indegree),
                        "biological_indegree": None if row.pre in EXCITATORY else float(row.biological_indegree),
                    })
                total = sum(x["eligible_segment_count"] for x in row_records)
                records.append({
                    "pre": rows[0].pre, "post": target,
                    "template": template, "rows": row_records,
                    "eligible_segment_count": total,
                    "domain_probabilities": {
                        domain: sum(
                            x["eligible_segment_count"] for x in row_records
                            if x["reduced_domain"] == domain
                        ) / total for domain in DOMAIN_NAMES
                    },
                })
        queue.put((True, records))
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        queue.put((False, repr(exc)))


def eligible_report(config: Path) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_eligible_worker, args=(str(config), queue))
    process.start()
    ok, value = queue.get()
    process.join()
    if process.exitcode != 0 or not ok:
        raise RuntimeError(f"eligible-site probe failed: {value}")
    return value


def _port_counts(eligible: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[int]]:
    return {
        (str(item["pre"]), str(item["post"])): [
            int(row["eligible_segment_count"]) for row in item["rows"]
        ] for item in eligible
    }


def _source_train(
    pre: str, source_id: int, run: h5py.File,
    stores: Mapping[str, tuple[np.ndarray, np.ndarray]], aff_dt_ms: float,
    *, cck_scale: float | None = None, thinning_seed: int = 0,
) -> np.ndarray:
    train = EXACT._source_times_ms(pre, source_id, run, stores, aff_dt_ms)
    if pre == "CCK_Basket" and cck_scale is not None and cck_scale < 1.0 and train.size:
        keep = _rng(thinning_seed, "cck-thin", source_id).random(train.size) < cck_scale
        train = train[keep]
    return train


def _event_bins(train: np.ndarray, delay_ms: float, dt_ms: float, n_steps: int) -> np.ndarray:
    bins = np.ceil((train + delay_ms) / dt_ms - 1e-12).astype(np.int64)
    return bins[(bins >= 0) & (bins < n_steps)]


def replay_exact_reduced(
    edges: h5py.File, run: h5py.File, descriptors: Sequence[Any], spec: Any,
    stores: Mapping[str, tuple[np.ndarray, np.ndarray]], target: str, target_id: int,
    *, dt_ms: float, duration_ms: float, aff_dt_ms: float, contact_seed: int,
    eligible_counts: Mapping[tuple[str, str], Sequence[int]],
    cck_scale: float | None = None, thinning_seed: int = 0,
    m4_params: Sequence[float] | None = None,
    m5_params: Sequence[float] | None = None,
    m7_params: Sequence[float] | None = None,
) -> dict[str, Any]:
    rows = EXACT._target_rows(edges, descriptors, target, target_id, spec)
    n_steps = int(round(duration_ms / dt_ms))
    event_rows: list[np.ndarray] = []
    branch_event_rows: list[np.ndarray] = []
    amplitudes: list[float] = []
    tau_rise: list[float] = []
    tau_decay: list[float] = []
    e_rev: list[float] = []
    domains: list[int] = []
    allocation_audit = []
    cursor = 0
    for descriptor in descriptors:
        if descriptor.post != target:
            continue
        projection_rows = rows[cursor:cursor + len(descriptor.ports)]
        cursor += len(descriptor.ports)
        counts = list(eligible_counts[(descriptor.pre, target)])
        if len(counts) != len(projection_rows):
            raise ValueError(
                f"eligible row/port mismatch for {descriptor.pre}->{target}: "
                f"{counts} vs {len(projection_rows)} ports"
            )
        base_sources = EXACT._projection_sources(edges[descriptor.group_path], target_id)
        contacts = {int(row["contacts"]) for row in projection_rows}
        if len(contacts) != 1:
            raise ValueError(f"contact mismatch within {descriptor.name}")
        n_contacts = contacts.pop()
        rng = _rng(contact_seed, "contact", target, target_id, descriptor.name)
        draws = rng.integers(0, sum(counts), size=(len(base_sources), n_contacts))
        bounds = np.cumsum(counts)
        port_draws = np.searchsorted(bounds, draws, side="right")
        realized = [int(np.sum(port_draws == port)) for port in range(len(projection_rows))]
        allocation_audit.append({
            "projection": descriptor.name, "pre": descriptor.pre,
            "biological_sources": len(base_sources), "contacts_per_source": n_contacts,
            "eligible_counts_by_port": counts, "realized_contacts_by_port": realized,
        })
        for port_index, row in enumerate(projection_rows):
            events = np.zeros(n_steps, dtype=np.uint32)
            branch_events = np.zeros((4, n_steps), dtype=np.uint32)
            for source_index, source_id in enumerate(base_sources):
                multiplicity = int(np.sum(port_draws[source_index] == port_index))
                if not multiplicity:
                    continue
                train = _source_train(
                    str(row["pre"]), int(source_id), run, stores, aff_dt_ms,
                    cck_scale=cck_scale, thinning_seed=thinning_seed,
                )
                bins = _event_bins(train, float(row["delay_ms"]), dt_ms, n_steps)
                if bins.size:
                    events += np.bincount(bins, minlength=n_steps).astype(np.uint32) * multiplicity
                    if m7_params is not None and int(row["domain"]) > 0:
                        from ca1.sim.user_m7 import PV_LANE_SITES, route_contacts
                        if str(row["pre"]) == "Pyramidal": site_key = "apical_gt_100"
                        elif str(row["pre"]) == "O_LM": site_key = "apical_gt_200"
                        elif str(row["pre"]) == "CCK_Basket": site_key = "dend_lt_50"
                        else: site_key = "dend_50_200"
                        lane_counts = route_contacts(
                            int(source_id), int(target_id),
                            f"{row['pre']}|{row.get('receptor', port_index)}|{site_key}",
                            multiplicity, PV_LANE_SITES[site_key])
                        base_bins = np.bincount(bins, minlength=n_steps).astype(np.uint32)
                        for lane, count in enumerate(lane_counts):
                            if count: branch_events[lane] += base_bins * count
            if events.max(initial=0) > np.iinfo(np.uint16).max:
                raise OverflowError("exact-contact event bin overflow")
            event_rows.append(events.astype(np.uint16))
            branch_event_rows.append(branch_events.astype(np.uint16))
            tr = float(row["tau_rise"]); td = float(row["tau_decay"])
            amplitudes.append(float(row["weight_nS"]) * EXACT.BARRAGE.PAIRED._beta_g0(tr, td))
            tau_rise.append(tr); tau_decay.append(td); e_rev.append(float(row["e_rev"]))
            domains.append(int(row["domain"]))
    event_matrix = np.asarray(event_rows, dtype=np.uint16)
    enabled = np.ones((1, len(event_rows)), dtype=np.uint8)
    if m7_params is not None:
        measured = EXACT.CLAMP_KERNEL.simulate_user_m7(
            np.asarray(branch_event_rows, dtype=np.uint16), event_matrix,
            np.asarray(amplitudes), np.asarray(tau_rise), np.asarray(tau_decay),
            np.asarray(e_rev), np.asarray(domains, dtype=np.int64), dt_ms,
            duration_ms, EXACT._status_vector(target), np.asarray(m7_params, dtype=float))[0]
    else:
        measured = EXACT.CLAMP_KERNEL.simulate_user_m2(
            event_matrix, np.asarray(amplitudes), np.asarray(tau_rise), np.asarray(tau_decay),
            np.asarray(e_rev), np.asarray(domains, dtype=np.int64), enabled,
            dt_ms, duration_ms, EXACT._status_vector(target),
            m4_params=None if m4_params is None else np.asarray(m4_params, dtype=float),
            m5_params=None if m5_params is None else np.asarray(m5_params, dtype=float),
        )[0]
    return {
        "target_type": target, "target_id": target_id,
        "arm": ("M7_exact_contact" if m7_params is not None else
                "M5_exact_contact" if m5_params is not None else
                "M4_exact_contact" if m4_params is not None else "B_exact_contact"),
        "dt_ms": dt_ms, "contact_seed": contact_seed,
        "afferent_seed": CONTROL_AFFERENT_SEED if stores.get("__control__") else SAVED_AFFERENT_SEED,
        "n_spikes": int(measured[0]), "rate_hz": float(measured[1]),
        "mean_v_soma_mV": float(measured[2]), "max_v_soma_mV": float(measured[3]),
        "threshold_margin_mV": float(measured[4]),
        "mean_v_prox_mV": float(measured[5]), "mean_v_dist_mV": float(measured[6]),
        "peak_g_soma_nS": float(measured[7]), "peak_g_prox_nS": float(measured[8]),
        "peak_g_dist_nS": float(measured[9]), "allocation_audit": allocation_audit,
        "cck_rate_scale": cck_scale,
    }


def replay_deployed(
    edges: h5py.File, run: h5py.File, descriptors: Sequence[Any], spec: Any,
    stores: Mapping[str, tuple[np.ndarray, np.ndarray]], target: str, target_id: int,
    *, dt_ms: float, duration_ms: float, aff_dt_ms: float, afferent_seed: int,
) -> dict[str, Any]:
    record = EXACT.replay_target(
        edges, run, descriptors, spec, stores, target, target_id,
        dt_ms=dt_ms, duration_ms=duration_ms, afferent_dt_ms=aff_dt_ms, arms=("all",),
    )[0]
    return {
        **record, "arm": "A_deployed", "contact_seed": None,
        "afferent_seed": afferent_seed,
        "mean_v_soma_mV": record["mean_v_mV"],
    }


def _native_params(row: Any) -> tuple[float, float, float, float]:
    if row.pre in EXCITATORY:
        return (
            float(row.tau_rise_ms), float(row.tau_decay_ms),
            float(row.e_rev_mV), float(row.source_gmax_nS),
        )
    if not str(row.receptor_class).startswith("GABA_A"):
        raise ValueError(f"native replay supports active GABA_A row only: {row.row_key}")
    return (
        float(row.source_tau_rise_ms), float(row.source_tau_decay_ms),
        float(row.source_e_rev_mV), float(row.source_gmax_nS),
    )


def _native_task(task: Mapping[str, Any], queue: Any) -> None:
    try:
        config_path = Path(str(task["config"]))
        spec = build_network_spec(config_path, scale=1.0, seed=SAVED_AFFERENT_SEED)
        contracts = _contracts(config_path)
        target = str(task["target"]); target_id = int(task["target_id"])
        dt_ms = float(task["dt_ms"]); duration_ms = float(task["duration_ms"])
        aff_dt_ms = float(task["afferent_dt_ms"]); contact_seed = int(task["contact_seed"])
        h = neuron_session()
        template = contracts[next(key for key in contracts if key[1] == target)][0].template
        h.load_file(str(_MODELDB / "cells" / f"class_{template}.hoc"))
        h.dt = dt_ms; h.steps_per_ms = 1.0 / dt_ms; h.cvode.active(0)
        cells = {arm: getattr(h, template)(index, index, 0) for index, arm in enumerate(("C_native_all", "D_native_no_inhibition"))}
        keepalive: list[Any] = []
        scheduled: list[tuple[Any, np.ndarray]] = []
        stores = {
            source: (np.load(paths[0], mmap_mode="r"), np.load(paths[1], mmap_mode="r"))
            for source, paths in task["stores"].items()
        }
        with h5py.File(str(task["edges"]), "r") as edges, h5py.File(str(task["run"]), "r") as run:
            descriptors = EXACT.projections(edges)
            for descriptor in descriptors:
                if descriptor.post != target:
                    continue
                rows = contracts[(descriptor.pre, target)]
                sources = EXACT._projection_sources(edges[descriptor.group_path], target_id)
                n_contacts = {int(port["synapses_per_connection"]) for port in descriptor.ports}
                if len(n_contacts) != 1:
                    raise ValueError(f"contact mismatch for {descriptor.name}")
                contacts = n_contacts.pop()
                for arm, cell in cells.items():
                    if arm == "D_native_no_inhibition" and descriptor.pre not in EXCITATORY:
                        continue
                    sites = _site_records(h, cell, rows)
                    # Reinitialize from the identical key so C and D receive
                    # bit-identical excitation contact draws.
                    draws = _rng(
                        contact_seed, "contact", target, target_id, descriptor.name
                    ).integers(0, len(sites), size=(len(sources), contacts))
                    per_site: list[list[np.ndarray]] = [[] for _ in sites]
                    for source_index, source_id in enumerate(sources):
                        train = _source_train(
                            descriptor.pre, int(source_id), run, stores, aff_dt_ms,
                        ) + float(descriptor.delay_ms)
                        train = train[(train >= 0.0) & (train < duration_ms)]
                        if not train.size:
                            continue
                        for site_index in draws[source_index]:
                            per_site[int(site_index)].append(train)
                    for site_index, arrivals in enumerate(per_site):
                        if not arrivals:
                            continue
                        row, segment = sites[site_index]
                        tr, td, er, weight = _native_params(row)
                        synapse = h.MyExp2Sid(segment.x, sec=segment.sec)
                        synapse.tau1 = tr; synapse.tau2 = td; synapse.e = er
                        connection = h.NetCon(None, synapse)
                        connection.weight[0] = weight / 1000.0
                        ordered = np.sort(np.concatenate(arrivals))
                        keepalive.extend((synapse, connection, ordered))
                        scheduled.append((connection, ordered))
        records = []
        detectors = []
        for arm, cell in cells.items():
            soma = _soma(cell)
            voltage = h.Vector(); voltage.record(soma(0.5)._ref_v, dt_ms)
            spikes = h.Vector()
            detector = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
            detector.threshold = -20.0 if target == "O_LM" else -10.0
            detector.record(spikes)
            keepalive.extend((voltage, spikes, detector)); detectors.append((arm, voltage, spikes, detector.threshold))
        h.finitialize(float(cells["C_native_all"].Vrest))
        for connection, arrivals in scheduled:
            for arrival in arrivals:
                connection.event(float(arrival))
        h.continuerun(duration_ms)
        for arm, voltage, spikes, threshold in detectors:
            values = np.asarray(voltage, dtype=float)
            spike_times = np.asarray(spikes, dtype=float)
            measured = spike_times[(spike_times >= 0.0) & (spike_times < duration_ms)]
            records.append({
                "target_type": target, "target_id": target_id, "arm": arm,
                "dt_ms": dt_ms, "contact_seed": contact_seed,
                "afferent_seed": int(task["afferent_seed"]),
                "n_spikes": int(measured.size),
                "rate_hz": float(measured.size / (duration_ms / 1000.0)),
                "mean_v_soma_mV": float(values.mean()), "max_v_soma_mV": float(values.max()),
                "threshold_mV": float(threshold),
            })
        queue.put((True, records))
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        queue.put((False, repr(exc)))


def run_native(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_native_task, args=(task, queue))
    process.start()
    ok, value = queue.get()
    process.join()
    if process.exitcode != 0 or not ok:
        raise RuntimeError(f"native replay failed for {task}: {value}")
    return value


def _summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        stability = str(row.get("stability_condition", "primary"))
        seed = int(row["contact_seed"] or 0)
        groups[(str(row["target_type"]), str(row["arm"]), float(row["dt_ms"]), seed, stability)].append(row)
    output = []
    metrics = (
        "rate_hz", "mean_v_soma_mV", "mean_v_prox_mV", "mean_v_dist_mV",
        "peak_g_soma_nS", "peak_g_prox_nS", "peak_g_dist_nS",
    )
    for (target, arm, dt_ms, contact_seed, stability), rows in sorted(groups.items()):
        item: dict[str, Any] = {
            "target_type": target, "arm": arm, "dt_ms": dt_ms,
            "contact_seed": None if contact_seed == 0 else contact_seed,
            "stability_condition": stability, "n_cells": len(rows),
            "firing_cells": sum(float(row["rate_hz"]) > 0.0 for row in rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if metric in row]
            if values:
                item[metric] = EXACT._summarize(values)
        output.append(item)
    return output


def _primary_type_stats(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, dict[str, float]]:
    output = {}
    for target in TARGETS:
        rows = [
            row for row in records if row["target_type"] == target and row["arm"] == arm
            and row.get("stability_condition", "primary") == "primary"
            and float(row["dt_ms"]) == PRIMARY_DT_MS
        ]
        # Pool contact seeds for stochastic arms but keep one copy of deterministic A.
        if arm == "A_deployed":
            unique = {(int(row["target_id"]), int(row["afferent_seed"])): row for row in rows}
            rows = list(unique.values())
        rates = np.asarray([float(row["rate_hz"]) for row in rows])
        output[target] = {
            "mean_rate_hz": float(rates.mean()),
            "firing_fraction": float(np.mean(rates > 0.0)),
            "recruited_fraction": float(np.mean(rates >= RECRUIT_RATE_HZ)),
            "mean_v_soma_mV": float(np.mean([float(row["mean_v_soma_mV"]) for row in rows])),
        }
    return output


def _is_recruited(stats: Mapping[str, Mapping[str, float]]) -> bool:
    return all(
        values["mean_rate_hz"] >= RECRUIT_RATE_HZ
        and values["recruited_fraction"] >= RECRUIT_FRACTION
        for values in stats.values()
    )


def _stability_detail(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = ("A_deployed", "B_exact_contact", "C_native_all", "D_native_no_inhibition")

    def rate(target: str, arm: str, condition: str, dt_ms: float, seed: int | None, ids: set[int] | None = None) -> float | None:
        rows = [
            row for row in records if row["target_type"] == target and row["arm"] == arm
            and row.get("stability_condition", "primary") == condition
            and float(row["dt_ms"]) == dt_ms
            and (seed is None or row.get("contact_seed") == seed)
            and (ids is None or int(row["target_id"]) in ids)
        ]
        return None if not rows else float(np.mean([float(row["rate_hz"]) for row in rows]))

    rows = []
    for target in TARGETS:
        for arm in arms:
            seed = None if arm == "A_deployed" else CONTACT_SEEDS[0]
            dt_ids = {int(row["target_id"]) for row in records if row["target_type"] == target and row["arm"] == arm and row.get("stability_condition") == "dt_check"}
            aff_ids = {int(row["target_id"]) for row in records if row["target_type"] == target and row["arm"] == arm and row.get("stability_condition") == "afferent_seed_control"}
            primary = rate(target, arm, "primary", PRIMARY_DT_MS, seed)
            primary_dt = rate(target, arm, "primary", PRIMARY_DT_MS, seed, dt_ids or None)
            primary_aff = rate(target, arm, "primary", PRIMARY_DT_MS, seed, aff_ids or None)
            checked = rate(target, arm, "dt_check", CHECK_DT_MS, seed)
            afferent = rate(target, arm, "afferent_seed_control", PRIMARY_DT_MS, seed)
            contact_values = [
                value for value in (
                    rate(target, arm, "primary", PRIMARY_DT_MS, contact_seed)
                    for contact_seed in CONTACT_SEEDS
                ) if value is not None
            ]
            rows.append({
                "target_type": target, "arm": arm,
                "primary_rate_hz": primary, "dt0p05_rate_hz": checked,
                "dt_abs_difference_hz": None if primary_dt is None or checked is None else abs(checked - primary_dt),
                "afferent_control_rate_hz": afferent,
                "afferent_abs_difference_hz": None if primary_aff is None or afferent is None else abs(afferent - primary_aff),
                "contact_seed_rate_range_hz": None if not contact_values else [min(contact_values), max(contact_values)],
            })
    dt_differences = [row["dt_abs_difference_hz"] for row in rows if row["dt_abs_difference_hz"] is not None]
    aff_differences = [row["afferent_abs_difference_hz"] for row in rows if row["afferent_abs_difference_hz"] is not None]
    return {
        "rows": rows,
        "max_dt_abs_difference_hz": max(dt_differences, default=None),
        "max_afferent_abs_difference_hz": max(aff_differences, default=None),
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    primary = report["decision"]["primary_rates_hz"]
    lines = [
        "# Exact-contact four-arm CPU replay", "",
        f"Decision: **{report['decision']['branch']}** — {report['decision']['interpretation']}", "",
        "No deployed parameter, weight, in-degree, contact count, reversal, delay, source rule, or artifact was changed. No Table-5 rate was used.", "",
        "## Verified contact semantics", "",
        "| target | eligible dendrite | eligible soma | source contact probability | deployed K allocation | contacts/source |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["eligible_segments"]:
        if row["pre"] != "CCK_Basket":
            continue
        dend = sum(x["eligible_segment_count"] for x in row["rows"] if x["reduced_domain"] != "soma")
        soma = sum(x["eligible_segment_count"] for x in row["rows"] if x["reduced_domain"] == "soma")
        k = [x["deployed_port_indegree"] for x in row["rows"]]
        lines.append(f"| {row['post']} | {dend} | {soma} | {dend}/{dend+soma} dend, {soma}/{dend+soma} soma | {k} | 8 |")
    lines += ["", "Source ModelDB draws each contact independently/uniformly from the union of eligible synapse objects. Deployment partitions the K biological sources across ports and puts all contacts for a selected edge on that port domain.", "", "Thus CCK→PV/Bistratified changes from a source expectation of 64 proximal + 32 somatic contacts to deployed 48 + 48; CCK→O-LM changes from 106.67 + 53.33 to 80 + 80. Deployment over-allocates the somatic contact expectation by 50% and replaces mixed-domain source events with eight-contact single-domain events.", "", "## Per-arm firing rates", "", "| target | A deployed | B exact contacts | C native all | D native no inhibition | B' combined (if run) |", "|---|---:|---:|---:|---:|---:|"]
    for target in TARGETS:
        def rate(arm: str) -> str:
            value = primary.get(arm, {}).get(target)
            return "—" if value is None else f"{value:.3f}"
        lines.append(f"| {target} | {rate('A_deployed')} | {rate('B_exact_contact')} | {rate('C_native_all')} | {rate('D_native_no_inhibition')} | {rate('Bprime_exact_contact_cck24')} |")
    lines += ["", "## Reduced-arm voltage and conductance diagnostics", "", "| target | arm | mean Vm soma/prox/dist (mV) | peak g soma/prox/dist (nS) |", "|---|---|---:|---:|"]
    records = report["per_cell"]
    for target in TARGETS:
        for arm in ("A_deployed", "B_exact_contact"):
            selected = [row for row in records if row["target_type"] == target and row["arm"] == arm and row.get("stability_condition") == "primary" and float(row["dt_ms"]) == PRIMARY_DT_MS]
            if arm == "A_deployed":
                selected = list({int(row["target_id"]): row for row in selected}.values())
            means = [np.mean([float(row[key]) for row in selected]) for key in ("mean_v_soma_mV", "mean_v_prox_mV", "mean_v_dist_mV")]
            peaks = [np.mean([float(row[key]) for row in selected]) for key in ("peak_g_soma_nS", "peak_g_prox_nS", "peak_g_dist_nS")]
            lines.append(f"| {target} | {arm} | {means[0]:.3f} / {means[1]:.3f} / {means[2]:.3f} | {peaks[0]:.3f} / {peaks[1]:.3f} / {peaks[2]:.3f} |")
    lines += ["", "## Stability", "", report["stability"]["interpretation"], "", f"Maximum firing-rate change at dt 0.05 ms: {report['stability']['max_dt_abs_difference_hz']:.3f} Hz. Maximum alternate-afferent-seed change: {report['stability']['max_afferent_abs_difference_hz']:.3f} Hz.", "", "Per-type contact-seed ranges and dt/afferent comparisons are retained under `stability.rows` in the JSON.", "", "## Decision and next lever", "", report["decision"]["interpretation"], "", report["decision"]["next_step"], "", "## Verification", "", f"Pytest: **{report['verification']['pytest']}**", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXACT.DEFAULT_CONFIG)
    parser.add_argument("--edges", type=Path, default=EXACT.DEFAULT_EDGES)
    parser.add_argument("--run", type=Path, default=EXACT.DEFAULT_RUN)
    parser.add_argument("--candidate", type=Path, default=ROOT / "results" / "gaba_into_cck_candidate.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eligible-output", type=Path, default=DEFAULT_ELIGIBLE_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--reduced-cells", type=int, default=10)
    parser.add_argument("--native-cells", type=int, default=3)
    parser.add_argument("--skip-native-stability", action="store_true", help="development only")
    args = parser.parse_args()

    eligible = eligible_report(args.config.resolve())
    args.eligible_output.parent.mkdir(parents=True, exist_ok=True)
    args.eligible_output.write_text(json.dumps({"schema": "contact-eligible-segments/v1", "records": eligible}, indent=2) + "\n", encoding="utf-8")
    counts = _port_counts(eligible)
    spec = build_network_spec(args.config, scale=1.0, seed=SAVED_AFFERENT_SEED)
    all_records: list[dict[str, Any]] = []
    native_records: list[dict[str, Any]] = []
    with h5py.File(args.edges, "r") as edges, h5py.File(args.run, "r") as run:
        descriptors = EXACT.projections(edges)
        duration_s = float(run["meta"].attrs["duration_s"])
        if duration_s < 10.0:
            raise ValueError("the decisive replay requires at least 10 s")
        duration_ms = duration_s * 1000.0
        aff_dt_ms = float(run["meta"].attrs["dt_s"]) * 1000.0
        reduced_ids = {target: _spatial_panel(run, target, args.reduced_cells) for target in TARGETS}
        native_ids = {target: _spatial_panel(run, target, args.native_cells) for target in TARGETS}
        _, aff_counts, _, projection_summaries = EXACT.reconstruct_step2(
            edges, run, descriptors, reduced_ids, duration_s=duration_s,
            seed=SAVED_AFFERENT_SEED, afferent_rate_hz=0.65, network_spec=spec,
        )
        with tempfile.TemporaryDirectory(prefix="contact-alloc-afferents-") as tmp:
            store_paths: dict[int, dict[str, tuple[str, str]]] = {}
            stores_by_seed: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
            for aff_seed in (SAVED_AFFERENT_SEED, CONTROL_AFFERENT_SEED):
                local_counts = aff_counts if aff_seed == SAVED_AFFERENT_SEED else {
                    source: EXACT._afferent_counts(source, len(values), 0.65, duration_s, aff_seed)
                    for source, values in aff_counts.items()
                }
                seed_dir = Path(tmp) / str(aff_seed); seed_dir.mkdir()
                stores = {
                    source: EXACT.build_afferent_slot_store(
                        seed_dir, source, values, duration_ms=duration_ms,
                        dt_ms=aff_dt_ms, seed=aff_seed, rate_hz=0.65,
                    ) for source, values in local_counts.items()
                }
                stores_by_seed[aff_seed] = stores
                store_paths[aff_seed] = {
                    source: (str(offsets.filename), str(slots.filename))
                    for source, (offsets, slots) in stores.items()
                }

            # Primary A/B panel: A is deterministic; B has three contact draws.
            primary_stores = stores_by_seed[SAVED_AFFERENT_SEED]
            for target in TARGETS:
                for target_id in reduced_ids[target]:
                    a = replay_deployed(
                        edges, run, descriptors, spec, primary_stores, target, target_id,
                        dt_ms=PRIMARY_DT_MS, duration_ms=duration_ms,
                        aff_dt_ms=aff_dt_ms, afferent_seed=SAVED_AFFERENT_SEED,
                    ); a["stability_condition"] = "primary"; all_records.append(a)
                    for contact_seed in CONTACT_SEEDS:
                        b = replay_exact_reduced(
                            edges, run, descriptors, spec, primary_stores, target, target_id,
                            dt_ms=PRIMARY_DT_MS, duration_ms=duration_ms, aff_dt_ms=aff_dt_ms,
                            contact_seed=contact_seed, eligible_counts=counts,
                        ); b["stability_condition"] = "primary"; all_records.append(b)

            # Same 10-cell panel, primary contact draw at dt=0.05.
            for target in TARGETS:
                for target_id in reduced_ids[target]:
                    a = replay_deployed(
                        edges, run, descriptors, spec, primary_stores, target, target_id,
                        dt_ms=CHECK_DT_MS, duration_ms=duration_ms,
                        aff_dt_ms=aff_dt_ms, afferent_seed=SAVED_AFFERENT_SEED,
                    ); a["stability_condition"] = "dt_check"; all_records.append(a)
                    b = replay_exact_reduced(
                        edges, run, descriptors, spec, primary_stores, target, target_id,
                        dt_ms=CHECK_DT_MS, duration_ms=duration_ms, aff_dt_ms=aff_dt_ms,
                        contact_seed=CONTACT_SEEDS[0], eligible_counts=counts,
                    ); b["stability_condition"] = "dt_check"; all_records.append(b)

            # Same 10-cell panel, alternate afferent source realization.
            control_stores = stores_by_seed[CONTROL_AFFERENT_SEED]
            for target in TARGETS:
                for target_id in reduced_ids[target]:
                    a = replay_deployed(
                        edges, run, descriptors, spec, control_stores, target, target_id,
                        dt_ms=PRIMARY_DT_MS, duration_ms=duration_ms,
                        aff_dt_ms=aff_dt_ms, afferent_seed=CONTROL_AFFERENT_SEED,
                    ); a["stability_condition"] = "afferent_seed_control"; all_records.append(a)
                    b = replay_exact_reduced(
                        edges, run, descriptors, spec, control_stores, target, target_id,
                        dt_ms=PRIMARY_DT_MS, duration_ms=duration_ms, aff_dt_ms=aff_dt_ms,
                        contact_seed=CONTACT_SEEDS[0], eligible_counts=counts,
                    ); b["afferent_seed"] = CONTROL_AFFERENT_SEED
                    b["stability_condition"] = "afferent_seed_control"; all_records.append(b)

            a_stats = _primary_type_stats(all_records, "A_deployed")
            b_stats = _primary_type_stats(all_records, "B_exact_contact")
            partially_depolarized = all(
                b_stats[target]["mean_v_soma_mV"] - a_stats[target]["mean_v_soma_mV"]
                >= PARTIAL_DEPOLARIZATION_MV for target in TARGETS
            )
            b_recruited = _is_recruited(b_stats)

            # Conditional B': thin the realized CCK trains to the independently
            # source-gated arm-iii rate.  The candidate supplies that provenance;
            # no PING weight or cell parameter changes here.
            bprime_run = (not b_recruited) and partially_depolarized
            cck_target_hz = None
            if bprime_run:
                candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
                if candidate.get("schema") != "gaba-into-cck-sca-transfer-candidate/v1":
                    raise ValueError("unexpected GABA-into-CCK candidate schema")
                combined = json.loads((ROOT / "results" / "gaba_into_cck_combined_replay.json").read_text(encoding="utf-8"))
                arm_rows = [x for x in combined["summary"] if x["target"] == "CCK_Basket" and x["arm"] == "iii_corrected_inhibition_plus_cck_user_m3" and float(x["dt_ms"]) == PRIMARY_DT_MS]
                cck_target_hz = float(arm_rows[0]["rate_hz"]["mean"])
                recorded_cck_hz = next(x["population_rate_hz"] for x in projection_summaries if x["pre"] == "CCK_Basket")
                scale = cck_target_hz / float(recorded_cck_hz)
                for target in TARGETS:
                    for target_id in reduced_ids[target]:
                        row = replay_exact_reduced(
                            edges, run, descriptors, spec, primary_stores, target, target_id,
                            dt_ms=PRIMARY_DT_MS, duration_ms=duration_ms, aff_dt_ms=aff_dt_ms,
                            contact_seed=CONTACT_SEEDS[0], eligible_counts=counts,
                            cck_scale=scale, thinning_seed=20260712,
                        )
                        row["arm"] = "Bprime_exact_contact_cck24"
                        row["stability_condition"] = "conditional_combined"
                        all_records.append(row)

            # Native panel: 3 cells/type, three contact draws.  Stability checks
            # use one spatial representative/type to bound the CPU cost.
            native_tasks = []
            for target in TARGETS:
                for target_id in native_ids[target]:
                    for contact_seed in CONTACT_SEEDS:
                        native_tasks.append((target, target_id, PRIMARY_DT_MS, contact_seed, SAVED_AFFERENT_SEED, "primary"))
                if not args.skip_native_stability:
                    representative = native_ids[target][len(native_ids[target]) // 2]
                    native_tasks.extend((
                        (target, representative, CHECK_DT_MS, CONTACT_SEEDS[0], SAVED_AFFERENT_SEED, "dt_check"),
                        (target, representative, PRIMARY_DT_MS, CONTACT_SEEDS[0], CONTROL_AFFERENT_SEED, "afferent_seed_control"),
                    ))
            for index, (target, target_id, dt_ms, contact_seed, aff_seed, condition) in enumerate(native_tasks, 1):
                print(f"native {index}/{len(native_tasks)} {target}[{target_id}] dt={dt_ms} contact={contact_seed} aff={aff_seed}", flush=True)
                task = {
                    "config": str(args.config.resolve()), "edges": str(args.edges.resolve()),
                    "run": str(args.run.resolve()), "stores": store_paths[aff_seed],
                    "target": target, "target_id": target_id, "dt_ms": dt_ms,
                    "duration_ms": duration_ms, "afferent_dt_ms": aff_dt_ms,
                    "contact_seed": contact_seed, "afferent_seed": aff_seed,
                }
                rows = run_native(task)
                for row in rows:
                    row["stability_condition"] = condition
                native_records.extend(rows); all_records.extend(rows)

        c_stats = _primary_type_stats(all_records, "C_native_all")
        d_stats = _primary_type_stats(all_records, "D_native_no_inhibition")
        c_recruited = _is_recruited(c_stats)
        d_recruited = _is_recruited(d_stats)
        if not d_recruited:
            branch = "D_SILENT_INVALID_STREAM_MAPPING_STOP"
            interpretation = "Native excitation-only arm D did not recruit every target type; A-C are not interpreted."
            next_step = "Stop and repair the exact-stream/source-placement mapping before drawing a model conclusion."
        elif b_recruited:
            branch = "B_FIRES_CONTACT_ALLOCATION_DOMINANT"
            interpretation = "Exact ModelDB contact allocation recruits all three reduced PING types; the dominant wall is graph-reduction contact semantics."
            next_step = "Smallest faithful fix: preserve one K-sized biological source edge set and store per-source multinomial contact multiplicities to every eligible reduced port (or equivalent same-source multi-port edges), keeping K, per-contact gmax, kinetics, Erev, delay, contact count, and site probabilities unchanged."
        elif c_recruited:
            branch = "B_SILENT_C_FIRES_REDUCTION_LIMIT"
            interpretation = "Exact-contact user_m2 remains unrecruited while native templates recruit under the same streams; the three-domain fixed-threshold reduction misses the mixed E/I recruitment surface."
            next_step = "Next lever: a PV/Bistratified/O-LM user_mX with source-fitted dendritic Na regeneration, validated on held-out mixed E/I recruitment—not another CCK state or Table-5 tuning."
        else:
            branch = "B_AND_C_SILENT_UPSTREAM_WORKING_POINT"
            interpretation = "Both exact-contact user_m2 and native templates remain unrecruited at the recorded 36-45 Hz CCK state."
            next_step = "The mismatch is upstream: establish how the paper full network reaches a different CCK/PING working point or locate another network/data semantic difference."

        primary_rates: dict[str, dict[str, float]] = {}
        for arm in ("A_deployed", "B_exact_contact", "C_native_all", "D_native_no_inhibition", "Bprime_exact_contact_cck24"):
            rows = [x for x in all_records if x["arm"] == arm and x.get("stability_condition", "primary") in ("primary", "conditional_combined") and float(x["dt_ms"]) == PRIMARY_DT_MS]
            if rows:
                primary_rates[arm] = {
                    target: float(np.mean([float(x["rate_hz"]) for x in rows if x["target_type"] == target]))
                    for target in TARGETS
                }
        stability = _stability_detail(all_records)
        report = {
            "schema": "contact-allocation-four-arm/v1",
            "protocol": {
                "cpu_only": True, "gpu_used": False, "mpi_used": False,
                "duration_s": duration_s, "dt_ms": [PRIMARY_DT_MS, CHECK_DT_MS],
                "contact_seeds": list(CONTACT_SEEDS), "saved_afferent_seed": SAVED_AFFERENT_SEED,
                "control_afferent_seed": CONTROL_AFFERENT_SEED,
                "reduced_panel_ids": reduced_ids, "native_panel_ids": native_ids,
                "native_panel_declaration": "3 spatially stratified cells/type (allowed smaller native panel); primary uses all 3 cells/type x 3 contact seeds; dt and afferent checks use the middle spatial representative/type",
                "exact_saved_graph": True, "recorded_recurrent_and_pyramidal_spikes": True,
                "reconstructed_ca3_eciii_hz": 0.65, "table5_rate_tuning": False,
                "deployed_artifacts_unchanged": True,
                "immutable": ["biological K", "per-contact gmax", "kinetics", "Erev", "3 ms delay", "source contact count", "uniform-over-eligible-segments draw"],
                "recruitment_criterion": f"every type mean >= {RECRUIT_RATE_HZ} Hz and >= {RECRUIT_FRACTION:.0%} of replay records/type >= {RECRUIT_RATE_HZ} Hz",
                "partial_depolarization_criterion": f"B mean soma Vm at least {PARTIAL_DEPOLARIZATION_MV} mV above A for every type",
            },
            "provenance": {
                "config": str(args.config), "edges": str(args.edges), "run": str(args.run),
                "edge_sha256": str(edges.attrs["edge_sha256"]),
                "candidate_only": True,
            },
            "eligible_segments": eligible,
            "source_vs_deployed": {
                "source": "K biological sources; every source retains the exact source contact count; each contact independently samples uniformly from the union of eligible synapse-object segments",
                "deployed": "one K-sized base source list partitioned into disjoint equal per-port subsets; every selected port edge carries the full contact count on one reduced domain",
                "cck_to_ping_expected_and_deployed_contacts": [
                    {"target": "PV_Basket", "K": 12, "total_contacts": 96, "source_expected_proximal": 64.0, "source_expected_soma": 32.0, "deployed_proximal": 48, "deployed_soma": 48},
                    {"target": "Bistratified", "K": 12, "total_contacts": 96, "source_expected_proximal": 64.0, "source_expected_soma": 32.0, "deployed_proximal": 48, "deployed_soma": 48},
                    {"target": "O_LM", "K": 20, "total_contacts": 160, "source_expected_proximal": 320.0 / 3.0, "source_expected_soma": 160.0 / 3.0, "deployed_proximal": 80, "deployed_soma": 80},
                ],
            },
            "incoming_projection_summaries": projection_summaries,
            "per_cell": all_records, "summary": _summary(all_records),
            "conditional_Bprime": {
                "triggered": bprime_run, "cck_target_hz": cck_target_hz,
                "candidate": str(args.candidate),
            },
            "decision": {
                "branch": branch, "interpretation": interpretation, "next_step": next_step,
                "primary_rates_hz": primary_rates,
                "A_stats": a_stats, "B_stats": b_stats, "C_stats": c_stats, "D_stats": d_stats,
                "B_recruited": b_recruited, "C_recruited": c_recruited,
                "D_recruited": d_recruited, "B_partially_depolarized": partially_depolarized,
            },
            "stability": {**stability,
                "interpretation": "Primary reduced results use 10 cells/type and three exact-contact seeds; dt=0.05 and afferent-seed 12346 repeat the 10-cell panel at contact seed 12345. Native primary uses 3 cells/type and all three contact seeds; dt and afferent controls use one declared representative/type. See condition-labelled summaries for numerical ranges.",
            },
            "verification": {"pytest": "pending"},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    _write_markdown(report, args.markdown)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
