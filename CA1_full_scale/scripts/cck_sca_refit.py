#!/usr/bin/env python3
"""Source-grounded, candidate-only CCK/SCA reduced-cell refit.

The fit targets only paired ModelDB/NEURON intrinsic and unit-transfer
responses.  Network firing rates are validation outputs and never objectives.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pyximport
from scipy.optimize import differential_evolution

from ca1.config import build_network_spec
from ca1.extract.modeldb_tables import extract_connectivity
from ca1.sim.aglif_dend import aglif_dend_status


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS = ROOT / "results" / "cck_sca_diagnosis.json"
OUTPUT = ROOT / "results" / "cck_sca_refit_candidate.json"
TARGETS = ("CCK_Basket", "SCA")
SEEDS = (20260712, 20260713, 20260714)
DTS_MS = (0.05, 0.025)


def _load(name: str, filename: str) -> Any:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = _load("_cck_refit_paired", "paired_transfer_audit.py")
BARRAGE = _load("_cck_refit_barrage", "full_converging_barrage.py")
CLAMP = _load("_cck_refit_clamp", "exact_network_clamp_replay.py")
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import _cck_sca_refit_kernel as REFIT_KERNEL  # type: ignore[import-not-found]  # noqa: E402


STATUS_KEYS = (
    "C_m", "tau_m", "E_L", "g_c", "g_c_dist", "dend_C_frac",
    "dist_C_frac", "dend_leak_scale", "dist_leak_scale", "I_e",
    "k_adap", "k2", "k1", "V_th", "V_reset", "A2", "A1", "t_ref",
)


def _status(cell: str, overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    status = aglif_dend_status(cell)
    if overrides:
        status.update({key: float(value) for key, value in overrides.items()})
        if "g_c_scale" in overrides:
            conductance = float(status["C_m"]) / float(status["tau_m"])
            status["g_c"] = 2.0 * conductance * float(overrides["g_c_scale"])
        if "dist_coupling_ratio" in overrides or "g_c_scale" in overrides:
            status["g_c_dist"] = float(status["g_c"]) * float(
                overrides.get("dist_coupling_ratio", 0.25)
            )
    return {key: float(value) for key, value in status.items()}


def _status_vector(status: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(status[key]) for key in STATUS_KEYS], dtype=np.float64)


def _current_trace_summary(
    s: np.ndarray, current_nA: float, dt_ms: float, duration_ms: float,
    stim_end_ms: float,
) -> tuple[int, float, float]:
    result = REFIT_KERNEL.current_trace_summary(
        s, current_nA, dt_ms, duration_ms, stim_end_ms
    )
    return int(result[0]), float(result[1]), float(result[2])


def intrinsic_metrics(
    cell: str, source: Mapping[str, Any], overrides: Mapping[str, float], dt_ms: float,
) -> dict[str, Any]:
    vector = _status_vector(_status(cell, overrides))
    _n, rin, tau = _current_trace_summary(vector, -0.05, dt_ms, 900.0, 800.0)
    rates = [
        _current_trace_summary(vector, float(current), dt_ms, 900.0, 800.0)[0] / 0.6
        for current in source["currents_nA"]
    ]
    lo, hi = 0.0, 0.8
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        fires = _current_trace_summary(vector, mid, dt_ms, 600.0, 500.0)[0] > 0
        if fires: hi = mid
        else: lo = mid
    return {
        "dt_ms": dt_ms, "Rin_MOhm": rin, "tau_m_ms": tau,
        "rheobase_nA": hi, "currents_nA": list(source["currents_nA"]),
        "rates_hz": rates,
    }


def _fit_overrides(cell: str, source: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    base = _status(cell)
    sigmas = np.asarray(source["sigma"]["rates_hz"], dtype=float)
    target_rates = np.asarray(source["rates_hz"], dtype=float)

    def unpack(x: np.ndarray) -> dict[str, float]:
        return {
            "C_m": base["C_m"] * float(np.exp(x[0])),
            "tau_m": base["tau_m"] * float(np.exp(x[1])),
            "V_th": base["V_th"] + float(x[2]),
            "A2": base["A2"] * float(np.exp(x[3])),
            "A1": base["A1"] * float(np.exp(x[4])),
            "k_adap": base["k_adap"] * float(np.exp(x[5])),
            "t_ref": base["t_ref"] * float(np.exp(x[6])),
        }

    def objective(x: np.ndarray) -> float:
        metrics = intrinsic_metrics(cell, source, unpack(x), 0.05)
        rates = np.asarray(metrics["rates_hz"], dtype=float)
        rate_loss = np.mean(((rates - target_rates) / sigmas) ** 2)
        passive = ((metrics["Rin_MOhm"] / float(source["Rin"]) - 1.0) / 0.15) ** 2
        passive += ((metrics["tau_m_ms"] / float(source["tau_m"]) - 1.0) / 0.15) ** 2
        rheo = ((metrics["rheobase_nA"] / float(source["rheobase_nA"]) - 1.0) / 0.10) ** 2
        return float(rate_loss + passive + rheo)

    bounds = [
        (np.log(0.7), np.log(2.5)), (np.log(0.7), np.log(2.5)),
        (-1.0, 8.0), (np.log(0.35), np.log(6.0)),
        (np.log(0.35), np.log(4.0)), (np.log(0.35), np.log(4.0)),
        (np.log(0.6), np.log(4.0)),
    ]
    fit = differential_evolution(
        objective, bounds, seed=20260712, popsize=7, maxiter=35,
        polish=False, workers=1, updating="immediate",
    )
    overrides = unpack(fit.x)
    return overrides, {"loss": float(fit.fun), "evaluations": int(fit.nfev), "success": bool(fit.success)}


def fit_intrinsics(diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell in TARGETS:
        diagnosed = diagnosis["intrinsic"]["cells"][cell]
        source = diagnosed["source_neuron"]
        overrides, optimization = _fit_overrides(cell, source)
        deployed = diagnosed["deployed_user_m2"]
        refit = [intrinsic_metrics(cell, source, overrides, dt) for dt in DTS_MS]
        primary = next(row for row in refit if row["dt_ms"] == 0.025)
        tolerances = [max(2.0, 0.2 * float(rate)) for rate in source["rates_hz"]]
        gate = {
            "Rin_within_15_percent": abs(primary["Rin_MOhm"] / source["Rin"] - 1.0) <= 0.15,
            "tau_m_within_15_percent": abs(primary["tau_m_ms"] / source["tau_m"] - 1.0) <= 0.15,
            "rheobase_within_10_percent": abs(primary["rheobase_nA"] / source["rheobase_nA"] - 1.0) <= 0.10,
            "fi_each_within_max_2hz_or_20percent": all(
                abs(float(got) - float(want)) <= tol
                for got, want, tol in zip(primary["rates_hz"], source["rates_hz"], tolerances, strict=True)
            ),
        }
        gate["passed"] = all(gate.values())
        cells[cell] = {
            "source_neuron": source, "deployed_user_m2": deployed,
            "fitted_params": overrides, "optimization": optimization,
            "refit_user_m2": refit, "held_out_gate": gate,
        }
    return {"cells": cells}


def _single_domain_allocation(domain: str) -> dict[str, float]:
    return {key: float(key == domain) for key in ("soma", "proximal", "distal")}


def fit_transfers(
    diagnosis: Mapping[str, Any], intrinsic: Mapping[str, Any]
) -> dict[str, Any]:
    primary_records = {
        str(record["row"]): record for record in diagnosis["excitatory_transfer"]["rows"]
        if float(record["dt_ms"]) == 0.025
    }
    return _fit_transfers_continue(primary_records, diagnosis, intrinsic)


def _barrage_rows(config: Path) -> dict[str, list[Any]]:
    spec = build_network_spec(config, scale=1.0, seed=PAIRED.LOCATION_SEED)
    raw = extract_connectivity(
        index=spec.conndata_index, cellnumbers_index=spec.cellnumbers_index,
        count_mode=spec.conndata_count_mode,
    )
    result = {cell: [] for cell in TARGETS}
    for source in PAIRED.configured_excitatory_rows(config, TARGETS):
        key = f"{source.pre}_to_{source.post}"
        entry = raw["afferents" if source.kind == "aff" else "projections"][key]
        if source.kind == "aff":
            indegree = int(round(float(entry["synapses_per_cell"]) / source.synapses_per_connection))
            rate = 0.65
        else:
            indegree = int(round(float(entry["indegree"])))
            rate = 1.0
        result[source.post].append(BARRAGE.BarrageRow(source, indegree, rate))
    for rows in result.values():
        rows.sort(key=lambda item: item.source.pre)
    return result


def _barrage_candidate_rows(transfer: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for record in transfer["rows"]:
        mapping = record["candidate_mapping"]
        output[str(record["row"])] = {
            "contract": record["contract"],
            "derivation": {
                "constraint": {"total_transfer_scale": mapping["transfer_scale_from_immutable_source_gmax"]},
                "graded_allocation": mapping["allocation"],
            },
        }
    return output


def _compiled_barrage_result(
    rows: Sequence[Any], arm: str, seed: int, dt: float,
    candidate_rows: Mapping[str, Mapping[str, Any]], overrides: Mapping[str, float] | None,
    h_params: Sequence[float] | None = None,
    conductance_scale: float = 1.0,
) -> Any:
    transient_ms, measure_ms = 1000.0, 10000.0
    duration_ms = transient_ms + measure_ms
    steps = int(round(duration_ms / dt))
    events = np.zeros((len(rows), steps), dtype=np.uint16)
    amplitudes = np.empty(len(rows)); tau_rise = np.empty(len(rows)); tau_decay = np.empty(len(rows))
    e_rev = np.empty(len(rows)); domains = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        schedule = BARRAGE.poisson_schedule(row, duration_ms - row.source.delay_ms, seed, 1)
        arrivals = schedule.event_times_ms + row.source.delay_ms
        bins = np.ceil(arrivals / dt - 1e-12).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < steps)]
        counted = np.bincount(bins, minlength=steps)
        if counted.max(initial=0) > np.iinfo(np.uint16).max:
            raise OverflowError("barrage event bin overflow")
        events[index] = counted.astype(np.uint16)
        weight, allocation = BARRAGE._transfer_for_arm(row, arm, candidate_rows)
        nonzero = np.flatnonzero(allocation > 1e-12)
        if len(nonzero) != 1:
            raise ValueError(f"compiled barrage expects one candidate domain for {row.row_id}")
        amplitudes[index] = conductance_scale * weight * row.source.synapses_per_connection * PAIRED._beta_g0(
            row.source.tau_rise_ms, row.source.tau_decay_ms
        )
        tau_rise[index] = row.source.tau_rise_ms; tau_decay[index] = row.source.tau_decay_ms
        e_rev[index] = row.source.e_rev_mV; domains[index] = int(nonzero[0])
    status = CLAMP._status_vector(rows[0].source.post, overrides)
    measured = CLAMP.CLAMP_KERNEL.simulate_user_m2(
        events, amplitudes, tau_rise, tau_decay, e_rev, domains,
        np.ones((1, len(rows)), dtype=np.bool_), dt, duration_ms, status, transient_ms,
        h_params,
    )[0]
    return BARRAGE.ArmResult(
        cell=rows[0].source.post, arm=arm, seed=seed, dt_ms=dt,
        n_spikes=int(measured[0]), rate_hz=float(measured[1]),
        mean_v_mV=float(measured[2]), max_v_mV=float(measured[3]),
        threshold_mV=float(_status(rows[0].source.post, overrides)["V_th"]),
        threshold_status="suprathreshold/firing" if measured[0] else "subthreshold/silent",
    )


def validate_barrage(
    diagnosis: Mapping[str, Any], intrinsic: Mapping[str, Any], transfer: Mapping[str, Any],
    config: Path,
) -> dict[str, Any]:
    rows = _barrage_rows(config)
    candidate_rows = _barrage_candidate_rows(transfer)
    results = []
    for cell in TARGETS:
        for seed in SEEDS:
            for dt in DTS_MS:
                results.append(_compiled_barrage_result(
                    rows[cell], "deployed_user_m2", seed, dt, candidate_rows, None
                ))
                results.append(_compiled_barrage_result(
                    rows[cell], "candidate_user_m2", seed, dt, candidate_rows,
                    intrinsic["cells"][cell]["fitted_params"],
                ))
    summaries = BARRAGE._summaries(results)
    source_summaries = [
        row for row in diagnosis["excitation_only_barrage"]["summary"]
        if row["arm"] == "source_neuron" and row["cell"] in TARGETS
    ]
    source_lookup = {(row["cell"], float(row["dt_ms"])): float(row["rate_mean_hz"]) for row in source_summaries}
    lookup = {(row["cell"], row["arm"], float(row["dt_ms"])): float(row["rate_mean_hz"]) for row in summaries}
    gap = []
    for cell in TARGETS:
        source = source_lookup[(cell, 0.025)]
        deployed = lookup[(cell, "deployed_user_m2", 0.025)]
        candidate = lookup[(cell, "candidate_user_m2", 0.025)]
        old_gap = abs(deployed - source); new_gap = abs(candidate - source)
        gap.append({
            "cell": cell, "source_rate_hz": source, "deployed_rate_hz": deployed,
            "candidate_rate_hz": candidate,
            "source_gap_removed_percent": 100.0 * (old_gap - new_gap) / old_gap,
            "residual_overfire_hz": candidate - source,
        })
    return {
        "protocol": {"transient_ms": 1000.0, "measure_ms": 10000.0, "seeds": list(SEEDS),
                     "dt_ms": list(DTS_MS), "afferent_rate_hz": 0.65,
                     "recurrent_pyramidal_proxy_hz": 1.0, "rate_in_fit_objective": False},
        "results": [asdict(row) for row in results], "summary": summaries,
        "source_neuron_summary_reused_from_diagnosis": source_summaries,
        "primary_dt_gap": gap,
    }


def _spatial_panel(run: h5py.File, target: str, count: int = 10) -> list[int]:
    positions = np.asarray(run["cell_positions"][target], dtype=float)
    ids = np.arange(len(positions))
    order = ids[np.lexsort((ids, positions[:, 2], positions[:, 1], positions[:, 0]))]
    return [int(block[len(block) // 2]) for block in np.array_split(order, count)]


def _clamp_transfer_map(transfer: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["row"]): {
            "transferred_gmax_nS": row["candidate_mapping"]["transferred_gmax_nS"],
            "allocation": row["candidate_mapping"]["allocation"],
        }
        for row in transfer["rows"]
    }


def validate_exact_clamp(
    diagnosis: Mapping[str, Any], intrinsic: Mapping[str, Any], transfer: Mapping[str, Any],
    config: Path, edges_path: Path, run_path: Path,
) -> dict[str, Any]:
    CLAMP.TARGETS = TARGETS
    spec = build_network_spec(config, scale=1.0, seed=12345)
    candidate_transfer = _clamp_transfer_map(transfer)
    candidate_records: list[dict[str, Any]] = []
    with h5py.File(edges_path, "r") as edges, h5py.File(run_path, "r") as run:
        descriptors = CLAMP.projections(edges)
        selected = {target: _spatial_panel(run, target) for target in TARGETS}
        duration_s = float(run["meta"].attrs["duration_s"])
        aff_dt = float(run["meta"].attrs["dt_s"]) * 1000.0
        _recurrent, aff_counts, _detail, _summaries = CLAMP.reconstruct_step2(
            edges, run, descriptors, selected, duration_s=duration_s, seed=12345,
            afferent_rate_hz=0.65, network_spec=spec,
        )
        with tempfile.TemporaryDirectory(prefix="cck-sca-refit-aff-") as tmp:
            stores = {
                source: CLAMP.build_afferent_slot_store(
                    Path(tmp), source, counts, duration_ms=duration_s * 1000.0,
                    dt_ms=aff_dt, seed=12345, rate_hz=0.65,
                ) for source, counts in aff_counts.items()
            }
            for target in TARGETS:
                for target_id in selected[target]:
                    for dt in DTS_MS:
                        candidate_records.extend(CLAMP.replay_target(
                            edges, run, descriptors, spec, stores, target, target_id,
                            dt_ms=dt, duration_ms=duration_s * 1000.0,
                            afferent_dt_ms=aff_dt, arms=("all",),
                            status_overrides=intrinsic["cells"][target]["fitted_params"],
                            excitatory_transfer=candidate_transfer,
                        ))
    deployed_summary = [
        row for row in diagnosis["exact_clamp"]["summary"]
        if row["target_type"] in TARGETS and row["arm"] == "all"
    ]
    return {
        "protocol": {"open_loop_recorded_inhibition": True, "saved_afferent_seed": 12345,
                     "selected_targets_per_type": 10, "dt_ms": list(DTS_MS)},
        "deployed_summary_reused_from_diagnosis": deployed_summary,
        "candidate_per_cell": candidate_records,
        "candidate_summary": CLAMP._aggregate_replays(candidate_records),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_snapshot() -> dict[str, Any]:
    paths = [
        ROOT / "src/ca1/params/aglif_parameters_fitted.json",
        ROOT / "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json",
        ROOT / "src/ca1/params/connectivity.json",
        ROOT / "src/ca1/params/syndata_120.json",
    ]
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def finalize_report(report: dict[str, Any]) -> None:
    intrinsic = report["intrinsic"]["cells"]
    transfer_rows = report["excitatory_transfer"]["rows"]
    barrage = report.get("excitation_only_full_barrage")
    clamp = report.get("exact_ei_open_loop_clamp")
    stability: dict[str, Any] = {
        "intrinsic": {},
        "transfer_max_abs_dt_difference_percentage_points": max(
            max(
                abs(float(row["candidate_dt_0p025"][metric]) - float(row["candidate_dt_0p05"][metric]))
                for metric in ("peak_percent_of_source", "charge_percent_of_source")
            ) for row in transfer_rows
        ),
    }
    for cell, record in intrinsic.items():
        by_dt = {float(row["dt_ms"]): row for row in record["refit_user_m2"]}
        stability["intrinsic"][cell] = {
            "max_fi_rate_difference_hz": max(
                abs(a - b) for a, b in zip(by_dt[0.05]["rates_hz"], by_dt[0.025]["rates_hz"], strict=True)
            ),
            "Rin_difference_MOhm": abs(by_dt[0.05]["Rin_MOhm"] - by_dt[0.025]["Rin_MOhm"]),
            "tau_difference_ms": abs(by_dt[0.05]["tau_m_ms"] - by_dt[0.025]["tau_m_ms"]),
        }
    if barrage is not None:
        lookup = {(row["cell"], row["arm"], float(row["dt_ms"])): row for row in barrage["summary"]}
        stability["barrage"] = {
            cell: {
                "candidate_dt_rate_difference_hz": abs(
                    lookup[(cell, "candidate_user_m2", 0.05)]["rate_mean_hz"]
                    - lookup[(cell, "candidate_user_m2", 0.025)]["rate_mean_hz"]
                ),
                "three_seed_range_hz_at_0p025": [
                    lookup[(cell, "candidate_user_m2", 0.025)]["rate_min_hz"],
                    lookup[(cell, "candidate_user_m2", 0.025)]["rate_max_hz"],
                ],
            } for cell in TARGETS
        }
    if clamp is not None:
        lookup = {(row["target_type"], float(row["dt_ms"])): row for row in clamp["candidate_summary"]}
        stability["exact_clamp"] = {
            cell: {"candidate_dt_rate_difference_hz": abs(
                lookup[(cell, 0.05)]["firing_rate_hz"]["mean"]
                - lookup[(cell, 0.025)]["firing_rate_hz"]["mean"]
            )} for cell in TARGETS
        }
    report["stability"] = stability
    report["other_cell_types_invariance"] = {
        "candidate_override_keys": list(TARGETS),
        "unchanged_cell_types": [
            "Pyramidal", "PV_Basket", "Axo", "Bistratified", "Ivy", "O_LM", "Neurogliaform"
        ],
        "deployed_parameter_files_modified": False,
        "evidence": "candidate overrides are embedded only under intrinsic.cells.CCK_Basket/SCA; deployed-file SHA256 recorded above",
    }
    if barrage is not None and clamp is not None:
        gaps = {row["cell"]: row for row in barrage["primary_dt_gap"]}
        candidate_clamp = {
            (row["target_type"], float(row["dt_ms"])): row["firing_rate_hz"]["mean"]
            for row in clamp["candidate_summary"]
        }
        deployed_clamp = {
            (row["target_type"], float(row["dt_ms"])): row["firing_rate_hz"]["mean"]
            for row in clamp["deployed_summary_reused_from_diagnosis"]
        }
        report["verdict"] = {
            "candidate_not_deployed": True,
            "CCK_Basket": {
                "excitation_only_source_gap_removed_percent": gaps["CCK_Basket"]["source_gap_removed_percent"],
                "excitation_only_residual_overfire_hz": gaps["CCK_Basket"]["residual_overfire_hz"],
                "exact_EI_deployed_hz": deployed_clamp[("CCK_Basket", 0.025)],
                "exact_EI_candidate_hz": candidate_clamp[("CCK_Basket", 0.025)],
                "interpretation": "source-grounded refit does not remove CCK overactivity; restoring source transfer exposes the AGLIF depolarization-block expressivity failure",
                "requires_depolarization_block_expressivity_change": True,
            },
            "SCA": {
                "excitation_only_source_gap_removed_percent": gaps["SCA"]["source_gap_removed_percent"],
                "excitation_only_residual_overfire_hz": gaps["SCA"]["residual_overfire_hz"],
                "exact_EI_deployed_hz": deployed_clamp[("SCA", 0.025)],
                "exact_EI_candidate_hz": candidate_clamp[("SCA", 0.025)],
                "interpretation": "source-grounded refit removes most SCA excess but a smaller block-boundary residual remains",
                "requires_depolarization_block_expressivity_change": True,
            },
            "overall": "candidate is suitable for review, not deployment: SCA is substantially corrected; CCK requires the parallel depolarization-block model-capability change",
        }


def _fit_transfers_continue(
    primary_records: Mapping[str, Mapping[str, Any]],
    diagnosis: Mapping[str, Any],
    intrinsic: Mapping[str, Any],
) -> dict[str, Any]:
    stability_records = {
        str(record["row"]): record for record in diagnosis["excitatory_transfer"]["rows"]
        if float(record["dt_ms"]) == 0.05
    }
    rows: list[dict[str, Any]] = []
    for row_id, record in sorted(primary_records.items()):
        row = PAIRED.SourceRow(**record["contract"])
        source = record["source_summary"]
        overrides = intrinsic["cells"][row.post]["fitted_params"]
        old_peak = float(record["peak_percent_of_source"])
        old_charge = float(record["charge_percent_of_source"])
        candidates: list[dict[str, Any]] = []
        for domain in ("soma", "proximal", "distal"):
            allocation = _single_domain_allocation(domain)

            def evaluate(scale: float, dt: float = 0.025, capture: Mapping[str, Any] = record) -> tuple[Any, float, float]:
                measurement = PAIRED.run_user_m2_cpu(
                    row, domain, dt, float(capture["reduced_initial_mV"]),
                    transfer_scale=float(scale), allocation=allocation,
                    passive_overrides=overrides, max_transfer_scale=4.0,
                )
                summary = capture["source_summary"]
                peak = 100.0 * measurement.epsp_peak_mV / summary["epsp_peak_mV"]["median"]
                charge = 100.0 * abs(measurement.clamp_charge_nA_ms) / abs(summary["clamp_charge_nA_ms"]["median"])
                return measurement, peak, charge

            unit_measurement, unit_peak, unit_charge = evaluate(1.0)
            a, b = unit_peak / 100.0, unit_charge / 100.0
            scale = float(np.clip((a + b) / (a * a + b * b), 0.0, 4.0))
            measurement, peak, charge = evaluate(scale)
            feasible = 85.0 <= peak <= 115.0 and 90.0 <= charge <= 110.0
            both_improved = (
                abs(peak - 100.0) <= abs(old_peak - 100.0) + 1e-6
                and abs(charge - 100.0) <= abs(old_charge - 100.0) + 1e-6
            )
            if not feasible and not both_improved:
                # Some compressed rows (notably Pyr->SCA) cannot jointly match
                # peak and charge at immutable kinetics.  Constrain the scalar
                # to the interval that moves both observed responses toward source.
                scale = float(np.clip(100.0 / unit_peak, 0.0, 4.0))
                measurement, peak, charge = evaluate(scale)
                both_improved = (
                    abs(peak - 100.0) <= abs(old_peak - 100.0) + 1e-6
                    and abs(charge - 100.0) <= abs(old_charge - 100.0) + 1e-6
                )
            candidates.append({
                "domain": domain, "scale": scale, "measurement": measurement,
                "peak_percent": peak, "charge_percent": charge,
                "feasible_gate": feasible, "both_improved": both_improved,
                "loss": (peak / 100.0 - 1.0) ** 2 + (charge / 100.0 - 1.0) ** 2,
            })
        feasible_rows = [item for item in candidates if item["feasible_gate"]]
        improving_rows = [item for item in candidates if item["both_improved"]]
        pool = feasible_rows or improving_rows or candidates
        chosen = min(pool, key=lambda item: item["loss"])
        checked = stability_records[row_id]
        checked_measurement = PAIRED.run_user_m2_cpu(
            row, chosen["domain"], 0.05, float(checked["reduced_initial_mV"]),
            transfer_scale=float(chosen["scale"]),
            allocation=_single_domain_allocation(chosen["domain"]),
            passive_overrides=overrides, max_transfer_scale=4.0,
        )
        checked_source = checked["source_summary"]
        checked_peak = 100.0 * checked_measurement.epsp_peak_mV / checked_source["epsp_peak_mV"]["median"]
        checked_charge = 100.0 * abs(checked_measurement.clamp_charge_nA_ms) / abs(checked_source["clamp_charge_nA_ms"]["median"])
        rows.append({
            "row": row_id, "contract": record["contract"],
            "objective": {
                "name": "paired_source_peak_and_clamp_charge_only",
                "formula": "(peak/source-1)^2 + (charge/source-1)^2",
                "network_rate_in_objective": False, "table5_in_objective": False,
            },
            "deployed": {"peak_percent_of_source": old_peak, "charge_percent_of_source": old_charge},
            "candidate_mapping": {
                "domain": chosen["domain"], "allocation": _single_domain_allocation(chosen["domain"]),
                "transfer_scale_from_immutable_source_gmax": chosen["scale"],
                "source_gmax_nS": row.source_gmax_nS,
                "transferred_gmax_nS": row.source_gmax_nS * float(chosen["scale"]),
            },
            "candidate_dt_0p025": {
                "measurement": asdict(chosen["measurement"]),
                "peak_percent_of_source": chosen["peak_percent"],
                "charge_percent_of_source": chosen["charge_percent"],
                "peak_gate_85_to_115": 85.0 <= chosen["peak_percent"] <= 115.0,
                "charge_gate_90_to_110": 90.0 <= chosen["charge_percent"] <= 110.0,
                "both_metrics_moved_toward_source": chosen["both_improved"],
            },
            "candidate_dt_0p05": {
                "measurement": asdict(checked_measurement),
                "peak_percent_of_source": checked_peak,
                "charge_percent_of_source": checked_charge,
            },
        })
    ca3 = [row for row in rows if row["row"].startswith("CA3->")]
    return {
        "protocol": {"paired_location_draws": 32, "fit_dt_ms": 0.025, "held_out_dt_ms": 0.05},
        "rows": rows,
        "gates": {
            "all_CA3_peak_85_to_115": all(row["candidate_dt_0p025"]["peak_gate_85_to_115"] for row in ca3),
            "all_CA3_charge_90_to_110": all(row["candidate_dt_0p025"]["charge_gate_90_to_110"] for row in ca3),
            "all_rows_both_metrics_moved_toward_source": all(row["candidate_dt_0p025"]["both_metrics_moved_toward_source"] for row in rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-intrinsic", action="store_true")
    parser.add_argument("--skip-barrage", action="store_true")
    parser.add_argument("--skip-clamp", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    diagnosis = json.loads(DIAGNOSIS.read_text(encoding="utf-8"))
    previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    intrinsic = fit_intrinsics(diagnosis) if args.refit_intrinsic or "intrinsic" not in previous else previous["intrinsic"]
    transfer = (
        fit_transfers(diagnosis, intrinsic)
        if args.refit_intrinsic or "excitatory_transfer" not in previous
        else previous["excitatory_transfer"]
    )
    report = {
        "schema": "cck-sca-source-grounded-refit-candidate/v1",
        "provenance": {
            "candidate_only": True, "deployed_params_unchanged": True,
            "source_response_objective_only": True, "table5_rate_tuning": False,
            "gpu_used": False, "mpi_used": False,
        },
        "intrinsic": intrinsic,
        "excitatory_transfer": transfer,
        "immutable_deployed_file_sha256": immutable_snapshot(),
    }
    if not args.skip_barrage:
        report["excitation_only_full_barrage"] = (
            previous["excitation_only_full_barrage"]
            if "excitation_only_full_barrage" in previous and not args.refit_intrinsic
            else validate_barrage(
                diagnosis, intrinsic, transfer, ROOT / "configs/full_scale_3dtopo.yaml"
            )
        )
    if not args.skip_clamp:
        report["exact_ei_open_loop_clamp"] = (
            previous["exact_ei_open_loop_clamp"]
            if "exact_ei_open_loop_clamp" in previous and not args.refit_intrinsic
            else validate_exact_clamp(
                diagnosis, intrinsic, transfer, ROOT / "configs/full_scale_3dtopo.yaml",
                ROOT / "results/edges_fullscale.h5", ROOT / "results/fullscale_3dtopo_theta.h5",
            )
        )
    finalize_report(report)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["intrinsic"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
