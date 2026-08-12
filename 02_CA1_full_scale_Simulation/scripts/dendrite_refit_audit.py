#!/usr/bin/env python3
"""Candidate-only joint cell-level dendritic refit from paired source captures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from ca1.params.aglif import aglif_params_for_cell_type
from ca1.params.dendritic_transfer import dendritic_transfer_for_cell_type
from ca1.params.dendritic_transfer_fit import (
    CandidateResponse,
    CellDendriteParams,
    SourceResponseTarget,
    fit_joint_cell_transfer,
    response_constraints,
    response_ratios,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results" / "charge_matched_transfer_candidate.json"
DEFAULT_OUTPUT = ROOT / "results" / "dendrite_refit_candidate.json"
TARGET_CELLS = ("PV_Basket", "Bistratified", "O_LM")


def _load_paired_module() -> Any:
    path = ROOT / "scripts" / "paired_transfer_audit.py"
    spec = importlib.util.spec_from_file_location("_paired_transfer_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = _load_paired_module()


def _source_target(record: Mapping[str, Any]) -> SourceResponseTarget:
    summary = record["source_neuron"]["summary"]
    contract = record["contract"]
    return SourceResponseTarget(
        row_id=f"{contract['pre']}->{contract['post']}",
        peak_mV=float(summary["epsp_peak_mV"]["median"]),
        clamp_charge_nA_ms=float(summary["clamp_charge_nA_ms"]["median"]),
        voltage_area_mV_ms=float(summary["voltage_area_mV_ms"]["median"]),
        time_to_peak_ms=float(summary["time_to_peak_ms"]["median"]),
    )


def _old_params(cell: str) -> CellDendriteParams:
    old = dendritic_transfer_for_cell_type(cell)
    return CellDendriteParams(
        dend_C_frac=old.dend_C_frac,
        dend_leak_scale=old.dend_leak_scale,
        g_c_scale=old.g_c_scale,
    )


class PairedEvaluator:
    """Replay immutable row contracts with candidate-only passive overrides."""

    def __init__(self, records: list[Mapping[str, Any]], dt_ms: float) -> None:
        self.records = {f"{x['contract']['pre']}->{x['contract']['post']}": x for x in records}
        self.dt_ms = dt_ms
        self.cache: dict[tuple[str, CellDendriteParams], CandidateResponse] = {}

    def __call__(self, row_id: str, params: CellDendriteParams) -> CandidateResponse:
        key = (row_id, params)
        if key not in self.cache:
            record = self.records[row_id]
            row = PAIRED.SourceRow(**record["contract"])
            derived = record["derivation"]
            measurement = PAIRED.run_user_m2_cpu(
                row,
                "soma",
                self.dt_ms,
                float(record["source_rest_mV"]),
                transfer_scale=float(derived["constraint"]["total_transfer_scale"]),
                allocation=derived["graded_allocation"],
                passive_overrides=params.as_status_overrides(),
            )
            self.cache[key] = CandidateResponse(
                peak_mV=measurement.epsp_peak_mV,
                clamp_charge_nA_ms=measurement.clamp_charge_nA_ms,
                voltage_area_mV_ms=measurement.voltage_area_mV_ms,
                time_to_peak_ms=measurement.time_to_peak_ms,
            )
        return self.cache[key]


class FastPairedEvaluator(PairedEvaluator):
    """Adaptive CPU replay used only inside the optimizer; final gates use RK4."""

    def __call__(self, row_id: str, params: CellDendriteParams) -> CandidateResponse:
        key = (row_id, params)
        if key not in self.cache:
            self.cache[key] = self._solve(self.records[row_id], params)
        return self.cache[key]

    def _solve(self, record: Mapping[str, Any], passive: CellDendriteParams) -> CandidateResponse:
        row = record["contract"]
        derived = record["derivation"]
        intrinsic = aglif_params_for_cell_type(str(row["post"]))
        c_dend = intrinsic.C_m * passive.dend_C_frac
        caps = np.asarray([intrinsic.C_m-c_dend, c_dend*(1-passive.dist_C_frac), c_dend*passive.dist_C_frac])
        gc = 2.0*(intrinsic.C_m/intrinsic.tau_m)*passive.g_c_scale
        gcd = gc*passive.dist_coupling_ratio
        allocation = derived["graded_allocation"]
        fractions = np.asarray([allocation[x] for x in ("soma", "proximal", "distal")], dtype=float)
        scale = float(derived["constraint"]["total_transfer_scale"])
        tau_r, tau_d = float(row["tau_rise_ms"]), float(row["tau_decay_ms"])
        peak_t = tau_d*tau_r*np.log(tau_d/tau_r)/(tau_d-tau_r)
        g0 = (1/tau_r-1/tau_d)/(np.exp(-peak_t/tau_d)-np.exp(-peak_t/tau_r))
        beta_integrator_factor = tau_r*tau_d/(tau_d-tau_r)
        amplitude = (
            float(row["source_gmax_nS"])*int(row["synapses_per_connection"])
            * scale*g0*beta_integrator_factor
        )
        e_rev, rest, event = float(row["e_rev_mV"]), float(record["source_rest_mV"]), PAIRED.EVENT_MS

        def conductance(t: float) -> np.ndarray:
            if t < event:
                return np.zeros(3)
            elapsed = t-event
            return amplitude*(np.exp(-elapsed/tau_d)-np.exp(-elapsed/tau_r))*fractions

        def current_rhs(t: float, y: np.ndarray) -> np.ndarray:
            vm, vd, vx, ia, idep = y
            gs = conductance(t)
            return np.asarray([
                (-(caps[0]/intrinsic.tau_m)*(vm-intrinsic.E_L)+gc*(vd-vm)-ia+idep+gs[0]*(e_rev-vm))/caps[0],
                (-(caps[1]/intrinsic.tau_m)*passive.dend_leak_scale*(vd-intrinsic.E_L)+gc*(vm-vd)+gcd*(vx-vd)+gs[1]*(e_rev-vd))/caps[1],
                (-(caps[2]/intrinsic.tau_m)*passive.dist_leak_scale*(vx-intrinsic.E_L)+gcd*(vd-vx)+gs[2]*(e_rev-vx))/caps[2],
                intrinsic.k_adap*(vm-intrinsic.E_L)-intrinsic.k2*ia,
                -intrinsic.k1*idep,
            ])

        def clamp_rhs(t: float, y: np.ndarray) -> np.ndarray:
            vd, vx, ia, idep = y
            gs = conductance(t)
            return np.asarray([
                (-(caps[1]/intrinsic.tau_m)*passive.dend_leak_scale*(vd-intrinsic.E_L)+gc*(rest-vd)+gcd*(vx-vd)+gs[1]*(e_rev-vd))/caps[1],
                (-(caps[2]/intrinsic.tau_m)*passive.dist_leak_scale*(vx-intrinsic.E_L)+gcd*(vd-vx)+gs[2]*(e_rev-vx))/caps[2],
                intrinsic.k_adap*(rest-intrinsic.E_L)-intrinsic.k2*ia,
                -intrinsic.k1*idep,
            ])

        times = np.arange(0.0, event+PAIRED.POST_EVENT_MS+self.dt_ms/2, self.dt_ms)
        common = dict(t_span=(0.0, float(times[-1])), t_eval=times, rtol=2e-5, atol=2e-7, max_step=0.5)
        current = solve_ivp(current_rhs, y0=[rest, rest, rest, 0.0, 0.0], **common)
        clamp = solve_ivp(clamp_rhs, y0=[rest, rest, 0.0, 0.0], **common)
        vm = current.y[0]
        base_mask = (times >= event-20.0) & (times < event)
        response_mask = (times >= event) & (times <= event+PAIRED.POST_EVENT_MS)
        response_time = times[response_mask]
        delta_v = vm[response_mask]-float(vm[base_mask].mean())
        peak_index = int(np.argmax(delta_v))
        vd, ia, idep = clamp.y[0], clamp.y[2], clamp.y[3]
        gsoma = np.asarray([conductance(t)[0] for t in times])
        hold = -(
            -(caps[0]/intrinsic.tau_m)*(rest-intrinsic.E_L)+gc*(vd-rest)-ia+idep
            + gsoma*(e_rev-rest)
        )/1000.0
        hold_delta = hold[response_mask]-float(hold[base_mask].mean())
        return CandidateResponse(
            peak_mV=float(delta_v[peak_index]),
            clamp_charge_nA_ms=float(-np.trapz(hold_delta, response_time)),
            voltage_area_mV_ms=float(np.trapz(delta_v, response_time)),
            time_to_peak_ms=float(response_time[peak_index]-event),
        )


def _linear_matrix(cell: str, params: CellDendriteParams) -> tuple[np.ndarray, np.ndarray]:
    intrinsic = aglif_params_for_cell_type(cell)
    c_dend = intrinsic.C_m * params.dend_C_frac
    caps = np.asarray(
        [
            intrinsic.C_m - c_dend,
            c_dend * (1.0 - params.dist_C_frac),
            c_dend * params.dist_C_frac,
        ]
    )
    gc = 2.0 * (intrinsic.C_m / intrinsic.tau_m) * params.g_c_scale
    gcd = gc * params.dist_coupling_ratio
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0] = -(caps[0] / intrinsic.tau_m + gc) / caps[0]
    matrix[0, 1] = gc / caps[0]
    matrix[0, 3] = -1.0 / caps[0]
    matrix[1, 0] = gc / caps[1]
    matrix[1, 1] = -(caps[1] / intrinsic.tau_m * params.dend_leak_scale + gc + gcd) / caps[1]
    matrix[1, 2] = gcd / caps[1]
    matrix[2, 1] = gcd / caps[2]
    matrix[2, 2] = -(caps[2] / intrinsic.tau_m * params.dist_leak_scale + gcd) / caps[2]
    matrix[3, 0] = intrinsic.k_adap
    matrix[3, 3] = -intrinsic.k2
    drive = np.asarray([1.0 / caps[0], 0.0, 0.0, 0.0])
    return matrix, drive


def intrinsic_features(cell: str, params: CellDendriteParams) -> dict[str, Any]:
    """Held-out DC/passive gates of the same three-compartment equations."""
    matrix, drive = _linear_matrix(cell, params)
    steady_per_pA = np.linalg.solve(matrix, -drive)
    rin_mohm = float(1000.0 * steady_per_pA[0])
    # 63.2% step time is protocol-grounded and robust to the fast internal modes.
    times = np.linspace(0.0, 100.0, 2001)
    eigenvalues, vectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(vectors)
    coefficients = inverse @ (-steady_per_pA)
    voltage = np.asarray(
        [steady_per_pA[0] + (vectors @ (np.exp(eigenvalues * t) * coefficients))[0] for t in times],
        dtype=complex,
    ).real
    threshold = 0.6321205588 * steady_per_pA[0]
    indices = np.flatnonzero(voltage >= threshold)
    tau_ms = float(times[indices[0]]) if len(indices) else float("nan")
    currents = np.asarray(json.loads((ROOT / "src/ca1/params/ground_truth.json").read_text())[cell]["currents_nA"])
    rates = _fi_replay(cell, params, currents)
    positive = np.flatnonzero(rates > 0.0)
    rheobase = float(currents[positive[0]]) if len(positive) else float("nan")
    return {
        "Rin_MOhm": rin_mohm,
        "tau_m_ms": tau_ms,
        "rheobase_nA_on_held_out_ladder": rheobase,
        "fi_currents_nA": currents.tolist(),
        "fi_rates_hz": rates.tolist(),
    }


def _fi_replay(cell: str, passive: CellDendriteParams, currents_nA: np.ndarray) -> np.ndarray:
    """Deterministic CPU replay of user_m2 DC current injection (0.05 ms)."""
    p = aglif_params_for_cell_type(cell)
    dt, duration, settle = 0.05, 600.0, 100.0
    c_dend = p.C_m * passive.dend_C_frac
    caps = np.asarray([p.C_m - c_dend, c_dend * (1-passive.dist_C_frac), c_dend*passive.dist_C_frac])
    gc = 2.0 * (p.C_m / p.tau_m) * passive.g_c_scale
    gcd = gc * passive.dist_coupling_ratio
    state = np.zeros((len(currents_nA), 5), dtype=float)
    state[:, :3] = p.E_L
    refractory = np.zeros(len(currents_nA), dtype=int)
    counts = np.zeros(len(currents_nA), dtype=int)

    def derivative(y: np.ndarray) -> np.ndarray:
        vm, vd, vx, ia, idep = y.T
        out = np.empty_like(y)
        out[:, 0] = (-(caps[0]/p.tau_m)*(vm-p.E_L)+gc*(vd-vm)-ia+idep+currents_nA*1000.0)/caps[0]
        out[:, 1] = (-(caps[1]/p.tau_m)*passive.dend_leak_scale*(vd-p.E_L)+gc*(vm-vd)+gcd*(vx-vd))/caps[1]
        out[:, 2] = (-(caps[2]/p.tau_m)*passive.dist_leak_scale*(vx-p.E_L)+gcd*(vd-vx))/caps[2]
        out[:, 3] = p.k_adap*(vm-p.E_L)-p.k2*ia
        out[:, 4] = -p.k1*idep
        out[refractory > 0, 0] = 0.0
        return out

    for step in range(int(duration/dt)):
        k1 = derivative(state)
        k2 = derivative(state+0.5*dt*k1)
        k3 = derivative(state+0.5*dt*k2)
        k4 = derivative(state+dt*k3)
        state += dt*(k1+2*k2+2*k3+k4)/6.0
        active_refrac = refractory > 0
        state[active_refrac, 0] = p.V_reset
        refractory[active_refrac] -= 1
        spiked = (~active_refrac) & (state[:, 0] >= p.V_th)
        if step*dt >= settle:
            counts += spiked
        state[spiked, 0] = p.V_reset
        state[spiked, 3] += p.A2
        state[spiked, 4] = p.A1
        refractory[spiked] = int(round(p.t_ref/dt))
    return counts / ((duration-settle)/1000.0)


def _row_report(
    target: SourceResponseTarget,
    evaluator: PairedEvaluator,
    old: CellDendriteParams,
    new: CellDendriteParams,
) -> dict[str, Any]:
    old_response, new_response = evaluator(target.row_id, old), evaluator(target.row_id, new)
    old_ratios, new_ratios = response_ratios(target, old_response), response_ratios(target, new_response)
    return {
        "row": target.row_id,
        "source_target": asdict(target),
        "old": {"response": asdict(old_response), **{f"{k}_percent": 100*v for k, v in old_ratios.items()}},
        "new": {
            "response": asdict(new_response),
            **{f"{k}_percent": 100*v for k, v in new_ratios.items()},
            "hard_constraints_pass": response_constraints(target, new_response),
        },
    }


def run(source_path: Path, output: Path, maxiter: int) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cells: dict[str, Any] = {}
    for cell in TARGET_CELLS:
        records = [x for x in source["rows"] if x["contract"]["post"] == cell]
        targets = [_source_target(x) for x in records]
        evaluator = PairedEvaluator(records, 0.025)
        fit_evaluator = FastPairedEvaluator(records, 0.025)
        old = _old_params(cell)
        # O_LM's only row is already a hard-gate pass and is somatic; distal
        # variables are therefore non-identifiable and remain audited-fixed.
        if cell == "O_LM" and all(response_constraints(t, evaluator(t.row_id, old)) for t in targets):
            fitted, fit_meta = old, {
                "loss": None, "constraints_satisfied": True, "evaluations": 0,
                "opened_distal_params": False,
                "distal_audit": "not identifiable from the all-somatic O_LM row; retained deployed values",
            }
        else:
            fixed_distal_fit = fit_joint_cell_transfer(
                targets, fit_evaluator, old, open_distal=False, maxiter=maxiter
            )
            # Open distal quantities only if the preregistered hard gates cannot
            # be met with the three formerly hidden proximal/cell variables.
            fit = fixed_distal_fit
            if not fixed_distal_fit.constraints_satisfied:
                fit = fit_joint_cell_transfer(
                    targets, fit_evaluator, fixed_distal_fit.params,
                    open_distal=True, maxiter=maxiter,
                )
            fitted, fit_meta = fit.params, asdict(fit)
            fit_meta.pop("params")
            fit_meta["distal_audit"] = (
                "retained deployed distal values: three-parameter joint fit met all hard gates"
                if not fit.opened_distal_params else
                "opened after the three-parameter joint fit failed at least one hard gate"
            )
        rows = [_row_report(t, evaluator, old, fitted) for t in targets]
        cells[cell] = {
            "old_params": asdict(old), "fitted_params": asdict(fitted), "fit": fit_meta,
            "rows": rows,
            "intrinsic_before": intrinsic_features(cell, old),
            "intrinsic_after": intrinsic_features(cell, fitted),
            "all_rows_pass": all(x["new"]["hard_constraints_pass"] for x in rows),
        }
    report = {
        "schema": "dendrite-refit-candidate/v1",
        "provenance": {
            "method": "joint-cell-level paired source-NEURON vs user_m2 CPU replay",
            "fit_response": "somatic peak, voltage-clamp charge, voltage area, and time-to-peak",
            "hard_acceptance": "charge >= 90%; peak in [85%, 115%]",
            "shared_vector_per_cell": True,
            "source_gmax_kinetics_locations_contacts_immutable": True,
            "table5_rates_used": False,
            "rate_tuned": False,
            "deployed_params_unchanged": True,
            "source_capture": str(source_path),
        },
        "cells": cells,
        "all_cells_all_rows_pass": all(x["all_rows_pass"] for x in cells.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def add_heldout_validation(
    source_path: Path,
    output: Path,
    capture_paths: list[Path],
) -> dict[str, Any]:
    """Replay an existing candidate against independent location captures."""
    report = json.loads(output.read_text(encoding="utf-8"))
    fit_source = json.loads(source_path.read_text(encoding="utf-8"))
    fit_records = {
        f"{x['contract']['pre']}->{x['contract']['post']}": x
        for x in fit_source["rows"]
    }
    validation: list[dict[str, Any]] = []
    for path in capture_paths:
        capture = json.loads(path.read_text(encoding="utf-8"))
        contract = capture["contract"]
        row_id = f"{contract['pre']}->{contract['post']}"
        base = fit_records[row_id]
        record = {
            "contract": contract,
            "source_rest_mV": capture["source_rest_mV"],
            "source_neuron": {"summary": capture["source_summary"]},
            "derivation": base["derivation"],
        }
        target = _source_target(record)
        cell = str(contract["post"])
        params = CellDendriteParams(**report["cells"][cell]["fitted_params"])
        evaluator = PairedEvaluator([record], float(capture["dt_ms"]))
        response = evaluator(row_id, params)
        ratios = response_ratios(target, response)
        validation.append({
            "row": row_id,
            "dt_ms": float(capture["dt_ms"]),
            "location_seed": capture["location_seed"],
            "n_location_draws": capture["n_location_draws"],
            "capture": str(path),
            "response": asdict(response),
            **{f"{key}_percent": 100.0*value for key, value in ratios.items()},
            "hard_constraints_pass": response_constraints(target, response),
        })
    report["held_out_validation"] = validation
    report["held_out_all_pass"] = all(x["hard_constraints_pass"] for x in validation)
    report["dt_stable"] = all(
        abs(a["peak_percent"]-b["peak_percent"]) <= 2.0
        and abs(a["charge_percent"]-b["charge_percent"]) <= 2.0
        for index, a in enumerate(validation)
        for b in validation[index+1:]
        if a["row"] == b["row"] and a["location_seed"] == b["location_seed"]
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument(
        "--validate-capture", action="append", type=Path, default=[],
        help="add an independent paired-source-response capture (repeatable)",
    )
    args = parser.parse_args()
    report = (
        add_heldout_validation(args.source, args.output, args.validate_capture)
        if args.validate_capture else run(args.source, args.output, args.maxiter)
    )
    print(json.dumps({cell: {"params": x["fitted_params"], "pass": x["all_rows_pass"]} for cell, x in report["cells"].items()}, indent=2))


if __name__ == "__main__":
    main()
