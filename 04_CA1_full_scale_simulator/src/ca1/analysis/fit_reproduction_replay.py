from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ca1.analysis.fit_reproduction_data import (
    aglif_passive_estimate,
    json_field,
    json_mapping,
    json_number,
    load_targets,
    read_json_mapping,
)
from ca1.analysis.fit_reproduction_schema import (
    CELL_ORDER,
    FloatArray,
    JsonValue,
    PassiveValues,
    TargetCell,
)
from ca1.params.aglif import AGLIFParams


def build_aglif_replay_report(
    gt_path: Path,
    aglif_path: Path,
    cell_order: tuple[str, ...] = CELL_ORDER,
) -> dict[str, JsonValue]:
    from ca1.params.aglif_fit import BatchAGLIFFI

    targets = load_targets(gt_path, cell_order)
    raw = read_json_mapping(aglif_path)
    n_currents = int(targets[cell_order[0]].currents_nA.size)
    batch = BatchAGLIFFI(pop=1, n_currents=n_currents)
    report: dict[str, JsonValue] = {}
    for cell_name in cell_order:
        target = targets[cell_name]
        if int(target.currents_nA.size) != n_currents:
            raise ValueError("A-GLIF replay requires a fixed current count per cell")
        record = json_mapping(json_field(raw, cell_name), f"{aglif_path}:{cell_name}")
        params = _aglif_params(record)
        currents_pA = [float(value) * 1000.0 for value in target.currents_nA]
        rate_matrix, _trains = batch.evaluate([params], currents_pA)
        rates = np.asarray(rate_matrix[0], dtype=float)
        passive = aglif_passive_estimate(record)
        z_values = _z_values(target, rates, passive)
        report[cell_name] = {
            "rates_hz": [float(value) for value in rates],
            "count_window_ms": 500.0,
            "passive": _passive_json(passive),
            "passed": bool(np.median(z_values) <= 1.5 and np.max(z_values) <= 4.0),
            "median_z": float(np.median(z_values)),
            "max_z": float(np.max(z_values)),
            "hard_fails": _hard_fails(target, rates, passive),
            "protocol": "nestgpu-fi-replay-with-analytic-passive",
        }
    return report


def write_aglif_replay_report(report: dict[str, JsonValue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _aglif_params(record: dict[str, JsonValue]) -> AGLIFParams:
    return AGLIFParams(
        V_th=json_number(json_field(record, "V_th"), "V_th"),
        E_L=json_number(json_field(record, "E_L"), "E_L"),
        C_m=json_number(json_field(record, "C_m"), "C_m"),
        tau_m=json_number(json_field(record, "tau_m"), "tau_m"),
        k_adap=json_number(json_field(record, "k_adap"), "k_adap"),
        k1=json_number(json_field(record, "k1"), "k1"),
        k2=json_number(json_field(record, "k2"), "k2"),
        A1=json_number(json_field(record, "A1"), "A1"),
        A2=json_number(json_field(record, "A2"), "A2"),
        I_e=json_number(json_field(record, "I_e"), "I_e"),
        V_peak=json_number(json_field(record, "V_peak"), "V_peak"),
        V_reset=json_number(json_field(record, "V_reset"), "V_reset"),
        t_ref=json_number(json_field(record, "t_ref"), "t_ref"),
    )


def _z_values(target: TargetCell, rates: FloatArray, passive: PassiveValues) -> FloatArray:
    stop = min(target.peak_index + 1, int(rates.size), int(target.rates_hz.size))
    rate_z = np.abs((rates[:stop] - target.rates_hz[:stop]) / target.rate_sigma_hz[:stop])
    model_passive = passive.as_array()
    target_passive = target.passive.as_array()
    sigma = target.passive_sigma.as_array()
    finite = np.isfinite(model_passive) & (sigma > 0.0)
    passive_z = np.abs((model_passive[finite] - target_passive[finite]) / sigma[finite])
    return np.concatenate([rate_z, passive_z])


def _hard_fails(target: TargetCell, rates: FloatArray, passive: PassiveValues) -> list[JsonValue]:
    failures: list[JsonValue] = []
    stop = min(target.peak_index + 1, int(rates.size), int(target.rates_hz.size))
    for idx in range(stop):
        current = float(target.currents_nA[idx])
        model_rate = float(rates[idx])
        target_rate = float(target.rates_hz[idx])
        diff = abs(model_rate - target_rate)
        limit = max(4.0, 0.30 * target_rate)
        if diff > limit:
            failures.append(f"rate[{current:.3f}nA] {model_rate:.1f}!={target_rate:.1f}Hz")
    if abs(passive.rin_mohm - target.passive.rin_mohm) > 0.25 * target.passive.rin_mohm:
        failures.append(f"Rin {passive.rin_mohm:.0f}!={target.passive.rin_mohm:.0f}")
    if abs(passive.tau_ms - target.passive.tau_ms) > 0.25 * target.passive.tau_ms:
        failures.append(f"tau_m {passive.tau_ms:.1f}!={target.passive.tau_ms:.1f}")
    if abs(passive.e_l_mv - target.passive.e_l_mv) > 2.0:
        failures.append(f"E_L {passive.e_l_mv:.1f}!={target.passive.e_l_mv:.1f}")
    return failures


def _passive_json(passive: PassiveValues) -> dict[str, JsonValue]:
    return {
        "Rin": passive.rin_mohm,
        "tau_m": passive.tau_ms,
        "E_L": passive.e_l_mv,
        "sag": passive.sag_mv,
    }
