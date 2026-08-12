from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

import numpy as np
import numpy.typing as npt

from ca1.analysis.location_transfer import TransferMode
from ca1.params.provenance import (
    diagnostic_config_provenance,
    diagnostic_environment_provenance,
    stamp_clean_diagnostic_audit,
)
from ca1.params.receptor_ports import PortCompressionStrategy
from ca1.validation.targets import MODEL_RATES_HZ


@dataclass(frozen=True, slots=True)
class LocationTransferCase:
    case: str
    out_dir: Path
    model: str
    conndata_index: int
    syndata_variant: int
    duration_s: float
    crop_ms: float
    afferent_rate_hz: float
    recurrent_scale: float | None
    afferent_weight_scale: float | None
    ca3_source_scale: float | None
    eciii_source_scale: float | None
    gfast_scale: float | None
    gslow_scale: float | None
    gb_scale: float | None
    dend_ampa_scale: float | None
    delay_ms: float | None
    compartment_aware_synapses: bool
    receptor_port_strategy: PortCompressionStrategy
    transfer_mode: TransferMode
    transfer_table: Path
    allow_incomplete_transfer_for_prototype: bool
    projection_weight_scales: dict[str, float] | None = None
    afferent_post_weight_scales: dict[str, float] | None = None


def calibration_for(case: LocationTransferCase) -> dict[str, object]:
    calibration: dict[str, object] = {"mode": "paper_reduction"}
    if case.recurrent_scale is not None:
        calibration["recurrent_weight_scale"] = case.recurrent_scale
    if case.afferent_weight_scale is not None:
        calibration["afferent_weight_scale"] = case.afferent_weight_scale
    if case.dend_ampa_scale is not None:
        calibration["dendritic_ampa_weight_scale"] = case.dend_ampa_scale

    source_scales: dict[str, float] = {}
    if case.ca3_source_scale is not None:
        source_scales["CA3"] = case.ca3_source_scale
    if case.eciii_source_scale is not None:
        source_scales["ECIII"] = case.eciii_source_scale
    if source_scales:
        calibration["afferent_source_weight_scales"] = source_scales

    receptor_scales: dict[str, float] = {}
    if case.gfast_scale is not None:
        receptor_scales["GABA_A_fast"] = case.gfast_scale
    if case.gslow_scale is not None:
        receptor_scales["GABA_A_slow"] = case.gslow_scale
    if case.gb_scale is not None:
        receptor_scales["GABA_B"] = case.gb_scale
    if receptor_scales:
        calibration["recurrent_receptor_weight_scales"] = receptor_scales
    if case.projection_weight_scales:
        calibration["mode"] = "diagnostic"
        calibration["projection_weight_scales"] = case.projection_weight_scales
    if case.afferent_post_weight_scales:
        calibration["mode"] = "diagnostic"
        calibration["afferent_post_weight_scales"] = case.afferent_post_weight_scales
    return calibration


def diagnostic_gc_scale_env() -> dict[str, str]:
    prefix = "CA1_AGLIF_DEND_GC_SCALE"
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key == prefix or key.startswith(f"{prefix}_")
    }


def diagnostic_runtime_env() -> dict[str, str]:
    return diagnostic_environment_provenance(os.environ)


def diagnostic_case_config(case: LocationTransferCase) -> dict[str, str]:
    return diagnostic_config_provenance({"calibration": calibration_for(case)})


def rates_hz(
    spikes: dict[str, list[npt.NDArray[np.float64]]],
    crop_s: float,
    window_s: float,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for cell_type, cells in spikes.items():
        spike_count = sum(int(np.count_nonzero(cell >= crop_s)) for cell in cells)
        rates[cell_type] = round(spike_count / (len(cells) * window_s), 4) if cells else 0.0
    return rates


def score_rates(rates: dict[str, float]) -> tuple[dict[str, float], float]:
    errors = {
        cell_type: round(abs(rates.get(cell_type, 0.0) - target) / target, 5)
        for cell_type, target in MODEL_RATES_HZ.items()
    }
    mean_score = round(float(np.mean(list(errors.values()))), 5)
    return errors, mean_score


def fail_count(rates: dict[str, float]) -> int:
    failures = 0
    for cell_type, target in MODEL_RATES_HZ.items():
        measured = rates.get(cell_type, 0.0)
        if measured < target * 0.7 or measured > target * 1.3:
            failures += 1
    return failures


def _transfer_table_provenance(
    case: LocationTransferCase,
    transfer_missing: list[str],
) -> str:
    if case.transfer_mode == "none":
        return "unused"
    parts = [
        "unvalidated-prototype-source-location-transfer",
        f"mode={case.transfer_mode}",
        f"table={case.transfer_table.name}",
    ]
    if case.allow_incomplete_transfer_for_prototype:
        parts.append("incomplete-prototype-override")
    if transfer_missing:
        parts.append(f"missing_rows={len(set(transfer_missing))}")
    return ";".join(parts)


def _is_spike_dict(
    value: object,
) -> TypeGuard[dict[str, list[npt.NDArray[np.float64]]]]:
    if not isinstance(value, dict):
        return False
    for cell_type, cells in cast(dict[object, object], value).items():
        if not isinstance(cell_type, str) or not isinstance(cells, list):
            return False
        if not all(isinstance(cell, np.ndarray) for cell in cast(list[object], cells)):
            return False
    return True


def postprocess(
    case: LocationTransferCase,
    spike_path: Path,
    elapsed_s: float,
    parameter_provenance: dict[str, str],
    transfer_applied: list[dict[str, object]],
    transfer_missing: list[str],
) -> dict[str, object]:
    with spike_path.open("rb") as handle:
        loaded = cast(object, pickle.load(handle))
    if not _is_spike_dict(loaded):
        raise TypeError(f"{spike_path} does not contain spike arrays by cell type")
    spikes = loaded
    n_cells_total = sum(len(cells) for cells in spikes.values())
    if n_cells_total == 0:
        raise ValueError(f"{spike_path} must contain at least one analyzed cell")
    crop_s = case.crop_ms / 1000.0
    window_s = max(case.duration_s - crop_s, 1.0e-9)
    rates = rates_hz(spikes, crop_s, window_s)
    errors, mean_score = score_rates(rates)
    transfer_provenance = _transfer_table_provenance(case, transfer_missing)
    audited_parameter_provenance = dict(parameter_provenance)
    if case.transfer_mode != "none":
        audited_parameter_provenance[
            "source_location_transfer.table"
        ] = transfer_provenance
    diagnostic_environment = diagnostic_runtime_env()
    diagnostic_config = diagnostic_case_config(case)
    diagnostic_provenance = {**diagnostic_environment, **diagnostic_config}
    summary: dict[str, object] = {
        "case": case.case,
        "elapsed_s": round(elapsed_s, 3),
        "neuron_model": case.model,
        "n_cells_total": n_cells_total,
        "conndata_index": case.conndata_index,
        "syndata_variant": case.syndata_variant,
        "duration_s": case.duration_s,
        "crop_ms": case.crop_ms,
        "afferent_rate_hz": case.afferent_rate_hz,
        "recurrent_scale": case.recurrent_scale,
        "afferent_weight_scale": case.afferent_weight_scale,
        "ca3_source_scale": case.ca3_source_scale,
        "eciii_source_scale": case.eciii_source_scale,
        "gfast_scale": case.gfast_scale,
        "gslow_scale": case.gslow_scale,
        "gb_scale": case.gb_scale,
        "dendritic_ampa_weight_scale": case.dend_ampa_scale,
        "delay_ms": case.delay_ms,
        "compartment_aware_synapses": case.compartment_aware_synapses,
        "receptor_port_strategy": case.receptor_port_strategy,
        "transfer_mode": case.transfer_mode,
        "allow_incomplete_transfer_for_prototype": (
            case.allow_incomplete_transfer_for_prototype
        ),
        "projection_weight_scales": case.projection_weight_scales,
        "afferent_post_weight_scales": case.afferent_post_weight_scales,
        "transfer_table": str(case.transfer_table),
        "transfer_table_provenance": transfer_provenance,
        "transfer_applied_count": len(transfer_applied),
        "transfer_applied": transfer_applied,
        "transfer_missing_count": len(transfer_missing),
        "transfer_missing": transfer_missing,
        "diagnostic_gc_scale_env": diagnostic_gc_scale_env(),
        "diagnostic_environment_provenance": diagnostic_environment,
        "diagnostic_config_provenance": diagnostic_config,
        "diagnostic_provenance": stamp_clean_diagnostic_audit(
            diagnostic_provenance
        ),
        "parameter_provenance": audited_parameter_provenance,
        "rates_hz": rates,
        "targets_hz": MODEL_RATES_HZ,
        "relative_abs_error": errors,
        "score_mean_relative_abs_error": mean_score,
        "rate_fail_count": fail_count(rates),
        "active_types": sorted(cell_type for cell_type, rate in rates.items() if rate > 0.0),
        "spikes_total": sum(int(cell.size) for cells in spikes.values() for cell in cells),
    }
    _ = (case.out_dir / f"{case.case}_postprocessed.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
