from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ca1.analysis.fit_reproduction_schema import (
    CELL_ORDER,
    FitCell,
    FitMetrics,
    FloatArray,
    JsonValue,
    PassiveValues,
    ReproductionDataset,
    TargetCell,
)

_NAN = float("nan")


def load_reproduction_dataset(
    gt_path: Path,
    aeif_path: Path,
    aglif_path: Path,
    aglif_report_path: Path | None = None,
    cell_order: tuple[str, ...] = CELL_ORDER,
) -> ReproductionDataset:
    targets = load_targets(gt_path, cell_order)
    return ReproductionDataset(
        cell_order=cell_order,
        targets=targets,
        fits={
            "AEIF": load_aeif_fits(aeif_path, cell_order),
            "A-GLIF": load_aglif_fits(aglif_path, cell_order, aglif_report_path),
        },
    )


def load_targets(path: Path, cell_order: tuple[str, ...] = CELL_ORDER) -> dict[str, TargetCell]:
    raw = _read_mapping(path)
    targets: dict[str, TargetCell] = {}
    for cell_name in cell_order:
        record = _mapping(_field(raw, cell_name), f"{path}:{cell_name}")
        sigma = _mapping(_field(record, "sigma"), f"{path}:{cell_name}.sigma")
        currents = np.asarray(_number_list(_field(record, "currents_nA"), "currents_nA"))
        rates = np.asarray(_number_list(_field(record, "rates_hz"), "rates_hz"))
        rate_sigma = np.asarray(_number_list(_field(sigma, "rates_hz"), "sigma.rates_hz"))
        if currents.shape != rates.shape or rates.shape != rate_sigma.shape:
            raise ValueError(f"{cell_name} current/rate/sigma lengths do not match")
        targets[cell_name] = TargetCell(
            name=cell_name,
            currents_nA=currents,
            rates_hz=rates,
            rate_sigma_hz=rate_sigma,
            passive=_passive(record),
            passive_sigma=_passive(sigma),
            rheobase_nA=_number(_field(record, "rheobase_nA"), "rheobase_nA"),
            count_window_ms=600.0,
        )
    return targets


def load_aeif_fits(path: Path, cell_order: tuple[str, ...] = CELL_ORDER) -> dict[str, FitCell]:
    raw = _read_mapping(path)
    fits: dict[str, FitCell] = {}
    for cell_name in cell_order:
        record = _mapping(_field(raw, cell_name), f"{path}:{cell_name}")
        validation = _optional_mapping(record.get("validation"))
        rates = None
        passive = None
        passed = None
        median_z = None
        max_z = None
        hard_fails: tuple[str, ...] = ()
        protocol = "fit-record"
        if validation is not None:
            rates = np.asarray(_number_list(_field(validation, "nest_rates_hz"), "nest_rates_hz"))
            passive = _passive(
                _mapping(_field(validation, "nest_passive"), "validation.nest_passive")
            )
            passed = _bool(_field(validation, "passed"), "validation.passed")
            median_z = _number(_field(validation, "median_z"), "validation.median_z")
            max_z = _number(_field(validation, "max_z"), "validation.max_z")
            hard_fails = _string_tuple(validation.get("hard_fails"))
            protocol = "cpu-nest-validation"
        fits[cell_name] = FitCell(
            model="AEIF",
            name=cell_name,
            rates_hz=rates,
            passive=passive,
            loss=_maybe_number(record.get("loss"), "loss"),
            passed=passed,
            median_z=median_z,
            max_z=max_z,
            hard_fails=hard_fails,
            protocol=protocol,
            count_window_ms=600.0,
        )
    return fits


def load_aglif_fits(
    path: Path,
    cell_order: tuple[str, ...] = CELL_ORDER,
    report_path: Path | None = None,
) -> dict[str, FitCell]:
    raw = _read_mapping(path)
    report = _read_mapping(report_path) if report_path is not None and report_path.exists() else {}
    fits: dict[str, FitCell] = {}
    for cell_name in cell_order:
        record = _mapping(_field(raw, cell_name), f"{path}:{cell_name}")
        validation = _optional_mapping(report.get(cell_name))
        rates = None
        passive = aglif_passive_estimate(record)
        passed = None
        median_z = None
        max_z = None
        hard_fails: tuple[str, ...] = ()
        protocol = "fit-objective-passive-estimate"
        if validation is not None:
            rates = np.asarray(_number_list(_field(validation, "rates_hz"), "rates_hz"))
            passive = _passive(_mapping(_field(validation, "passive"), "aglif.passive"))
            passed = _bool(_field(validation, "passed"), "passed")
            median_z = _number(_field(validation, "median_z"), "median_z")
            max_z = _number(_field(validation, "max_z"), "max_z")
            hard_fails = _string_tuple(validation.get("hard_fails"))
            protocol = str(validation.get("protocol", "nestgpu-fi-replay"))
            count_window_ms = _maybe_number(validation.get("count_window_ms"), "count_window_ms")
        else:
            count_window_ms = None
        fits[cell_name] = FitCell(
            model="A-GLIF",
            name=cell_name,
            rates_hz=rates,
            passive=passive,
            loss=_maybe_number(record.get("loss"), "loss"),
            passed=passed,
            median_z=median_z,
            max_z=max_z,
            hard_fails=hard_fails,
            protocol=protocol,
            count_window_ms=500.0 if count_window_ms is None else count_window_ms,
        )
    return fits


def fit_metrics(target: TargetCell, fit: FitCell) -> FitMetrics:
    return FitMetrics(
        model=fit.model,
        cell_name=target.name,
        rate_rmse_z=_rate_rmse_z(target, fit.rates_hz),
        passive_rmse_z=_passive_rmse_z(target, fit.passive),
        loss=_NAN if fit.loss is None else fit.loss,
    )


def read_json_mapping(path: Path) -> dict[str, JsonValue]:
    return _read_mapping(path)


def json_field(record: dict[str, JsonValue], key: str) -> JsonValue:
    return _field(record, key)


def json_mapping(value: JsonValue, label: str) -> dict[str, JsonValue]:
    return _mapping(value, label)


def json_number(value: JsonValue, label: str) -> float:
    return _number(value, label)


def _read_mapping(path: Path) -> dict[str, JsonValue]:
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(raw, str(path))


def _field(record: dict[str, JsonValue], key: str) -> JsonValue:
    try:
        return record[key]
    except KeyError as exc:
        raise KeyError(f"missing JSON field {key!r}") from exc


def _mapping(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return value
    raise TypeError(f"{label} must be a JSON object")


def _optional_mapping(value: JsonValue | None) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    return _mapping(value, "optional mapping")


def _number(value: JsonValue, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"{label} must be numeric")


def _maybe_number(value: JsonValue | None, label: str) -> float | None:
    if value is None:
        return None
    return _number(value, label)


def _bool(value: JsonValue, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{label} must be boolean")


def _number_list(value: JsonValue, label: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return [_number(item, label) for item in value]


def _string_tuple(value: JsonValue | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("expected a list of strings")
    return tuple(item for item in value if isinstance(item, str))


def _passive(record: dict[str, JsonValue]) -> PassiveValues:
    return PassiveValues(
        rin_mohm=_number(_field(record, "Rin"), "Rin"),
        tau_ms=_number(_field(record, "tau_m"), "tau_m"),
        e_l_mv=_number(_field(record, "E_L"), "E_L"),
        sag_mv=_number(record.get("sag", _NAN), "sag"),
    )


def aglif_passive_estimate(record: dict[str, JsonValue]) -> PassiveValues:
    c_m = _number(_field(record, "C_m"), "C_m")
    tau_m = _number(_field(record, "tau_m"), "tau_m")
    k_adap = _number(_field(record, "k_adap"), "k_adap")
    k2 = max(_number(_field(record, "k2"), "k2"), 1.0e-9)
    g_eff = c_m / tau_m + k_adap / k2
    return PassiveValues(
        rin_mohm=1000.0 / g_eff,
        tau_ms=c_m / g_eff,
        e_l_mv=_number(_field(record, "E_L"), "E_L"),
        sag_mv=_NAN,
    )


def _rate_rmse_z(target: TargetCell, rates: FloatArray | None) -> float:
    if rates is None:
        return _NAN
    stop = min(target.peak_index + 1, int(rates.size), int(target.rates_hz.size))
    z_values = (rates[:stop] - target.rates_hz[:stop]) / target.rate_sigma_hz[:stop]
    return float(np.sqrt(np.mean(z_values * z_values)))


def _passive_rmse_z(target: TargetCell, passive: PassiveValues | None) -> float:
    if passive is None:
        return _NAN
    target_values = target.passive.as_array()
    model_values = passive.as_array()
    sigmas = target.passive_sigma.as_array()
    finite = np.isfinite(model_values) & np.isfinite(sigmas) & (sigmas > 0.0)
    if not np.any(finite):
        return _NAN
    z_values = (model_values[finite] - target_values[finite]) / sigmas[finite]
    return float(np.sqrt(np.mean(z_values * z_values)))
