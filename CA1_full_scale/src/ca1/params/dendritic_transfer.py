from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from ca1.params.groundtruth import CELL_TEMPLATES
from .json_io import (
    JsonValue,
    load_json_mapping,
    numeric_field,
    reject_malformed_validation_passed,
    required_string_field,
)

_DEFAULT_PATH: Final[Path] = Path(__file__).with_name(
    "dendritic_transfer_fitted.json"
)


@dataclass(frozen=True, slots=True)
class DendriticTransferParams:
    dend_C_frac: float
    dend_leak_scale: float
    g_c_scale: float
    fit_provenance: str

    def g_c_nS(self, membrane_conductance_nS: float) -> float:
        return 2.0 * membrane_conductance_nS * self.g_c_scale


def _float_field(record: Mapping[str, JsonValue], key: str) -> float:
    return numeric_field(record, key, context="dendritic-transfer")


def _str_field(record: Mapping[str, JsonValue], key: str) -> str:
    return required_string_field(
        record,
        key,
        context="dendritic-transfer",
    )


def _from_record(record: Mapping[str, JsonValue]) -> DendriticTransferParams:
    params = DendriticTransferParams(
        dend_C_frac=_float_field(record, "dend_C_frac"),
        dend_leak_scale=_float_field(record, "dend_leak_scale"),
        g_c_scale=_float_field(record, "g_c_scale"),
        fit_provenance=_str_field(record, "fit_provenance"),
    )
    if not 0.0 < params.dend_C_frac < 1.0:
        raise ValueError("dend_C_frac must be in (0, 1)")
    if params.dend_leak_scale <= 0.0:
        raise ValueError("dend_leak_scale must be positive")
    if params.g_c_scale <= 0.0:
        raise ValueError("g_c_scale must be positive")
    return params


def _raise_if_rejected_validation(
    cell_type: str,
    record: Mapping[str, JsonValue],
    path: Path,
) -> None:
    try:
        reject_malformed_validation_passed(
            record,
            context=f"dendritic-transfer fit for {cell_type!r}",
        )
    except ValueError as exc:
        raise ValueError(
            f"dendritic-transfer fit for {cell_type!r} in {path} "
            + f"failed validation ({exc}); regenerate or remove the rejected record"
        ) from exc


def load_dendritic_transfer_params(
    path: Path | None = None,
) -> dict[str, DendriticTransferParams]:
    fit_path = path or _DEFAULT_PATH
    if not fit_path.exists():
        raise FileNotFoundError(
            f"missing dendritic-transfer params: {fit_path}. "
            + "Use an explicit complete transfer file so rejected/default "
            + "values are visible in provenance."
        )

    raw = load_json_mapping(fit_path, context="dendritic transfer params")

    expected = set(CELL_TEMPLATES)
    unknown = set(raw) - expected
    if unknown:
        unknown_list = sorted(unknown)
        unknown_msg = (
            f"unknown cell type {unknown_list[0]!r}"
            if len(unknown_list) == 1
            else f"unknown cell types: {unknown_list}"
        )
        raise ValueError(
            f"dendritic-transfer fits in {fit_path} contain {unknown_msg}"
        )

    result: dict[str, DendriticTransferParams] = {}
    for cell_type, record in raw.items():
        if not isinstance(record, dict):
            raise TypeError(f"invalid dendritic-transfer record for {cell_type!r}")
        _raise_if_rejected_validation(cell_type, record, fit_path)
        params = _from_record(record)
        if params.fit_provenance == "FAILED":
            raise ValueError(
                f"dendritic-transfer fit for {cell_type!r} in {fit_path} "
                + "is marked FAILED; remove the record or regenerate a validated fit"
            )
        result[cell_type] = params
    missing = expected - set(raw)
    if missing:
        raise ValueError(
            f"missing dendritic-transfer fit in {fit_path}: {sorted(missing)}"
        )
    return result


@lru_cache(maxsize=1)
def _dendritic_transfer_params() -> dict[str, DendriticTransferParams]:
    return load_dendritic_transfer_params()


def dendritic_transfer_for_cell_type(name: str) -> DendriticTransferParams:
    return _dendritic_transfer_params()[name]
