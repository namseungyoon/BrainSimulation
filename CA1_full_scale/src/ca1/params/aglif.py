from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from .json_io import (
    JsonValue,
    load_json_mapping,
    numeric_field,
    reject_malformed_validation_passed,
    required_string_field,
)
from ..types import NestParams

EXPECTED_CELL_TYPES: Final[frozenset[str]] = frozenset({
    "Pyramidal",
    "PV_Basket",
    "CCK_Basket",
    "Axo",
    "Bistratified",
    "Ivy",
    "O_LM",
    "SCA",
    "Neurogliaform",
})


@dataclass(frozen=True, slots=True)
class AGLIFParams:
    V_th: float
    E_L: float
    C_m: float
    tau_m: float
    k_adap: float
    k1: float
    k2: float
    A1: float
    A2: float
    I_e: float
    V_peak: float
    V_reset: float
    t_ref: float

    def as_nest(self) -> NestParams:
        return {
            "V_th": self.V_th,
            "E_L": self.E_L,
            "C_m": self.C_m,
            "tau_m": self.tau_m,
            "k_adap": self.k_adap,
            "k1": self.k1,
            "k2": self.k2,
            "A1": self.A1,
            "A2": self.A2,
            "I_e": self.I_e,
            "V_peak": self.V_peak,
            "V_reset": self.V_reset,
            "t_ref": self.t_ref,
        }


def _float_field(record: Mapping[str, JsonValue], key: str) -> float:
    return numeric_field(record, key, context="A-GLIF")


def _from_fit_record(record: Mapping[str, JsonValue]) -> AGLIFParams:
    return AGLIFParams(
        V_th=_float_field(record, "V_th"),
        E_L=_float_field(record, "E_L"),
        C_m=_float_field(record, "C_m"),
        tau_m=_float_field(record, "tau_m"),
        k_adap=_float_field(record, "k_adap"),
        k1=_float_field(record, "k1"),
        k2=_float_field(record, "k2"),
        A1=_float_field(record, "A1"),
        A2=_float_field(record, "A2"),
        I_e=_float_field(record, "I_e"),
        V_peak=_float_field(record, "V_peak"),
        V_reset=_float_field(record, "V_reset"),
        t_ref=_float_field(record, "t_ref"),
    )


def load_aglif_params(path: Path | None = None) -> dict[str, AGLIFParams]:
    expected = set(EXPECTED_CELL_TYPES)
    fit_path = path or Path(__file__).with_name("aglif_parameters_fitted.json")
    if not fit_path.exists():
        raise FileNotFoundError(
            f"missing A-GLIF fitted params: {fit_path}. "
            + "Do not rely on implicit AdEx-derived fallback for production runs."
        )

    raw = load_json_mapping(fit_path, context="A-GLIF params file")

    unknown = set(raw) - expected
    if unknown:
        unknown_list = sorted(unknown)
        unknown_msg = (
            f"unknown cell type {unknown_list[0]!r}"
            if len(unknown_list) == 1
            else f"unknown cell types: {unknown_list}"
        )
        raise ValueError(
            f"A-GLIF params in {fit_path} contain {unknown_msg}"
        )

    params: dict[str, AGLIFParams] = {}
    for cell_type, record in raw.items():
        if not isinstance(record, dict):
            raise TypeError(f"invalid A-GLIF record in {fit_path}: {cell_type!r}")
        provenance = required_string_field(
            record,
            "fit_provenance",
            context="A-GLIF",
        )
        if provenance == "FAILED":
            raise ValueError(
                f"A-GLIF params for {cell_type!r} in {fit_path} are marked FAILED; "
                + "remove the record or regenerate a validated fit"
            )
        reject_malformed_validation_passed(
            record,
            context=f"A-GLIF params for {cell_type!r}",
        )
        params[cell_type] = _from_fit_record(record)
    missing = expected - set(raw)
    if missing:
        raise ValueError(
            f"missing A-GLIF params in {fit_path}: {sorted(missing)}"
        )
    return params


@lru_cache(maxsize=1)
def _aglif_params() -> dict[str, AGLIFParams]:
    return load_aglif_params()


def aglif_params_for_cell_type(name: str) -> AGLIFParams:
    return _aglif_params()[name]
