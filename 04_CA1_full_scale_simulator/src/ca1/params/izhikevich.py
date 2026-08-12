from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .json_io import (
    JsonValue,
    load_json_mapping,
    numeric_field,
    reject_malformed_validation_passed,
    required_string_field,
)

@dataclass(frozen=True, slots=True)
class IzhikevichParams:
    V_m: float
    u: float
    V_th: float
    a: float
    b: float
    c: float
    d: float
    I_e: float = 0.0
    t_ref: float = 0.0
    h_min_rel: float = 0.1
    h0_rel: float = 0.1
    current_gain: float = 1.0

    def as_nest(self) -> dict[str, float]:
        return {
            "V_m": self.V_m,
            "u": self.u,
            "V_th": self.V_th,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "I_e": self.I_e,
            "t_ref": self.t_ref,
            "h_min_rel": self.h_min_rel,
            "h0_rel": self.h0_rel,
        }


_RS: Final[IzhikevichParams] = IzhikevichParams(
    V_m=-70.0,
    u=-14.0,
    V_th=30.0,
    a=0.02,
    b=0.2,
    c=-65.0,
    d=8.0,
)
_FS: Final[IzhikevichParams] = IzhikevichParams(
    V_m=-70.0,
    u=-14.0,
    V_th=30.0,
    a=0.1,
    b=0.2,
    c=-65.0,
    d=2.0,
)
_LTS: Final[IzhikevichParams] = IzhikevichParams(
    V_m=-70.0,
    u=-17.5,
    V_th=30.0,
    a=0.02,
    b=0.25,
    c=-65.0,
    d=2.0,
)

_DEFAULTS: Final[dict[str, IzhikevichParams]] = {
    "Pyramidal": _RS,
    "PV_Basket": _FS,
    "Axo": _FS,
    "Bistratified": _FS,
    "CCK_Basket": _LTS,
    "Ivy": _LTS,
    "O_LM": _LTS,
    "SCA": _LTS,
    "Neurogliaform": _LTS,
}


def _float_field(record: Mapping[str, JsonValue], key: str) -> float:
    return numeric_field(record, key, context="Izhikevich")


def _from_fit_record(record: Mapping[str, JsonValue]) -> IzhikevichParams:
    return IzhikevichParams(
        V_m=_float_field(record, "V_m"),
        u=_float_field(record, "u"),
        V_th=_float_field(record, "V_th"),
        a=_float_field(record, "a"),
        b=_float_field(record, "b"),
        c=_float_field(record, "c"),
        d=_float_field(record, "d"),
        I_e=_float_field(record, "I_bias"),
        current_gain=_float_field(record, "I_gain"),
    )


def load_izhikevich_params(
    path: Path | None = None,
) -> dict[str, IzhikevichParams]:
    expected = set(_DEFAULTS)
    fit_path = path or Path(__file__).with_name("izhikevich_parameters_fitted.json")
    if not fit_path.exists():
        raise FileNotFoundError(
            f"missing Izhikevich fitted params: {fit_path}. "
            + "Do not rely on implicit preset fallback for production runs."
        )

    raw = load_json_mapping(fit_path, context="Izhikevich params file")

    unknown = set(raw) - expected
    if unknown:
        unknown_list = sorted(unknown)
        unknown_msg = (
            f"unknown cell type {unknown_list[0]!r}"
            if len(unknown_list) == 1
            else f"unknown cell types: {unknown_list}"
        )
        raise ValueError(
            f"Izhikevich params in {fit_path} contain {unknown_msg}"
        )

    params: dict[str, IzhikevichParams] = {}
    for cell_type, record in raw.items():
        if not isinstance(record, dict):
            raise TypeError(f"invalid Izhikevich record in {fit_path}: {cell_type!r}")
        provenance = required_string_field(
            record,
            "fit_provenance",
            context="Izhikevich",
        )
        if provenance == "FAILED":
            raise ValueError(
                f"Izhikevich params for {cell_type!r} in {fit_path} are "
                + "marked FAILED; remove the record or regenerate a validated fit"
            )
        try:
            reject_malformed_validation_passed(
                record,
                context=f"Izhikevich params for {cell_type!r}",
            )
        except ValueError as exc:
            raise ValueError(
                f"Izhikevich params for {cell_type!r} in {fit_path} failed "
                + f"validation ({exc}); remove the record or regenerate a validated fit"
            ) from exc
        params[cell_type] = _from_fit_record(record)
    missing = expected - set(raw)
    if missing:
        raise ValueError(
            f"missing Izhikevich params in {fit_path}: {sorted(missing)}"
        )
    return params


_ACTIVE_PARAMS: Final[dict[str, IzhikevichParams]] = load_izhikevich_params()


def izhikevich_params_for_cell_type(cell_type: str) -> IzhikevichParams:
    return _ACTIVE_PARAMS[cell_type]
