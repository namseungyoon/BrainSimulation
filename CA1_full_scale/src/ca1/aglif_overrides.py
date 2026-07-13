from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING, Final, TypeGuard

if TYPE_CHECKING:
    from ca1.types import CellType

SOMA_EXCITATORY: Final = "soma_excitatory"


@dataclass(frozen=True)
class AglifDendOverride:
    receive_domain: str | None = None
    g_c_scale: float = 1.0
    model: str | None = None


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _config_mapping(value: object, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if not _is_object_mapping(value):
        raise TypeError(f"config field {field!r} must be a mapping")

    parsed: dict[str, object] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field} keys must be strings")
        parsed[key] = raw_value
    return parsed


def parse_aglif_receive_domain_overrides(
    value: object,
    cell_types: Mapping[str, CellType],
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for cell_type, mode in _config_mapping(
        value,
        "aglif_receive_domain_overrides",
    ).items():
        if cell_type not in cell_types:
            raise ValueError(f"unknown AGLIF override cell type: {cell_type!r}")
        if mode != SOMA_EXCITATORY:
            message = (
                f"unsupported AGLIF receive-domain mode for {cell_type!r}: "
                f"{mode!r}; expected {SOMA_EXCITATORY!r}"
            )
            raise ValueError(message)
        parsed[cell_type] = SOMA_EXCITATORY
    return parsed


def parse_aglif_gc_scale_overrides(
    value: object,
    cell_types: Mapping[str, CellType],
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for cell_type, scale in _config_mapping(
        value,
        "aglif_gc_scale_overrides",
    ).items():
        if cell_type not in cell_types:
            raise ValueError(f"unknown AGLIF override cell type: {cell_type!r}")
        if isinstance(scale, bool) or not isinstance(scale, int | float):
            raise ValueError(
                f"aglif_gc_scale_overrides.{cell_type} must be a positive number"
            )
        number = float(scale)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(
                f"aglif_gc_scale_overrides.{cell_type} must be a positive number"
            )
        parsed[cell_type] = number
    return parsed


def parse_aglif_dend_overrides(
    value: object,
    cell_types: Mapping[str, CellType],
) -> dict[str, AglifDendOverride]:
    parsed: dict[str, AglifDendOverride] = {}
    for cell_type, raw_override in _config_mapping(
        value,
        "aglif_dend_overrides",
    ).items():
        if cell_type not in cell_types:
            raise ValueError(f"unknown AGLIF override cell type: {cell_type!r}")
        if not _is_object_mapping(raw_override):
            raise TypeError(
                f"aglif_dend_overrides.{cell_type} must be a mapping"
            )

        unknown_fields = set(raw_override) - {"receive_domain", "g_c_scale", "model"}
        if unknown_fields:
            raise ValueError(
                f"aglif_dend_overrides.{cell_type} has unknown fields: "
                f"{sorted(unknown_fields, key=str)}"
            )

        receive_domain = raw_override.get("receive_domain")
        if receive_domain is not None and receive_domain != SOMA_EXCITATORY:
            raise ValueError(
                f"receive_domain for {cell_type!r} is {receive_domain!r}; "
                f"expected {SOMA_EXCITATORY!r}"
            )

        raw_scale = raw_override.get("g_c_scale", 1.0)
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, int | float):
            raise TypeError(
                f"g_c_scale for {cell_type!r} must be a positive finite number"
            )
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"g_c_scale for {cell_type!r} must be a positive finite number"
            )

        model = raw_override.get("model")
        if model is not None and model not in {
            "user_m2", "user_m3", "user_m4", "user_m5", "user_m7"
        }:
            raise ValueError(
                f"model for {cell_type!r} is {model!r}; expected "
                "user_m2/user_m3/user_m4/user_m5/user_m7"
            )
        if model == "user_m3" and cell_type != "CCK_Basket":
            raise ValueError("user_m3 is an opt-in model for CCK_Basket only")
        if model == "user_m4" and cell_type not in {
            "PV_Basket", "Bistratified", "O_LM"
        }:
            raise ValueError(
                "user_m4 is an opt-in model for PV_Basket/Bistratified/O_LM only"
            )
        if model == "user_m5" and cell_type not in {
            "PV_Basket", "Bistratified", "O_LM"
        }:
            raise ValueError(
                "user_m5 is an opt-in model for PV_Basket/Bistratified/O_LM only"
            )
        if model == "user_m7" and cell_type != "PV_Basket":
            raise ValueError("user_m7 is an opt-in model for PV_Basket only")

        parsed[cell_type] = AglifDendOverride(
            receive_domain=receive_domain,
            g_c_scale=scale,
            model=model,
        )
    return parsed


def aglif_receive_domain_overrides_provenance(
    overrides: Mapping[str, str],
) -> str:
    return json.dumps(dict(sorted(overrides.items())), sort_keys=True)


def aglif_gc_scale_overrides_provenance(overrides: Mapping[str, float]) -> str:
    return json.dumps(dict(sorted(overrides.items())), sort_keys=True)
