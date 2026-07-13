from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from typing_extensions import override

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonDict: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SyndataNumericFieldError(TypeError):
    field: str
    actual: JsonValue

    @override
    def __str__(self) -> str:
        return f"syndata field {self.field!r} must be numeric, got {self.actual!r}"


@dataclass(frozen=True, slots=True)
class UnsupportedSyndataVariantError(ValueError):
    variant: int

    @override
    def __str__(self) -> str:
        return f"variant must be 120 or 137, got {self.variant!r}"


@dataclass(frozen=True, slots=True)
class MissingSyndataKineticsError(KeyError):
    variant: int
    component: str
    pre: str
    post: str

    @override
    def __str__(self) -> str:
        return (
            f"syndata_{self.variant} has no {self.component}-component "
            + f"kinetics for {self.pre}->{self.post}"
        )


def component_params(
    params: JsonDict,
    *,
    suffix: str,
) -> tuple[float, float, float] | None:
    e_key = f"e_rev{suffix}"
    rise_key = f"tau_rise{suffix}"
    decay_key = f"tau_decay{suffix}"
    if not all(key in params for key in (e_key, rise_key, decay_key)):
        return None
    return (
        as_float(params[e_key], e_key),
        as_float(params[rise_key], rise_key),
        as_float(params[decay_key], decay_key),
    )


def entry_compartment(entry: JsonDict) -> str:
    section = str(entry.get("section_list", "")).lower()
    if "soma" in section or "axon" in section:
        return "soma"
    return "dend"


def as_float(value: JsonValue, field: str) -> float:
    if isinstance(value, bool):
        raise SyndataNumericFieldError(field, value)
    if isinstance(value, str | int | float):
        return float(value)
    raise SyndataNumericFieldError(field, value)
