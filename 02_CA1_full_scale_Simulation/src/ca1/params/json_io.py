from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final, NoReturn, TypeAlias, cast

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
_BOOL_IS_NUMERIC_ERROR: Final[str] = "must be numeric, not boolean"


def load_json_mapping(path: Path, *, context: str) -> dict[str, JsonValue]:
    raw = cast(
        JsonValue,
        json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        ),
    )
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"{context} must be a mapping: {path}")


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(
        f"non-standard JSON constant {value!r}; use null or a finite number"
    )


def mapping_field(
    record: Mapping[str, JsonValue],
    key: str,
    *,
    context: str,
) -> dict[str, JsonValue]:
    value = record[key]
    if isinstance(value, dict):
        return value
    raise TypeError(f"{context} field {key!r} must be a mapping")


def numeric_field(
    record: Mapping[str, JsonValue],
    key: str,
    *,
    context: str,
) -> float:
    return _numeric_value(record[key], key=key, context=context)


def _numeric_value(value: JsonValue, *, key: str, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} field {key!r} {_BOOL_IS_NUMERIC_ERROR}")
    if isinstance(value, str | int | float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{context} field {key!r} must be finite")
        return number
    raise TypeError(f"{context} field {key!r} must be numeric")


def required_string_field(
    record: Mapping[str, JsonValue],
    key: str,
    *,
    context: str,
) -> str:
    if key not in record:
        raise ValueError(f"{context} field {key!r} is required")
    value = record[key]
    if isinstance(value, str):
        return value
    raise TypeError(f"{context} field {key!r} must be a string")


def reject_malformed_validation_passed(
    record: Mapping[str, JsonValue],
    *,
    context: str,
) -> None:
    validation = record.get("validation")
    if validation is None:
        return
    if not isinstance(validation, dict):
        raise TypeError(f"{context} field 'validation' must be a mapping")
    passed = validation.get("passed")
    if passed is False:
        raise ValueError(f"{context} validation failed")
    if passed is not True:
        raise ValueError(f"{context} validation.passed must be true")
    _reject_nonfinite_json_values(validation, context=context, path="validation")


def _reject_nonfinite_json_values(
    value: JsonValue,
    *,
    context: str,
    path: str,
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} {path} must not contain non-finite numbers")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_json_values(
                child,
                context=context,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_json_values(
                child,
                context=context,
                path=f"{path}[{index}]",
            )
