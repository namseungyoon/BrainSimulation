from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final, cast

from ca1.analysis.location_transfer_validation import unvalidated_transfer_rows
from ca1.params.groundtruth import CELL_TEMPLATES
from ca1.params.json_io import JsonValue, load_json_mapping, mapping_field
from ca1.types import NetworkSpec

_CONNDATA_RE: Final = re.compile(r"conndata(?P<index>\d+)")
_SYNDATA_RE: Final = re.compile(r"syndata(?P<variant>\d+)")
_VALIDATION_FIELD: Final = "validation"
_SOURCE_TABLE_FIELD: Final = "source_location_table"
_REFIT_METHOD: Final = "conndata430-weighted-morphology-target-refit"
_ROW_VALIDATION_METHOD: Final = "user_m2-row-level-source-location-response-fidelity"
_PEAK_RATIO_TOLERANCE: Final[float] = 0.10
_AREA_RATIO_TOLERANCE: Final[float] = 0.15


def _optional_string(record: dict[str, JsonValue], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise TypeError(f"dendritic-transfer provenance field {key!r} must be a string")


def _source_marker(note: str | None) -> str | None:
    if note is None:
        return None
    conndata = _CONNDATA_RE.search(note)
    syndata = _SYNDATA_RE.search(note)
    parts: list[str] = []
    if conndata is not None:
        parts.append(f"conndata{conndata.group('index')}")
    if syndata is not None:
        parts.append(f"syndata{syndata.group('variant')}")
    if not parts:
        return None
    return "/".join(parts)


def _spec_marker(spec: NetworkSpec) -> str:
    conndata = (
        "conndata-unspecified"
        if spec.conndata_index is None
        else f"conndata{spec.conndata_index}"
    )
    return (
        f"{conndata};count_mode={spec.conndata_count_mode};"
        f"cellnumbers={spec.cellnumbers_index}"
    )


def _validation_mapping(record: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    value = record.get(_VALIDATION_FIELD)
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    raise TypeError("dendritic-transfer validation must be a mapping")


def _source_table_path(validation: dict[str, JsonValue]) -> Path | None:
    value = validation.get(_SOURCE_TABLE_FIELD)
    if value is None:
        return None
    if isinstance(value, str):
        return Path(value)
    raise TypeError("dendritic-transfer validation source_location_table must be a string")


def _numeric_validation_field(
    validation: dict[str, JsonValue],
    field: str,
) -> float | None:
    value = validation.get(field)
    if isinstance(value, bool):
        return None
    if isinstance(value, str | int | float):
        return float(value)
    return None


def _load_transfer_row_list(path: Path) -> list[dict[str, object]]:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, list):
        raise TypeError(f"source-location transfer table must be a list: {path}")
    rows: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"source-location transfer row must be a mapping: {item!r}")
        rows.append(dict(cast(dict[str, object], item)))
    return rows


def _source_budget_validation_blocker(
    validation: dict[str, JsonValue],
    spec: NetworkSpec,
) -> str | None:
    conndata = validation.get("source_budget_conndata_index")
    count_mode = validation.get("source_budget_count_mode")
    cellnumbers = validation.get("source_budget_cellnumbers_index")
    if (
        conndata == spec.conndata_index
        and count_mode == spec.conndata_count_mode
        and cellnumbers == spec.cellnumbers_index
    ):
        return None
    return (
        "source-domain-refit-source-budget-mismatch;"
        f"fit_conndata={conndata};fit_count_mode={count_mode};"
        f"fit_cellnumbers={cellnumbers};spec={_spec_marker(spec)}"
    )


def _response_validation_blocker(
    validation: dict[str, JsonValue],
) -> str | None:
    target_peak = _numeric_validation_field(validation, "target_peak_ratio")
    response_peak = _numeric_validation_field(validation, "response_peak_ratio")
    target_area = _numeric_validation_field(validation, "target_area_ratio")
    response_area = _numeric_validation_field(validation, "response_area_ratio")
    if (
        target_peak is None
        or response_peak is None
        or target_area is None
        or response_area is None
    ):
        return "source-domain-refit-validation-incomplete"
    peak_error = abs(response_peak - target_peak)
    area_error = abs(response_area - target_area)
    if (
        peak_error <= _PEAK_RATIO_TOLERANCE
        and area_error <= _AREA_RATIO_TOLERANCE
    ):
        return None
    return (
        "source-domain-refit-out-of-tolerance;"
        f"peak_error={peak_error:g};peak_tolerance={_PEAK_RATIO_TOLERANCE:g};"
        f"area_error={area_error:g};area_tolerance={_AREA_RATIO_TOLERANCE:g}"
    )


def _source_table_validation_record(
    validation: dict[str, JsonValue] | None,
    spec: NetworkSpec,
    note: str | None,
) -> str | None:
    if validation is None or validation.get("passed") is not True:
        return None
    method = validation.get("method")
    if method == _ROW_VALIDATION_METHOD:
        if note is None or "carried-forward" not in note:
            return None
        if _source_marker(note) != f"conndata{spec.conndata_index}":
            return None
        table_record = _validated_source_location_table_record(validation, spec)
        if table_record is None:
            return None
        return table_record + ";cell_g_c=carried-forward"
    if method != _REFIT_METHOD:
        return None
    budget_blocker = _source_budget_validation_blocker(validation, spec)
    if budget_blocker is not None:
        return budget_blocker
    response_blocker = _response_validation_blocker(validation)
    if response_blocker is not None:
        return response_blocker
    return _validated_source_location_table_record(validation, spec)


def _validated_source_location_table_record(
    validation: dict[str, JsonValue],
    spec: NetworkSpec,
) -> str | None:
    table_path = _source_table_path(validation)
    if table_path is None or not table_path.exists():
        return "source-domain-refit-validation-incomplete"
    rows = _load_transfer_row_list(table_path)
    keyed_rows = {
        (
            str(row["pre"]),
            str(row["post"]),
            str(row["receptor"]),
            str(row["port"]),
        ): row
        for row in rows
    }
    if unvalidated_transfer_rows(keyed_rows):
        return "source-domain-refit-validation-incomplete"
    conndata_values = {
        row.get("source_budget_conndata_index")
        for row in rows
        if "source_budget_conndata_index" in row
    }
    count_mode_values = {
        row.get("source_budget_count_mode")
        for row in rows
        if "source_budget_count_mode" in row
    }
    cellnumbers_values = {
        row.get("source_budget_cellnumbers_index")
        for row in rows
        if "source_budget_cellnumbers_index" in row
    }
    if conndata_values != {spec.conndata_index}:
        return None
    if count_mode_values != {spec.conndata_count_mode}:
        return None
    if cellnumbers_values != {spec.cellnumbers_index}:
        return None
    method = validation.get("method", "unknown")
    return (
        "source-location-transfer-table-validation-passed;"
        f"conndata{spec.conndata_index};"
        f"count_mode={spec.conndata_count_mode};"
        f"cellnumbers={spec.cellnumbers_index};"
        f"rows={len(rows)};"
        f"method={method}"
    )


def dendritic_transfer_source_provenance(
    *,
    path: Path,
    expected_cells: set[str],
    spec: NetworkSpec,
) -> dict[str, str]:
    raw = load_json_mapping(path, context="dendritic transfer source provenance")
    missing = expected_cells - set(raw)
    if missing:
        return {
            f"dendritic_transfer_source.{cell_type}": "source-domain-unspecified"
            for cell_type in sorted(missing)
        }

    spec_marker = _spec_marker(spec)
    expected_conndata = (
        None if spec.conndata_index is None else f"conndata{spec.conndata_index}"
    )
    provenance: dict[str, str] = {}
    for cell_type in sorted(expected_cells & set(CELL_TEMPLATES)):
        record = mapping_field(
            raw,
            cell_type,
            context="dendritic transfer source provenance",
        )
        note = _optional_string(record, "note")
        validated = _source_table_validation_record(
            _validation_mapping(record),
            spec,
            note,
        )
        if validated is not None:
            provenance[f"dendritic_transfer_source.{cell_type}"] = validated
            continue
        marker = _source_marker(note)
        if marker is None:
            provenance[f"dendritic_transfer_source.{cell_type}"] = (
                f"source-domain-unspecified;spec={spec_marker}"
            )
            continue
        if expected_conndata is not None and expected_conndata not in marker:
            provenance[f"dendritic_transfer_source.{cell_type}"] = (
                f"source-domain-mismatch;fit={marker};spec={spec_marker}"
            )
    return provenance
