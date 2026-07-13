from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Final, Literal, cast

from .location_transfer_validation import unvalidated_transfer_rows
from ..params.receptors import receptor_prefix
from ..types import Afferent, NetworkSpec, Projection

TransferMode = Literal[
    "none",
    "inhibitory_distal",
    "inhibitory_dend",
    "all_dend",
]
_CANONICAL_TRANSFER_TABLE = "source_location_transfer_syndata120_budget_weighted.json"
_CANONICAL_TRANSFER_SHA256 = (
    "6153610d0460f474831d0175f218245cf18692c63f667fd2a58bee8b72338917"
)
_SOURCE_BUDGET_FIELD: Final[str] = "conductance_per_cell_nS"
_SOURCE_BUDGET_REL_TOLERANCE: Final[float] = 0.05


class IncompleteLocationTransferError(RuntimeError):
    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing: tuple[str, ...] = missing
        preview = ", ".join(self.missing[:8])
        suffix = "" if len(self.missing) <= 8 else f", ... (+{len(self.missing) - 8})"
        message = (
            "location-transfer table is incomplete; refusing implicit 1.0 fallback "
            + f"for {len(self.missing)} dendritic rows: {preview}{suffix}"
        )
        super().__init__(message)


class UnvalidatedLocationTransferError(RuntimeError):
    def __init__(self, rows: tuple[str, ...]) -> None:
        self.rows: tuple[str, ...] = rows
        preview = ", ".join(self.rows[:8])
        suffix = "" if len(self.rows) <= 8 else f", ... (+{len(self.rows) - 8})"
        message = (
            "location-transfer table is not final-validated; refusing "
            + f"{len(self.rows)} diagnostic/prototype rows: {preview}{suffix}"
        )
        super().__init__(message)


class IncompatibleLocationTransferBudgetError(RuntimeError):
    def __init__(
        self,
        *,
        row: str,
        table_budget_nS: float,
        spec_budget_nS: float,
        tolerance: float,
    ) -> None:
        self.row: str = row
        self.table_budget_nS: float = table_budget_nS
        self.spec_budget_nS: float = spec_budget_nS
        self.tolerance: float = tolerance
        rel_diff = _relative_difference(table_budget_nS, spec_budget_nS)
        message = (
            "source-budget metadata is incompatible with NetworkSpec for "
            f"{row}: table conductance_per_cell_nS={table_budget_nS:g}, "
            f"spec conductance_per_cell_nS={spec_budget_nS:g}, "
            f"relative_difference={rel_diff:g}, tolerance={tolerance:g}"
        )
        super().__init__(message)


def parse_transfer_mode(value: str) -> TransferMode:
    match value:
        case "none":
            return "none"
        case "inhibitory_distal":
            return "inhibitory_distal"
        case "inhibitory_dend":
            return "inhibitory_dend"
        case "all_dend":
            return "all_dend"
        case unreachable:
            raise ValueError(f"unsupported transfer mode: {unreachable}")


def _float_field(row: dict[str, object], field: str) -> float:
    value = row[field]
    if isinstance(value, bool):
        raise TypeError(f"transfer row field {field!r} must be numeric, got {value!r}")
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"transfer row field {field!r} must be numeric, got {value!r}")


def _relative_difference(table_budget_nS: float, spec_budget_nS: float) -> float:
    if spec_budget_nS == 0:
        if table_budget_nS == 0:
            return 0.0
        return float("inf")
    return abs(table_budget_nS - spec_budget_nS) / abs(spec_budget_nS)


def _assert_compatible_source_budget(
    row: dict[str, object],
    *,
    row_name: str,
    spec_budget_nS: float,
) -> None:
    if _SOURCE_BUDGET_FIELD not in row:
        return
    table_budget_nS = _float_field(row, _SOURCE_BUDGET_FIELD)
    if (
        _relative_difference(table_budget_nS, spec_budget_nS)
        <= _SOURCE_BUDGET_REL_TOLERANCE
    ):
        return
    raise IncompatibleLocationTransferBudgetError(
        row=row_name,
        table_budget_nS=table_budget_nS,
        spec_budget_nS=spec_budget_nS,
        tolerance=_SOURCE_BUDGET_REL_TOLERANCE,
    )


def _load_transfer_rows(
    transfer_table: Path,
) -> dict[tuple[str, str, str, str], dict[str, object]]:
    raw = cast(object, json.loads(transfer_table.read_text(encoding="utf-8")))
    if not isinstance(raw, list):
        raise ValueError(f"{transfer_table} must contain a list")
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for item_raw in cast(list[object], raw):
        if not isinstance(item_raw, dict):
            raise TypeError(f"invalid transfer row in {transfer_table}: {item_raw!r}")
        item = {
            str(key): value
            for key, value in cast(dict[object, object], item_raw).items()
        }
        if "transfer_scale" in item:
            _ = _float_field(item, "transfer_scale")
        if _SOURCE_BUDGET_FIELD in item:
            _ = _float_field(item, _SOURCE_BUDGET_FIELD)
        key = (
            str(item["pre"]),
            str(item["post"]),
            str(item["receptor"]),
            str(item["port"]),
        )
        if key in rows:
            pre, post, receptor, port = key
            raise ValueError(
                f"duplicate source-location transfer row: {pre}->{post}:{receptor}:{port}"
            )
        rows[key] = item
    return rows


def _row_scale(row: dict[str, object], mode: TransferMode) -> float:
    match mode:
        case "none":
            return 1.0
        case "all_dend":
            if row["aglif_compartment"] != "dend":
                return 1.0
            return _float_field(row, "transfer_scale")
        case "inhibitory_dend":
            if (
                row["aglif_compartment"] != "dend"
                or not str(row["receptor"]).startswith("GABA")
            ):
                return 1.0
            return _float_field(row, "transfer_scale")
        case "inhibitory_distal":
            if (
                row["aglif_compartment"] != "dend"
                or not str(row["receptor"]).startswith("GABA")
                or "dist" not in str(row["loc"])
            ):
                return 1.0
            return _float_field(row, "transfer_scale")


def apply_location_transfer(
    spec: NetworkSpec,
    mode: TransferMode,
    transfer_table: Path,
    *,
    allow_incomplete_transfer_for_prototype: bool = False,
) -> tuple[NetworkSpec, list[dict[str, object]], list[str]]:
    if mode == "none":
        return spec, [], []
    rows = _load_transfer_rows(transfer_table)
    if not allow_incomplete_transfer_for_prototype:
        unvalidated = unvalidated_transfer_rows(rows)
        if unvalidated:
            raise UnvalidatedLocationTransferError(unvalidated)
    applied: list[dict[str, object]] = []
    missing: list[str] = []
    projections: list[Projection] = []
    for projection in spec.projections:
        prefix = receptor_prefix(projection.receptor)
        key = (projection.pre, projection.post, prefix, projection.receptor)
        row = rows.get(key)
        if row is None:
            if projection.receptor.endswith("__dend"):
                missing.append(
                    f"{projection.pre}->{projection.post}:{projection.receptor}"
                )
            projections.append(projection)
            continue
        row_name = f"{projection.pre}->{projection.post}:{projection.receptor}"
        _assert_compatible_source_budget(
            row,
            row_name=row_name,
            spec_budget_nS=projection.total_conductance_per_cell(),
        )
        scale = _row_scale(row, mode)
        if scale != 1.0:
            applied.append(
                {
                    "pre": projection.pre,
                    "post": projection.post,
                    "receptor": projection.receptor,
                    "loc": row["loc"],
                    "scale": scale,
                }
            )
        projections.append(replace(projection, weight_nS=projection.weight_nS * scale))
    afferents: list[Afferent] = []
    for afferent in spec.afferents:
        prefix = receptor_prefix(afferent.receptor)
        pre = afferent.name.split("_to_", maxsplit=1)[0]
        key = (pre, afferent.post, prefix, afferent.receptor)
        row = rows.get(key)
        if row is None:
            if afferent.receptor.endswith("__dend"):
                missing.append(f"{pre}->{afferent.post}:{afferent.receptor}")
            afferents.append(afferent)
            continue
        row_name = f"{pre}->{afferent.post}:{afferent.receptor}"
        _assert_compatible_source_budget(
            row,
            row_name=row_name,
            spec_budget_nS=(
                afferent.weight_nS
                * afferent.synapses_per_cell
                * afferent.synapses_per_connection
            ),
        )
        scale = _row_scale(row, mode)
        if scale != 1.0:
            applied.append(
                {
                    "pre": pre,
                    "post": afferent.post,
                    "receptor": afferent.receptor,
                    "loc": row["loc"],
                    "scale": scale,
                }
            )
        afferents.append(replace(afferent, weight_nS=afferent.weight_nS * scale))
    unique_missing = sorted(set(missing))
    if unique_missing and not allow_incomplete_transfer_for_prototype:
        raise IncompleteLocationTransferError(tuple(unique_missing))
    updated = replace(
        spec,
        projections=projections,
        afferents=afferents,
        source_location_transfer_provenance=_transfer_provenance(
            mode=mode,
            transfer_table=transfer_table,
            applied_count=len(applied),
            missing_count=len(unique_missing),
            prototype_override=allow_incomplete_transfer_for_prototype,
        ),
        source_location_transfer_table=str(transfer_table),
    )
    return updated, applied, unique_missing


def _transfer_provenance(
    *,
    mode: TransferMode,
    transfer_table: Path,
    applied_count: int,
    missing_count: int,
    prototype_override: bool,
) -> str:
    digest = hashlib.sha256(transfer_table.read_bytes()).hexdigest()
    if prototype_override:
        parts = [
            "unvalidated-prototype-source-location-transfer",
            f"mode={mode}",
            f"table={transfer_table.name}",
            f"sha256={digest}",
            f"applied={applied_count}",
            "incomplete-prototype-override",
        ]
        if missing_count:
            parts.append(f"missing_rows={missing_count}")
        return ";".join(parts)
    marker = "source-location-transfer-m2-row-validation-passed"
    if (
        transfer_table.name != _CANONICAL_TRANSFER_TABLE
        or digest != _CANONICAL_TRANSFER_SHA256
    ):
        marker = "diagnostic-noncanonical-source-location-transfer"
    return f"{marker};mode={mode};table={transfer_table.name};sha256={digest};applied={applied_count}"
