from __future__ import annotations

import json
from pathlib import Path
from typing import Final, cast

from ca1.extract.connectivity import canonical_name
from ca1.params.receptors import receptor_class_for_pre
from ca1.params.synapses import load_afferents, load_projections

from ca1.analysis.receptor_compression_types import (
    ComponentName,
    ComponentRow,
    CompressionContext,
    JsonValue,
    KernelItem,
    KernelKey,
    ReceptorCompressionInputError,
)
from ca1.analysis.receptor_compression_utils import kernel_response

_PARAMS_DIR: Final = Path(__file__).resolve().parents[1] / "params"
_V_OPERATING_MV: Final = -55.0
_TARGET_UTILITY: Final[dict[str, float]] = {
    "Pyramidal": 4.0,
    "PV_Basket": 2.5,
    "O_LM": 2.5,
    "CCK_Basket": 2.0,
    "SCA": 2.0,
}


def build_compression_context(variant: int) -> CompressionContext:
    rows = _load_component_rows(variant)
    pair_counts = _pair_component_counts(rows)
    conductance = _conductance_budgets(pair_counts, variant)
    items: dict[KernelKey, tuple[float, float, int]] = {}
    for row in rows:
        row_budget = conductance.get((row.pre, row.post, row.component), 0.0)
        row_budget /= pair_counts[(row.pre, row.post, row.component)]
        utility = (
            row_budget
            * abs(row.key.e_rev - _V_OPERATING_MV)
            * _TARGET_UTILITY.get(row.post, 1.0)
        )
        old_budget, old_utility, old_count = items.get(row.key, (0.0, 0.0, 0))
        items[row.key] = (
            old_budget + row_budget,
            old_utility + utility,
            old_count + 1,
        )
    kernel_items = tuple(
        KernelItem(key, conductance_budget, utility_budget, row_count)
        for key, (conductance_budget, utility_budget, row_count) in sorted(items.items())
    )
    return CompressionContext(
        items=kernel_items,
        responses={item.key: kernel_response(item.key) for item in kernel_items},
        uniform_weights={item.key: 1.0 for item in kernel_items},
        conductance_weights={item.key: item.conductance_budget for item in kernel_items},
        utility_weights={item.key: item.utility_budget for item in kernel_items},
    )


def _load_component_rows(variant: int) -> tuple[ComponentRow, ...]:
    raw = cast(
        JsonValue,
        json.loads((_PARAMS_DIR / f"syndata_{variant}.json").read_text(encoding="utf-8")),
    )
    if not isinstance(raw, dict):
        raise ReceptorCompressionInputError("syndata", "a mapping", type(raw).__name__)
    data = cast(dict[str, JsonValue], raw)
    entries = data["entries"]
    if not isinstance(entries, list):
        raise ReceptorCompressionInputError("entries", "a list", type(entries).__name__)
    rows: list[ComponentRow] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ReceptorCompressionInputError(
                "entries[]", "a mapping", type(raw_entry).__name__
            )
        rows.extend(_entry_rows(raw_entry))
    return tuple(rows)


def _entry_rows(entry: dict[str, JsonValue]) -> tuple[ComponentRow, ...]:
    post = canonical_name(_str_field(entry, "postsynaptic"))
    pre = canonical_name(_str_field(entry, "presynaptic"))
    params = _mapping_field(entry, "parameters")
    compartment = _entry_compartment(_str_field(entry, "section_list"))
    rows: list[ComponentRow] = []
    primary = _component_key(params, "", receptor_class_for_pre(pre), compartment)
    if primary is None:
        primary = _component_key(params, "_A", receptor_class_for_pre(pre), compartment)
    if primary is not None:
        rows.append(ComponentRow(post=post, pre=pre, component="A", key=primary))
    secondary = _component_key(params, "_B", "GABA_B", compartment)
    if secondary is not None:
        rows.append(ComponentRow(post=post, pre=pre, component="B", key=secondary))
    return tuple(rows)


def _component_key(
    params: dict[str, JsonValue],
    suffix: str,
    receptor: str,
    compartment: str,
) -> KernelKey | None:
    required = (f"e_rev{suffix}", f"tau_rise{suffix}", f"tau_decay{suffix}")
    if not all(key in params for key in required):
        return None
    return KernelKey(
        receptor=receptor,
        e_rev=_float_field(params, required[0]),
        tau_rise=_float_field(params, required[1]),
        tau_decay=_float_field(params, required[2]),
        compartment=compartment,
    )


def _conductance_budgets(
    pair_counts: dict[tuple[str, str, ComponentName], int],
    variant: int,
) -> dict[tuple[str, str, ComponentName], float]:
    budgets: dict[tuple[str, str, ComponentName], float] = {}
    projections = load_projections(
        conndata_index=430,
        conndata_count_mode="per_cell",
        synapse_variant=variant,
    )
    for projection in projections:
        component: ComponentName = "B" if projection.receptor.startswith("GABA_B") else "A"
        key = (projection.pre, projection.post, component)
        if key in pair_counts:
            budgets[key] = budgets.get(key, 0.0) + projection.total_conductance_per_cell()
    afferents = load_afferents(
        conndata_index=430,
        conndata_count_mode="per_cell",
        synapse_variant=variant,
    )
    for afferent in afferents:
        pre = afferent.name.split("_to_", maxsplit=1)[0]
        key = (pre, afferent.post, "A")
        if key in pair_counts:
            budgets[key] = budgets.get(key, 0.0) + (
                afferent.synapses_per_cell * afferent.weight_nS
            )
    return budgets


def _pair_component_counts(
    rows: tuple[ComponentRow, ...],
) -> dict[tuple[str, str, ComponentName], int]:
    counts: dict[tuple[str, str, ComponentName], int] = {}
    for row in rows:
        key = (row.pre, row.post, row.component)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _entry_compartment(section_list: str) -> str:
    lowered = section_list.lower()
    return "soma" if "soma" in lowered or "axon" in lowered else "dend"


def _str_field(mapping: dict[str, JsonValue], field: str) -> str:
    value = mapping[field]
    if isinstance(value, str):
        return value
    raise ReceptorCompressionInputError(field, "a string", type(value).__name__)


def _float_field(mapping: dict[str, JsonValue], field: str) -> float:
    value = mapping[field]
    if isinstance(value, bool):
        raise ReceptorCompressionInputError(field, "numeric", type(value).__name__)
    if isinstance(value, int | float | str):
        return float(value)
    raise ReceptorCompressionInputError(field, "numeric", type(value).__name__)


def _mapping_field(mapping: dict[str, JsonValue], field: str) -> dict[str, JsonValue]:
    value = mapping[field]
    if isinstance(value, dict):
        return value
    raise ReceptorCompressionInputError(field, "a mapping", type(value).__name__)
