from __future__ import annotations

from typing import Final

from ca1.params.receptor_ports import PortKey, port_replacements, representative_port

from ca1.analysis.receptor_compression_types import (
    CompressionContext,
    KernelItem,
    KernelKey,
    StrategyName,
)
from ca1.analysis.receptor_compression_utils import (
    response_for_key,
    response_loss,
    weighted_sum,
)

_EPS: Final = 1e-12


def assignments_for_strategy(
    name: StrategyName,
    context: CompressionContext,
    variant: int,
    n_budget: int,
) -> dict[KernelKey, tuple[KernelKey, ...]]:
    if name == "current_safe20":
        replacements = port_replacements(variant, compartment_aware=True)
        return {
            item.key: (_safe20_key(item.key, replacements),)
            for item in context.items
        }
    if name == "uniform_medoids":
        return _medoid_assignments(context, context.uniform_weights, n_budget)
    if name == "conductance_weighted_medoids":
        return _medoid_assignments(context, context.conductance_weights, n_budget)
    if name == "utility_weighted_medoids" or name == "utility_sparse2":
        return _medoid_assignments(context, context.utility_weights, n_budget)


def _medoid_assignments(
    context: CompressionContext,
    weights: dict[KernelKey, float],
    n_budget: int,
) -> dict[KernelKey, tuple[KernelKey, ...]]:
    selected: set[KernelKey] = set()
    for group_items in _groups(context.items).values():
        selected.add(_best_group_medoid(group_items, weights, context))
    while len(selected) < n_budget:
        candidate = _best_next_medoid(context, weights, selected)
        if candidate is None:
            break
        selected.add(candidate)
    return {
        item.key: (_nearest_key(item.key, selected, context),)
        for item in context.items
    }


def _safe20_key(key: KernelKey, replacements: dict[PortKey, PortKey]) -> KernelKey:
    source = key.port_key()
    replaced = replacements[source] if source in replacements else source
    final_class, e_rev, tau_rise, tau_decay = representative_port(replaced, "budget_weighted")
    return KernelKey(final_class, e_rev, tau_rise, tau_decay, key.compartment)


def _best_group_medoid(
    items: tuple[KernelItem, ...],
    weights: dict[KernelKey, float],
    context: CompressionContext,
) -> KernelKey:
    return min(
        (item.key for item in items),
        key=lambda candidate: _assignment_loss(
            tuple(item.key for item in items), (candidate,), weights, context
        ),
    )


def _best_next_medoid(
    context: CompressionContext,
    weights: dict[KernelKey, float],
    selected: set[KernelKey],
) -> KernelKey | None:
    keys = tuple(item.key for item in context.items)
    baseline = _assignment_loss(keys, tuple(selected), weights, context)
    gains = [
        (
            baseline - _assignment_loss(keys, (*selected, item.key), weights, context),
            item.key,
        )
        for item in context.items
        if item.key not in selected
    ]
    positive = [gain for gain in gains if gain[0] > _EPS]
    if not positive:
        return None
    return max(positive, key=lambda value: (value[0], value[1].label()))[1]


def _assignment_loss(
    keys: tuple[KernelKey, ...],
    selected: tuple[KernelKey, ...],
    weights: dict[KernelKey, float],
    context: CompressionContext,
) -> float:
    losses = {
        key: response_loss(
            context.responses[key],
            response_for_key(_nearest_key(key, set(selected), context), context),
        )
        for key in keys
    }
    return weighted_sum(losses, weights)


def _nearest_key(
    key: KernelKey,
    candidates: set[KernelKey],
    context: CompressionContext,
) -> KernelKey:
    same_group = [
        candidate for candidate in candidates
        if candidate.merge_group() == key.merge_group()
    ]
    return min(
        same_group,
        key=lambda candidate: response_loss(
            context.responses[key],
            response_for_key(candidate, context),
        ),
    )


def _groups(items: tuple[KernelItem, ...]) -> dict[tuple[str, float, str], tuple[KernelItem, ...]]:
    grouped: dict[tuple[str, float, str], list[KernelItem]] = {}
    for item in items:
        grouped.setdefault(item.key.merge_group(), []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}
