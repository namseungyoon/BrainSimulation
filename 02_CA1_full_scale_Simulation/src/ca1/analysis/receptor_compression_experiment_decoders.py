from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ca1.analysis.receptor_compression_types import (
    CompressionContext,
    FloatArray,
    KernelItem,
    KernelKey,
)
from ca1.analysis.receptor_compression_utils import (
    response_for_key,
    response_loss,
    weighted_sum,
)

CODE_BITS: Final = 4
MAX_CODEWORDS_PER_GROUP: Final = 2**CODE_BITS
_EPS: Final = 1.0e-12


@dataclass(frozen=True, slots=True)
class DecodedAssignments:
    assignments: dict[KernelKey, tuple[KernelKey, ...]]
    losses: dict[KernelKey, float]
    decoder_parameter_count: int


def binary4_cdm_assignments(context: CompressionContext) -> DecodedAssignments:
    basis_by_group = _binary_basis_by_group(context)
    assignments: dict[KernelKey, tuple[KernelKey, ...]] = {}
    losses: dict[KernelKey, float] = {}
    for item in context.items:
        decoded, loss = _best_binary_decode(
            item.key,
            basis_by_group[item.key.merge_group()],
            context,
        )
        assignments[item.key] = decoded
        losses[item.key] = loss
    return DecodedAssignments(
        assignments=assignments,
        losses=losses,
        decoder_parameter_count=2 * sum(len(group) for group in basis_by_group.values()),
    )


def event_select2_mix_assignments(
    utility_assignments: dict[KernelKey, tuple[KernelKey, ...]],
    context: CompressionContext,
) -> DecodedAssignments:
    ports = tuple(sorted({decoded[0] for decoded in utility_assignments.values()}))
    assignments: dict[KernelKey, tuple[KernelKey, ...]] = {}
    losses: dict[KernelKey, float] = {}
    for item in context.items:
        candidates = tuple(
            port for port in ports if port.merge_group() == item.key.merge_group()
        )
        decoded, loss = _best_select2_decode(item.key, candidates, context)
        assignments[item.key] = decoded
        losses[item.key] = loss
    return DecodedAssignments(
        assignments=assignments,
        losses=losses,
        decoder_parameter_count=2 * len(ports) + 2 * len(context.items),
    )


def _binary_basis_by_group(
    context: CompressionContext,
) -> dict[tuple[str, float, str], tuple[KernelKey, ...]]:
    result: dict[tuple[str, float, str], tuple[KernelKey, ...]] = {}
    for group, items in _groups(context.items).items():
        selected: tuple[KernelKey, ...] = ()
        n_basis = min(CODE_BITS, len(items))
        while len(selected) < n_basis:
            selected = (*selected, _best_next_binary_basis(items, selected, context))
        result[group] = selected
    return result


def _best_next_binary_basis(
    items: tuple[KernelItem, ...],
    selected: tuple[KernelKey, ...],
    context: CompressionContext,
) -> KernelKey:
    candidates = tuple(item.key for item in items if item.key not in selected)
    return min(
        candidates,
        key=lambda candidate: _binary_group_loss(items, (*selected, candidate), context),
    )


def _binary_group_loss(
    items: tuple[KernelItem, ...],
    basis: tuple[KernelKey, ...],
    context: CompressionContext,
) -> float:
    losses = {
        item.key: _best_binary_decode(item.key, basis, context)[1]
        for item in items
    }
    return weighted_sum(losses, context.utility_weights)


def _best_binary_decode(
    key: KernelKey,
    basis: tuple[KernelKey, ...],
    context: CompressionContext,
) -> tuple[tuple[KernelKey, ...], float]:
    target = context.responses[key]
    best = (basis[0],)
    best_loss = response_loss(target, response_for_key(best[0], context))
    for mask in range(1, 1 << len(basis)):
        decoded = tuple(
            basis[index]
            for index in range(len(basis))
            if mask & (1 << index)
        )
        estimate = _normalized_sum(decoded, context)
        loss = response_loss(target, estimate)
        if loss < best_loss:
            best = decoded
            best_loss = loss
    return best, best_loss


def _best_select2_decode(
    key: KernelKey,
    candidates: tuple[KernelKey, ...],
    context: CompressionContext,
) -> tuple[tuple[KernelKey, ...], float]:
    target = context.responses[key]
    best = (_nearest_key(key, candidates, context),)
    best_loss = response_loss(target, response_for_key(best[0], context))
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            decoded, estimate = _select2_estimate(target, left, right, context)
            loss = response_loss(target, estimate)
            if loss < best_loss:
                best = decoded
                best_loss = loss
    return best, best_loss


def _select2_estimate(
    target: FloatArray,
    left: KernelKey,
    right: KernelKey,
    context: CompressionContext,
) -> tuple[tuple[KernelKey, ...], FloatArray]:
    basis = np.column_stack((
        response_for_key(left, context),
        response_for_key(right, context),
    ))
    coeffs = np.asarray(np.linalg.lstsq(basis, target, rcond=None)[0], dtype=np.float64)
    clipped = np.maximum(coeffs, 0.0)
    return (left, right), basis @ clipped


def _normalized_sum(
    keys: tuple[KernelKey, ...],
    context: CompressionContext,
) -> FloatArray:
    estimate = np.zeros(response_for_key(keys[0], context).shape, dtype=np.float64)
    for key in keys:
        estimate = estimate + response_for_key(key, context)
    return estimate / float(len(keys))


def _nearest_key(
    key: KernelKey,
    candidates: tuple[KernelKey, ...],
    context: CompressionContext,
) -> KernelKey:
    return min(
        candidates,
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
