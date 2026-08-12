from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ca1.analysis.receptor_compression_assignments import assignments_for_strategy
from ca1.analysis.receptor_compression_inputs import build_compression_context
from ca1.analysis.receptor_compression_types import (
    CompressionContext,
    CompressionReport,
    KernelKey,
    ReceptorCompressionInputError,
    StrategyName,
    StrategyScore,
)
from ca1.analysis.receptor_compression_utils import (
    response_for_key,
    response_loss,
    weighted_average,
)


def rank_receptor_compression_strategies(
    variant: int = 120,
    n_budget: int = 20,
) -> CompressionReport:
    if n_budget != 20:
        raise ReceptorCompressionInputError("n_budget", "20", str(n_budget))
    context = build_compression_context(variant)
    hard_scores = [
        _score_hard_strategy(
            name,
            assignments_for_strategy(name, context, variant, n_budget),
            context,
        )
        for name in (
            "current_safe20",
            "uniform_medoids",
            "conductance_weighted_medoids",
            "utility_weighted_medoids",
        )
    ]
    sparse_score = _score_sparse_strategy(
        "utility_sparse2",
        assignments_for_strategy("utility_weighted_medoids", context, variant, n_budget),
        context,
    )
    scores = sorted(
        (*hard_scores, sparse_score),
        key=lambda score: (score.rank_objective, score.utility_loss, score.strategy),
    )
    return CompressionReport(
        variant=variant,
        n_original_items=len(context.items),
        n_budget=n_budget,
        scores=tuple(scores),
    )


def write_report(report: CompressionReport, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_write(json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    ))
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        _ensure_write(fh.write(",".join(_CSV_HEADER) + "\n"))
        for score in report.scores:
            _ensure_write(fh.write(",".join(_score_csv_values(score)) + "\n"))


def _ensure_write(bytes_written: int) -> None:
    if bytes_written < 1:
        raise ReceptorCompressionInputError("report", "writable output", "zero bytes")


_CSV_HEADER: tuple[str, ...] = (
    "strategy",
    "rank_objective",
    "utility_loss",
    "conductance_loss",
    "uniform_loss",
    "max_item_loss",
    "n_ports",
    "effective_ports_per_item",
    "worst_original",
    "worst_decoded",
)


def _score_csv_values(score: StrategyScore) -> tuple[str, ...]:
    return (
        score.strategy,
        str(score.rank_objective),
        str(score.utility_loss),
        str(score.conductance_loss),
        str(score.uniform_loss),
        str(score.max_item_loss),
        str(score.n_ports),
        str(score.effective_ports_per_item),
        score.worst_original,
        score.worst_decoded,
    )


def _score_hard_strategy(
    name: StrategyName,
    assignments: dict[KernelKey, tuple[KernelKey, ...]],
    context: CompressionContext,
) -> StrategyScore:
    losses = {
        key: response_loss(
            context.responses[key],
            response_for_key(decoded[0], context),
        )
        for key, decoded in assignments.items()
    }
    return _strategy_score(name, assignments, losses, context)


def _score_sparse_strategy(
    name: StrategyName,
    assignments: dict[KernelKey, tuple[KernelKey, ...]],
    context: CompressionContext,
) -> StrategyScore:
    ports = tuple(sorted({decoded[0] for decoded in assignments.values()}))
    sparse: dict[KernelKey, tuple[KernelKey, ...]] = {}
    losses: dict[KernelKey, float] = {}
    for item in context.items:
        candidates = tuple(port for port in ports if port.merge_group() == item.key.merge_group())
        decoded, loss = _best_sparse_decode(item.key, candidates, context)
        sparse[item.key] = decoded
        losses[item.key] = loss
    return _strategy_score(name, sparse, losses, context)


def _strategy_score(
    name: StrategyName,
    assignments: dict[KernelKey, tuple[KernelKey, ...]],
    losses: dict[KernelKey, float],
    context: CompressionContext,
) -> StrategyScore:
    utility_loss = weighted_average(losses, context.utility_weights)
    effective_ports = weighted_average(
        {key: float(len(value)) for key, value in assignments.items()},
        context.utility_weights,
    )
    penalty = 1.0 + 0.2 * max(0.0, effective_ports - 1.0)
    worst_key = max(losses, key=lambda key: losses[key])
    return StrategyScore(
        strategy=name,
        rank_objective=utility_loss * penalty,
        utility_loss=utility_loss,
        conductance_loss=weighted_average(losses, context.conductance_weights),
        uniform_loss=weighted_average(losses, context.uniform_weights),
        max_item_loss=losses[worst_key],
        n_ports=len({decoded for values in assignments.values() for decoded in values}),
        effective_ports_per_item=effective_ports,
        worst_original=worst_key.label(),
        worst_decoded="+".join(key.label() for key in assignments[worst_key]),
    )


def _best_sparse_decode(
    key: KernelKey,
    candidates: tuple[KernelKey, ...],
    context: CompressionContext,
) -> tuple[tuple[KernelKey, ...], float]:
    target = context.responses[key]
    best = (_nearest_sparse_key(key, candidates, context),)
    best_loss = response_loss(target, response_for_key(best[0], context))
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            basis = np.column_stack((
                response_for_key(left, context),
                response_for_key(right, context),
            ))
            coeffs, *_ = np.linalg.lstsq(basis, target, rcond=None)
            estimate = basis @ np.maximum(coeffs, 0.0)
            loss = response_loss(target, estimate)
            if loss < best_loss:
                best = (left, right)
                best_loss = loss
    return best, best_loss


def _nearest_sparse_key(
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
