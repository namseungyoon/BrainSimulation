from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

from ca1.analysis.receptor_compression_assignments import assignments_for_strategy
from ca1.analysis.receptor_compression_experiment_decoders import (
    CODE_BITS,
    MAX_CODEWORDS_PER_GROUP,
    DecodedAssignments,
    binary4_cdm_assignments,
    event_select2_mix_assignments,
)
from ca1.analysis.receptor_compression_inputs import build_compression_context
from ca1.analysis.receptor_compression_types import (
    CompressionContext,
    KernelKey,
    ReceptorCompressionInputError,
)
from ca1.analysis.receptor_compression_utils import (
    weighted_average,
)

ExperimentalStrategyName: TypeAlias = Literal[
    "binary4_cdm",
    "event_select2_mix",
]


@dataclass(frozen=True, slots=True)
class ExperimentalStrategyScore:
    strategy: ExperimentalStrategyName
    rank_objective: float
    utility_loss: float
    conductance_loss: float
    uniform_loss: float
    max_item_loss: float
    n_ports: int
    code_bits: int
    max_codewords_per_group: int
    effective_inputs_per_item: float
    decoder_parameter_count: int
    worst_original: str
    worst_decoded: str


@dataclass(frozen=True, slots=True)
class ExperimentalCompressionReport:
    variant: int
    n_original_items: int
    n_budget: int
    scores: tuple[ExperimentalStrategyScore, ...]


def rank_experimental_receptor_compression_strategies(
    variant: int = 120,
    n_budget: int = 20,
) -> ExperimentalCompressionReport:
    if n_budget != 20:
        raise ReceptorCompressionInputError("n_budget", "20", str(n_budget))
    context = build_compression_context(variant)
    utility_assignments = assignments_for_strategy(
        "utility_weighted_medoids",
        context,
        variant,
        n_budget,
    )
    scores = sorted(
        (
            _score_binary4_cdm(context),
            _score_event_select2_mix(utility_assignments, context),
        ),
        key=lambda score: (score.rank_objective, score.strategy),
    )
    return ExperimentalCompressionReport(
        variant=variant,
        n_original_items=len(context.items),
        n_budget=n_budget,
        scores=tuple(scores),
    )


def write_experimental_report(
    report: ExperimentalCompressionReport,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_write(json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    ))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        _ensure_write(handle.write(",".join(_CSV_HEADER) + "\n"))
        for score in report.scores:
            _ensure_write(handle.write(",".join(_score_csv_values(score)) + "\n"))


_CSV_HEADER: Final[tuple[str, ...]] = (
    "strategy",
    "rank_objective",
    "utility_loss",
    "conductance_loss",
    "uniform_loss",
    "max_item_loss",
    "n_ports",
    "code_bits",
    "max_codewords_per_group",
    "effective_inputs_per_item",
    "decoder_parameter_count",
    "worst_original",
    "worst_decoded",
)


def _score_csv_values(score: ExperimentalStrategyScore) -> tuple[str, ...]:
    return (
        score.strategy,
        str(score.rank_objective),
        str(score.utility_loss),
        str(score.conductance_loss),
        str(score.uniform_loss),
        str(score.max_item_loss),
        str(score.n_ports),
        str(score.code_bits),
        str(score.max_codewords_per_group),
        str(score.effective_inputs_per_item),
        str(score.decoder_parameter_count),
        score.worst_original,
        score.worst_decoded,
    )


def _ensure_write(bytes_written: int) -> None:
    if bytes_written < 1:
        raise ReceptorCompressionInputError("report", "writable output", "zero bytes")


def _score_binary4_cdm(context: CompressionContext) -> ExperimentalStrategyScore:
    decoded = binary4_cdm_assignments(context)
    return _experimental_score(
        "binary4_cdm",
        decoded,
        context,
    )


def _score_event_select2_mix(
    utility_assignments: dict[KernelKey, tuple[KernelKey, ...]],
    context: CompressionContext,
) -> ExperimentalStrategyScore:
    decoded = event_select2_mix_assignments(utility_assignments, context)
    return _experimental_score(
        "event_select2_mix",
        decoded,
        context,
    )


def _experimental_score(
    name: ExperimentalStrategyName,
    decoded: DecodedAssignments,
    context: CompressionContext,
) -> ExperimentalStrategyScore:
    assignments = decoded.assignments
    losses = decoded.losses
    utility_loss = weighted_average(losses, context.utility_weights)
    effective_inputs = weighted_average(
        {key: float(len(value)) for key, value in assignments.items()},
        context.utility_weights,
    )
    penalty = 1.0 + 0.2 * max(0.0, effective_inputs - 1.0)
    worst_key = max(losses, key=lambda key: losses[key])
    return ExperimentalStrategyScore(
        strategy=name,
        rank_objective=utility_loss * penalty,
        utility_loss=utility_loss,
        conductance_loss=weighted_average(losses, context.conductance_weights),
        uniform_loss=weighted_average(losses, context.uniform_weights),
        max_item_loss=losses[worst_key],
        n_ports=len({decoded for values in assignments.values() for decoded in values}),
        code_bits=CODE_BITS,
        max_codewords_per_group=MAX_CODEWORDS_PER_GROUP,
        effective_inputs_per_item=effective_inputs,
        decoder_parameter_count=decoded.decoder_parameter_count,
        worst_original=worst_key.label(),
        worst_decoded="+".join(key.label() for key in assignments[worst_key]),
    )
