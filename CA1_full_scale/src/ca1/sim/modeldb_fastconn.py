from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

from ca1.sim.modeldb_interval_budget import (
    contiguous_index_intervals,
    split_indegree_over_intervals,
    validated_repaired_intervals,
)

LONGITUDINAL_LENGTH_UM: Final = 4_000.0
_FASTCONN_STEPS: Final = 5


@dataclass(frozen=True, slots=True)
class FastconnAxonDistribution:
    """Gaussian parameters read from ``cells/axondists/dist_*.hoc``."""

    a: float
    b_um: float
    c_um: float


_AXON_DISTRIBUTIONS: Final[dict[str, FastconnAxonDistribution]] = {
    "Axo": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "Bistratified": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "CA3": FastconnAxonDistribution(1.0, 0.0, 2_000.0),
    "CCK_Basket": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "ECIII": FastconnAxonDistribution(1.0, 0.0, 1_000.0),
    "Ivy": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "Neurogliaform": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "O_LM": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "PV_Basket": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "Pyramidal": FastconnAxonDistribution(1.0, 0.0, 250.0),
    "SCA": FastconnAxonDistribution(1.0, 0.0, 250.0),
}


@dataclass(frozen=True, slots=True)
class FastconnSourceInterval:
    source_start: int
    source_stop: int
    indegree: int


def fastconn_source_intervals(
    *,
    pre_type: str,
    source_ranges: tuple[tuple[int, int], ...],
    source_total: int,
    target_center_um: float,
    requested_indegree: int,
) -> tuple[FastconnSourceInterval, ...]:
    """Return source intervals using ModelDB fastconn's five Gaussian bins."""
    extent_um = fastconn_extent_um(pre_type)
    step_indegrees = _fastconn_step_indegrees(
        requested_indegree,
        extent_um=extent_um,
        c_um=_axon_distribution_c_um(pre_type),
    )
    intervals = _source_ring_intervals(
        source_ranges=source_ranges,
        source_total=source_total,
        target_center_um=target_center_um,
        extent_um=extent_um,
        step_indegrees=step_indegrees,
        requested_indegree=requested_indegree,
    )
    return tuple(
        FastconnSourceInterval(
            source_start=start,
            source_stop=stop,
            indegree=indegree,
        )
        for start, stop, indegree in intervals
    )


def partition_ranges(count: int, bins: int) -> tuple[tuple[int, int], ...]:
    base = count // bins
    remainder = count % bins
    start = 0
    ranges: list[tuple[int, int]] = []
    for idx in range(bins):
        stop = start + base + (1 if idx < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def range_center_um(start: int, stop: int, total_count: int) -> float:
    midpoint = (float(start) + float(stop)) / 2.0
    return midpoint / float(total_count) * LONGITUDINAL_LENGTH_UM


def fastconn_extent_um(pre_type: str) -> float:
    return 4.0 * _axon_distribution_c_um(pre_type)


def fastconn_axon_distribution(pre_type: str) -> FastconnAxonDistribution:
    try:
        return _AXON_DISTRIBUTIONS[pre_type]
    except KeyError as exc:
        raise ValueError(f"unknown fastconn pre_type {pre_type!r}") from exc


def _axon_distribution_c_um(pre_type: str) -> float:
    return fastconn_axon_distribution(pre_type).c_um


def _fastconn_step_indegrees(
    requested_indegree: int,
    *,
    extent_um: float,
    c_um: float,
    steps: int = _FASTCONN_STEPS,
) -> tuple[int, ...]:
    _require_positive(steps, "steps")
    weights = tuple(
        math.exp(-((extent_um * float(step + 1) / float(steps)) / c_um) ** 2)
        for step in range(steps)
    )
    total = sum(weights)
    raw = [weight / total * float(requested_indegree) for weight in weights]
    base = [int(math.floor(value)) for value in raw]
    remaining = requested_indegree - sum(base)
    order = sorted(
        range(steps),
        key=lambda idx: raw[idx] - float(base[idx]),
        reverse=True,
    )
    for idx in order[:remaining]:
        base[idx] += 1
    return tuple(base)


def _source_ring_intervals(
    *,
    source_ranges: tuple[tuple[int, int], ...],
    source_total: int,
    target_center_um: float,
    extent_um: float,
    step_indegrees: tuple[int, ...],
    requested_indegree: int,
) -> tuple[tuple[int, int, int], ...]:
    centers = tuple(
        range_center_um(start, stop, source_total)
        for start, stop in source_ranges
    )
    intervals: list[tuple[int, int, int]] = []
    previous_um = 0.0
    steps = len(step_indegrees)
    for step, step_indegree in enumerate(step_indegrees):
        current_um = extent_um * float(step + 1) / float(steps)
        if step_indegree == 0:
            previous_um = current_um
            continue
        selected = [
            idx
            for idx, center_um in enumerate(centers)
            if _inside_distance_ring(
                abs(center_um - target_center_um),
                lower_um=previous_um,
                upper_um=current_um,
                step=step,
            )
        ]
        if step == 0 and not selected:
            selected = [
                min(
                    range(len(source_ranges)),
                    key=lambda idx: abs(centers[idx] - target_center_um),
                )
            ]
        selected_intervals = contiguous_index_intervals(selected)
        intervals.extend(
            split_indegree_over_intervals(
                source_ranges,
                selected_intervals,
                step_indegree,
            )
        )
        previous_um = current_um
    return validated_repaired_intervals(intervals, requested_indegree)


def _inside_distance_ring(
    distance_um: float,
    *,
    lower_um: float,
    upper_um: float,
    step: int,
) -> bool:
    if step == 0:
        return 0.0 <= distance_um <= upper_um
    return lower_um < distance_um <= upper_um


def _require_positive(value: int, field: str) -> None:
    if value < 1:
        raise ValueError(f"{field} must be positive, got {value}")
