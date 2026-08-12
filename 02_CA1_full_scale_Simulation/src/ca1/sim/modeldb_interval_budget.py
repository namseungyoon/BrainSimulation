from __future__ import annotations

import math


def contiguous_index_intervals(indices: list[int]) -> tuple[tuple[int, int], ...]:
    if not indices:
        return ()
    intervals: list[tuple[int, int]] = []
    start = indices[0]
    previous = indices[0]
    for idx in indices[1:]:
        if idx == previous + 1:
            previous = idx
            continue
        intervals.append((start, previous))
        start = idx
        previous = idx
    intervals.append((start, previous))
    return tuple(intervals)


def split_indegree_over_intervals(
    source_ranges: tuple[tuple[int, int], ...],
    intervals: tuple[tuple[int, int], ...],
    indegree: int,
) -> tuple[tuple[int, int, int], ...]:
    if not intervals:
        return ()
    capacities = [
        _window_count(source_ranges, left, right)
        for left, right in intervals
    ]
    total_capacity = sum(capacities)
    if total_capacity < indegree:
        return ()
    raw = [
        float(indegree) * float(capacity) / float(total_capacity)
        for capacity in capacities
    ]
    allocated = [
        min(capacity, int(math.floor(value)))
        for value, capacity in zip(raw, capacities)
    ]
    _allocate_remaining_by_largest_remainder(allocated, capacities, raw)
    result: list[tuple[int, int, int]] = []
    for (left, right), interval_indegree in zip(intervals, allocated):
        if interval_indegree <= 0:
            continue
        result.append(
            (
                source_ranges[left][0],
                source_ranges[right][1],
                interval_indegree,
            )
        )
    return tuple(result)


def validated_repaired_intervals(
    intervals: list[tuple[int, int, int]],
    requested_indegree: int,
) -> tuple[tuple[int, int, int], ...]:
    if sum(indegree for _, _, indegree in intervals) != requested_indegree:
        intervals = list(
            _repair_interval_indegrees(
                intervals,
                requested_indegree,
            )
        )
    if not intervals:
        raise ValueError("binned fastconn produced no source intervals")
    if any(stop - start < indegree for start, stop, indegree in intervals):
        raise ValueError(
            "binned fastconn source window cannot satisfy requested indegree"
        )
    if sum(indegree for _, _, indegree in intervals) != requested_indegree:
        raise ValueError("binned fastconn cannot preserve requested indegree")
    return tuple(intervals)


def _allocate_remaining_by_largest_remainder(
    allocated: list[int],
    capacities: list[int],
    raw: list[float],
) -> None:
    remaining = round(sum(raw)) - sum(allocated)
    order = sorted(
        range(len(allocated)),
        key=lambda idx: raw[idx] - float(allocated[idx]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for idx in order:
            if remaining == 0:
                break
            if allocated[idx] >= capacities[idx]:
                continue
            allocated[idx] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break


def _repair_interval_indegrees(
    intervals: list[tuple[int, int, int]],
    requested_indegree: int,
) -> tuple[tuple[int, int, int], ...]:
    total = sum(indegree for _, _, indegree in intervals)
    repaired = list(intervals)
    while total < requested_indegree:
        candidates = [
            idx
            for idx, (start, stop, indegree) in enumerate(repaired)
            if indegree < stop - start
        ]
        if not candidates:
            break
        idx = candidates[0]
        start, stop, indegree = repaired[idx]
        repaired[idx] = (start, stop, indegree + 1)
        total += 1
    while total > requested_indegree:
        candidates = [
            idx
            for idx, (_, _, indegree) in enumerate(repaired)
            if indegree > 1
        ]
        if not candidates:
            break
        idx = candidates[-1]
        start, stop, indegree = repaired[idx]
        repaired[idx] = (start, stop, indegree - 1)
        total -= 1
    return tuple(repaired)


def _window_count(
    ranges: tuple[tuple[int, int], ...],
    left: int,
    right: int,
) -> int:
    return ranges[right][1] - ranges[left][0]
