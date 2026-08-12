from __future__ import annotations

from ca1.sim.modeldb_topology import (
    binned_fixed_indegree_connections,
    gaussian_binned_fixed_indegree_connections,
)


def test_binned_fixed_indegree_covers_every_target_once() -> None:
    calls = binned_fixed_indegree_connections(
        pre_type="Pyramidal",
        post_type="PV_Basket",
        pre_count=311_500,
        post_count=5_530,
        indegree=813,
        n_x_bins=32,
    )

    assert calls[0].target_start == 0
    assert calls[-1].target_stop == 5_530
    assert len(calls) == 32
    assert sum(call.target_count for call in calls) == 5_530
    assert all(call.indegree == 813 for call in calls)
    assert sum(call.target_count * call.indegree for call in calls) == (
        5_530 * 813
    )


def test_binned_fixed_indegree_uses_local_source_windows() -> None:
    calls = binned_fixed_indegree_connections(
        pre_type="Pyramidal",
        post_type="PV_Basket",
        pre_count=311_500,
        post_count=5_530,
        indegree=813,
        n_x_bins=32,
    )

    assert len(calls) == 32
    assert all(call.source_count < 311_500 for call in calls)
    assert calls[0].source_start < calls[-1].source_start
    assert calls[0].source_stop < calls[-1].source_stop


def test_binned_fixed_indegree_expands_small_source_windows_to_indegree() -> None:
    calls = binned_fixed_indegree_connections(
        pre_type="O_LM",
        post_type="Pyramidal",
        pre_count=1_640,
        post_count=311_500,
        indegree=21,
        n_x_bins=64,
    )

    assert len(calls) == 64
    assert all(call.source_count >= 21 for call in calls)
    assert all(call.source_count < 1_640 for call in calls[1:-1])


def test_gaussian_binned_fixed_indegree_splits_target_budget_over_rings() -> None:
    calls = gaussian_binned_fixed_indegree_connections(
        pre_type="Pyramidal",
        post_type="PV_Basket",
        pre_count=311_500,
        post_count=5_530,
        indegree=813,
        n_x_bins=32,
    )

    by_target: dict[tuple[int, int], int] = {}
    for call in calls:
        key = (call.target_start, call.target_stop)
        by_target[key] = by_target.get(key, 0) + call.indegree
    assert len(calls) > 32
    assert len(by_target) == 32
    assert all(target_indegree == 813 for target_indegree in by_target.values())
    assert sum(call.target_count * call.indegree for call in calls) == (
        5_530 * 813
    )
    first_target = [
        call for call in calls
        if call.target_start == calls[0].target_start
    ]
    assert first_target[0].indegree > first_target[1].indegree
