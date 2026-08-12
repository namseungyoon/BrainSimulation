from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import multiprocessing
import os
import threading

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ca1.sim.modeldb_fastconn import (
    FastconnSourceInterval,
    fastconn_axon_distribution,
    fastconn_extent_um,
    fastconn_source_intervals,
    partition_ranges,
    range_center_um,
)
from ca1.types import Projection


_FASTCONN_STEPS = 5
_PARALLEL_MIN_DISTANCE_PAIRS = 1_000_000
_POST_CHUNKS_PER_WORKER = 4
_MAX_RESULT_EDGES_PER_CHUNK = 262_144
_MAX_TOPOLOGY_WORKERS = 32
# Keep this import graph simulator-free: the forkserver must never load NEST,
# NEST-GPU, MPI, or their fork-unsafe libinfinipath dependency.
_FORKSERVER_PRELOAD_MODULES = (
    "numpy",
    "scipy.spatial",
    "ca1.sim.modeldb_topology",
)
_FORKSERVER_CONTEXT = multiprocessing.get_context("forkserver")
_FORKSERVER_CONTEXT.set_forkserver_preload(list(_FORKSERVER_PRELOAD_MODULES))
_FORKSERVER_BOOT_LOCK = threading.Lock()
_FORKSERVER_BOOTED = False


def _forkserver_boot_probe() -> None:
    """Give the clean forkserver one harmless child to prove it is ready."""


def boot_modeldb_topology_forkserver() -> None:
    """Start the shared topology forkserver before fork-unsafe libraries load."""
    global _FORKSERVER_BOOTED
    if _FORKSERVER_BOOTED:
        return
    with _FORKSERVER_BOOT_LOCK:
        if _FORKSERVER_BOOTED:
            return
        # This MUST run before nestgpu/MPI is imported.  All later topology
        # workers then fork from this clean server and cannot inherit the
        # fork-unsafe libinfinipath loaded in the simulator process.
        probe = _FORKSERVER_CONTEXT.Process(target=_forkserver_boot_probe)
        probe.start()
        probe.join()
        if probe.exitcode != 0:
            raise RuntimeError(
                "could not boot the ModelDB topology forkserver "
                f"(probe exit code {probe.exitcode})"
            )
        _FORKSERVER_BOOTED = True


@dataclass(frozen=True, slots=True)
class BinnedFixedIndegreeConnection:
    source_start: int
    source_stop: int
    target_start: int
    target_stop: int
    indegree: int

    @property
    def source_count(self) -> int:
        return self.source_stop - self.source_start

    @property
    def target_count(self) -> int:
        return self.target_stop - self.target_start


@dataclass(frozen=True, slots=True)
class ModelDbFastconnPostEdges:
    """One post cell's distinct biological edges and their 3-D rings."""

    post_index: int
    source_indices: npt.NDArray[np.int64]
    ring_indices: npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class ProjectionPortPlan:
    projection: Projection
    edges_per_post: int


@dataclass(frozen=True, slots=True)
class ProjectionComponentPlan:
    name: str
    ports: tuple[ProjectionPortPlan, ...]


@dataclass(frozen=True, slots=True)
class RecurrentProjectionPlan:
    """One sampled biological pair set shared by all release components."""

    pre: str
    post: str
    delay_ms: float
    indegree: int
    components: tuple[ProjectionComponentPlan, ...]


@dataclass(frozen=True, slots=True)
class ProjectionPortEdges:
    component: str
    projection: Projection
    source_indices: npt.NDArray[np.int64]
    target_indices: npt.NDArray[np.int64]


def apportion_integer(total: int, weights: Sequence[float]) -> tuple[int, ...]:
    """Largest-remainder integer apportionment with stable index tie-breaking."""
    if total < 0:
        raise ValueError(f"total must be nonnegative, got {total}")
    if not weights:
        raise ValueError("cannot apportion over zero weights")
    values = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("apportionment weights must be finite and nonnegative")
    denominator = float(values.sum())
    if denominator <= 0.0:
        raise ValueError("at least one apportionment weight must be positive")
    raw = values / denominator * float(total)
    allocated = np.floor(raw).astype(np.int64)
    remainder = total - int(allocated.sum())
    if remainder:
        order = sorted(
            range(len(weights)),
            key=lambda idx: (-(raw[idx] - allocated[idx]), idx),
        )
        allocated[np.asarray(order[:remainder], dtype=np.int64)] += 1
    return tuple(int(value) for value in allocated)


def gaussian_ring_weights(pre_type: str) -> tuple[float, ...]:
    """ModelDB ``fastconn.mod`` Gaussian mass evaluated at five ring bounds."""
    distribution = fastconn_axon_distribution(pre_type)
    extent_um = 4.0 * distribution.c_um
    raw = tuple(
        math.exp(
            -distribution.a
            * (
                (
                    extent_um * float(step + 1) / float(_FASTCONN_STEPS)
                    - distribution.b_um
                )
                / distribution.c_um
            )
            ** 2
        )
        for step in range(_FASTCONN_STEPS)
    )
    denominator = sum(raw)
    return tuple(value / denominator for value in raw)


def gaussian_ring_indegrees(pre_type: str, indegree: int) -> tuple[int, ...]:
    """Exact-K integer ring targets derived from the ModelDB Gaussian weights."""
    _require_positive(indegree, "indegree")
    return apportion_integer(indegree, gaussian_ring_weights(pre_type))


def feasible_gaussian_ring_indegrees(
    candidate_counts: Sequence[int],
    desired_counts: Sequence[int],
) -> tuple[int, ...]:
    """Apply ``fastconn.mod:264`` shortage redistribution without losing K."""
    if len(candidate_counts) != len(desired_counts):
        raise ValueError("candidate and desired ring counts must have equal length")
    if any(count < 0 for count in candidate_counts) or any(
        count < 0 for count in desired_counts
    ):
        raise ValueError("ring counts must be nonnegative")
    requested = sum(int(count) for count in desired_counts)
    if sum(int(count) for count in candidate_counts) < requested:
        raise ValueError(
            "fewer distinct presynaptic candidates exist within the ModelDB "
            f"4c extent than requested indegree {requested}"
        )

    feasible = [int(count) for count in desired_counts]
    rem = 0
    steps = len(feasible)
    for step in range(steps):
        available = int(candidate_counts[step])
        if feasible[step] + rem <= available:
            continue
        rem = feasible[step] + rem - available
        if step > 0:
            for delta in range(1, step + 1):
                previous = step - delta
                spare = int(candidate_counts[previous]) - feasible[previous]
                if spare <= 0:
                    continue
                extra = min(spare, rem)
                feasible[previous] += extra
                feasible[step] -= extra
                rem -= extra
        if rem > 0 and step < steps - 1:
            for later in range(step + 1, steps):
                spare = int(candidate_counts[later]) - feasible[later]
                if spare <= 0:
                    continue
                extra = min(spare, rem)
                feasible[later] += extra
                feasible[step] -= extra
                rem -= extra
                if rem == 0:
                    break

    if rem or sum(feasible) != requested:
        raise ValueError("ModelDB feasibility redistribution could not preserve indegree")
    if any(
        count < 0 or count > int(candidate_counts[idx])
        for idx, count in enumerate(feasible)
    ):
        raise ValueError("ModelDB feasibility redistribution produced invalid counts")
    return tuple(feasible)


def _stable_projection_post_seed(
    seed: int,
    projection: str,
    post_index: int,
) -> int:
    payload = f"{int(seed)}\0{projection}\0{int(post_index)}".encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"ca1-3dtopo",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


@dataclass(slots=True)
class _ProjectionSampler:
    pre_positions: npt.NDArray[np.float64]
    post_positions: npt.NDArray[np.float64]
    tree: cKDTree
    source_indices: npt.NDArray[np.int64]
    boundaries: npt.NDArray[np.float64]
    squared_boundaries: npt.NDArray[np.float64]
    desired: tuple[int, ...]
    all_sources_inside: npt.NDArray[np.bool_]
    exact_half_micron_grid: bool

    def sample_post(
        self,
        post_index: int,
        *,
        seed: int,
        projection: str,
    ) -> ModelDbFastconnPostEdges:
        post_position = self.post_positions[post_index]
        if self.all_sources_inside[post_index]:
            candidates = self.source_indices
            if self.exact_half_micron_grid:
                distance_values = cdist(
                    post_position.reshape(1, 3),
                    self.pre_positions,
                    metric="sqeuclidean",
                )[0]
                ring_boundaries = self.squared_boundaries
            else:
                distance_values = np.linalg.norm(
                    self.pre_positions - post_position,
                    axis=1,
                )
                ring_boundaries = self.boundaries
        else:
            pairs = cKDTree(post_position.reshape(1, 3)).sparse_distance_matrix(
                self.tree,
                float(self.boundaries[-1]),
                output_type="ndarray",
            )
            order = np.argsort(pairs["j"])
            candidates = np.asarray(pairs["j"][order], dtype=np.int64)
            distance_values = np.asarray(pairs["v"][order], dtype=np.float64)
            ring_boundaries = self.boundaries

        within_two = distance_values <= ring_boundaries[2]
        ring_candidates_by_ring: tuple[npt.NDArray[np.int64], ...] | None
        if np.count_nonzero(within_two) >= sum(self.desired):
            within_zero = distance_values <= ring_boundaries[0]
            within_one = distance_values <= ring_boundaries[1]
            ring_candidates_by_ring = (
                candidates[within_zero],
                candidates[within_one & ~within_zero],
                candidates[within_two & ~within_one],
            )
            candidate_counts = tuple(
                len(ring_candidates) for ring_candidates in ring_candidates_by_ring
            ) + (0, 0)
        else:
            rings = np.zeros(len(distance_values), dtype=np.int8)
            for boundary in ring_boundaries:
                rings += distance_values > boundary
            counts = np.bincount(rings, minlength=_FASTCONN_STEPS)
            candidate_counts = tuple(int(count) for count in counts[:_FASTCONN_STEPS])
            ring_candidates_by_ring = None

        feasible = feasible_gaussian_ring_indegrees(candidate_counts, self.desired)
        rng = np.random.default_rng(
            _stable_projection_post_seed(seed, projection, post_index)
        )
        selected_sources: list[npt.NDArray[np.int64]] = []
        selected_rings: list[npt.NDArray[np.int8]] = []
        for ring, count in enumerate(feasible):
            if count == 0:
                continue
            ring_candidates = (
                ring_candidates_by_ring[ring]
                if ring_candidates_by_ring is not None
                else candidates[rings == ring]
            )
            chosen = np.asarray(
                rng.choice(ring_candidates, size=count, replace=False),
                dtype=np.int64,
            )
            selected_sources.append(chosen)
            selected_rings.append(np.full(count, ring + 1, dtype=np.int8))
        sources = np.concatenate(selected_sources)
        ring_numbers = np.concatenate(selected_rings)
        permutation = rng.permutation(sum(self.desired))
        return ModelDbFastconnPostEdges(
            post_index=post_index,
            source_indices=sources[permutation],
            ring_indices=ring_numbers[permutation],
        )


_WORKER_TOPOLOGY: ModelDbFastconn3D | None = None


def _initialize_topology_worker(
    positions_um: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    global _WORKER_TOPOLOGY
    _WORKER_TOPOLOGY = ModelDbFastconn3D(positions_um, max_workers=1)


def _sample_projection_chunk(
    job: tuple[str, str, int, int, str, int, int],
) -> tuple[ModelDbFastconnPostEdges, ...]:
    pre_type, post_type, indegree, seed, projection, start, stop = job
    if _WORKER_TOPOLOGY is None:
        raise RuntimeError("3-D topology worker was not initialized")
    sampler = _WORKER_TOPOLOGY._sampler(pre_type, post_type, indegree)
    return tuple(
        sampler.sample_post(post_index, seed=seed, projection=projection)
        for post_index in range(start, stop)
    )


class ModelDbFastconn3D:
    """Cached position-aware implementation of ModelDB's 3-D ``fastconn``."""

    def __init__(
        self,
        positions_um: Mapping[str, npt.NDArray[np.float64]],
        *,
        max_workers: int | None = None,
        force_parallel: bool = False,
    ) -> None:
        self._positions: dict[str, npt.NDArray[np.float64]] = {}
        for cell_type, raw_positions in positions_um.items():
            positions = np.asarray(raw_positions, dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError(
                    f"positions for {cell_type!r} must have shape (n_cells, 3)"
                )
            self._positions[cell_type] = positions
        self._trees: dict[str, cKDTree] = {}
        self._source_indices: dict[str, npt.NDArray[np.int64]] = {}
        self._samplers: dict[tuple[str, str, int], _ProjectionSampler] = {}
        requested_workers = os.cpu_count() if max_workers is None else max_workers
        self._max_workers = max(1, min(_MAX_TOPOLOGY_WORKERS, requested_workers or 1))
        self._force_parallel = force_parallel
        self._executor: ProcessPoolExecutor | None = None

    def close(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown()
            self._executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may already have torn down multiprocessing.
            pass

    def _sampler(
        self,
        pre_type: str,
        post_type: str,
        indegree: int,
    ) -> _ProjectionSampler:
        key = (pre_type, post_type, indegree)
        cached = self._samplers.get(key)
        if cached is not None:
            return cached

        try:
            pre_positions = self._positions[pre_type]
            post_positions = self._positions[post_type]
        except KeyError as exc:
            raise ValueError(f"positions missing for cell type {exc.args[0]!r}") from exc
        if indegree > len(pre_positions):
            raise ValueError(
                f"indegree {indegree} exceeds source population {pre_type}="
                f"{len(pre_positions)}"
            )
        tree = self._trees.get(pre_type)
        if tree is None:
            tree = cKDTree(pre_positions)
            self._trees[pre_type] = tree
        source_indices = self._source_indices.get(pre_type)
        if source_indices is None:
            source_indices = np.arange(len(pre_positions), dtype=np.int64)
            self._source_indices[pre_type] = source_indices
        distribution = fastconn_axon_distribution(pre_type)
        boundaries = (
            4.0
            * distribution.c_um
            * np.arange(1, _FASTCONN_STEPS + 1, dtype=np.float64)
            / float(_FASTCONN_STEPS)
        )
        extent_um = float(boundaries[-1])
        pre_min = np.min(pre_positions, axis=0)
        pre_max = np.max(pre_positions, axis=0)
        farthest_deltas = np.maximum(
            np.abs(post_positions - pre_min),
            np.abs(post_positions - pre_max),
        )
        sampler = _ProjectionSampler(
            pre_positions=pre_positions,
            post_positions=post_positions,
            tree=tree,
            source_indices=source_indices,
            boundaries=boundaries,
            squared_boundaries=boundaries * boundaries,
            desired=gaussian_ring_indegrees(pre_type, indegree),
            all_sources_inside=(
                np.einsum("ij,ij->i", farthest_deltas, farthest_deltas)
                <= extent_um * extent_um
            ),
            exact_half_micron_grid=bool(
                np.all(pre_positions * 2.0 == np.rint(pre_positions * 2.0))
                and np.all(post_positions * 2.0 == np.rint(post_positions * 2.0))
                and np.all(boundaries * 2.0 == np.rint(boundaries * 2.0))
            ),
        )
        self._samplers[key] = sampler
        return sampler

    def _process_pool(self, workers: int) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=_FORKSERVER_CONTEXT,
                initializer=_initialize_topology_worker,
                initargs=(self._positions,),
            )
        return self._executor

    def iter_post_edges(
        self,
        *,
        pre_type: str,
        post_type: str,
        indegree: int,
        seed: int,
        projection: str | None = None,
    ) -> Iterator[ModelDbFastconnPostEdges]:
        _require_positive(indegree, "indegree")
        sampler = self._sampler(pre_type, post_type, indegree)
        pre_positions = sampler.pre_positions
        post_positions = sampler.post_positions
        projection_key = projection or f"{pre_type}->{post_type}"

        workers = min(
            self._max_workers,
            len(post_positions),
        )
        if (
            workers <= 1
            or (
                not self._force_parallel
                and len(pre_positions) * len(post_positions)
                < _PARALLEL_MIN_DISTANCE_PAIRS
            )
        ):
            for post_index in range(len(post_positions)):
                yield sampler.sample_post(
                    post_index,
                    seed=seed,
                    projection=projection_key,
                )
            return

        target_chunk_size = math.ceil(
            len(post_positions) / float(workers * _POST_CHUNKS_PER_WORKER)
        )
        edge_limited_chunk_size = max(1, _MAX_RESULT_EDGES_PER_CHUNK // indegree)
        chunk_size = max(1, min(target_chunk_size, edge_limited_chunk_size))
        jobs = (
            (
                pre_type,
                post_type,
                indegree,
                seed,
                projection_key,
                chunk_start,
                min(chunk_start + chunk_size, len(post_positions)),
            )
            for chunk_start in range(0, len(post_positions), chunk_size)
        )
        for chunk in self._process_pool(workers).map(_sample_projection_chunk, jobs):
            yield from chunk


def recurrent_projection_plans(
    projections: Sequence[Projection],
    pre_counts: Mapping[str, int],
) -> tuple[RecurrentProjectionPlan, ...]:
    """Group split ports/co-release rows under one exact biological edge set."""
    grouped: dict[tuple[str, str, float], list[Projection]] = {}
    for projection in projections:
        grouped.setdefault(
            (projection.pre, projection.post, projection.delay_ms), []
        ).append(projection)

    plans: list[RecurrentProjectionPlan] = []
    for (pre, post, delay_ms), pair_projections in grouped.items():
        components: dict[str, list[Projection]] = {}
        for projection in pair_projections:
            components.setdefault(projection.release_component, []).append(projection)

        biological_values = [
            float(projection.biological_indegree)
            for projection in pair_projections
            if projection.biological_indegree is not None
        ]
        if biological_values:
            if len(biological_values) != len(pair_projections):
                raise ValueError(
                    f"mixed biological_indegree metadata for {pre}->{post}"
                )
            base_value = biological_values[0]
            if any(
                not math.isclose(value, base_value, rel_tol=0.0, abs_tol=1e-9)
                for value in biological_values[1:]
            ):
                raise ValueError(
                    f"inconsistent biological indegrees for {pre}->{post}: "
                    f"{biological_values}"
                )
        else:
            component_totals = [
                sum(float(projection.indegree) for projection in component)
                for component in components.values()
            ]
            base_value = component_totals[0]
            if any(
                not math.isclose(value, base_value, rel_tol=0.0, abs_tol=1e-9)
                for value in component_totals[1:]
            ):
                raise ValueError(
                    f"co-release component indegrees differ for {pre}->{post}: "
                    f"{component_totals}"
                )
        rounded = int(round(base_value))
        if rounded < 1 or not math.isfinite(base_value):
            raise ValueError(
                f"biological indegree for {pre}->{post} must round to a "
                f"positive integer, got {base_value}"
            )
        try:
            pre_count = int(pre_counts[pre])
        except KeyError as exc:
            raise ValueError(f"source count missing for {pre!r}") from exc
        base_indegree = min(rounded, pre_count)

        component_plans: list[ProjectionComponentPlan] = []
        for component_name, component in components.items():
            allocations = apportion_integer(
                base_indegree,
                [float(projection.indegree) for projection in component],
            )
            component_plans.append(
                ProjectionComponentPlan(
                    name=component_name,
                    ports=tuple(
                        ProjectionPortPlan(projection, allocation)
                        for projection, allocation in zip(
                            component, allocations, strict=True
                        )
                        if allocation > 0
                    ),
                )
            )
        plans.append(
            RecurrentProjectionPlan(
                pre=pre,
                post=post,
                delay_ms=delay_ms,
                indegree=base_indegree,
                components=tuple(component_plans),
            )
        )
    return tuple(plans)


def projection_port_edges(
    plan: RecurrentProjectionPlan,
    post_edges: ModelDbFastconnPostEdges,
) -> tuple[ProjectionPortEdges, ...]:
    """Assign one base edge set to ports; co-release components reuse it."""
    if len(post_edges.source_indices) != plan.indegree:
        raise ValueError(
            f"base edge count {len(post_edges.source_indices)} does not match "
            f"planned indegree {plan.indegree}"
        )
    assignments: list[ProjectionPortEdges] = []
    for component in plan.components:
        offset = 0
        for port in component.ports:
            stop = offset + port.edges_per_post
            sources = post_edges.source_indices[offset:stop]
            assignments.append(
                ProjectionPortEdges(
                    component=component.name,
                    projection=port.projection,
                    source_indices=sources,
                    target_indices=np.full(
                        len(sources), post_edges.post_index, dtype=np.int64
                    ),
                )
            )
            offset = stop
        if offset != plan.indegree:
            raise ValueError(
                f"port allocations for {plan.pre}->{plan.post} component "
                f"{component.name!r} sum to {offset}, expected {plan.indegree}"
            )
    return tuple(assignments)


def binned_fixed_indegree_connections(
    *,
    pre_type: str,
    post_type: str,
    pre_count: int,
    post_count: int,
    indegree: int,
    n_x_bins: int = 64,
) -> tuple[BinnedFixedIndegreeConnection, ...]:
    _ = post_type
    _require_positive(pre_count, "pre_count")
    _require_positive(post_count, "post_count")
    _require_positive(indegree, "indegree")
    _require_positive(n_x_bins, "n_x_bins")
    source_ranges = partition_ranges(pre_count, min(pre_count, n_x_bins))
    target_ranges = partition_ranges(post_count, min(post_count, n_x_bins))
    requested_indegree = min(indegree, pre_count)
    extent_um = fastconn_extent_um(pre_type)
    calls: list[BinnedFixedIndegreeConnection] = []
    for target_start, target_stop in target_ranges:
        target_center_um = range_center_um(
            target_start,
            target_stop,
            post_count,
        )
        interval = _uniform_source_interval(
            source_ranges=source_ranges,
            source_total=pre_count,
            target_center_um=target_center_um,
            extent_um=extent_um,
            indegree=requested_indegree,
        )
        calls.extend(
            _target_connections_from_intervals(
                target_start=target_start,
                target_stop=target_stop,
                intervals=(interval,),
                requested_indegree=requested_indegree,
            )
        )
    return tuple(calls)


def gaussian_binned_fixed_indegree_connections(
    *,
    pre_type: str,
    post_type: str,
    pre_count: int,
    post_count: int,
    indegree: int,
    n_x_bins: int = 64,
) -> tuple[BinnedFixedIndegreeConnection, ...]:
    _ = post_type
    _require_positive(pre_count, "pre_count")
    _require_positive(post_count, "post_count")
    _require_positive(indegree, "indegree")
    _require_positive(n_x_bins, "n_x_bins")
    source_ranges = partition_ranges(pre_count, min(pre_count, n_x_bins))
    target_ranges = partition_ranges(post_count, min(post_count, n_x_bins))
    requested_indegree = min(indegree, pre_count)
    calls: list[BinnedFixedIndegreeConnection] = []
    for target_start, target_stop in target_ranges:
        target_center_um = range_center_um(target_start, target_stop, post_count)
        intervals = fastconn_source_intervals(
            pre_type=pre_type,
            source_ranges=source_ranges,
            source_total=pre_count,
            target_center_um=target_center_um,
            requested_indegree=requested_indegree,
        )
        calls.extend(
            _target_connections_from_intervals(
                target_start=target_start,
                target_stop=target_stop,
                intervals=intervals,
                requested_indegree=requested_indegree,
            )
        )
    return tuple(calls)


def _target_connections_from_intervals(
    *,
    target_start: int,
    target_stop: int,
    intervals: tuple[FastconnSourceInterval, ...],
    requested_indegree: int,
) -> tuple[BinnedFixedIndegreeConnection, ...]:
    if sum(interval.indegree for interval in intervals) != requested_indegree:
        raise ValueError("binned fastconn cannot preserve requested indegree")
    return tuple(
        BinnedFixedIndegreeConnection(
            source_start=interval.source_start,
            source_stop=interval.source_stop,
            target_start=target_start,
            target_stop=target_stop,
            indegree=interval.indegree,
        )
        for interval in intervals
    )


def _uniform_source_interval(
    *,
    source_ranges: tuple[tuple[int, int], ...],
    source_total: int,
    target_center_um: float,
    extent_um: float,
    indegree: int,
) -> FastconnSourceInterval:
    centers = tuple(
        range_center_um(start, stop, source_total)
        for start, stop in source_ranges
    )
    selected = [
        idx
        for idx, center_um in enumerate(centers)
        if abs(center_um - target_center_um) <= extent_um
    ]
    if not selected:
        selected = [
            min(
                range(len(source_ranges)),
                key=lambda idx: abs(centers[idx] - target_center_um),
            )
        ]
    return FastconnSourceInterval(
        source_start=source_ranges[min(selected)][0],
        source_stop=source_ranges[max(selected)][1],
        indegree=indegree,
    )


def _require_positive(value: int, field: str) -> None:
    if value < 1:
        raise ValueError(f"{field} must be positive, got {value}")
