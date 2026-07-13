"""NEST GPU backend for the CA1 hippocampal model.

``import nestgpu as ngpu`` is deferred to ``setup()`` so this module is safe to
import on machines without CUDA.

Canonical full-scale validation runs on one A40.  MPI recurrent sharding is
kept for explicit benchmark/reproduction work only because recurrent sharded
outputs are not currently final-equivalent to one-GPU runs.

    Explicit MPI benchmark deployment::

    CA1_ALLOW_MPI_RECURRENT_SHARDING=1 \
    mpirun -np 3 python -m ca1.cli sim --backend gpu --scale 1.0

One rank per A40 (46 GB HBM).  Each rank instantiates only its shard of every
population and recurrent projections are created with NEST GPU's distributed
fixed-indegree MPI API.  This is distinct from the calibration scripts, which
may still use one independent process per GPU for parameter sweeps.

    Receptor ports are 0-based in NEST GPU, unlike CPU NEST which is 1-based.
    The concrete port count and kinetics come from the postsynaptic receptor
    table and may use up to the 20 NEST-GPU receptor ports per node group.

Inhibitory weights MUST be POSITIVE; the sign of inhibition is encoded by
E_rev on the port (GABA ports have E_rev below rest).

``spec.weight_compensation`` is applied to recurrent weights only; afferent
weights are kept verbatim (independent Poisson sources, not a scaled pop).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path
from typing import Final, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ca1.params.aglif import aglif_params_for_cell_type
from ca1.params.izhikevich import izhikevich_params_for_cell_type
from ca1.sim.afferents import afferent_poisson_drive
from ca1.sim.aglif_dend import (
    aglif_dend_compartments as _aglif_dend_compartments,
    aglif_dend_status as _aglif_dend_status,
)
from ca1.sim.backend import SimulatorBackend
from ca1.sim.intrinsic_heterogeneity import intrinsic_heterogeneity_status
from ca1.sim.edge_artifact import EdgeArtifact, load_edge_artifact_from_env
from ca1.sim.modeldb_topology import (
    ModelDbFastconn3D,
    binned_fixed_indegree_connections,
    boot_modeldb_topology_forkserver,
    gaussian_binned_fixed_indegree_connections,
    projection_port_edges,
    recurrent_projection_plans,
)
from ca1.sim.modeldb_positions import (
    MODELDB_NPOLE_ELECTRODE_ROI,
    electrode_roi_mask,
    modeldb_cell_positions,
    modeldb_connectivity_positions,
)
from ca1.sim.nestgpu_api import (
    NestGpuModule,
    NestGpuNodes,
    NestGpuRemoteNodes,
    nestgpu_module,
)
from ca1.sim.npole_lfp import (
    MODELDB_NPOLE_RHO_OHM_CM,
    reduced_domain_n_pole_lfp,
    reduced_domain_n_pole_weights,
)
from ca1.sim.source_rate_heterogeneity import source_rates_hz
from ca1.sim.weights import nonnegative_weight_nS
from ca1.types import (
    SUPPORTED_NEURON_MODELS,
    Afferent,
    NetworkSpec,
    NeuronParams,
    ReceptorConfig,
    SimMeta,
    SimResult,
)

_MAX_NESTGPU_RECEPTOR_PORTS: Final = 20
_MAX_NESTGPU_LOCAL_NODES: Final = 1_048_576
_SPIKE_TRAIN_BATCH_MAX_SPIKES: Final = 1_000_000
_AFFERENT_TOPOLOGY_ENV: Final = "CA1_AFFERENT_TOPOLOGY"
_AFFERENT_TOPOLOGY_COMPOUND: Final = "compound"
_AFFERENT_TOPOLOGY_SOURCE_POOL: Final = "source_pool"
_AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH: Final = "literal_source_graph"
_AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH_BINNED: Final = "literal_source_graph_binned"
_LITERAL_SOURCE_DRIVER: Final = "precomputed_poisson_spike_generator"
_SOURCE_POOL_SIZE_ENV: Final = "CA1_AFFERENT_SOURCE_POOL_SIZE"
_SOURCE_POOL_INDEGREE_ENV: Final = "CA1_AFFERENT_SOURCE_POOL_INDEGREE"
_LFP_SAMPLE_CELLS_ENV: Final = "CA1_GPU_LFP_SAMPLE_CELLS"
_LFP_RECORD_EVERY_ENV: Final = "CA1_LFP_RECORD_EVERY"
_MAX_REC_SPIKES_ENV: Final = "CA1_GPU_MAX_REC_SPIKES"
_DEFAULT_LFP_SAMPLE_CELLS: Final = 128
# At dt=0.1 ms this records LFP variables every 1 ms, retaining ample samples
# per gamma cycle while reducing multimeter device-to-host traffic tenfold.
_DEFAULT_LFP_RECORD_EVERY: Final = 10
_EXPLICIT_EDGE_CHUNK_SIZE: Final = 1_048_576
_FUSED_EXPLICIT_CONNECT_ENV: Final = "CA1_GPU_FUSED_EXPLICIT_CONNECT"
_ZERO_COPY_EXPLICIT_CONNECT_ENV: Final = "CA1_GPU_ZERO_COPY_EXPLICIT_CONNECT"
_MODELDB_FULL_PYRAMIDAL_COUNT: Final = 311_500
_LFP_PROXY_MODELDB_N_POLE_REDUCED: Final = "modeldb_n_pole_reduced_domain_lfp"
_LFP_PROXY_SYNAPTIC_CURRENT: Final = "pyramidal_synaptic_current"
_LFP_PROXY_SPIKE_DENSITY: Final = "pyramidal_spike_density"
_LFP_MODELDB_N_POLE_PROVENANCE_KEY: Final = "lfp.modeldb_n_pole_reduced_domain"
_LFP_MODELDB_N_POLE_PROVENANCE_VALUE: Final = "modeldb-n-pole-reduced-domain-lfp"
_AfferentTopology = Literal[
    "compound",
    "source_pool",
    "literal_source_graph",
    "literal_source_graph_binned",
]


@dataclass(frozen=True)
class _LfpConductanceColumn:
    sample_idx: int
    port_idx: int
    compartment: float
    e_rev_mv: float


def _mpi_rank_size_from_env() -> tuple[int, int]:
    rank = os.environ.get("PMI_RANK") or os.environ.get("OMPI_COMM_WORLD_RANK")
    size = os.environ.get("PMI_SIZE") or os.environ.get("OMPI_COMM_WORLD_SIZE")
    return int(rank or "0"), int(size or "1")


def _partition_counts(total: int, size: int) -> list[int]:
    base = total // size
    remainder = total % size
    return [base + (1 if rank < remainder else 0) for rank in range(size)]


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a nonnegative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {raw!r}")
    return value


def _record_spike_buffer_size(duration_s: float) -> int:
    requested = os.environ.get(_MAX_REC_SPIKES_ENV)
    if requested is not None:
        return _env_positive_int(_MAX_REC_SPIKES_ENV, 0)
    return min(
        NestGpuBackend.MAX_REC_SPIKES,
        max(16, int(np.ceil(duration_s * 250.0)) + 16),
    )


def _stable_source_seed(seed: int, source: str) -> int:
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(source))
    return (int(seed) + offset) % (2**32)


def _stable_working_point_seed(seed: int, cell_type: str, cell_index: int) -> int:
    payload = f"{int(seed)}\0{cell_type}\0{int(cell_index)}".encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"ca1-wp",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _sample_distinct_spike_slots(
    counts: npt.NDArray[np.integer],
    *,
    slot_count: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Sample sorted, distinct slots for variable-size source spike trains.

    The output is grouped by source index, with ``offsets`` delimiting each
    source's sorted slots.  It uses O(total_spikes) memory, rather than an
    infeasible source-by-slot matrix.  Collisions are re-drawn in bulk; at the
    low occupancies used for Poisson afferents this normally needs one pass.
    """
    counts = np.asarray(counts, dtype=np.int64)
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    total_spikes = int(offsets[-1])
    if total_spikes == 0:
        return offsets, np.empty(0, dtype=np.int64)

    # Re-drawing collisions is efficient at the sparse occupancy of Poisson
    # afferents.  Near a full slot grid it becomes coupon-collector-like, so
    # retain NumPy's exact without-replacement algorithm for that exceptional
    # case.  Passing an integer avoids materializing np.arange() ourselves.
    if int(counts.max()) > (slot_count - 1) // 8:
        slots = np.empty(total_spikes, dtype=np.int64)
        for source_idx, count in enumerate(counts):
            begin, end = offsets[source_idx:source_idx + 2]
            if count:
                slots[begin:end] = np.sort(
                    rng.choice(slot_count - 1, size=int(count), replace=False) + 1
                )
        return offsets, slots

    # One vectorized draw covers all source/node spike placements in this
    # batch.  The source id is retained only long enough to identify and
    # re-draw duplicate slots within the same source.
    source_indices = np.repeat(
        np.arange(counts.size, dtype=np.int32), counts
    )
    slots = rng.integers(1, slot_count, size=total_spikes, dtype=np.int64)
    order = np.lexsort((slots, source_indices))
    while True:
        ordered_sources = source_indices[order]
        ordered_slots = slots[order]
        duplicate_mask = (
            (ordered_sources[1:] == ordered_sources[:-1])
            & (ordered_slots[1:] == ordered_slots[:-1])
        )
        if not np.any(duplicate_mask):
            return offsets, ordered_slots
        duplicate_indices = order[1:][duplicate_mask]
        slots[duplicate_indices] = rng.integers(
            1,
            slot_count,
            size=duplicate_indices.size,
            dtype=np.int64,
        )
        order = np.lexsort((slots, source_indices))


def _spike_slot_batches(
    counts: npt.NDArray[np.integer],
    *,
    slot_count: int,
    rng: np.random.Generator,
    max_spikes: int = _SPIKE_TRAIN_BATCH_MAX_SPIKES,
) -> Iterable[tuple[int, npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Yield bounded-memory batches of source-indexed spike slots."""
    counts = np.asarray(counts, dtype=np.int64)
    start = 0
    while start < counts.size:
        cumulative = np.cumsum(counts[start:], dtype=np.int64)
        width = int(np.searchsorted(cumulative, max_spikes, side="right"))
        # A single source can be larger than max_spikes, while still being
        # valid because the caller limits it to slot_count - 1.
        end = start + max(1, width)
        offsets, slots = _sample_distinct_spike_slots(
            counts[start:end],
            slot_count=slot_count,
            rng=rng,
        )
        yield start, offsets, slots
        start = end


def _set_intrinsic_heterogeneity(
    ngpu: NestGpuModule,
    nodes: NestGpuNodes | NestGpuRemoteNodes,
    *,
    cell_type: str,
    params: NeuronParams,
    baseline_status: Mapping[str, float],
    count: int,
    seed: int,
    shard: int,
) -> None:
    for field, payload in intrinsic_heterogeneity_status(
        cell_type=cell_type,
        params=params,
        baseline_status=baseline_status,
        count=count,
        seed=seed,
        shard=shard,
    ).items():
        ngpu.SetStatus(nodes, field, payload)


def _validate_receptor_port_budget(
    spec: NetworkSpec,
    model_name: str,
) -> None:
    if model_name not in {
        "aeif_cond_beta_multisynapse",
        "izhikevich_cond_beta",
        "user_m1",
        "user_m2",
    }:
        return
    for post in spec.cell_types:
        receptors = spec.receptors_for_post(post)
        n_ports = receptors.n_ports()
        if n_ports <= _MAX_NESTGPU_RECEPTOR_PORTS:
            continue
        message = " ".join(
            (
                f"{spec.neuron_model} maps to NEST-GPU {model_name}, which",
                f"supports at most {_MAX_NESTGPU_RECEPTOR_PORTS} receptor ports",
                f"per postsynaptic group; post={post!r} got {n_ports}",
            )
        )
        raise ValueError(message)


def _port_index_for_post(
    receptors: ReceptorConfig,
    *,
    post: str,
    receptor: str,
) -> int:
    try:
        return receptors.port_index(receptor)
    except ValueError as exc:
        message = (
            f"receptor {receptor!r} is not present in the receptor table for "
            f"postsynaptic cell type {post!r}"
        )
        raise KeyError(message) from exc


def _explicit_node_ids(
    nodes: NestGpuNodes,
    indices: Sequence[int] | npt.NDArray[np.int64],
) -> list[int]:
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.size == 0:
        return []
    # Create() returns a contiguous NodeSeq.  Translate its local indices in
    # NumPy and cross the Python/C++ boundary once, instead of indexing the
    # NodeSeq once per edge in Python.
    return (index_array + int(nodes[0])).tolist()


def _explicit_node_ids_uint32(
    nodes: NestGpuNodes,
    indices: Sequence[int] | npt.NDArray[np.int64],
) -> npt.NDArray[np.uint32]:
    """Map local indices to ABI-ready global IDs without Python objects."""
    index_array = np.asarray(indices)
    if index_array.ndim != 1:
        raise ValueError("explicit node indices must be one-dimensional")
    if not np.issubdtype(index_array.dtype, np.integer):
        raise TypeError("explicit node indices must be integers")
    if index_array.size == 0:
        return np.empty(0, dtype=np.uint32)
    minimum = int(index_array.min())
    maximum = int(index_array.max())
    if minimum < 0 or maximum >= len(nodes):
        raise ValueError(
            f"explicit node index range [{minimum}, {maximum}] is outside "
            f"[0, {len(nodes)})"
        )
    first_node = int(nodes[0])
    maximum_id = first_node + maximum
    if first_node < 0 or maximum_id > np.iinfo(np.uint32).max:
        raise OverflowError("explicit global node ID is outside uint32 range")
    # astype performs the only required host operation: one contiguous uint32
    # buffer whose data pointer is consumed directly by NEST-GPU.
    return np.ascontiguousarray(index_array + first_node, dtype=np.uint32)


def _zero_copy_explicit_connect_enabled(ngpu: NestGpuModule) -> bool:
    raw = os.environ.get(_ZERO_COPY_EXPLICIT_CONNECT_ENV, "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw not in {"1", "true", "yes", "on"}:
        raise ValueError(
            f"{_ZERO_COPY_EXPLICIT_CONNECT_ENV} must be a boolean value"
        )
    return callable(getattr(ngpu, "ConnectGroupGroupUInt32", None))


def _fused_explicit_connect_enabled(ngpu: NestGpuModule) -> bool:
    raw = os.environ.get(_FUSED_EXPLICIT_CONNECT_ENV, "0").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw not in {"1", "true", "yes", "on"}:
        raise ValueError(f"{_FUSED_EXPLICIT_CONNECT_ENV} must be a boolean value")
    return callable(getattr(ngpu, "ConnectExplicitArrays", None))


def _fused_syn_spec_supported(
    syn_spec: Mapping[str, str | int | float | bool],
) -> bool:
    allowed = {"weight", "delay", "receptor", "synapse_group"}
    if set(syn_spec) - allowed:
        return False
    if not all(np.isscalar(value) for value in syn_spec.values()):
        return False
    return all(
        isinstance(syn_spec.get(name, 0), (int, np.integer))
        for name in ("receptor", "synapse_group")
    )


@dataclass(slots=True)
class _ExplicitEdgeBuffer:
    source_chunks: list[npt.NDArray[np.int64]] = dataclass_field(default_factory=list)
    target_chunks: list[npt.NDArray[np.int64]] = dataclass_field(default_factory=list)
    size: int = 0

    def append(
        self,
        source_indices: Sequence[int] | npt.NDArray[np.int64],
        target_indices: Sequence[int] | npt.NDArray[np.int64],
    ) -> None:
        sources = np.asarray(source_indices, dtype=np.int64)
        targets = np.asarray(target_indices, dtype=np.int64)
        if len(sources) != len(targets):
            raise ValueError("explicit source/target chunk lengths differ")
        self.source_chunks.append(sources)
        self.target_chunks.append(targets)
        self.size += len(sources)

    def arrays(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        return (
            np.concatenate(self.source_chunks),
            np.concatenate(self.target_chunks),
        )

    def clear(self) -> None:
        self.source_chunks.clear()
        self.target_chunks.clear()
        self.size = 0


def _connect_explicit_one_to_one(
    ngpu: NestGpuModule,
    pre_nodes: NestGpuNodes,
    post_nodes: NestGpuNodes,
    source_indices: Sequence[int] | npt.NDArray[np.int64],
    target_indices: Sequence[int] | npt.NDArray[np.int64],
    syn_spec: Mapping[str, str | int | float | bool],
    *,
    label: str,
) -> None:
    if len(source_indices) != len(target_indices):
        raise ValueError(f"{label} explicit source/target lengths differ")
    if len(source_indices) == 0:
        return
    try:
        if _fused_explicit_connect_enabled(ngpu) and _fused_syn_spec_supported(
            syn_spec
        ):
            ngpu.ConnectExplicitArrays(
                _explicit_node_ids_uint32(pre_nodes, source_indices),
                _explicit_node_ids_uint32(post_nodes, target_indices),
                syn_spec.get("weight", 0.0),
                syn_spec.get("delay", 0.0),
                syn_spec.get("receptor", 0),
                syn_spec.get("synapse_group", 0),
            )
        elif _zero_copy_explicit_connect_enabled(ngpu):
            ngpu.ConnectGroupGroupUInt32(
                _explicit_node_ids_uint32(pre_nodes, source_indices),
                _explicit_node_ids_uint32(post_nodes, target_indices),
                {"rule": "one_to_one"},
                syn_spec,
            )
        else:
            ngpu.Connect(
                _explicit_node_ids(pre_nodes, source_indices),
                _explicit_node_ids(post_nodes, target_indices),
                {"rule": "one_to_one"},
                syn_spec,
            )
    except Exception as exc:  # pragma: no cover - exercised with real NEST-GPU
        raise RuntimeError(f"Explicit 3-D connect failed {label}: {exc}") from exc


def _projection_syn_spec(
    spec: NetworkSpec,
    projection: object,
) -> dict[str, str | int | float | bool]:
    from ca1.types import Projection

    if not isinstance(projection, Projection):
        raise TypeError("projection plan contains a non-Projection value")
    receptors = spec.receptors_for_post(projection.post)
    port_idx = _port_index_for_post(
        receptors,
        post=projection.post,
        receptor=projection.receptor,
    )
    weight_nS = (
        nonnegative_weight_nS(
            projection.weight_nS,
            label=f"projection {projection.pre}->{projection.post}",
        )
        * projection.synapses_per_connection
        * spec.weight_compensation
        * _post_current_gain(spec, projection.post)
    )
    return {
        "weight": weight_nS,
        "delay": projection.delay_ms,
        "receptor": port_idx,
    }


def _connect_recurrent_3d_gaussian(
    ngpu: NestGpuModule,
    spec: NetworkSpec,
    nodes_by_type: Mapping[str, NestGpuNodes],
    n_cells: Mapping[str, int],
    topology: ModelDbFastconn3D | EdgeArtifact,
    working_point_rates: Mapping[str, float],
) -> None:
    active = [
        projection
        for projection in spec.projections
        if projection.pre in nodes_by_type
        and projection.post in nodes_by_type
        and projection.post not in working_point_rates
    ]
    for plan in recurrent_projection_plans(active, n_cells):
        buffers: dict[int, _ExplicitEdgeBuffer] = {}
        projections_by_id: dict[int, object] = {}

        def flush(port_id: int) -> None:
            buffer = buffers[port_id]
            source_indices, target_indices = buffer.arrays()
            projection = projections_by_id[port_id]
            _connect_explicit_one_to_one(
                ngpu,
                nodes_by_type[plan.pre],
                nodes_by_type[plan.post],
                source_indices,
                target_indices,
                _projection_syn_spec(spec, projection),
                label=(
                    f"{plan.pre}->{plan.post}:"
                    f"{getattr(projection, 'receptor', 'unknown')}"
                ),
            )
            buffer.clear()

        for post_edges in topology.iter_post_edges(
            pre_type=plan.pre,
            post_type=plan.post,
            indegree=plan.indegree,
            seed=spec.seed,
            projection=f"recurrent:{plan.pre}->{plan.post}",
        ):
            for assignment in projection_port_edges(plan, post_edges):
                port_id = id(assignment.projection)
                projections_by_id[port_id] = assignment.projection
                buffer = buffers.setdefault(port_id, _ExplicitEdgeBuffer())
                buffer.append(
                    assignment.source_indices,
                    assignment.target_indices,
                )
                if buffer.size >= _EXPLICIT_EDGE_CHUNK_SIZE:
                    flush(port_id)
        for port_id, buffer in buffers.items():
            if buffer.size:
                flush(port_id)


def _connect_afferent_3d_gaussian(
    ngpu: NestGpuModule,
    spec: NetworkSpec,
    topology: ModelDbFastconn3D | EdgeArtifact,
    source: str,
    afferent: Afferent,
    source_nodes: NestGpuNodes,
    target_nodes: NestGpuNodes,
    indegree: int,
    syn_spec: Mapping[str, str | int | float | bool],
) -> None:
    buffer = _ExplicitEdgeBuffer()
    for post_edges in topology.iter_post_edges(
        pre_type=source,
        post_type=afferent.post,
        indegree=indegree,
        seed=spec.seed,
        projection=f"afferent:{afferent.name}",
    ):
        buffer.append(
            post_edges.source_indices,
            np.full(
                len(post_edges.source_indices),
                post_edges.post_index,
                dtype=np.int64,
            ),
        )
        if buffer.size >= _EXPLICIT_EDGE_CHUNK_SIZE:
            source_indices, target_indices = buffer.arrays()
            _connect_explicit_one_to_one(
                ngpu,
                source_nodes,
                target_nodes,
                source_indices,
                target_indices,
                syn_spec,
                label=f"{afferent.name}->{afferent.post}",
            )
            buffer.clear()
    if buffer.size:
        source_indices, target_indices = buffer.arrays()
        _connect_explicit_one_to_one(
            ngpu,
            source_nodes,
            target_nodes,
            source_indices,
            target_indices,
            syn_spec,
            label=f"{afferent.name}->{afferent.post}",
        )


def _source_env_suffix(source: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in source.upper())


def _afferent_source_name(afferent_name: str) -> str:
    return afferent_name.split("_to_", maxsplit=1)[0]


def _afferent_topology(spec: NetworkSpec) -> _AfferentTopology:
    env_topology = os.environ.get(_AFFERENT_TOPOLOGY_ENV)
    if env_topology is not None and env_topology != spec.afferent_topology:
        raise ValueError(
            f"{_AFFERENT_TOPOLOGY_ENV}={env_topology!r} conflicts with "
            f"NetworkSpec.afferent_topology={spec.afferent_topology!r}; "
            "put diagnostic afferent topology in the config/spec so provenance "
            "can audit it"
        )
    raw = env_topology if env_topology is not None else spec.afferent_topology
    match raw:
        case "compound":
            return _AFFERENT_TOPOLOGY_COMPOUND
        case "source_pool":
            return _AFFERENT_TOPOLOGY_SOURCE_POOL
        case "literal_source_graph":
            return _AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH
        case "literal_source_graph_binned":
            return _AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH_BINNED
        case _:
            message = " ".join(
                (
                    f"unknown {_AFFERENT_TOPOLOGY_ENV}={raw!r};",
                    "expected 'compound', 'source_pool',",
                    "'literal_source_graph', or",
                    "'literal_source_graph_binned'",
                )
            )
            raise ValueError(message)


def _source_pool_size(aff: Afferent, source: str, spec: NetworkSpec) -> int:
    specific = f"{_SOURCE_POOL_SIZE_ENV}_{_source_env_suffix(source)}"
    requested = _env_positive_int(
        specific,
        _env_positive_int(
            _SOURCE_POOL_SIZE_ENV,
            int(spec.afferent_source_pool_size),
        ),
    )
    return min(requested, max(1, int(aff.n_source)))


def _source_pool_indegree(source: str, pool_size: int, spec: NetworkSpec) -> int:
    specific = f"{_SOURCE_POOL_INDEGREE_ENV}_{_source_env_suffix(source)}"
    requested = _env_positive_int(
        specific,
        _env_positive_int(
            _SOURCE_POOL_INDEGREE_ENV,
            int(spec.afferent_source_pool_indegree),
        ),
    )
    return min(requested, pool_size)


def _source_pool_rate_hz(aff: Afferent, pool_size: int, indegree: int) -> float:
    _ = pool_size
    return aff.rate_hz * aff.synapses_per_cell / float(indegree)


def _source_pool_weight_nS(aff: Afferent) -> float:
    return nonnegative_weight_nS(
        aff.weight_nS,
        label=f"afferent {aff.name}",
    )


def _source_pool_local_nodes_required(
    spec: NetworkSpec,
    n_cells: Mapping[str, int],
) -> int:
    total = sum(int(count) for count in n_cells.values())
    for aff in spec.afferents:
        if aff.post not in n_cells:
            continue
        source = _afferent_source_name(aff.name)
        total += _source_pool_size(aff, source, spec)
    return total


def _validate_source_pool_node_budget(
    spec: NetworkSpec,
    n_cells: Mapping[str, int],
) -> None:
    required = _source_pool_local_nodes_required(spec, n_cells)
    if required <= _MAX_NESTGPU_LOCAL_NODES:
        return
    raise ValueError(
        "source_pool path-specific pools require "
        f"{required} local nodes, exceeding the NEST-GPU local node cap "
        f"{_MAX_NESTGPU_LOCAL_NODES}; use compound topology or reduce "
        "afferent_source_pool_size with diagnostic provenance"
    )


def _literal_source_graph_local_nodes_required(
    spec: NetworkSpec,
    n_cells: Mapping[str, int],
) -> int:
    source_counts: dict[str, int] = {}
    for aff in spec.afferents:
        if aff.post not in n_cells:
            continue
        source = _afferent_source_name(aff.name)
        source_counts[source] = max(source_counts.get(source, 0), int(aff.n_source))
    return sum(int(count) for count in n_cells.values()) + sum(source_counts.values())


def _validate_literal_source_graph_node_budget(
    spec: NetworkSpec,
    n_cells: Mapping[str, int],
) -> None:
    required = _literal_source_graph_local_nodes_required(spec, n_cells)
    if required <= _MAX_NESTGPU_LOCAL_NODES:
        return
    raise ValueError(
        "literal_source_graph requires "
        f"{required} local nodes, exceeding the NEST-GPU local node cap "
        f"{_MAX_NESTGPU_LOCAL_NODES}"
    )


def _literal_source_graph_indegree(aff: Afferent) -> int:
    if aff.synapses_per_connection < 1:
        raise ValueError(
            f"{aff.name} synapses_per_connection must be positive, "
            f"got {aff.synapses_per_connection}"
        )
    contact_count = aff.synapses_per_cell / float(aff.synapses_per_connection)
    indegree = int(round(contact_count))
    if indegree < 1:
        raise ValueError(f"{aff.name} literal source contact indegree is < 1")
    if not np.isclose(contact_count, float(indegree), rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{aff.name} synapses_per_cell={aff.synapses_per_cell} is not an "
            f"integer multiple of synapses_per_connection={aff.synapses_per_connection}"
        )
    if indegree > aff.n_source:
        raise ValueError(
            f"{aff.name} literal source contact indegree {indegree} exceeds "
            f"source count {aff.n_source}"
        )
    return indegree


def _literal_source_graph_weight_nS(aff: Afferent) -> float:
    return (
        nonnegative_weight_nS(aff.weight_nS, label=f"afferent {aff.name}")
        * float(aff.synapses_per_connection)
    )


def _post_current_gain(spec: NetworkSpec, cell_type: str) -> float:
    match spec.neuron_model:
        case "aeif_cond_beta_multisynapse":
            return 1.0
        case "izhikevich_cond_beta":
            return izhikevich_params_for_cell_type(cell_type).current_gain
        case "aglif_cond_beta":
            return 1.0
        case "aglif_dend_cond_beta":
            return 1.0
        case _:
            raise ValueError(
                f"NestGpuBackend unsupported neuron_model {spec.neuron_model!r}; "
                f"expected one of {SUPPORTED_NEURON_MODELS}"
            )


def _required_dendritic_ports(spec: NetworkSpec, cell_type: str) -> frozenset[str]:
    ports = {
        projection.receptor
        for projection in spec.projections
        if projection.post == cell_type and projection.receptor.endswith("__dend")
    }
    ports.update(
        afferent.receptor
        for afferent in spec.afferents
        if afferent.post == cell_type and afferent.receptor.endswith("__dend")
    )
    return frozenset(ports)


def _nestgpu_model_name(spec: NetworkSpec) -> str:
    match spec.neuron_model:
        case "aglif_cond_beta":
            return "user_m1"
        case "aglif_dend_cond_beta":
            return "user_m2"
        case "aeif_cond_beta_multisynapse" | "izhikevich_cond_beta":
            return spec.neuron_model
        case _:
            raise ValueError(
                f"NestGpuBackend unsupported neuron_model {spec.neuron_model!r}; "
                f"expected one of {SUPPORTED_NEURON_MODELS}"
            )


def _nestgpu_model_name_for_cell(spec: NetworkSpec, cell_type: str) -> str:
    base = _nestgpu_model_name(spec)
    if base != "user_m2":
        return base
    override = spec.aglif_dend_overrides.get(cell_type)
    if override is not None and override.model in {"user_m3", "user_m4", "user_m5", "user_m7"}:
        return override.model
    return base


def _neuron_status(
    spec: NetworkSpec,
    cell_type: str,
    ct_params: NeuronParams,
) -> dict[str, float]:
    match spec.neuron_model:
        case "aeif_cond_beta_multisynapse":
            return _status_with_initial_v_m(ct_params.as_nest())
        case "izhikevich_cond_beta":
            return izhikevich_params_for_cell_type(cell_type).as_nest()
        case "aglif_cond_beta":
            return _status_with_initial_v_m(
                aglif_params_for_cell_type(cell_type).as_nest()
            )
        case "aglif_dend_cond_beta":
            override = spec.aglif_dend_overrides.get(cell_type)
            gc_scale = (
                override.g_c_scale
                if override is not None
                else spec.aglif_gc_scale_overrides.get(cell_type, 1.0)
            )
            status = _status_with_initial_v_m(_aglif_dend_status(
                cell_type,
                gc_scale,
            ))
            if override is not None and override.model == "user_m3":
                from ca1.sim.aglif_dend import cck_user_m3_status
                status.update(cck_user_m3_status(status["E_L"]))
            if override is not None and override.model == "user_m4":
                from ca1.sim.aglif_dend import user_m4_status
                status.update(user_m4_status(cell_type, status["E_L"]))
            if override is not None and override.model == "user_m5":
                from ca1.sim.aglif_dend import user_m5_status
                status.update(user_m5_status(cell_type, status["E_L"]))
            if override is not None and override.model == "user_m7":
                from ca1.sim.aglif_dend import user_m7_status
                status.update(user_m7_status(cell_type, status["E_L"]))
            status.update(spec.aglif_status_overrides.get(cell_type, {}))
            return status
        case _:
            raise ValueError(
                f"NestGpuBackend unsupported neuron_model {spec.neuron_model!r}; "
                f"expected one of {SUPPORTED_NEURON_MODELS}"
            )


def _status_with_initial_v_m(status: Mapping[str, float]) -> dict[str, float]:
    explicit = dict(status)
    if "V_m" in explicit:
        return explicit
    try:
        explicit["V_m"] = explicit["E_L"]
    except KeyError as exc:
        raise KeyError(
            "E_L is required to derive explicit initial V_m for NEST-GPU status"
        ) from exc
    return explicit


def _can_collect_modeldb_n_pole_lfp(
    spec: NetworkSpec | None,
    n_cells: Mapping[str, int],
) -> bool:
    return (
        spec is not None
        and spec.neuron_model == "aglif_dend_cond_beta"
        and spec.scale >= 0.999
        and spec.cellnumbers_index == 101
        and int(n_cells.get("Pyramidal", 0)) == _MODELDB_FULL_PYRAMIDAL_COUNT
    )


def _spatially_distributed_lfp_sample_indices(
    positions_um: npt.NDArray[np.float64],
    sample_count: int,
    *,
    seed: int,
) -> npt.NDArray[np.int64]:
    """Select deterministic x-stratified cells, prioritizing the electrode ROI."""
    positions = np.asarray(positions_um, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_um must have shape (n_cells, 3)")
    if not bool(np.isfinite(positions).all()):
        raise ValueError("positions_um must contain only finite values")
    if sample_count < 0:
        raise ValueError("sample_count must be nonnegative")
    count = min(int(sample_count), positions.shape[0])
    if count == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)

    def x_stratified(
        candidates: npt.NDArray[np.int64],
        n_select: int,
    ) -> npt.NDArray[np.int64]:
        if n_select >= candidates.size:
            return candidates.copy()
        ordered = candidates[
            np.lexsort((candidates, positions[candidates, 0]))
        ]
        strata = np.array_split(ordered, n_select)
        return np.asarray(
            [stratum[rng.integers(0, stratum.size)] for stratum in strata],
            dtype=np.int64,
        )

    roi_mask = electrode_roi_mask(positions, MODELDB_NPOLE_ELECTRODE_ROI)
    in_roi = np.flatnonzero(roi_mask).astype(np.int64)
    if in_roi.size >= count:
        selected = x_stratified(in_roi, count)
    else:
        outside_roi = np.flatnonzero(~roi_mask).astype(np.int64)
        selected = np.concatenate(
            (in_roi, x_stratified(outside_roi, count - in_roi.size))
        )
    return selected[
        np.lexsort((selected, positions[selected, 0]))
    ].astype(np.int64, copy=False)


def _meta_with_lfp_proxy(
    meta: SimMeta,
    lfp_proxy: str,
    lfp_provenance: Mapping[str, str] | None = None,
) -> SimMeta:
    if lfp_proxy != _LFP_PROXY_MODELDB_N_POLE_REDUCED:
        return replace(meta, lfp_proxy=lfp_proxy)
    provenance = dict(meta.parameter_provenance)
    provenance[_LFP_MODELDB_N_POLE_PROVENANCE_KEY] = (
        _LFP_MODELDB_N_POLE_PROVENANCE_VALUE
    )
    if lfp_provenance is not None:
        provenance.update(dict(lfp_provenance))
    return replace(
        meta,
        lfp_proxy=lfp_proxy,
        parameter_provenance=provenance,
    )


class NestGpuBackend(SimulatorBackend):
    """NEST GPU multi-GPU backend.

    Lifecycle::

        b = NestGpuBackend()
        b.setup(dt_ms=0.1, seed=12345)
        b.build(spec, n_cells)
        b.attach_recorders()
        b.run(duration_ms=10_000.0)
        spikes = b.collect_spikes()
    """

    name: str = "nestgpu"

    def __init__(self) -> None:
        self._ngpu: Optional[NestGpuModule] = None       # bound lazily in setup()
        self._nodes: dict[str, NestGpuNodes] = {}
        self._poisson: dict[str, NestGpuNodes] = {}
        self._recorders: dict[str, NestGpuNodes] = {}
        self._n_cells: dict[str, int] = {}
        self._rank_counts: dict[str, list[int]] = {}
        self._node_shards: dict[str, list[NestGpuNodes]] = {}
        self._literal_source_rates_hz: dict[str, float] = {}
        self._working_point_rates_hz: dict[str, float] = {}
        self._working_point_spikes_ms: dict[
            str, list[npt.NDArray[np.float64]]
        ] = {}
        self._spec: Optional[NetworkSpec] = None
        self._lfp_record: int | None = None
        self._lfp_sample_count: int = 0
        self._lfp_record_every: int = _DEFAULT_LFP_RECORD_EVERY
        self._lfp_g_columns: dict[int, _LfpConductanceColumn] = {}
        self._lfp_vm_columns: list[int] = []
        self._lfp_vd_columns: list[int] = []
        self._lfp_vdist_columns: list[int] = []
        self._lfp_sample_indices = np.empty(0, dtype=np.int64)
        self._lfp_sample_positions_um = np.empty((0, 3), dtype=np.float64)
        self._last_lfp_proxy: str = "unrecorded"
        self._last_lfp_provenance: dict[str, str] = {}
        self._dt_ms: float = 0.1
        self._mpi_rank: int = 0
        self._mpi_size: int = 1
        self._host_group: int = 0
        self._max_rec_spikes: int = self.MAX_REC_SPIKES

    # ------------------------------------------------------------------
    # SimulatorBackend interface
    # ------------------------------------------------------------------

    def setup(self, dt_ms: float = 0.1, seed: int = 12345,
              n_threads: int = 1) -> None:
        """Reset the NEST GPU kernel; bind the lazy ``nestgpu`` import.

        ``n_threads`` is ignored (GPU-resident; thread count irrelevant).
        ``ngpu.SetKernelStatus`` does not exist in NEST GPU; kernel parameters
        are set via ``ngpu.SetRandomSeed`` and ``ngpu.SetTimeResolution``.
        """
        # libinfinipath (pulled in by nestgpu via MPI/PSM) is not fork-safe and
        # its on_exit handler segfaults at teardown whenever a multiprocessing
        # forkserver/pool coexists in this process -- even with zero live pool
        # workers.  So the 3-D topology edge generator runs single-process
        # (serial) inside the GPU backend by default, matching the pre-pool code
        # path.  Only boot the forkserver when the operator explicitly opts into
        # the parallel pool via CA1_TOPOLOGY_MAX_WORKERS > 1, and only then
        # before nestgpu pulls in MPI/libinfinipath.
        if int(os.environ.get("CA1_TOPOLOGY_MAX_WORKERS", "1")) > 1:
            boot_modeldb_topology_forkserver()
        import nestgpu as raw_ngpu  # noqa: PLC0415

        ngpu = nestgpu_module(raw_ngpu)
        self._ngpu = ngpu
        self._dt_ms = dt_ms

        self._mpi_rank, self._mpi_size = _mpi_rank_size_from_env()
        if self._mpi_size > 1:
            ngpu.ConnectMpiInit()
            self._mpi_rank = ngpu.HostId()
            self._mpi_size = ngpu.HostNum()
            self._host_group = ngpu.CreateHostGroup(list(range(self._mpi_size)))
        else:
            self._host_group = 0

        ngpu.SetRandomSeed(seed + self._mpi_rank)
        ngpu.SetTimeResolution(dt_ms)

        self._nodes = {}
        self._poisson = {}
        self._recorders = {}
        self._n_cells = {}
        self._rank_counts = {}
        self._node_shards = {}
        self._literal_source_rates_hz = {}
        self._working_point_rates_hz = {}
        self._working_point_spikes_ms = {}
        self._spec = None
        self._lfp_record = None
        self._lfp_sample_count = 0
        self._lfp_record_every = _DEFAULT_LFP_RECORD_EVERY
        self._lfp_g_columns = {}
        self._lfp_vm_columns = []
        self._lfp_vd_columns = []
        self._lfp_vdist_columns = []
        self._lfp_sample_indices = np.empty(0, dtype=np.int64)
        self._lfp_sample_positions_um = np.empty((0, 3), dtype=np.float64)
        self._last_lfp_proxy = "unrecorded"
        self._last_lfp_provenance = {}
        self._max_rec_spikes = self.MAX_REC_SPIKES

    def build(self, spec: NetworkSpec, n_cells: Mapping[str, int]) -> None:
        """Create all 9 populations, recurrent projections, and Poisson afferents.

        Receptor ports are 0-based.  ``spec.weight_compensation`` is applied to
        recurrent weights.  Afferent ``synapses_per_cell`` is kept verbatim and
        NOT capped to presynaptic population size (see types.py Afferent note).
        """
        if self._ngpu is None:
            raise RuntimeError("Call setup() before build().")

        ngpu = self._ngpu
        model_name = _nestgpu_model_name(spec)
        _validate_receptor_port_budget(spec, model_name)
        topology = _afferent_topology(spec)
        if (
            spec.recurrent_topology == "modeldb_fastconn_3d_gaussian"
            and self._mpi_size > 1
        ):
            raise ValueError(
                "recurrent_topology='modeldb_fastconn_3d_gaussian' currently "
                "requires the canonical single-GPU deployment"
            )
        working_point_rates = (
            dict(spec.working_point_clamp_rates_hz)
            if spec.working_point_mode == "clamp"
            else {}
        )
        if working_point_rates and self._mpi_size > 1:
            raise ValueError(
                "working_point_mode='clamp' is only implemented for "
                "single-GPU runs"
            )
        if topology == _AFFERENT_TOPOLOGY_SOURCE_POOL and self._mpi_size > 1:
            raise ValueError(
                "CA1_AFFERENT_TOPOLOGY=source_pool is only implemented for "
                "single-GPU runs"
            )
        if topology in {
            _AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH,
            _AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH_BINNED,
        } and self._mpi_size > 1:
            raise ValueError(
                f"CA1_AFFERENT_TOPOLOGY={topology} is only implemented for "
                "single-GPU runs"
            )
        if self._mpi_size > 1:
            match spec.recurrent_topology:
                case "modeldb_fastconn_binned" | "modeldb_fastconn_gaussian_binned":
                    message = " ".join(
                        (
                            f"recurrent_topology={spec.recurrent_topology!r} is only",
                            "implemented for single-GPU final-equivalence runs; MPI",
                            "recurrent sharding remains a diagnostic benchmark path",
                        )
                    )
                    raise ValueError(message)
                case _:
                    pass
        if self._mpi_size > 1 and any(
            override.model in {"user_m3", "user_m4", "user_m5", "user_m7"}
            for override in spec.aglif_dend_overrides.values()
        ):
            raise ValueError("user_m3/user_m4/user_m5/user_m7 validation is single-GPU only; MPI is forbidden")
        if topology == _AFFERENT_TOPOLOGY_SOURCE_POOL:
            _validate_source_pool_node_budget(spec, n_cells)
        if topology in {
            _AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH,
            _AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH_BINNED,
        }:
            _validate_literal_source_graph_node_budget(spec, n_cells)
        self._spec = spec
        self._working_point_rates_hz = working_point_rates
        self._working_point_spikes_ms = {
            cell_type: [
                np.empty(0, dtype=np.float64)
                for _ in range(int(n_cells[cell_type]))
            ]
            for cell_type in working_point_rates
        }
        self._n_cells = {}
        self._rank_counts = {
            ct_name: _partition_counts(int(count), self._mpi_size)
            for ct_name, count in n_cells.items()
        }
        if self._mpi_size > 1:
            undersized = [
                ct_name for ct_name, count in n_cells.items()
                if int(count) < self._mpi_size
            ]
            if undersized:
                raise ValueError(
                    "Distributed NEST GPU runs require every cell type count "
                    f"to be >= mpi_size={self._mpi_size}; undersized={undersized}"
                )

        # 1. Populations -----------------------------------------------
        for ct_name, ct in spec.cell_types.items():
            # Receptor count is a Create() argument in NEST GPU (not a SetStatus
            # param).  Scalar AdEx params and the per-port receptor arrays are set
            # in two SetStatus calls.
            count = int(n_cells[ct_name])
            local_count = self._rank_counts[ct_name][self._mpi_rank]
            if ct_name in working_point_rates:
                nodes = ngpu.Create("spike_generator", count)
                self._poisson[f"working_point:{ct_name}"] = nodes
                self._node_shards[ct_name] = [nodes]
                self._nodes[ct_name] = nodes
                self._n_cells[ct_name] = local_count
                continue
            status = _neuron_status(spec, ct_name, ct.params)
            cell_model_name = _nestgpu_model_name_for_cell(spec, ct_name)
            receptors = spec.receptors_for_post(ct_name)
            n_ports = receptors.n_ports()
            receptor_status = {
                "E_rev": list(receptors.E_rev),
                "tau_rise": list(receptors.tau_rise),
                "tau_decay": list(receptors.tau_decay),
            }
            if cell_model_name in {"user_m2", "user_m3", "user_m4", "user_m5", "user_m7"}:
                override = spec.aglif_dend_overrides.get(ct_name)
                receptor_status["compartment"] = _aglif_dend_compartments(
                    receptors.names,
                    ct_name,
                    _required_dendritic_ports(spec, ct_name),
                    spec.source_location_transfer_table,
                    spec.aglif_receive_domain_overrides,
                    spec.aglif_compartment_overrides,
                    receive_domain=(
                        None if override is None else override.receive_domain
                    ),
                )
            if self._mpi_size > 1:
                shards: list[NestGpuNodes] = []
                for host, host_count in enumerate(self._rank_counts[ct_name]):
                    remote = ngpu.RemoteCreate(
                        host, cell_model_name, host_count, n_ports
                    )
                    ngpu.SetStatus(remote, status)
                    ngpu.SetStatus(remote, receptor_status)
                    _set_intrinsic_heterogeneity(
                        ngpu,
                        remote,
                        cell_type=ct_name,
                        params=ct.params,
                        baseline_status=status,
                        count=host_count,
                        seed=spec.seed,
                        shard=host,
                    )
                    shards.append(remote.node_seq)
                self._node_shards[ct_name] = shards
                self._nodes[ct_name] = shards[self._mpi_rank]
            else:
                nodes = ngpu.Create(cell_model_name, count, n_ports)
                ngpu.SetStatus(nodes, status)
                ngpu.SetStatus(nodes, receptor_status)
                _set_intrinsic_heterogeneity(
                    ngpu,
                    nodes,
                    cell_type=ct_name,
                    params=ct.params,
                    baseline_status=status,
                    count=count,
                    seed=spec.seed,
                    shard=0,
                )
                self._node_shards[ct_name] = [nodes]
                self._nodes[ct_name] = nodes
            self._n_cells[ct_name] = local_count

        topology_3d: ModelDbFastconn3D | EdgeArtifact | None = None
        if spec.recurrent_topology == "modeldb_fastconn_3d_gaussian":
            connectivity_counts = {
                cell_type: int(count) for cell_type, count in n_cells.items()
            }
            for afferent in spec.afferents:
                source = _afferent_source_name(afferent.name)
                previous = connectivity_counts.get(source)
                if previous is not None and previous != int(afferent.n_source):
                    raise ValueError(
                        f"3-D topology source {source!r} has inconsistent counts "
                        f"{previous} and {afferent.n_source}"
                    )
                connectivity_counts[source] = int(afferent.n_source)
            topology_3d = load_edge_artifact_from_env(spec, n_cells)
            if topology_3d is None:
                topology_3d = ModelDbFastconn3D(
                    modeldb_connectivity_positions(connectivity_counts),
                    max_workers=int(os.environ.get("CA1_TOPOLOGY_MAX_WORKERS", "1")),
                )
            _connect_recurrent_3d_gaussian(
                ngpu,
                spec,
                self._nodes,
                n_cells,
                topology_3d,
                working_point_rates,
            )

        # 2. Recurrent connections -------------------------------------
        wcomp = spec.weight_compensation
        for proj in spec.projections:
            if spec.recurrent_topology == "modeldb_fastconn_3d_gaussian":
                continue  # connected above from one shared biological edge set
            if proj.pre not in self._nodes or proj.post not in self._nodes:
                continue  # absent from this MPI rank's partition
            if proj.post in working_point_rates:
                continue  # minimal clamp has no postsynaptic dynamics

            post_receptors = spec.receptors_for_post(proj.post)
            port_idx = _port_index_for_post(
                post_receptors,
                post=proj.post,
                receptor=proj.receptor,
            )
            # Preserve in-degree (matches NestBackend); clamp to pre-pop size so
            # fixed_indegree never asks for more presynaptic cells than exist.
            pre_count = int(n_cells[proj.pre])
            indegree = min(max(1, int(round(proj.indegree))), pre_count)
            # Always positive; inhibition sign encoded by E_rev on port.
            weight_nS = (
                nonnegative_weight_nS(
                    proj.weight_nS,
                    label=f"projection {proj.pre}->{proj.post}",
                )
                * proj.synapses_per_connection
                * wcomp
                * _post_current_gain(spec, proj.post)
            )
            syn_spec = {
                "weight": weight_nS,
                "delay": proj.delay_ms,
                "receptor": port_idx,
            }

            match spec.recurrent_topology:
                case "fixed_indegree":
                    if self._mpi_size > 1:
                        try:
                            hosts = list(range(self._mpi_size))
                            ngpu.ConnectDistributedFixedIndegree(
                                hosts,
                                self._node_shards[proj.pre],
                                hosts,
                                self._node_shards[proj.post],
                                indegree,
                                self._host_group,
                                syn_spec,
                            )
                        except Exception as exc:  # pragma: no cover
                            raise RuntimeError(
                                f"Distributed connect failed {proj.pre}->{proj.post}: {exc}"
                            ) from exc
                    else:
                        try:
                            ngpu.Connect(
                                self._nodes[proj.pre],
                                self._nodes[proj.post],
                                {"rule": "fixed_indegree", "indegree": indegree},
                                syn_spec,
                            )
                        except Exception as exc:  # pragma: no cover
                            raise RuntimeError(
                                f"Connect failed {proj.pre}->{proj.post}: {exc}"
                            ) from exc
                case "modeldb_fastconn_binned" | "modeldb_fastconn_gaussian_binned":
                    if spec.recurrent_topology == "modeldb_fastconn_binned":
                        binned_connections = binned_fixed_indegree_connections
                    else:
                        binned_connections = gaussian_binned_fixed_indegree_connections
                    for conn in binned_connections(
                        pre_type=proj.pre,
                        post_type=proj.post,
                        pre_count=pre_count,
                        post_count=int(n_cells[proj.post]),
                        indegree=indegree,
                    ):
                        try:
                            ngpu.Connect(
                                self._nodes[proj.pre][
                                    conn.source_start:conn.source_stop
                                ],
                                self._nodes[proj.post][
                                    conn.target_start:conn.target_stop
                                ],
                                {
                                    "rule": "fixed_indegree",
                                    "indegree": conn.indegree,
                                },
                                syn_spec,
                            )
                        except Exception as exc:  # pragma: no cover
                            message = " ".join(
                                (
                                    "Binned fastconn connect failed",
                                    f"{proj.pre}->{proj.post}",
                                    f"sources={conn.source_start}:{conn.source_stop}",
                                    f"targets={conn.target_start}:{conn.target_stop}:",
                                    str(exc),
                                )
                            )
                            raise RuntimeError(
                                message
                            ) from exc

        # 3. Poisson afferents -----------------------------------------
        literal_source_pools: dict[str, NestGpuNodes] = {}
        literal_source_sizes: dict[str, int] = {}
        literal_source_rates: dict[str, float] = {}
        for aff in spec.afferents:
            if aff.post not in self._nodes:
                continue
            if aff.post in working_point_rates:
                continue  # minimal clamp has no postsynaptic dynamics

            post_receptors = spec.receptors_for_post(aff.post)
            port_idx = _port_index_for_post(
                post_receptors,
                post=aff.post,
                receptor=aff.receptor,
            )
            match topology:
                case "compound":
                    pgen = ngpu.Create("poisson_generator", self._n_cells[aff.post])
                    drive = afferent_poisson_drive(aff)
                    ngpu.SetStatus(pgen, {"rate": drive.rate_hz})
                    self._poisson[aff.name] = pgen
                    conn_spec = {"rule": "one_to_one"}
                    weight_nS = drive.weight_nS
                case "source_pool":
                    source = _afferent_source_name(aff.name)
                    pool_key = f"source_pool:{aff.name}"
                    pool_size = _source_pool_size(aff, source, spec)
                    indegree = _source_pool_indegree(source, pool_size, spec)
                    rate_hz = _source_pool_rate_hz(aff, pool_size, indegree)
                    if pool_key in self._poisson:
                        raise ValueError(
                            f"duplicate source_pool afferent key {pool_key!r}"
                        )
                    pgen = ngpu.Create("poisson_generator", pool_size)
                    ngpu.SetStatus(pgen, {"rate": rate_hz})
                    self._poisson[pool_key] = pgen
                    conn_spec = {"rule": "fixed_indegree", "indegree": indegree}
                    weight_nS = _source_pool_weight_nS(aff)
                case "literal_source_graph" | "literal_source_graph_binned":
                    source = _afferent_source_name(aff.name)
                    source_size = int(aff.n_source)
                    existing_size = literal_source_sizes.get(source)
                    if existing_size is not None and existing_size != source_size:
                        raise ValueError(
                            f"literal_source_graph source {source!r} has "
                            f"inconsistent n_source values: {existing_size} "
                            f"and {source_size}"
                        )
                    literal_source_sizes[source] = source_size
                    existing_rate = literal_source_rates.get(source)
                    if existing_rate is not None and not np.isclose(
                        existing_rate,
                        aff.rate_hz,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"literal_source_graph source {source!r} has "
                            f"inconsistent rates: {existing_rate} and "
                            f"{aff.rate_hz}"
                        )
                    literal_source_rates[source] = aff.rate_hz
                    pgen = literal_source_pools.get(source)
                    if pgen is None:
                        pgen = ngpu.Create("spike_generator", source_size)
                        literal_source_pools[source] = pgen
                        self._poisson[f"literal_source:{source}"] = pgen
                    indegree = _literal_source_graph_indegree(aff)
                    weight_nS = _literal_source_graph_weight_nS(aff)
                    if topology_3d is not None:
                        syn_spec = {
                            "weight": weight_nS * _post_current_gain(spec, aff.post),
                            "delay": aff.delay_ms,
                            "receptor": port_idx,
                        }
                        _connect_afferent_3d_gaussian(
                            ngpu,
                            spec,
                            topology_3d,
                            source,
                            aff,
                            pgen,
                            self._nodes[aff.post],
                            indegree,
                            syn_spec,
                        )
                        continue
                    if topology == _AFFERENT_TOPOLOGY_LITERAL_SOURCE_GRAPH_BINNED:
                        syn_spec = {
                            "weight": weight_nS * _post_current_gain(spec, aff.post),
                            "delay": aff.delay_ms,
                            "receptor": port_idx,
                        }
                        for conn in gaussian_binned_fixed_indegree_connections(
                            pre_type=source,
                            post_type=aff.post,
                            pre_count=source_size,
                            post_count=int(n_cells[aff.post]),
                            indegree=indegree,
                        ):
                            try:
                                ngpu.Connect(
                                    pgen[conn.source_start:conn.source_stop],
                                    self._nodes[aff.post][
                                        conn.target_start:conn.target_stop
                                    ],
                                    {
                                        "rule": "fixed_indegree",
                                        "indegree": conn.indegree,
                                    },
                                    syn_spec,
                                )
                            except Exception as exc:
                                message = " ".join((
                                    "Binned literal afferent connect failed",
                                    f"{aff.name}->{aff.post}",
                                    f"sources={conn.source_start}:{conn.source_stop}",
                                    f"targets={conn.target_start}:{conn.target_stop}:",
                                    str(exc),
                                ))
                                raise RuntimeError(message) from exc
                        continue
                    conn_spec = {"rule": "fixed_indegree", "indegree": indegree}

            try:
                ngpu.Connect(
                    pgen, self._nodes[aff.post],
                    conn_spec,
                    {
                        "weight": weight_nS * _post_current_gain(spec, aff.post),
                        "delay": aff.delay_ms,
                        "receptor": port_idx,
                    },
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Connect afferent {aff.name}->{aff.post}: {exc}"
                ) from exc
        self._literal_source_rates_hz = literal_source_rates

    #: per-cell spike-time buffer (full-scale: 338740 * 4096 * 4B ~= 5.5 GB).
    MAX_REC_SPIKES: int = 4096

    def attach_recorders(
        self, record_types: Optional[Iterable[str]] = None
    ) -> None:
        if self._ngpu is None:
            raise RuntimeError("Call setup() before attach_recorders().")

        ngpu = self._ngpu
        targets = set(record_types) if record_types is not None else set(self._nodes)
        unknown = sorted(targets - set(self._nodes))
        if unknown:
            raise KeyError(
                "record_types contains unknown cell types "
                f"{unknown}; available={list(self._nodes.keys())}"
            )
        self._lfp_record = None
        self._lfp_sample_count = 0
        self._lfp_record_every = _env_positive_int(
            _LFP_RECORD_EVERY_ENV,
            _DEFAULT_LFP_RECORD_EVERY,
        )
        self._lfp_g_columns = {}
        self._lfp_vm_columns = []
        self._lfp_vd_columns = []
        self._lfp_vdist_columns = []
        self._lfp_sample_indices = np.empty(0, dtype=np.int64)
        self._lfp_sample_positions_um = np.empty((0, 3), dtype=np.float64)
        self._last_lfp_proxy = "unrecorded"
        self._last_lfp_provenance = {}

        for ct_name in targets:
            if ct_name not in self._nodes:
                raise RuntimeError(f"validated record type disappeared: {ct_name}")
            if ct_name in self._working_point_rates_hz:
                self._recorders[ct_name] = self._nodes[ct_name]
                continue
            try:
                ngpu.ActivateRecSpikeTimes(self._nodes[ct_name], self._max_rec_spikes)
                self._recorders[ct_name] = self._nodes[ct_name]
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"ActivateRecSpikeTimes failed for {ct_name}: {exc}"
                ) from exc

        self._attach_lfp_record()

    def _attach_lfp_record(self) -> None:
        if self._ngpu is None:
            raise RuntimeError("Call setup() before attach_recorders().")
        if self._spec is None:
            raise RuntimeError("Call build() before attach_recorders().")
        if self._spec.neuron_model != "aglif_dend_cond_beta":
            return
        if "Pyramidal" not in self._nodes:
            return

        sample_count = min(
            _env_nonnegative_int(_LFP_SAMPLE_CELLS_ENV, _DEFAULT_LFP_SAMPLE_CELLS),
            self._n_cells.get("Pyramidal", 0),
        )
        if sample_count == 0:
            return

        receptors = self._spec.receptors_for_post("Pyramidal")
        override = self._spec.aglif_dend_overrides.get("Pyramidal")
        compartments = _aglif_dend_compartments(
            receptors.names,
            "Pyramidal",
            _required_dendritic_ports(self._spec, "Pyramidal"),
            self._spec.source_location_transfer_table,
            self._spec.aglif_receive_domain_overrides,
            self._spec.aglif_compartment_overrides,
            receive_domain=None if override is None else override.receive_domain,
        )
        nodes = self._nodes["Pyramidal"]
        var_names: list[str] = []
        record_nodes: list[int] = []
        ports: list[int] = []
        g_columns: dict[int, _LfpConductanceColumn] = {}
        vm_columns: list[int] = []
        vd_columns: list[int] = []
        vdist_columns: list[int] = []

        positions = modeldb_cell_positions(
            {"Pyramidal": self._n_cells["Pyramidal"]}
        )["Pyramidal"]
        sample_indices = _spatially_distributed_lfp_sample_indices(
            positions,
            sample_count,
            seed=self._spec.seed,
        )

        for sample_idx, cell_idx in enumerate(sample_indices):
            node = nodes[int(cell_idx)]
            vm_columns.append(len(var_names) + 1)
            var_names.append("V_m")
            record_nodes.append(node)
            ports.append(0)

            vd_columns.append(len(var_names) + 1)
            var_names.append("V_d")
            record_nodes.append(node)
            ports.append(0)

            # The deployed user_m2 model registers V_dist as a scalar recordable
            # (nest-gpu/src/user_m2_kernel.h) and uses it for distal port current.
            vdist_columns.append(len(var_names) + 1)
            var_names.append("V_dist")
            record_nodes.append(node)
            ports.append(0)

            for port_idx, (compartment, e_rev_mv) in enumerate(
                zip(compartments, receptors.E_rev, strict=True)
            ):
                col_idx = len(var_names) + 1
                var_names.append("g")
                record_nodes.append(node)
                ports.append(port_idx)
                g_columns[col_idx] = _LfpConductanceColumn(
                    sample_idx=sample_idx,
                    port_idx=port_idx,
                    compartment=float(compartment),
                    e_rev_mv=float(e_rev_mv),
                )

        try:
            self._lfp_record = self._ngpu.CreateRecord(
                "",
                var_names,
                record_nodes,
                ports,
            )
            self._ngpu.SetRecordStride(
                self._lfp_record,
                self._lfp_record_every,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "CreateRecord/SetRecordStride failed for GPU LFP current probe"
            ) from exc

        self._lfp_sample_count = sample_count
        self._lfp_g_columns = g_columns
        self._lfp_vm_columns = vm_columns
        self._lfp_vd_columns = vd_columns
        self._lfp_vdist_columns = vdist_columns
        self._lfp_sample_indices = sample_indices
        self._lfp_sample_positions_um = positions[sample_indices].copy()

    def run(self, duration_ms: float) -> None:
        """Advance simulation by ``duration_ms`` of model time.

        NEST GPU ``Simulate`` takes the total duration in one call -- do not
        use the chunked ``range(0, dur, chunk) + Simulate(chunk)`` pattern
        because NEST-GPU passes a chunk-local update index to neuron ``Update``;
        restarting that index corrupts spike generation.
        """
        if self._ngpu is None:
            raise RuntimeError("Call setup() before run().")
        self._ngpu.Simulate(duration_ms)

    def _set_literal_source_spike_trains(
        self,
        *,
        duration_s: float,
        seed: int,
    ) -> None:
        if self._ngpu is None:
            raise RuntimeError("Call setup() before setting source spike trains.")
        if self._spec is None:
            raise RuntimeError("Call build() before setting source spike trains.")
        if not self._literal_source_rates_hz and not self._working_point_rates_hz:
            return

        duration_ms = duration_s * 1e3
        if duration_ms <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        slot_count = int(np.floor(duration_ms / self._dt_ms))
        available_slot_count = slot_count - 1
        if available_slot_count <= 0:
            raise ValueError(
                "duration_s must cover at least two NEST-GPU time steps, "
                f"got duration_ms={duration_ms} and dt_ms={self._dt_ms}"
            )

        for source_key, nodes in sorted(self._poisson.items()):
            if not source_key.startswith("literal_source:"):
                continue
            source = source_key.split(":", maxsplit=1)[1]
            rate_hz = self._literal_source_rates_hz[source]
            rng = np.random.default_rng(_stable_source_seed(seed, source))
            if self._spec.afferent_source_rate_cv == 0.0:
                counts = rng.poisson(rate_hz * duration_s, size=len(nodes))
            else:
                rates_hz = source_rates_hz(
                    base_rate_hz=rate_hz,
                    count=len(nodes),
                    cv=self._spec.afferent_source_rate_cv,
                    seed=seed,
                    source=source,
                )
                counts = rng.poisson(rates_hz * duration_s)
            overflow = np.flatnonzero(counts > available_slot_count)
            if overflow.size:
                count = int(counts[overflow[0]])
                raise ValueError(
                    "literal_source_graph generated more source spikes "
                    "than available NEST-GPU time slots; reduce dt or "
                    "rate. source="
                    f"{source!r} count={count} slots={available_slot_count}"
                )
            for batch_start, offsets, spike_slots in _spike_slot_batches(
                counts,
                slot_count=slot_count,
                rng=rng,
            ):
                for local_idx in np.flatnonzero(np.diff(offsets)):
                    begin, end = offsets[local_idx:local_idx + 2]
                    spike_times = spike_slots[begin:end] * self._dt_ms
                    # NEST GPU's spike_generator calibrates a missing
                    # spike_gen_mul to 1.0, so the scalar parameter form does
                    # one Python-to-C call per node instead of the dict form's
                    # two.  Its API cannot install distinct arrays for a node
                    # sequence in one call (see spike_generator::SetArrayParam).
                    self._ngpu.SetStatus(
                        nodes[batch_start + int(local_idx):batch_start + int(local_idx) + 1],
                        "spike_times",
                        spike_times.tolist(),
                    )

        for source_key, nodes in sorted(self._poisson.items()):
            if not source_key.startswith("working_point:"):
                continue
            cell_type = source_key.split(":", maxsplit=1)[1]
            rate_hz = self._working_point_rates_hz[cell_type]
            rng = np.random.default_rng(_stable_source_seed(seed, source_key))
            counts = rng.poisson(rate_hz * duration_s, size=len(nodes))
            overflow = np.flatnonzero(counts > available_slot_count)
            if overflow.size:
                node_idx = int(overflow[0])
                count = int(counts[node_idx])
                raise ValueError(
                    "working-point clamp generated more spikes than available "
                    "NEST-GPU time slots; reduce dt or rate. cell_type="
                    f"{cell_type!r} cell_index={node_idx} count={count} "
                    f"slots={available_slot_count}"
                )
            for batch_start, offsets, spike_slots in _spike_slot_batches(
                counts,
                slot_count=slot_count,
                rng=rng,
            ):
                for local_idx in np.flatnonzero(np.diff(offsets)):
                    begin, end = offsets[local_idx:local_idx + 2]
                    node_idx = batch_start + int(local_idx)
                    spike_times = spike_slots[begin:end] * self._dt_ms
                    self._working_point_spikes_ms[cell_type][node_idx] = spike_times
                    self._ngpu.SetStatus(
                        nodes[node_idx:node_idx + 1],
                        "spike_times",
                        spike_times.tolist(),
                    )

    def collect_spikes(self) -> dict[str, list[npt.NDArray[np.float64]]]:
        """Return spike times in seconds, grouped by cell type and cell index.

        ``ngpu.GetRecordData(rec_id)`` returns ``[[time_ms, gid], ...]``.
        GIDs are converted to 0-based cell indices relative to population start.
        Silent cells appear as empty arrays so downstream analyses see all cells.
        """
        if self._ngpu is None:
            raise RuntimeError("Call setup() before collect_spikes().")

        ngpu = self._ngpu
        spikes: dict[str, list[npt.NDArray[np.float64]]] = {}

        for ct_name, nodes in self._recorders.items():
            n = self._n_cells.get(ct_name, 0)
            if ct_name in self._working_point_spikes_ms:
                recorded = self._working_point_spikes_ms[ct_name]
                spikes[ct_name] = [
                    np.asarray(spike_times, dtype=float) * 1e-3
                    for spike_times in recorded[:n]
                ]
                continue
            try:
                rec = ngpu.GetRecSpikeTimes(nodes)   # list per node of spike times (ms)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"GetRecSpikeTimes failed for {ct_name}: {exc}"
                ) from exc

            cell_spikes: list[npt.NDArray[np.float64]] = []
            for i in range(n):
                spike_times = rec[i] if rec is not None and i < len(rec) else None
                if spike_times:
                    if len(spike_times) >= self._max_rec_spikes:
                        raise RuntimeError(
                            f"GPU spike recorder saturated for {ct_name}[{i}] at "
                            f"{self._max_rec_spikes} spikes; increase "
                            f"{_MAX_REC_SPIKES_ENV}"
                        )
                    cell_spikes.append(np.asarray(spike_times, dtype=float) * 1e-3)  # ms -> s
                else:
                    cell_spikes.append(np.empty(0, dtype=float))
            spikes[ct_name] = cell_spikes

        return spikes

    def collect_lfp(self) -> tuple[Optional[npt.NDArray[np.float64]], Optional[float]]:
        if self._ngpu is None:
            raise RuntimeError("Call setup() before collect_lfp().")
        if self._lfp_record is None:
            return None, None

        try:
            rows = self._ngpu.GetRecordData(self._lfp_record)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("GetRecordData failed for GPU LFP current probe") from exc

        data = np.asarray(rows, dtype=np.float64)
        expected_cols = 1 + 3 * self._lfp_sample_count + len(self._lfp_g_columns)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != expected_cols:
            raise RuntimeError(
                "GPU LFP current probe returned malformed data: "
                f"shape={data.shape}, expected columns={expected_cols}"
            )
        if not np.isfinite(data).all():
            raise RuntimeError("GPU LFP current probe returned non-finite data")

        times_ms = data[:, 0]
        observed_dt_ms = np.diff(times_ms)
        lfp_dt_ms = self._lfp_record_every * self._dt_ms
        # NEST GPU's record-data ABI exposes float32 rows.  At sub-millisecond
        # strides, timestamp subtraction can therefore differ from the nominal
        # interval by one float32 ULP even though the native time indexes are
        # uniform.  Scale the absolute tolerance to that storage precision,
        # while separately rejecting duplicate or decreasing timestamps.
        time_axis_atol_ms = max(
            1e-6,
            float(np.spacing(np.float32(np.max(np.abs(times_ms))))),
        )
        if (
            not np.isfinite(observed_dt_ms).all()
            or not np.all(observed_dt_ms > 0.0)
            or not np.allclose(
                observed_dt_ms,
                lfp_dt_ms,
                rtol=1e-5,
                atol=time_axis_atol_ms,
            )
        ):
            raise RuntimeError("GPU LFP current probe returned invalid time axis")

        v_m = data[:, self._lfp_vm_columns]
        v_d = data[:, self._lfp_vd_columns]
        v_dist = data[:, self._lfp_vdist_columns]
        currents = np.zeros((data.shape[0], self._lfp_sample_count), dtype=np.float64)
        for col_idx, column in self._lfp_g_columns.items():
            if column.compartment < 0.5:
                voltage = v_m[:, column.sample_idx]
            elif column.compartment < 1.5:
                voltage = v_d[:, column.sample_idx]
            else:
                voltage = v_dist[:, column.sample_idx]
            currents[:, column.sample_idx] += data[:, col_idx] * (
                column.e_rev_mv - voltage
            )

        if _can_collect_modeldb_n_pole_lfp(self._spec, self._n_cells):
            positions = self._lfp_sample_positions_um
            weights = reduced_domain_n_pole_weights(
                positions,
                MODELDB_NPOLE_ELECTRODE_ROI,
            )
            selected_cells = int(np.count_nonzero(weights > 0.0))
            self._last_lfp_provenance = {
                "lfp.modeldb_n_pole_reduced_domain.current_source": (
                    "sampled-pyramidal-port-current-point-source"
                ),
                "lfp.modeldb_n_pole_reduced_domain.rho_ohm_cm": (
                    f"{MODELDB_NPOLE_RHO_OHM_CM:g}"
                ),
                "lfp.modeldb_n_pole_reduced_domain.sampled_cells": (
                    str(self._lfp_sample_count)
                ),
                "lfp.modeldb_n_pole_reduced_domain.selected_cells": (
                    str(selected_cells)
                ),
                "lfp.modeldb_n_pole_reduced_domain.sampling_strategy": (
                    "seeded-x-stratified-electrode-roi"
                ),
                "lfp.modeldb_n_pole_reduced_domain.sample_seed": str(
                    self._spec.seed
                ),
                "lfp.modeldb_n_pole_reduced_domain.sampled_cell_indices_zero_based": (
                    ",".join(str(int(idx)) for idx in self._lfp_sample_indices)
                ),
                "lfp.modeldb_n_pole_reduced_domain.sampled_x_um": (
                    ",".join(f"{x:g}" for x in positions[:, 0])
                ),
                "lfp.modeldb_n_pole_reduced_domain.compartment_voltages": (
                    "soma:V_m;proximal:V_d;distal:V_dist"
                ),
            }
            self._last_lfp_proxy = _LFP_PROXY_MODELDB_N_POLE_REDUCED
            return (
                reduced_domain_n_pole_lfp(
                    currents,
                    positions,
                    MODELDB_NPOLE_ELECTRODE_ROI,
                ),
                lfp_dt_ms * 1e-3,
            )

        self._last_lfp_proxy = _LFP_PROXY_SYNAPTIC_CURRENT
        self._last_lfp_provenance = {}
        return currents.mean(axis=1), lfp_dt_ms * 1e-3

    # ------------------------------------------------------------------
    # Persistence and template override
    # ------------------------------------------------------------------

    def persist_spikes(
        self,
        spikes: dict[str, list[npt.NDArray[np.float64]]],
        out_path: os.PathLike[str] | str,
    ) -> None:
        """Pickle ``spikes`` dict to ``out_path`` (parent dirs created)."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}")
        with tmp.open("wb") as fh:
            pickle.dump(spikes, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out)

    def _rank_spikes_path(self, run_dir: Path, rank: int) -> Path:
        return run_dir / f"spikes_raw_rank{rank}.pkl"

    def _merge_rank_spikes(
        self,
        run_dir: Path,
        n_cells: Mapping[str, int],
    ) -> dict[str, list[npt.NDArray[np.float64]]]:
        timeout_s = float(os.environ.get("CA1_MPI_MERGE_TIMEOUT_S", "600"))
        deadline = time.monotonic() + timeout_s
        rank_paths = [
            self._rank_spikes_path(run_dir, rank)
            for rank in range(self._mpi_size)
        ]
        while not all(path.exists() for path in rank_paths):
            if time.monotonic() > deadline:
                missing = [str(path) for path in rank_paths if not path.exists()]
                raise TimeoutError(f"Timed out waiting for MPI spike files: {missing}")
            time.sleep(0.1)

        merged: dict[str, list[npt.NDArray[np.float64]]] = {
            ct_name: [] for ct_name in n_cells
        }
        for path in rank_paths:
            with path.open("rb") as fh:
                rank_spikes: dict[str, list[npt.NDArray[np.float64]]] = pickle.load(fh)
            for ct_name in merged:
                merged[ct_name].extend(rank_spikes.get(ct_name, []))

        for ct_name, expected_count in n_cells.items():
            actual_count = len(merged[ct_name])
            if actual_count != int(expected_count):
                raise RuntimeError(
                    f"Merged spike count mismatch for {ct_name}: "
                    f"{actual_count} cells, expected {expected_count}"
                )
        return merged

    def simulate(self, spec: NetworkSpec, meta: SimMeta,  # type: ignore[override]
                 record_types: Optional[Iterable[str]] = None) -> SimResult:
        """Template override: adds per-rank raw-spike persistence before crop."""
        n_cells = dict(meta.n_cells_per_type)
        _phase_timing = os.environ.get("CA1_PHASE_TIMING")
        _t0 = time.perf_counter()
        self.setup(dt_ms=meta.dt_s * 1e3, seed=meta.seed)
        _t1 = time.perf_counter()
        self.build(spec, n_cells)
        _t2 = time.perf_counter()
        self._set_literal_source_spike_trains(
            duration_s=meta.duration_s,
            seed=meta.seed,
        )
        _t3 = time.perf_counter()
        self._max_rec_spikes = _record_spike_buffer_size(meta.duration_s)
        self.attach_recorders(record_types)
        _t4 = time.perf_counter()
        self.run(meta.duration_s * 1e3)
        _t5 = time.perf_counter()
        if _phase_timing:
            print(
                f"[CA1_PHASE_TIMING] setup={_t1 - _t0:.1f}s "
                f"build+connect={_t2 - _t1:.1f}s spike_setup={_t3 - _t2:.1f}s "
                f"recorders={_t4 - _t3:.1f}s run={_t5 - _t4:.1f}s",
                flush=True,
            )

        spikes = self.collect_spikes()

        run_dir = Path(
            os.environ.get("CA1_RUN_DIR", str(Path(tempfile.gettempdir()) / "ca1_runs"))
        )
        self.persist_spikes(spikes, self._rank_spikes_path(run_dir, self._mpi_rank))

        if self._mpi_size > 1 and self._mpi_rank == 0:
            spikes = self._merge_rank_spikes(run_dir, meta.n_cells_per_type)

        crop_s = meta.crop_first_ms * 1e-3
        if crop_s > 0:
            spikes = {
                ct: [a[a >= crop_s] - crop_s for a in cells]
                for ct, cells in spikes.items()
            }

        lfp, lfp_dt = self.collect_lfp()
        if lfp is not None and lfp_dt is not None and crop_s > 0.0:
            crop_steps = int(round(crop_s / lfp_dt))
            lfp = lfp[crop_steps:]
        lfp_proxy = (
            self._last_lfp_proxy
            if lfp is not None and lfp_dt is not None
            else _LFP_PROXY_SPIKE_DENSITY
        )
        return SimResult(
            spikes=spikes,
            meta=_meta_with_lfp_proxy(meta, lfp_proxy, self._last_lfp_provenance),
            lfp=lfp,
            lfp_dt_s=lfp_dt,
        )
