"""Persisted ModelDB 3-D Gaussian edge graphs.

This module deliberately does not import :mod:`ca1.sim.gpu_backend` (or
``nestgpu``).  ``ca1 build-edges`` can therefore use the topology process pool
from a CPU-only process without loading MPI/libinfinipath.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Mapping

import h5py
import numpy as np
import numpy.typing as npt

from ca1.sim.modeldb_positions import modeldb_connectivity_positions
from ca1.sim.modeldb_topology import (
    ModelDbFastconn3D,
    ModelDbFastconnPostEdges,
    recurrent_projection_plans,
)
from ca1.types import Afferent, NetworkSpec


EDGE_ARTIFACT_ENV = "CA1_EDGE_ARTIFACT"
EDGE_ARTIFACT_FORMAT = 1
_FORMAT_NAME = "ca1-modeldb-3d-edges"


@dataclass(frozen=True, slots=True)
class EdgeArtifactBuildStats:
    path: Path
    artifact_key: str
    edge_count: int
    projection_count: int
    digest: str
    used_process_pool: bool


class EdgeArtifact:
    """Validated compact edge arrays, streamed from HDF5 per projection."""

    def __init__(self, path: Path, projection_paths: Mapping[str, str]) -> None:
        self._path = path
        self._projection_paths = dict(projection_paths)

    def iter_post_edges(
        self,
        *,
        pre_type: str,
        post_type: str,
        indegree: int,
        seed: int,
        projection: str | None = None,
    ) -> Iterator[ModelDbFastconnPostEdges]:
        """Match ``ModelDbFastconn3D.iter_post_edges`` for GPU connect callers."""
        del pre_type, post_type, indegree, seed
        if projection is None:
            raise ValueError("persisted 3-D edges require an explicit projection key")
        try:
            group_path = self._projection_paths[projection]
        except KeyError as exc:
            raise ValueError(f"edge artifact lacks projection {projection!r}") from exc
        with h5py.File(self._path, "r") as h5:
            stored = h5[group_path]
            offsets = np.asarray(stored["offsets"], dtype=np.uint64)
            sources_dataset = stored["sources"]
            for post_index in range(len(offsets) - 1):
                start = int(offsets[post_index])
                stop = int(offsets[post_index + 1])
                sources = np.asarray(sources_dataset[start:stop], dtype=np.int64)
                # Rings are intentionally not persisted: GPU connection expansion only
                # consumes the sampled source order and post index.
                yield ModelDbFastconnPostEdges(
                    post_index=post_index,
                    source_indices=sources,
                    ring_indices=np.empty(len(sources), dtype=np.int8),
                )


def artifact_key(spec: NetworkSpec) -> str:
    """Stable cache key for the topology inputs named in the user contract."""
    conndata = "none" if spec.conndata_index is None else str(spec.conndata_index)
    scale = format(float(spec.scale), ".17g")
    return "_".join(
        (
            f"seed-{spec.seed}",
            f"scale-{scale}",
            f"topology-{spec.recurrent_topology}",
            f"conndata-{conndata}",
            f"cellnumbers-{spec.cellnumbers_index}",
        )
    )


def default_artifact_path(spec: NetworkSpec) -> Path:
    return Path("edge_artifacts") / f"{artifact_key(spec)}.h5"


def _literal_source_indegree(afferent: Afferent) -> int:
    if afferent.synapses_per_connection < 1:
        raise ValueError(f"{afferent.name} synapses_per_connection must be positive")
    contacts = afferent.synapses_per_cell / float(afferent.synapses_per_connection)
    indegree = int(round(contacts))
    if indegree < 1 or not np.isclose(contacts, float(indegree), rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{afferent.name} synapses_per_cell is not an integer contact count"
        )
    if indegree > afferent.n_source:
        raise ValueError(f"{afferent.name} contact indegree exceeds source count")
    return indegree


def _connectivity_counts(spec: NetworkSpec, n_cells: Mapping[str, int]) -> dict[str, int]:
    counts = {cell_type: int(count) for cell_type, count in n_cells.items()}
    for afferent in spec.afferents:
        source = afferent.name.split("_to_", maxsplit=1)[0]
        previous = counts.get(source)
        if previous is not None and previous != int(afferent.n_source):
            raise ValueError(
                f"3-D topology source {source!r} has inconsistent counts "
                f"{previous} and {afferent.n_source}"
            )
        counts[source] = int(afferent.n_source)
    return counts


def _projection_descriptors(spec: NetworkSpec, n_cells: Mapping[str, int]) -> list[dict[str, object]]:
    descriptors: list[dict[str, object]] = []
    for plan in recurrent_projection_plans(spec.projections, n_cells):
        descriptors.append(
            {
                "name": f"recurrent:{plan.pre}->{plan.post}",
                "kind": "recurrent",
                "pre": plan.pre,
                "post": plan.post,
                "indegree": plan.indegree,
                "delay_ms": plan.delay_ms,
                "release_components": [
                    {
                        "name": component.name,
                        "ports": [
                            {
                                "receptor": port.projection.receptor,
                                "receptor_port": spec.receptors_for_post(plan.post).port_index(
                                    port.projection.receptor
                                ),
                                "weight_nS": port.projection.weight_nS,
                                "synapses_per_connection": port.projection.synapses_per_connection,
                                "edges_per_post": port.edges_per_post,
                            }
                            for port in component.ports
                        ],
                    }
                    for component in plan.components
                ],
            }
        )
    if spec.afferent_topology in {"literal_source_graph", "literal_source_graph_binned"}:
        for afferent in spec.afferents:
            descriptors.append(
                {
                    "name": f"afferent:{afferent.name}",
                    "kind": "afferent",
                    "pre": afferent.name.split("_to_", maxsplit=1)[0],
                    "post": afferent.post,
                    "indegree": _literal_source_indegree(afferent),
                    "delay_ms": afferent.delay_ms,
                    "receptor": afferent.receptor,
                    "receptor_port": spec.receptors_for_post(afferent.post).port_index(
                        afferent.receptor
                    ),
                    "weight_nS": afferent.weight_nS,
                    "synapses_per_connection": afferent.synapses_per_connection,
                }
            )
    return descriptors


def _provenance(spec: NetworkSpec, n_cells: Mapping[str, int]) -> dict[str, object]:
    # Include every input that changes sampled pairs, as well as the requested
    # human-auditable cache-key fields.
    return {
        "format": _FORMAT_NAME,
        "format_version": EDGE_ARTIFACT_FORMAT,
        "artifact_key": artifact_key(spec),
        "seed": spec.seed,
        "scale": float(spec.scale),
        "recurrent_topology": spec.recurrent_topology,
        "afferent_topology": spec.afferent_topology,
        "conndata_index": spec.conndata_index,
        "cellnumbers_index": spec.cellnumbers_index,
        "conndata_count_mode": spec.conndata_count_mode,
        "cell_counts": {name: int(count) for name, count in sorted(n_cells.items())},
        "projections": _projection_descriptors(spec, n_cells),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _without_synaptic_weights(value: object) -> object:
    """Return the sampled-pair contract, excluding Connect-time weights."""
    if isinstance(value, dict):
        return {
            key: _without_synaptic_weights(item)
            for key, item in value.items()
            if key != "weight_nS"
        }
    if isinstance(value, list):
        return [_without_synaptic_weights(item) for item in value]
    return value


def graph_identity_digest(
    spec: NetworkSpec,
    n_cells: Mapping[str, int] | None = None,
) -> str:
    """Digest every input to sampled pairs while ignoring synaptic weights."""
    counts = spec.scaled_counts() if n_cells is None else n_cells
    identity = _without_synaptic_weights(_provenance(spec, counts))
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _write_projection(
    group: h5py.Group,
    topology: ModelDbFastconn3D,
    descriptor: Mapping[str, object],
    *,
    seed: int,
) -> int:
    """Append one projection without retaining the entire graph in RAM."""
    pre = str(descriptor["pre"])
    post = str(descriptor["post"])
    indegree = int(descriptor["indegree"])
    name = str(descriptor["name"])
    source_dataset = group.create_dataset(
        "sources", shape=(0,), maxshape=(None,), dtype=np.uint32,
        chunks=True, compression="gzip",
    )
    offsets: list[int] = [0]
    buffered: list[npt.NDArray[np.uint32]] = []
    buffered_count = 0
    edge_count = 0
    for post_edges in topology.iter_post_edges(
        pre_type=pre, post_type=post, indegree=indegree, seed=seed, projection=name,
    ):
        sources = np.asarray(post_edges.source_indices, dtype=np.uint32)
        buffered.append(sources)
        buffered_count += len(sources)
        edge_count += len(sources)
        offsets.append(edge_count)
        if buffered_count >= 1_048_576:
            start = len(source_dataset)
            source_dataset.resize((start + buffered_count,))
            source_dataset[start:] = np.concatenate(buffered)
            buffered.clear()
            buffered_count = 0
    if buffered_count:
        start = len(source_dataset)
        source_dataset.resize((start + buffered_count,))
        source_dataset[start:] = np.concatenate(buffered)
    group.create_dataset("offsets", data=np.asarray(offsets, dtype=np.uint64), compression="gzip")
    return edge_count


def _digest_h5(provenance: Mapping[str, object], h5: h5py.File) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(provenance).encode("utf-8"))
    for _key, group in sorted(h5["projections"].items()):
        digest.update(str(group.attrs["name"]).encode("utf-8"))
        digest.update(np.asarray(group["offsets"], dtype="<u8").tobytes())
        sources = group["sources"]
        for start in range(0, len(sources), 1_048_576):
            digest.update(
                np.asarray(sources[start:start + 1_048_576], dtype="<u4").tobytes()
            )
    return digest.hexdigest()


def build_edge_artifact(
    spec: NetworkSpec,
    out_path: str | Path | None = None,
    *,
    max_workers: int | None = None,
) -> EdgeArtifactBuildStats:
    """Generate all 3-D recurrent/literal-afferent pairs and atomically persist them."""
    if spec.recurrent_topology != "modeldb_fastconn_3d_gaussian":
        raise ValueError("edge artifacts require recurrent_topology='modeldb_fastconn_3d_gaussian'")
    n_cells = spec.scaled_counts()
    provenance = _provenance(spec, n_cells)
    destination = Path(out_path) if out_path is not None else default_artifact_path(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    topology = ModelDbFastconn3D(
        modeldb_connectivity_positions(_connectivity_counts(spec, n_cells)),
        max_workers=max_workers,
        # This CPU-only command is specifically the amortized parallel build
        # path; avoid silently reverting a small projection to serial work.
        force_parallel=max_workers != 1,
    )
    try:
        descriptors = _projection_descriptors(spec, n_cells)
        used_process_pool = False
        edge_count = 0
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with h5py.File(temporary, "w") as h5:
            h5.attrs["format"] = _FORMAT_NAME
            h5.attrs["format_version"] = EDGE_ARTIFACT_FORMAT
            h5.attrs["provenance_json"] = _canonical_json(provenance)
            group = h5.create_group("projections")
            for index, descriptor in enumerate(descriptors):
                item = group.create_group(f"{index:04d}")
                item.attrs["name"] = str(descriptor["name"])
                item.attrs["metadata_json"] = _canonical_json(descriptor)
                edge_count += _write_projection(item, topology, descriptor, seed=spec.seed)
                used_process_pool = used_process_pool or topology._executor is not None
            digest = _digest_h5(provenance, h5)
            h5.attrs["edge_sha256"] = digest
    finally:
        topology.close()
    temporary.replace(destination)
    return EdgeArtifactBuildStats(
        path=destination,
        artifact_key=artifact_key(spec),
        edge_count=edge_count,
        projection_count=len(descriptors),
        digest=digest,
        used_process_pool=used_process_pool,
    )


def load_edge_artifact(path: str | Path, spec: NetworkSpec, n_cells: Mapping[str, int]) -> EdgeArtifact:
    """Load and validate an artifact before any GPU ``Connect`` calls happen."""
    expected = _provenance(spec, n_cells)
    source_path = Path(path)
    with h5py.File(source_path, "r") as h5:
        if h5.attrs.get("format") != _FORMAT_NAME or int(h5.attrs.get("format_version", -1)) != EDGE_ARTIFACT_FORMAT:
            raise ValueError(f"unsupported 3-D edge artifact format: {source_path}")
        raw_provenance = h5.attrs.get("provenance_json")
        if isinstance(raw_provenance, bytes):
            raw_provenance = raw_provenance.decode("utf-8")
        actual = json.loads(str(raw_provenance))
        if _without_synaptic_weights(actual) != _without_synaptic_weights(expected):
            raise ValueError(
                "3-D edge artifact sampled-pair provenance does not match this "
                f"simulation: {source_path}"
            )
        expected_descriptors = _projection_descriptors(spec, n_cells)
        groups = [group for _key, group in sorted(h5["projections"].items())]
        if len(groups) != len(expected_descriptors):
            raise ValueError(f"3-D edge artifact projection count mismatch: {source_path}")
        projection_paths: dict[str, str] = {}
        for group, descriptor in zip(groups, expected_descriptors, strict=True):
            raw_metadata = group.attrs.get("metadata_json")
            if isinstance(raw_metadata, bytes):
                raw_metadata = raw_metadata.decode("utf-8")
            stored_descriptor = json.loads(str(raw_metadata))
            if _without_synaptic_weights(stored_descriptor) != (
                _without_synaptic_weights(descriptor)
            ):
                raise ValueError(f"3-D edge artifact projection metadata mismatch: {source_path}")
            name = str(group.attrs["name"])
            projection_paths[name] = group.name
        expected_digest = str(h5.attrs.get("edge_sha256", ""))
        # Verify the artifact against its own persisted provenance.  A compatible
        # source-grounded deck may change only Connect-time weights while reusing
        # these exact sampled source/target arrays.
        actual_digest = _digest_h5(actual, h5)
    if actual_digest != expected_digest:
        raise ValueError(f"3-D edge artifact checksum mismatch: {source_path}")
    return EdgeArtifact(source_path, projection_paths)


def load_edge_artifact_from_env(
    spec: NetworkSpec,
    n_cells: Mapping[str, int],
) -> EdgeArtifact | None:
    """Resolve ``CA1_EDGE_ARTIFACT`` (file or cache directory), or fall back."""
    configured = os.environ.get(EDGE_ARTIFACT_ENV)
    if not configured:
        return None
    candidate = Path(configured)
    if candidate.is_dir() or configured.endswith(os.sep):
        candidate = candidate / f"{artifact_key(spec)}.h5"
    if not candidate.exists():
        return None
    return load_edge_artifact(candidate, spec, n_cells)
