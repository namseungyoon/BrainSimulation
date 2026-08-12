from __future__ import annotations

import builtins
from collections import Counter
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from ca1.config import build_network_spec
import ca1.sim.gpu_backend as gpu_backend
from ca1.sim.edge_artifact import build_edge_artifact, load_edge_artifact
import ca1.sim.modeldb_topology as modeldb_topology
from ca1.sim.gpu_backend import NestGpuBackend
from ca1.sim.modeldb_positions import modeldb_connectivity_positions
from ca1.sim.modeldb_topology import (
    ModelDbFastconn3D,
    boot_modeldb_topology_forkserver,
    feasible_gaussian_ring_indegrees,
    gaussian_ring_indegrees,
    projection_port_edges,
    recurrent_projection_plans,
)
from ca1.types import (
    Afferent,
    CellType,
    NetworkSpec,
    NeuronParams,
    Projection,
)


ROOT = Path(__file__).resolve().parents[1]


def _edge_digest(topology: ModelDbFastconn3D) -> str:
    digest = hashlib.sha256()
    for post_edges in topology.iter_post_edges(
        pre_type="CA3",
        post_type="Pyramidal",
        indegree=197,
        seed=12_345,
        projection="afferent:CA3_to_Pyramidal",
    ):
        digest.update(np.asarray([post_edges.post_index], dtype="<i8").tobytes())
        digest.update(post_edges.source_indices.astype("<i8", copy=False).tobytes())
        digest.update(post_edges.ring_indices.astype("i1", copy=False).tobytes())
    return digest.hexdigest()


def test_modeldb_3d_gaussian_ring_fraction_matches_fastconn_source_mass() -> None:
    positions = modeldb_connectivity_positions(
        {"Pyramidal": 10_000, "PV_Basket": 100}
    )
    topology = ModelDbFastconn3D(positions, max_workers=1)
    rings: Counter[int] = Counter()

    for post_edges in topology.iter_post_edges(
        pre_type="Pyramidal",
        post_type="PV_Basket",
        indegree=197,
        seed=20_260_710,
    ):
        rings.update(int(ring) for ring in post_edges.ring_indices)

    total = sum(rings.values())
    fractions = np.asarray([rings[ring] / total for ring in range(1, 6)])
    np.testing.assert_allclose(
        fractions,
        [0.868, 0.127, 0.005, 0.0, 0.0],
        atol=6e-4,
    )
    assert gaussian_ring_indegrees("Pyramidal", 197) == (171, 25, 1, 0, 0)


def _sample_edges(seed: int) -> list[tuple[int, tuple[int, ...]]]:
    positions = modeldb_connectivity_positions(
        {"Pyramidal": 2_000, "PV_Basket": 12}
    )
    topology = ModelDbFastconn3D(positions)
    return [
        (post_edges.post_index, tuple(int(x) for x in post_edges.source_indices))
        for post_edges in topology.iter_post_edges(
            pre_type="Pyramidal",
            post_type="PV_Basket",
            indegree=32,
            seed=seed,
            projection="determinism:Pyramidal->PV_Basket",
        )
    ]


def test_modeldb_3d_gaussian_is_seed_deterministic_and_has_no_duplicate_sources() -> None:
    first = _sample_edges(1234)
    repeated = _sample_edges(1234)
    changed = _sample_edges(1235)

    assert first == repeated
    assert first != changed
    assert all(len(sources) == len(set(sources)) == 32 for _post, sources in first)


def test_modeldb_3d_gaussian_matches_preoptimization_edge_digest() -> None:
    positions = modeldb_connectivity_positions({"CA3": 4_096, "Pyramidal": 23})
    assert _edge_digest(ModelDbFastconn3D(positions)) == (
        "41314045da9efc8cac3c8417e871110c5"
        "97093384db3dab6b9e8ded74daed420"
    )


def test_modeldb_3d_gaussian_forkserver_matches_serial_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = modeldb_connectivity_positions({"CA3": 4_096, "Pyramidal": 23})
    serial = ModelDbFastconn3D(positions, max_workers=1)
    serial_digest = _edge_digest(serial)

    try:
        boot_modeldb_topology_forkserver()
    except PermissionError as exc:
        pytest.skip(f"forkserver sockets are blocked by this sandbox: {exc}")
    monkeypatch.setattr(modeldb_topology, "_PARALLEL_MIN_DISTANCE_PAIRS", 0)
    parallel = ModelDbFastconn3D(positions, max_workers=2)
    try:
        assert _edge_digest(parallel) == serial_digest
    finally:
        parallel.close()


def test_gpu_setup_boots_forkserver_before_importing_nestgpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_ngpu = SimpleNamespace(
        SetRandomSeed=lambda seed: None,
        SetTimeResolution=lambda resolution_ms: None,
    )
    original_import = builtins.__import__

    def tracked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "nestgpu":
            events.append("nestgpu import")
            return fake_ngpu
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        gpu_backend,
        "boot_modeldb_topology_forkserver",
        lambda: events.append("forkserver boot"),
    )
    monkeypatch.setattr(builtins, "__import__", tracked_import)
    monkeypatch.delenv("PMI_RANK", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    # Operator opts into the parallel pool -> forkserver must boot first, while
    # the process is still free of nestgpu/MPI/libinfinipath.
    monkeypatch.setenv("CA1_TOPOLOGY_MAX_WORKERS", "2")

    NestGpuBackend().setup()

    assert events == ["forkserver boot", "nestgpu import"]


def test_gpu_setup_skips_forkserver_when_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default GPU path is single-process: no forkserver, so libinfinipath's
    teardown handler cannot race a multiprocessing atexit and segfault."""
    events: list[str] = []
    fake_ngpu = SimpleNamespace(
        SetRandomSeed=lambda seed: None,
        SetTimeResolution=lambda resolution_ms: None,
    )
    original_import = builtins.__import__

    def tracked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "nestgpu":
            events.append("nestgpu import")
            return fake_ngpu
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        gpu_backend,
        "boot_modeldb_topology_forkserver",
        lambda: events.append("forkserver boot"),
    )
    monkeypatch.setattr(builtins, "__import__", tracked_import)
    monkeypatch.delenv("PMI_RANK", raising=False)
    monkeypatch.delenv("PMI_SIZE", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.delenv("CA1_TOPOLOGY_MAX_WORKERS", raising=False)

    NestGpuBackend().setup()

    assert events == ["nestgpu import"]


def test_modeldb_feasibility_redistributes_short_inner_ring_without_losing_k() -> None:
    assert feasible_gaussian_ring_indegrees(
        candidate_counts=(2, 20, 0, 0, 0),
        desired_counts=(9, 1, 0, 0, 0),
    ) == (2, 8, 0, 0, 0)


def test_external_ca3_and_eciii_positions_use_modeldb_layer_flags() -> None:
    positions = modeldb_connectivity_positions({"CA3": 9, "ECIII": 9})

    assert positions["CA3"].shape == (9, 3)
    assert positions["ECIII"].shape == (9, 3)
    assert np.all((positions["CA3"][:, 2] >= 104.0) & (positions["CA3"][:, 2] < 154.0))
    assert np.all((positions["ECIII"][:, 2] >= 154.0) & (positions["ECIII"][:, 2] < 354.0))


def test_cck_split_ports_keep_exact_source_pair_indegrees() -> None:
    spec = build_network_spec(ROOT / "configs/full_scale_3dtopo.yaml")
    plans = {
        (plan.pre, plan.post): plan
        for plan in recurrent_projection_plans(spec.projections, spec.scaled_counts())
    }

    expected = {
        ("CCK_Basket", "CCK_Basket"): (35, [18, 17]),
        ("CCK_Basket", "Pyramidal"): (13, [7, 6]),
        ("CCK_Basket", "SCA"): (27, [14, 13]),
    }
    for pair, (indegree, allocations) in expected.items():
        plan = plans[pair]
        assert plan.indegree == indegree
        assert [port.edges_per_post for port in plan.components[0].ports] == allocations
        assert sum(port.edges_per_post for port in plan.components[0].ports) == indegree


def test_ngf_gabaa_gabab_components_reuse_identical_base_edges() -> None:
    spec = build_network_spec(ROOT / "configs/full_scale_3dtopo.yaml")
    plan = next(
        plan
        for plan in recurrent_projection_plans(spec.projections, spec.scaled_counts())
        if plan.pre == "Neurogliaform" and plan.post == "Pyramidal"
    )
    positions = modeldb_connectivity_positions(
        {"Neurogliaform": spec.scaled_counts()["Neurogliaform"], "Pyramidal": 4}
    )
    post_edges = next(
        ModelDbFastconn3D(positions).iter_post_edges(
            pre_type="Neurogliaform",
            post_type="Pyramidal",
            indegree=plan.indegree,
            seed=spec.seed,
            projection="recurrent:Neurogliaform->Pyramidal",
        )
    )

    by_component: dict[str, set[tuple[int, int]]] = {}
    for assignment in projection_port_edges(plan, post_edges):
        by_component.setdefault(assignment.component, set()).update(
            zip(
                (int(value) for value in assignment.source_indices),
                (int(value) for value in assignment.target_indices),
                strict=True,
            )
        )

    assert by_component["primary"] == by_component["GABA_B"]
    assert len(by_component["primary"]) == plan.indegree == 14


class _Nodes:
    def __init__(self, start: int, count: int, label: str) -> None:
        self.start = start
        self.count = count
        self.label = label

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, key: int | slice) -> int | _Nodes:
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = self.count if key.stop is None else key.stop
            return _Nodes(self.start + start, stop - start, self.label)
        return self.start + key


class _FakeNestGpu:
    def __init__(self) -> None:
        self.next_node = 0
        self.connect_calls: list[tuple[Any, Any, dict[str, Any], dict[str, Any]]] = []

    def Create(self, model: str, count: int = 1, n_ports: int | None = None) -> _Nodes:  # noqa: N802
        _ = n_ports
        nodes = _Nodes(self.next_node, int(count), model)
        self.next_node += int(count)
        return nodes

    def SetStatus(self, nodes: _Nodes, params: Any, val: Any = None) -> None:  # noqa: N802
        _ = nodes, params, val

    def Connect(self, pre: Any, post: Any, conn: dict[str, Any], syn: dict[str, Any]) -> None:  # noqa: N802
        self.connect_calls.append((pre, post, conn, syn))


def _params() -> NeuronParams:
    return NeuronParams(
        C_m=100.0,
        g_L=5.0,
        E_L=-65.0,
        V_th=-50.0,
        V_reset=-60.0,
        Delta_T=2.0,
        a=0.0,
        b=0.0,
        tau_w=100.0,
        t_ref=2.0,
    )


def test_gpu_3d_path_connects_ngf_corelease_as_explicit_identical_edges() -> None:
    spec = NetworkSpec(
        name="ngf-explicit-3d",
        cell_types={
            "Neurogliaform": CellType("Neurogliaform", 20, ("SLM",), _params()),
            "Pyramidal": CellType("Pyramidal", 6, ("SP",), _params()),
        },
        projections=[
            Projection(
                "Neurogliaform", "Pyramidal", 4.0, 1, 0.3, "GABA_A_slow",
                biological_indegree=4.0,
                release_component="primary",
            ),
            Projection(
                "Neurogliaform", "Pyramidal", 4.0, 1, 0.1, "GABA_B",
                biological_indegree=4.0,
                release_component="GABA_B",
            ),
        ],
        afferents=[],
        recurrent_topology="modeldb_fastconn_3d_gaussian",
        seed=17,
    )
    fake = _FakeNestGpu()
    backend = NestGpuBackend()
    backend._ngpu = fake

    backend.build(spec, spec.scaled_counts())

    calls = [call for call in fake.connect_calls if call[2] == {"rule": "one_to_one"}]
    by_receptor = {
        call[3]["receptor"]: set(zip(call[0], call[1], strict=True)) for call in calls
    }
    gabaa = spec.receptors.port_index("GABA_A_slow")
    gabab = spec.receptors.port_index("GABA_B")
    assert by_receptor[gabaa] == by_receptor[gabab]
    assert len(by_receptor[gabaa]) == 6 * 4
    assert spec.receptors.E_rev[gabaa] < 0.0
    assert spec.receptors.E_rev[gabab] < 0.0
    assert all(call[3]["delay"] == pytest.approx(3.0) for call in calls)
    assert all(call[3]["weight"] > 0.0 for call in calls)


def test_gpu_literal_afferent_reuses_3d_generator_and_preserves_budget() -> None:
    spec = NetworkSpec(
        name="afferent-explicit-3d",
        cell_types={
            "Pyramidal": CellType("Pyramidal", 3, ("SP",), _params()),
        },
        projections=[],
        afferents=[
            Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=50,
                synapses_per_cell=10.0,
                synapses_per_connection=2,
                weight_nS=0.2,
                receptor="AMPA_fast",
            )
        ],
        recurrent_topology="modeldb_fastconn_3d_gaussian",
        afferent_topology="literal_source_graph",
        seed=19,
    )
    fake = _FakeNestGpu()
    backend = NestGpuBackend()
    backend._ngpu = fake

    backend.build(spec, spec.scaled_counts())

    calls = [call for call in fake.connect_calls if call[2] == {"rule": "one_to_one"}]
    assert len(calls) == 1
    sources, targets, _conn, syn_spec = calls[0]
    assert len(sources) == len(targets) == 3 * 5
    for target in set(targets):
        target_sources = [source for source, value in zip(sources, targets, strict=True) if value == target]
        assert len(target_sources) == len(set(target_sources)) == 5
    assert syn_spec == {
        "weight": pytest.approx(0.4),
        "delay": pytest.approx(3.0),
        "receptor": spec.receptors.port_index("AMPA_fast"),
    }


def test_persisted_3d_edges_match_regeneration_and_gpu_connect_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compact source/offset arrays preserve exact sampled edge order."""
    spec = NetworkSpec(
        name="persisted-explicit-3d",
        cell_types={
            "Neurogliaform": CellType("Neurogliaform", 20, ("SLM",), _params()),
            "Pyramidal": CellType("Pyramidal", 6, ("SP",), _params()),
        },
        projections=[
            Projection(
                "Neurogliaform", "Pyramidal", 4.0, 1, 0.3, "GABA_A_slow",
                biological_indegree=4.0,
            ),
        ],
        afferents=[
            Afferent(
                name="CA3_to_Pyramidal", post="Pyramidal", n_source=50,
                synapses_per_cell=10.0, synapses_per_connection=2, weight_nS=0.2,
            ),
        ],
        recurrent_topology="modeldb_fastconn_3d_gaussian",
        afferent_topology="literal_source_graph",
        seed=19,
    )
    counts = spec.scaled_counts()
    artifact_path = tmp_path / "edges.h5"
    stats = build_edge_artifact(spec, artifact_path, max_workers=1)
    loaded = load_edge_artifact(artifact_path, spec, counts)
    direct = ModelDbFastconn3D(
        modeldb_connectivity_positions({"Neurogliaform": 20, "Pyramidal": 6, "CA3": 50}),
        max_workers=1,
    )
    try:
        for key, pre, post, indegree in (
            ("recurrent:Neurogliaform->Pyramidal", "Neurogliaform", "Pyramidal", 4),
            ("afferent:CA3_to_Pyramidal", "CA3", "Pyramidal", 5),
        ):
            generated = list(direct.iter_post_edges(
                pre_type=pre, post_type=post, indegree=indegree, seed=19, projection=key,
            ))
            persisted = list(loaded.iter_post_edges(
                pre_type=pre, post_type=post, indegree=indegree, seed=19, projection=key,
            ))
            assert [item.post_index for item in persisted] == [item.post_index for item in generated]
            for from_disk, regenerated in zip(persisted, generated, strict=True):
                np.testing.assert_array_equal(from_disk.source_indices, regenerated.source_indices)
    finally:
        direct.close()
    assert len(stats.digest) == 64
    assert "nestgpu" not in sys.modules

    direct_gpu = _FakeNestGpu()
    backend = NestGpuBackend()
    backend._ngpu = direct_gpu
    backend.build(spec, counts)
    monkeypatch.setenv("CA1_EDGE_ARTIFACT", str(artifact_path))
    loaded_gpu = _FakeNestGpu()
    backend = NestGpuBackend()
    backend._ngpu = loaded_gpu
    backend.build(spec, counts)
    assert loaded_gpu.connect_calls == direct_gpu.connect_calls


def test_3d_ab_configs_are_identical_except_name_and_recurrent_topology() -> None:
    with open(ROOT / "configs/smoke_3dtopo_vs_uniform_3d.yaml", encoding="utf-8") as fh:
        three_d = yaml.safe_load(fh)
    with open(ROOT / "configs/smoke_3dtopo_vs_uniform_uniform.yaml", encoding="utf-8") as fh:
        uniform = yaml.safe_load(fh)

    assert three_d.pop("name") != uniform.pop("name")
    assert three_d.pop("recurrent_topology") == "modeldb_fastconn_3d_gaussian"
    assert uniform.pop("recurrent_topology") == "modeldb_fastconn_binned"
    assert three_d == uniform


def test_full_3d_config_only_changes_recurrent_topology() -> None:
    with open(ROOT / "configs/full_scale.yaml", encoding="utf-8") as fh:
        baseline = yaml.safe_load(fh)
    with open(ROOT / "configs/full_scale_3dtopo.yaml", encoding="utf-8") as fh:
        three_d = yaml.safe_load(fh)

    assert baseline.pop("recurrent_topology") == "modeldb_fastconn_binned"
    assert three_d.pop("recurrent_topology") == "modeldb_fastconn_3d_gaussian"
    assert baseline == three_d
