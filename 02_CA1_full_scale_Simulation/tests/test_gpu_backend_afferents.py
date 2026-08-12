from __future__ import annotations

from typing import Any

import pytest


class _FakeNodeCollection:
    def __init__(self, count: int, label: str = "") -> None:
        self.count = count
        self.label = label

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, key: slice) -> "_FakeNodeCollection":
        if not isinstance(key, slice):
            raise TypeError("fake nodes only support slicing")
        start = 0 if key.start is None else key.start
        stop = self.count if key.stop is None else key.stop
        return _FakeNodeCollection(stop - start, f"{self.label}[{start}:{stop}]")


class _FakeRemoteNodeCollection:
    def __init__(self, host: int, nodes: _FakeNodeCollection) -> None:
        self.i_host = host
        self.node_seq = nodes


class _FakeNestGpu:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, int, int | None]] = []
        self.remote_create_calls: list[tuple[int, str, int, int | None]] = []
        self.connect_calls: list[
            tuple[_FakeNodeCollection, _FakeNodeCollection, dict[str, Any], dict[str, Any]]
        ] = []
        self.distributed_connect_calls: list[
            tuple[
                list[int],
                list[_FakeNodeCollection],
                list[int],
                list[_FakeNodeCollection],
                int,
                int,
                dict[str, Any],
            ]
        ] = []
        self.set_status_calls: list[tuple[_FakeNodeCollection, dict[str, Any]]] = []
        self.rec_spike_times: list[list[float] | None] | None = None
        self.host_id = 0
        self.host_num = 1
        self.host_group = 7

    def Create(self, model: str, count: int, n_ports: int | None = None) -> _FakeNodeCollection:  # noqa: N802
        self.create_calls.append((model, int(count), n_ports))
        return _FakeNodeCollection(int(count), model)

    def RemoteCreate(  # noqa: N802
        self,
        host: int,
        model: str,
        count: int,
        n_ports: int | None = None,
    ) -> _FakeRemoteNodeCollection:
        self.remote_create_calls.append((host, model, int(count), n_ports))
        return _FakeRemoteNodeCollection(host, _FakeNodeCollection(int(count), f"{host}:{model}"))

    def SetStatus(  # noqa: N802
        self,
        nodes: _FakeNodeCollection,
        params: dict[str, Any] | str,
        val: Any | None = None,
    ) -> None:
        status = {params: val} if isinstance(params, str) else params
        self.set_status_calls.append((nodes, status))

    def Connect(  # noqa: N802
        self,
        pre: _FakeNodeCollection,
        post: _FakeNodeCollection,
        conn_spec: dict[str, Any],
        syn_spec: dict[str, Any],
    ) -> None:
        self.connect_calls.append((pre, post, conn_spec, syn_spec))

    def ConnectDistributedFixedIndegree(  # noqa: N802
        self,
        source_hosts: list[int],
        source_groups: list[_FakeNodeCollection],
        target_hosts: list[int],
        target_groups: list[_FakeNodeCollection],
        indegree: int,
        host_group: int,
        syn_spec: dict[str, Any],
    ) -> None:
        self.distributed_connect_calls.append(
            (source_hosts, source_groups, target_hosts, target_groups, indegree, host_group, syn_spec)
        )

    def CreateHostGroup(self, hosts: list[int]) -> int:  # noqa: N802
        assert hosts == list(range(self.host_num))
        return self.host_group

    def HostId(self) -> int:  # noqa: N802
        return self.host_id

    def HostNum(self) -> int:  # noqa: N802
        return self.host_num

    def GetRecSpikeTimes(  # noqa: N802
        self,
        _nodes: _FakeNodeCollection,
    ) -> list[list[float] | None] | None:
        return self.rec_spike_times


def _params(types_mod):
    return types_mod.NeuronParams(
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


def test_gpu_afferent_poisson_drive_uses_independent_source_per_post_cell() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-independent-afferents",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="ECIII_to_Pyramidal",
                post="Pyramidal",
                n_source=250_000,
                synapses_per_cell=8.0,
                weight_nS=0.1,
                receptor="AMPA_slow",
                rate_hz=0.65,
            )
        ],
    )

    backend.build(spec, {"Pyramidal": 4})

    assert ("poisson_generator", 4, None) in fake_ngpu.create_calls
    afferent_connections = [
        call for call in fake_ngpu.connect_calls
        if call[2] == {"rule": "one_to_one"}
    ]
    assert afferent_connections
    _pre, _post, _conn, syn_spec = afferent_connections[-1]
    assert syn_spec["weight"] == pytest.approx(0.1)
    assert syn_spec["receptor"] == spec.receptors.port_index("AMPA_slow")


def test_gpu_intrinsic_heterogeneity_env_applies_per_cell_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_VTH_SIGMA_MV", "1.0")
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_VM_SIGMA_MV", "2.0")
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_CLIP_SIGMA", "1.0")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-intrinsic-heterogeneity",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
    )

    backend.build(spec, {"Pyramidal": 4})

    scalar_status = [
        params
        for nodes, params in fake_ngpu.set_status_calls
        if nodes.label == "aeif_cond_beta_multisynapse" and "E_L" in params
    ]
    assert scalar_status
    assert scalar_status[0]["E_L"] == -65.0
    assert scalar_status[0]["V_m"] == -65.0

    array_status = {
        field: payload["array"]
        for nodes, params in fake_ngpu.set_status_calls
        for field, payload in params.items()
        if nodes.label == "aeif_cond_beta_multisynapse"
        and isinstance(payload, dict)
        and "array" in payload
    }
    assert set(array_status) >= {"V_th", "V_m"}
    assert len(array_status["V_th"]) == 4
    assert len(array_status["V_m"]) == 4
    assert len(set(array_status["V_th"])) > 1
    assert all(-51.0 <= value <= -49.0 for value in array_status["V_th"])
    assert all(-67.0 <= value <= -63.0 for value in array_status["V_m"])


def test_gpu_intrinsic_heterogeneity_env_can_target_one_cell_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_VTH_SIGMA_MV_PYRAMIDAL", "1.0")
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_CLIP_SIGMA", "1.0")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-targeted-intrinsic-heterogeneity",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "SCA": types_mod.CellType(
                name="SCA",
                count=4,
                layers=("SR",),
                params=_params(types_mod),
            ),
        },
        projections=[],
        afferents=[],
    )

    backend.build(spec, {"Pyramidal": 4, "SCA": 4})

    heterogeneity_targets = [
        nodes.label
        for nodes, params in fake_ngpu.set_status_calls
        if any(isinstance(value, dict) and "array" in value for value in params.values())
    ]
    assert heterogeneity_targets == ["aeif_cond_beta_multisynapse"]


def test_gpu_backend_rejects_unknown_record_type_instead_of_skip() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    backend._nodes = {"Pyramidal": _FakeNodeCollection(1, "Pyramidal")}

    with pytest.raises(KeyError, match="record_types contains unknown"):
        backend.attach_recorders(record_types=["TypoCell"])


def test_gpu_afferent_source_pool_preserves_poisson_superposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_AFFERENT_TOPOLOGY", "source_pool")
    monkeypatch.setenv("CA1_AFFERENT_SOURCE_POOL_SIZE", "100")
    monkeypatch.setenv("CA1_AFFERENT_SOURCE_POOL_INDEGREE", "10")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-source-pool-afferents",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="ECIII_to_Pyramidal",
                post="Pyramidal",
                n_source=1000,
                synapses_per_cell=50.0,
                weight_nS=0.2,
                receptor="AMPA_slow",
                rate_hz=0.5,
            )
        ],
        afferent_topology="source_pool",
    )

    backend.build(spec, {"Pyramidal": 4})

    assert ("poisson_generator", 100, None) in fake_ngpu.create_calls
    pool_rates = [
        params["rate"]
        for nodes, params in fake_ngpu.set_status_calls
        if nodes.label == "poisson_generator"
    ]
    assert pool_rates == [pytest.approx(2.5)]
    afferent_connections = [
        call for call in fake_ngpu.connect_calls
        if call[2] == {"rule": "fixed_indegree", "indegree": 10}
    ]
    assert afferent_connections
    _pre, _post, _conn, syn_spec = afferent_connections[-1]
    assert syn_spec["weight"] == pytest.approx(0.2)
    assert syn_spec["receptor"] == spec.receptors.port_index("AMPA_slow")


def test_gpu_afferent_source_pool_uses_spec_parameters_without_env() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-source-pool-spec-afferents",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=1000,
                synapses_per_cell=50.0,
                weight_nS=0.2,
                receptor="AMPA_fast",
                rate_hz=0.5,
            )
        ],
        afferent_topology="source_pool",
        afferent_source_pool_size=100,
        afferent_source_pool_indegree=10,
    )

    backend.build(spec, {"Pyramidal": 4})

    assert ("poisson_generator", 100, None) in fake_ngpu.create_calls
    pool_rates = [
        params["rate"]
        for nodes, params in fake_ngpu.set_status_calls
        if nodes.label == "poisson_generator"
    ]
    assert pool_rates == [pytest.approx(2.5)]
    afferent_connections = [
        call for call in fake_ngpu.connect_calls
        if call[2] == {"rule": "fixed_indegree", "indegree": 10}
    ]
    assert afferent_connections
    _pre, _post, _conn, syn_spec = afferent_connections[-1]
    assert syn_spec["weight"] == pytest.approx(0.2)
    assert syn_spec["receptor"] == spec.receptors.port_index("AMPA_fast")


def test_gpu_afferent_source_pool_separates_paths_with_distinct_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_AFFERENT_TOPOLOGY", "source_pool")
    monkeypatch.setenv("CA1_AFFERENT_SOURCE_POOL_SIZE", "100")
    monkeypatch.setenv("CA1_AFFERENT_SOURCE_POOL_INDEGREE", "10")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-source-pool-path-specific-afferents",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "SCA": types_mod.CellType(
                name="SCA",
                count=4,
                layers=("SR",),
                params=_params(types_mod),
            ),
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=1000,
                synapses_per_cell=50.0,
                weight_nS=0.2,
                receptor="AMPA_fast",
                rate_hz=0.5,
            ),
            types_mod.Afferent(
                name="CA3_to_SCA",
                post="SCA",
                n_source=1000,
                synapses_per_cell=20.0,
                weight_nS=0.2,
                receptor="AMPA_fast",
                rate_hz=0.5,
            ),
        ],
        afferent_topology="source_pool",
    )

    backend.build(spec, {"Pyramidal": 4, "SCA": 4})

    pool_creates = [
        call for call in fake_ngpu.create_calls
        if call == ("poisson_generator", 100, None)
    ]
    assert len(pool_creates) == 2
    pool_rates = [
        params["rate"]
        for nodes, params in fake_ngpu.set_status_calls
        if nodes.label == "poisson_generator"
    ]
    assert pool_rates == [pytest.approx(2.5), pytest.approx(1.0)]


def test_gpu_afferent_source_pool_rejects_distributed_mpi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_AFFERENT_TOPOLOGY", "source_pool")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    backend._mpi_size = 2
    spec = types_mod.NetworkSpec(
        name="gpu-source-pool-mpi-rejected",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=1000,
                synapses_per_cell=50.0,
                weight_nS=0.2,
                receptor="AMPA_fast",
            )
        ],
        afferent_topology="source_pool",
    )

    with pytest.raises(ValueError, match="source_pool.*single-GPU"):
        backend.build(spec, {"Pyramidal": 2})


def test_gpu_afferent_literal_source_graph_shares_source_pool() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-literal-source-graph",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "SCA": types_mod.CellType(
                name="SCA",
                count=3,
                layers=("SR",),
                params=_params(types_mod),
            ),
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=100,
                synapses_per_cell=10.0,
                weight_nS=0.2,
                synapses_per_connection=2,
                receptor="AMPA_fast",
                rate_hz=0.5,
            ),
            types_mod.Afferent(
                name="CA3_to_SCA",
                post="SCA",
                n_source=100,
                synapses_per_cell=6.0,
                weight_nS=0.3,
                synapses_per_connection=2,
                receptor="AMPA_fast",
                rate_hz=0.5,
            ),
        ],
        afferent_topology="literal_source_graph",
    )

    backend.build(spec, {"Pyramidal": 4, "SCA": 3})

    pool_creates = [
        call for call in fake_ngpu.create_calls
        if call == ("spike_generator", 100, None)
    ]
    assert len(pool_creates) == 1
    assert ("poisson_generator", 100, None) not in fake_ngpu.create_calls
    assert not any(
        nodes.label == "spike_generator" and "rate" in params
        for nodes, params in fake_ngpu.set_status_calls
    )
    afferent_connections = [
        call for call in fake_ngpu.connect_calls
        if call[0].label == "spike_generator"
    ]
    assert len(afferent_connections) == 2
    assert all(
        call[0] is afferent_connections[0][0]
        for call in afferent_connections
    )
    assert [call[2] for call in afferent_connections] == [
        {"rule": "fixed_indegree", "indegree": 5},
        {"rule": "fixed_indegree", "indegree": 3},
    ]
    assert [call[3]["weight"] for call in afferent_connections] == [
        pytest.approx(0.4),
        pytest.approx(0.6),
    ]


def test_gpu_afferent_literal_source_graph_sets_precomputed_spike_trains() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-literal-source-spike-trains",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=4,
                synapses_per_cell=2.0,
                weight_nS=0.2,
                synapses_per_connection=1,
                receptor="AMPA_fast",
                rate_hz=50.0,
            )
        ],
        afferent_topology="literal_source_graph",
    )
    backend.build(spec, {"Pyramidal": 2})
    assert ("spike_generator", 4, None) in fake_ngpu.create_calls
    assert not any(
        nodes.label == "spike_generator" and "rate" in params
        for nodes, params in fake_ngpu.set_status_calls
    )
    fake_ngpu.set_status_calls.clear()

    backend._set_literal_source_spike_trains(duration_s=0.5, seed=7)

    assert len(fake_ngpu.set_status_calls) == 4
    for nodes, params in fake_ngpu.set_status_calls:
        assert nodes.label.startswith("spike_generator[")
        assert len(nodes) == 1
        assert "rate" not in params
        spike_times = params["spike_times"]
        assert spike_times
        # NEST GPU defaults an omitted spike_gen_mul array to 1.0 at
        # calibration time.  Supplying only spike_times avoids a second C call
        # for every literal source.
        assert "spike_gen_mul" not in params
        assert all(0.0 <= time_ms < 500.0 for time_ms in spike_times)
        assert all(
            after - before >= backend._dt_ms
            for before, after in zip(spike_times, spike_times[1:], strict=False)
        )


def test_gpu_afferent_literal_source_graph_keeps_source_spikes_in_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    class ZeroSlotRng:
        def poisson(self, _lam: float, size: int) -> object:
            return gpu_backend_mod.np.array([1, *([0] * (size - 1))])

        def integers(
            self,
            low: int,
            high: int,
            *,
            size: int,
            dtype: object,
        ) -> object:
            assert low == 1
            assert high > low
            return gpu_backend_mod.np.full(size, low, dtype=dtype)

    monkeypatch.setattr(
        gpu_backend_mod.np.random,
        "default_rng",
        lambda _seed: ZeroSlotRng(),
    )
    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="gpu-literal-source-spike-trains-future",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=4,
                synapses_per_cell=2.0,
                weight_nS=0.2,
                synapses_per_connection=1,
                receptor="AMPA_fast",
                rate_hz=50.0,
            )
        ],
        afferent_topology="literal_source_graph",
    )
    backend.build(spec, {"Pyramidal": 2})
    fake_ngpu.set_status_calls.clear()

    backend._set_literal_source_spike_trains(duration_s=0.5, seed=7)

    spike_times = fake_ngpu.set_status_calls[0][1]["spike_times"]
    assert spike_times == [pytest.approx(backend._dt_ms)]


def test_gpu_afferent_literal_source_graph_rejects_distributed_mpi() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    backend._mpi_size = 2
    spec = types_mod.NetworkSpec(
        name="gpu-literal-source-graph-mpi-rejected",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=100,
                synapses_per_cell=10.0,
                weight_nS=0.2,
                synapses_per_connection=2,
                receptor="AMPA_fast",
            )
        ],
        afferent_topology="literal_source_graph",
    )

    with pytest.raises(ValueError, match="literal_source_graph.*single-GPU"):
        backend.build(spec, {"Pyramidal": 2})


def test_gpu_afferent_literal_source_graph_rejects_nonintegral_contacts() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    spec = types_mod.NetworkSpec(
        name="gpu-literal-source-graph-nonintegral",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=100,
                synapses_per_cell=9.0,
                weight_nS=0.2,
                synapses_per_connection=2,
                receptor="AMPA_fast",
            )
        ],
        afferent_topology="literal_source_graph",
    )

    with pytest.raises(ValueError, match="integer multiple"):
        backend.build(spec, {"Pyramidal": 2})


def test_gpu_collect_spikes_rejects_saturated_recorder() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    fake_nodes = _FakeNodeCollection(1, "Pyramidal")
    fake_ngpu.rec_spike_times = [[0.1, 0.2]]
    backend._ngpu = fake_ngpu
    backend._recorders = {"Pyramidal": fake_nodes}
    backend._n_cells = {"Pyramidal": 1}
    backend._max_rec_spikes = 2

    with pytest.raises(RuntimeError, match="spike recorder saturated"):
        backend.collect_spikes()


def test_gpu_afferent_source_pool_rejects_node_cap_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_AFFERENT_TOPOLOGY", "source_pool")
    monkeypatch.setenv("CA1_AFFERENT_SOURCE_POOL_SIZE", "250000")
    monkeypatch.setenv("CA1_AFFERENT_SOURCE_POOL_INDEGREE", "64")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    spec = types_mod.NetworkSpec(
        name="gpu-source-pool-node-cap",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=338_740,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name=f"CA3_to_Pyramidal_{idx}",
                post="Pyramidal",
                n_source=250_000,
                synapses_per_cell=50.0,
                weight_nS=0.2,
                receptor="AMPA_fast",
            )
            for idx in range(4)
        ],
        afferent_topology="source_pool",
        afferent_source_pool_size=250_000,
        afferent_source_pool_indegree=64,
    )

    with pytest.raises(ValueError, match="source_pool.*local node cap"):
        backend.build(spec, {"Pyramidal": 338_740})


def test_gpu_afferent_topology_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_AFFERENT_TOPOLOGY", "unexpected")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    spec = types_mod.NetworkSpec(
        name="gpu-unknown-afferent-topology",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=1,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
    )

    with pytest.raises(ValueError, match="CA1_AFFERENT_TOPOLOGY"):
        backend.build(spec, {"Pyramidal": 1})


def test_gpu_afferent_topology_rejects_env_spec_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    monkeypatch.setenv("CA1_AFFERENT_TOPOLOGY", "source_pool")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    spec = types_mod.NetworkSpec(
        name="gpu-afferent-topology-conflict",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=1,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
        afferent_topology="compound",
    )

    with pytest.raises(ValueError, match="conflicts with NetworkSpec"):
        backend.build(spec, {"Pyramidal": 1})


def test_gpu_backend_can_build_izhikevich_cond_beta_population() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="izh-gpu-model",
        cell_types={
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
        neuron_model="izhikevich_cond_beta",
    )

    backend.build(spec, {"PV_Basket": 2})

    assert ("izhikevich_cond_beta", 2, spec.receptors.n_ports()) in fake_ngpu.create_calls
    merged_params: dict[str, Any] = {}
    for _nodes, params in fake_ngpu.set_status_calls:
        merged_params.update(params)
    assert merged_params["a"] == pytest.approx(0.083696002323701)
    assert merged_params["b"] == pytest.approx(0.2802478181536967)
    assert merged_params["c"] == pytest.approx(-72.84304949826722)
    assert merged_params["d"] == pytest.approx(4.239480143813836)
    assert merged_params["E_rev"] == list(spec.receptors.E_rev)


def test_izhikevich_backend_scales_recurrent_weight_by_post_current_gain() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="izh-recurrent-current-units",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            ),
        },
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=1.0,
                synapses_per_connection=2,
                weight_nS=0.5,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
        neuron_model="izhikevich_cond_beta",
    )

    backend.build(spec, {"Pyramidal": 3, "PV_Basket": 2})

    recurrent = [
        call for call in fake_ngpu.connect_calls
        if call[2] == {"rule": "fixed_indegree", "indegree": 1}
    ][0]
    _pre, _post, _conn, syn_spec = recurrent
    assert syn_spec["weight"] == pytest.approx(1.0 * 0.010192412369208436)


def test_izhikevich_backend_scales_afferent_weight_by_post_current_gain() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="izh-afferent-current-units",
        cell_types={
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=2,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_PV_Basket",
                post="PV_Basket",
                n_source=100,
                synapses_per_cell=3.0,
                weight_nS=0.5,
                receptor="AMPA_fast",
                rate_hz=0.65,
            )
        ],
        neuron_model="izhikevich_cond_beta",
    )

    backend.build(spec, {"PV_Basket": 2})

    afferent_connections = [
        call for call in fake_ngpu.connect_calls
        if call[2] == {"rule": "one_to_one"}
    ]
    assert afferent_connections
    _pre, _post, _conn, syn_spec = afferent_connections[-1]
    assert syn_spec["weight"] == pytest.approx(0.5 * 0.010192412369208436)


def test_gpu_backend_modeldb_fastconn_binned_uses_local_recurrent_windows() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="modeldb-fastconn-binned-gpu",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3_200,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=64,
                layers=("SP",),
                params=_params(types_mod),
            ),
        },
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=32.0,
                synapses_per_connection=2,
                weight_nS=0.25,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
        recurrent_topology="modeldb_fastconn_binned",
    )

    backend.build(spec, {"Pyramidal": 3_200, "PV_Basket": 64})

    recurrent = fake_ngpu.connect_calls
    assert len(recurrent) == 64
    assert all(call[2] == {"rule": "fixed_indegree", "indegree": 32} for call in recurrent)
    assert all(call[0].count < 3_200 for call in recurrent)
    assert sum(call[1].count for call in recurrent) == 64
    assert recurrent[0][0].label != "aeif_cond_beta_multisynapse"
    assert recurrent[0][1].label.endswith("[0:1]")
    assert recurrent[-1][1].label.endswith("[63:64]")
    assert recurrent[0][0].label < recurrent[-1][0].label
    assert all(call[3]["weight"] == pytest.approx(0.5) for call in recurrent)


def test_gpu_backend_modeldb_fastconn_gaussian_binned_uses_distance_rings() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="modeldb-fastconn-gaussian-binned-gpu",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3_200,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=64,
                layers=("SP",),
                params=_params(types_mod),
            ),
        },
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=32.0,
                synapses_per_connection=2,
                weight_nS=0.25,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
        recurrent_topology="modeldb_fastconn_gaussian_binned",
    )

    backend.build(spec, {"Pyramidal": 3_200, "PV_Basket": 64})

    recurrent = fake_ngpu.connect_calls
    assert len(recurrent) > 64
    assert all(call[2]["rule"] == "fixed_indegree" for call in recurrent)
    assert sum(call[1].count * call[2]["indegree"] for call in recurrent) == (
        64 * 32
    )
    assert all(call[0].count < 3_200 for call in recurrent)
    assert sum(call[1].count for call in recurrent) > 64
    assert recurrent[0][1].label.endswith("[0:1]")
    assert recurrent[-1][1].label.endswith("[63:64]")
    assert all(call[3]["weight"] == pytest.approx(0.5) for call in recurrent)


def test_gpu_backend_fixed_indegree_topology_keeps_global_recurrent_connect() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="fixed-indegree-gpu",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3_200,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=64,
                layers=("SP",),
                params=_params(types_mod),
            ),
        },
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=32.0,
                synapses_per_connection=2,
                weight_nS=0.25,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
    )

    backend.build(spec, {"Pyramidal": 3_200, "PV_Basket": 64})

    assert len(fake_ngpu.connect_calls) == 1
    pre, post, conn, syn_spec = fake_ngpu.connect_calls[0]
    assert pre.count == 3_200
    assert post.count == 64
    assert conn == {"rule": "fixed_indegree", "indegree": 32}
    assert syn_spec["weight"] == pytest.approx(0.5)


def test_gpu_backend_partitions_recurrent_network_across_mpi_hosts() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    fake_ngpu.host_id = 1
    fake_ngpu.host_num = 3
    backend._ngpu = fake_ngpu
    backend._mpi_rank = 1
    backend._mpi_size = 3
    backend._host_group = fake_ngpu.host_group
    spec = types_mod.NetworkSpec(
        name="distributed-gpu",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=10,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=7,
                layers=("SP",),
                params=_params(types_mod),
            ),
        },
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=4.0,
                synapses_per_connection=2,
                weight_nS=0.25,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
    )

    backend.build(spec, {"Pyramidal": 10, "PV_Basket": 7})

    assert fake_ngpu.remote_create_calls == [
        (0, "aeif_cond_beta_multisynapse", 4, spec.receptors.n_ports()),
        (1, "aeif_cond_beta_multisynapse", 3, spec.receptors.n_ports()),
        (2, "aeif_cond_beta_multisynapse", 3, spec.receptors.n_ports()),
        (0, "aeif_cond_beta_multisynapse", 3, spec.receptors.n_ports()),
        (1, "aeif_cond_beta_multisynapse", 2, spec.receptors.n_ports()),
        (2, "aeif_cond_beta_multisynapse", 2, spec.receptors.n_ports()),
    ]
    assert backend._n_cells == {"Pyramidal": 3, "PV_Basket": 2}
    assert len(fake_ngpu.connect_calls) == 0
    assert len(fake_ngpu.distributed_connect_calls) == 1
    source_hosts, source_groups, target_hosts, target_groups, indegree, host_group, syn_spec = (
        fake_ngpu.distributed_connect_calls[0]
    )
    assert source_hosts == [0, 1, 2]
    assert [len(group) for group in source_groups] == [4, 3, 3]
    assert target_hosts == [0, 1, 2]
    assert [len(group) for group in target_groups] == [3, 2, 2]
    assert indegree == 4
    assert host_group == fake_ngpu.host_group
    assert syn_spec["weight"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "recurrent_topology",
    ["modeldb_fastconn_binned", "modeldb_fastconn_gaussian_binned"],
)
def test_gpu_backend_rejects_modeldb_fastconn_binned_under_mpi(
    recurrent_topology: str,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    fake_ngpu.host_id = 1
    fake_ngpu.host_num = 2
    backend._ngpu = fake_ngpu
    backend._mpi_rank = 1
    backend._mpi_size = 2
    backend._host_group = fake_ngpu.host_group
    spec = types_mod.NetworkSpec(
        name=f"{recurrent_topology.replace('_', '-')}-mpi-rejected",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=4,
                layers=("SP",),
                params=_params(types_mod),
            ),
        },
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=2.0,
                synapses_per_connection=1,
                weight_nS=0.25,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
        recurrent_topology=recurrent_topology,
    )

    with pytest.raises(ValueError, match=f"{recurrent_topology}.*single-GPU"):
        backend.build(spec, {"Pyramidal": 4, "PV_Basket": 4})


def test_gpu_backend_rejects_distributed_run_with_zero_sized_shard() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    fake_ngpu.host_id = 2
    fake_ngpu.host_num = 3
    backend._ngpu = fake_ngpu
    backend._mpi_rank = 2
    backend._mpi_size = 3
    backend._host_group = fake_ngpu.host_group
    spec = types_mod.NetworkSpec(
        name="too-small-for-distributed-gpu",
        cell_types={
            "O_LM": types_mod.CellType(
                name="O_LM",
                count=2,
                layers=("SO",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
    )

    with pytest.raises(ValueError, match="undersized=\\['O_LM'\\]"):
        backend.build(spec, {"O_LM": 2})


def test_aglif_backend_maps_public_model_to_user_m1_without_current_gain() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="aglif-gpu",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3,
                layers=("SP",),
                params=_params(types_mod),
            ),
            "PV_Basket": types_mod.CellType(
                name="PV_Basket",
                count=3,
                layers=("SP",),
                params=_params(types_mod),
            ),
        },
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=1.0,
                synapses_per_connection=2,
                weight_nS=0.25,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
        neuron_model="aglif_cond_beta",
    )

    backend.build(spec, {"Pyramidal": 3, "PV_Basket": 3})

    assert fake_ngpu.create_calls == [
        ("user_m1", 3, spec.receptors.n_ports()),
        ("user_m1", 3, spec.receptors.n_ports()),
    ]
    first_status = fake_ngpu.set_status_calls[0][1]
    assert first_status["tau_m"] > 0.0
    assert "k_adap" in first_status
    assert "A1" in first_status
    assert "A2" in first_status
    assert "Delta_T" not in first_status
    assert fake_ngpu.connect_calls[0][3]["weight"] == pytest.approx(0.5)


def test_aglif_dend_backend_maps_public_model_to_user_m2_with_compartments() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="aglif-dend-gpu",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
        neuron_model="aglif_dend_cond_beta",
    )

    backend.build(spec, {"Pyramidal": 3})

    assert fake_ngpu.create_calls == [("user_m2", 3, spec.receptors.n_ports())]
    merged_status: dict[str, Any] = {}
    for _nodes, status in fake_ngpu.set_status_calls:
        merged_status.update(status)
    assert merged_status["tau_m"] > 0.0
    assert merged_status["dend_C_frac"] == pytest.approx(0.4)
    assert merged_status["g_c"] > 0.0
    assert merged_status["compartment"] == [1, 1, 0, 1, 1]


def test_aglif_dend_intrinsic_vth_heterogeneity_uses_backend_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: aglif-dend backend status has a different V_th than CellType params.
    aglif_dend_mod = pytest.importorskip("ca1.sim.aglif_dend")
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")
    expected_v_th = aglif_dend_mod.aglif_dend_status("Pyramidal")["V_th"]
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_VTH_SIGMA_MV", "0.0")
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_EL_SIGMA_MV", "0.0")
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_VM_SIGMA_MV", "0.0")
    monkeypatch.setenv(
        "CA1_INTRINSIC_HETEROGENEITY_VTH_SIGMA_MV_PYRAMIDAL",
        "0.000001",
    )
    monkeypatch.setenv("CA1_INTRINSIC_HETEROGENEITY_CLIP_SIGMA", "0.000001")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = types_mod.NetworkSpec(
        name="aglif-dend-vth-heterogeneity-baseline",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
        neuron_model="aglif_dend_cond_beta",
    )

    # When: intrinsic heterogeneity writes per-cell V_th arrays for user_m2.
    backend.build(spec, {"Pyramidal": 3})

    # Then: the array is centered on the aglif-dend backend status baseline.
    array_status = {
        field: payload["array"]
        for nodes, params in fake_ngpu.set_status_calls
        for field, payload in params.items()
        if nodes.label == "user_m2"
        and isinstance(payload, dict)
        and "array" in payload
    }
    assert array_status["V_th"] == pytest.approx(
        [expected_v_th] * 3,
        abs=1e-6,
    )


def test_aglif_backend_rejects_more_than_twenty_receptor_ports() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    backend._ngpu = _FakeNestGpu()
    spec = types_mod.NetworkSpec(
        name="aglif-too-many-ports",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=1,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[],
        receptors=types_mod.ReceptorConfig(
            names=tuple(f"port_{idx}" for idx in range(21)),
            E_rev=tuple(0.0 for _idx in range(21)),
            tau_rise=tuple(1.0 for _idx in range(21)),
            tau_decay=tuple(2.0 for _idx in range(21)),
        ),
        neuron_model="aglif_cond_beta",
    )

    with pytest.raises(ValueError, match="at most 20 receptor ports"):
        backend.build(spec, {"Pyramidal": 1})


def _working_point_spec(types_mod, *, mode: str):
    cell_types = {
        name: types_mod.CellType(
            name=name,
            count=count,
            layers=("SP",),
            params=_params(types_mod),
        )
        for name, count in {
            "Pyramidal": 4,
            "PV_Basket": 3,
            "SCA": 2,
        }.items()
    }
    return types_mod.NetworkSpec(
        name=f"working-point-{mode}",
        cell_types=cell_types,
        projections=[
            types_mod.Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=2.0,
                synapses_per_connection=1,
                weight_nS=0.1,
                receptor="AMPA_fast",
            ),
            types_mod.Projection(
                pre="PV_Basket",
                post="Pyramidal",
                indegree=3.0,
                synapses_per_connection=2,
                weight_nS=0.25,
                receptor="GABA_A_fast",
            ),
            types_mod.Projection(
                pre="SCA",
                post="Pyramidal",
                indegree=2.0,
                synapses_per_connection=1,
                weight_nS=0.2,
                receptor="GABA_A_slow",
            ),
            types_mod.Projection(
                pre="PV_Basket",
                post="SCA",
                indegree=3.0,
                synapses_per_connection=1,
                weight_nS=0.3,
                receptor="GABA_A_fast",
            ),
        ],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=100,
                synapses_per_cell=2.0,
                weight_nS=0.4,
                receptor="AMPA_fast",
            ),
            types_mod.Afferent(
                name="CA3_to_PV_Basket",
                post="PV_Basket",
                n_source=100,
                synapses_per_cell=2.0,
                weight_nS=0.4,
                receptor="AMPA_fast",
            ),
        ],
        working_point_mode=mode,
        working_point_clamp_rates_hz={"PV_Basket": 54.4},
    )


def test_gpu_working_point_clamp_replaces_output_and_preserves_efferent_synapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = _working_point_spec(types_mod, mode="clamp")

    backend.build(spec, spec.scaled_counts())

    assert fake_ngpu.create_calls[:3] == [
        ("aeif_cond_beta_multisynapse", 4, spec.receptors.n_ports()),
        ("spike_generator", 3, None),
        ("aeif_cond_beta_multisynapse", 2, spec.receptors.n_ports()),
    ]
    assert backend._nodes["Pyramidal"].label == "aeif_cond_beta_multisynapse"
    assert backend._nodes["PV_Basket"].label == "spike_generator"
    assert backend._nodes["SCA"].label == "aeif_cond_beta_multisynapse"
    assert backend._working_point_rates_hz == {"PV_Basket": 54.4}

    recurrent = [
        call for call in fake_ngpu.connect_calls
        if call[2].get("rule") == "fixed_indegree"
    ]
    assert len(recurrent) == 3
    assert not any(call[1].label == "spike_generator" for call in recurrent)
    pv_efferents = [call for call in recurrent if call[0].label == "spike_generator"]
    assert len(pv_efferents) == 2
    assert [call[2]["indegree"] for call in pv_efferents] == [3, 3]
    assert [call[3]["weight"] for call in pv_efferents] == [
        pytest.approx(0.5),
        pytest.approx(0.3),
    ]
    assert [call[3]["receptor"] for call in pv_efferents] == [
        spec.receptors.port_index("GABA_A_fast"),
        spec.receptors.port_index("GABA_A_fast"),
    ]
    sca_efferent = next(call for call in recurrent if call[0].label != "spike_generator")
    assert sca_efferent[3]["weight"] == pytest.approx(0.2)
    assert sca_efferent[3]["receptor"] == spec.receptors.port_index("GABA_A_slow")

    afferents = [call for call in fake_ngpu.connect_calls if call[2] == {"rule": "one_to_one"}]
    assert len(afferents) == 1
    assert afferents[0][1] is backend._nodes["Pyramidal"]
    assert ("poisson_generator", 4, None) in fake_ngpu.create_calls
    assert ("poisson_generator", 3, None) not in fake_ngpu.create_calls

    poisson_lambdas: list[float] = []
    rng_seeds: list[int] = []

    class CaptureRng:
        def poisson(self, lam: float, size: int | None = None) -> object:
            poisson_lambdas.append(lam)
            assert size == 3
            return gpu_backend_mod.np.ones(size, dtype=int)

        def integers(
            self,
            low: int,
            high: int,
            *,
            size: int,
            dtype: object,
        ) -> object:
            assert low == 1
            assert high > low
            return gpu_backend_mod.np.full(size, low, dtype=dtype)

    def capture_rng(seed: int) -> CaptureRng:
        rng_seeds.append(seed)
        return CaptureRng()

    monkeypatch.setattr(gpu_backend_mod.np.random, "default_rng", capture_rng)
    fake_ngpu.set_status_calls.clear()
    backend._set_literal_source_spike_trains(duration_s=0.5, seed=123)

    assert poisson_lambdas == [pytest.approx(54.4 * 0.5)]
    assert rng_seeds == [
        gpu_backend_mod._stable_source_seed(123, "working_point:PV_Basket")
    ]
    assert len(fake_ngpu.set_status_calls) == 3
    assert all(
        nodes.label.startswith("spike_generator[")
        and "spike_gen_mul" not in params
        for nodes, params in fake_ngpu.set_status_calls
    )


def test_gpu_working_point_mode_off_is_a_no_op() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    types_mod = pytest.importorskip("ca1.types")

    backend = gpu_backend_mod.NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend._ngpu = fake_ngpu
    spec = _working_point_spec(types_mod, mode="off")

    backend.build(spec, spec.scaled_counts())

    assert fake_ngpu.create_calls[:3] == [
        ("aeif_cond_beta_multisynapse", 4, spec.receptors.n_ports()),
        ("aeif_cond_beta_multisynapse", 3, spec.receptors.n_ports()),
        ("aeif_cond_beta_multisynapse", 2, spec.receptors.n_ports()),
    ]
    assert backend._working_point_rates_hz == {}
    assert not any(key.startswith("working_point:") for key in backend._poisson)
    recurrent = [
        call for call in fake_ngpu.connect_calls
        if call[2].get("rule") == "fixed_indegree"
    ]
    assert len(recurrent) == 4
    assert len([call for call in fake_ngpu.connect_calls if call[2] == {"rule": "one_to_one"}]) == 2
