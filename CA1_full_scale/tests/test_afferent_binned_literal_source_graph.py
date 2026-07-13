from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, overload, override

from ca1.config import build_network_spec
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.sim.gpu_backend import NestGpuBackend
from ca1.sim.nestgpu_api import (
    NestGpuConnValue,
    NestGpuModule,
    NestGpuNodes,
    NestGpuRemoteNodes,
    NestGpuStatusValue,
)
from ca1.types import Afferent, CellType, NetworkSpec, NeuronParams
from ca1.validation.network_provenance import final_tier_network_structure_blockers

_ConnectCall: TypeAlias = tuple[
    NestGpuNodes,
    NestGpuNodes,
    dict[str, NestGpuConnValue],
    dict[str, NestGpuConnValue],
]
_SetStatusCall: TypeAlias = tuple[
    NestGpuNodes | NestGpuRemoteNodes,
    dict[str, NestGpuStatusValue],
]


class _FakeNodeCollection(NestGpuNodes):
    def __init__(self, count: int, label: str = "") -> None:
        self.count: int = count
        self.label: str = label

    @override
    def __len__(self) -> int:
        return self.count

    @overload
    def __getitem__(self, key: int) -> int: ...

    @overload
    def __getitem__(self, key: slice) -> "_FakeNodeCollection": ...

    @override
    def __getitem__(self, key: int | slice) -> int | _FakeNodeCollection:
        match key:
            case int():
                return key
            case slice():
                return self._slice(key)

    def _slice(self, key: slice) -> "_FakeNodeCollection":
        start, stop, _ = key.indices(self.count)
        return _FakeNodeCollection(stop - start, f"{self.label}[{start}:{stop}]")


@dataclass(frozen=True, slots=True)
class _FakeRemoteNodes(NestGpuRemoteNodes):
    i_host: int
    node_seq: NestGpuNodes


class _FakeNestGpu:
    def __init__(self) -> None:
        self.create_calls: list[tuple[str, int, int | None]] = []
        self.connect_calls: list[_ConnectCall] = []
        self.connect_labels: list[tuple[str, str]] = []
        self.set_status_calls: list[_SetStatusCall] = []
        self.random_seed: int | None = None
        self.time_resolution_ms: float | None = None

    def SetRandomSeed(self, seed: int) -> None:  # noqa: N802
        self.random_seed = seed

    def SetTimeResolution(self, resolution_ms: float) -> None:  # noqa: N802
        self.time_resolution_ms = resolution_ms

    def Create(  # noqa: N802
        self,
        model: str,
        count: int = 1,
        n_ports: int | None = None,
    ) -> _FakeNodeCollection:
        self.create_calls.append((model, int(count), n_ports))
        return _FakeNodeCollection(int(count), model)

    def RemoteCreate(  # noqa: N802
        self,
        host: int,
        model: str,
        count: int = 1,
        n_ports: int | None = None,
    ) -> _FakeRemoteNodes:
        self.create_calls.append((model, int(count), n_ports))
        return _FakeRemoteNodes(
            i_host=host,
            node_seq=_FakeNodeCollection(int(count), f"{model}@{host}"),
        )

    @overload
    def SetStatus(  # noqa: N802
        self,
        nodes: NestGpuNodes | NestGpuRemoteNodes,
        params: Mapping[str, NestGpuStatusValue],
    ) -> None: ...

    @overload
    def SetStatus(  # noqa: N802
        self,
        nodes: NestGpuNodes | NestGpuRemoteNodes,
        params: str,
        val: NestGpuStatusValue,
    ) -> None: ...

    def SetStatus(  # noqa: N802
        self,
        nodes: NestGpuNodes | NestGpuRemoteNodes,
        params: Mapping[str, NestGpuStatusValue] | str,
        val: NestGpuStatusValue | None = None,
    ) -> None:
        match params:
            case str():
                if val is None:
                    raise ValueError("string SetStatus requires a value")
                status = {params: val}
            case _:
                status = dict(params)
        self.set_status_calls.append((nodes, status))

    def Connect(  # noqa: N802
        self,
        pre: NestGpuNodes,
        post: NestGpuNodes,
        conn_spec: Mapping[str, NestGpuConnValue],
        syn_spec: Mapping[str, NestGpuConnValue],
    ) -> None:
        self.connect_calls.append((pre, post, dict(conn_spec), dict(syn_spec)))
        self.connect_labels.append((_node_label(pre), _node_label(post)))

    def ConnectMpiInit(self) -> None:  # noqa: N802
        return None

    def HostId(self) -> int:  # noqa: N802
        return 0

    def HostNum(self) -> int:  # noqa: N802
        return 1

    def CreateHostGroup(self, hosts: list[int]) -> int:  # noqa: N802
        return len(hosts)

    def ConnectDistributedFixedIndegree(  # noqa: N802
        self,
        source_hosts: list[int],
        source_groups: list[NestGpuNodes],
        target_hosts: list[int],
        target_groups: list[NestGpuNodes],
        indegree: int,
        host_group: int,
        syn_spec: Mapping[str, NestGpuConnValue],
    ) -> None:
        _ = (
            source_hosts,
            source_groups,
            target_hosts,
            target_groups,
            indegree,
            host_group,
            syn_spec,
        )

    def ActivateRecSpikeTimes(  # noqa: N802
        self,
        nodes: NestGpuNodes,
        max_spikes: int,
    ) -> None:
        _ = (nodes, max_spikes)

    def CreateRecord(  # noqa: N802
        self,
        file_name: str,
        var_names: list[str],
        nodes: list[int],
        ports: list[int],
    ) -> int:
        _ = (file_name, var_names, nodes, ports)
        return 0

    def Simulate(self, duration_ms: float) -> None:  # noqa: N802
        _ = duration_ms

    def GetRecSpikeTimes(  # noqa: N802
        self,
        nodes: NestGpuNodes,
    ) -> list[list[float] | None] | None:
        _ = nodes
        return None

    def GetRecordData(self, record: int) -> list[list[float]]:  # noqa: N802
        _ = record
        return []


class _TestNestGpuBackend(NestGpuBackend):
    _ngpu: NestGpuModule | None

    def install_ngpu(self, fake_ngpu: _FakeNestGpu) -> None:
        self._ngpu = fake_ngpu


def _fixed_indegree(conn_spec: Mapping[str, NestGpuConnValue]) -> int:
    value = conn_spec["indegree"]
    match value:
        case bool():
            raise AssertionError("fixed_indegree indegree must be an int")
        case int():
            return value
        case _:
            raise AssertionError("fixed_indegree indegree must be an int")


def _node_label(nodes: NestGpuNodes) -> str:
    match nodes:
        case _FakeNodeCollection():
            return nodes.label
        case _:
            raise AssertionError("test fake expected _FakeNodeCollection")


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


def test_literal_source_graph_binned_is_explicit_final_ineligible_topology() -> None:
    # Given: a full-scale-like spec asks for source-domain binned afferents.
    spec = build_network_spec({
        "name": "binned_source_domain_probe",
        "neuron_model": "aglif_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "afferent_topology": "literal_source_graph_binned",
        "recurrent_topology": "modeldb_fastconn_binned",
        "afferent_rate_hz": 0.65,
    })

    # When: final-tier structural provenance is checked.
    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_network_structure_blockers(provenance, spec.scaled_counts())

    # Then: the topology is visible and rejected until audited as final-equivalent.
    assert provenance["network.afferent_topology"] == "literal_source_graph_binned"
    assert provenance["network.afferent_poisson_rule"] == (
        "literal_shared_source_graph_gaussian_binned_fastconn"
    )
    assert provenance["network.afferent_source_driver"] == (
        "precomputed_poisson_spike_generator"
    )
    assert any("literal_source_graph_binned" in blocker for blocker in blockers)


def test_gpu_literal_source_graph_binned_uses_source_and_target_slices() -> None:
    # Given: a tiny binned literal-source graph with enough cells for bins.
    backend = _TestNestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    backend.install_ngpu(fake_ngpu)
    spec = NetworkSpec(
        name="gpu-binned-literal-source-graph",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=16,
                layers=("SP",),
                params=_params(),
            )
        },
        projections=[],
        afferents=[
            Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=64,
                synapses_per_cell=8.0,
                weight_nS=0.2,
                synapses_per_connection=2,
                receptor="AMPA_fast",
                rate_hz=0.65,
            )
        ],
        afferent_topology="literal_source_graph_binned",
    )

    # When: the GPU backend builds afferent connections.
    backend.build(spec, {"Pyramidal": 16})

    # Then: it reuses one literal source pool but connects spatial slices.
    assert ("spike_generator", 64, None) in fake_ngpu.create_calls
    assert len(fake_ngpu.connect_calls) > 1
    assert all(
        source.startswith("spike_generator[")
        for source, _ in fake_ngpu.connect_labels
    )
    assert all("[" in target for _, target in fake_ngpu.connect_labels)
    assert sum(_fixed_indegree(call[2]) for call in fake_ngpu.connect_calls) >= 4
