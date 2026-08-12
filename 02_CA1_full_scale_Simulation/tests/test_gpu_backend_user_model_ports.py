from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, overload

import numpy as np
import pytest

from ca1.sim import gpu_backend as gpu_backend_mod
from ca1.sim.gpu_backend import NestGpuBackend
from ca1.sim.modeldb_positions import (
    MODELDB_NPOLE_ELECTRODE_ROI,
    modeldb_cell_positions,
)
from ca1.sim.npole_lfp import reduced_domain_n_pole_lfp
from ca1.sim.nestgpu_api import (
    NestGpuConnValue,
    NestGpuNodes,
    NestGpuRemoteNodes,
    NestGpuStatusValue,
)
from ca1.types import (
    CellType,
    NetworkSpec,
    NeuronParams,
    Projection,
    ReceptorConfig,
    SimMeta,
)


class _FakeNodes:
    def __init__(self, count: int) -> None:
        self.count: int = count

    def __len__(self) -> int:
        return self.count

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> _FakeNodes: ...

    def __getitem__(self, index: int | slice) -> int | _FakeNodes:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.count)
            return _FakeNodes(len(range(start, stop, step)))
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return index


class _FakeNestGpu:
    def __init__(self) -> None:
        self.seed: int | None = None
        self.resolution_ms: float | None = None
        self.last_status_target: NestGpuNodes | NestGpuRemoteNodes | None = None
        self.last_status_size: int = 0
        self.connect_call_size: int = 0
        self.distributed_call_size: int = 0
        self.record_call_size: int = 0
        self.simulated_ms: float = 0.0
        self.create_calls: list[tuple[str, int, int | None]] = []
        self.remote_create_calls: list[tuple[int, str, int, int | None]] = []
        self.set_status_calls: list[
            tuple[NestGpuNodes | NestGpuRemoteNodes, dict[str, NestGpuStatusValue]]
        ] = []
        self.connect_calls: list[
            tuple[dict[str, NestGpuConnValue], dict[str, NestGpuConnValue]]
        ] = []
        self.create_record_calls: list[
            tuple[str, list[str], list[int], list[int]]
        ] = []
        self.set_record_stride_calls: list[tuple[int, int]] = []
        self.record_data: list[list[float]] = []

    def SetRandomSeed(self, seed: int) -> None:  # noqa: N802
        self.seed = seed
        return None

    def SetTimeResolution(self, resolution_ms: float) -> None:  # noqa: N802
        self.resolution_ms = resolution_ms
        return None

    def Create(self, model: str, count: int = 1, n_ports: int | None = None) -> NestGpuNodes:  # noqa: N802
        self.create_calls.append((model, count, n_ports))
        return _FakeNodes(count)

    def SetStatus(
        self,
        nodes: NestGpuNodes | NestGpuRemoteNodes,
        params: Mapping[str, NestGpuStatusValue],
    ) -> None:  # noqa: N802
        self.last_status_target = nodes
        self.last_status_size = len(params)
        self.set_status_calls.append((nodes, dict(params)))
        return None

    def RemoteCreate(  # noqa: N802
        self,
        host: int,
        model: str,
        count: int = 1,
        n_ports: int | None = None,
    ) -> NestGpuRemoteNodes:
        self.remote_create_calls.append((host, model, count, n_ports))
        return _FakeRemoteNodes(host, _FakeNodes(count))

    def Connect(
        self,
        pre: NestGpuNodes,
        post: NestGpuNodes,
        conn_spec: Mapping[str, NestGpuConnValue],
        syn_spec: Mapping[str, NestGpuConnValue],
    ) -> None:  # noqa: N802
        self.connect_call_size += len(pre) + len(post) + len(conn_spec) + len(syn_spec)
        self.connect_calls.append((dict(conn_spec), dict(syn_spec)))
        return None

    def ConnectMpiInit(self) -> None:  # noqa: N802
        return None

    def HostId(self) -> int:  # noqa: N802
        return 0

    def HostNum(self) -> int:  # noqa: N802
        return 1

    def CreateHostGroup(self, hosts: list[int]) -> int:  # noqa: N802
        self.distributed_call_size += len(hosts)
        return 0

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
        self.distributed_call_size += (
            len(source_hosts)
            + len(source_groups)
            + len(target_hosts)
            + len(target_groups)
            + indegree
            + host_group
            + len(syn_spec)
        )
        return None

    def ActivateRecSpikeTimes(self, nodes: NestGpuNodes, max_spikes: int) -> None:  # noqa: N802
        self.record_call_size += len(nodes) + max_spikes
        return None

    def CreateRecord(  # noqa: N802
        self,
        file_name: str,
        var_names: list[str],
        nodes: list[int],
        ports: list[int],
    ) -> int:
        self.create_record_calls.append((file_name, var_names, nodes, ports))
        return len(self.create_record_calls) - 1

    def SetRecordStride(self, record: int, sampling_stride: int) -> None:  # noqa: N802
        self.set_record_stride_calls.append((record, sampling_stride))
        return None

    def Simulate(self, duration_ms: float) -> None:  # noqa: N802
        self.simulated_ms += duration_ms
        return None

    def GetRecSpikeTimes(self, nodes: NestGpuNodes) -> list[list[float] | None] | None:  # noqa: N802
        self.record_call_size += len(nodes)
        return []

    def GetRecordData(self, record: int) -> list[list[float]]:  # noqa: N802
        assert record == 0
        return self.record_data


class _FakeRemoteNodes:
    def __init__(self, host: int, nodes: _FakeNodes) -> None:
        self.i_host: int = host
        self.node_seq: NestGpuNodes = nodes


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


def _too_many_receptors() -> ReceptorConfig:
    n_ports = 21
    return ReceptorConfig(
        names=tuple(f"port_{idx}" for idx in range(n_ports)),
        E_rev=tuple(0.0 for _idx in range(n_ports)),
        tau_rise=tuple(1.0 for _idx in range(n_ports)),
        tau_decay=tuple(2.0 for _idx in range(n_ports)),
    )


def _receptors(
    names: tuple[str, ...],
    values: Mapping[str, tuple[float, float, float]],
) -> ReceptorConfig:
    return ReceptorConfig(
        names=names,
        E_rev=tuple(values[name][0] for name in names),
        tau_rise=tuple(values[name][1] for name in names),
        tau_decay=tuple(values[name][2] for name in names),
    )


def test_aglif_dend_backend_rejects_more_than_twenty_receptor_ports() -> None:
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    spec = NetworkSpec(
        name="aglif-dend-too-many-ports",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=1,
                layers=("SP",),
                params=_params(),
            )
        },
        projections=[],
        afferents=[],
        receptors=_too_many_receptors(),
        neuron_model="aglif_dend_cond_beta",
    )

    with pytest.raises(ValueError, match="at most 20 receptor ports"):
        backend.build(spec, {"Pyramidal": 1})
    assert fake_ngpu.create_calls == []


def test_gpu_backend_uses_target_local_receptor_tables_for_create_and_connect() -> None:
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    receptor_values = {
        "olm_only": (-77.0, 0.31, 6.1),
        "shared": (-66.5, 0.42, 7.2),
        "pyr_only": (3.25, 0.53, 8.3),
    }
    spec = NetworkSpec(
        name="target-local-receptors",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=2,
                layers=("SP",),
                params=_params(),
            ),
            "O_LM": CellType(
                name="O_LM",
                count=2,
                layers=("SO",),
                params=_params(),
            ),
        },
        projections=[
            Projection(
                pre="O_LM",
                post="Pyramidal",
                indegree=1.0,
                synapses_per_connection=1,
                weight_nS=1.0,
                receptor="shared",
            ),
            Projection(
                pre="Pyramidal",
                post="O_LM",
                indegree=1.0,
                synapses_per_connection=1,
                weight_nS=1.0,
                receptor="shared",
            ),
        ],
        afferents=[],
        receptors=_receptors(("olm_only", "shared", "pyr_only"), receptor_values),
        neuron_model="aeif_cond_beta_multisynapse",
    )
    setattr(spec, "receptor_table_scope", "per_target")
    setattr(
        spec,
        "target_receptors",
        {
            "Pyramidal": _receptors(("shared",), receptor_values),
            "O_LM": _receptors(("olm_only", "shared"), receptor_values),
        },
    )

    backend.build(spec, {"Pyramidal": 2, "O_LM": 2})

    population_creates = [
        call for call in fake_ngpu.create_calls
        if call[0] == "aeif_cond_beta_multisynapse"
    ]
    assert population_creates == [
        ("aeif_cond_beta_multisynapse", 2, 1),
        ("aeif_cond_beta_multisynapse", 2, 2),
    ]
    assert [syn_spec["receptor"] for _conn_spec, syn_spec in fake_ngpu.connect_calls] == [
        0,
        1,
    ]
    receptor_statuses = [
        params for _nodes, params in fake_ngpu.set_status_calls
        if {"E_rev", "tau_rise", "tau_decay"} <= set(params)
    ]
    assert receptor_statuses == [
        {
            "E_rev": [-66.5],
            "tau_rise": [0.42],
            "tau_decay": [7.2],
        },
        {
            "E_rev": [-77.0, -66.5],
            "tau_rise": [0.31, 0.42],
            "tau_decay": [6.1, 7.2],
        },
    ]


def test_gpu_backend_rejects_unknown_model_before_create() -> None:
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    spec = NetworkSpec(
        name="unknown-gpu-model",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=1,
                layers=("SP",),
                params=_params(),
            )
        },
        projections=[],
        afferents=[],
        neuron_model="aeif_cond_beta_multisynapse",
    )
    cast(Any, spec).neuron_model = "eglif_cond_beta"

    with pytest.raises(ValueError, match="unsupported neuron_model"):
        backend.build(spec, {"Pyramidal": 1})
    assert fake_ngpu.create_calls == []


def _aglif_dend_spec() -> NetworkSpec:
    return NetworkSpec(
        name="aglif-dend-lfp",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=1,
                layers=("SP",),
                params=_params(),
            )
        },
        projections=[],
        afferents=[],
        neuron_model="aglif_dend_cond_beta",
    )


def _full_pyramidal_aglif_dend_spec() -> NetworkSpec:
    return NetworkSpec(
        name="aglif-dend-full-lfp",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=311_500,
                layers=("SP",),
                params=_params(),
            )
        },
        projections=[],
        afferents=[],
        scale=1.0,
        cellnumbers_index=101,
        neuron_model="aglif_dend_cond_beta",
    )


def test_gpu_lfp_record_reconstructs_pyramidal_synaptic_current() -> None:
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    setattr(backend, "_dt_ms", 0.1)
    spec = _aglif_dend_spec()

    backend.build(spec, {"Pyramidal": 1})
    backend.attach_recorders()
    fake_ngpu.record_data = [
        [0.0, -65.0, -70.0, -80.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, -65.0, -70.0, -80.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    ]

    lfp, lfp_dt_s = backend.collect_lfp()

    assert fake_ngpu.create_record_calls
    assert fake_ngpu.create_record_calls[0][1][:3] == ["V_m", "V_d", "V_dist"]
    assert fake_ngpu.set_record_stride_calls == [(0, 10)]
    assert lfp_dt_s == pytest.approx(0.001)
    assert lfp == pytest.approx([165.0, 165.0])


def test_gpu_lfp_stride_one_preserves_every_step_time_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CA1_LFP_RECORD_EVERY", "1")
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    setattr(backend, "_dt_ms", 0.1)

    backend.build(_aglif_dend_spec(), {"Pyramidal": 1})
    backend.attach_recorders()
    fake_ngpu.record_data = [
        [0.0, -65.0, -70.0, -80.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.1, -65.0, -70.0, -80.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    ]

    backend.run(100.0)
    _lfp, lfp_dt_s = backend.collect_lfp()

    assert fake_ngpu.set_record_stride_calls == [(0, 1)]
    assert fake_ngpu.simulated_ms == pytest.approx(100.0)
    assert lfp_dt_s == pytest.approx(0.0001)


def test_gpu_lfp_stride_one_accepts_native_float32_timestamp_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CA1_LFP_RECORD_EVERY", "1")
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    setattr(backend, "_dt_ms", 0.1)

    backend.build(_aglif_dend_spec(), {"Pyramidal": 1})
    backend.attach_recorders()
    payload = [-65.0, -70.0, -80.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    fake_ngpu.record_data = [
        [float(np.float32(time_ms)), *payload]
        for time_ms in (127.8, 127.9, 128.0)
    ]

    _lfp, lfp_dt_s = backend.collect_lfp()

    assert lfp_dt_s == pytest.approx(0.0001)


def test_gpu_lfp_sample_selection_is_seeded_spatial_and_counted() -> None:
    positions = np.column_stack(
        (
            np.linspace(0.0, 1100.0, 100),
            np.full(100, 100.0),
            np.full(100, 120.0),
        )
    )

    first = gpu_backend_mod._spatially_distributed_lfp_sample_indices(
        positions,
        8,
        seed=12345,
    )
    repeated = gpu_backend_mod._spatially_distributed_lfp_sample_indices(
        positions,
        8,
        seed=12345,
    )

    np.testing.assert_array_equal(first, repeated)
    assert first.size == 8
    assert np.unique(first).size == 8
    assert np.ptp(positions[first, 0]) > 900.0
    assert np.all(
        np.linalg.norm(
            positions[first] - np.asarray(MODELDB_NPOLE_ELECTRODE_ROI.center_um),
            axis=1,
        )
        < MODELDB_NPOLE_ELECTRODE_ROI.radius_um
    )


def test_gpu_full_lfp_record_returns_modeldb_reduced_n_pole_lfp() -> None:
    # Given: a full-scale canonical Pyramidal LFP recording with two sampled cells.
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    setattr(backend, "_spec", _full_pyramidal_aglif_dend_spec())
    setattr(backend, "_n_cells", {"Pyramidal": 311_500})
    setattr(backend, "_lfp_record", 0)
    setattr(backend, "_lfp_record_every", 1)
    setattr(backend, "_lfp_sample_count", 2)
    setattr(backend, "_lfp_vm_columns", [1, 4])
    setattr(backend, "_lfp_vd_columns", [2, 5])
    setattr(backend, "_lfp_vdist_columns", [3, 6])
    all_positions = modeldb_cell_positions({"Pyramidal": 311_500})["Pyramidal"]
    sample_indices = gpu_backend_mod._spatially_distributed_lfp_sample_indices(
        all_positions,
        2,
        seed=backend._spec.seed,  # type: ignore[union-attr]
    )
    sample_positions = all_positions[sample_indices]
    setattr(backend, "_lfp_sample_indices", sample_indices)
    setattr(backend, "_lfp_sample_positions_um", sample_positions)
    setattr(
        backend,
        "_lfp_g_columns",
        {
            7: gpu_backend_mod._LfpConductanceColumn(
                sample_idx=0,
                port_idx=0,
                compartment=0.0,
                e_rev_mv=0.0,
            ),
            8: gpu_backend_mod._LfpConductanceColumn(
                sample_idx=1,
                port_idx=0,
                compartment=2.0,
                e_rev_mv=0.0,
            ),
        },
    )
    fake_ngpu.record_data = [
        [0.0, -65.0, -70.0, -75.0, -65.0, -70.0, -80.0, 1.0, 2.0],
        [0.1, -65.0, -70.0, -75.0, -65.0, -70.0, -80.0, 1.0, 2.0],
    ]

    # When: LFP is collected for a full-scale canonical run.
    lfp, lfp_dt_s = backend.collect_lfp()

    # Then: the signal is the reduced-domain N-pole weighted sum, not mean current.
    currents = np.asarray([[65.0, 160.0], [65.0, 160.0]], dtype=np.float64)
    expected = reduced_domain_n_pole_lfp(
        currents,
        sample_positions,
        MODELDB_NPOLE_ELECTRODE_ROI,
    )
    assert getattr(backend, "_last_lfp_proxy") == "modeldb_n_pole_reduced_domain_lfp"
    assert getattr(backend, "_last_lfp_provenance")[
        "lfp.modeldb_n_pole_reduced_domain.sampled_cells"
    ] == "2"
    assert getattr(backend, "_last_lfp_provenance")[
        "lfp.modeldb_n_pole_reduced_domain.selected_cells"
    ] == "2"
    assert getattr(backend, "_last_lfp_provenance")[
        "lfp.modeldb_n_pole_reduced_domain.sampled_cell_indices_zero_based"
    ] == ",".join(str(int(idx)) for idx in sample_indices)
    assert getattr(backend, "_last_lfp_provenance")[
        "lfp.modeldb_n_pole_reduced_domain.compartment_voltages"
    ] == "soma:V_m;proximal:V_d;distal:V_dist"
    assert lfp_dt_s == pytest.approx(0.0001)
    assert lfp is not None
    np.testing.assert_allclose(lfp, expected)
    assert not np.allclose(lfp, currents.mean(axis=1))


def test_gpu_modeldb_n_pole_lfp_provenance_record_is_stamped() -> None:
    # Given: full-tier metadata and a modeldb N-pole LFP proxy.
    meta = SimMeta(
        duration_s=1.0,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 311_500},
        scale=1.0,
        seed=1,
        backend="gpu",
        config_name="full",
        parameter_provenance={"network.neuron_model": "aglif_dend_cond_beta"},
    )

    # When: GPU backend stamps result metadata for the collected proxy.
    stamped = gpu_backend_mod._meta_with_lfp_proxy(
        meta,
        "modeldb_n_pole_reduced_domain_lfp",
    )

    # Then: validation can audit the LFP path from explicit provenance.
    assert stamped.lfp_proxy == "modeldb_n_pole_reduced_domain_lfp"
    assert stamped.parameter_provenance["lfp.modeldb_n_pole_reduced_domain"] == (
        "modeldb-n-pole-reduced-domain-lfp"
    )
    assert "lfp.modeldb_n_pole_reduced_domain" not in meta.parameter_provenance


def test_gpu_lfp_record_uses_configured_source_location_table(
    tmp_path: Path,
) -> None:
    target = "GABA_A_slow__em60__tr0p11__td9p7__distal__dend"
    transfer_table = tmp_path / "source_location_transfer.json"
    transfer_table.write_text(
        json.dumps([
            {
                "pre": "O_LM",
                "post": "Pyramidal",
                "receptor": "GABA_A_slow",
                "port": target,
                "loc": "dist",
                "aglif_compartment": "dend",
                "transfer_scale": 1.0,
            }
        ]),
        encoding="utf-8",
    )
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    spec = NetworkSpec(
        name="aglif-dend-lfp-configured-source-location-table",
        cell_types={
            "Pyramidal": CellType(
                name="Pyramidal",
                count=1,
                layers=("SP",),
                params=_params(),
            )
        },
        projections=[
            Projection(
                pre="O_LM",
                post="Pyramidal",
                indegree=1.0,
                synapses_per_connection=1,
                weight_nS=1.0,
                receptor=target,
            )
        ],
        afferents=[],
        receptors=ReceptorConfig(
            names=(target,),
            E_rev=(-60.0,),
            tau_rise=(0.11,),
            tau_decay=(9.7,),
        ),
        neuron_model="aglif_dend_cond_beta",
        source_location_transfer_table=str(transfer_table),
    )

    backend.build(spec, {"Pyramidal": 1})
    backend.attach_recorders()

    assert fake_ngpu.create_record_calls


def test_gpu_lfp_record_failure_is_not_silent_fallback() -> None:
    backend = NestGpuBackend()
    fake_ngpu = _FakeNestGpu()
    setattr(backend, "_ngpu", fake_ngpu)
    spec = _aglif_dend_spec()

    backend.build(spec, {"Pyramidal": 1})
    backend.attach_recorders()
    fake_ngpu.record_data = [[0.0, -65.0]]

    with pytest.raises(RuntimeError, match="malformed data"):
        backend.collect_lfp()
