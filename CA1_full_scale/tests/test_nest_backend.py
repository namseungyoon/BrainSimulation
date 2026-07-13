from __future__ import annotations

from typing import Any

import pytest

nest_backend_mod = pytest.importorskip("ca1.sim.nest_backend")
types_mod = pytest.importorskip("ca1.types")


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


class _FakeNodeCollection:
    def __init__(self, node_ids: list[int]) -> None:
        self._node_ids = node_ids

    def __len__(self) -> int:
        return len(self._node_ids)

    def tolist(self) -> list[int]:
        return list(self._node_ids)


class _FakeNest:
    def __init__(self) -> None:
        self._next_id = 1
        self.connect_calls = 0
        self.create_calls: list[tuple[str, int]] = []
        self.status_events: dict[str, list[float]] = {"I_syn": [1.0]}
        self.connections: list[
            tuple[object, object, dict[str, Any] | None, dict[str, Any] | None]
        ] = []
        self.status_error: RuntimeError | None = None

    def Create(self, _model, count=1, params=None) -> _FakeNodeCollection:  # noqa: N802
        del params
        self.create_calls.append((str(_model), int(count)))
        node_ids = list(range(self._next_id, self._next_id + int(count)))
        self._next_id += int(count)
        return _FakeNodeCollection(node_ids)

    def Connect(self, pre, post, conn_spec=None, syn_spec=None) -> None:  # noqa: N802
        self.connections.append((pre, post, conn_spec, syn_spec))
        self.connect_calls += 1
        if conn_spec is None:
            return
        if pre is post and not bool(conn_spec.get("allow_autapses", True)) and int(
            conn_spec["indegree"]
        ) >= len(pre):
            raise AssertionError("impossible fixed_indegree self-projection without autapses")

    def GetStatus(self, _node, _key) -> list[dict[str, list[float]]]:  # noqa: N802
        if self.status_error is not None:
            raise self.status_error
        return [self.status_events]


def test_build_skips_one_cell_self_projection_when_autapses_are_disabled() -> None:
    backend = nest_backend_mod.NestBackend()
    backend._nest = _FakeNest()
    spec = types_mod.NetworkSpec(
        name="one-cell-self-projection",
        cell_types={
            "Bistratified": types_mod.CellType(
                name="Bistratified",
                count=1,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[
            types_mod.Projection(
                pre="Bistratified",
                post="Bistratified",
                indegree=1.0,
                synapses_per_connection=1,
                weight_nS=1.0,
                receptor="GABA_A_slow",
            )
        ],
        afferents=[],
    )

    backend.build(spec, {"Bistratified": 1})

    assert backend._nest.connect_calls == 0


def test_lfp_recorder_attaches_to_canonical_pyramidal_population() -> None:
    backend = nest_backend_mod.NestBackend()
    fake_nest = _FakeNest()
    backend._nest = fake_nest
    pyr_pop = _FakeNodeCollection([1, 2, 3])
    backend._populations = {"Pyramidal": pyr_pop}

    backend.attach_recorders(record_types=[])

    assert backend._record_lfp is True
    assert any(post is pyr_pop for _pre, post, _conn, _syn in fake_nest.connections)


@pytest.mark.parametrize(
    ("events", "status_error", "pyramidal_ids", "message"),
    [
        ({"I_syn": [1.0]}, RuntimeError("backend read failed"), [1], "Failed to collect"),
        ({"I_syn": []}, None, [1], "empty I_syn"),
        ({"I_syn": [1.0, 2.0, 3.0]}, None, [1, 2], "malformed I_syn"),
    ],
)
def test_collect_lfp_raises_when_attached_recorder_data_is_invalid(
    events: dict[str, list[float]],
    status_error: RuntimeError | None,
    pyramidal_ids: list[int],
    message: str,
) -> None:
    backend = nest_backend_mod.NestBackend()
    fake_nest = _FakeNest()
    fake_nest.status_events = events
    fake_nest.status_error = status_error
    backend._nest = fake_nest
    backend._record_lfp = True
    backend._multimeter = object()
    backend._populations = {"Pyramidal": _FakeNodeCollection(pyramidal_ids)}
    with pytest.raises(RuntimeError, match=message):
        backend.collect_lfp()


def test_nest_backend_rejects_unknown_record_type_instead_of_skip() -> None:
    backend = nest_backend_mod.NestBackend()
    fake_nest = _FakeNest()
    backend._nest = fake_nest
    backend._populations = {"Pyramidal": _FakeNodeCollection([1])}

    with pytest.raises(KeyError, match="record_types contains unknown"):
        backend.attach_recorders(record_types=["TypoCell"])
    assert fake_nest.connections == []


def test_afferent_poisson_drive_uses_independent_source_per_post_cell() -> None:
    backend = nest_backend_mod.NestBackend()
    fake_nest = _FakeNest()
    backend._nest = fake_nest
    spec = types_mod.NetworkSpec(
        name="independent-afferents",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=204_700,
                synapses_per_cell=10.0,
                weight_nS=0.2,
                receptor="AMPA_fast",
                rate_hz=0.65,
                delay_ms=7.0,
            )
        ],
    )

    backend.build(spec, {"Pyramidal": 3})

    assert ("poisson_generator", 3) in fake_nest.create_calls
    afferent_connections = [
        call for call in fake_nest.connections
        if call[2] == {"rule": "one_to_one"}
    ]
    assert afferent_connections
    _pre, _post, _conn, syn_spec = afferent_connections[-1]
    assert syn_spec is not None
    assert syn_spec["weight"] == pytest.approx(0.2)
    assert syn_spec["delay"] == pytest.approx(7.0)


def test_nest_backend_rejects_literal_source_graph_instead_of_compound_fallback() -> None:
    backend = nest_backend_mod.NestBackend()
    backend._nest = _FakeNest()
    spec = types_mod.NetworkSpec(
        name="literal-source-graph-not-cpu",
        cell_types={
            "Pyramidal": types_mod.CellType(
                name="Pyramidal",
                count=3,
                layers=("SP",),
                params=_params(types_mod),
            )
        },
        projections=[],
        afferents=[
            types_mod.Afferent(
                name="CA3_to_Pyramidal",
                post="Pyramidal",
                n_source=204_700,
                synapses_per_cell=10.0,
                weight_nS=0.2,
                synapses_per_connection=2,
                receptor="AMPA_fast",
            )
        ],
        afferent_topology="literal_source_graph",
    )

    with pytest.raises(ValueError, match="literal_source_graph"):
        backend.build(spec, {"Pyramidal": 3})


def test_nest_backend_rejects_modeldb_fastconn_binned_instead_of_fixed_indegree_fallback() -> None:
    backend = nest_backend_mod.NestBackend()
    fake_nest = _FakeNest()
    backend._nest = fake_nest
    spec = types_mod.NetworkSpec(
        name="modeldb-fastconn-binned-not-cpu",
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
                synapses_per_connection=1,
                weight_nS=0.2,
                receptor="AMPA_fast",
            )
        ],
        afferents=[],
        recurrent_topology="modeldb_fastconn_binned",
    )

    with pytest.raises(ValueError, match="recurrent_topology='fixed_indegree'"):
        backend.build(spec, {"Pyramidal": 3, "PV_Basket": 3})
    assert fake_nest.connect_calls == 0


def test_nest_backend_rejects_per_target_receptors_instead_of_global_fallback() -> None:
    backend = nest_backend_mod.NestBackend()
    fake_nest = _FakeNest()
    backend._nest = fake_nest
    spec = types_mod.NetworkSpec(
        name="per-target-receptors-not-cpu",
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
            names=("global_only", "local_only"),
            E_rev=(0.0, 0.0),
            tau_rise=(1.0, 1.0),
            tau_decay=(2.0, 2.0),
        ),
        target_receptors={
            "Pyramidal": types_mod.ReceptorConfig(
                names=("local_only",),
                E_rev=(0.0,),
                tau_rise=(1.0,),
                tau_decay=(2.0,),
            )
        },
        receptor_table_scope="per_target",
    )

    with pytest.raises(ValueError, match="receptor_table_scope='per_target'"):
        backend.build(spec, {"Pyramidal": 1})
    assert fake_nest.create_calls == []


def test_nest_backend_rejects_non_aeif_model_before_unused_param_fallback() -> None:
    backend = nest_backend_mod.NestBackend()
    fake_nest = _FakeNest()
    backend._nest = fake_nest
    spec = types_mod.NetworkSpec(
        name="aglif-dend-not-cpu-aeif",
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
        neuron_model="aglif_dend_cond_beta",
    )

    with pytest.raises(ValueError, match="NestBackend only supports"):
        backend.build(spec, {"Pyramidal": 1})
    assert fake_nest.create_calls == []
