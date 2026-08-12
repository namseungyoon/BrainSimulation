from __future__ import annotations

import json
from pathlib import Path

import pytest


def _receptor_prefix(receptor: str) -> str:
    return receptor.split("__", maxsplit=1)[0]


def test_load_afferents_preserves_source_receptor_from_connectivity() -> None:
    synapses_mod = pytest.importorskip("ca1.params.synapses")

    afferents = {aff.name: aff for aff in synapses_mod.load_afferents()}

    assert afferents["ECIII_to_Pyramidal"].receptor.startswith("AMPA_slow__")
    assert afferents["ECIII_to_SCA"].receptor.startswith("AMPA_slow__")
    assert afferents["CA3_to_Pyramidal"].receptor.startswith("AMPA_fast__")
    assert afferents["CA3_to_Ivy"].receptor.startswith("AMPA_fast__")


def test_load_afferents_rejects_unknown_source_instead_of_nsource_fallback(
    tmp_path: Path,
) -> None:
    synapses_mod = pytest.importorskip("ca1.params.synapses")
    path = tmp_path / "connectivity.json"
    path.write_text(
        json.dumps({
            "populations_used": {
                "ca3cell": 204700,
                "eccell": 250000,
            },
            "afferents": {
                "septum_to_Pyramidal": {
                    "presynaptic": "Septum",
                    "postsynaptic": "Pyramidal",
                    "indegree_true": 1.0,
                    "weight_nS": 0.1,
                },
            },
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown afferent source"):
        synapses_mod.load_afferents(path=path)


def test_receptor_config_exposes_distinct_fast_and_slow_ampa_ports() -> None:
    synapses_mod = pytest.importorskip("ca1.params.synapses")

    receptors = synapses_mod.load_receptor_config(variant=120)
    fast_ports = [name for name in receptors.names if name.startswith("AMPA_fast__")]
    slow_ports = [name for name in receptors.names if name.startswith("AMPA_slow__")]

    assert fast_ports
    assert slow_ports
    assert set(fast_ports).isdisjoint(slow_ports)
    assert all(receptors.E_rev[receptors.port_index(name)] == pytest.approx(0.0)
               for name in slow_ports)


def test_neurogliaform_gabab_weight_matches_modeldb_ab_scale() -> None:
    synapses_mod = pytest.importorskip("ca1.params.synapses")

    # Given: recurrent projections with Neurogliaform co-release.
    projections = synapses_mod.load_projections()
    posts = sorted({proj.post for proj in projections if proj.pre == "Neurogliaform"})

    # When: each GABA_B component is paired with its GABA_A component.
    projection_pairs = [
        (
            next(
                proj for proj in projections
                if (
                    proj.pre == "Neurogliaform"
                    and proj.post == post
                    and _receptor_prefix(proj.receptor) == "GABA_A_slow"
                )
            ),
            next(
                proj for proj in projections
                if (
                    proj.pre == "Neurogliaform"
                    and proj.post == post
                    and _receptor_prefix(proj.receptor) == "GABA_B"
                )
            ),
        )
        for post in posts
    ]

    # Then: ExpGABAab B-component weight is exactly A-component weight / 3.37.
    assert projection_pairs
    for gaba_a, gaba_b in projection_pairs:
        assert gaba_b.weight_nS == gaba_a.weight_nS / 3.37, (
            f"{gaba_b.pre}->{gaba_b.post} GABA_B weight {gaba_b.weight_nS} "
            f"must equal GABA_A weight {gaba_a.weight_nS} / 3.37"
        )


def test_modeldb_syndata_pair_kinetics_are_preserved() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "pair_kinetics",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
    })

    ca3_to_ivy = next(aff for aff in spec.afferents if aff.name == "CA3_to_Ivy")
    ca3_ivy_port = spec.receptors.port_index(ca3_to_ivy.receptor)
    assert ca3_to_ivy.synapses_per_connection == 2
    assert spec.receptors.tau_rise[ca3_ivy_port] == pytest.approx(2.0)
    assert spec.receptors.tau_decay[ca3_ivy_port] == pytest.approx(6.3)

    pyr_to_ivy = next(
        proj for proj in spec.projections
        if proj.pre == "Pyramidal" and proj.post == "Ivy"
    )
    pyr_ivy_port = spec.receptors.port_index(pyr_to_ivy.receptor)
    assert spec.receptors.tau_rise[pyr_ivy_port] == pytest.approx(0.3)
    assert spec.receptors.tau_decay[pyr_ivy_port] == pytest.approx(0.6)


def test_syndata_137_preserves_ngf_pyramidal_gabaa_reversal() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "syndata_137_reversal",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 137,
    })

    ngf_to_pyramidal = next(
        proj for proj in spec.projections
        if (
            proj.pre == "Neurogliaform"
            and proj.post == "Pyramidal"
            and _receptor_prefix(proj.receptor) == "GABA_A_slow"
        )
    )
    port = spec.receptors.port_index(ngf_to_pyramidal.receptor)

    assert "em75" in ngf_to_pyramidal.receptor
    assert spec.receptors.E_rev[port] == pytest.approx(-75.0)
    assert spec.receptors.tau_rise[port] == pytest.approx(9.0)
    assert spec.receptors.tau_decay[port] == pytest.approx(39.0)


def test_receptor_config_fits_nest_gpu_port_budget() -> None:
    synapses_mod = pytest.importorskip("ca1.params.synapses")

    for variant in (120, 137):
        receptors = synapses_mod.load_receptor_config(variant=variant)

        assert receptors.n_ports() <= 20
        assert receptors.port_index("GABA_B__em90__tr180__td200") < 20


def test_syndata_137_port_budget_preserves_em75_gabaa() -> None:
    synapses_mod = pytest.importorskip("ca1.params.synapses")

    receptors = synapses_mod.load_receptor_config(variant=137)

    assert receptors.n_ports() == 20
    assert "GABA_A_slow__em75__tr9__td39" in receptors.names
    assert "GABA_B__em90__tr180__td200" in receptors.names


def test_compartment_aware_syndata_splits_duplicate_cck_locations() -> None:
    config_mod = pytest.importorskip("ca1.config")

    base_config = {
        "name": "base",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 137,
    }
    baseline = config_mod.build_network_spec(base_config)
    split = config_mod.build_network_spec({
        **base_config,
        "name": "compartment-aware",
        "compartment_aware_synapses": True,
    })

    baseline_cck_pyr = [
        proj for proj in baseline.projections
        if proj.pre == "CCK_Basket" and proj.post == "Pyramidal"
    ]
    split_cck_pyr = [
        proj for proj in split.projections
        if proj.pre == "CCK_Basket" and proj.post == "Pyramidal"
    ]

    assert len(baseline_cck_pyr) == 1
    assert len(split_cck_pyr) == 2
    assert {proj.receptor.rsplit("__", maxsplit=1)[-1] for proj in split_cck_pyr} == {
        "dend",
        "soma",
    }
    assert sum(proj.indegree for proj in split_cck_pyr) == pytest.approx(
        baseline_cck_pyr[0].indegree
    )
    assert all(
        proj.weight_nS == pytest.approx(baseline_cck_pyr[0].weight_nS)
        for proj in split_cck_pyr
    )
    assert split.receptors.n_ports() == 20
    assert sum(proj.total_conductance_per_cell() for proj in split_cck_pyr) == pytest.approx(
        baseline_cck_pyr[0].total_conductance_per_cell()
    )


def test_compartment_aware_preserves_pyramidal_pv_ultrafast_ampa() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "compartment-aware-pv",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "compartment_aware_synapses": True,
    })

    pyr_to_pv = next(
        proj for proj in spec.projections
        if proj.pre == "Pyramidal" and proj.post == "PV_Basket"
    )
    port = spec.receptors.port_index(pyr_to_pv.receptor)

    assert spec.receptors.n_ports() == 20
    assert pyr_to_pv.receptor.endswith("__dend")
    assert spec.receptors.tau_rise[port] == pytest.approx(0.07)
    assert spec.receptors.tau_decay[port] == pytest.approx(0.2)


def test_fast_basket_bistratified_strategy_preserves_source_kinetics() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "compartment-aware-pv-bistratified",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "preserve_fast_basket_bistratified",
    })

    pv_to_bistratified = next(
        proj for proj in spec.projections
        if proj.pre == "PV_Basket" and proj.post == "Bistratified"
    )
    port = spec.receptors.port_index(pv_to_bistratified.receptor)

    assert spec.receptors.n_ports() <= 20
    assert pv_to_bistratified.receptor.endswith("__soma")
    assert spec.receptors.tau_rise[port] == pytest.approx(0.18)
    assert spec.receptors.tau_decay[port] == pytest.approx(0.45)


def test_demix_pyramidal_olm_strategy_splits_mixed_distal_port() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "compartment-aware-demix-pyramidal-olm",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "demix_pyramidal_olm_gabaa_slow_distal",
    })

    blocker_port = "GABA_A_slow__em60__tr0p11__td9p7__dend"
    distal_port = "GABA_A_slow__em60__tr0p11__td9p7__distal__dend"
    merged_soma_port = "GABA_A_slow__em60__tr1__td8__soma"
    merged_target_port = "GABA_A_slow__em60__tr1__td8__dend"

    assert spec.receptors.n_ports() == 20
    assert distal_port in spec.receptors.names
    assert merged_soma_port not in spec.receptors.names

    olm_to_pyramidal = next(
        proj for proj in spec.projections
        if proj.pre == "O_LM" and proj.post == "Pyramidal"
    )
    bistratified_to_pyramidal = next(
        proj for proj in spec.projections
        if proj.pre == "Bistratified" and proj.post == "Pyramidal"
    )
    cck_to_olm = [
        proj for proj in spec.projections
        if proj.pre == "CCK_Basket" and proj.post == "O_LM"
    ]

    assert olm_to_pyramidal.receptor == distal_port
    assert bistratified_to_pyramidal.receptor == blocker_port
    assert {proj.receptor for proj in cck_to_olm} == {merged_target_port}
    assert sum(proj.indegree for proj in cck_to_olm) == pytest.approx(20.0)


def test_compartment_aware_ports_drive_aglif_dend_compartment_status() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    receptors = (
        "AMPA_fast__e0__tr0p3__td0p6__dend",
        "GABA_A_slow__em60__tr0p432__td4p49__soma",
        "GABA_A_fast__em60__tr0p287__td2p67",
    )

    assert gpu_backend_mod._aglif_dend_compartments(
        receptors,
        "PV_Basket",
        frozenset(),
    ) == [1.0, 0.0, 0.0]


def test_source_location_ports_drive_aglif_dend_domain_status() -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    pv_receptors = (
        "AMPA_fast__e0__tr2__td6p3__dend",
        "AMPA_fast__e0__tr0p07__td0p2__dend",
        "AMPA_slow__e0__tr2__td6p3__dend",
        "GABA_A_fast__em60__tr0p287__td2p67__soma",
    )

    assert gpu_backend_mod._aglif_dend_compartments(
        pv_receptors,
        "PV_Basket",
        frozenset(port for port in pv_receptors if port.endswith("__dend")),
    ) == [1.0, 2.0, 2.0, 0.0]
    bist_receptors = (
        "AMPA_fast__e0__tr2__td6p3__dend",
        "AMPA_fast__e0__tr0p07__td0p2__dend",
    )

    assert gpu_backend_mod._aglif_dend_compartments(
        bist_receptors,
        "Bistratified",
        frozenset(bist_receptors),
    ) == [1.0, 2.0]


def test_aglif_dend_exc_soma_override_is_global(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    monkeypatch.setenv("CA1_AGLIF_DEND_EXC_SOMA", "1")
    receptors = (
        "AMPA_fast__e0__tr0p3__td0p6__dend",
        "AMPA_slow__e0__tr2__td6p3__dend",
        "GABA_A_slow__em60__tr0p432__td4p49__dend",
    )

    assert gpu_backend_mod._aglif_dend_compartments(
        receptors,
        "PV_Basket",
        frozenset(receptors),
    ) == [0.0, 0.0, 1.0]


def test_aglif_dend_exc_soma_override_can_be_cell_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    receptors = (
        "AMPA_fast__e0__tr0p3__td0p6__dend",
        "AMPA_slow__e0__tr2__td6p3__dend",
        "GABA_A_slow__em60__tr0p432__td4p49__dend",
    )
    monkeypatch.setenv("CA1_AGLIF_DEND_EXC_SOMA_CELL_TYPES", "PV_Basket,Bistratified")

    assert gpu_backend_mod._aglif_dend_compartments(
        receptors,
        "Bistratified",
        frozenset(receptors),
    ) == [
        0.0,
        0.0,
        1.0,
    ]
    assert gpu_backend_mod._aglif_dend_compartments(
        receptors,
        "Pyramidal",
        frozenset({"AMPA_fast__e0__tr0p3__td0p6__dend"}),
    ) == [
        2.0,
        1.0,
        1.0,
    ]


def test_aglif_dend_gc_cell_specific_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer_mod = pytest.importorskip("ca1.params.dendritic_transfer")
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    monkeypatch.setenv("CA1_AGLIF_DEND_GC_SCALE", "2.0")
    monkeypatch.setenv("CA1_AGLIF_DEND_GC_SCALE_BISTRATIFIED", "5.0")

    pyramidal = gpu_backend_mod._aglif_dend_status("Pyramidal")
    bistratified = gpu_backend_mod._aglif_dend_status("Bistratified")
    pyramidal_params = gpu_backend_mod.aglif_params_for_cell_type("Pyramidal")
    bistratified_params = gpu_backend_mod.aglif_params_for_cell_type("Bistratified")
    pyramidal_transfer = transfer_mod.dendritic_transfer_for_cell_type("Pyramidal")
    bistratified_transfer = transfer_mod.dendritic_transfer_for_cell_type("Bistratified")

    assert pyramidal["g_c"] == pytest.approx(
        2.0
        * (pyramidal_params.C_m / pyramidal_params.tau_m)
        * pyramidal_transfer.g_c_scale
        * 2.0
    )
    assert bistratified["g_c"] == pytest.approx(
        2.0
        * (bistratified_params.C_m / bistratified_params.tau_m)
        * bistratified_transfer.g_c_scale
        * 5.0
    )


def test_per_target_exact_receptor_tables_preserve_original_kinetics_under_port_limit() -> None:
    config_mod = pytest.importorskip("ca1.config")

    # Given: the paper Table 1 connectivity and compartment-aware syndata kinetics.
    spec = config_mod.build_network_spec({
        "name": "per-target-exact-receptors",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "compartment_aware_synapses": True,
        "receptor_table_scope": "per_target",
    })

    # Then: exact kinetics may exceed the NEST-GPU per-group limit globally, while
    # each postsynaptic target still has a local table inside the 20-port limit.
    assert spec.receptor_table_scope == "per_target"
    assert spec.receptors.n_ports() == 39
    assert set(spec.target_receptors) == set(spec.cell_types)
    target_counts = {
        target: receptors.n_ports()
        for target, receptors in spec.target_receptors.items()
    }
    assert target_counts == {
        "Axo": 10,
        "Bistratified": 10,
        "CCK_Basket": 9,
        "Ivy": 8,
        "Neurogliaform": 5,
        "O_LM": 5,
        "PV_Basket": 9,
        "Pyramidal": 13,
        "SCA": 10,
    }
    assert max(target_counts.values()) <= 20

    missing_projection_ports = [
        (proj.pre, proj.post, proj.receptor)
        for proj in spec.projections
        if proj.receptor not in spec.receptors_for_post(proj.post).names
    ]
    missing_afferent_ports = [
        (aff.name, aff.post, aff.receptor)
        for aff in spec.afferents
        if aff.receptor not in spec.receptors_for_post(aff.post).names
    ]
    assert missing_projection_ports == []
    assert missing_afferent_ports == []
