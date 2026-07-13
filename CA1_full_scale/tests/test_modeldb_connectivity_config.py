from __future__ import annotations

import pytest

from ca1.analysis.location_transfer import UnvalidatedLocationTransferError


def _receptor_prefix(receptor: str) -> str:
    return receptor.split("__", maxsplit=1)[0]


def test_extract_connectivity_can_pair_conn211_with_cellnumbers101() -> None:
    modeldb_tables = pytest.importorskip("ca1.extract.modeldb_tables")

    data = modeldb_tables.extract_connectivity(index=211, cellnumbers_index=101)

    ca3_ivy = data["afferents"]["CA3_to_Ivy"]
    ec3_ivy = data["afferents"]["ECIII_to_Ivy"]
    pyr_ivy = data["projections"]["Pyramidal_to_Ivy"]

    assert "CA3_to_Neurogliaform" not in data["afferents"]
    assert ca3_ivy["synapses_per_cell"] == pytest.approx((17086995 * 2) / 8810)
    assert ca3_ivy["weight_nS"] == pytest.approx(0.1)
    assert ec3_ivy["synapses_per_cell"] == pytest.approx((321565 * 2) / 8810)
    assert ec3_ivy["weight_nS"] == pytest.approx(0.15)
    assert pyr_ivy["indegree"] == pytest.approx(76353 / 8810)
    assert pyr_ivy["weight_nS"] == pytest.approx(0.405)
    assert pyr_ivy["synapses_per_connection"] == 3


def test_extract_connectivity_conn430_uses_per_cell_count_mode() -> None:
    modeldb_tables = pytest.importorskip("ca1.extract.modeldb_tables")

    data = modeldb_tables.extract_connectivity(
        index=430,
        cellnumbers_index=101,
        count_mode="per_cell",
    )

    pyr_pv = data["projections"]["Pyramidal_to_PV_Basket"]
    ca3_pv = data["afferents"]["CA3_to_PV_Basket"]
    ec3_pyr = data["afferents"]["ECIII_to_Pyramidal"]

    assert data["conndata_count_mode"] == "per_cell"
    assert pyr_pv["indegree"] == pytest.approx(424)
    assert pyr_pv["weight_nS"] == pytest.approx(0.7)
    assert pyr_pv["synapses_per_connection"] == 3
    assert ca3_pv["synapses_per_cell"] == pytest.approx(6047 * 2)
    assert ca3_pv["weight_nS"] == pytest.approx(0.22)
    assert ec3_pyr["synapses_per_cell"] == pytest.approx(1299 * 2)


def test_extract_connectivity_rejects_conn430_network_total_mode() -> None:
    modeldb_tables = pytest.importorskip("ca1.extract.modeldb_tables")

    with pytest.raises(ValueError, match="conndata_430.*per-cell.*per_cell"):
        modeldb_tables.extract_connectivity(
            index=430,
            cellnumbers_index=101,
            count_mode="network_total",
        )


def test_load_afferents_rejects_conn430_default_count_mode() -> None:
    synapses_mod = pytest.importorskip("ca1.params.synapses")

    with pytest.raises(ValueError, match="conndata_430.*per-cell.*per_cell"):
        synapses_mod.load_afferents(conndata_index=430, cellnumbers_index=101)


def test_extract_connectivity_conn430_matches_paper_table1_totals() -> None:
    modeldb_tables = pytest.importorskip("ca1.extract.modeldb_tables")

    data = modeldb_tables.extract_connectivity(
        index=430,
        cellnumbers_index=101,
        count_mode="per_cell",
    )

    ca3_pyr = data["afferents"]["CA3_to_Pyramidal"]
    ec3_pyr = data["afferents"]["ECIII_to_Pyramidal"]
    ca3_sca = data["afferents"]["CA3_to_SCA"]
    ec3_sca = data["afferents"]["ECIII_to_SCA"]
    ec3_ngf = data["afferents"]["ECIII_to_Neurogliaform"]

    assert ca3_pyr["estimated_total_connections"] == 5985 * 311_500
    assert ca3_pyr["synapses_per_cell"] == pytest.approx(11970)
    assert ec3_pyr["estimated_total_connections"] == 1299 * 311_500
    assert ec3_pyr["synapses_per_cell"] == pytest.approx(2598)
    assert ca3_sca["synapses_per_cell"] == pytest.approx(3880)
    assert ec3_sca["synapses_per_cell"] == pytest.approx(1146)
    assert ec3_ngf["synapses_per_cell"] == pytest.approx(1046)


def test_build_network_spec_rejects_conn430_without_per_cell_mode() -> None:
    config_mod = pytest.importorskip("ca1.config")

    with pytest.raises(ValueError, match="conndata_430.*per_cell"):
        _ = config_mod.build_network_spec({
            "name": "paper_conn430_missing_count_mode",
            "conndata_index": 430,
            "cellnumbers_index": 101,
        })


def test_build_network_spec_uses_configured_conn430_with_cellnumbers101() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "paper_conn430",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
    })

    ca3_pyr = next(a for a in spec.afferents if a.name == "CA3_to_Pyramidal")
    ec3_pyr = next(a for a in spec.afferents if a.name == "ECIII_to_Pyramidal")
    ca3_sca = next(a for a in spec.afferents if a.name == "CA3_to_SCA")
    ec3_ngf = next(a for a in spec.afferents if a.name == "ECIII_to_Neurogliaform")
    pyr_sca = next(
        p for p in spec.projections
        if (
            p.pre == "Pyramidal"
            and p.post == "SCA"
            and _receptor_prefix(p.receptor) == "AMPA_fast"
        )
    )
    pyr_pyr = next(
        p for p in spec.projections
        if (
            p.pre == "Pyramidal"
            and p.post == "Pyramidal"
            and _receptor_prefix(p.receptor) == "AMPA_fast"
        )
    )

    assert all(a.name != "CA3_to_Neurogliaform" for a in spec.afferents)
    assert len(spec.afferents) == 13
    assert len(spec.projections) == 68
    assert ca3_pyr.synapses_per_cell == pytest.approx(11970)
    assert ca3_pyr.weight_nS == pytest.approx(0.2)
    assert ec3_pyr.synapses_per_cell == pytest.approx(2598)
    assert ec3_pyr.weight_nS == pytest.approx(0.2)
    assert ca3_sca.synapses_per_cell == pytest.approx(3880)
    assert ec3_ngf.synapses_per_cell == pytest.approx(1046)
    assert pyr_pyr.indegree == pytest.approx(197)
    assert pyr_pyr.synapses_per_connection == 1
    assert pyr_sca.indegree == pytest.approx(105)
    assert pyr_sca.weight_nS == pytest.approx(0.405)


def test_modeldb_connectivity_uses_default_three_ms_delay() -> None:
    config_mod = pytest.importorskip("ca1.config")

    spec = config_mod.build_network_spec({
        "name": "paper_conn430_delay",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
    })

    assert {projection.delay_ms for projection in spec.projections} == {3.0}
    assert {afferent.delay_ms for afferent in spec.afferents} == {3.0}


def test_full_scale_config_declares_canonical_modeldb_indices() -> None:
    config_mod = pytest.importorskip("ca1.config")

    config = config_mod.load_config("configs/full_scale.yaml")

    assert config["cellnumbers_index"] == 101
    assert config["conndata_index"] == 430
    assert config["conndata_count_mode"] == "per_cell"
    assert config["syndata_variant"] == 120
    assert config["afferent_topology"] == "literal_source_graph"
    assert config["afferent_source_pool_size"] == 250000
    assert config["afferent_source_pool_indegree"] == 64


def test_full_scale_config_declares_canonical_gpu_neuron_surface() -> None:
    config_mod = pytest.importorskip("ca1.config")

    config = dict(config_mod.load_config("configs/full_scale.yaml"))
    config["source_location_transfer_mode"] = "none"
    spec = config_mod.build_network_spec(config)

    assert config["neuron_model"] == "aglif_dend_cond_beta"
    assert config["compartment_aware_synapses"] is True
    assert config["receptor_port_strategy"] == "budget_weighted"
    assert spec.neuron_model == "aglif_dend_cond_beta"
    assert spec.receptor_provenance.startswith(
        "syndata120-compartment-aware-20port-budget_weighted"
    )
    assert ";sha256=" in spec.receptor_provenance
    assert spec.source_location_transfer_provenance == ""


def test_aglif_dend_spec_does_not_embed_unused_adex_fallback_provenance() -> None:
    config_mod = pytest.importorskip("ca1.config")

    config = dict(config_mod.load_config("configs/full_scale.yaml"))
    config["source_location_transfer_mode"] = "none"
    spec = config_mod.build_network_spec(config)

    assert spec.neuron_model == "aglif_dend_cond_beta"
    assert {
        cell_type.params.fit_provenance
        for cell_type in spec.cell_types.values()
    } == {"unused-by-aglif_dend_cond_beta-runtime"}


def test_aglif_dend_parameter_provenance_exposes_active_runtime_keys() -> None:
    config_mod = pytest.importorskip("ca1.config")
    provenance_mod = pytest.importorskip("ca1.params.provenance")

    config = dict(config_mod.load_config("configs/full_scale.yaml"))
    config["source_location_transfer_mode"] = "none"
    spec = config_mod.build_network_spec(config)
    provenance = provenance_mod.parameter_provenance_for_spec(spec)

    assert provenance["network.neuron_model"] == "aglif_dend_cond_beta"
    assert provenance["aglif.Pyramidal"].startswith("nestgpu-fi-fit")
    assert provenance["dendritic_transfer.Pyramidal"].startswith(
        "neuron-epsp-location-compressed-fit"
    )
    assert "neuron.Pyramidal" not in provenance


def test_full_scale_config_uses_final_validated_source_transfer() -> None:
    config_mod = pytest.importorskip("ca1.config")
    provenance_mod = pytest.importorskip("ca1.params.provenance")
    validation_mod = pytest.importorskip("ca1.validation.provenance")

    config = config_mod.load_config("configs/full_scale.yaml")
    spec = config_mod.build_network_spec(config)
    provenance = provenance_mod.parameter_provenance_for_spec(spec)

    assert config["source_location_transfer_mode"] == "all_dend"
    assert provenance["source_location_transfer.table"].startswith(
        "source-location-transfer-m2-row-validation-passed"
    )
    assert "sha256=" in provenance["source_location_transfer.table"]
    blockers = validation_mod.final_tier_parameter_provenance_blockers(
        provenance,
        spec.scaled_counts(),
    )
    assert "source_location_transfer.table=missing" not in blockers
    assert not any(
        blocker.startswith("source_location_transfer.")
        for blocker in blockers
    )


def test_config_rejects_unvalidated_source_location_transfer_table() -> None:
    config_mod = pytest.importorskip("ca1.config")

    config = config_mod.load_config("configs/full_scale.yaml")
    config["source_location_transfer_mode"] = "all_dend"
    config["source_location_transfer_table"] = (
        ".omo/ulw-loop/g002-continuation/evidence/"
        "fullscale_source_kinetic_ports_wave108/"
        "location_transfer_compensated_budget_with_afferents.json"
    )

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="M2 response validation",
    ):
        _ = config_mod.build_network_spec(config)
