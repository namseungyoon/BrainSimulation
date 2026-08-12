from __future__ import annotations

import numpy as np

from ca1.config import build_network_spec
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.types import SimMeta, SimResult
from ca1.validation.harness import validate

_CANONICAL_RECEPTOR_PORTS = (
    "syndata120-compartment-aware-20port-budget_weighted;"
    "sha256=26774704b306d1bd0461fd7df69491cfacd0e1a2e6385877ece2150c9e05e46c"
)


def _quiet_result(parameter_provenance: dict[str, str]) -> SimResult:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type={"Pyramidal": 338_740},
        scale=1.0,
        seed=1,
        backend="test",
        config_name="structural_provenance",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_spike_density",
        parameter_provenance=parameter_provenance,
        diagnostic_provenance={"diagnostic.audit": "no-overrides"},
    )
    return SimResult(
        spikes={"Pyramidal": [np.array([], dtype=float)]},
        meta=meta,
    )


def _full_counts() -> dict[str, int]:
    return {
        "Pyramidal": 311_500,
        "PV_Basket": 5_530,
        "CCK_Basket": 3_600,
        "Axo": 1_470,
        "Bistratified": 2_210,
        "Ivy": 8_810,
        "O_LM": 1_640,
        "SCA": 400,
        "Neurogliaform": 3_580,
    }


def _full_result(parameter_provenance: dict[str, str]) -> SimResult:
    meta = SimMeta(
        duration_s=0.05,
        dt_s=0.001,
        n_cells_per_type=_full_counts(),
        scale=1.0,
        seed=1,
        backend="test",
        config_name="structural_provenance",
        crop_first_ms=0.0,
        lfp_proxy="pyramidal_spike_density",
        parameter_provenance=parameter_provenance,
        diagnostic_provenance={"diagnostic.audit": "no-overrides"},
    )
    return SimResult(
        spikes={name: [np.array([], dtype=float)] for name in _full_counts()},
        meta=meta,
    )


def _paper_network_provenance(
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    provenance = {
        "aglif.Pyramidal": "nestgpu-fi-fit",
        "network.total_cells": "338740",
        "network.cell_types": "9",
        "network.recurrent_projections": "68",
        "network.afferents": "13",
        "network.afferent_sources": "CA3,ECIII",
        "network.afferent_rate_hz": "0.65",
        "network.afferent_source_rate_rule": "homogeneous",
        "network.afferent_source_count_total": "454700",
        "network.afferent_source_count_max": "250000",
        "network.recurrent_topology": "modeldb_fastconn_3d_gaussian",
        "network.afferent_topology": "source_pool",
        "network.afferent_source_pool_size": "250000",
        "network.afferent_source_pool_indegree": "64",
        "network.afferent_source_pool_weight_rule": (
            "source_pool_path_rate_preserving"
        ),
        "network.afferent_poisson_rule": "source_pool_path_rate_preserving",
        "network.afferent_source_driver": "rate_preserving_poisson_generator",
        "network.conndata_index": "430",
        "network.cellnumbers_index": "101",
        "network.conndata_count_mode": "per_cell",
        "network.recurrent_synapses": "441375540",
        "network.afferent_synapses": "4704026540",
        "network.total_synapses": "5145402080",
        "network.multisynapse_rule": "same_source_same_delay_weight_aggregation",
        "network.neuron_model": "aglif_cond_beta",
        "synapse.short_term_plasticity": "static_exp2syn_no_stp",
        "synapse.receptor_ports": _CANONICAL_RECEPTOR_PORTS,
    }
    if overrides:
        provenance.update(overrides)
    return provenance


def test_validate_full_tier_fails_when_network_structure_provenance_is_missing() -> None:
    result = _quiet_result({"aglif.Pyramidal": "nestgpu-fi-fit"})

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.total_cells" in checks[0].detail


def test_validate_full_tier_fails_when_recurrent_or_afferents_are_missing() -> None:
    result = _quiet_result(
        _paper_network_provenance({
            "network.recurrent_projections": "0",
            "network.afferents": "0",
            "network.afferent_sources": "missing",
            "network.afferent_source_count_total": "0",
            "network.afferent_source_count_max": "0",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "recurrent_projections" in checks[0].detail
    assert "afferents" in checks[0].detail
    assert "afferent_sources" in checks[0].detail


def test_validate_full_tier_rejects_compound_poisson_superposition() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "compound",
            "network.afferent_poisson_rule": (
                "postcell_independent_poisson_superposition"
            ),
            "network.afferent_source_driver": "compound_poisson_generator",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.afferent_topology=compound is a diagnostic" in checks[0].detail


def test_validate_full_tier_rejects_compound_without_poisson_rule() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "compound",
            "network.afferent_poisson_rule": "",
            "network.afferent_source_driver": "compound_poisson_generator",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.afferent_poisson_rule" in checks[0].detail


def test_validate_full_tier_rejects_non_paper_conndata() -> None:
    result = _quiet_result(
        _paper_network_provenance({
            "network.conndata_index": "211",
            "network.conndata_count_mode": "network_total",
            "network.recurrent_synapses": "706411822",
            "network.afferent_synapses": "9662533570",
            "network.total_synapses": "10368945392",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.conndata_index must be 430" in checks[0].detail
    assert "network.conndata_count_mode must be per_cell" in checks[0].detail


def test_validate_full_tier_rejects_non_paper_cellnumbers() -> None:
    result = _quiet_result(
        _paper_network_provenance({"network.cellnumbers_index": "211"})
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.cellnumbers_index must be 101" in checks[0].detail


def test_validate_full_tier_rejects_wrong_modeldb_synapse_budget() -> None:
    result = _quiet_result(
        _paper_network_provenance({
            "network.recurrent_synapses": "1",
            "network.afferent_synapses": "2",
            "network.total_synapses": "3",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.recurrent_synapses must be 441375540" in checks[0].detail
    assert "network.afferent_synapses must be 4704026540" in checks[0].detail
    assert "network.total_synapses must be 5145402080" in checks[0].detail


def test_validate_full_tier_rejects_hidden_multisynapse_rule() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_pool_weight_rule": (
                "unused_for_literal_source_graph"
            ),
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
            "network.multisynapse_rule": "physical_multapses",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.multisynapse_rule" in checks[0].detail
    assert "same_source_same_delay_weight_aggregation" in checks[0].detail


def test_validate_full_tier_rejects_hidden_stp_reduction() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_pool_weight_rule": (
                "unused_for_literal_source_graph"
            ),
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
            "synapse.short_term_plasticity": "modeldb_exp_gabaab_stp",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "synapse.short_term_plasticity" in checks[0].detail
    assert "static_exp2syn_no_stp" in checks[0].detail


def test_validate_full_tier_rejects_partial_path_counts() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.recurrent_projections": "12",
            "network.afferents": "2",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.recurrent_projections must be 68" in checks[0].detail
    assert "network.afferents must be 13" in checks[0].detail


def test_validate_full_tier_rejects_source_pool_without_audited_parameters() -> None:
    provenance = _paper_network_provenance()
    del provenance["network.afferent_source_pool_size"]
    del provenance["network.afferent_source_pool_indegree"]
    del provenance["network.afferent_source_pool_weight_rule"]
    result = _quiet_result(provenance)

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.afferent_source_pool_size" in checks[0].detail
    assert "network.afferent_source_pool_indegree" in checks[0].detail
    assert "network.afferent_source_pool_weight_rule" in checks[0].detail


def test_validate_full_tier_rejects_path_preserving_source_pool() -> None:
    result = _full_result(_paper_network_provenance())

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "source_pool_path_rate_preserving is diagnostic" in checks[0].detail


def test_validate_full_tier_accepts_literal_source_graph_structure() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_pool_weight_rule": (
                "unused_for_literal_source_graph"
            ),
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert checks[0].passed


def test_validate_full_tier_rejects_noncanonical_receptor_ports() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_pool_weight_rule": (
                "unused_for_literal_source_graph"
            ),
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
            "synapse.receptor_ports": (
                "syndata120-compartment-aware-20port-"
                "preserve_fast_basket_bistratified"
            ),
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "synapse.receptor_ports" in checks[0].detail
    assert "syndata120-compartment-aware-20port-budget_weighted" in checks[0].detail


def test_validate_full_tier_rejects_literal_source_graph_without_driver() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_pool_weight_rule": (
                "unused_for_literal_source_graph"
            ),
            "network.afferent_source_driver": "",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.afferent_source_driver" in checks[0].detail
    assert "precomputed_poisson_spike_generator" in checks[0].detail


def test_validate_full_tier_rejects_literal_source_graph_with_wrong_rule() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "source_pool_path_rate_preserving",
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "literal_shared_source_graph" in checks[0].detail


def test_validate_full_tier_rejects_literal_source_graph_with_wrong_driver() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_driver": "runtime_poisson_generator",
            "network.afferent_source_pool_weight_rule": (
                "unused_for_literal_source_graph"
            ),
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "precomputed_poisson_spike_generator" in checks[0].detail


def test_validate_full_tier_rejects_compressed_source_count_fallback() -> None:
    result = _quiet_result(
        _paper_network_provenance({"network.afferent_source_pool_size": "4096"})
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "compressed diagnostic source-count fallback" in checks[0].detail
    assert "source_count_max=250000" in checks[0].detail


def test_validate_full_tier_rejects_fixed_indegree_recurrent_topology() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
            "network.afferent_source_pool_weight_rule": "unused_for_literal_source_graph",
            "network.recurrent_topology": "fixed_indegree",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.recurrent_topology=fixed_indegree" in checks[0].detail


def test_validate_full_tier_rejects_gaussian_fastconn_recurrent_topology() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
            "network.afferent_source_pool_weight_rule": "unused_for_literal_source_graph",
            "network.recurrent_topology": "modeldb_fastconn_gaussian_binned",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert (
        "network.recurrent_topology=modeldb_fastconn_gaussian_binned"
        in checks[0].detail
    )


def test_validate_full_tier_rejects_x_binned_fastconn_recurrent_topology() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
            "network.afferent_source_pool_weight_rule": "unused_for_literal_source_graph",
            "network.recurrent_topology": "modeldb_fastconn_binned",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert not checks[0].passed
    assert "network.recurrent_topology=modeldb_fastconn_binned" in checks[0].detail


def test_validate_full_tier_accepts_modeldb_fastconn_3d_recurrent_topology() -> None:
    result = _full_result(
        _paper_network_provenance({
            "network.afferent_topology": "literal_source_graph",
            "network.afferent_poisson_rule": "literal_shared_source_graph",
            "network.afferent_source_driver": "precomputed_poisson_spike_generator",
            "network.afferent_source_pool_weight_rule": "unused_for_literal_source_graph",
        })
    )

    report = validate(result, tier="full")

    checks = [
        check for check in report.checks
        if check.name == "provenance/network_structure"
    ]
    assert len(checks) == 1
    assert checks[0].required
    assert checks[0].passed


def test_parameter_provenance_records_network_structure() -> None:
    spec = build_network_spec({
        "name": "full_scale_structure_without_source_transfer",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
    })

    provenance = parameter_provenance_for_spec(spec)

    assert provenance["network.total_cells"] == "338740"
    assert provenance["network.cell_types"] == "9"
    assert provenance["network.recurrent_projections"] == "68"
    assert provenance["network.afferents"] == "13"
    assert provenance["network.afferent_sources"] == "CA3,ECIII"
    assert provenance["network.afferent_rate_hz"] == "0.65"
    assert provenance["network.afferent_source_rate_rule"] == "homogeneous"
    assert provenance["network.afferent_source_count_total"] == "454700"
    assert provenance["network.afferent_source_count_max"] == "250000"
    assert provenance["network.recurrent_topology"] == "fixed_indegree"
    assert provenance["network.afferent_topology"] == "compound"
    assert (
        provenance["network.afferent_poisson_rule"]
        == "postcell_independent_poisson_superposition"
    )
    assert (
        provenance["network.afferent_source_driver"]
        == "compound_poisson_generator"
    )
    assert provenance["network.afferent_source_pool_size"] == "4096"
    assert provenance["network.afferent_source_pool_indegree"] == "64"
    assert (
        provenance["network.afferent_source_pool_weight_rule"]
        == "unused_for_compound"
    )
    assert provenance["network.conndata_index"] == "430"
    assert provenance["network.cellnumbers_index"] == "101"
    assert provenance["network.conndata_count_mode"] == "per_cell"
    assert provenance["network.recurrent_synapses"] == "441375540"
    assert provenance["network.afferent_synapses"] == "4704026540"
    assert provenance["network.total_synapses"] == "5145402080"
    assert (
        provenance["network.multisynapse_rule"]
        == "same_source_same_delay_weight_aggregation"
    )
    assert provenance["synapse.short_term_plasticity"] == "static_exp2syn_no_stp"


def test_parameter_provenance_records_literal_source_graph_rule() -> None:
    spec = build_network_spec({
        "name": "literal_source_graph_provenance",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "afferent_topology": "literal_source_graph",
    })

    provenance = parameter_provenance_for_spec(spec)

    assert provenance["network.afferent_topology"] == "literal_source_graph"
    assert provenance["network.afferent_poisson_rule"] == "literal_shared_source_graph"
    assert (
        provenance["network.afferent_source_driver"]
        == "precomputed_poisson_spike_generator"
    )
    assert (
        provenance["network.afferent_source_pool_weight_rule"]
        == "unused_for_literal_source_graph"
    )
