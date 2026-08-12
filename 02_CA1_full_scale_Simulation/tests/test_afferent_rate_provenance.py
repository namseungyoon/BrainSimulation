from __future__ import annotations

import pytest

from ca1.config import build_network_spec
from ca1.params.provenance import (
    diagnostic_config_provenance,
    parameter_provenance_for_spec,
)
from ca1.validation.network_provenance import (
    final_tier_network_structure_blockers,
)
from ca1.validation.provenance import final_tier_diagnostic_provenance_blockers


def test_parameter_provenance_records_homogeneous_afferent_control_rate() -> None:
    spec = build_network_spec({
        "name": "literal_source_control_rate",
        "neuron_model": "aglif_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "afferent_topology": "literal_source_graph",
        "recurrent_topology": "modeldb_fastconn_binned",
        "afferent_rate_hz": 0.65,
    })

    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert provenance["network.afferent_rate_hz"] == "0.65"
    assert provenance["network.afferent_source_rate_rule"] == "homogeneous"
    assert not any("network.afferent_rate_hz" in blocker for blocker in blockers)
    assert not any(
        "network.afferent_source_rate_rule" in blocker
        for blocker in blockers
    )


def test_full_tier_rejects_non_control_afferent_rate() -> None:
    spec = build_network_spec({
        "name": "literal_source_wrong_rate",
        "neuron_model": "aglif_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "afferent_topology": "literal_source_graph",
        "recurrent_topology": "modeldb_fastconn_binned",
        "afferent_rate_hz": 0.8,
    })

    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert provenance["network.afferent_rate_hz"] == "0.8"
    assert any(
        "network.afferent_rate_hz must be 0.65" in blocker
        for blocker in blockers
    )


def test_full_tier_rejects_heterogeneous_source_rate_rule() -> None:
    spec = build_network_spec({
        "name": "literal_source_rate_heterogeneous",
        "neuron_model": "aglif_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "conndata_index": 430,
        "cellnumbers_index": 101,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "afferent_topology": "literal_source_graph",
        "recurrent_topology": "modeldb_fastconn_binned",
        "afferent_rate_hz": 0.65,
        "afferent_source_rate_cv": 0.25,
    })

    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert provenance["network.afferent_source_rate_rule"] == (
        "mean_preserving_lognormal_cv=0.25"
    )
    assert any(
        "network.afferent_source_rate_rule must be 'homogeneous'" in blocker
        for blocker in blockers
    )


def test_diagnostic_provenance_records_source_rate_cv_config() -> None:
    provenance = diagnostic_config_provenance({"afferent_source_rate_cv": 0.25})

    assert provenance == {"config.afferent_source_rate_cv": "0.25"}
    assert final_tier_diagnostic_provenance_blockers(provenance) == [
        "config.afferent_source_rate_cv=0.25",
    ]


def test_build_network_spec_rejects_negative_source_rate_cv() -> None:
    with pytest.raises(ValueError, match="afferent_source_rate_cv"):
        _ = build_network_spec({
            "name": "negative_source_rate_cv",
            "afferent_source_rate_cv": -0.1,
        })
