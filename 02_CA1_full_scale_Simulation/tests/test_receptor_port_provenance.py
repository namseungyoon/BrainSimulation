from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from ca1.analysis.location_transfer_validation import unvalidated_transfer_rows
from ca1.config import build_network_spec
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.types import ReceptorConfig
from ca1.validation.network_provenance import final_tier_network_structure_blockers


def _full_scale_budget_weighted_spec():
    return build_network_spec({
        "name": "canonical_receptor_port_provenance",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "syndata_variant": 120,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "recurrent_topology": "modeldb_fastconn_3d_gaussian",
        "afferent_topology": "literal_source_graph",
        "afferent_source_pool_size": 250000,
        "afferent_source_pool_indegree": 64,
    })


def _full_scale_per_target_exact_spec():
    return build_network_spec({
        "name": "canonical_per_target_exact_receptor_port_provenance",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_table_scope": "per_target",
        "syndata_variant": 120,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "recurrent_topology": "modeldb_fastconn_3d_gaussian",
        "afferent_topology": "literal_source_graph",
        "afferent_source_pool_size": 250000,
        "afferent_source_pool_indegree": 64,
    })


def _fingerprint(receptors: ReceptorConfig) -> str:
    rows = [
        {
            "name": name,
            "e_rev": receptors.E_rev[index],
            "tau_rise": receptors.tau_rise[index],
            "tau_decay": receptors.tau_decay[index],
        }
        for index, name in enumerate(receptors.names)
    ]
    payload = json.dumps(rows, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_canonical_receptor_port_strategy_passes_full_network_gate() -> None:
    # Given: a full-scale syndata120 compartment-aware budget-weighted spec.
    spec = _full_scale_budget_weighted_spec()
    provenance = parameter_provenance_for_spec(spec)

    # When: final-tier network structure provenance is checked.
    blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    # Then: the canonical 20-port strategy is accepted.
    assert blockers == []


def test_per_target_exact_receptor_strategy_passes_full_network_gate() -> None:
    # Given: a full-scale syndata120 compartment-aware spec with exact target-local
    # receptor tables.
    spec = _full_scale_per_target_exact_spec()
    provenance = parameter_provenance_for_spec(spec)

    # When: final-tier network structure provenance is checked.
    blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    # Then: the exact 39-port global table is accepted because each target-local
    # table stays within the NEST-GPU 20-port limit.
    assert blockers == []


def test_receptor_port_provenance_changes_when_resolved_table_changes() -> None:
    # Given: a canonical spec and the same strategy name with mutated resolved kinetics.
    spec = _full_scale_budget_weighted_spec()
    mutated_receptors = replace(
        spec.receptors,
        tau_decay=(spec.receptors.tau_decay[0] + 0.001, *spec.receptors.tau_decay[1:]),
    )
    mutated_spec = replace(spec, receptors=mutated_receptors)

    # When: parameter provenance is emitted.
    canonical = parameter_provenance_for_spec(spec)
    mutated = parameter_provenance_for_spec(mutated_spec)

    # Then: the resolved table content is represented in receptor provenance.
    assert _fingerprint(spec.receptors) != _fingerprint(mutated_receptors)
    assert canonical["synapse.receptor_ports"] != mutated["synapse.receptor_ports"]
    assert f"sha256={_fingerprint(spec.receptors)}" in canonical[
        "synapse.receptor_ports"
    ]


def test_final_network_gate_rejects_missing_receptor_port_hash() -> None:
    # Given: old-style final provenance that only names the strategy.
    spec = _full_scale_budget_weighted_spec()
    provenance = parameter_provenance_for_spec(spec)
    provenance["synapse.receptor_ports"] = (
        "syndata120-compartment-aware-20port-budget_weighted"
    )

    # When: final-tier network structure provenance is checked.
    blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    # Then: missing content address is a loud final-tier blocker.
    assert any("missing receptor port sha256" in blocker for blocker in blockers)


def test_final_network_gate_reports_receptor_port_hash_mismatch() -> None:
    # Given: the canonical strategy name with a mutated resolved-table hash.
    spec = _full_scale_budget_weighted_spec()
    provenance = parameter_provenance_for_spec(spec)
    provenance["synapse.receptor_ports"] = (
        "syndata120-compartment-aware-20port-budget_weighted;sha256="
        f"{'0' * 64}"
    )

    # When: final-tier network structure provenance is checked.
    blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    # Then: final validation reports a content-address mismatch.
    assert any("receptor port sha256 mismatch" in blocker for blocker in blockers)


@pytest.mark.parametrize(
    "field",
    [
        "measured_reduced_ratio",
        "compensated_ratio",
        "abs_error",
        "tolerance",
    ],
)
def test_location_transfer_rejects_boolean_m2_numeric_fields(field: str) -> None:
    # Given: an otherwise valid M2 validation record with a boolean numeric field.
    validation: dict[str, object] = {
        "method": "user_m2-row-level-response-fidelity",
        "evidence_path": "evidence/m2_row_validation.json",
        "passed": True,
        "sign_preserved": True,
        "low_signal": False,
        "measured_reduced_ratio": 0.04,
        "compensated_ratio": 0.1,
        "abs_error": 0.0,
        "tolerance": 0.01,
    }
    row: dict[str, object] = {
        "port": "AMPA_fast__e0__tr0p07__td0p2__dend",
        "aglif_compartment": "dend",
        "morph_ratio_est": 0.1,
        "transfer_scale": 2.5,
        "m2_validation": validation,
    }
    validation[field] = True

    # When: the source-location table is scanned for unsafe rows.
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {
        ("Pyramidal", "PV_Basket", "AMPA_fast", str(row["port"])): row
    }
    blockers = unvalidated_transfer_rows(rows)

    # Then: bool is rejected instead of accepted through int subclassing.
    assert any(field in blocker for blocker in blockers)
