from __future__ import annotations

import json
from pathlib import Path

import pytest

from ca1.analysis.location_transfer import (
    IncompatibleLocationTransferBudgetError,
    IncompleteLocationTransferError,
    UnvalidatedLocationTransferError,
    apply_location_transfer,
)
from ca1.config import build_network_spec
from ca1.types import NetworkSpec, Projection


def _write_not_final_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "location_transfer_not_final.json"
    rows = [
        {
            "kind": "rec",
            "pre": "Pyramidal",
            "post": "PV_Basket",
            "receptor": "AMPA_fast",
            "port": "AMPA_fast__e0__tr0p07__td0p2__dend",
            "loc": "prox",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.1,
            "reduced_ratio_est": 0.04,
            "transfer_scale": 2.5,
            "provenance": (
                "diagnostic-wave116-neuron-receptor-specific-peak-ratio;not-final"
            ),
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    return transfer_table


def _write_empty_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "empty_location_transfer.json"
    _ = transfer_table.write_text(json.dumps([]), encoding="utf-8")
    return transfer_table


def _write_missing_scale_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "location_transfer_missing_scale.json"
    rows = [
        {
            "kind": "rec",
            "pre": "Pyramidal",
            "post": "PV_Basket",
            "receptor": "AMPA_fast",
            "port": "AMPA_fast__e0__tr0p07__td0p2__dend",
            "loc": "prox",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.1,
            "reduced_ratio_est": 0.04,
            "m2_validation": {
                "method": "user_m2-row-level-response-fidelity",
                "evidence_path": "evidence/m2_row_validation.json",
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.04,
                "compensated_ratio": 0.1,
                "abs_error": 0.0,
                "tolerance": 0.01,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    return transfer_table


def _write_incompatible_budget_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "location_transfer_incompatible_budget.json"
    rows = [
        {
            "kind": "rec",
            "pre": "Pyramidal",
            "post": "PV_Basket",
            "receptor": "AMPA_fast",
            "port": "AMPA_fast__e0__tr0p07__td0p2__dend",
            "loc": "prox",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.1,
            "reduced_ratio_est": 0.04,
            "transfer_scale": 2.5,
            "conductance_per_cell_nS": 20.0,
            "m2_validation": {
                "method": "user_m2-row-level-response-fidelity",
                "evidence_path": "evidence/m2_row_validation.json",
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.04,
                "compensated_ratio": 0.1,
                "abs_error": 0.0,
                "tolerance": 0.01,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    return transfer_table


def _write_disguised_active_probe_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "location_transfer_disguised_active_probe.json"
    rows = [
        {
            "kind": "rec",
            "pre": "CCK_Basket",
            "post": "PV_Basket",
            "receptor": "GABA_A_slow",
            "port": "GABA_A_slow__em60__tr0p287__td2p67__dend",
            "loc": "distal",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.1,
            "reduced_ratio_est": 0.04,
            "transfer_scale": 2.5,
            "provenance": (
                "syndata120-budget_weighted-section-distance + "
                "active-inhibitory-m2-probe"
            ),
            "m2_validation": {
                "method": "user_m2-active-inhibitory-row-response-fidelity",
                "evidence_path": (
                    ".omo/ulw-loop/g002-continuation/evidence/"
                    "active_inhibitory_m2_probe_wave226/summary.jsonl"
                ),
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.04,
                "compensated_ratio": 0.1,
                "abs_error": 0.0,
                "tolerance": 0.01,
                "probe_e_rev_mV": -60.0,
                "probe_baseline_mV": -55.0,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    return transfer_table


def _write_self_certified_inhibitory_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "location_transfer_self_certified_inhibitory.json"
    rows = [
        {
            "kind": "rec",
            "pre": "CCK_Basket",
            "post": "PV_Basket",
            "receptor": "GABA_A_slow",
            "port": "GABA_A_slow__em60__tr0p287__td2p67__dend",
            "loc": "distal",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.1,
            "reduced_ratio_est": 0.04,
            "transfer_scale": 2.5,
            "provenance": (
                "syndata120-budget_weighted-section-distance / "
                "user_m2-inhibitory-row-response-validated"
            ),
            "m2_validation": {
                "method": "user_m2-inhibitory-row-response-fidelity",
                "evidence_path": (
                    ".omo/ulw-loop/g002-continuation/evidence/"
                    "final_inhibitory_m2_validation_wave227/summary.jsonl"
                ),
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.04,
                "compensated_ratio": 0.1,
                "abs_error": 0.0,
                "tolerance": 0.01,
                "probe_e_rev_mV": -60.0,
                "probe_baseline_mV": -55.0,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    return transfer_table


def _write_underscore_diagnostic_inhibitory_transfer_table(
    tmp_path: Path,
    *,
    hidden_field: str,
) -> Path:
    transfer_table = tmp_path / f"location_transfer_underscore_{hidden_field}.json"
    provenance = (
        "syndata120-budget_weighted-section-distance / "
        "user_m2-inhibitory-row-response-final-fidelity"
    )
    method = "user_m2-inhibitory-row-response-fidelity"
    evidence_path = (
        ".omo/ulw-loop/g002-continuation/evidence/"
        "final_inhibitory_m2_validation_wave227/summary.jsonl"
    )
    if hidden_field == "provenance":
        provenance = "diagnostic_active_probe_transfer_table"
    elif hidden_field == "method":
        method = "diagnostic_active_probe_transfer_method"
    elif hidden_field == "evidence_path":
        evidence_path = "diagnostic_active_probe_transfer_summary.jsonl"
    else:
        raise ValueError(f"unknown hidden_field {hidden_field!r}")
    rows = [
        {
            "kind": "rec",
            "pre": "CCK_Basket",
            "post": "PV_Basket",
            "receptor": "GABA_A_slow",
            "port": "GABA_A_slow__em60__tr0p287__td2p67__dend",
            "loc": "distal",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.1,
            "reduced_ratio_est": 0.04,
            "transfer_scale": 2.5,
            "provenance": provenance,
            "m2_validation": {
                "method": method,
                "evidence_path": evidence_path,
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.04,
                "compensated_ratio": 0.1,
                "abs_error": 0.0,
                "tolerance": 0.01,
                "probe_e_rev_mV": -60.0,
                "probe_baseline_mV": -55.0,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    return transfer_table


def _transfer_config(transfer_table: Path) -> dict[str, object]:
    return {
        "name": "source_location_transfer_config_gate",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "syndata_variant": 120,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": str(transfer_table),
    }


def _spec_with_dend_projection() -> NetworkSpec:
    return NetworkSpec(
        name="not-final-transfer",
        cell_types={},
        projections=[
            Projection(
                pre="Pyramidal",
                post="PV_Basket",
                indegree=1.0,
                synapses_per_connection=1,
                weight_nS=0.2,
                receptor="AMPA_fast__e0__tr0p07__td0p2__dend",
            )
        ],
        afferents=[],
        neuron_model="aglif_dend_cond_beta",
    )


def _spec_with_gaba_dend_projection() -> NetworkSpec:
    return NetworkSpec(
        name="gaba-transfer",
        cell_types={},
        projections=[
            Projection(
                pre="CCK_Basket",
                post="PV_Basket",
                indegree=1.0,
                synapses_per_connection=1,
                weight_nS=0.2,
                receptor="GABA_A_slow__em60__tr0p287__td2p67__dend",
            )
        ],
        afferents=[],
        neuron_model="aglif_dend_cond_beta",
    )


def test_location_transfer_refuses_not_final_table_without_prototype_override(
    tmp_path: Path,
) -> None:
    transfer_table = _write_not_final_transfer_table(tmp_path)
    spec = _spec_with_dend_projection()

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="not final-validated",
    ):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_location_transfer_refuses_disguised_active_probe_table(
    tmp_path: Path,
) -> None:
    transfer_table = _write_disguised_active_probe_transfer_table(tmp_path)
    spec = _spec_with_gaba_dend_projection()

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="active-inhibitory-m2-probe",
    ):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_location_transfer_refuses_self_certified_inhibitory_validation(
    tmp_path: Path,
) -> None:
    transfer_table = _write_self_certified_inhibitory_transfer_table(tmp_path)
    spec = _spec_with_gaba_dend_projection()

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="user_m2-inhibitory-row-response-validated",
    ):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


@pytest.mark.parametrize(
    ("hidden_field", "error_match"),
    [
        ("provenance", "diagnostic_"),
        ("method", "unvalidated M2 evidence fields"),
        ("evidence_path", "unvalidated M2 evidence fields"),
    ],
)
def test_location_transfer_refuses_underscore_diagnostic_tokens(
    tmp_path: Path,
    hidden_field: str,
    error_match: str,
) -> None:
    transfer_table = _write_underscore_diagnostic_inhibitory_transfer_table(
        tmp_path,
        hidden_field=hidden_field,
    )
    spec = _spec_with_gaba_dend_projection()

    with pytest.raises(UnvalidatedLocationTransferError, match=error_match):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_location_transfer_refuses_matched_dend_row_missing_transfer_scale(
    tmp_path: Path,
) -> None:
    transfer_table = _write_missing_scale_transfer_table(tmp_path)
    spec = _spec_with_dend_projection()

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="missing transfer_scale",
    ):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_location_transfer_refuses_incompatible_source_budget_metadata(
    tmp_path: Path,
) -> None:
    transfer_table = _write_incompatible_budget_transfer_table(tmp_path)
    spec = _spec_with_dend_projection()

    with pytest.raises(IncompatibleLocationTransferBudgetError, match="source-budget"):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_location_transfer_allows_not_final_table_only_as_explicit_prototype(
    tmp_path: Path,
) -> None:
    transfer_table = _write_not_final_transfer_table(tmp_path)
    spec = _spec_with_dend_projection()

    updated, applied, missing = apply_location_transfer(
        spec,
        "all_dend",
        transfer_table,
        allow_incomplete_transfer_for_prototype=True,
    )

    assert missing == []
    assert applied == [
        {
            "pre": "Pyramidal",
            "post": "PV_Basket",
            "receptor": "AMPA_fast__e0__tr0p07__td0p2__dend",
            "loc": "prox",
            "scale": 2.5,
        }
    ]
    assert updated.projections[0].weight_nS == 0.5
    assert updated.source_location_transfer_provenance.startswith(
        "unvalidated-prototype-source-location-transfer"
    )
    assert "incomplete-prototype-override" in updated.source_location_transfer_provenance


def test_location_transfer_prototype_override_is_parameter_provenance_visible(
    tmp_path: Path,
) -> None:
    transfer_table = _write_not_final_transfer_table(tmp_path)
    spec = _spec_with_dend_projection()

    updated, _, _ = apply_location_transfer(
        spec,
        "all_dend",
        transfer_table,
        allow_incomplete_transfer_for_prototype=True,
    )

    assert "prototype" in updated.source_location_transfer_provenance
    assert "not-final" not in updated.source_location_transfer_provenance


def test_build_network_spec_refuses_missing_source_location_transfer_table(
    tmp_path: Path,
) -> None:
    transfer_table = tmp_path / "missing_location_transfer.json"

    with pytest.raises(FileNotFoundError, match="missing_location_transfer"):
        _ = build_network_spec(_transfer_config(transfer_table))


def test_build_network_spec_refuses_incomplete_source_location_transfer_table(
    tmp_path: Path,
) -> None:
    transfer_table = _write_empty_transfer_table(tmp_path)

    with pytest.raises(
        IncompleteLocationTransferError,
        match="refusing implicit 1.0 fallback",
    ):
        _ = build_network_spec(_transfer_config(transfer_table))


def test_build_network_spec_refuses_not_final_source_location_transfer_table(
    tmp_path: Path,
) -> None:
    transfer_table = _write_not_final_transfer_table(tmp_path)

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="not final-validated",
    ):
        _ = build_network_spec(_transfer_config(transfer_table))
