from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ca1.analysis.location_transfer import (
    IncompleteLocationTransferError,
    apply_location_transfer,
)
from ca1.config import build_network_spec
from ca1.params.aglif import load_aglif_params
from ca1.params.dendritic_transfer import load_dendritic_transfer_params
from ca1.params.groundtruth import CELL_TEMPLATES
from ca1.params.neurons import _load_fitted, load_neuron_params
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.sim.aglif_dend import aglif_dend_compartments
from ca1.types import NetworkSpec, Projection
from ca1.validation.provenance import final_tier_parameter_provenance_blockers


def test_connectivity_rejects_declared_receptor_that_would_be_ignored(
    tmp_path: Path,
) -> None:
    from ca1.params.synapses import load_projections

    connectivity = {
        "excitatory_connections": {
            "pyr_to_sca": {
                "presynaptic": "Pyramidal",
                "postsynaptic": "SCA",
                "indegree": 1,
                "synapses_per_connection": 1,
                "weight_nS": 0.1,
                "receptor": "NMDA",
            }
        },
        "inhibitory_connections": {},
    }
    path = tmp_path / "connectivity_nmda.json"
    path.write_text(json.dumps(connectivity), encoding="utf-8")

    with pytest.raises(ValueError, match="declared receptor"):
        _ = load_projections(path=path)


@pytest.mark.parametrize(
    "calibration",
    [
        {"recurrent_weight_scale": -1.0},
        {"afferent_weight_scale": -1.0},
        {"recurrent_receptor_weight_scales": {"AMPA_fast": -1.0}},
        {"projection_weight_scales": {"Pyramidal->SCA": -1.0}, "mode": "diagnostic"},
        {"afferent_weight_scales": {"CA3_to_Pyramidal": -1.0}, "mode": "diagnostic"},
        {"afferent_post_weight_scales": {"SCA": -1.0}, "mode": "diagnostic"},
    ],
)
def test_calibration_rejects_negative_weight_scales(
    calibration: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _ = build_network_spec({"name": "negative_scale", "calibration": calibration})


def test_calibration_rejects_unknown_projection_target() -> None:
    with pytest.raises(ValueError, match="unknown calibration.projection_weight_scales"):
        _ = build_network_spec(
            {
                "name": "unknown_projection_scale",
                "calibration": {
                    "mode": "diagnostic",
                    "projection_weight_scales": {"CA3_to_Pyramidal": 1.1},
                },
            }
        )


def test_calibration_rejects_unknown_afferent_target() -> None:
    with pytest.raises(ValueError, match="unknown calibration.afferent_weight_scales"):
        _ = build_network_spec(
            {
                "name": "unknown_afferent_scale",
                "calibration": {
                    "mode": "diagnostic",
                    "afferent_weight_scales": {"Pyramidal->SCA": 1.1},
                },
            }
        )


def test_calibration_rejects_boolean_numeric_scale() -> None:
    with pytest.raises(ValueError, match="calibration.recurrent_weight_scale"):
        _ = build_network_spec(
            {
                "name": "boolean_scale",
                "calibration": {"recurrent_weight_scale": True},
            }
        )


def test_config_rejects_quoted_false_boolean() -> None:
    with pytest.raises(TypeError, match="compartment_aware_synapses"):
        _ = build_network_spec(
            {
                "name": "quoted_false_compartment_aware",
                "compartment_aware_synapses": "false",
            }
        )


@pytest.mark.parametrize("field", ["transfer_scale", "conductance_per_cell_nS"])
def test_location_transfer_rejects_boolean_numeric_fields(
    field: str,
    tmp_path: Path,
) -> None:
    canonical = (
        Path(__file__).parents[1]
        / "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
    )
    rows = json.loads(canonical.read_text(encoding="utf-8"))
    rows[0][field] = True
    transfer_table = tmp_path / "boolean_transfer_table.json"
    transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    spec = build_network_spec(
        {
            "name": "boolean_transfer_source",
            "conndata_index": 430,
            "conndata_count_mode": "per_cell",
            "compartment_aware_synapses": True,
            "receptor_port_strategy": "budget_weighted",
        }
    )

    with pytest.raises(TypeError, match=field):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_final_tier_rejects_bare_analytic_aeif_provenance() -> None:
    blockers = final_tier_parameter_provenance_blockers(
        {
            "network.neuron_model": "aeif_cond_beta_multisynapse",
            "calibration.mode": "paper_reduction",
            "neuron.Pyramidal": "analytic",
        },
        {"Pyramidal": 1},
    )

    assert "neuron.Pyramidal=analytic" in blockers


def test_final_tier_rejects_partial_source_location_transfer_mode() -> None:
    blockers = final_tier_parameter_provenance_blockers(
        {
            "network.neuron_model": "aglif_dend_cond_beta",
            "calibration.mode": "paper_reduction",
            "aglif.Pyramidal": "nestgpu-fi-fit",
            "dendritic_transfer.Pyramidal": "neuron-epsp-location-compressed-fit",
            "source_location_transfer.table": (
                "source-location-transfer-m2-row-validation-passed;"
                "mode=inhibitory_dend;"
                "table=source_location_transfer_syndata120_budget_weighted.json;"
                "applied=45"
            ),
        },
        {"Pyramidal": 1},
    )

    assert any(
        blocker.startswith("source_location_transfer.table=")
        for blocker in blockers
    )


def test_aeif_fitted_loader_rejects_malformed_validation_passed(
    tmp_path: Path,
) -> None:
    analytic_path = Path(__file__).parents[1] / "src/ca1/params/neuron_parameters.json"
    analytic = load_neuron_params(analytic_path)
    records: dict[str, dict[str, object]] = {
        name: {
            "C_m": params.C_m,
            "g_L": params.g_L,
            "E_L": params.E_L,
            "V_th": params.V_th,
            "V_reset": params.V_reset,
            "Delta_T": params.Delta_T,
            "a": params.a,
            "b": params.b,
            "tau_w": params.tau_w,
            "t_ref": params.t_ref,
            "V_peak": params.V_peak,
            "fit_provenance": "nest-validated",
            "validation": {"passed": True},
        }
        for name, params in analytic.items()
    }
    records["Pyramidal"]["validation"] = {"passed": "true"}
    fit_path = tmp_path / "neuron_parameters_fitted.json"
    _ = fit_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="validation.passed.*true"):
        _ = _load_fitted(fit_path, analytic)


def test_aeif_fitted_loader_rejects_nonfinite_validation_payload(
    tmp_path: Path,
) -> None:
    analytic_path = Path(__file__).parents[1] / "src/ca1/params/neuron_parameters.json"
    analytic = load_neuron_params(analytic_path)
    records: dict[str, dict[str, object]] = {
        name: {
            "C_m": params.C_m,
            "g_L": params.g_L,
            "E_L": params.E_L,
            "V_th": params.V_th,
            "V_reset": params.V_reset,
            "Delta_T": params.Delta_T,
            "a": params.a,
            "b": params.b,
            "tau_w": params.tau_w,
            "t_ref": params.t_ref,
            "V_peak": params.V_peak,
            "fit_provenance": "nest-validated",
            "validation": {"passed": True},
        }
        for name, params in analytic.items()
    }
    records["Pyramidal"]["validation"] = {
        "passed": True,
        "metrics": {"max_z": math.nan},
    }
    fit_path = tmp_path / "neuron_parameters_fitted.json"
    _ = fit_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        _ = _load_fitted(fit_path, analytic)


def _aglif_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "V_th": -52.0,
        "E_L": -64.0,
        "C_m": 100.0,
        "tau_m": 8.0,
        "k_adap": 0.02,
        "k1": 0.01,
        "k2": 0.03,
        "A1": 7.0,
        "A2": 11.0,
        "I_e": 0.0,
        "V_peak": -47.0,
        "V_reset": -66.0,
        "t_ref": 2.0,
        "fit_provenance": "nestgpu-fi-fit",
        "validation": {"passed": True},
    }
    record.update(overrides)
    return record


def test_aglif_loader_rejects_malformed_validation_passed(
    tmp_path: Path,
) -> None:
    records = {name: _aglif_record() for name in CELL_TEMPLATES}
    records["Pyramidal"]["validation"] = {"passed": "true"}
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    _ = fit_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="validation.passed.*true"):
        _ = load_aglif_params(fit_path)


def test_aglif_loader_rejects_nonfinite_validation_payload(
    tmp_path: Path,
) -> None:
    records = {name: _aglif_record() for name in CELL_TEMPLATES}
    records["Pyramidal"]["validation"] = {
        "passed": True,
        "passive": {"sag": math.inf},
    }
    fit_path = tmp_path / "aglif_parameters_fitted.json"
    _ = fit_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        _ = load_aglif_params(fit_path)


def _transfer_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "dend_C_frac": 0.4,
        "dend_leak_scale": 1.0,
        "g_c_scale": 1.0,
        "fit_provenance": "neuron-synaptic-transfer-fit",
        "validation": {"passed": True},
    }
    record.update(overrides)
    return record


def test_dendritic_transfer_loader_rejects_malformed_validation_passed(
    tmp_path: Path,
) -> None:
    records = {name: _transfer_record() for name in CELL_TEMPLATES}
    records["Pyramidal"]["validation"] = {"passed": "true"}
    fit_path = tmp_path / "dendritic_transfer_fitted.json"
    _ = fit_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="validation.passed.*true"):
        _ = load_dendritic_transfer_params(fit_path)


def test_location_transfer_rejects_duplicate_rows(tmp_path: Path) -> None:
    row = {
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
    table = tmp_path / "duplicate_transfer_rows.json"
    _ = table.write_text(json.dumps([row, row]), encoding="utf-8")
    spec = NetworkSpec(
        name="duplicate-transfer-row",
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

    with pytest.raises(ValueError, match="duplicate source-location transfer row"):
        _ = apply_location_transfer(spec, "all_dend", table)


def test_aglif_dend_compartments_reject_missing_source_location_domain() -> None:
    with pytest.raises(ValueError, match="missing source-location domain"):
        _ = aglif_dend_compartments(
            ("AMPA_fast__e0__tr99__td99__dend",),
            "Pyramidal",
            frozenset({"AMPA_fast__e0__tr99__td99__dend"}),
        )


def test_aglif_dend_compartments_reject_cell_type_free_dendritic_fallback() -> None:
    with pytest.raises(ValueError, match="cell_type is required"):
        _ = aglif_dend_compartments(
            ("GABA_A_slow__em60__tr0p11__td9p7__distal__dend",),
        )


def test_aglif_dend_compartments_require_declared_used_dendritic_ports() -> None:
    with pytest.raises(ValueError, match="required_dendritic_ports is required"):
        _ = aglif_dend_compartments(
            ("AMPA_fast__e0__tr0p3__td0p6__dend",),
            "PV_Basket",
        )


def test_syndata137_aglif_dend_requires_validated_em75_transfer_row() -> None:
    with pytest.raises(
        IncompleteLocationTransferError,
        match="Neurogliaform->Pyramidal:GABA_A_slow__em75__tr9__td39__dend",
    ):
        _ = build_network_spec({
            "name": "syndata137_aglif_dend_transfer_gate",
            "neuron_model": "aglif_dend_cond_beta",
            "compartment_aware_synapses": True,
            "receptor_port_strategy": "budget_weighted",
            "syndata_variant": 137,
            "conndata_index": 430,
            "conndata_count_mode": "per_cell",
            "cellnumbers_index": 101,
            "source_location_transfer_mode": "all_dend",
            "source_location_transfer_table": (
                "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
            ),
        })


def test_aglif_dend_compartments_reject_syndata137_em75_domain_fallback() -> None:
    receptor = "GABA_A_slow__em75__tr9__td39__dend"

    with pytest.raises(
        ValueError,
        match="missing source-location domain for Pyramidal:GABA_A_slow__em75",
    ):
        _ = aglif_dend_compartments(
            (receptor,),
            "Pyramidal",
            frozenset({receptor}),
        )


def test_final_tier_rejects_source_location_table_digest_mismatch(
    tmp_path: Path,
) -> None:
    canonical = (
        Path(__file__).parents[1]
        / "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
    )
    rows = json.loads(canonical.read_text(encoding="utf-8"))
    for row in rows:
        if (
            row.get("post") == "Pyramidal"
            and row.get("port") == "GABA_A_slow__em60__tr0p11__td9p7__dend"
        ):
            row["loc"] = "distal"
    transfer_table = tmp_path / canonical.name
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")

    spec = build_network_spec({
        "name": "digest_mismatch_transfer_table",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "syndata_variant": 120,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": str(transfer_table),
    })

    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_parameter_provenance_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert "diagnostic-noncanonical-source-location-transfer" in provenance[
        "source_location_transfer.table"
    ]
    assert any(
        blocker.startswith("source_location_transfer.table=")
        for blocker in blockers
    )


def test_aglif_dend_compartments_use_configured_source_location_table(
    tmp_path: Path,
) -> None:
    canonical = (
        Path(__file__).parents[1]
        / "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
    )
    receptor = "GABA_A_slow__em60__tr0p11__td9p7__dend"
    rows = json.loads(canonical.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("post") == "Pyramidal" and row.get("port") == receptor:
            row["loc"] = "distal"
    transfer_table = tmp_path / canonical.name
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")

    spec = build_network_spec({
        "name": "configured_transfer_domain_table",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "syndata_variant": 120,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": str(transfer_table),
    })

    canonical_domain = aglif_dend_compartments(
        (receptor,),
        "Pyramidal",
        frozenset({receptor}),
    )
    configured_domain = aglif_dend_compartments(
        (receptor,),
        "Pyramidal",
        frozenset({receptor}),
        spec.source_location_transfer_table,
    )

    assert canonical_domain == [1.0]
    assert configured_domain == [2.0]
