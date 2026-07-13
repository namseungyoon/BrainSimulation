from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ca1.config import build_network_spec
from ca1.params.provenance import (
    diagnostic_config_provenance,
    diagnostic_environment_provenance,
    fit_file_provenance,
    parameter_provenance_for_spec,
)
from ca1.validation.provenance import (
    final_tier_diagnostic_provenance_blockers,
    final_tier_parameter_provenance_blockers,
)
from ca1.validation.network_provenance import final_tier_network_structure_blockers


def test_aeif_provenance_reports_active_neuron_params() -> None:
    spec = build_network_spec({
        "name": "aeif",
        "neuron_model": "aeif_cond_beta_multisynapse",
    })
    provenance = parameter_provenance_for_spec(spec)

    assert "neuron.Pyramidal" in provenance
    assert not any(key.startswith("aglif.") for key in provenance)
    assert not any(key.startswith("izhikevich.") for key in provenance)


def test_aglif_dend_provenance_omits_inactive_aeif_params() -> None:
    spec = build_network_spec({
        "name": "aglif_dend",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
    })
    provenance = parameter_provenance_for_spec(spec)

    assert not any(key.startswith("neuron.") for key in provenance)
    assert "aglif.Pyramidal" in provenance
    assert "dendritic_transfer.Pyramidal" in provenance
    assert provenance["synapse.receptor_ports"].startswith(
        "syndata120-compartment-aware-20port-budget_weighted"
    )
    assert ";sha256=" in provenance["synapse.receptor_ports"]


def test_aglif_dend_final_tier_requires_source_location_transfer_record() -> None:
    spec = build_network_spec({
        "name": "aglif_dend_without_source_location_transfer",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
    })
    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_parameter_provenance_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert "source_location_transfer.table" not in provenance
    assert "source_location_transfer.table=missing" in blockers


def test_aglif_dend_provenance_accepts_source_backed_shared_port_resolution() -> None:
    spec = build_network_spec({
        "name": "aglif_dend_source_location_transfer",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": (
            "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
        ),
    })

    provenance = parameter_provenance_for_spec(spec)
    blockers = final_tier_parameter_provenance_blockers(
        provenance,
        spec.scaled_counts(),
    )

    transfer_rows = json.loads(
        Path(
            "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
        ).read_text(encoding="utf-8")
    )
    olm_pyramidal = next(
        row for row in transfer_rows
        if row["pre"] == "O_LM"
        and row["post"] == "Pyramidal"
        and row["port"] == "GABA_A_slow__em60__tr0p11__td9p7__dend"
    )

    assert "source_location_transfer.mixed_domain_ports" not in provenance
    shared_resolution = provenance[
        "source_location_transfer.shared_port_resolution"
    ]
    assert (
        "O_LM->Pyramidal:GABA_A_slow__em60__tr0p11__td9p7__dend:"
        "dist->prox:conductance-weighted-proximal-representative"
    ) in shared_resolution
    assert olm_pyramidal["loc"] == "prox"
    assert olm_pyramidal["loc_original"] == "dist"
    assert olm_pyramidal["shared_port_domain_resolution"] == (
        "conductance-weighted-proximal-representative"
    )
    assert "source-location-transfer-m2-row-validation-passed" in provenance[
        "source_location_transfer.table"
    ]
    assert not any(
        blocker.startswith("source_location_transfer.")
        for blocker in blockers
    )


def test_final_tier_parameter_blockers_reject_hidden_mixed_domain_record() -> None:
    blockers = final_tier_parameter_provenance_blockers(
        {
            "network.neuron_model": "aglif_dend_cond_beta",
            "aglif.Pyramidal": "nestgpu-fi-fit",
            "dendritic_transfer.Pyramidal": (
                "neuron-epsp-location-compressed-fit;validation-passed"
            ),
            "source_location_transfer.table": (
                "source-location-transfer-m2-row-validation-passed"
            ),
            "source_location_transfer.mixed_domain_ports": (
                "hidden-compressed-prox-dist"
            ),
        },
        {"Pyramidal": 1},
    )

    assert "source_location_transfer.mixed_domain_ports=hidden-compressed-prox-dist" in blockers


def test_receptor_port_strategy_is_visible_in_provenance() -> None:
    spec = build_network_spec({
        "name": "aglif_dend_source_kinetics",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "preserve_fast_basket_bistratified",
    })
    provenance = parameter_provenance_for_spec(spec)

    assert provenance["synapse.receptor_ports"].startswith(
        "syndata120-compartment-aware-20port-preserve_fast_basket_bistratified"
    )
    assert ";sha256=" in provenance["synapse.receptor_ports"]


def test_demix_strategy_requires_visible_noncanonical_provenance(
    tmp_path: Path,
) -> None:
    source = Path(
        "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
    )
    rows = json.loads(source.read_text(encoding="utf-8"))
    for row in rows:
        if (
            row["pre"] == "O_LM"
            and row["post"] == "Pyramidal"
            and row["port"] == "GABA_A_slow__em60__tr0p11__td9p7__dend"
        ):
            row["port"] = "GABA_A_slow__em60__tr0p11__td9p7__distal__dend"

    transfer_table = tmp_path / "demix_transfer.json"
    transfer_table.write_text(json.dumps(rows), encoding="utf-8")
    spec = build_network_spec({
        "name": "aglif_dend_demix_source_location_transfer",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "demix_pyramidal_olm_gabaa_slow_distal",
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": str(transfer_table),
    })

    provenance = parameter_provenance_for_spec(spec)
    parameter_blockers = final_tier_parameter_provenance_blockers(
        provenance,
        spec.scaled_counts(),
    )
    structure_blockers = final_tier_network_structure_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert "source_location_transfer.mixed_domain_ports" not in provenance
    assert provenance["synapse.receptor_ports"].startswith(
        "syndata120-compartment-aware-20port-"
        "demix_pyramidal_olm_gabaa_slow_distal"
    )
    assert ";sha256=" in provenance["synapse.receptor_ports"]
    assert any(
        blocker.startswith(
            "source_location_transfer.table=diagnostic-noncanonical"
        )
        for blocker in parameter_blockers
    )
    assert any("synapse.receptor_ports" in blocker for blocker in structure_blockers)


@pytest.mark.parametrize(
    ("topology", "source_driver"),
    [
        ("compound", "compound_poisson_generator"),
        ("literal_source_graph", "precomputed_poisson_spike_generator"),
    ],
)
def test_parameter_provenance_records_afferent_source_driver(
    topology: str,
    source_driver: str,
) -> None:
    spec = build_network_spec({
        "name": f"{topology}_source_driver",
        "afferent_topology": topology,
    })
    provenance = parameter_provenance_for_spec(spec)

    assert provenance["network.afferent_topology"] == topology
    assert provenance["network.afferent_source_driver"] == source_driver


def test_parameter_provenance_records_configured_recurrent_topology() -> None:
    spec = build_network_spec({
        "name": "modeldb_fastconn_recurrent_topology",
        "recurrent_topology": "modeldb_fastconn_binned",
    })

    provenance = parameter_provenance_for_spec(spec)

    assert spec.recurrent_topology == "modeldb_fastconn_binned"
    assert provenance["network.recurrent_topology"] == "modeldb_fastconn_binned"


def test_build_network_spec_rejects_unknown_recurrent_topology() -> None:
    with pytest.raises(ValueError, match="unsupported recurrent_topology"):
        _ = build_network_spec({
            "name": "unknown_recurrent_topology",
            "recurrent_topology": "silent_fallback",
        })


def test_diagnostic_provenance_records_intrinsic_heterogeneity_env() -> None:
    provenance = diagnostic_environment_provenance({
        "CA1_INTRINSIC_HETEROGENEITY_VTH_SIGMA_MV": "1.0",
    })

    assert provenance == {
        "env.CA1_INTRINSIC_HETEROGENEITY_VTH_SIGMA_MV": "1.0",
    }
    assert final_tier_diagnostic_provenance_blockers(provenance) == [
        "env.CA1_INTRINSIC_HETEROGENEITY_VTH_SIGMA_MV=1.0",
    ]


def test_diagnostic_provenance_records_prototype_transfer_override_config() -> None:
    provenance = diagnostic_config_provenance({
        "allow_incomplete_transfer_for_prototype": True,
    })

    assert provenance == {
        "config.allow_incomplete_transfer_for_prototype": "True",
    }
    assert final_tier_diagnostic_provenance_blockers(provenance) == [
        "config.allow_incomplete_transfer_for_prototype=True",
    ]


def test_diagnostic_provenance_records_all_calibration_families() -> None:
    provenance = diagnostic_config_provenance({
        "calibration": {
            "mode": "diagnostic",
            "recurrent_weight_scale": 0.7,
            "recurrent_receptor_weight_scales": {"GABA_A_fast": 0.8},
            "afferent_weight_scale": 0.6,
            "afferent_source_weight_scales": {"CA3": 0.5},
            "dendritic_ampa_weight_scale": 1.3,
            "projection_weight_scales": {"Pyramidal->Pyramidal": 0.4},
            "afferent_weight_scales": {"CA3_to_Pyramidal": 0.9},
            "afferent_post_weight_scales": {"CCK_Basket": 0.2},
        },
    })

    assert provenance == {
        "config.calibration.mode": "diagnostic",
        "config.calibration.recurrent_weight_scale": "0.7",
        "config.calibration.recurrent_receptor_weight_scales": (
            '{"GABA_A_fast": 0.8}'
        ),
        "config.calibration.afferent_weight_scale": "0.6",
        "config.calibration.afferent_source_weight_scales": '{"CA3": 0.5}',
        "config.calibration.dendritic_ampa_weight_scale": "1.3",
        "config.calibration.projection_weight_scales": (
            '{"Pyramidal->Pyramidal": 0.4}'
        ),
        "config.calibration.afferent_weight_scales": '{"CA3_to_Pyramidal": 0.9}',
        "config.calibration.afferent_post_weight_scales": '{"CCK_Basket": 0.2}',
    }
    blockers = final_tier_diagnostic_provenance_blockers(provenance)
    assert (
        "config.calibration.recurrent_receptor_weight_scales="
        '{"GABA_A_fast": 0.8}'
    ) in blockers
    assert "config.calibration.afferent_source_weight_scales={\"CA3\": 0.5}" in blockers


def test_diagnostic_provenance_records_zero_calibration_scales() -> None:
    provenance = diagnostic_config_provenance({
        "calibration": {
            "mode": "diagnostic",
            "recurrent_weight_scale": 0.0,
            "projection_weight_scales": {"Pyramidal->Pyramidal": 0.0},
            "afferent_weight_scales": {"CA3_to_Pyramidal": 0.0},
        },
    })

    assert provenance == {
        "config.calibration.mode": "diagnostic",
        "config.calibration.recurrent_weight_scale": "0.0",
        "config.calibration.projection_weight_scales": (
            '{"Pyramidal->Pyramidal": 0.0}'
        ),
        "config.calibration.afferent_weight_scales": '{"CA3_to_Pyramidal": 0.0}',
    }
    assert final_tier_diagnostic_provenance_blockers(provenance) == [
        "config.calibration.afferent_weight_scales={\"CA3_to_Pyramidal\": 0.0}",
        "config.calibration.mode=diagnostic",
        "config.calibration.projection_weight_scales={\"Pyramidal->Pyramidal\": 0.0}",
        "config.calibration.recurrent_weight_scale=0.0",
    ]


def test_izhikevich_provenance_omits_inactive_aeif_params() -> None:
    spec = build_network_spec({
        "name": "izh",
        "neuron_model": "izhikevich_cond_beta",
    })
    provenance = parameter_provenance_for_spec(spec)

    assert not any(key.startswith("neuron.") for key in provenance)
    assert "izhikevich.Pyramidal" in provenance
    assert not any(key.startswith("aglif.") for key in provenance)


def test_fit_file_provenance_requires_explicit_fit_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fit.json"
    _ = path.write_text(
        json.dumps({"Pyramidal": {"loss": 0.0}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fit_provenance.*required"):
        _ = fit_file_provenance(
            path=path,
            prefix="aglif",
            expected_cells={"Pyramidal"},
        )


def test_fit_file_provenance_marks_missing_validation_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fit.json"
    _ = path.write_text(
        json.dumps({"Pyramidal": {"fit_provenance": "nestgpu-fi-fit"}}),
        encoding="utf-8",
    )

    provenance = fit_file_provenance(
        path=path,
        prefix="aglif",
        expected_cells={"Pyramidal"},
    )

    assert provenance["aglif.Pyramidal"] == "nestgpu-fi-fit;validation-missing"


def test_fit_file_provenance_rejects_nonfinite_validation_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fit.json"
    _ = path.write_text(
        json.dumps({
            "Pyramidal": {
                "fit_provenance": "nestgpu-fi-fit",
                "validation": {
                    "passed": True,
                    "metrics": {"max_z": math.nan},
                },
            },
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-standard JSON constant 'NaN'"):
        _ = fit_file_provenance(
            path=path,
            prefix="aglif",
            expected_cells={"Pyramidal"},
        )


def test_fit_file_provenance_rejects_self_certified_passed_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fit.json"
    _ = path.write_text(
        json.dumps(
            {
                "Pyramidal": {
                    "fit_provenance": "nestgpu-fi-fit",
                    "validation": {"passed": True},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires non-empty protocol"):
        _ = fit_file_provenance(
            path=path,
            prefix="aglif",
            expected_cells={"Pyramidal"},
        )


def test_final_tier_parameter_blockers_require_model_specific_cell_records() -> None:
    blockers = final_tier_parameter_provenance_blockers(
        {
            "network.neuron_model": "aglif_dend_cond_beta",
            "dendritic_transfer.Pyramidal": (
                "neuron-epsp-location-compressed-fit;validation-passed"
            ),
        },
        {"Pyramidal": 1},
    )

    assert "aglif.Pyramidal=missing" in blockers


def test_final_tier_parameter_blockers_reject_non_neutral_calibration() -> None:
    blockers = final_tier_parameter_provenance_blockers(
        {
            "network.neuron_model": "aglif_cond_beta",
            "aglif.Pyramidal": "nestgpu-fi-fit",
            "calibration.mode": "paper_reduction",
            "calibration.afferent_weight_scale": "0.2",
        },
        {"Pyramidal": 1},
    )

    assert "calibration.afferent_weight_scale=0.2" in blockers


def test_final_tier_parameter_blockers_require_explicit_paper_calibration_mode() -> None:
    blockers = final_tier_parameter_provenance_blockers(
        {
            "network.neuron_model": "aglif_cond_beta",
            "aglif.Pyramidal": "nestgpu-fi-fit",
        },
        {"Pyramidal": 1},
    )

    assert "calibration.mode=missing" in blockers


def test_final_tier_rejects_source_location_mode_substring_match() -> None:
    blockers = final_tier_parameter_provenance_blockers(
        {
            "network.neuron_model": "aglif_dend_cond_beta",
            "calibration.mode": "paper_reduction",
            "aglif.Pyramidal": "nestgpu-fi-fit",
            "dendritic_transfer.Pyramidal": (
                "neuron-epsp-location-compressed-fit;validation-passed"
            ),
            "source_location_transfer.table": (
                "source-location-transfer-m2-row-validation-passed;"
                "mode=all_dend_extra;"
                "table=source_location_transfer_syndata120_budget_weighted.json"
            ),
        },
        {"Pyramidal": 1},
    )

    assert any(
        blocker.startswith("source_location_transfer.table=")
        for blocker in blockers
    )


@pytest.mark.parametrize(
    "fit_provenance",
    [
        "diagnostic-fit;not-final",
        "diagnostic-fit;not_final",
        "diagnostic-source-location-transfer",
        "diagnostic_source_location_transfer",
        "user_m2-inhibitory-row-response-validated",
    ],
)
def test_final_tier_parameter_blockers_reject_not_final_and_diagnostic_tokens(
    fit_provenance: str,
) -> None:
    spec = build_network_spec({
        "name": "aglif_dend_token_gate",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
    })
    provenance = parameter_provenance_for_spec(spec)
    provenance["aglif.Pyramidal"] = fit_provenance

    blockers = final_tier_parameter_provenance_blockers(
        provenance,
        spec.scaled_counts(),
    )

    assert f"aglif.Pyramidal={fit_provenance}" in blockers
