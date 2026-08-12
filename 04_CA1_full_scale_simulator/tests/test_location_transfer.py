from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from ca1.analysis.location_transfer import (
    IncompleteLocationTransferError,
    UnvalidatedLocationTransferError,
    apply_location_transfer,
)
from ca1.analysis.location_transfer_wave import LocationTransferCase, postprocess
from ca1.types import Afferent, NetworkSpec, Projection


def _write_empty_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "location_transfer.json"
    _ = transfer_table.write_text(json.dumps([]), encoding="utf-8")
    return transfer_table


def _write_complete_transfer_table(tmp_path: Path) -> Path:
    transfer_table = tmp_path / "location_transfer.json"
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


def _spec_with_missing_dend_projection() -> NetworkSpec:
    return NetworkSpec(
        name="missing-transfer",
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
                pre="SCA",
                post="PV_Basket",
                indegree=1.0,
                synapses_per_connection=1,
                weight_nS=0.2,
                receptor="GABA_A_slow__em60__tr0p432__td4p49__dend",
            )
        ],
        afferents=[],
        neuron_model="aglif_dend_cond_beta",
    )


def _spec_with_missing_dend_afferent() -> NetworkSpec:
    return NetworkSpec(
        name="missing-afferent-transfer",
        cell_types={},
        projections=[],
        afferents=[
            Afferent(
                name="CA3_to_Bistratified",
                post="Bistratified",
                n_source=100,
                synapses_per_cell=10.0,
                weight_nS=0.2,
                receptor="AMPA_fast__e0__tr2__td6p3__dend",
            )
        ],
        neuron_model="aglif_dend_cond_beta",
    )


def test_location_transfer_refuses_implicit_missing_row_fallback(
    tmp_path: Path,
) -> None:
    # Given: a source-location table that lacks a dendritic projection row.
    transfer_table = _write_empty_transfer_table(tmp_path)
    spec = _spec_with_missing_dend_projection()

    # When / Then: applying a transfer mode fails before defaulting to scale 1.0.
    with pytest.raises(
        IncompleteLocationTransferError,
        match="refusing implicit 1.0 fallback",
    ):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_location_transfer_prototype_override_returns_missing_rows(
    tmp_path: Path,
) -> None:
    # Given: a deliberately incomplete exploratory source-location table.
    transfer_table = _write_empty_transfer_table(tmp_path)
    spec = _spec_with_missing_dend_projection()

    # When: the caller explicitly marks the run as an incomplete prototype.
    updated, applied, missing = apply_location_transfer(
        spec,
        "all_dend",
        transfer_table,
        allow_incomplete_transfer_for_prototype=True,
    )

    # Then: the missing row is observable and no hidden attenuation is applied.
    assert applied == []
    assert missing == ["Pyramidal->PV_Basket:AMPA_fast__e0__tr0p07__td0p2__dend"]
    assert updated.projections[0].weight_nS == 0.2


def test_location_transfer_prefers_explicit_compensation_scale(
    tmp_path: Path,
) -> None:
    transfer_table = _write_complete_transfer_table(tmp_path)
    spec = _spec_with_missing_dend_projection()

    updated, applied, missing = apply_location_transfer(
        spec,
        "all_dend",
        transfer_table,
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


def test_location_transfer_refuses_boolean_only_m2_validation(
    tmp_path: Path,
) -> None:
    transfer_table = tmp_path / "location_transfer_boolean_only.json"
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
            "m2_validation": {
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="M2 .*validation",
    ):
        _ = apply_location_transfer(
            _spec_with_missing_dend_projection(),
            "all_dend",
            transfer_table,
        )


def test_location_transfer_refuses_failed_m2_response_validation(
    tmp_path: Path,
) -> None:
    transfer_table = tmp_path / "location_transfer_failed_response.json"
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
            "m2_validation": {
                "method": "user_m2-row-level-response-fidelity",
                "evidence_path": "evidence/m2_row_validation.json",
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.04,
                "compensated_ratio": 0.2,
                "abs_error": 0.1,
                "tolerance": 0.01,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="failed M2 response validation",
    ):
        _ = apply_location_transfer(
            _spec_with_missing_dend_projection(),
            "all_dend",
            transfer_table,
        )


def test_location_transfer_refuses_gaba_transfer_without_inhibitory_probe_metadata(
    tmp_path: Path,
) -> None:
    transfer_table = tmp_path / "location_transfer_gaba_passive_only.json"
    rows = [
        {
            "kind": "rec",
            "pre": "SCA",
            "post": "PV_Basket",
            "receptor": "GABA_A_slow",
            "port": "GABA_A_slow__em60__tr0p432__td4p49__dend",
            "loc": "prox",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.9,
            "reduced_ratio_est": 0.8,
            "transfer_scale": 1.125,
            "m2_validation": {
                "method": "user_m2-row-level-response-fidelity",
                "evidence_path": "evidence/m2_row_validation.json",
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.8,
                "compensated_ratio": 0.9,
                "abs_error": 0.0,
                "tolerance": 0.01,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="missing inhibitory M2 validation field probe_e_rev_mV",
    ):
        _ = apply_location_transfer(
            _spec_with_gaba_dend_projection(),
            "all_dend",
            transfer_table,
        )


def test_location_transfer_refuses_gaba_transfer_validated_with_epsp_probe(
    tmp_path: Path,
) -> None:
    transfer_table = tmp_path / "location_transfer_gaba_epsp_probe.json"
    rows = [
        {
            "kind": "rec",
            "pre": "SCA",
            "post": "PV_Basket",
            "receptor": "GABA_A_slow",
            "port": "GABA_A_slow__em60__tr0p432__td4p49__dend",
            "loc": "prox",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.9,
            "reduced_ratio_est": 0.8,
            "transfer_scale": 1.125,
            "m2_validation": {
                "method": "user_m2-row-level-response-fidelity",
                "evidence_path": "evidence/m2_row_validation.json",
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.8,
                "compensated_ratio": 0.9,
                "abs_error": 0.0,
                "tolerance": 0.01,
                "probe_e_rev_mV": 0.0,
                "probe_baseline_mV": -55.0,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(
        UnvalidatedLocationTransferError,
        match="does not match receptor E_rev=-60",
    ):
        _ = apply_location_transfer(
            _spec_with_gaba_dend_projection(),
            "all_dend",
            transfer_table,
        )


def test_location_transfer_accepts_gaba_transfer_with_inhibitory_probe_metadata(
    tmp_path: Path,
) -> None:
    transfer_table = tmp_path / "location_transfer_gaba_inhibitory_probe.json"
    rows = [
        {
            "kind": "rec",
            "pre": "SCA",
            "post": "PV_Basket",
            "receptor": "GABA_A_slow",
            "port": "GABA_A_slow__em60__tr0p432__td4p49__dend",
            "loc": "prox",
            "aglif_compartment": "dend",
            "morph_ratio_est": 0.9,
            "reduced_ratio_est": 0.8,
            "transfer_scale": 1.125,
            "m2_validation": {
                "method": "user_m2-row-level-inhibitory-response-fidelity",
                "evidence_path": "evidence/m2_row_validation.json",
                "passed": True,
                "sign_preserved": True,
                "low_signal": False,
                "measured_reduced_ratio": 0.8,
                "compensated_ratio": 0.9,
                "abs_error": 0.0,
                "tolerance": 0.01,
                "probe_e_rev_mV": -60.0,
                "probe_baseline_mV": -55.0,
            },
        }
    ]
    _ = transfer_table.write_text(json.dumps(rows), encoding="utf-8")

    updated, applied, missing = apply_location_transfer(
        _spec_with_gaba_dend_projection(),
        "all_dend",
        transfer_table,
    )

    assert missing == []
    assert applied == [
        {
            "pre": "SCA",
            "post": "PV_Basket",
            "receptor": "GABA_A_slow__em60__tr0p432__td4p49__dend",
            "loc": "prox",
            "scale": 1.125,
        }
    ]
    assert updated.projections[0].weight_nS == pytest.approx(0.225)


def test_location_transfer_refuses_missing_afferent_transfer_rows(
    tmp_path: Path,
) -> None:
    transfer_table = _write_empty_transfer_table(tmp_path)
    spec = _spec_with_missing_dend_afferent()

    with pytest.raises(
        IncompleteLocationTransferError,
        match="CA3->Bistratified:AMPA_fast__e0__tr2__td6p3__dend",
    ):
        _ = apply_location_transfer(spec, "all_dend", transfer_table)


def test_location_transfer_postprocess_rejects_zero_analyzed_cells(
    tmp_path: Path,
) -> None:
    # Given: a spike artifact with no analyzed cells.
    spike_path = tmp_path / "spikes.pkl"
    with spike_path.open("wb") as handle:
        pickle.dump({}, handle)
    case = LocationTransferCase(
        case="empty",
        out_dir=tmp_path,
        model="aglif_dend_cond_beta",
        conndata_index=430,
        syndata_variant=120,
        duration_s=1.0,
        crop_ms=50.0,
        afferent_rate_hz=0.65,
        recurrent_scale=None,
        afferent_weight_scale=None,
        ca3_source_scale=None,
        eciii_source_scale=None,
        gfast_scale=None,
        gslow_scale=None,
        gb_scale=None,
        dend_ampa_scale=None,
        delay_ms=None,
        compartment_aware_synapses=True,
        receptor_port_strategy="budget_weighted",
        transfer_mode="all_dend",
        transfer_table=tmp_path / "location_transfer.json",
        allow_incomplete_transfer_for_prototype=False,
    )

    # When / Then: postprocessing refuses to emit a report with 0 analyzed cells.
    with pytest.raises(ValueError, match="at least one analyzed cell"):
        _ = postprocess(
            case,
            spike_path,
            elapsed_s=0.0,
            parameter_provenance={},
            transfer_applied=[],
            transfer_missing=[],
        )


def test_location_transfer_postprocess_records_transfer_table_in_parameter_provenance(
    tmp_path: Path,
) -> None:
    # Given: a prototype source-location transfer run with analyzed spikes.
    spike_path = tmp_path / "spikes.pkl"
    with spike_path.open("wb") as handle:
        pickle.dump({"Pyramidal": [np.array([], dtype=float)]}, handle)
    case = LocationTransferCase(
        case="prototype-transfer",
        out_dir=tmp_path,
        model="aglif_dend_cond_beta",
        conndata_index=430,
        syndata_variant=120,
        duration_s=1.0,
        crop_ms=50.0,
        afferent_rate_hz=0.65,
        recurrent_scale=None,
        afferent_weight_scale=None,
        ca3_source_scale=None,
        eciii_source_scale=None,
        gfast_scale=None,
        gslow_scale=None,
        gb_scale=None,
        dend_ampa_scale=None,
        delay_ms=None,
        compartment_aware_synapses=True,
        receptor_port_strategy="budget_weighted",
        transfer_mode="all_dend",
        transfer_table=tmp_path / "location_transfer.json",
        allow_incomplete_transfer_for_prototype=False,
    )

    # When: postprocessing emits the JSON summary consumed as evidence.
    summary = postprocess(
        case,
        spike_path,
        elapsed_s=0.0,
        parameter_provenance={"aglif.Pyramidal": "nestgpu-fi-fit"},
        transfer_applied=[],
        transfer_missing=[],
    )

    # Then: the transfer table is also visible to parameter-provenance gates.
    provenance = cast(dict[str, object], summary["parameter_provenance"])
    transfer_record = provenance["source_location_transfer.table"]
    assert isinstance(transfer_record, str)
    assert transfer_record.startswith(
        "unvalidated-prototype-source-location-transfer"
    )


def test_location_transfer_postprocess_records_diagnostic_runtime_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a diagnostic run that reroutes selected excitatory ports to soma.
    spike_path = tmp_path / "spikes.pkl"
    with spike_path.open("wb") as handle:
        pickle.dump({"PV_Basket": [np.array([], dtype=float)]}, handle)
    monkeypatch.setenv(
        "CA1_AGLIF_DEND_EXC_SOMA_CELL_TYPES",
        "PV_Basket,Bistratified",
    )
    case = LocationTransferCase(
        case="diagnostic-routing",
        out_dir=tmp_path,
        model="aglif_dend_cond_beta",
        conndata_index=430,
        syndata_variant=120,
        duration_s=1.0,
        crop_ms=50.0,
        afferent_rate_hz=0.65,
        recurrent_scale=None,
        afferent_weight_scale=None,
        ca3_source_scale=None,
        eciii_source_scale=None,
        gfast_scale=None,
        gslow_scale=None,
        gb_scale=None,
        dend_ampa_scale=None,
        delay_ms=None,
        compartment_aware_synapses=True,
        receptor_port_strategy="budget_weighted",
        transfer_mode="none",
        transfer_table=tmp_path / "location_transfer.json",
        allow_incomplete_transfer_for_prototype=False,
    )

    # When: postprocessing writes the diagnostic summary artifact.
    summary = postprocess(
        case,
        spike_path,
        elapsed_s=0.0,
        parameter_provenance={},
        transfer_applied=[],
        transfer_missing=[],
    )

    # Then: the routing override is visible and cannot become a hidden fallback.
    assert summary["diagnostic_environment_provenance"] == {
        "env.CA1_AGLIF_DEND_EXC_SOMA_CELL_TYPES": "PV_Basket,Bistratified"
    }
    assert summary["diagnostic_provenance"] == {
        "env.CA1_AGLIF_DEND_EXC_SOMA_CELL_TYPES": "PV_Basket,Bistratified"
    }


def test_location_transfer_postprocess_records_clean_diagnostic_audit(
    tmp_path: Path,
) -> None:
    # Given: a non-diagnostic run with no active CA1 diagnostic env overrides.
    spike_path = tmp_path / "spikes.pkl"
    with spike_path.open("wb") as handle:
        pickle.dump({"PV_Basket": [np.array([], dtype=float)]}, handle)
    case = LocationTransferCase(
        case="clean-routing",
        out_dir=tmp_path,
        model="aglif_dend_cond_beta",
        conndata_index=430,
        syndata_variant=120,
        duration_s=1.0,
        crop_ms=50.0,
        afferent_rate_hz=0.65,
        recurrent_scale=None,
        afferent_weight_scale=None,
        ca3_source_scale=None,
        eciii_source_scale=None,
        gfast_scale=None,
        gslow_scale=None,
        gb_scale=None,
        dend_ampa_scale=None,
        delay_ms=None,
        compartment_aware_synapses=True,
        receptor_port_strategy="budget_weighted",
        transfer_mode="none",
        transfer_table=tmp_path / "location_transfer.json",
        allow_incomplete_transfer_for_prototype=False,
    )

    # When: postprocessing writes the diagnostic summary artifact.
    summary = postprocess(
        case,
        spike_path,
        elapsed_s=0.0,
        parameter_provenance={},
        transfer_applied=[],
        transfer_missing=[],
    )

    # Then: absence of diagnostic overrides is audited explicitly.
    assert summary["diagnostic_environment_provenance"] == {}
    assert summary["diagnostic_provenance"] == {"diagnostic.audit": "no-overrides"}
