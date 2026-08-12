from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import h5py

from ca1.analysis.hdf_rate_summary import summarize_hdf_rates

_FULL_COUNTS: Final = {
    "Axo": 1_470,
    "Bistratified": 2_210,
    "CCK_Basket": 3_600,
    "Ivy": 8_810,
    "Neurogliaform": 3_580,
    "O_LM": 1_640,
    "PV_Basket": 5_530,
    "Pyramidal": 311_500,
    "SCA": 400,
}


def _final_parameter_provenance() -> dict[str, str]:
    provenance = {
        "calibration.mode": "paper_reduction",
        "lfp.modeldb_n_pole_reduced_domain": "modeldb-n-pole-reduced-domain-lfp",
        "network.total_cells": "338740",
        "network.cell_types": "9",
        "network.recurrent_projections": "68",
        "network.afferents": "13",
        "network.afferent_sources": "CA3,ECIII",
        "network.afferent_source_count_total": "454700",
        "network.afferent_source_count_max": "250000",
        "network.afferent_topology": "literal_source_graph",
        "network.afferent_poisson_rule": "literal_shared_source_graph",
        "network.afferent_source_driver": "precomputed_poisson_spike_generator",
        "network.conndata_index": "430",
        "network.cellnumbers_index": "101",
        "network.conndata_count_mode": "per_cell",
        "network.recurrent_synapses": "441375540",
        "network.afferent_synapses": "4704026540",
        "network.total_synapses": "5145402080",
        "network.neuron_model": "aglif_cond_beta",
        "synapse.receptor_ports": "syndata120-single-compartment-20port-budget_weighted",
    }
    provenance.update({f"aglif.{cell_type}": "nestgpu-fi-fit" for cell_type in _FULL_COUNTS})
    return provenance


def _write_full_tier_result(
    path: Path,
    *,
    lfp_proxy: str,
    write_lfp: bool,
) -> None:
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["duration_s"] = 1.0
        meta.attrs["scale"] = 1.0
        meta.attrs["tier"] = "full"
        meta.attrs["crop_first_ms"] = 0.0
        meta.attrs["lfp_proxy"] = lfp_proxy
        meta.attrs["parameter_provenance_json"] = json.dumps(
            _final_parameter_provenance(),
            sort_keys=True,
        )
        meta.attrs["diagnostic_provenance_json"] = json.dumps(
            {"diagnostic.audit": "no-overrides"},
            sort_keys=True,
        )
        spikes = h5.create_group("spikes")
        spikes.create_group("Pyramidal").create_dataset("0", data=[])
        n_cells = h5.create_group("n_cells_per_type")
        for cell_type, count in _FULL_COUNTS.items():
            n_cells.attrs[cell_type] = count
        if write_lfp:
            h5.create_dataset("lfp", data=[0.0, 1.0, 0.0])


def test_summary_marks_spike_density_lfp_proxy_ineligible(tmp_path: Path) -> None:
    # Given: a full-tier HDF with final-looking parameters but spike-density LFP fallback.
    result_path = tmp_path / "spike_density_lfp.hdf5"
    _write_full_tier_result(
        result_path,
        lfp_proxy="pyramidal_spike_density",
        write_lfp=False,
    )

    # When: the posthoc HDF rate summary reports final-tier eligibility.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: the summary cannot call spike-density spectral evidence final-eligible.
    assert provenance["final_tier_eligible"] is False
    assert (
        "lfp: LFP proxy source recorded as pyramidal_spike_density"
    ) in "\n".join(provenance["eligibility_failures"])


def test_summary_marks_incomplete_spike_datasets_ineligible(tmp_path: Path) -> None:
    # Given: a full-tier HDF whose metadata declares more cells than stored spike trains
    # and stores only the reduced current LFP proxy.
    result_path = tmp_path / "incomplete_spikes.hdf5"
    _write_full_tier_result(
        result_path,
        lfp_proxy="pyramidal_synaptic_current",
        write_lfp=True,
    )

    # When: the posthoc HDF rate summary reports final-tier eligibility.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: incomplete spike artifacts are explicit blockers, not warning-only text.
    assert provenance["final_tier_eligible"] is False
    assert (
        "artifact: spikes.Pyramidal has 1 datasets but "
        "n_cells_per_type.Pyramidal declares 311500"
    ) in "\n".join(provenance["eligibility_failures"])
    assert (
        "lfp: LFP proxy source recorded as pyramidal_synaptic_current, "
        "a diagnostic/scaled proxy"
    ) in "\n".join(provenance["eligibility_failures"])


def test_summary_marks_claimed_lfp_current_without_lfp_dataset_ineligible(
    tmp_path: Path,
) -> None:
    # Given: metadata claims a current-derived LFP proxy, but no LFP dataset is stored.
    result_path = tmp_path / "missing_lfp_current.hdf5"
    _write_full_tier_result(
        result_path,
        lfp_proxy="pyramidal_synaptic_current",
        write_lfp=False,
    )

    # When: the posthoc HDF rate summary reports final-tier eligibility.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: the missing real LFP artifact is a final-tier blocker.
    assert provenance["final_tier_eligible"] is False
    assert (
        "lfp: LFP proxy metadata claims pyramidal_synaptic_current, but stored "
        "LFP presence is False; refusing hidden spectral fallback"
    ) in "\n".join(provenance["eligibility_failures"])


def test_summary_rejects_modeldb_n_pole_lfp_without_roi_context(
    tmp_path: Path,
) -> None:
    # Given: metadata claims final N-pole LFP, but the HDF lacks electrode ROI
    # and pyramidal position context needed to audit that claim.
    result_path = tmp_path / "claimed_n_pole_without_context.hdf5"
    _write_full_tier_result(
        result_path,
        lfp_proxy="modeldb_n_pole_reduced_domain_lfp",
        write_lfp=True,
    )

    # When: the posthoc HDF rate summary reports final-tier eligibility.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: a relabeled current trace cannot become final spectral evidence.
    assert provenance["final_tier_eligible"] is False
    assert (
        "lfp: modeldb_n_pole_reduced_domain_lfp requires electrode ROI and "
        "Pyramidal cell_positions context"
    ) in "\n".join(provenance["eligibility_failures"])
