from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import h5py

from ca1.analysis.hdf_rate_summary import summarize_hdf_rates
from ca1.validation.targets import MODEL_RATES_HZ, RATE_REL_TOL

_PYRAMIDAL_TARGET: Final = MODEL_RATES_HZ["Pyramidal"]


def _write_result(
    path: Path,
    *,
    duration_s: float,
    crop_first_ms: float,
    spike_trains: list[list[float]],
    tier: str = "full",
    scale: float = 1.0,
    parameter_provenance: dict[str, str] | None = None,
    diagnostic_provenance: dict[str, str] | None = None,
) -> None:
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["duration_s"] = duration_s
        meta.attrs["scale"] = scale
        meta.attrs["tier"] = tier
        meta.attrs["crop_first_ms"] = crop_first_ms
        meta.attrs["lfp_proxy"] = "pyramidal_synaptic_current"
        meta.attrs["parameter_provenance_json"] = json.dumps(
            parameter_provenance or {},
            sort_keys=True,
        )
        meta.attrs["diagnostic_provenance_json"] = json.dumps(
            diagnostic_provenance or {},
            sort_keys=True,
        )
        spikes = h5.create_group("spikes")
        pyramidal = spikes.create_group("Pyramidal")
        for index, train in enumerate(spike_trains):
            pyramidal.create_dataset(str(index), data=train)
        n_cells = h5.create_group("n_cells_per_type")
        n_cells.attrs["Pyramidal"] = len(spike_trains)
        h5.create_dataset("lfp", data=[0.0, 1.0, 0.0])


def test_summary_counts_stored_post_crop_spikes_without_double_crop(tmp_path: Path) -> None:
    # Given: HDF spikes are already post-crop and shifted, even though meta records a crop.
    result_path = tmp_path / "post_crop.h5"
    _write_result(
        result_path,
        duration_s=1.0,
        crop_first_ms=200.0,
        spike_trains=[[0.1], []],
    )

    # When: the summary computes rates from the persisted HDF.
    summary = summarize_hdf_rates(result_path)
    pyramidal = summary["cell_types"]["Pyramidal"]

    # Then: the 0.1s spike is still counted and only the denominator is adjusted.
    assert summary["analysis_window_s"] == 0.8
    assert pyramidal["total_spikes"] == 1
    assert pyramidal["raw_spike_count"] == 1
    assert pyramidal["cropped_as_stored_spike_count"] == 1
    assert pyramidal["mean_rate_hz"] == 0.625


def test_summary_reports_target_band_pass_and_fail(tmp_path: Path) -> None:
    # Given: one file lands inside the Pyramidal target band and one lands below it.
    passing_path = tmp_path / "passing.h5"
    failing_path = tmp_path / "failing.h5"
    _write_result(
        passing_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        spike_trains=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    )
    _write_result(
        failing_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        spike_trains=[[0.1, 0.2]],
    )

    # When: both summaries are computed.
    passing = summarize_hdf_rates(passing_path)["cell_types"]["Pyramidal"]
    failing = summarize_hdf_rates(failing_path)["cell_types"]["Pyramidal"]

    # Then: target, band, and pass status are explicit.
    expected_band = [
        _PYRAMIDAL_TARGET * (1.0 - RATE_REL_TOL),
        _PYRAMIDAL_TARGET * (1.0 + RATE_REL_TOL),
    ]
    assert passing["target_hz"] == _PYRAMIDAL_TARGET
    assert passing["target_band_hz"] == expected_band
    assert passing["target_pass"] is True
    assert failing["target_pass"] is False


def test_summary_reports_electrode_roi_validation_rates(tmp_path: Path) -> None:
    # Given: two stored Pyramidal cells, but only the first lies inside the ROI.
    result_path = tmp_path / "roi_rates.h5"
    _write_result(
        result_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        spike_trains=[[0.1, 0.2], [0.3, 0.4]],
    )
    with h5py.File(result_path, "a") as h5:
        meta = h5["meta"]
        meta.attrs["analysis_roi_center_um"] = [0.0, 0.0, 0.0]
        meta.attrs["analysis_roi_radius_um"] = 5.0
        meta.attrs["analysis_roi_distance_mode"] = "xyz"
        positions = h5.create_group("cell_positions")
        positions.create_dataset(
            "Pyramidal",
            data=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        )

    # When: HDF rates are summarized.
    summary = summarize_hdf_rates(result_path)
    pyramidal = summary["cell_types"]["Pyramidal"]

    # Then: all-cell and ROI validation rates are both visible and distinct.
    assert summary["provenance"]["has_n_pole_lfp_context"] is True
    assert pyramidal["mean_rate_hz"] == 2.0
    assert pyramidal["validation_scope"] == "electrode_roi"
    assert pyramidal["validation_n_cells"] == 1
    assert pyramidal["roi_cells"] == 1
    assert pyramidal["roi_total_spikes"] == 2
    assert pyramidal["validation_mean_rate_hz"] == 2.0


def test_summary_exposes_source_pool_fallback_and_diagnostic_keys(
    tmp_path: Path,
) -> None:
    # Given: source-pool provenance records a compressed diagnostic fallback.
    result_path = tmp_path / "provenance.h5"
    _write_result(
        result_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        spike_trains=[[]],
        parameter_provenance={
            "network.afferent_topology": "source_pool",
            "network.afferent_source_count_max": "250000",
            "network.afferent_source_pool_size": "4096",
            "network.afferent_source_pool_indegree": "64",
            "network.afferent_source_pool_weight_rule": (
                "source_pool_path_rate_preserving"
            ),
            "network.afferent_poisson_rule": "source_pool_path_rate_preserving",
        },
        diagnostic_provenance={
            "diagnostic.audit": "no-overrides",
            "diagnostic.config": "CA1_TEST_OVERRIDE=1",
        },
    )

    # When: provenance is summarized.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: fallback visibility and diagnostic keys are machine-readable.
    assert provenance["diagnostic_keys"] == [
        "diagnostic.audit",
        "diagnostic.config",
    ]
    assert provenance["source_pool_compressed"] is True
    assert provenance["source_pool_size"] == 4096
    assert provenance["source_count_max"] == 250000
    assert provenance["final_tier_eligible"] is False
    assert (
        "structure: network.afferent_poisson_rule="
        "source_pool_path_rate_preserving is diagnostic; full-tier requires "
        "literal CA3/ECIII source-pool connectivity"
    ) in provenance["eligibility_failures"]
    assert (
        "structure: network.afferent_source_pool_size is a compressed "
        "diagnostic source-count fallback; full-tier requires at least "
        "network.afferent_source_count_max=250000, got 4096"
    ) in provenance["warnings"]
    assert (
        "diagnostic: diagnostic.config=CA1_TEST_OVERRIDE=1"
    ) in provenance["eligibility_failures"]


def test_summary_marks_compound_afferent_superposition_diagnostic(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "compound_provenance.h5"
    _write_result(
        result_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        spike_trains=[[]],
        parameter_provenance={
            "network.afferent_topology": "compound",
            "network.afferent_poisson_rule": (
                "postcell_independent_poisson_superposition"
            ),
            "network.afferent_source_count_max": "250000",
            "network.afferent_source_pool_size": "250000",
        },
        diagnostic_provenance={"diagnostic.audit": "no-overrides"},
    )

    provenance = summarize_hdf_rates(result_path)["provenance"]

    assert provenance["final_tier_eligible"] is False
    assert (
        "structure: network.afferent_topology=compound is a diagnostic "
        "rate-superposition fallback; full-tier requires literal CA3/ECIII "
        "source-pool connectivity"
    ) in provenance["eligibility_failures"]


def test_summary_marks_missing_provenance_ineligible(tmp_path: Path) -> None:
    # Given: a legacy HDF result lacks both provenance JSON attributes.
    result_path = tmp_path / "missing_provenance.h5"
    _write_result(
        result_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        spike_trains=[[]],
    )
    with h5py.File(result_path, "r+") as h5:
        del h5["meta"].attrs["parameter_provenance_json"]
        del h5["meta"].attrs["diagnostic_provenance_json"]

    # When: provenance is summarized.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: final-tier ineligibility is explicit, not only warning text.
    assert provenance["final_tier_eligible"] is False
    assert "parameter_provenance_json missing" in provenance["eligibility_failures"]
    assert "diagnostic_provenance_json missing" in provenance["eligibility_failures"]
    assert any(
        failure.startswith(
            "lfp: LFP proxy source recorded as pyramidal_synaptic_current"
        )
        for failure in provenance["eligibility_failures"]
    )
    assert provenance["warnings"] == provenance["eligibility_failures"]


def test_summary_marks_scaled_tier_ineligible_even_at_full_scale(
    tmp_path: Path,
) -> None:
    # Given: a scale=1.0 diagnostic run explicitly stored as scaled-tier evidence.
    result_path = tmp_path / "scaled_tier.h5"
    _write_result(
        result_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        tier="scaled",
        scale=1.0,
        spike_trains=[[]],
        parameter_provenance={
            "network.neuron_model": "aeif_cond_beta_multisynapse",
            "neuron.Pyramidal": "nest-validated",
            "calibration.mode": "paper_reduction",
        },
        diagnostic_provenance={"diagnostic.audit": "no-overrides"},
    )

    # When: provenance is summarized.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: a diagnostic/scaled run cannot masquerade as final-tier evidence.
    assert provenance["final_tier_eligible"] is False
    assert "tier=scaled; final-tier evidence requires tier=full" in provenance[
        "eligibility_failures"
    ]


def test_summary_marks_diagnostic_calibration_ineligible(
    tmp_path: Path,
) -> None:
    # Given: a full-scale result whose parameter provenance records diagnostic tuning.
    result_path = tmp_path / "diagnostic_calibration.h5"
    _write_result(
        result_path,
        duration_s=1.0,
        crop_first_ms=0.0,
        spike_trains=[[]],
        parameter_provenance={
            "network.neuron_model": "aeif_cond_beta_multisynapse",
            "neuron.Pyramidal": "nest-validated",
            "calibration.mode": "diagnostic",
            "calibration.afferent_weight_scale": "0.2",
        },
        diagnostic_provenance={"diagnostic.audit": "no-overrides"},
    )

    # When: provenance is summarized.
    provenance = summarize_hdf_rates(result_path)["provenance"]

    # Then: diagnostic calibration is a visible final-tier blocker.
    assert provenance["final_tier_eligible"] is False
    assert "parameter: calibration.mode=diagnostic" in provenance[
        "eligibility_failures"
    ]
    assert "parameter: calibration.afferent_weight_scale=0.2" in provenance[
        "eligibility_failures"
    ]
