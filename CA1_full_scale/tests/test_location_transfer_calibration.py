from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ca1.analysis.location_transfer_wave import LocationTransferCase, calibration_for
from ca1.analysis.location_transfer_wave import postprocess


def _case(
    *,
    projection_weight_scales: dict[str, float] | None = None,
    afferent_post_weight_scales: dict[str, float] | None = None,
) -> LocationTransferCase:
    return LocationTransferCase(
        case="calibration",
        out_dir=Path("/tmp"),
        model="aglif_dend_cond_beta",
        conndata_index=430,
        syndata_variant=120,
        duration_s=0.4,
        crop_ms=50.0,
        afferent_rate_hz=0.65,
        recurrent_scale=0.2,
        afferent_weight_scale=None,
        ca3_source_scale=None,
        eciii_source_scale=None,
        gfast_scale=None,
        gslow_scale=None,
        gb_scale=None,
        dend_ampa_scale=3.0,
        delay_ms=3.0,
        compartment_aware_synapses=True,
        receptor_port_strategy="budget_weighted",
        transfer_mode="all_dend",
        transfer_table=Path("/tmp/location_transfer.json"),
        allow_incomplete_transfer_for_prototype=True,
        projection_weight_scales=projection_weight_scales,
        afferent_post_weight_scales=afferent_post_weight_scales,
    )


def test_targeted_transfer_case_calibration_is_diagnostic() -> None:
    calibration = calibration_for(
        _case(
            projection_weight_scales={"Pyramidal->PV_Basket": 3.0},
            afferent_post_weight_scales={"O_LM": 0.3},
        )
    )

    assert calibration["mode"] == "diagnostic"
    assert calibration["projection_weight_scales"] == {"Pyramidal->PV_Basket": 3.0}
    assert calibration["afferent_post_weight_scales"] == {"O_LM": 0.3}
    assert calibration["recurrent_weight_scale"] == 0.2
    assert calibration["dendritic_ampa_weight_scale"] == 3.0


def test_postprocess_records_targeted_calibration_diagnostic_provenance(
    tmp_path: Path,
) -> None:
    spike_path = tmp_path / "spikes.pkl"
    with spike_path.open("wb") as handle:
        pickle.dump({"PV_Basket": [np.array([], dtype=float)]}, handle)

    summary = postprocess(
        _case(
            projection_weight_scales={"Pyramidal->PV_Basket": 3.0},
            afferent_post_weight_scales={"O_LM": 0.3},
        ),
        spike_path,
        elapsed_s=0.0,
        parameter_provenance={},
        transfer_applied=[],
        transfer_missing=[],
    )

    assert summary["diagnostic_config_provenance"] == {
        "config.calibration.afferent_post_weight_scales": '{"O_LM": 0.3}',
        "config.calibration.dendritic_ampa_weight_scale": "3.0",
        "config.calibration.mode": "diagnostic",
        "config.calibration.projection_weight_scales": (
            '{"Pyramidal->PV_Basket": 3.0}'
        ),
        "config.calibration.recurrent_weight_scale": "0.2",
    }
    assert summary["diagnostic_provenance"] == summary[
        "diagnostic_config_provenance"
    ]
