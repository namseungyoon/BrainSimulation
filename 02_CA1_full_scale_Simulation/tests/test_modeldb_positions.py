from __future__ import annotations

import numpy as np

from ca1.sim.modeldb_positions import (
    MODELDB_NPOLE_ELECTRODE_ROI,
    ModelDbGeometry,
    electrode_roi_mask,
    filter_spikes_to_roi,
    modeldb_cell_positions,
)
from ca1.types import ElectrodeRoi


def test_modeldb_cell_positions_match_hoc_bin_centers_for_scaled_pyramidal() -> None:
    # Given: eight Pyramidal cells in the ModelDB layer-1 prism.
    geometry = ModelDbGeometry(
        longitudinal_um=4000.0,
        transverse_um=1000.0,
        layer_heights_um=(100.0, 50.0, 200.0, 100.0),
    )

    # When: positions are generated from the same bin-center equations as positionfcns.mod.
    positions = modeldb_cell_positions(
        {"Pyramidal": 8},
        geometry=geometry,
    )["Pyramidal"]

    # Then: HOC setBins yields 3 X bins, 3 Y bins, and 1 Z bin for this case.
    assert positions.shape == (8, 3)
    np.testing.assert_allclose(positions[0], [666.5, 166.5, 125.0])
    np.testing.assert_allclose(positions[-1], [3332.5, 499.5, 125.0])


def test_electrode_roi_mask_uses_strict_modeldb_radius_and_distance_mode() -> None:
    # Given: one point inside the 3D radius, one only inside the XY radius,
    # and one exactly on the boundary.
    positions = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 6.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )

    # When: the N-pole LFP ROI uses 3D distance and ModelDB's strict "< MaxEDist".
    xyz_mask = electrode_roi_mask(
        positions,
        ElectrodeRoi(center_um=(0.0, 0.0, 0.0), radius_um=5.0, distance_mode="xyz"),
    )
    xy_mask = electrode_roi_mask(
        positions,
        ElectrodeRoi(center_um=(0.0, 0.0, 0.0), radius_um=5.0, distance_mode="xy"),
    )

    # Then: Z distance matters for N-pole LFP, while XY mode mirrors spontpos_stimulation.hoc.
    assert xyz_mask.tolist() == [True, False, False]
    assert xy_mask.tolist() == [True, True, False]


def test_filter_spikes_to_roi_preserves_cell_order_and_drops_outside_cells() -> None:
    # Given: three spike trains with only the middle cell inside the ROI.
    spikes = {
        "Pyramidal": [
            np.asarray([0.1], dtype=np.float64),
            np.asarray([0.2, 0.3], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        ]
    }
    positions = {
        "Pyramidal": np.asarray(
            [[10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    }

    # When: the validation spike set is restricted to the electrode ROI.
    filtered = filter_spikes_to_roi(
        spikes,
        positions,
        ElectrodeRoi(center_um=(0.0, 0.0, 0.0), radius_um=5.0, distance_mode="xyz"),
    )

    # Then: spike train indices still align with the selected cell positions.
    assert list(filtered) == ["Pyramidal"]
    assert len(filtered["Pyramidal"]) == 1
    np.testing.assert_allclose(filtered["Pyramidal"][0], [0.2, 0.3])


def test_modeldb_default_electrode_roi_records_n_pole_defaults() -> None:
    # Given/When/Then: defaults mirror parameters.hoc for N-pole LFP validation.
    assert MODELDB_NPOLE_ELECTRODE_ROI.center_um == (200.0, 100.0, 120.0)
    assert MODELDB_NPOLE_ELECTRODE_ROI.radius_um == 1000.0
    assert MODELDB_NPOLE_ELECTRODE_ROI.distance_mode == "xyz"


def test_modeldb_default_full_scale_roi_counts_match_positionfcns() -> None:
    positions = modeldb_cell_positions(
        {
            "Axo": 1_470,
            "Bistratified": 2_210,
            "CCK_Basket": 3_600,
            "Ivy": 8_810,
            "Neurogliaform": 3_580,
            "O_LM": 1_640,
            "PV_Basket": 5_530,
            "Pyramidal": 311_500,
            "SCA": 400,
        },
    )

    roi_counts = {
        cell_type: int(electrode_roi_mask(pos, MODELDB_NPOLE_ELECTRODE_ROI).sum())
        for cell_type, pos in positions.items()
    }

    assert roi_counts == {
        "Axo": 397,
        "Bistratified": 601,
        "CCK_Basket": 956,
        "Ivy": 2_375,
        "Neurogliaform": 944,
        "O_LM": 443,
        "PV_Basket": 1_497,
        "Pyramidal": 89_678,
        "SCA": 108,
    }
