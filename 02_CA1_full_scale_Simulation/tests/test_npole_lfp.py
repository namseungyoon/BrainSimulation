from __future__ import annotations

import numpy as np
import pytest

from ca1.sim.npole_lfp import (
    reduced_domain_n_pole_lfp,
    reduced_domain_n_pole_weights,
)
from ca1.types import ElectrodeRoi


def test_reduced_domain_n_pole_lfp_uses_roi_weighted_sink_currents() -> None:
    # Given: positive sampled currents represent inward transmembrane sinks.
    currents = np.asarray(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ],
        dtype=np.float64,
    )
    positions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    roi = ElectrodeRoi(center_um=(0.0, 0.0, 0.0), radius_um=10.0)

    # When: the reduced-domain N-pole signal is computed.
    weights = reduced_domain_n_pole_weights(positions, roi)
    lfp = reduced_domain_n_pole_lfp(currents, positions, roi)

    # Then: the outside-ROI cell contributes zero and inward current is negative LFP.
    assert weights[2] == 0.0
    np.testing.assert_allclose(lfp, -(currents @ weights))
    mean_current = np.asarray(currents.mean(axis=1), dtype=np.float64)
    assert not np.allclose(lfp, mean_current)


def test_reduced_domain_n_pole_lfp_rejects_empty_roi() -> None:
    # Given: sampled Pyramidal positions all outside the electrode ROI.
    currents = np.asarray([[10.0, 20.0]], dtype=np.float64)
    positions = np.asarray([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=np.float64)
    roi = ElectrodeRoi(center_um=(0.0, 0.0, 0.0), radius_um=10.0)

    # When / Then: the final LFP path fails loudly instead of falling back to a mean.
    with pytest.raises(ValueError, match="no sampled Pyramidal cells inside"):
        _ = reduced_domain_n_pole_lfp(currents, positions, roi)
