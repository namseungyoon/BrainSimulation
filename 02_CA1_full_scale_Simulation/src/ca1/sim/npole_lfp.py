from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

from ca1.sim.modeldb_positions import electrode_roi_mask
from ca1.types import ElectrodeRoi

MODELDB_NPOLE_RHO_OHM_CM: Final = 333.0
MODELDB_NPOLE_POINT_SOURCE_SCALE: Final = 0.0001
_MIN_DISTANCE_UM: Final = 1e-9


def reduced_domain_n_pole_weights(
    positions_um: npt.NDArray[np.float64],
    roi: ElectrodeRoi,
    *,
    rho_ohm_cm: float = MODELDB_NPOLE_RHO_OHM_CM,
) -> npt.NDArray[np.float64]:
    positions = np.asarray(positions_um, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_um must have shape (n_cells, 3)")
    if not bool(np.isfinite(positions).all()):
        raise ValueError("positions_um must contain only finite values")
    center = np.asarray(roi.center_um, dtype=np.float64)
    delta = positions[:, :2] - center[:2]
    if roi.distance_mode == "xyz":
        delta = positions - center
    elif roi.distance_mode != "xy":
        raise ValueError(f"unknown ROI distance mode: {roi.distance_mode}")
    norm_um = np.asarray(np.linalg.norm(delta, axis=1), dtype=np.float64)
    distances_um = np.maximum(norm_um, _MIN_DISTANCE_UM)
    selected = electrode_roi_mask(positions, roi)
    weights = np.zeros(positions.shape[0], dtype=np.float64)
    weights[selected] = (
        MODELDB_NPOLE_POINT_SOURCE_SCALE
        * float(rho_ohm_cm)
        / (4.0 * math.pi * distances_um[selected])
    )
    return weights


def reduced_domain_n_pole_lfp(
    currents: npt.NDArray[np.float64],
    positions_um: npt.NDArray[np.float64],
    roi: ElectrodeRoi,
) -> npt.NDArray[np.float64]:
    current_matrix = np.asarray(currents, dtype=np.float64)
    if current_matrix.ndim != 2:
        raise ValueError("currents must have shape (n_samples, n_cells)")
    if not bool(np.isfinite(current_matrix).all()):
        raise ValueError("currents must contain only finite values")
    weights = reduced_domain_n_pole_weights(positions_um, roi)
    if current_matrix.shape[1] != weights.shape[0]:
        message = " ".join(
            (
                "currents cell axis does not match positions:",
                f"{current_matrix.shape[1]} != {weights.shape[0]}",
            )
        )
        raise ValueError(message)
    if not bool(np.any(weights > 0.0)):
        raise ValueError("no sampled Pyramidal cells inside ModelDB N-pole ROI")
    return -np.asarray(current_matrix @ weights, dtype=np.float64)
