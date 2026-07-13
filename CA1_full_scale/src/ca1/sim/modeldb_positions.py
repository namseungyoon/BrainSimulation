"""ModelDB cell-position and electrode-ROI helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from ca1.types import ElectrodeRoi


@dataclass(frozen=True, slots=True)
class ModelDbGeometry:
    longitudinal_um: float = 4000.0
    transverse_um: float = 1000.0
    layer_heights_um: tuple[float, ...] = (4.0, 100.0, 50.0, 200.0, 100.0)


@dataclass(frozen=True, slots=True)
class _Placement:
    layerflag: int
    external: bool = False


MODELDB_NPOLE_ELECTRODE_ROI = ElectrodeRoi(
    center_um=(200.0, 100.0, 120.0),
    radius_um=1000.0,
    distance_mode="xyz",
)

_PLACEMENTS: dict[str, _Placement] = {
    "Axo": _Placement(layerflag=1),
    "Bistratified": _Placement(layerflag=1),
    "CCK_Basket": _Placement(layerflag=1),
    "Ivy": _Placement(layerflag=1),
    "Neurogliaform": _Placement(layerflag=3),
    "O_LM": _Placement(layerflag=0),
    "Pyramidal": _Placement(layerflag=1),
    "PV_Basket": _Placement(layerflag=1),
    "SCA": _Placement(layerflag=2),
    "CA3": _Placement(layerflag=2, external=True),
    "ECIII": _Placement(layerflag=3, external=True),
}


def modeldb_cell_positions(
    n_cells_per_type: dict[str, int],
    *,
    geometry: ModelDbGeometry | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    resolved_geometry = ModelDbGeometry() if geometry is None else geometry
    positions: dict[str, npt.NDArray[np.float64]] = {}
    for cell_type, count in n_cells_per_type.items():
        placement = _PLACEMENTS.get(cell_type)
        if placement is None or placement.external:
            continue
        positions[cell_type] = _positions_for_count(
            int(count),
            placement.layerflag,
            resolved_geometry,
        )
    return positions


def modeldb_connectivity_positions(
    n_cells_per_type: dict[str, int],
    *,
    geometry: ModelDbGeometry | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Return ModelDB grid positions for internal CA1 and external sources.

    Unlike :func:`modeldb_cell_positions`, this connectivity-specific entry
    point includes the CA3 and ECIII source populations using their ModelDB
    layer flags.  Keeping the external populations out of the default helper
    avoids adding non-recorded sources to result/ROI position metadata.
    """
    resolved_geometry = ModelDbGeometry() if geometry is None else geometry
    positions: dict[str, npt.NDArray[np.float64]] = {}
    for cell_type, count in n_cells_per_type.items():
        placement = _PLACEMENTS.get(cell_type)
        if placement is None:
            continue
        positions[cell_type] = _positions_for_count(
            int(count),
            placement.layerflag,
            resolved_geometry,
        )
    return positions


def electrode_roi_mask(
    positions_um: npt.NDArray[np.float64],
    roi: ElectrodeRoi,
) -> npt.NDArray[np.bool_]:
    positions = np.asarray(positions_um, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_um must have shape (n_cells, 3)")
    center = np.asarray(roi.center_um, dtype=np.float64)
    delta: npt.NDArray[np.float64] = positions[:, :2] - center[:2]
    if roi.distance_mode == "xyz":
        delta = positions - center
    elif roi.distance_mode != "xy":
        raise ValueError(f"unknown ROI distance mode: {roi.distance_mode}")
    distances = np.asarray(np.linalg.norm(delta, axis=1), dtype=np.float64)
    return distances < float(roi.radius_um)


def filter_spikes_to_roi(
    spikes: dict[str, list[npt.NDArray[np.float64]]],
    positions_um_by_type: dict[str, npt.NDArray[np.float64]],
    roi: ElectrodeRoi,
) -> dict[str, list[npt.NDArray[np.float64]]]:
    filtered: dict[str, list[npt.NDArray[np.float64]]] = {}
    for cell_type, trains in spikes.items():
        positions = positions_um_by_type.get(cell_type)
        if positions is None:
            filtered[cell_type] = trains
            continue
        mask = electrode_roi_mask(positions, roi)
        if mask.size != len(trains):
            message = (
                f"positions for {cell_type} have {mask.size} rows, but spikes "
                f"contain {len(trains)} cells"
            )
            raise ValueError(message)
        mask_values = cast(list[bool], mask.astype(bool).tolist())
        filtered[cell_type] = [
            train for train, keep in zip(trains, mask_values, strict=True) if keep
        ]
    return filtered


def _positions_for_count(
    count: int,
    layerflag: int,
    geometry: ModelDbGeometry,
) -> npt.NDArray[np.float64]:
    if count < 0:
        raise ValueError(f"cell count must be nonnegative, got {count}")
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    if layerflag >= len(geometry.layer_heights_um):
        raise ValueError(
            f"layerflag {layerflag} outside {len(geometry.layer_heights_um)} layers"
        )

    z_len = float(geometry.layer_heights_um[layerflag])
    y_len = float(geometry.transverse_um)
    x_len = float(geometry.longitudinal_um)
    x_bins, y_bins, z_bins = _set_bins(count, x_len, y_len, z_len)
    x_bin_size = int(x_len / x_bins)
    y_bin_size = int(y_len / y_bins)
    z_bin_size = int(z_len / z_bins)
    z_offset = float(sum(geometry.layer_heights_um[:layerflag]))

    cell_nums = np.arange(1, count + 1, dtype=np.float64)
    zero_based = cell_nums - 1.0
    x_tmp = np.floor(zero_based / (y_bins * z_bins))
    y_tmp = np.floor(zero_based / z_bins)
    x = np.mod(x_tmp, x_bins) * x_bin_size + x_bin_size / 2.0
    y = np.mod(y_tmp, y_bins) * y_bin_size + y_bin_size / 2.0
    z = np.mod(zero_based, z_bins) * z_bin_size + z_bin_size / 2.0 + z_offset
    return np.column_stack((x, y, z)).astype(np.float64)


def _set_bins(
    count: int,
    x_len: float,
    y_len: float,
    z_len: float,
) -> tuple[int, int, int]:
    base = math.pow(float(count) * z_len * z_len / (y_len * x_len), 1.0 / 3.0)
    z_bins = int(base)
    if z_bins == 0:
        z_bins = 1
    y_bins = int((y_len / z_len) * base)
    if y_bins == 0:
        y_bins = 1
    x_bins = int((x_len / z_len) * base)
    if x_bins == 0:
        x_bins = 1

    if z_len >= y_len and z_len >= x_len:
        axis = "z"
        num_to_min = x_bins * y_bins
    elif y_len >= z_len and y_len >= x_len:
        axis = "y"
        num_to_min = x_bins * z_bins
    else:
        axis = "x"
        num_to_min = y_bins * z_bins

    while x_bins * y_bins * z_bins < count:
        if axis == "x":
            x_bins += 1
        elif axis == "y":
            y_bins += 1
        else:
            z_bins += 1

    too_high = x_bins * y_bins * z_bins - num_to_min
    while too_high >= count:
        if axis == "x":
            x_bins -= 1
        elif axis == "y":
            y_bins -= 1
        else:
            z_bins -= 1
        too_high = x_bins * y_bins * z_bins - num_to_min

    return x_bins, y_bins, z_bins
