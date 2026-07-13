from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TypedDict

import h5py
import numpy as np
import numpy.typing as npt

from ca1.analysis.hdf_artifact_summary import final_tier_spike_artifact_failures
from ca1.analysis.hdf_provenance_summary import (
    HdfProvenanceSummary,
    summarize_hdf_provenance,
)
from ca1.sim.modeldb_positions import electrode_roi_mask
from ca1.types import ElectrodeRoi
from ca1.validation.targets import MODEL_RATES_HZ, RATE_REL_TOL


class HdfRateSummaryError(ValueError): ...


@dataclass(frozen=True, slots=True)
class HdfMeta:
    duration_s: float
    scale: float | None
    tier: str | None
    lfp_proxy: str | None
    crop_first_ms: float
    parameter_provenance: dict[str, str]
    diagnostic_provenance: dict[str, str]
    parameter_provenance_missing: bool
    diagnostic_provenance_missing: bool
    analysis_roi: ElectrodeRoi | None

    @property
    def analysis_window_s(self) -> float:
        window_s = self.duration_s - self.crop_first_ms * 1.0e-3
        if window_s <= 0.0:
            raise HdfRateSummaryError(
                "analysis window must be positive after subtracting crop_first_ms"
            )
        return window_s


@dataclass(frozen=True, slots=True)
class CellCounts:
    n_cells: int
    spike_datasets: int
    active_cells: int
    total_spikes: int
    validation_scope: str = "all_cells"
    validation_n_cells: int | None = None
    validation_total_spikes: int | None = None
    roi_cells: int | None = None
    roi_total_spikes: int | None = None


class HdfRateSummary(TypedDict):
    path: str
    analysis_window_s: float
    duration_s: float
    crop_first_ms: float
    cell_types: dict[str, dict[str, object]]
    provenance: HdfProvenanceSummary


def summarize_hdf_rates(path: str | Path) -> HdfRateSummary:
    result_path = Path(path)
    with h5py.File(result_path, "r") as h5:
        meta = _read_meta(_require_group(h5, "meta"))
        n_cells = _read_n_cells(_require_group(h5, "n_cells_per_type"))
        positions = _read_cell_positions(h5)
        counts = _read_spike_counts(
            _require_group(h5, "spikes"),
            n_cells,
            positions,
            meta.analysis_roi,
        )
        has_lfp = _has_lfp_dataset(h5)

        window_s = meta.analysis_window_s
        return {
            "path": str(result_path),
            "analysis_window_s": window_s,
            "duration_s": meta.duration_s,
            "crop_first_ms": meta.crop_first_ms,
            "cell_types": {
                cell_type: _summarize_cell_type(cell_type, cell_counts, window_s)
                for cell_type, cell_counts in sorted(counts.items())
            },
            "provenance": summarize_hdf_provenance(
                parameter_provenance=meta.parameter_provenance,
                diagnostic_provenance=meta.diagnostic_provenance,
                n_cells_per_type=n_cells,
                parameter_provenance_missing=meta.parameter_provenance_missing,
                diagnostic_provenance_missing=meta.diagnostic_provenance_missing,
                tier=meta.tier,
                scale=meta.scale,
                lfp_proxy=meta.lfp_proxy,
                has_lfp=has_lfp,
                has_n_pole_lfp_context=_has_n_pole_lfp_context(h5, meta),
                artifact_failures=final_tier_spike_artifact_failures(
                    n_cells,
                    {
                        cell_type: cell_counts.spike_datasets
                        for cell_type, cell_counts in counts.items()
                    },
                ),
            ),
        }


def _read_meta(meta: h5py.Group) -> HdfMeta:
    parameter, parameter_missing = _read_provenance(meta, "parameter_provenance_json")
    diagnostic, diagnostic_missing = _read_provenance(
        meta,
        "diagnostic_provenance_json",
    )
    return HdfMeta(
        duration_s=_float_attr(meta, "duration_s"),
        scale=_optional_float_attr(meta, "scale"),
        tier=_optional_text_attr(meta, "tier"),
        lfp_proxy=_optional_text_attr(meta, "lfp_proxy"),
        crop_first_ms=_float_attr(meta, "crop_first_ms"),
        parameter_provenance=parameter,
        diagnostic_provenance=diagnostic,
        parameter_provenance_missing=parameter_missing,
        diagnostic_provenance_missing=diagnostic_missing,
        analysis_roi=_read_analysis_roi(meta),
    )


def _read_n_cells(group: h5py.Group) -> dict[str, int]:
    return {
        str(name): _int_scalar(_hdf_scalar(group, str(name)), f"{group.name}.{name}")
        for name in group.attrs
    }


def _read_analysis_roi(meta: h5py.Group) -> ElectrodeRoi | None:
    center_raw = meta.attrs.get("analysis_roi_center_um")
    radius_raw = meta.attrs.get("analysis_roi_radius_um")
    mode_raw = meta.attrs.get("analysis_roi_distance_mode")
    if center_raw is None and radius_raw is None and mode_raw is None:
        return None
    if center_raw is None or radius_raw is None or mode_raw is None:
        raise HdfRateSummaryError(
            "analysis ROI metadata requires center, radius, and distance_mode"
        )
    center_arr = np.asarray(center_raw, dtype=np.float64)
    if center_arr.shape != (3,):
        raise HdfRateSummaryError("analysis_roi_center_um must contain 3 values")
    mode = _text_scalar(mode_raw, "analysis_roi_distance_mode")
    if mode not in {"xyz", "xy"}:
        raise HdfRateSummaryError("analysis_roi_distance_mode must be 'xyz' or 'xy'")
    distance_mode = "xyz" if mode == "xyz" else "xy"
    return ElectrodeRoi(
        center_um=(
            float(center_arr[0]),
            float(center_arr[1]),
            float(center_arr[2]),
        ),
        radius_um=_float_scalar(radius_raw, "analysis_roi_radius_um"),
        distance_mode=distance_mode,
    )


def _read_cell_positions(
    parent: h5py.File,
) -> dict[str, npt.NDArray[np.float64]]:
    node = parent.get("cell_positions")
    if node is None:
        return {}
    if not isinstance(node, h5py.Group):
        raise HdfRateSummaryError("cell_positions must be an HDF5 group")
    positions: dict[str, npt.NDArray[np.float64]] = {}
    for cell_type in node:
        dataset = node[cell_type]
        if not isinstance(dataset, h5py.Dataset):
            raise HdfRateSummaryError(f"cell_positions.{cell_type} must be a dataset")
        array = np.asarray(dataset[()], dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise HdfRateSummaryError(
                f"cell_positions.{cell_type} must have shape (n_cells, 3)"
            )
        positions[str(cell_type)] = array
    return positions


def _read_spike_counts(
    spikes: h5py.Group,
    n_cells: Mapping[str, int],
    positions_um_by_type: Mapping[str, npt.NDArray[np.float64]],
    analysis_roi: ElectrodeRoi | None,
) -> dict[str, CellCounts]:
    counts: dict[str, CellCounts] = {}
    for cell_type in sorted(set(n_cells) | set(spikes.keys())):
        declared = int(n_cells.get(cell_type, 0))
        group = spikes.get(cell_type)
        if group is None:
            counts[cell_type] = CellCounts(declared, 0, 0, 0)
            continue
        if not isinstance(group, h5py.Group):
            raise HdfRateSummaryError(f"spikes.{cell_type} must be an HDF5 group")
        active = 0
        total = 0
        datasets = 0
        per_cell_counts: dict[int, int] = {}
        for cell_index in group:
            dataset = group[cell_index]
            if not isinstance(dataset, h5py.Dataset):
                path = f"spikes.{cell_type}.{cell_index}"
                raise HdfRateSummaryError(f"{path} must be a dataset")
            spike_count = _spike_count(dataset, f"spikes.{cell_type}.{cell_index}")
            total += spike_count
            datasets += 1
            active += int(spike_count > 0)
            per_cell_counts[int(str(cell_index))] = spike_count
        roi_cells = None
        roi_total = None
        validation_scope = "all_cells"
        validation_n_cells = None
        validation_total = None
        if analysis_roi is not None and cell_type in positions_um_by_type:
            mask = electrode_roi_mask(positions_um_by_type[cell_type], analysis_roi)
            roi_cells = int(mask.sum())
            roi_total = sum(
                count
                for index, count in per_cell_counts.items()
                if index < mask.size and bool(mask[index])
            )
            validation_scope = "electrode_roi"
            validation_n_cells = roi_cells
            validation_total = roi_total
        counts[cell_type] = CellCounts(
            declared if declared > 0 else datasets,
            datasets,
            active,
            total,
            validation_scope=validation_scope,
            validation_n_cells=validation_n_cells,
            validation_total_spikes=validation_total,
            roi_cells=roi_cells,
            roi_total_spikes=roi_total,
        )
    return counts


def _summarize_cell_type(
    cell_type: str,
    counts: CellCounts,
    window_s: float,
) -> dict[str, object]:
    mean_rate = (
        counts.total_spikes / counts.n_cells / window_s if counts.n_cells > 0 else 0.0
    )
    active_mean_rate = (
        counts.total_spikes / counts.active_cells / window_s
        if counts.active_cells > 0
        else 0.0
    )
    validation_n_cells = (
        counts.validation_n_cells
        if counts.validation_n_cells is not None
        else counts.n_cells
    )
    validation_total_spikes = (
        counts.validation_total_spikes
        if counts.validation_total_spikes is not None
        else counts.total_spikes
    )
    validation_mean_rate = (
        validation_total_spikes / validation_n_cells / window_s
        if validation_n_cells > 0
        else 0.0
    )
    target = MODEL_RATES_HZ.get(cell_type)
    band = None if target is None else _target_band(target)
    return {
        "n_cells": counts.n_cells,
        "spike_datasets": counts.spike_datasets,
        "active_cells": counts.active_cells,
        "total_spikes": counts.total_spikes,
        "raw_spike_count": counts.total_spikes,
        "cropped_as_stored_spike_count": counts.total_spikes,
        "mean_rate_hz": mean_rate,
        "active_mean_rate_hz": active_mean_rate,
        "validation_scope": counts.validation_scope,
        "validation_n_cells": validation_n_cells,
        "validation_total_spikes": validation_total_spikes,
        "validation_mean_rate_hz": validation_mean_rate,
        "roi_cells": counts.roi_cells,
        "roi_total_spikes": counts.roi_total_spikes,
        "target_hz": target,
        "target_band_hz": band,
        "target_pass": (
            None if band is None else band[0] <= validation_mean_rate <= band[1]
        ),
    }


def _target_band(target_hz: float) -> list[float]:
    return [target_hz * (1.0 - RATE_REL_TOL), target_hz * (1.0 + RATE_REL_TOL)]


def _read_provenance(group: h5py.Group, attr_name: str) -> tuple[dict[str, str], bool]:
    raw = group.attrs.get(attr_name)
    if raw is None:
        return {}, True
    parsed = json.loads(_text_scalar(raw, attr_name))
    if not isinstance(parsed, dict):
        raise HdfRateSummaryError(f"{attr_name} must encode a JSON object")
    records: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise HdfRateSummaryError(f"{attr_name} must contain string keys and values")
        records[key] = value
    return records, False


def _float_attr(group: h5py.Group, attr_name: str) -> float:
    value = _float_scalar(_hdf_scalar(group, attr_name), f"{group.name}.{attr_name}")
    if value < 0.0:
        raise HdfRateSummaryError(f"{group.name}.{attr_name} must be non-negative")
    return value


def _optional_float_attr(group: h5py.Group, attr_name: str) -> float | None:
    value = group.attrs.get(attr_name)
    if value is None:
        return None
    numeric = _float_scalar(value, f"{group.name}.{attr_name}")
    if numeric < 0.0:
        raise HdfRateSummaryError(f"{group.name}.{attr_name} must be non-negative")
    return numeric


def _optional_text_attr(group: h5py.Group, attr_name: str) -> str | None:
    value = group.attrs.get(attr_name)
    return None if value is None else _text_scalar(value, f"{group.name}.{attr_name}")


def _spike_count(dataset: h5py.Dataset, path: str) -> int:
    if dataset.ndim != 1:
        raise HdfRateSummaryError(f"{path} must be 1-D")
    return int(dataset.shape[0])


def _require_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    node = parent.get(name)
    if not isinstance(node, h5py.Group):
        raise HdfRateSummaryError(f"{name} must be an HDF5 group")
    return node


def _has_lfp_dataset(parent: h5py.File) -> bool:
    node = parent.get("lfp")
    if node is None:
        return False
    if not isinstance(node, h5py.Dataset):
        raise HdfRateSummaryError("lfp must be an HDF5 dataset")
    return True


def _has_n_pole_lfp_context(parent: h5py.File, meta: HdfMeta) -> bool:
    if meta.analysis_roi is None:
        return False
    positions = parent.get("cell_positions")
    if not isinstance(positions, h5py.Group):
        return False
    return isinstance(positions.get("Pyramidal"), h5py.Dataset)


def _hdf_scalar(group: h5py.Group, attr_name: str) -> str | bytes | Real:
    if attr_name not in group.attrs:
        raise HdfRateSummaryError(f"{group.name}.{attr_name} missing")
    value = group.attrs[attr_name]
    if isinstance(value, str | bytes | Real):
        return value
    raise HdfRateSummaryError(f"{group.name}.{attr_name} must be scalar")


def _text_scalar(value: str | bytes | Real, attr_name: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise HdfRateSummaryError(f"{attr_name} must be a string")


def _float_scalar(value: str | bytes | Real, attr_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise HdfRateSummaryError(f"{attr_name} must be numeric") from exc


def _int_scalar(value: str | bytes | Real, attr_name: str) -> int:
    try:
        if isinstance(value, str | bytes):
            return int(value)
        numeric = float(value)
        if not numeric.is_integer():
            raise HdfRateSummaryError(f"{attr_name} must be integral")
        return int(numeric)
    except ValueError as exc:
        raise HdfRateSummaryError(f"{attr_name} must be integral") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    summary = summarize_hdf_rates(args.result)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    for cell_type, cell_summary in summary["cell_types"].items():
        print(f"{cell_type}: {cell_summary['mean_rate_hz']:.3f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
