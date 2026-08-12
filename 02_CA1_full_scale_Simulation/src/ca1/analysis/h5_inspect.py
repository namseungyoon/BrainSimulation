"""BSB-HDF5 network file inspector.

Provides ``inspect_network(path) -> dict`` for programmatic use and a
``__main__`` CLI entry point for interactive exploration.

BSB-HDF5 schema (as written by bsb >= 4.x)
-------------------------------------------
Root attributes: may contain 'version', 'config_string', etc.

/placement/
    <cell_type>/          (one group per cell type)
        <partition_id>/   (integer labels, usually "0", "1", ...)
            position      Dataset  shape (N, 3)  float64  [µm]
            rotation      Dataset  shape (N, 4)  float64  quaternion (optional)
            identifiers   Dataset  shape (N,)    int64    global cell IDs

/connectivity/
    <pre>_to_<post>/      (one group per connection type)
        inc/              (incoming connections for post cells)
            <partition>/
                global_locs   Dataset  shape (M, 2) int64  [pre_id, post_id]
                local_locs    Dataset  shape (M, 2) int64  local indices
        out/              (outgoing -- symmetric view, may be absent)
            ...

/morphologies/            (optional, absent in point-neuron models)

Notes
-----
* ``indegree_true`` and ``weight_nS`` are not stored in the HDF5 file; they
  live in ``ca1/params/connectivity.json``.  This module reads structural
  counts only.
* All paths are resolved via ``h5py`` so the caller never touches raw bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _format_size(n_bytes: int) -> str:
    if n_bytes == 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    i = min(int(np.floor(np.log(n_bytes) / np.log(1024))), len(units) - 1)
    value = n_bytes / (1024 ** i)
    return f"{value:.2f} {units[i]}"


def _walk(group: h5py.Group, max_depth: int, depth: int = 0) -> list[dict[str, Any]]:
    """Recursively collect metadata for every item in a group."""
    items: list[dict[str, Any]] = []
    if depth >= max_depth:
        return items
    for key in group.keys():
        item = group[key]
        entry: dict[str, Any] = {"path": item.name, "depth": depth}
        if isinstance(item, h5py.Dataset):
            entry["kind"] = "Dataset"
            entry["shape"] = tuple(item.shape)
            entry["dtype"] = str(item.dtype)
            entry["size_bytes"] = item.nbytes
            entry["attrs"] = dict(item.attrs)
        else:
            entry["kind"] = "Group"
            entry["n_children"] = len(item.keys())
            entry["attrs"] = dict(item.attrs)
            items.append(entry)
            items.extend(_walk(item, max_depth, depth + 1))
            continue
        items.append(entry)
    return items


# ---------------------------------------------------------------------------
# placement analysis
# ---------------------------------------------------------------------------

def _inspect_placement(placement: h5py.Group) -> dict[str, Any]:
    """Return per-cell-type counts and spatial extents."""
    result: dict[str, Any] = {}
    for cell_type in placement.keys():
        grp = placement[cell_type]
        total = 0
        all_pos: list[np.ndarray] = []
        for part_id in grp.keys():
            part = grp[part_id]
            if "position" in part:
                pos: np.ndarray = part["position"][:]
                total += pos.shape[0]
                if pos.ndim == 2 and pos.shape[1] == 3:
                    all_pos.append(pos)
        entry: dict[str, Any] = {"n_cells": total}
        if all_pos:
            combined = np.vstack(all_pos)
            entry["x_range"] = (float(combined[:, 0].min()), float(combined[:, 0].max()))
            entry["y_range"] = (float(combined[:, 1].min()), float(combined[:, 1].max()))
            entry["z_range"] = (float(combined[:, 2].min()), float(combined[:, 2].max()))
        result[cell_type] = entry
    return result


# ---------------------------------------------------------------------------
# connectivity analysis
# ---------------------------------------------------------------------------

def _inspect_connectivity(connectivity: h5py.Group) -> dict[str, Any]:
    """Return per-connection-type approximate synapse counts."""
    result: dict[str, Any] = {}
    for conn_type in connectivity.keys():
        grp = connectivity[conn_type]
        n_synapses = 0
        # Try both 'inc' and direct datasets
        for direction in ("inc", "out"):
            if direction not in grp:
                continue
            dir_grp = grp[direction]
            for part in dir_grp.keys():
                part_grp = dir_grp[part]
                for ds_name in ("global_locs", "local_locs", "connections"):
                    if ds_name in part_grp:
                        n_synapses += part_grp[ds_name].shape[0]
                        break
            break  # count only 'inc' to avoid double-counting
        result[conn_type] = {"n_synapses_approx": n_synapses}
    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def inspect_network(path: str | Path) -> dict[str, Any]:
    """Inspect a BSB-HDF5 network file and return a structured summary dict.

    Parameters
    ----------
    path:
        Path to the ``.hdf5`` / ``.h5`` file.

    Returns
    -------
    dict with keys:
        ``file_path``, ``file_size_bytes``, ``root_attrs``,
        ``placement`` (per-type dicts with n_cells + spatial extents),
        ``connectivity`` (per-projection n_synapses_approx),
        ``tree`` (flat list of all items up to depth 4).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    file_size = path.stat().st_size
    result: dict[str, Any] = {
        "file_path": str(path.resolve()),
        "file_size_bytes": file_size,
        "placement": {},
        "connectivity": {},
        "tree": [],
    }

    with h5py.File(path, "r") as f:
        result["root_keys"] = list(f.keys())
        result["root_attrs"] = {k: _coerce_attr(v) for k, v in f.attrs.items()}
        result["tree"] = _walk(f, max_depth=4)
        if "placement" in f:
            result["placement"] = _inspect_placement(f["placement"])
        if "connectivity" in f:
            result["connectivity"] = _inspect_connectivity(f["connectivity"])

    return result


def _coerce_attr(v: Any) -> Any:
    """Convert h5py attribute values to plain Python types."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(info: dict[str, Any]) -> None:
    p = Path(info["file_path"])
    print(f"\n{'=' * 60}")
    print(f"BSB Network File: {p.name}")
    print(f"{'=' * 60}")
    print(f"Size : {_format_size(info['file_size_bytes'])}")
    print(f"Keys : {info['root_keys']}")
    if info["root_attrs"]:
        print(f"Attrs: {info['root_attrs']}")

    print(f"\n--- Placement ({len(info['placement'])} cell types) ---")
    total_cells = 0
    for ct, data in sorted(info["placement"].items()):
        n = data["n_cells"]
        total_cells += n
        spatial = ""
        if "x_range" in data:
            x = data["x_range"]
            y = data["y_range"]
            z = data["z_range"]
            spatial = (f"  [{x[0]:.0f}..{x[1]:.0f}]"
                       f" x [{y[0]:.0f}..{y[1]:.0f}]"
                       f" x [{z[0]:.0f}..{z[1]:.0f}] µm")
        print(f"  {ct:<25} {n:>8,} cells{spatial}")
    print(f"  {'TOTAL':<25} {total_cells:>8,} cells")

    print(f"\n--- Connectivity ({len(info['connectivity'])} projection types) ---")
    for ct, data in sorted(info["connectivity"].items()):
        n = data["n_synapses_approx"]
        print(f"  {ct:<35} ~{n:>10,} synapses")

    print("\n--- Full tree (depth <= 4, first 60 items) ---")
    header = f"{'Path':<50} {'Type':<10} {'Shape/Children':<20} {'DType':<12}"
    print(header)
    print("-" * len(header))
    for item in info["tree"][:60]:
        kind = item["kind"]
        if kind == "Dataset":
            shape_s = str(item["shape"])
            dtype_s = item["dtype"]
            size_s = _format_size(item["size_bytes"])
        else:
            shape_s = f"({item['n_children']} items)"
            dtype_s = "-"
            size_s = ""
        print(f"  {item['path']:<48} {kind:<10} {shape_s:<20} {dtype_s:<12} {size_s}")

    if len(info["tree"]) > 60:
        print(f"  ... {len(info['tree'])} total items, showing first 60")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m ca1.analysis.h5_inspect <path>``."""
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m ca1.analysis.h5_inspect <hdf5_path>")
        print()
        print("Examples:")
        print("  python -m ca1.analysis.h5_inspect ca1_complete_network.hdf5")
        print("  python -m ca1.analysis.h5_inspect ca1_scaled_1_50.hdf5")
        sys.exit(0)

    h5_path = Path(args[0])
    try:
        info = inspect_network(h5_path)
        _print_summary(info)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error inspecting {h5_path}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
