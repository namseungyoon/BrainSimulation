"""CA1 command-line interface.

Entry point is ``ca1`` (registered in pyproject.toml [project.scripts]).
All heavy imports are deferred inside subcommand handlers so the CLI is
importable even without NEST / NEST-GPU installed.

Subcommands
-----------
build     Build a scaled network from a YAML config -> HDF5 + manifest.json.
build-edges  Generate CPU-only persisted ModelDB 3-D Gaussian edges -> HDF5.
sim       Run a simulation -> SimResult persisted as HDF5.
validate  Score a persisted SimResult against Bezaire 2016 targets.
regen     Rebuild artifacts and verify SHA-256 checksums against manifest.json.
"""

from __future__ import annotations

import argparse
import os
import json
from numbers import Integral, Real
import shutil
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest_path(out_path: Path) -> Path:
    return out_path.parent / "manifest.json"


def _write_manifest(out_path: Path, entries: dict[str, str]) -> None:
    manifest = _manifest_path(out_path)
    existing: dict[str, str] = {}
    if manifest.exists():
        existing = json.loads(manifest.read_text())
    existing.update(entries)
    manifest.write_text(json.dumps(existing, indent=2))
    print(f"Manifest updated: {manifest}")


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _wait_for_path(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.1)


def _cli_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric, got {value!r}")
    if isinstance(value, str | Real):
        return float(value)
    raise TypeError(f"{field} must be numeric, got {value!r}")


def _cli_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be integral, got {value!r}")
    if isinstance(value, str | Integral):
        return int(value)
    raise TypeError(f"{field} must be integral, got {value!r}")


def _cli_str(value: object, field: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"{field} must be a string, got {value!r}")


def _read_provenance_json(raw: str | bytes, field: str) -> dict[str, str]:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    loaded: object = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{field} must be a JSON object")
    return {
        _cli_str(key, f"{field} key"): _cli_str(value, f"{field}.{key}")
        for key, value in loaded.items()
    }


def read_parameter_provenance_json(raw: str | bytes) -> dict[str, str]:
    return _read_provenance_json(raw, "parameter_provenance_json")


def read_diagnostic_provenance_json(raw: str | bytes) -> dict[str, str]:
    return _read_provenance_json(raw, "diagnostic_provenance_json")


# ---------------------------------------------------------------------------
# build subcommand
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    """Load YAML config, build network, write HDF5 + manifest."""
    from ca1.config import build_network_spec  # lazy import
    from ca1.build.builder import build_network  # lazy import

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else config_path.with_suffix(".hdf5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale = float(args.scale)
    seed = int(args.seed)

    print(f"Building network from {config_path} (scale={scale}, seed={seed}) ...")
    spec = build_network_spec(config_path, scale=scale, seed=seed)

    stats = build_network(spec, out_path=out_path, scale=scale)
    print(f"Network built: {stats}")
    print(f"HDF5 written: {out_path}")

    sha = _sha256(out_path)
    _write_manifest(out_path, {str(out_path): sha})
    return 0


def cmd_build_edges(args: argparse.Namespace) -> int:
    """Build a reusable 3-D edge artifact without importing the GPU backend."""
    from ca1.config import build_network_spec, load_config  # lazy, CPU-only
    from ca1.sim.edge_artifact import build_edge_artifact, default_artifact_path

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1
    config = load_config(config_path)
    scale = _cli_float(
        args.scale if args.scale is not None else config.get("scale", 1.0),
        "scale",
    )
    seed = _cli_int(
        args.seed if args.seed is not None else config.get("seed", 12345),
        "seed",
    )
    spec = build_network_spec(config_path, scale=scale, seed=seed)
    out_path = Path(args.output) if args.output else default_artifact_path(spec)
    print(
        f"Building CPU-only 3-D edges from {config_path} "
        f"(scale={scale}, seed={seed}, workers={args.workers or 'auto'}) ..."
    )
    started = time.perf_counter()
    try:
        stats = build_edge_artifact(spec, out_path, max_workers=args.workers)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    pool = "used" if stats.used_process_pool else "not needed at this scale"
    print(
        f"3-D edge artifact written: {stats.path} ({stats.edge_count} edges, "
        f"{stats.projection_count} projections, ProcessPool {pool}, {elapsed:.2f}s)"
    )
    print(f"edge_sha256={stats.digest}")
    print(
        "Set CA1_EDGE_ARTIFACT to this file (or its containing directory) "
        "for GPU simulations to load and validate it."
    )
    return 0


# ---------------------------------------------------------------------------
# sim subcommand
# ---------------------------------------------------------------------------

def cmd_sim(args: argparse.Namespace) -> int:
    """Run a simulation from a YAML config, persist spikes."""
    from ca1.analysis.location_transfer import (
        IncompatibleLocationTransferBudgetError,
        IncompleteLocationTransferError,
        UnvalidatedLocationTransferError,
    )
    from ca1.config import build_network_spec, load_config  # lazy import

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    backend_name = _cli_str(
        args.backend if args.backend is not None else config.get("backend", "gpu"),
        "backend",
    )
    scale = _cli_float(
        args.scale if args.scale is not None else config.get("scale", 1.0),
        "scale",
    )
    duration_s = _cli_float(
        args.duration if args.duration is not None else config.get("duration_s", 10.0),
        "duration_s",
    )
    dt_s = _cli_float(config.get("dt_s", 0.0001), "dt_s")
    seed = _cli_int(
        args.seed if args.seed is not None else config.get("seed", 12345),
        "seed",
    )
    crop_first_ms = _cli_float(config.get("crop_first_ms", 50.0), "crop_first_ms")

    print(f"Loading config {config_path} (backend={backend_name}, scale={scale}) ...")
    try:
        spec = build_network_spec(config_path, scale=scale, seed=seed)
    except (
        FileNotFoundError,
        IncompatibleLocationTransferBudgetError,
        IncompleteLocationTransferError,
        UnvalidatedLocationTransferError,
        ValueError,
    ) as exc:
        print(
            "Error: network spec is not final-buildable; simulation refused "
            "before backend execution.",
            file=sys.stderr,
        )
        print(f"  {exc}", file=sys.stderr)
        return 1
    from ca1.params.provenance import (
        diagnostic_config_provenance,
        diagnostic_environment_provenance,
        parameter_provenance_for_spec,
        stamp_clean_diagnostic_audit,
    )

    parameter_provenance = parameter_provenance_for_spec(spec)
    diagnostic_provenance = diagnostic_config_provenance(config)
    diagnostic_provenance.update(diagnostic_environment_provenance(os.environ))
    diagnostic_provenance = stamp_clean_diagnostic_audit(diagnostic_provenance)
    tier_raw = config.get("tier")
    tier = (
        _cli_str(tier_raw, "tier")
        if tier_raw is not None
        else ("full" if scale >= 0.999 else "scaled")
    )
    if tier not in {"scaled", "full"}:
        print("Error: tier must be 'scaled' or 'full'.", file=sys.stderr)
        return 1
    n_cells = spec.scaled_counts()
    if tier == "full":
        from ca1.validation.network_provenance import (
            final_tier_network_structure_blockers,
        )
        from ca1.validation.provenance import (
            final_tier_diagnostic_provenance_blockers,
            final_tier_parameter_provenance_blockers,
        )

        parameter_blockers = final_tier_parameter_provenance_blockers(
            parameter_provenance,
            n_cells,
        )
        diagnostic_blockers = final_tier_diagnostic_provenance_blockers(
            diagnostic_provenance,
        )
        structure_blockers = final_tier_network_structure_blockers(
            parameter_provenance,
            n_cells,
        )
        blockers = [
            *(f"parameter: {blocker}" for blocker in parameter_blockers),
            *(f"structure: {blocker}" for blocker in structure_blockers),
            *(f"diagnostic: {blocker}" for blocker in diagnostic_blockers),
        ]
        if blockers:
            print(
                "Error: full-tier provenance is not final-eligible; "
                "simulation refused before backend execution.",
                file=sys.stderr,
            )
            for blocker in blockers:
                print(f"  {blocker}", file=sys.stderr)
            return 1

    out_path = Path(args.output) if args.output else (
        config_path.parent / f"result_{config_path.stem}_{backend_name}.hdf5"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_run_dir: Path | None = None
    initial_gpu_rank = 0
    initial_gpu_size = 1

    # Select backend lazily
    if backend_name == "gpu":
        from ca1.sim.gpu_backend import (  # lazy import
            NestGpuBackend,
            _mpi_rank_size_from_env,
        )
        initial_gpu_rank, initial_gpu_size = _mpi_rank_size_from_env()
        if (
            initial_gpu_size > 1
            and os.environ.get("CA1_ALLOW_MPI_RECURRENT_SHARDING") != "1"
        ):
            print(
                "Error: multi-rank NEST-GPU recurrent sharding is disabled by "
                "default. Run canonical full-scale validation on one GPU, or set "
                "CA1_ALLOW_MPI_RECURRENT_SHARDING=1 for explicit benchmark-only "
                "MPI runs.",
                file=sys.stderr,
            )
            return 1
        done_marker = out_path.with_suffix(out_path.suffix + ".rank0_done")
        if initial_gpu_size > 1 and initial_gpu_rank == 0 and done_marker.exists():
            done_marker.unlink()
        if initial_gpu_size > 1 and "CA1_RUN_DIR" not in os.environ:
            gpu_run_dir = out_path.parent / f".{out_path.stem}_rank_spikes"
            if initial_gpu_rank == 0 and gpu_run_dir.exists():
                shutil.rmtree(gpu_run_dir)
            os.environ["CA1_RUN_DIR"] = str(gpu_run_dir)
        backend = NestGpuBackend()
    elif backend_name == "nest":
        from ca1.sim.nest_backend import NestBackend  # lazy import
        backend = NestBackend()
    else:
        print(f"Error: unknown backend '{backend_name}'. Choose nest or gpu.",
              file=sys.stderr)
        return 1

    from ca1.types import SimMeta  # already pure-Python, safe to import

    meta = SimMeta(
        duration_s=duration_s,
        dt_s=dt_s,
        n_cells_per_type=n_cells,
        scale=scale,
        seed=seed,
        backend=backend_name,
        config_name=config_path.stem,
        crop_first_ms=crop_first_ms,
        parameter_provenance=parameter_provenance,
        diagnostic_provenance=diagnostic_provenance,
    )

    print(f"Running {duration_s}s simulation with {spec.total_cells()} cells ...")
    result = backend.simulate(spec, meta)
    if spec.scale >= 0.999 and spec.cellnumbers_index == 101:
        from ca1.sim.modeldb_positions import (
            MODELDB_NPOLE_ELECTRODE_ROI,
            modeldb_cell_positions,
        )
        result.cell_positions_um = modeldb_cell_positions(dict(result.meta.n_cells_per_type))
        result.analysis_roi = MODELDB_NPOLE_ELECTRODE_ROI
    result_meta = result.meta
    print(f"Simulation done. Total spikes: {result.n_spikes()}")

    backend_rank = int(getattr(backend, "_mpi_rank", initial_gpu_rank))
    backend_size = int(getattr(backend, "_mpi_size", initial_gpu_size))
    if backend_name == "gpu" and backend_size > 1 and backend_rank != 0:
        done_marker = out_path.with_suffix(out_path.suffix + ".rank0_done")
        _wait_for_path(done_marker, float(os.environ.get("CA1_MPI_MERGE_TIMEOUT_S", "600")))
        print(f"MPI rank {backend_rank}/{backend_size} wrote raw spikes; rank 0 wrote HDF5.")
        return 0
    from ca1.validation.lfp_artifact import lfp_artifact_failures

    lfp_failures = lfp_artifact_failures(result.lfp, result.lfp_dt_s)
    if lfp_failures:
        print(
            "Error: runtime LFP artifact is malformed; simulation refused before "
            "HDF5 persistence.",
            file=sys.stderr,
        )
        for failure in lfp_failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    if tier == "full":
        from ca1.validation.runtime_artifacts import (
            final_tier_runtime_artifact_failures,
        )

        runtime_failures = final_tier_runtime_artifact_failures(result)
        if runtime_failures:
            print(
                "Error: full-tier runtime provenance is not final-eligible; "
                "simulation refused before HDF5 persistence.",
                file=sys.stderr,
            )
            for failure in runtime_failures:
                print(f"  {failure}", file=sys.stderr)
            return 1

    try:
        import h5py  # lazy import

        with h5py.File(out_path, "w") as f:
            meta_grp = f.create_group("meta")
            meta_grp.attrs["duration_s"] = result_meta.duration_s
            meta_grp.attrs["dt_s"] = result_meta.dt_s
            meta_grp.attrs["scale"] = result_meta.scale
            meta_grp.attrs["tier"] = tier
            meta_grp.attrs["seed"] = result_meta.seed
            meta_grp.attrs["backend"] = result_meta.backend
            meta_grp.attrs["config_name"] = result_meta.config_name
            meta_grp.attrs["crop_first_ms"] = result_meta.crop_first_ms
            meta_grp.attrs["lfp_proxy"] = result_meta.lfp_proxy
            if result.analysis_roi is not None:
                meta_grp.attrs["analysis_roi_center_um"] = result.analysis_roi.center_um
                meta_grp.attrs["analysis_roi_radius_um"] = result.analysis_roi.radius_um
                meta_grp.attrs[
                    "analysis_roi_distance_mode"
                ] = result.analysis_roi.distance_mode
            meta_grp.attrs["parameter_provenance_json"] = json.dumps(
                dict(result_meta.parameter_provenance),
                sort_keys=True,
            )
            meta_grp.attrs["diagnostic_provenance_json"] = json.dumps(
                dict(result_meta.diagnostic_provenance),
                sort_keys=True,
            )

            spikes_grp = f.create_group("spikes")
            for ct, cell_arrays in result.spikes.items():
                ct_grp = spikes_grp.create_group(ct)
                for idx, arr in enumerate(cell_arrays):
                    ct_grp.create_dataset(str(idx), data=arr)

            ncells_grp = f.create_group("n_cells_per_type")
            for ct, n in n_cells.items():
                ncells_grp.attrs[ct] = n

            if result.cell_positions_um is not None:
                positions_grp = f.create_group("cell_positions")
                for ct, positions in result.cell_positions_um.items():
                    positions_grp.create_dataset(ct, data=positions)

            if result.lfp is not None:
                f.create_dataset("lfp", data=result.lfp)
                f.attrs["lfp_dt_s"] = result.lfp_dt_s

        print(f"Result written: {out_path}")
        sha = _sha256(out_path)
        _write_manifest(out_path, {str(out_path): sha})
        if gpu_run_dir is not None and backend_rank == 0 and gpu_run_dir.exists():
            shutil.rmtree(gpu_run_dir)
        if backend_name == "gpu" and backend_size > 1 and backend_rank == 0:
            out_path.with_suffix(out_path.suffix + ".rank0_done").write_text("done\n")
    except ImportError as exc:
        print(
            f"Error: h5py not available; spikes could not be persisted: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


# ---------------------------------------------------------------------------
# validate subcommand
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    """Load a persisted SimResult and run the validation harness."""
    result_path = Path(args.result)
    if not result_path.exists():
        print(f"Error: result file not found: {result_path}", file=sys.stderr)
        return 1

    try:
        import h5py
        import numpy as np
        import numpy.typing as npt
    except ImportError:
        print("Error: h5py and numpy are required for validate.", file=sys.stderr)
        return 1

    from ca1.types import ElectrodeRoi, SimMeta, SimResult  # lazy but pure-Python

    stored_tier: str | None = None
    with h5py.File(result_path, "r") as f:
        meta_node = f["meta"]
        if not isinstance(meta_node, h5py.Group):
            raise TypeError("result meta must be an HDF5 group")
        m = meta_node
        raw_tier = m.attrs.get("tier")
        if isinstance(raw_tier, str | bytes):
            stored_tier = _cli_str(raw_tier, "tier")
        requested_tier = args.tier
        if requested_tier is None:
            if stored_tier is None:
                print(
                    "Error: tier metadata missing; cannot infer final-tier eligibility.",
                    file=sys.stderr,
                )
                return 1
            tier = stored_tier
        elif requested_tier == "full" and stored_tier != "full":
            if stored_tier is None:
                print(
                    "Error: tier metadata missing; cannot validate as full-tier.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Error: stored tier={stored_tier}; cannot validate as full-tier.",
                    file=sys.stderr,
                )
            return 1
        else:
            tier = requested_tier
        n_cells_node = f["n_cells_per_type"]
        if not isinstance(n_cells_node, h5py.Group):
            raise TypeError("result n_cells_per_type must be an HDF5 group")
        n_cells: dict[str, int] = {
            str(key): _cli_int(value, f"n_cells_per_type.{key}")
            for key, value in n_cells_node.attrs.items()
        }

        provenance_raw = m.attrs.get("parameter_provenance_json")
        parameter_provenance = (
            read_parameter_provenance_json(provenance_raw)
            if isinstance(provenance_raw, str | bytes)
            else {}
        )
        diagnostic_raw = m.attrs.get("diagnostic_provenance_json")
        diagnostic_provenance = (
            read_diagnostic_provenance_json(diagnostic_raw)
            if isinstance(diagnostic_raw, str | bytes)
            else {}
        )

        meta = SimMeta(
            duration_s=_cli_float(m.attrs["duration_s"], "duration_s"),
            dt_s=_cli_float(m.attrs["dt_s"], "dt_s"),
            n_cells_per_type=n_cells,
            scale=_cli_float(m.attrs["scale"], "scale"),
            seed=_cli_int(m.attrs["seed"], "seed"),
            backend=_cli_str(m.attrs["backend"], "backend"),
            config_name=_cli_str(m.attrs["config_name"], "config_name"),
            lfp_proxy=_cli_str(m.attrs.get("lfp_proxy", "unrecorded"), "lfp_proxy"),
            crop_first_ms=_cli_float(
                m.attrs.get("crop_first_ms", 50.0),
                "crop_first_ms",
            ),
            parameter_provenance=parameter_provenance,
            diagnostic_provenance=diagnostic_provenance,
        )
        roi: ElectrodeRoi | None = None
        roi_center = m.attrs.get("analysis_roi_center_um")
        roi_radius = m.attrs.get("analysis_roi_radius_um")
        roi_mode = m.attrs.get("analysis_roi_distance_mode")
        if roi_center is not None or roi_radius is not None or roi_mode is not None:
            if roi_center is None or roi_radius is None or roi_mode is None:
                raise TypeError("analysis ROI metadata is incomplete")
            center = np.asarray(roi_center, dtype=np.float64)
            if center.shape != (3,):
                raise TypeError("analysis_roi_center_um must contain 3 values")
            mode = _cli_str(roi_mode, "analysis_roi_distance_mode")
            if mode not in {"xyz", "xy"}:
                raise TypeError("analysis_roi_distance_mode must be 'xyz' or 'xy'")
            roi = ElectrodeRoi(
                center_um=(float(center[0]), float(center[1]), float(center[2])),
                radius_um=_cli_float(roi_radius, "analysis_roi_radius_um"),
                distance_mode="xyz" if mode == "xyz" else "xy",
            )

        spikes_node = f["spikes"]
        if not isinstance(spikes_node, h5py.Group):
            raise TypeError("result spikes must be an HDF5 group")
        spikes: dict[str, list[npt.NDArray[np.float64]]] = {}
        for ct in spikes_node:
            grp_node = spikes_node[ct]
            if not isinstance(grp_node, h5py.Group):
                raise TypeError(f"spikes.{ct} must be an HDF5 group")
            cell_arrays: list[npt.NDArray[np.float64]] = []
            for idx in range(len(grp_node)):
                dataset = grp_node[str(idx)]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError(f"spikes.{ct}.{idx} must be an HDF5 dataset")
                cell_arrays.append(np.asarray(dataset[()], dtype=np.float64))
            spikes[str(ct)] = cell_arrays

        lfp: npt.NDArray[np.float64] | None = None
        lfp_dt_s: float | None = None
        if "lfp" in f:
            lfp_node = f["lfp"]
            if not isinstance(lfp_node, h5py.Dataset):
                raise TypeError("result lfp must be an HDF5 dataset")
            if "lfp_dt_s" not in f.attrs:
                raise TypeError("lfp_dt_s metadata missing for stored lfp")
            lfp = np.asarray(lfp_node[()], dtype=np.float64)
            raw_lfp_dt_s = np.asarray(f.attrs["lfp_dt_s"], dtype=np.float64)
            if raw_lfp_dt_s.shape != ():
                raise TypeError("lfp_dt_s must be scalar")
            lfp_dt_s = float(raw_lfp_dt_s.item())
            from ca1.validation.lfp_artifact import require_valid_lfp_artifact

            require_valid_lfp_artifact(lfp, lfp_dt_s)
        cell_positions: dict[str, npt.NDArray[np.float64]] | None = None
        positions_node = f.get("cell_positions")
        if positions_node is not None:
            if not isinstance(positions_node, h5py.Group):
                raise TypeError("result cell_positions must be an HDF5 group")
            cell_positions = {}
            for ct in positions_node:
                dataset = positions_node[ct]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError(f"cell_positions.{ct} must be an HDF5 dataset")
                positions = np.asarray(dataset[()], dtype=np.float64)
                if positions.ndim != 2 or positions.shape[1] != 3:
                    raise TypeError(f"cell_positions.{ct} must have shape (n_cells, 3)")
                cell_positions[str(ct)] = positions

    result = SimResult(
        spikes=spikes,
        meta=meta,
        lfp=lfp,
        lfp_dt_s=lfp_dt_s,
        cell_positions_um=cell_positions,
        analysis_roi=roi,
    )

    from ca1.validation.harness import validate  # lazy import

    report = validate(result, tier=tier)
    print(report.summary())
    return 0 if report.passed else 2


# ---------------------------------------------------------------------------
# regen subcommand
# ---------------------------------------------------------------------------

def cmd_regen(args: argparse.Namespace) -> int:
    """Rebuild all artifacts and verify SHA-256 checksums against manifest."""
    from ca1.config import build_network_spec  # lazy import
    from ca1.build.builder import build_network  # lazy import

    configs_dir = Path(args.configs_dir)
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest: dict[str, str] = json.loads(manifest_path.read_text())
    errors: list[str] = []

    for config_file in sorted(configs_dir.glob("*.yaml")):
        print(f"Regenerating from {config_file} ...")
        spec = build_network_spec(config_file)
        out_path = config_file.with_suffix(".hdf5")

        try:
            build_network(spec, out_path=out_path)
        except Exception as exc:
            errors.append(f"{config_file}: build failed: {exc}")
            continue

        sha = _sha256(out_path)
        expected = manifest.get(str(out_path))
        if expected is None:
            print(f"  New artifact (not in manifest): {out_path}")
            manifest[str(out_path)] = sha
        elif sha != expected:
            errors.append(
                f"  CHECKSUM MISMATCH: {out_path}\n"
                f"    expected: {expected}\n"
                f"    got:      {sha}"
            )
        else:
            print(f"  OK: {out_path}")

    if errors:
        print("\nErrors:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    # Persist updated manifest
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("All artifacts verified and manifest updated.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``ca1`` CLI."""
    parser = argparse.ArgumentParser(
        prog="ca1",
        description="CA1 full-scale hippocampal model — build, simulate, validate.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ---- build ----
    p_build = sub.add_parser("build", help="Build network HDF5 from a YAML config.")
    p_build.add_argument("config", help="Path to YAML config file.")
    p_build.add_argument("-o", "--output", default=None,
                         help="Output HDF5 path (default: <config>.hdf5).")
    p_build.add_argument("--scale", default=1.0, type=float,
                         help="Network scale factor (default 1.0).")
    p_build.add_argument("--seed", default=12345, type=int,
                         help="Random seed (default 12345).")

    # ---- build-edges ----
    p_edges = sub.add_parser(
        "build-edges",
        help="CPU-only persisted ModelDB 3-D Gaussian edge graph.",
    )
    p_edges.add_argument("config", help="Path to YAML config file.")
    p_edges.add_argument(
        "-o", "--output", default=None,
        help="Artifact HDF5 path (default: edge_artifacts/<topology-key>.h5).",
    )
    p_edges.add_argument("--scale", default=None, type=float,
                         help="Network scale factor (default: config scale).")
    p_edges.add_argument("--seed", default=None, type=int,
                         help="Random seed (default: config seed).")
    p_edges.add_argument("--workers", default=None, type=int,
                         help="Topology ProcessPool workers (default: CPU count).")

    # ---- sim ----
    p_sim = sub.add_parser("sim", help="Run a simulation from a YAML config.")
    p_sim.add_argument("config", help="Path to YAML config file.")
    p_sim.add_argument("--backend", choices=["nest", "gpu"], default=None,
                       help="Simulator backend (default: config backend or gpu).")
    p_sim.add_argument("--scale", default=None, type=float,
                       help="Network scale factor (default: config scale or 1.0).")
    p_sim.add_argument("--duration", default=None, type=float,
                       help="Simulation duration in seconds (default: config duration_s or 10.0).")
    p_sim.add_argument("--seed", default=None, type=int,
                       help="Random seed (default: config seed or 12345).")
    p_sim.add_argument("-o", "--output", default=None,
                       help="Output HDF5 path for persisted result.")

    # ---- validate ----
    p_val = sub.add_parser("validate", help="Validate a persisted SimResult.")
    p_val.add_argument("result", help="Path to HDF5 result file.")
    p_val.add_argument("--tier", default=None, choices=["scaled", "full"],
                       help="Override validation tier (default: stored HDF tier).")

    # ---- regen ----
    p_regen = sub.add_parser("regen", help="Rebuild artifacts and verify checksums.")
    p_regen.add_argument("--configs-dir", default="configs",
                         help="Directory containing YAML configs (default: configs/).")
    p_regen.add_argument("--manifest", default="configs/manifest.json",
                         help="Path to manifest.json (default: configs/manifest.json).")

    args = parser.parse_args()

    handlers = {
        "build": cmd_build,
        "build-edges": cmd_build_edges,
        "sim": cmd_sim,
        "validate": cmd_validate,
        "regen": cmd_regen,
    }
    rc = handlers[args.subcommand](args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
