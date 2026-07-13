from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Literal

import h5py
import numpy as np

from ca1 import cli
from ca1.types import Afferent, CellType, NeuronParams, Projection, SimMeta, SimResult


class _FakeSpec:
    scale: float = 0.001
    neuron_model: Literal["aeif_cond_beta_multisynapse"] = (
        "aeif_cond_beta_multisynapse"
    )
    receptor_provenance: str = ""
    calibration_provenance: ClassVar[dict[str, str]] = {
        "calibration.mode": "paper_reduction"
    }
    aglif_receive_domain_overrides: ClassVar[dict[str, str]] = {}
    aglif_gc_scale_overrides: ClassVar[dict[str, float]] = {}
    aglif_dend_overrides: ClassVar[dict[str, object]] = {}
    source_location_transfer_provenance: str = ""
    afferent_topology: Literal["compound"] = "compound"
    recurrent_topology: Literal["fixed_indegree"] = "fixed_indegree"
    afferent_source_pool_size: int = 4096
    afferent_source_pool_indegree: int = 64
    afferent_source_rate_cv: float = 0.0
    conndata_index: int | None = None
    cellnumbers_index: int = 101
    conndata_count_mode: Literal["network_total"] = "network_total"
    projections: ClassVar[list[Projection]] = []
    afferents: ClassVar[list[Afferent]] = []
    cell_types: ClassVar[dict[str, CellType]] = {
        "Pyramidal": CellType(
            name="Pyramidal",
            count=1,
            layers=("SP",),
            params=NeuronParams(
                C_m=100.0,
                g_L=5.0,
                E_L=-65.0,
                V_th=-50.0,
                V_reset=-60.0,
                Delta_T=2.0,
                a=0.0,
                b=0.0,
                tau_w=100.0,
                t_ref=2.0,
                fit_provenance="analytic-fallback-after-failed-fit",
            ),
        )
    }

    def scaled_counts(self) -> dict[str, int]:
        return {"Pyramidal": 1}

    def total_cells(self) -> int:
        return 1


class _FakeNestBackend:
    captured_meta: SimMeta | None = None

    def simulate(self, _spec: _FakeSpec, meta: SimMeta) -> SimResult:
        type(self).captured_meta = meta
        return SimResult(spikes={"Pyramidal": [np.array([], dtype=float)]}, meta=meta)


class _FakeGpuBackend:
    _mpi_rank = 1
    _mpi_size = 3
    captured_meta: SimMeta | None = None

    def simulate(self, _spec: _FakeSpec, meta: SimMeta) -> SimResult:
        type(self).captured_meta = meta
        return SimResult(spikes={"Pyramidal": [np.array([], dtype=float)]}, meta=meta)


def test_cmd_sim_uses_yaml_scale_duration_and_seed_when_cli_flags_are_omitted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "\n".join([
            "name: yaml_defaults",
            "scale: 0.000532",
            "duration_s: 0.25",
            "dt_s: 0.0002",
            "seed: 42",
            "crop_first_ms: 25.0",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"
    captured: dict[str, float | int] = {}

    def fake_build_network_spec(
        config: dict[str, str | int | float] | str | Path,
        scale: float,
        seed: int,
    ) -> _FakeSpec:
        del config
        captured["scale"] = scale
        captured["seed"] = seed
        return _FakeSpec()

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _FakeNestBackend)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="nest",
            scale=None,
            duration=None,
            seed=None,
            output=str(output_path),
        )
    )

    assert rc == 0
    assert captured == {"scale": 0.000532, "seed": 42}
    assert _FakeNestBackend.captured_meta is not None
    assert _FakeNestBackend.captured_meta.duration_s == 0.25
    assert _FakeNestBackend.captured_meta.dt_s == 0.0002
    assert _FakeNestBackend.captured_meta.scale == 0.000532
    assert _FakeNestBackend.captured_meta.seed == 42
    assert _FakeNestBackend.captured_meta.crop_first_ms == 25.0
    assert (
        _FakeNestBackend.captured_meta.parameter_provenance["neuron.Pyramidal"]
        == "analytic-fallback-after-failed-fit"
    )

    with h5py.File(output_path, "r") as f:
        meta = f["meta"].attrs
        provenance_raw = meta["parameter_provenance_json"]
        assert isinstance(provenance_raw, str | bytes)
        provenance_text = (
            provenance_raw.decode("utf-8")
            if isinstance(provenance_raw, bytes)
            else provenance_raw
        )
        provenance = json.loads(provenance_text)
        assert provenance["neuron.Pyramidal"] == "analytic-fallback-after-failed-fit"
        assert provenance["network.total_cells"] == "1"
        assert provenance["network.afferent_sources"] == "missing"
        assert meta["tier"] == "scaled"
        assert meta["diagnostic_provenance_json"] == (
            '{"diagnostic.audit": "no-overrides"}'
        )


def test_cmd_sim_records_diagnostic_runtime_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "\n".join([
            "name: diagnostic_env",
            "scale: 0.001",
            "duration_s: 0.01",
            "backend: nest",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"

    def fake_build_network_spec(
        config: dict[str, str | int | float] | str | Path,
        scale: float,
        seed: int,
    ) -> _FakeSpec:
        del config, scale, seed
        return _FakeSpec()

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _FakeNestBackend)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.setenv("CA1_AGLIF_DEND_GC_SCALE_BISTRATIFIED", "5")
    monkeypatch.setenv("CA1_AFFERENT_TOPOLOGY", "source_pool")
    monkeypatch.setenv("CA1_AFFERENT_SOURCE_POOL_SIZE_CA3", "1024")
    monkeypatch.setenv("CA1_ALLOW_MPI_RECURRENT_SHARDING", "1")
    monkeypatch.setenv("CA1_GPU_LFP_SAMPLE_CELLS", "0")

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="nest",
            scale=None,
            duration=None,
            seed=None,
            output=str(output_path),
        )
    )

    assert rc == 0
    assert _FakeNestBackend.captured_meta is not None
    assert _FakeNestBackend.captured_meta.diagnostic_provenance == {
        "env.CA1_ALLOW_MPI_RECURRENT_SHARDING": "1",
        "env.CA1_AFFERENT_SOURCE_POOL_SIZE_CA3": "1024",
        "env.CA1_AFFERENT_TOPOLOGY": "source_pool",
        "env.CA1_AGLIF_DEND_GC_SCALE_BISTRATIFIED": "5",
        "env.CA1_GPU_LFP_SAMPLE_CELLS": "0",
    }

    with h5py.File(output_path, "r") as f:
        assert (
            f["meta"].attrs["diagnostic_provenance_json"]
            == (
                '{"env.CA1_AFFERENT_SOURCE_POOL_SIZE_CA3": "1024", '
                '"env.CA1_AFFERENT_TOPOLOGY": "source_pool", '
                '"env.CA1_AGLIF_DEND_GC_SCALE_BISTRATIFIED": "5", '
                '"env.CA1_ALLOW_MPI_RECURRENT_SHARDING": "1", '
                '"env.CA1_GPU_LFP_SAMPLE_CELLS": "0"}'
            )
        )


def test_cmd_sim_records_diagnostic_config_calibration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "diagnostic.yaml"
    config_path.write_text(
        "\n".join([
            "name: diagnostic_config",
            "scale: 0.001",
            "duration_s: 0.01",
            "backend: nest",
            "calibration:",
            "  mode: diagnostic",
            "  projection_weight_scales:",
            "    Pyramidal->SCA: 0.0",
            "  afferent_post_weight_scales:",
            "    SCA: 0.3",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"

    def fake_build_network_spec(
        config: dict[str, str | int | float] | str | Path,
        scale: float,
        seed: int,
    ) -> _FakeSpec:
        del config, scale, seed
        return _FakeSpec()

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _FakeNestBackend)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="nest",
            scale=None,
            duration=None,
            seed=None,
            output=str(output_path),
        )
    )

    assert rc == 0
    assert _FakeNestBackend.captured_meta is not None
    assert _FakeNestBackend.captured_meta.diagnostic_provenance == {
        "config.calibration.afferent_post_weight_scales": '{"SCA": 0.3}',
        "config.calibration.mode": "diagnostic",
        "config.calibration.projection_weight_scales": '{"Pyramidal->SCA": 0.0}',
    }

    with h5py.File(output_path, "r") as f:
        assert (
            f["meta"].attrs["diagnostic_provenance_json"]
            == (
                '{"config.calibration.afferent_post_weight_scales": "{\\"SCA\\": 0.3}", '
                '"config.calibration.mode": "diagnostic", '
                '"config.calibration.projection_weight_scales": '
                '"{\\"Pyramidal->SCA\\": 0.0}"}'
            )
        )


def test_cmd_sim_fails_if_result_cannot_be_persisted(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "\n".join([
            "name: persist_required",
            "scale: 0.001",
            "duration_s: 0.01",
            "backend: nest",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"

    def fake_build_network_spec(
        config: dict[str, str | int | float] | str | Path,
        scale: float,
        seed: int,
    ) -> _FakeSpec:
        del config, scale, seed
        return _FakeSpec()

    original_import = builtins.__import__

    def import_without_h5py(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if name == "h5py":
            raise ImportError("h5py unavailable for test")
        return original_import(name, globals, locals, fromlist, level)

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _FakeNestBackend)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.setattr(builtins, "__import__", import_without_h5py)

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="nest",
            scale=None,
            duration=None,
            seed=None,
            output=str(output_path),
        )
    )

    assert rc == 1
    assert not output_path.exists()
    assert "spikes could not be persisted" in capsys.readouterr().err


def test_cmd_sim_gpu_nonzero_mpi_rank_does_not_write_shared_hdf5(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gpu.yaml"
    config_path.write_text(
        "\n".join([
            "name: gpu_rank",
            "tier: scaled",
            "scale: 1.0",
            "duration_s: 0.001",
            "backend: gpu",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"
    output_path.with_suffix(output_path.suffix + ".rank0_done").write_text("done\n")

    def fake_build_network_spec(
        config: dict[str, str | int | float] | str | Path,
        scale: float,
        seed: int,
    ) -> _FakeSpec:
        del config, scale, seed
        return _FakeSpec()

    fake_backend_module = ModuleType("ca1.sim.gpu_backend")
    setattr(fake_backend_module, "NestGpuBackend", _FakeGpuBackend)
    setattr(fake_backend_module, "_mpi_rank_size_from_env", lambda: (1, 3))
    monkeypatch.setitem(sys.modules, "ca1.sim.gpu_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.setenv("CA1_ALLOW_MPI_RECURRENT_SHARDING", "1")

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="gpu",
            scale=None,
            duration=None,
            seed=None,
            output=str(output_path),
        )
    )

    assert rc == 0
    assert _FakeGpuBackend.captured_meta is not None
    assert not output_path.exists()
    assert Path(os.environ["CA1_RUN_DIR"]).name == ".result_rank_spikes"


def test_cmd_sim_gpu_rejects_mpi_sharding_without_explicit_opt_in(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "gpu.yaml"
    config_path.write_text(
        "\n".join([
            "name: gpu_rank",
            "tier: scaled",
            "scale: 1.0",
            "duration_s: 0.001",
            "backend: gpu",
        ]),
        encoding="utf-8",
    )

    def fake_build_network_spec(
        config: dict[str, str | int | float] | str | Path,
        scale: float,
        seed: int,
    ) -> _FakeSpec:
        del config, scale, seed
        return _FakeSpec()

    fake_backend_module = ModuleType("ca1.sim.gpu_backend")
    setattr(fake_backend_module, "NestGpuBackend", _FakeGpuBackend)
    setattr(fake_backend_module, "_mpi_rank_size_from_env", lambda: (0, 3))
    monkeypatch.setitem(sys.modules, "ca1.sim.gpu_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.delenv("CA1_ALLOW_MPI_RECURRENT_SHARDING", raising=False)
    monkeypatch.delenv("CA1_RUN_DIR", raising=False)

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="gpu",
            scale=None,
            duration=None,
            seed=None,
            output=str(tmp_path / "result.hdf5"),
        )
    )

    stderr = capsys.readouterr().err
    assert rc == 1
    assert "CA1_ALLOW_MPI_RECURRENT_SHARDING=1" in stderr
    assert "CA1_RUN_DIR" not in os.environ
