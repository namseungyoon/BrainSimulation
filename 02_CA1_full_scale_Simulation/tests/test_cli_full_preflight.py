from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Literal

import numpy as np
import pytest

from ca1 import cli
from ca1.types import SimMeta, SimResult

_FULL_COUNTS = {
    "Axo": 1_470,
    "Bistratified": 2_210,
    "CCK_Basket": 3_600,
    "Ivy": 8_810,
    "Neurogliaform": 3_580,
    "O_LM": 1_640,
    "PV_Basket": 5_530,
    "Pyramidal": 311_500,
    "SCA": 400,
}


class _PreflightSpec:
    neuron_model: Literal["aglif_dend_cond_beta"] = "aglif_dend_cond_beta"
    source_location_transfer_provenance: str = ""

    def scaled_counts(self) -> dict[str, int]:
        return {"Pyramidal": 1}

    def total_cells(self) -> int:
        return 1


class _FullPostrunSpec:
    neuron_model: Literal["aglif_cond_beta"] = "aglif_cond_beta"
    source_location_transfer_provenance: str = ""
    scale: float = 1.0
    cellnumbers_index: int = 0

    def scaled_counts(self) -> dict[str, int]:
        return dict(_FULL_COUNTS)

    def total_cells(self) -> int:
        return sum(_FULL_COUNTS.values())


class _BackendMustNotRun:
    constructed: ClassVar[bool] = False
    simulated: ClassVar[bool] = False

    def __init__(self) -> None:
        type(self).constructed = True

    def simulate(self, _spec: _PreflightSpec, meta: SimMeta) -> SimResult:
        type(self).simulated = True
        return SimResult(spikes={"Pyramidal": [np.array([], dtype=float)]}, meta=meta)


class _BackendReturnsReducedLfp:
    simulated: ClassVar[bool] = False

    def simulate(self, _spec: _FullPostrunSpec, meta: SimMeta) -> SimResult:
        type(self).simulated = True
        return SimResult(
            spikes={"Pyramidal": [np.array([], dtype=float)]},
            meta=replace(meta, lfp_proxy="pyramidal_synaptic_current"),
            lfp=np.array([0.0, 1.0, 0.0], dtype=float),
            lfp_dt_s=0.001,
        )


def _full_parameter_provenance() -> dict[str, str]:
    provenance = {
        "calibration.mode": "paper_reduction",
        "lfp.modeldb_n_pole_reduced_domain": "modeldb-n-pole-reduced-domain-lfp",
        "network.total_cells": "338740",
        "network.cell_types": "9",
        "network.recurrent_projections": "68",
        "network.afferents": "13",
        "network.afferent_sources": "CA3,ECIII",
        "network.afferent_rate_hz": "0.65",
        "network.afferent_source_rate_rule": "homogeneous",
        "network.afferent_source_count_total": "454700",
        "network.afferent_source_count_max": "250000",
        "network.recurrent_topology": "modeldb_fastconn_3d_gaussian",
        "network.afferent_topology": "literal_source_graph",
        "network.afferent_poisson_rule": "literal_shared_source_graph",
        "network.afferent_source_driver": "precomputed_poisson_spike_generator",
        "network.conndata_index": "430",
        "network.cellnumbers_index": "101",
        "network.conndata_count_mode": "per_cell",
        "network.recurrent_synapses": "441375540",
        "network.afferent_synapses": "4704026540",
        "network.total_synapses": "5145402080",
        "network.multisynapse_rule": "same_source_same_delay_weight_aggregation",
        "network.neuron_model": "aglif_cond_beta",
        "synapse.receptor_ports": (
            "syndata120-compartment-aware-20port-budget_weighted;sha256="
            "26774704b306d1bd0461fd7df69491cfacd0e1a2e6385877ece2150c9e05e46c"
        ),
        "synapse.short_term_plasticity": "static_exp2syn_no_stp",
    }
    provenance.update({f"aglif.{cell_type}": "nestgpu-fi-fit" for cell_type in _FULL_COUNTS})
    return provenance


def test_cmd_sim_full_tier_rejects_unvalidated_provenance_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a full-tier simulation whose fitted dendritic transfer record is not final-validated.
    config_path = tmp_path / "full.yaml"
    _ = config_path.write_text(
        "\n".join([
            "name: full_preflight",
            "tier: full",
            "scale: 1.0",
            "duration_s: 0.001",
            "backend: nest",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"

    def fake_build_network_spec(
        config: dict[str, str | int | float],
        scale: float,
        seed: int,
    ) -> _PreflightSpec:
        del config, scale, seed
        return _PreflightSpec()

    def fake_parameter_provenance_for_spec(_spec: _PreflightSpec) -> dict[str, str]:
        return {
            "network.neuron_model": "aglif_dend_cond_beta",
            "dendritic_transfer.Pyramidal": (
                "neuron-epsp-location-compressed-fit;validation-missing"
            ),
        }

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _BackendMustNotRun)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.setattr(
        "ca1.params.provenance.parameter_provenance_for_spec",
        fake_parameter_provenance_for_spec,
    )

    # When: the CLI sim path is invoked.
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

    # Then: it fails loudly before any backend execution can hide the fallback.
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "validation-missing" in stderr
    assert "dendritic_transfer.Pyramidal" in stderr
    assert not _BackendMustNotRun.constructed
    assert not _BackendMustNotRun.simulated
    assert not output_path.exists()


def test_cmd_sim_full_tier_rejects_non_final_lfp_after_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: full-tier provenance passes preflight, but the backend returns reduced current LFP.
    config_path = tmp_path / "full_postrun_lfp.yaml"
    _ = config_path.write_text(
        "\n".join([
            "name: full_postrun_lfp",
            "tier: full",
            "scale: 1.0",
            "duration_s: 0.001",
            "backend: nest",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"
    _BackendReturnsReducedLfp.simulated = False

    def fake_build_network_spec(
        config: dict[str, str | int | float],
        scale: float,
        seed: int,
    ) -> _FullPostrunSpec:
        del config, scale, seed
        return _FullPostrunSpec()

    def fake_parameter_provenance_for_spec(_spec: _FullPostrunSpec) -> dict[str, str]:
        return _full_parameter_provenance()

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _BackendReturnsReducedLfp)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.setattr(
        "ca1.params.provenance.parameter_provenance_for_spec",
        fake_parameter_provenance_for_spec,
    )

    # When: the CLI sim path receives the backend result.
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

    # Then: the artifact is rejected before HDF or manifest persistence.
    stderr = capsys.readouterr().err
    assert rc == 1
    assert _BackendReturnsReducedLfp.simulated
    assert "full-tier runtime provenance is not final-eligible" in stderr
    assert "pyramidal_synaptic_current" in stderr
    assert not output_path.exists()
    assert not output_path.with_suffix(output_path.suffix + ".manifest.json").exists()


def test_cmd_sim_full_tier_rejects_missing_structural_provenance_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: full-tier model-fit provenance without the structural audit records.
    config_path = tmp_path / "full.yaml"
    _ = config_path.write_text(
        "\n".join([
            "name: missing_structural_preflight",
            "tier: full",
            "scale: 1.0",
            "duration_s: 0.001",
            "backend: nest",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"

    def fake_build_network_spec(
        config: dict[str, str | int | float],
        scale: float,
        seed: int,
    ) -> _PreflightSpec:
        del config, scale, seed
        return _PreflightSpec()

    def fake_parameter_provenance_for_spec(_spec: _PreflightSpec) -> dict[str, str]:
        return {
            "network.neuron_model": "aglif_dend_cond_beta",
            "aglif.Pyramidal": "nestgpu-fi-fit;validation-passed",
            "dendritic_transfer.Pyramidal": (
                "neuron-epsp-location-compressed-fit;validation-passed"
            ),
        }

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _BackendMustNotRun)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.setattr(
        "ca1.params.provenance.parameter_provenance_for_spec",
        fake_parameter_provenance_for_spec,
    )
    _BackendMustNotRun.constructed = False
    _BackendMustNotRun.simulated = False

    # When: the CLI sim path is invoked.
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

    # Then: structural provenance gaps are also preflight blockers.
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "network.total_cells" in stderr
    assert "network.recurrent_projections" in stderr
    assert not _BackendMustNotRun.constructed
    assert not _BackendMustNotRun.simulated
    assert not output_path.exists()


def test_cmd_sim_full_tier_rejects_diagnostic_config_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a full-tier simulation with a diagnostic calibration override.
    config_path = tmp_path / "full_diagnostic.yaml"
    _ = config_path.write_text(
        "\n".join([
            "name: full_diagnostic_preflight",
            "tier: full",
            "scale: 1.0",
            "duration_s: 0.001",
            "backend: nest",
            "calibration:",
            "  mode: diagnostic",
            "  projection_weight_scales:",
            "    Pyramidal->PV_Basket: 7.0",
        ]),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.hdf5"
    _BackendMustNotRun.constructed = False
    _BackendMustNotRun.simulated = False

    def fake_build_network_spec(
        config: dict[str, str | int | float],
        scale: float,
        seed: int,
    ) -> _PreflightSpec:
        del config, scale, seed
        return _PreflightSpec()

    def fake_parameter_provenance_for_spec(_spec: _PreflightSpec) -> dict[str, str]:
        return {
            "network.neuron_model": "aglif_dend_cond_beta",
            "aglif.Pyramidal": "nestgpu-fi-fit;validation-passed",
            "dendritic_transfer.Pyramidal": (
                "neuron-epsp-location-compressed-fit;validation-passed"
            ),
        }

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _BackendMustNotRun)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)
    monkeypatch.setattr("ca1.config.build_network_spec", fake_build_network_spec)
    monkeypatch.setattr(
        "ca1.params.provenance.parameter_provenance_for_spec",
        fake_parameter_provenance_for_spec,
    )

    # When: the CLI sim path is invoked.
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

    # Then: diagnostic calibration is not allowed to run as hidden full-tier input.
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "diagnostic" in stderr
    assert "config.calibration.projection_weight_scales" in stderr
    assert not _BackendMustNotRun.constructed
    assert not _BackendMustNotRun.simulated
    assert not output_path.exists()


def test_cmd_sim_full_scale_config_rejects_diagnostic_afferents_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "full_scale_compound.yaml"
    config_text = (
        Path("configs/full_scale.yaml").read_text(encoding="utf-8")
        .replace("afferent_topology: literal_source_graph", "afferent_topology: compound")
        .replace("source_location_transfer_mode: all_dend", "source_location_transfer_mode: none")
    )
    _ = config_path.write_text(config_text, encoding="utf-8")
    output_path = tmp_path / "result.hdf5"
    _BackendMustNotRun.constructed = False
    _BackendMustNotRun.simulated = False

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _BackendMustNotRun)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="nest",
            scale=None,
            duration=0.001,
            seed=None,
            output=str(output_path),
        )
    )

    stderr = capsys.readouterr().err
    assert rc == 1
    assert "network.afferent_topology=compound is a diagnostic" in stderr
    assert "simulation refused before backend execution" in stderr
    assert not _BackendMustNotRun.constructed
    assert not _BackendMustNotRun.simulated
    assert not output_path.exists()


def test_cmd_sim_full_scale_config_rejects_unvalidated_transfer_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "full_scale_unvalidated_transfer.yaml"
    config_text = (
        Path("configs/full_scale.yaml").read_text(encoding="utf-8")
        .replace("duration_s: 10.0", "duration_s: 0.001")
        .replace("backend: gpu", "backend: nest")
        .replace(
            (
                "source_location_transfer_table: "
                "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
            ),
            (
                "source_location_transfer_table: "
                ".omo/ulw-loop/g002-continuation/evidence/"
                "active_inhibitory_m2_probe_wave226/"
                "location_transfer_active_inhibitory_diag.json"
            ),
        )
    )
    _ = config_path.write_text(config_text, encoding="utf-8")
    output_path = tmp_path / "result.hdf5"
    _BackendMustNotRun.constructed = False
    _BackendMustNotRun.simulated = False

    fake_backend_module = ModuleType("ca1.sim.nest_backend")
    setattr(fake_backend_module, "NestBackend", _BackendMustNotRun)
    monkeypatch.setitem(sys.modules, "ca1.sim.nest_backend", fake_backend_module)

    rc = cli.cmd_sim(
        argparse.Namespace(
            config=str(config_path),
            backend="nest",
            scale=None,
            duration=0.001,
            seed=None,
            output=str(output_path),
        )
    )

    stderr = capsys.readouterr().err
    assert rc == 1
    assert "network spec is not final-buildable" in stderr
    assert "unvalidated M2 evidence fields" in stderr
    assert "simulation refused before backend execution" in stderr
    assert not _BackendMustNotRun.constructed
    assert not _BackendMustNotRun.simulated
    assert not output_path.exists()
