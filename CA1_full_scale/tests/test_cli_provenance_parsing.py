from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pytest

from ca1 import cli


@pytest.mark.parametrize(
    "raw",
    [
        '{"neuron.Pyramidal": null}',
        '{"neuron.Pyramidal": 0}',
        '{"neuron.Pyramidal": ["nest-validated"]}',
    ],
)
def test_read_parameter_provenance_requires_string_values(raw: str) -> None:
    with pytest.raises(TypeError, match="parameter_provenance_json"):
        _ = cli.read_parameter_provenance_json(raw)


def test_read_diagnostic_provenance_requires_string_values() -> None:
    with pytest.raises(TypeError, match="diagnostic_provenance_json"):
        _ = cli.read_diagnostic_provenance_json('{"diagnostic.audit": null}')


def _write_cli_result(path: Path, *, tier: str | None) -> None:
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["duration_s"] = 1.0
        meta.attrs["dt_s"] = 0.001
        meta.attrs["scale"] = 1.0
        if tier is not None:
            meta.attrs["tier"] = tier
        meta.attrs["seed"] = 1
        meta.attrs["backend"] = "gpu"
        meta.attrs["config_name"] = "stored_scaled"
        meta.attrs["crop_first_ms"] = 0.0
        meta.attrs["lfp_proxy"] = "pyramidal_spike_density"
        meta.attrs["parameter_provenance_json"] = json.dumps(
            {"network.neuron_model": "aeif_cond_beta_multisynapse"},
            sort_keys=True,
        )
        meta.attrs["diagnostic_provenance_json"] = json.dumps(
            {"diagnostic.audit": "no-overrides"},
            sort_keys=True,
        )
        spikes = h5.create_group("spikes")
        pyramidal = spikes.create_group("Pyramidal")
        _ = pyramidal.create_dataset("0", data=[])
        n_cells = h5.create_group("n_cells_per_type")
        n_cells.attrs["Pyramidal"] = 1


def test_cmd_validate_refuses_to_promote_stored_scaled_tier(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a result file explicitly persisted as scaled-tier evidence.
    result_path = tmp_path / "scaled_result.h5"
    _write_cli_result(result_path, tier="scaled")

    # When: validation is asked to promote that artifact to full-tier.
    rc = cli.cmd_validate(
        argparse.Namespace(result=str(result_path), tier="full"),
    )

    # Then: the CLI refuses before harness validation can hide the tier mismatch.
    captured = capsys.readouterr()
    assert rc == 1
    assert "stored tier=scaled" in captured.err
    assert "cannot validate as full-tier" in captured.err


def test_cmd_validate_refuses_to_promote_legacy_missing_tier_to_full(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a legacy result file with no stored tier metadata.
    result_path = tmp_path / "legacy_result.h5"
    _write_cli_result(result_path, tier=None)

    # When: validation is asked to treat that artifact as full-tier.
    rc = cli.cmd_validate(
        argparse.Namespace(result=str(result_path), tier="full"),
    )

    # Then: missing tier metadata is a hard blocker for final-tier evidence.
    captured = capsys.readouterr()
    assert rc == 1
    assert "tier metadata missing" in captured.err
    assert "cannot validate as full-tier" in captured.err


def test_cmd_validate_rejects_scaled_tier_before_reading_spikes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a scaled-tier artifact whose spike payload is malformed.
    result_path = tmp_path / "scaled_bad_spikes.h5"
    _write_cli_result(result_path, tier="scaled")
    with h5py.File(result_path, "a") as h5:
        del h5["spikes"]
        _ = h5.create_dataset("spikes", data=[0])

    # When: validation is asked to promote it to full-tier.
    rc = cli.cmd_validate(
        argparse.Namespace(result=str(result_path), tier="full"),
    )

    # Then: the CLI rejects the stored tier before spike payload parsing.
    captured = capsys.readouterr()
    assert rc == 1
    assert "stored tier=scaled" in captured.err
    assert "cannot validate as full-tier" in captured.err


def test_cmd_validate_default_refuses_legacy_missing_tier(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a legacy scale-1.0 result whose stored tier metadata is missing.
    result_path = tmp_path / "legacy_default_result.h5"
    _write_cli_result(result_path, tier=None)

    # When: validation uses its default stored-tier mode.
    rc = cli.cmd_validate(
        argparse.Namespace(result=str(result_path), tier=None),
    )

    # Then: the CLI refuses instead of re-inferring full tier from scale.
    captured = capsys.readouterr()
    assert rc == 1
    assert "tier metadata missing" in captured.err
    assert "cannot infer final-tier eligibility" in captured.err


def test_cmd_validate_rejects_stored_lfp_without_explicit_dt(
    tmp_path: Path,
) -> None:
    # Given: a persisted artifact with an LFP dataset but no LFP sample interval.
    result_path = tmp_path / "missing_lfp_dt.h5"
    _write_cli_result(result_path, tier="scaled")
    with h5py.File(result_path, "a") as h5:
        _ = h5.create_dataset("lfp", data=[0.0, 1.0, 0.0])

    # When / Then: validation refuses instead of falling back to meta.dt_s.
    with pytest.raises(TypeError, match="lfp_dt_s metadata missing"):
        _ = cli.cmd_validate(
            argparse.Namespace(result=str(result_path), tier=None),
        )
