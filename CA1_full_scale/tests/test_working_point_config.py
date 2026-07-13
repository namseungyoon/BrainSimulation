from __future__ import annotations

import math

import pytest

from ca1.config import build_network_spec, load_config
from ca1.params.provenance import diagnostic_config_provenance
from ca1.validation.provenance import final_tier_diagnostic_provenance_blockers
from ca1.validation.targets import MODEL_RATES_HZ


def test_working_point_defaults_off() -> None:
    spec = build_network_spec({"name": "working-point-off"})

    assert spec.working_point_mode == "off"
    assert spec.working_point_clamp_rates_hz == {}


@pytest.mark.parametrize("rates", [None, "table5"])
def test_working_point_clamp_defaults_to_all_table5_interneurons(
    rates: str | None,
) -> None:
    config: dict[str, object] = {
        "name": "working-point-table5",
        "working_point_mode": "clamp",
    }
    if rates is not None:
        config["working_point_clamp_rates_hz"] = rates

    spec = build_network_spec(config)

    assert spec.working_point_mode == "clamp"
    assert spec.working_point_clamp_rates_hz == {
        cell_type: rate_hz
        for cell_type, rate_hz in MODEL_RATES_HZ.items()
        if cell_type != "Pyramidal"
    }
    assert len(spec.working_point_clamp_rates_hz) == 8


def test_working_point_clamp_accepts_explicit_real_cell_rates() -> None:
    spec = build_network_spec({
        "name": "working-point-custom",
        "working_point_mode": "clamp",
        "working_point_clamp_rates_hz": {
            "PV_Basket": 7,
            "Pyramidal": 6.0,
        },
    })

    assert spec.working_point_clamp_rates_hz == {
        "PV_Basket": 7.0,
        "Pyramidal": 6.0,
    }


def test_working_point_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported working_point_mode"):
        _ = build_network_spec({"working_point_mode": "shadow"})


def test_working_point_rejects_unknown_cell_type() -> None:
    with pytest.raises(ValueError, match="unknown cell types.*NotACell"):
        _ = build_network_spec({
            "working_point_mode": "clamp",
            "working_point_clamp_rates_hz": {"NotACell": 1.0},
        })


@pytest.mark.parametrize("rate", [0.0, -1.0, math.inf, math.nan])
def test_working_point_rejects_nonpositive_or_nonfinite_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _ = build_network_spec({
            "working_point_mode": "clamp",
            "working_point_clamp_rates_hz": {"PV_Basket": rate},
        })


@pytest.mark.parametrize("rates", ["not-table5", [], True])
def test_working_point_rejects_nonmapping_rates(rates: object) -> None:
    with pytest.raises(TypeError, match="must be a mapping or 'table5'"):
        _ = build_network_spec({
            "working_point_mode": "clamp",
            "working_point_clamp_rates_hz": rates,
        })


def test_working_point_rejects_boolean_rate() -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        _ = build_network_spec({
            "working_point_mode": "clamp",
            "working_point_clamp_rates_hz": {"PV_Basket": True},
        })


def test_working_point_config_is_diagnostic_provenance_blocker() -> None:
    provenance = diagnostic_config_provenance({
        "working_point_mode": "clamp",
        "working_point_clamp_rates_hz": "table5",
    })

    assert provenance == {
        "config.working_point_mode": "clamp",
        "config.working_point_clamp_rates_hz": "table5",
    }
    assert final_tier_diagnostic_provenance_blockers(provenance) == [
        "config.working_point_clamp_rates_hz=table5",
        "config.working_point_mode=clamp",
    ]


def test_full_scale_clamp_config_is_scaled_table5_diagnostic() -> None:
    config = load_config("configs/full_scale_clamp_all_table5.yaml")

    assert config["scale"] == 1.0
    assert config["duration_s"] == 2.5
    assert config["tier"] == "scaled"
    assert config["neuron_model"] == "aglif_dend_cond_beta"
    assert config["afferent_topology"] == "literal_source_graph"
    assert config["recurrent_topology"] == "modeldb_fastconn_binned"
    assert config["source_location_transfer_mode"] == "all_dend"
    assert config["conndata_index"] == 430
    assert config["working_point_mode"] == "clamp"
    assert config["working_point_clamp_rates_hz"] == "table5"

    spec = build_network_spec(config)
    assert set(spec.working_point_clamp_rates_hz) == set(MODEL_RATES_HZ) - {
        "Pyramidal"
    }
