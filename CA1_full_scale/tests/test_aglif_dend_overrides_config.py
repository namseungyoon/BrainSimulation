from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import hashlib
import json

import pytest

from ca1.config import build_network_spec
from ca1.params.provenance import parameter_provenance_for_spec
from ca1.sim.aglif_dend import aglif_dend_compartments, aglif_dend_status


_TRANSFER_TABLE = (
    "src/ca1/params/source_location_transfer_syndata120_budget_weighted.json"
)


def _aglif_config() -> dict[str, object]:
    return {
        "name": "aglif_dend_override_config",
        "neuron_model": "aglif_dend_cond_beta",
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "budget_weighted",
        "syndata_variant": 120,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "cellnumbers_index": 101,
        "source_location_transfer_mode": "all_dend",
        "source_location_transfer_table": _TRANSFER_TABLE,
    }


def test_build_network_spec_rejects_unknown_aglif_override_cell_type() -> None:
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"Basket": {"g_c_scale": 2.0}}

    with pytest.raises(ValueError, match="unknown AGLIF override cell type.*Basket"):
        _ = build_network_spec(config)


def test_build_network_spec_rejects_unknown_aglif_receive_domain_mode() -> None:
    config = _aglif_config()
    config["aglif_dend_overrides"] = {
        "Bistratified": {"receive_domain": "axon"}
    }

    with pytest.raises(ValueError, match="receive_domain.*Bistratified.*axon"):
        _ = build_network_spec(config)


@pytest.mark.parametrize(
    "scale",
    [0.0, -1.0, float("inf"), float("nan"), True],
)
def test_build_network_spec_rejects_invalid_aglif_gc_scale(scale: object) -> None:
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"O_LM": {"g_c_scale": scale}}

    with pytest.raises((TypeError, ValueError), match="g_c_scale.*O_LM"):
        _ = build_network_spec(config)


def test_configured_bistratified_soma_excitatory_routes_ampa_to_soma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CA1_AGLIF_DEND_EXC_SOMA", raising=False)
    monkeypatch.delenv("CA1_AGLIF_DEND_EXC_SOMA_CELL_TYPES", raising=False)
    config = _aglif_config()
    config["aglif_dend_overrides"] = {
        "Bistratified": {"receive_domain": "soma_excitatory"}
    }
    receptor = "AMPA_fast__e0__tr0p07__td0p2__dend"

    spec = build_network_spec(config)
    override = spec.aglif_dend_overrides["Bistratified"]
    compartments = aglif_dend_compartments(
        (receptor,),
        "Bistratified",
        frozenset({receptor}),
        spec.source_location_transfer_table,
        receive_domain=override.receive_domain,
    )

    assert Path(spec.source_location_transfer_table).name == Path(_TRANSFER_TABLE).name
    assert compartments == [0.0]


def test_configured_olm_gc_scale_routes_through_neuron_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_backend_mod = pytest.importorskip("ca1.sim.gpu_backend")
    monkeypatch.delenv("CA1_AGLIF_DEND_GC_SCALE", raising=False)
    monkeypatch.delenv("CA1_AGLIF_DEND_GC_SCALE_O_LM", raising=False)
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"O_LM": {"g_c_scale": 3.5}}

    spec = build_network_spec(config)
    status = gpu_backend_mod._neuron_status(
        spec,
        "O_LM",
        spec.cell_types["O_LM"].params,
    )
    baseline = aglif_dend_status("O_LM")

    assert status["g_c"] == pytest.approx(baseline["g_c"] * 3.5)


def test_parameter_provenance_records_explicit_aglif_overrides() -> None:
    config = _aglif_config()
    config["aglif_dend_overrides"] = {
        "Bistratified": {"receive_domain": "soma_excitatory"},
        "O_LM": {"g_c_scale": 3.5},
    }

    spec = build_network_spec(config)
    provenance = parameter_provenance_for_spec(spec)

    assert provenance["aglif_dend_override.Bistratified.receive_domain"] == (
        "soma_excitatory"
    )
    assert provenance["aglif_dend_override.O_LM.g_c_scale"] == "3.5"


def test_user_m3_is_explicit_cck_only_opt_in_and_default_stays_user_m2() -> None:
    gpu = pytest.importorskip("ca1.sim.gpu_backend")
    base = build_network_spec(_aglif_config())
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"CCK_Basket": {"model": "user_m3"}}
    opted = build_network_spec(config)

    assert gpu._nestgpu_model_name_for_cell(base, "CCK_Basket") == "user_m2"
    assert gpu._nestgpu_model_name_for_cell(opted, "CCK_Basket") == "user_m3"
    assert all(
        gpu._nestgpu_model_name_for_cell(opted, cell_type) == "user_m2"
        for cell_type in opted.cell_types
        if cell_type != "CCK_Basket"
    )
    status = gpu._neuron_status(
        opted, "CCK_Basket", opted.cell_types["CCK_Basket"].params
    )
    assert 0.0 < status["h"] < 1.0
    assert status["h_crit"] > 0.0


def test_user_m3_override_cannot_be_applied_to_non_cck_cell() -> None:
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"SCA": {"model": "user_m3"}}
    with pytest.raises(ValueError, match="CCK_Basket only"):
        _ = build_network_spec(config)


def test_user_m3_model_swap_preserves_connection_graph_digest() -> None:
    def digest(spec: object) -> str:
        payload = {
            "cell_counts": {k: v.count for k, v in spec.cell_types.items()},
            "receptors": {
                k: asdict(spec.receptors_for_post(k)) for k in spec.cell_types
            },
            "projections": [asdict(item) for item in spec.projections],
            "afferents": [asdict(item) for item in spec.afferents],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    base = build_network_spec(_aglif_config())
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"CCK_Basket": {"model": "user_m3"}}
    opted = build_network_spec(config)
    assert digest(base) == digest(opted)
    for cell_type in ("PV_Basket", "Bistratified", "O_LM"):
        m4_config = _aglif_config()
        m4_config["aglif_dend_overrides"] = {
            cell_type: {"model": "user_m4"}
        }
        assert digest(base) == digest(build_network_spec(m4_config))
        m5_config = _aglif_config()
        m5_config["aglif_dend_overrides"] = {
            cell_type: {"model": "user_m5"}
        }
        assert digest(base) == digest(build_network_spec(m5_config))
    m7_config = _aglif_config()
    m7_config["aglif_dend_overrides"] = {"PV_Basket": {"model": "user_m7"}}
    assert digest(base) == digest(build_network_spec(m7_config))


@pytest.mark.parametrize("cell_type", ["PV_Basket", "Bistratified", "O_LM"])
def test_user_m4_is_ping_only_opt_in_with_source_status(cell_type: str) -> None:
    gpu = pytest.importorskip("ca1.sim.gpu_backend")
    config = _aglif_config()
    config["aglif_dend_overrides"] = {cell_type: {"model": "user_m4"}}
    spec = build_network_spec(config)
    assert gpu._nestgpu_model_name_for_cell(spec, cell_type) == "user_m4"
    status = gpu._neuron_status(spec, cell_type, spec.cell_types[cell_type].params)
    assert status["gbar_Na_prox"] > 0.0
    assert status["gbar_Kd_dist"] > 0.0
    assert 0.0 < status["h_Na_prox"] < 1.0
    assert 0.0 < status["n_Kd_dist"] < 1.0


def test_user_m4_rejects_non_ping_cell() -> None:
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"CCK_Basket": {"model": "user_m4"}}
    with pytest.raises(ValueError, match="PV_Basket/Bistratified/O_LM only"):
        _ = build_network_spec(config)


@pytest.mark.parametrize("cell_type", ["PV_Basket", "Bistratified", "O_LM"])
def test_user_m5_is_ping_only_opt_in_with_private_branch_status(cell_type: str) -> None:
    gpu = pytest.importorskip("ca1.sim.gpu_backend")
    config = _aglif_config()
    config["aglif_dend_overrides"] = {cell_type: {"model": "user_m5"}}
    spec = build_network_spec(config)
    assert gpu._nestgpu_model_name_for_cell(spec, cell_type) == "user_m5"
    status = gpu._neuron_status(spec, cell_type, spec.cell_types[cell_type].params)
    assert status["C_b_prox"] > 0.0
    assert status["g_b_dist"] > 0.0
    assert status["V_b_prox"] == status["E_L"]
    assert 0.0 < status["h_Na_dist"] < 1.0


def test_user_m5_rejects_non_ping_cell() -> None:
    config = _aglif_config()
    config["aglif_dend_overrides"] = {"CCK_Basket": {"model": "user_m5"}}
    with pytest.raises(ValueError, match="PV_Basket/Bistratified/O_LM only"):
        _ = build_network_spec(config)
