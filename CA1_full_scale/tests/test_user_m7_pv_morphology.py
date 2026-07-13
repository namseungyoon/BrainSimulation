from __future__ import annotations
from pathlib import Path
import pytest
import yaml
from ca1.config import build_network_spec
from ca1.sim import gpu_backend as gpu
from ca1.sim.aglif_dend import user_m7_status
from ca1.sim.user_m7 import (PV_LANE_SITES, PV_ROUTE_TABLE_SHA256,
                             branch_derivatives_per_ms, route_contacts)


def test_user_m7_is_pv_only_opt_in() -> None:
    base = Path(__file__).resolve().parents[1] / "configs/full_scale_3dtopo.yaml"
    config = yaml.safe_load(base.read_text())
    config["aglif_dend_overrides"] = {"PV_Basket": {"model": "user_m7"}}
    spec = build_network_spec(config)
    assert gpu._nestgpu_model_name_for_cell(spec, "PV_Basket") == "user_m7"
    assert gpu._nestgpu_model_name_for_cell(spec, "Bistratified") == "user_m2"
    with pytest.raises(ValueError, match="PV_Basket only"):
        config["aglif_dend_overrides"] = {"Bistratified": {"model": "user_m7"}}
        build_network_spec(config)


def test_native_site_lane_multiplicities_are_exact() -> None:
    assert tuple(PV_LANE_SITES["dend_50_200"].count(x) for x in range(4)) == (4, 4, 5, 5)
    assert tuple(PV_LANE_SITES["apical_gt_100"].count(x) for x in range(4)) == (12, 12, 0, 0)
    assert tuple(PV_LANE_SITES["apical_gt_200"].count(x) for x in range(4)) == (9, 9, 0, 0)
    assert tuple(PV_LANE_SITES["dend_gt_200"].count(x) for x in range(4)) == (9, 9, 3, 3)
    assert len(PV_ROUTE_TABLE_SHA256) == 64


def test_routing_is_deterministic_target_and_port_aware() -> None:
    sites = PV_LANE_SITES["dend_50_200"]
    a = route_contacts(11, 29, "CA3|AMPA", 64, sites)
    assert a == route_contacts(11, 29, "CA3|AMPA", 64, sites)
    assert sum(a) == 64
    assert a != route_contacts(11, 30, "CA3|AMPA", 64, sites)
    assert a != route_contacts(11, 29, "Bist|GABA", 64, sites)


def test_lanes_are_heterogeneous_and_axial_current_is_bidirectional() -> None:
    p = user_m7_status("PV_Basket", -65.0); p["E_L"] = -65.0
    assert p["C_b_prox_0"] != p["C_b_prox_2"]
    assert p["g_ax_b_dist_0"] != p["g_ax_b_dist_2"]
    _, into_lane = branch_derivatives_per_ms(-70, -60, 0, p["h_Na_prox_0"],
                                              p["n_Kd_prox_0"], p, "prox", 0)
    _, out_of_lane = branch_derivatives_per_ms(-50, -60, 0, p["h_Na_prox_0"],
                                                p["n_Kd_prox_0"], p, "prox", 0)
    assert into_lane > 0 and out_of_lane < 0


def test_static_port_and_event_abi_matches_user_m2() -> None:
    root = Path(__file__).resolve().parents[1] / "nest-gpu/src"
    m2 = (root / "user_m2_kernel.h").read_text()
    m7 = (root / "user_m7_kernel.h").read_text()
    for literal in ('{ "g", "g1" }',
                    '{ "E_rev", "tau_rise", "tau_decay", "g0", "compartment" }'):
        assert literal in m2 and literal in m7
    assert "i_V_m=0, i_V_d, i_V_dist, i_I_adap, i_I_dep" in m7
    assert "else if(V_m>=V_th)" in m7
    assert "PushSpike(data.i_node_0_+idx,1.0)" in m7
    assert "domain_reaction[d]-=axial" in m7
