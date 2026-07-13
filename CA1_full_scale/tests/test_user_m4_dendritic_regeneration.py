from __future__ import annotations

from pathlib import Path
import pytest

from ca1.sim.aglif_dend import user_m4_status
from ca1.sim.user_m4 import (
    activation_inf, dendritic_currents_pA, inactivation_inf,
    recovery_derivatives_per_ms,
)

TARGETS = ("PV_Basket", "Bistratified", "O_LM")


@pytest.mark.parametrize("target", TARGETS)
def test_user_m4_activation_is_regenerative_but_quiescent_at_rest(target: str) -> None:
    p = user_m4_status(target, -65.0)
    rest = dendritic_currents_pA(-65.0, p["h_Na_prox"], p["n_Kd_prox"], p)
    recruited = dendritic_currents_pA(p["Vm_half"] + 5.0, 1.0, 0.0, p)
    assert abs(rest[0]) < 0.1
    assert recruited[0] > 100.0
    assert activation_inf(-40.0, p["Vm_half"], p["km"]) > activation_inf(
        -60.0, p["Vm_half"], p["km"])


@pytest.mark.parametrize("target", TARGETS)
def test_user_m4_native_na_and_k_recovery_have_correct_direction(target: str) -> None:
    p = user_m4_status(target)
    assert inactivation_inf(-70.0, p["Vh_half"], p["kh"]) > inactivation_inf(
        -20.0, p["Vh_half"], p["kh"])
    dh, dn = recovery_derivatives_per_ms(-70.0, 0.0, 1.0, p)
    assert dh > 0.0
    assert dn < 0.0


def test_user_m4_parameter_contract_is_complete_and_cell_specific() -> None:
    required = {"gbar_Na_prox", "gbar_Na_dist", "E_Na", "Vm_half", "km",
                "Vh_half", "kh", "tau_h", "gbar_Kd_prox", "gbar_Kd_dist",
                "E_K", "Vn_half", "kn", "tau_n"}
    values = [user_m4_status(target, -65.0) for target in TARGETS]
    assert all(required <= set(value) for value in values)
    assert len({(value["Vm_half"], value["gbar_Na_prox"], value["Vh_half"])
                for value in values}) == len(TARGETS)


@pytest.mark.parametrize("target", TARGETS)
def test_user_m4_recovery_is_dt_stable(target: str) -> None:
    p = user_m4_status(target)
    def recover(dt: float) -> tuple[float, float]:
        h, n = 0.0, 1.0
        for _ in range(round(100.0 / dt)):
            dh, dn = recovery_derivatives_per_ms(-70.0, h, n, p)
            h += dt * dh; n += dt * dn
        return h, n
    assert recover(0.05) == pytest.approx(recover(0.025), abs=5e-4)


def test_user_m4_static_port_abi_matches_user_m2() -> None:
    root = Path(__file__).resolve().parents[1] / "nest-gpu/src"
    m2 = (root / "user_m2_kernel.h").read_text()
    m4 = (root / "user_m4_kernel.h").read_text()
    for literal in ('{ "g", "g1" }',
                    '{ "E_rev", "tau_rise", "tau_decay", "g0", "compartment" }'):
        assert literal in m2
        assert literal in m4
    assert "if(V_m>=V_th)" in m4
    assert "PushSpike(data.i_node_0_+idx,1.0)" in m4
