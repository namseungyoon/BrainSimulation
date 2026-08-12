from __future__ import annotations

from pathlib import Path

import pytest

from ca1.sim.aglif_dend import user_m5_status
from ca1.sim.user_m4 import recovery_derivatives_per_ms
from ca1.sim.user_m5 import branch_derivatives_per_ms

TARGETS = ("PV_Basket", "Bistratified", "O_LM")


@pytest.mark.parametrize("target", TARGETS)
def test_private_branch_is_quiescent_for_somatic_depolarization(target: str) -> None:
    p = user_m5_status(target, -65.0)
    p["E_L"] = -65.0
    dv, coupling = branch_derivatives_per_ms(
        -65.0, -45.0, 0.0, p["h_Na_prox"], p["n_Kd_prox"], p
    )
    # The source-fit active resting current is negligible; critically, the
    # depolarized domain cannot back-drive or load the private branch.
    assert abs(dv) < 0.02
    assert coupling == 0.0


@pytest.mark.parametrize("target", TARGETS)
def test_local_synaptic_drive_couples_out_of_private_branch(target: str) -> None:
    p = user_m5_status(target, -65.0)
    p["E_L"] = -65.0
    dv, coupling = branch_derivatives_per_ms(
        -30.0, -60.0, 2000.0, 1.0, 0.0, p
    )
    assert coupling > 0.0
    assert dv > 0.0


@pytest.mark.parametrize("target", TARGETS)
def test_user_m5_gate_recovery_is_dt_stable(target: str) -> None:
    p = user_m5_status(target)

    def recover(dt: float) -> tuple[float, float]:
        h, n = 0.0, 1.0
        for _ in range(round(100.0 / dt)):
            dh, dn = recovery_derivatives_per_ms(-70.0, h, n, p)
            h += dt * dh
            n += dt * dn
        return h, n

    assert recover(0.05) == pytest.approx(recover(0.025), abs=5e-4)


def test_user_m5_static_port_abi_matches_user_m2() -> None:
    root = Path(__file__).resolve().parents[1] / "nest-gpu/src"
    m2 = (root / "user_m2_kernel.h").read_text()
    m5 = (root / "user_m5_kernel.h").read_text()
    for literal in (
        '{ "g", "g1" }',
        '{ "E_rev", "tau_rise", "tau_decay", "g0", "compartment" }',
    ):
        assert literal in m2
        assert literal in m5
    assert "i_V_m=0, i_V_d, i_V_dist, i_I_adap, i_I_dep" in m5
    assert "if(V_m>=V_th)" in m5
    assert "PushSpike(data.i_node_0_+idx,1.0)" in m5
    assert "E_rev(i)-V_b_prox" in m5
    assert "g_b_prox*fmaxf(0.0f,V_b_prox-V_d)" in m5
