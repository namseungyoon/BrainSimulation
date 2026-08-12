"""Reference equations for the private active branches in ``user_m5``."""
from __future__ import annotations

from typing import Mapping

from ca1.sim.user_m4 import dendritic_currents_pA


def branch_derivatives_per_ms(
    branch_voltage_mV: float,
    domain_voltage_mV: float,
    synaptic_current_pA: float,
    h: float,
    n: float,
    params: Mapping[str, float],
    domain: str = "prox",
) -> tuple[float, float]:
    """Return (dV_b/dt, branch-to-domain current) for one private branch."""
    ina, ik = dendritic_currents_pA(branch_voltage_mV, h, n, params, domain)
    coupling = params[f"g_b_{domain}"] * max(
        0.0, branch_voltage_mV - domain_voltage_mV
    )
    rhs = (
        -params[f"g_leak_b_{domain}"]
        * (branch_voltage_mV - params.get("E_L", -65.0))
        - coupling
        + synaptic_current_pA
        + ina
        + ik
    )
    return rhs / params[f"C_b_{domain}"], coupling
