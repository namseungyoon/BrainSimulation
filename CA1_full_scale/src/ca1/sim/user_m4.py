"""Reference equations for the candidate active-dendrite ``user_m4`` model."""
from __future__ import annotations

import math
from typing import Mapping


def activation_inf(voltage_mV: float, half_mV: float, slope_mV: float) -> float:
    z = max(-80.0, min(80.0, -(voltage_mV-half_mV)/slope_mV))
    return 1.0 / (1.0 + math.exp(z))


def inactivation_inf(voltage_mV: float, half_mV: float, slope_mV: float) -> float:
    z = max(-80.0, min(80.0, (voltage_mV-half_mV)/slope_mV))
    return 1.0 / (1.0 + math.exp(z))


def dendritic_currents_pA(
    voltage_mV: float, h: float, n: float, params: Mapping[str, float],
    domain: str = "prox",
) -> tuple[float, float]:
    m = activation_inf(voltage_mV, params["Vm_half"], params["km"])
    gna = params[f"gbar_Na_{domain}"]
    gk = params[f"gbar_Kd_{domain}"]
    return (gna*m**3*h*(params["E_Na"]-voltage_mV),
            gk*n**4*(params["E_K"]-voltage_mV))


def recovery_derivatives_per_ms(
    voltage_mV: float, h: float, n: float, params: Mapping[str, float],
) -> tuple[float, float]:
    return ((inactivation_inf(voltage_mV, params["Vh_half"], params["kh"])-h)
            / params["tau_h"],
            (activation_inf(voltage_mV, params["Vn_half"], params["kn"])-n)
            / params["tau_n"])
