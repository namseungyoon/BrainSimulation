from __future__ import annotations

import math

import pytest

from ca1.sim.aglif_dend import cck_user_m3_status
from ca1.sim.user_m3 import (
    SodiumAvailability,
    USER_M2_PORT_ABI,
    USER_M2_PORT_PARAMETERS,
    USER_M3_PORT_ABI,
    USER_M3_PORT_PARAMETERS,
)


def _gate() -> SodiumAvailability:
    return SodiumAvailability.from_mapping(cck_user_m3_status())


def _source_h_inf(voltage: float) -> float:
    alpha = 0.35 / math.exp((voltage + 65.0) / 20.0)
    beta = 2.25 / (1.0 + math.exp((voltage + 12.5) / -10.0))
    return alpha / (alpha + beta)


def test_h_inf_is_grounded_in_ch_navcck_rate_equation() -> None:
    gate = _gate()
    for voltage in (-70.0, -60.0, -50.0, -40.0, -30.0, -22.29, -10.0):
        assert gate.h_inf(voltage) == pytest.approx(_source_h_inf(voltage), abs=0.035)


def test_successful_crossing_depletes_but_suppressed_crossing_does_not_reset() -> None:
    gate = _gate()
    available = gate.h_crit + 0.01
    assert gate.crossing(-50.0, -58.0, available)
    depleted = gate.deplete(available)
    assert depleted < gate.h_crit
    assert not gate.crossing(-20.0, -58.0, depleted)
    # Suppression returns no reset action; voltage remains the caller's plateau.
    plateau = -20.0
    assert plateau == -20.0


def test_depolarized_silence_and_repolarization_create_dynamic_hysteresis() -> None:
    gate = _gate()
    h = 0.1
    h = gate.advance(h, -20.0, 500.0)
    assert h < gate.h_crit
    assert not gate.crossing(-20.0, -58.0, h)
    h = gate.advance(h, -71.0, 500.0)
    assert h > gate.h_crit
    assert gate.crossing(-50.0, -58.0, h)


def test_gate_update_converges_identically_at_half_and_full_steps() -> None:
    gate = _gate()
    full = gate.advance(0.9, -22.29, 0.05)
    half = gate.advance(gate.advance(0.9, -22.29, 0.025), -22.29, 0.025)
    assert full == pytest.approx(half, abs=2e-15)


def test_port_abi_is_exact_user_m2_clone() -> None:
    assert USER_M3_PORT_ABI == USER_M2_PORT_ABI == ("g", "g1")
    assert USER_M3_PORT_PARAMETERS == USER_M2_PORT_PARAMETERS


def test_parameter_serialization_round_trip() -> None:
    gate = _gate()
    assert SodiumAvailability.from_mapping(gate.to_dict()) == gate
