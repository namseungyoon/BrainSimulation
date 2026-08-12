"""CPU reference primitives for the candidate-only NEST-GPU ``user_m3`` gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

USER_M2_PORT_ABI = ("g", "g1")
USER_M3_PORT_ABI = USER_M2_PORT_ABI
USER_M2_PORT_PARAMETERS = ("E_rev", "tau_rise", "tau_decay", "g0", "compartment")
USER_M3_PORT_PARAMETERS = USER_M2_PORT_PARAMETERS


@dataclass(frozen=True)
class SodiumAvailability:
    V_h_half: float
    k_h: float
    tau_h: float
    delta_h: float
    h_crit: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("sodium-availability parameters must be finite")
        if self.k_h <= 0.0 or self.tau_h <= 0.0:
            raise ValueError("k_h and tau_h must be positive")
        if not 0.0 <= self.delta_h <= 1.0 or not 0.0 <= self.h_crit <= 1.0:
            raise ValueError("delta_h and h_crit must lie in [0, 1]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> SodiumAvailability:
        return cls(**{name: float(values[name]) for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def h_inf(self, voltage_mV: float) -> float:
        exponent = max(-80.0, min(80.0, (voltage_mV - self.V_h_half) / self.k_h))
        return 1.0 / (1.0 + math.exp(exponent))

    def advance(self, h: float, voltage_mV: float, dt_ms: float) -> float:
        """Exact constant-voltage gate update, used as a timestep oracle."""
        if dt_ms < 0.0:
            raise ValueError("dt_ms must be nonnegative")
        target = self.h_inf(voltage_mV)
        updated = target + (min(1.0, max(0.0, h)) - target) * math.exp(
            -dt_ms / self.tau_h
        )
        return min(1.0, max(0.0, updated))

    def crossing(self, voltage_mV: float, threshold_mV: float, h: float) -> bool:
        return voltage_mV >= threshold_mV and h > self.h_crit

    def deplete(self, h: float) -> float:
        return max(0.0, min(1.0, h) - self.delta_h)
