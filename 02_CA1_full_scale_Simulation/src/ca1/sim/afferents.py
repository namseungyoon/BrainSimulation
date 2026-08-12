from __future__ import annotations

from dataclasses import dataclass

from ca1.sim.weights import nonnegative_weight_nS
from ca1.types import Afferent


@dataclass(frozen=True, slots=True)
class PoissonDrive:
    rate_hz: float
    weight_nS: float


def afferent_poisson_drive(afferent: Afferent) -> PoissonDrive:
    return PoissonDrive(
        rate_hz=afferent.rate_hz * afferent.synapses_per_cell,
        weight_nS=nonnegative_weight_nS(
            afferent.weight_nS,
            label=f"afferent {afferent.name}",
        ),
    )
