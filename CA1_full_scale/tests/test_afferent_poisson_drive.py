from __future__ import annotations

import pytest

from ca1.types import Afferent


def test_afferent_poisson_drive_keeps_synapse_budget_in_rate() -> None:
    from ca1.sim.afferents import afferent_poisson_drive

    afferent = Afferent(
        name="ECIII_to_Test",
        post="Test",
        n_source=250_000,
        synapses_per_cell=100.0,
        weight_nS=0.2,
        rate_hz=0.65,
    )

    drive = afferent_poisson_drive(afferent)

    assert drive.rate_hz == pytest.approx(65.0)
    assert drive.weight_nS == pytest.approx(0.2)
