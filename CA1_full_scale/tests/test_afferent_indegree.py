"""Tests for legacy JSON afferent synapse-budget preservation.

The critical invariant for the legacy ``connectivity.json`` path:
Bezaire's per-postsynaptic-cell
synapse budget (total_synapses / N_post) is kept VERBATIM in Afferent.synapses_per_cell
and must NOT be capped to any arbitrary ratio (e.g. N_source / N_post, or N_source alone).

The legacy motivating example: ECIII -> Neurogliaform has synapses_per_cell=58240, which
is total_synapses(208500000) / N_post(3580). This is a valid dense synapse budget
reflecting the fact that ECIII terminates densely in SLM. The value must be preserved
verbatim when explicitly using ``connectivity.json``. Final-tier full-scale runs use
raw ModelDB ``ConnData=430`` / ``per_cell`` instead.

Note on n_source: n_source = ECIII population size (250000). synapses_per_cell (58240)
happens to be less than n_source in this case, BUT it must NOT be capped to
n_source / N_post or any other derived ratio. The verbatim value is the invariant.
"""

from __future__ import annotations

import pytest

from ca1.types import Afferent

# Skip whole module if ca1.params.synapses is not yet implemented.
synapses_mod = pytest.importorskip("ca1.params.synapses")


# ---------------------------------------------------------------------------
# Legacy connectivity.json reference values
# ---------------------------------------------------------------------------
# ECIII_to_Neurogliaform: total_connections=208500000, N_post=3580
# synapses_per_cell = 208500000 / 3580 = 58240 (verbatim)
_ECIII_NGF_EXPECTED_SYNAPSES = 58240.0
_ECIII_NGF_N_SOURCE = 250000  # ECIII population size

# Minimum large value to detect any capping (58240 >> 10, >> 100, etc.)
_CAP_DETECTION_THRESHOLD = 50000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_afferents(rate_hz: float = 0.65) -> list[Afferent]:
    return synapses_mod.load_afferents(rate_hz=rate_hz)


def _afferent_by_name(afferents: list[Afferent], name: str) -> Afferent:
    matches = [a for a in afferents if a.name == name]
    assert matches, f"No afferent named {name!r}. Available: {[a.name for a in afferents]}"
    return matches[0]


# ---------------------------------------------------------------------------
# Core invariant: synapses_per_cell kept verbatim
# ---------------------------------------------------------------------------

class TestAfferentSynapsesBudgetPreserved:
    def test_eciii_to_neurogliaform_large_value_preserved(self) -> None:
        """ECIII->Neurogliaform synapses_per_cell must be ~58240 (verbatim from conndata).

        This is total_connections(208500000) / N_post(3580) = 58240.
        Any capping would reduce this to a tiny fraction, silencing the network.
        """
        afferents = _load_afferents()
        aff = _afferent_by_name(afferents, "ECIII_to_Neurogliaform")

        assert aff.synapses_per_cell >= _CAP_DETECTION_THRESHOLD, (
            f"ECIII->Neurogliaform synapses_per_cell={aff.synapses_per_cell} "
            f"was capped below the threshold {_CAP_DETECTION_THRESHOLD}. "
            f"Expected ~{_ECIII_NGF_EXPECTED_SYNAPSES} (verbatim from connectivity.json). "
            f"Capping silences the network."
        )

    def test_eciii_to_neurogliaform_exact_value(self) -> None:
        """synapses_per_cell must equal the connectivity.json value (58240) exactly."""
        afferents = _load_afferents()
        aff = _afferent_by_name(afferents, "ECIII_to_Neurogliaform")
        assert aff.synapses_per_cell == pytest.approx(_ECIII_NGF_EXPECTED_SYNAPSES, rel=0.01), (
            f"Got {aff.synapses_per_cell}; expected {_ECIII_NGF_EXPECTED_SYNAPSES}"
        )

    def test_synapses_per_cell_not_capped_to_naive_ratio(self) -> None:
        """synapses_per_cell must NOT equal n_source / N_post (naive wrong cap).

        For ECIII->Neurogliaform: n_source(250000) / n_post(3580) ~ 69.8.
        The correct value is 58240, not 69.8. If we see ~70, capping occurred.
        """
        afferents = _load_afferents()
        aff = _afferent_by_name(afferents, "ECIII_to_Neurogliaform")

        # Naive cap would give ~70. Correct value is 58240.
        naive_cap = aff.n_source / 3580  # ~69.8
        assert aff.synapses_per_cell > naive_cap * 100, (
            f"synapses_per_cell={aff.synapses_per_cell:.1f} appears to have been "
            f"capped to n_source/N_post={naive_cap:.1f}. "
            f"The verbatim conndata value is {_ECIII_NGF_EXPECTED_SYNAPSES}."
        )

    def test_eciii_n_source_is_ec_population(self) -> None:
        """n_source for ECIII afferents should be the ECIII population size."""
        afferents = _load_afferents()
        ec_afferents = [a for a in afferents if "ECIII" in a.name]
        assert ec_afferents, "No ECIII afferents found"
        for aff in ec_afferents:
            assert aff.n_source == _ECIII_NGF_N_SOURCE, (
                f"{aff.name}: n_source={aff.n_source}, expected {_ECIII_NGF_N_SOURCE}"
            )


# ---------------------------------------------------------------------------
# Afferent rate
# ---------------------------------------------------------------------------

class TestAfferentRate:
    def test_default_rate_is_0_65_hz(self) -> None:
        afferents = _load_afferents(rate_hz=0.65)
        for a in afferents:
            assert a.rate_hz == pytest.approx(0.65), (
                f"Afferent {a.name} has rate_hz={a.rate_hz}, expected 0.65"
            )

    def test_custom_rate_applied(self) -> None:
        afferents = _load_afferents(rate_hz=1.0)
        for a in afferents:
            assert a.rate_hz == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Afferent AMPA receptor routing
# ---------------------------------------------------------------------------

class TestAfferentReceptor:
    def test_afferents_use_ampa_receptor(self) -> None:
        """All external afferents are excitatory (AMPA)."""
        afferents = _load_afferents()
        for a in afferents:
            assert "AMPA" in a.receptor, (
                f"Afferent {a.name} uses receptor {a.receptor!r}; "
                "expected AMPA for excitatory external drive"
            )

    def test_afferents_have_positive_weight(self) -> None:
        """Weights must be positive (inhibition uses port routing, not negative weights)."""
        afferents = _load_afferents()
        for a in afferents:
            assert a.weight_nS > 0, (
                f"Afferent {a.name} has non-positive weight {a.weight_nS}"
            )


# ---------------------------------------------------------------------------
# Coverage: CA3 afferents also present
# ---------------------------------------------------------------------------

class TestCA3AfferentsPresent:
    def test_ca3_afferents_exist(self) -> None:
        afferents = _load_afferents()
        ca3 = [a for a in afferents if "CA3" in a.name]
        assert ca3, "No CA3 afferents found; expected Schaffer collateral inputs"

    def test_eciii_afferents_exist(self) -> None:
        afferents = _load_afferents()
        ec = [a for a in afferents if "ECIII" in a.name]
        assert ec, "No ECIII afferents found; expected perforant-path inputs"

    def test_twelve_afferents_total(self) -> None:
        """Bezaire conndata has exactly 12 afferent entries (5 ECIII + 7 CA3)."""
        afferents = _load_afferents()
        assert len(afferents) == 12, (
            f"Expected 12 afferents, got {len(afferents)}: {[a.name for a in afferents]}"
        )
