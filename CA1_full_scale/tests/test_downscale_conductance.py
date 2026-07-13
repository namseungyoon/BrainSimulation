"""Tests for ca1.build.downscale modes.

Verified properties
-------------------
1. mean-field mode: per-synapse weight is invariant (J_actual = J_full), where
   J_actual = Projection.weight_nS * spec.weight_compensation.
   K is scaled proportionally; weight_compensation = K_scaled/K_full ~ scale.
2. p-preserve mode: emits a UserWarning (DEBUG ONLY label).
3. preserve-indegree mode: in-degree K stays at the full-scale value (or is
   clamped to N_pre_scaled if the population is very small).
"""

from __future__ import annotations

import warnings

import pytest

# Skip whole module if the downscale module is not yet implemented.
downscale = pytest.importorskip("ca1.build.downscale")

from ca1.types import (  # noqa: E402
    CellType,
    NetworkSpec,
    NeuronParams,
    Projection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _params() -> NeuronParams:
    return NeuronParams(
        C_m=200.0, g_L=10.0, E_L=-70.0, V_th=-48.0,
        V_reset=-65.0, Delta_T=2.0, a=4.0, b=80.5,
        tau_w=144.0, t_ref=2.0,
    )


def _full_spec() -> NetworkSpec:
    """Minimal two-population spec at scale=1.0 with large enough N for mean-field."""
    pyr = CellType(name="pyramidalcell", count=311500, layers=("SP",), params=_params())
    pvb = CellType(name="pvbasketcell", count=5530, layers=("SP",), params=_params())

    proj = Projection(
        pre="pvbasketcell",
        post="pyramidalcell",
        indegree=25.0,
        synapses_per_connection=2,
        weight_nS=0.5,
        receptor="GABA_A_fast",
        delay_ms=1.0,
    )
    return NetworkSpec(
        name="test_full",
        cell_types={"pyramidalcell": pyr, "pvbasketcell": pvb},
        projections=[proj],
        afferents=[],
        scale=1.0,
        seed=12345,
    )


# ---------------------------------------------------------------------------
# Helper: per-synapse actual weight = J_nS * weight_compensation
# (This is what the backend applies; see NestBackend build())
# ---------------------------------------------------------------------------

def _actual_weight_per_synapse(spec: NetworkSpec, proj_index: int = 0) -> float:
    """weight_nS * weight_compensation -- what the backend applies per synapse."""
    p = spec.projections[proj_index]
    return p.weight_nS * spec.weight_compensation


def _total_g(spec: NetworkSpec, proj_index: int = 0) -> float:
    """K * synapses_per_connection * J_actual -- total conductance per post cell."""
    p = spec.projections[proj_index]
    return p.indegree * p.synapses_per_connection * _actual_weight_per_synapse(spec, proj_index)


# ---------------------------------------------------------------------------
# mean-field mode
# ---------------------------------------------------------------------------

class TestMeanFieldMode:
    """In mean-field mode:
       - K_scaled = K_full * scale (proportional to N_scaled / N_full)
       - weight_nS compensated so J_actual = weight_nS * weight_comp = J_full
       - weight_compensation = N_scaled / N_full ~ scale
    """

    @pytest.mark.parametrize("scale", [0.5, 0.1, 0.02])
    def test_actual_weight_per_synapse_invariant(self, scale: float) -> None:
        """J_actual = weight_nS * weight_comp must equal full-scale J_full."""
        full = _full_spec()
        j_full = _actual_weight_per_synapse(full)

        scaled = downscale.downscale_spec(full, scale=scale, mode="mean-field")
        j_scaled = _actual_weight_per_synapse(scaled)

        # Allow 1% relative tolerance for floating-point rounding
        assert abs(j_scaled - j_full) / max(j_full, 1e-12) < 0.01, (
            f"mean-field: J_actual changed from {j_full:.6f} to {j_scaled:.6f} "
            f"at scale={scale}. Synaptic strength should be invariant."
        )

    @pytest.mark.parametrize("scale", [0.5, 0.1])
    def test_indegree_scaled_proportionally(self, scale: float) -> None:
        """K_scaled should be proportional to scale * K_full (when not floor-clamped)."""
        full = _full_spec()
        k_full = full.projections[0].indegree

        scaled = downscale.downscale_spec(full, scale=scale, mode="mean-field")
        k_scaled = scaled.projections[0].indegree

        # K_scaled ~ K_full * scale; allow rounding tolerance.
        # Note: at very small scales K may be clamped to 1 (floor), so we only
        # test scales where K_full * scale >= 2 (well above the floor).
        expected = k_full * scale
        assert abs(k_scaled - expected) / max(expected, 1.0) < 0.15, (
            f"mean-field: K_scaled={k_scaled:.2f} not close to "
            f"K_full*scale={expected:.2f} at scale={scale}"
        )

    def test_indegree_at_very_small_scale_at_least_one(self) -> None:
        """At very small scale K_scaled may be floor-clamped to 1 (not zero)."""
        full = _full_spec()
        scaled = downscale.downscale_spec(full, scale=0.02, mode="mean-field")
        assert scaled.projections[0].indegree >= 1

    def test_weight_compensation_less_than_one_at_subscale(self) -> None:
        """weight_compensation = N_scaled / N_full < 1.0 when scale < 1.0."""
        full = _full_spec()
        scaled = downscale.downscale_spec(full, scale=0.02, mode="mean-field")
        # weight_comp encodes N_scaled/N_full which must be < 1
        assert scaled.weight_compensation < 1.0

    def test_weight_compensation_near_scale(self) -> None:
        """weight_compensation should be close to scale (within 10%)."""
        full = _full_spec()
        scale = 0.02
        scaled = downscale.downscale_spec(full, scale=scale, mode="mean-field")
        assert abs(scaled.weight_compensation - scale) / scale < 0.10, (
            f"weight_compensation={scaled.weight_compensation:.4f} deviates "
            f"more than 10% from scale={scale}"
        )

    def test_scaled_spec_has_correct_scale_attr(self) -> None:
        full = _full_spec()
        scaled = downscale.downscale_spec(full, scale=0.02, mode="mean-field")
        assert abs(scaled.scale - 0.02) < 1e-9


# ---------------------------------------------------------------------------
# p-preserve mode (DEBUG ONLY)
# ---------------------------------------------------------------------------

class TestPPreserveMode:
    def test_emits_warning(self) -> None:
        full = _full_spec()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            downscale.downscale_spec(full, scale=0.02, mode="p-preserve")

        assert len(caught) >= 1, "p-preserve mode must emit at least one warning"
        messages = " ".join(str(w.message) for w in caught)
        # Warning must mention debug / p-preserve context
        assert any(
            kw in messages.lower()
            for kw in ("debug", "p-preserve", "warning", "deprecated", "silence")
        ), f"Warning text did not mention debug context: {messages!r}"

    def test_emits_warning_category(self) -> None:
        """Warning should be UserWarning or subclass."""
        full = _full_spec()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            downscale.downscale_spec(full, scale=0.02, mode="p-preserve")
        categories = [w.category for w in caught]
        assert any(issubclass(c, (UserWarning, DeprecationWarning)) for c in categories)


# ---------------------------------------------------------------------------
# preserve-indegree mode (default)
# ---------------------------------------------------------------------------

class TestPreserveIndegreeMode:
    """In preserve-indegree mode K is held at full-scale value (or clamped to N_pre)."""

    @pytest.mark.parametrize("scale", [0.5, 0.1, 0.02])
    def test_indegree_unchanged_when_above_threshold(self, scale: float) -> None:
        """K should stay at 25.0 (N_pre_scaled is large enough at these scales)."""
        full = _full_spec()
        original_k = full.projections[0].indegree

        scaled = downscale.downscale_spec(full, scale=scale, mode="preserve-indegree")
        scaled_k = scaled.projections[0].indegree

        assert scaled_k == original_k, (
            f"preserve-indegree: K changed from {original_k} to {scaled_k}"
        )

    def test_is_default_mode(self) -> None:
        """Calling without mode= should behave like preserve-indegree."""
        full = _full_spec()
        original_k = full.projections[0].indegree
        scaled = downscale.downscale_spec(full, scale=0.02)
        assert scaled.projections[0].indegree == original_k

    def test_weight_compensation_is_one(self) -> None:
        """preserve-indegree does not change weights: weight_compensation = 1.0."""
        full = _full_spec()
        scaled = downscale.downscale_spec(full, scale=0.02, mode="preserve-indegree")
        assert scaled.weight_compensation == 1.0

    def test_cell_counts_scaled(self) -> None:
        full = _full_spec()
        scaled = downscale.downscale_spec(full, scale=0.02, mode="preserve-indegree")
        counts = scaled.scaled_counts()
        assert counts["pyramidalcell"] < full.scaled_counts()["pyramidalcell"]

    def test_invalid_mode_raises(self) -> None:
        full = _full_spec()
        with pytest.raises((ValueError, KeyError)):
            downscale.downscale_spec(full, scale=0.02, mode="not-a-real-mode")
