"""Tests for ca1.types contracts: NetworkSpec and ReceptorConfig."""

from __future__ import annotations

import pytest

from ca1.types import (
    parse_recurrent_topology,
    CellType,
    NetworkSpec,
    NeuronParams,
    ReceptorConfig,
    RECEPTOR_PORTS,
    parse_neuron_model,
)


# ---------------------------------------------------------------------------
# Minimal NeuronParams fixture (Pyramidal-like values)
# ---------------------------------------------------------------------------

def _pyr_params() -> NeuronParams:
    return NeuronParams(
        C_m=200.0, g_L=10.0, E_L=-70.0, V_th=-48.0,
        V_reset=-65.0, Delta_T=2.0, a=4.0, b=80.5,
        tau_w=144.0, t_ref=2.0, V_peak=0.0, I_e=0.0,
        fit_provenance="analytic",
    )


def _int_params() -> NeuronParams:
    return NeuronParams(
        C_m=134.4, g_L=19.2, E_L=-65.0, V_th=-48.4,
        V_reset=-60.0, Delta_T=1.0, a=0.0, b=0.0,
        tau_w=40.0, t_ref=2.0, V_peak=0.0, I_e=0.0,
        fit_provenance="analytic",
    )


# ---------------------------------------------------------------------------
# ReceptorConfig
# ---------------------------------------------------------------------------

class TestReceptorConfig:
    def test_default_names_match_receptor_ports_constant(self) -> None:
        rc = ReceptorConfig()
        assert rc.names == RECEPTOR_PORTS

    def test_n_ports(self) -> None:
        rc = ReceptorConfig()
        assert rc.n_ports() == 5

    def test_port_index_ampa_fast(self) -> None:
        rc = ReceptorConfig()
        assert rc.port_index("AMPA_fast") == 0

    def test_port_index_gaba_a_fast(self) -> None:
        rc = ReceptorConfig()
        assert rc.port_index("GABA_A_fast") == 2

    def test_port_index_gaba_a_slow(self) -> None:
        rc = ReceptorConfig()
        assert rc.port_index("GABA_A_slow") == 3

    def test_port_index_gaba_b(self) -> None:
        rc = ReceptorConfig()
        assert rc.port_index("GABA_B") == 4

    def test_port_index_ampa_slow(self) -> None:
        rc = ReceptorConfig()
        assert rc.port_index("AMPA_slow") == 1

    def test_port_index_unknown_raises(self) -> None:
        rc = ReceptorConfig()
        with pytest.raises(ValueError):
            rc.port_index("NMDA")

    def test_inhibitory_e_rev_below_rest(self) -> None:
        """All inhibitory ports must have E_rev well below resting potential."""
        rc = ReceptorConfig()
        for name, e_rev in zip(rc.names, rc.E_rev):
            if "GABA" in name:
                assert e_rev < -55.0, (
                    f"{name}: E_rev={e_rev} not below -55 mV -- "
                    "inhibition must use negative-E_rev ports with POSITIVE weights"
                )

    def test_frozen(self) -> None:
        rc = ReceptorConfig()
        with pytest.raises((AttributeError, TypeError)):
            setattr(rc, "names", ("AMPA_fast",))


# ---------------------------------------------------------------------------
# NetworkSpec
# ---------------------------------------------------------------------------

def _tiny_spec(scale: float = 1.0) -> NetworkSpec:
    """Minimal 2-cell-type NetworkSpec for unit testing."""
    pyr = CellType(
        name="pyramidalcell",
        count=311500,
        layers=("SP",),
        params=_pyr_params(),
    )
    pvb = CellType(
        name="pvbasketcell",
        count=5530,
        layers=("SP",),
        params=_int_params(),
    )
    return NetworkSpec(
        name="test_tiny",
        cell_types={"pyramidalcell": pyr, "pvbasketcell": pvb},
        projections=[],
        afferents=[],
        scale=scale,
        seed=42,
    )


class TestNetworkSpecScaledCounts:
    def test_full_scale_counts_equal_bezaire(self) -> None:
        spec = _tiny_spec(scale=1.0)
        counts = spec.scaled_counts()
        assert counts["pyramidalcell"] == 311500
        assert counts["pvbasketcell"] == 5530

    def test_half_scale_halves_counts(self) -> None:
        spec = _tiny_spec(scale=0.5)
        counts = spec.scaled_counts()
        assert counts["pyramidalcell"] == 155750
        assert counts["pvbasketcell"] == 2765

    def test_small_scale_minimum_one_cell(self) -> None:
        spec = _tiny_spec(scale=1e-6)
        counts = spec.scaled_counts()
        for v in counts.values():
            assert v >= 1, "scaled_counts must clamp to at least 1 cell per type"

    def test_total_cells_sum(self) -> None:
        spec = _tiny_spec(scale=1.0)
        assert spec.total_cells() == 311500 + 5530

    def test_total_cells_scaled(self) -> None:
        spec = _tiny_spec(scale=0.02)
        expected = sum(
            max(1, int(round(c * 0.02)))
            for c in [311500, 5530]
        )
        assert spec.total_cells() == expected


class TestNetworkSpecImmutableByConvention:
    """NetworkSpec is a plain dataclass (not frozen) but projections list
    should not be mutated externally after construction."""

    def test_spec_construction_succeeds(self) -> None:
        spec = _tiny_spec()
        assert spec.name == "test_tiny"
        assert len(spec.cell_types) == 2

    def test_default_weight_compensation_is_one(self) -> None:
        spec = _tiny_spec()
        assert spec.weight_compensation == 1.0

    def test_default_receptors_has_five_ports(self) -> None:
        spec = _tiny_spec()
        assert spec.receptors.n_ports() == 5

    def test_default_neuron_model_is_adex(self) -> None:
        spec = _tiny_spec()
        assert spec.neuron_model == "aeif_cond_beta_multisynapse"

    def test_aglif_neuron_model_is_supported(self) -> None:
        spec = _tiny_spec()
        spec.neuron_model = "aglif_cond_beta"
        spec.__post_init__()

        assert spec.neuron_model == "aglif_cond_beta"

    def test_aglif_dend_neuron_model_is_supported(self) -> None:
        spec = _tiny_spec()
        spec.neuron_model = "aglif_dend_cond_beta"
        spec.__post_init__()

        assert spec.neuron_model == "aglif_dend_cond_beta"

    def test_unknown_neuron_model_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported neuron_model"):
            parse_neuron_model("iaf_cond_alpha")

    def test_modeldb_fastconn_recurrent_topology_is_supported(self) -> None:
        assert parse_recurrent_topology("modeldb_fastconn_binned") == (
            "modeldb_fastconn_binned"
        )

    def test_modeldb_fastconn_gaussian_recurrent_topology_is_supported(self) -> None:
        assert parse_recurrent_topology("modeldb_fastconn_gaussian_binned") == (
            "modeldb_fastconn_gaussian_binned"
        )

    def test_modeldb_fastconn_3d_gaussian_topology_is_supported(self) -> None:
        assert parse_recurrent_topology("modeldb_fastconn_3d_gaussian") == (
            "modeldb_fastconn_3d_gaussian"
        )

    def test_unknown_recurrent_topology_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported recurrent_topology"):
            parse_recurrent_topology("silent_fallback")
