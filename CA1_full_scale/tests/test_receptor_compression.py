from __future__ import annotations

import pytest

from ca1.analysis.receptor_compression_rank import rank_receptor_compression_strategies
from ca1.analysis.receptor_compression_types import (
    CompressionReport,
    KernelKey,
    ReceptorCompressionInputError,
)
from ca1.analysis.receptor_compression_assignments import assignments_for_strategy
from ca1.analysis.receptor_compression_inputs import build_compression_context
from ca1.config import ConfigDict, build_network_spec
from ca1.params.receptor_ports import representative_port_for_compartment


@pytest.fixture(scope="module")
def compression_report() -> CompressionReport:
    return rank_receptor_compression_strategies()


def test_receptor_compression_ranks_information_weighted_candidates(
    compression_report: CompressionReport,
) -> None:
    # Given: the syndata120 compartment-aware 39-item response library.
    report = compression_report
    by_name = {score.strategy: score for score in report.scores}

    # When: candidate 20-port codebooks are ranked by utility-weighted loss.
    ranked_names = [score.strategy for score in report.scores]

    # Then: all candidates stay within budget, and response-space optimization
    # beats the current hand-selected safe20 table on this reconstruction metric.
    assert report.n_original_items == 39
    assert report.n_budget == 20
    assert all(score.n_ports <= 20 for score in report.scores)
    assert ranked_names[0] == "utility_sparse2"
    assert (
        by_name["utility_weighted_medoids"].utility_loss
        < by_name["current_safe20"].utility_loss
    )


def test_sparse2_keeps_loss_below_utility_hard_assignment(
    compression_report: CompressionReport,
) -> None:
    # Given: hard utility-weighted medoids and their sparse two-port relaxation.
    report = compression_report
    by_name = {score.strategy: score for score in report.scores}

    # When: the sparse relaxation is evaluated with an efficiency penalty.
    sparse = by_name["utility_sparse2"]
    hard = by_name["utility_weighted_medoids"]

    # Then: sparse2 does not worsen reconstruction and exposes its extra cost.
    assert sparse.utility_loss <= hard.utility_loss
    assert sparse.effective_ports_per_item >= 1.0
    assert sparse.rank_objective <= hard.rank_objective


def test_receptor_compression_rejects_non_20_port_budget() -> None:
    # Given: the analysis is specifically a 39-to-20 receptor-port comparison.
    # When / Then: lower budgets are rejected instead of returning over-budget rows.
    with pytest.raises(ReceptorCompressionInputError, match="n_budget"):
        blocked_report = rank_receptor_compression_strategies(n_budget=5)
        assert blocked_report.n_budget == 5


def test_utility_weighted_medoids_runtime_strategy_uses_ranked_ports() -> None:
    # Given: the top hard one-port candidate is selected in a compartment-aware spec.
    spec = build_network_spec({
        "name": "utility-weighted-runtime",
        "cellnumbers_index": 101,
        "conndata_index": 430,
        "conndata_count_mode": "per_cell",
        "syndata_variant": 120,
        "compartment_aware_synapses": True,
        "receptor_port_strategy": "utility_weighted_medoids",
    })

    # When: the receptor table is resolved through the normal runtime loader.
    names = frozenset(spec.receptors.names)
    expected = frozenset(
        _runtime_name(key)
        for key in (
            KernelKey("AMPA_fast", 0.0, 0.07, 0.2, "dend"),
            KernelKey("AMPA_fast", 0.0, 0.1, 1.5, "dend"),
            KernelKey("AMPA_fast", 0.0, 0.11, 0.25, "dend"),
            KernelKey("AMPA_fast", 0.0, 0.3, 0.6, "dend"),
            KernelKey("AMPA_fast", 0.0, 0.5, 3.0, "dend"),
            KernelKey("AMPA_fast", 0.0, 2.0, 6.3, "dend"),
            KernelKey("AMPA_slow", 0.0, 0.5, 3.0, "dend"),
            KernelKey("AMPA_slow", 0.0, 2.0, 6.3, "dend"),
            KernelKey("GABA_A_fast", -60.0, 0.08, 4.8, "soma"),
            KernelKey("GABA_A_fast", -60.0, 0.18, 0.45, "soma"),
            KernelKey("GABA_A_fast", -60.0, 0.28, 8.4, "soma"),
            KernelKey("GABA_A_slow", -60.0, 0.11, 9.7, "dend"),
            KernelKey("GABA_A_slow", -60.0, 0.25, 7.5, "dend"),
            KernelKey("GABA_A_slow", -60.0, 0.287, 2.67, "dend"),
            KernelKey("GABA_A_slow", -60.0, 0.432, 4.49, "dend"),
            KernelKey("GABA_A_slow", -60.0, 0.432, 4.49, "soma"),
            KernelKey("GABA_A_slow", -60.0, 0.728, 20.2, "dend"),
            KernelKey("GABA_A_slow", -60.0, 1.0, 8.0, "soma"),
            KernelKey("GABA_A_slow", -60.0, 2.9, 3.1, "dend"),
            KernelKey("GABA_B", -90.0, 180.0, 200.0, "dend"),
        )
    )

    # Then: runtime uses the ranked hard 20-port codebook with distinct provenance.
    assert spec.receptors.n_ports() == 20
    assert names == expected
    assert "20port-utility_weighted_medoids;sha256=" in spec.receptor_provenance


def test_utility_weighted_runtime_assignments_match_ranked_response_assignments(
) -> None:
    # Given: the response-space ranking selected hard utility medoid assignments.
    context = build_compression_context(120)
    ranked = assignments_for_strategy("utility_weighted_medoids", context, 120, 20)

    # When: each source kinetics is resolved through the runtime strategy.
    runtime = {
        source: KernelKey(
            *representative_port_for_compartment(
                source.port_key(),
                "utility_weighted_medoids",
                source.compartment,
            ),
            source.compartment,
        )
        for source in ranked
    }

    # Then: runtime preserves the exact response-loss assignment semantics.
    assert runtime == {source: decoded[0] for source, decoded in ranked.items()}


def test_utility_weighted_medoids_requires_compartment_aware_synapses() -> None:
    # Given: the utility-weighted medoid table is a 39 typed+compartment model.
    config: ConfigDict = {
        "name": "utility-weighted-without-compartments",
        "syndata_variant": 120,
        "receptor_port_strategy": "utility_weighted_medoids",
    }

    # When / Then: non-compartment-aware use is rejected instead of silently falling back.
    with pytest.raises(ValueError, match="requires compartment_aware_synapses=True"):
        _ = build_network_spec(config)


def _runtime_name(key: KernelKey) -> str:
    base = (
        f"{key.receptor}__e{_float_token(key.e_rev)}"
        f"__tr{_float_token(key.tau_rise)}__td{_float_token(key.tau_decay)}"
    )
    return f"{base}__{key.compartment}"


def _float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")
