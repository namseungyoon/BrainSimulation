from __future__ import annotations

import pytest

from ca1.analysis.receptor_compression_experiments import (
    rank_experimental_receptor_compression_strategies,
)
from ca1.analysis.receptor_compression_types import ReceptorCompressionInputError


def test_experimental_codebook_report_contains_two_logic_level_candidates() -> None:
    # Given: the syndata120 39-kernel receptor response library.
    # When: logic-level experimental compression candidates are ranked.
    report = rank_experimental_receptor_compression_strategies()
    by_name = {score.strategy: score for score in report.scores}

    # Then: both requested designs are evaluated without changing runtime ports.
    assert report.n_original_items == 39
    assert report.n_budget == 20
    assert set(by_name) == {"binary4_cdm", "event_select2_mix"}
    assert by_name["binary4_cdm"].code_bits == 4
    assert by_name["binary4_cdm"].max_codewords_per_group <= 16
    assert by_name["event_select2_mix"].effective_inputs_per_item >= 1.0


def test_event_select2_improves_over_binary4_cdm_on_reconstruction() -> None:
    # Given: CDMA-like bit masks and event-level two-input selection are both
    # evaluated only as reconstruction logic, independent of NEST-GPU support.
    report = rank_experimental_receptor_compression_strategies()
    by_name = {score.strategy: score for score in report.scores}

    # When / Then: an explicit second select channel should be at least as
    # expressive as a 4-bit equal-chip superposition decoder on utility loss.
    assert (
        by_name["event_select2_mix"].utility_loss
        <= by_name["binary4_cdm"].utility_loss
    )


def test_experimental_compression_still_rejects_non_20_budget() -> None:
    # Given: these experiments are anchored to the same 39-to-20 comparison.
    # When / Then: a different budget is not silently accepted.
    with pytest.raises(ReceptorCompressionInputError, match="n_budget"):
        _ = rank_experimental_receptor_compression_strategies(n_budget=16)
