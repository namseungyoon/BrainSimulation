from __future__ import annotations

import numpy as np
import numpy.testing as npt

from ca1.sim.gpu_backend import _spike_slot_batches


def _sample(seed: int) -> tuple[np.ndarray, list[tuple[int, np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(6.5, size=10_000)
    batches = [
        (start, offsets.copy(), slots.copy())
        for start, offsets, slots in _spike_slot_batches(
            counts,
            slot_count=25_000,
            rng=rng,
            max_spikes=10_000,
        )
    ]
    return counts, batches


def test_vectorized_spike_slot_generation_is_deterministic_and_distinct() -> None:
    counts_a, batches_a = _sample(12345)
    counts_b, batches_b = _sample(12345)

    npt.assert_array_equal(counts_a, counts_b)
    assert len(batches_a) == len(batches_b)
    for (start_a, offsets_a, slots_a), (start_b, offsets_b, slots_b) in zip(
        batches_a, batches_b, strict=True
    ):
        assert start_a == start_b
        npt.assert_array_equal(offsets_a, offsets_b)
        npt.assert_array_equal(slots_a, slots_b)
        assert np.all((slots_a >= 1) & (slots_a < 25_000))
        for begin, end in zip(offsets_a[:-1], offsets_a[1:], strict=True):
            source_slots = slots_a[begin:end]
            assert np.all(source_slots[1:] > source_slots[:-1])


def test_vectorized_spike_slot_generation_has_poisson_counts() -> None:
    expected_rate = 6.5
    rng = np.random.default_rng(9876)
    counts = rng.poisson(expected_rate, size=250_000)

    # At this sample size a 0.05-Hz tolerance is over 9 standard errors.
    assert abs(float(counts.mean()) - expected_rate) < 0.05


def test_dense_spike_train_fallback_keeps_slots_distinct() -> None:
    rng = np.random.default_rng(2468)
    counts = np.array([280, 275], dtype=np.int64)
    batches = list(
        _spike_slot_batches(
            counts,
            slot_count=300,
            rng=rng,
            max_spikes=1_000,
        )
    )

    assert len(batches) == 1
    _, offsets, slots = batches[0]
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        source_slots = slots[begin:end]
        assert np.all(source_slots[1:] > source_slots[:-1])
