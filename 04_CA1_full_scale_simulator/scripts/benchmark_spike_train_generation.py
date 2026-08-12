#!/usr/bin/env python3
"""CPU-only benchmark for literal-source spike-time placement.

The legacy path intentionally reproduces the expensive per-source
``choice(arange(...), replace=False)`` implementation.  It is expected to be
slow; use ``--skip-legacy`` for a quick regression check.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ca1.sim.gpu_backend import _spike_slot_batches


def legacy_slots(counts: np.ndarray, *, slot_count: int, seed: int) -> int:
    """Previous per-node placement loop, excluding NEST GPU SetStatus."""
    rng = np.random.default_rng(seed)
    total = 0
    for count in counts:
        if count:
            total += int(
                np.sort(
                    rng.choice(
                        np.arange(1, slot_count),
                        size=int(count),
                        replace=False,
                    )
                ).size
            )
    return total


def vectorized_slots(counts: np.ndarray, *, slot_count: int, seed: int) -> int:
    """Current bounded-memory vectorized placement loop, excluding SetStatus."""
    rng = np.random.default_rng(seed)
    return sum(
        int(slots.size)
        for _, _, slots in _spike_slot_batches(
            counts,
            slot_count=slot_count,
            rng=rng,
        )
    )


def timed(label: str, func: object) -> tuple[float, int]:
    start = time.perf_counter()
    spikes = func()  # type: ignore[operator]
    seconds = time.perf_counter() - start
    print(f"{label}: {seconds:.3f} s ({spikes:,} spikes)")
    return seconds, spikes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=250_000)
    parser.add_argument("--slot-count", type=int, default=25_000)
    parser.add_argument("--rate-hz", type=float, default=0.65)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()

    count_rng = np.random.default_rng(args.seed)
    counts = count_rng.poisson(args.rate_hz * args.duration_s, size=args.sources)
    print(
        f"sources={args.sources:,}, slots={args.slot_count:,}, "
        f"mean_count={counts.mean():.4f}, total_spikes={counts.sum():,}"
    )
    if not args.skip_legacy:
        timed(
            "legacy",
            lambda: legacy_slots(counts, slot_count=args.slot_count, seed=args.seed + 1),
        )
    timed(
        "vectorized",
        lambda: vectorized_slots(counts, slot_count=args.slot_count, seed=args.seed + 1),
    )


if __name__ == "__main__":
    main()
