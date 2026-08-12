from __future__ import annotations


def nonnegative_weight_nS(weight_nS: float, *, label: str) -> float:
    if weight_nS < 0.0:
        raise ValueError(f"{label} weight_nS must be non-negative")
    return weight_nS
