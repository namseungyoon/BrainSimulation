from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

HOMOGENEOUS_SOURCE_RATE_RULE: Final = "homogeneous"
LOGNORMAL_SOURCE_RATE_RULE_PREFIX: Final = "mean_preserving_lognormal_cv"


def source_rate_rule(cv: float) -> str:
    if cv < 0.0:
        raise ValueError(f"afferent_source_rate_cv must be nonnegative, got {cv}")
    if cv == 0.0:
        return HOMOGENEOUS_SOURCE_RATE_RULE
    return f"{LOGNORMAL_SOURCE_RATE_RULE_PREFIX}={cv:g}"


def source_rates_hz(
    *,
    base_rate_hz: float,
    count: int,
    cv: float,
    seed: int,
    source: str,
) -> npt.NDArray[np.float64]:
    if base_rate_hz < 0.0:
        raise ValueError(f"base_rate_hz must be nonnegative, got {base_rate_hz}")
    if count < 0:
        raise ValueError(f"count must be nonnegative, got {count}")
    if cv < 0.0:
        raise ValueError(f"afferent_source_rate_cv must be nonnegative, got {cv}")
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if cv == 0.0 or base_rate_hz == 0.0:
        return np.full(count, base_rate_hz, dtype=np.float64)

    sigma = math.sqrt(math.log1p(cv * cv))
    mu = -0.5 * sigma * sigma
    rng = np.random.default_rng(_stable_seed(seed, source, "source_rate"))
    multipliers = rng.lognormal(mean=mu, sigma=sigma, size=count)
    return np.asarray(base_rate_hz * multipliers, dtype=np.float64)


def _stable_seed(seed: int, source: str, field: str) -> int:
    token = f"{source}:{field}"
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(token))
    return (int(seed) + offset) % (2**32)
