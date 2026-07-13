from __future__ import annotations

import math
from typing import Final, cast

import numpy as np

from ca1.analysis.receptor_compression_types import (
    CompressionContext,
    FloatArray,
    KernelKey,
)

_EPS: Final = 1e-12


def kernel_response(key: KernelKey) -> FloatArray:
    time = np.linspace(0.0, 500.0, 5001, dtype=np.float64)
    rise = np.exp(-time / key.tau_rise)
    decay = np.exp(-time / key.tau_decay)
    conductance = decay - rise
    peak = float(cast(np.float64, conductance.max()))
    if peak > 0.0:
        conductance = conductance / peak
    return conductance


def response_for_key(key: KernelKey, context: CompressionContext) -> FloatArray:
    if key not in context.responses:
        context.responses[key] = kernel_response(key)
    return context.responses[key]


def response_loss(target: FloatArray, estimate: FloatArray) -> float:
    target_abs = np.abs(target)
    estimate_abs = np.abs(estimate)
    target_area = float(cast(np.float64, target_abs.sum())) + _EPS
    estimate_area = float(cast(np.float64, estimate_abs.sum())) + _EPS
    shape = float(
        cast(np.float64, np.abs(_probability(target) - _probability(estimate)).sum())
    )
    area = abs(math.log(target_area) - math.log(estimate_area))
    peak = abs(
        math.log(float(cast(np.float64, target_abs.max())) + _EPS)
        - math.log(float(cast(np.float64, estimate_abs.max())) + _EPS)
    )
    t_peak = abs(target.argmax() - estimate.argmax()) / target.size
    return float(shape + 0.35 * area + 0.2 * peak + 0.2 * t_peak)


def weighted_sum(values: dict[KernelKey, float], weights: dict[KernelKey, float]) -> float:
    return sum(values[key] * max(weights.get(key, 0.0), _EPS) for key in values)


def weighted_average(values: dict[KernelKey, float], weights: dict[KernelKey, float]) -> float:
    total = sum(max(weights.get(key, 0.0), _EPS) for key in values)
    return weighted_sum(values, weights) / total


def _probability(values: FloatArray) -> FloatArray:
    positive = np.abs(values)
    return positive / (float(cast(np.float64, positive.sum())) + _EPS)
