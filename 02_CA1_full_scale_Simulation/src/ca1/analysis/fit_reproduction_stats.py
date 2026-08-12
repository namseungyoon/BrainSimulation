from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ca1.analysis.fit_reproduction_schema import (
    CountStats,
    FitCell,
    FloatArray,
    ModelName,
    ReproductionDataset,
    TargetCell,
)

COUNT_MODELS: tuple[ModelName, ...] = ("AEIF", "A-GLIF")


def count_stats_for_dataset(dataset: ReproductionDataset) -> dict[tuple[ModelName, str], CountStats]:
    stats: dict[tuple[ModelName, str], CountStats] = {}
    for model in COUNT_MODELS:
        for cell_name in dataset.cell_order:
            target = dataset.targets[cell_name]
            fit = dataset.fits[model][cell_name]
            stats[(model, cell_name)] = count_stats(target, fit)
    return stats


def count_stats(target: TargetCell, fit: FitCell) -> CountStats:
    if fit.rates_hz is None:
        return CountStats(
            model=fit.model,
            cell_name=target.name,
            count_rmse_z=float("nan"),
            chi_square=float("nan"),
            chi_square_p=float("nan"),
            signed_count_bias=float("nan"),
            max_abs_count_delta=float("nan"),
            n_currents=0,
        )
    stop = min(target.peak_index + 1, int(target.rates_hz.size), int(fit.rates_hz.size))
    target_counts = _counts(target.rates_hz[:stop], target.count_window_ms)
    model_counts = _counts(fit.rates_hz[:stop], fit.count_window_ms)
    sigma_counts = np.maximum(1.0, _counts(target.rate_sigma_hz[:stop], target.count_window_ms))
    delta = model_counts - target_counts
    z_values = delta / sigma_counts
    expected = np.maximum(1.0, target_counts)
    chi_square_value = float(np.sum((delta * delta) / expected))
    return CountStats(
        model=fit.model,
        cell_name=target.name,
        count_rmse_z=float(np.sqrt(np.mean(z_values * z_values))),
        chi_square=chi_square_value,
        chi_square_p=float(chi2.sf(chi_square_value, max(1, stop))),
        signed_count_bias=float(np.mean(delta)),
        max_abs_count_delta=float(np.max(np.abs(delta))),
        n_currents=stop,
    )


def _counts(rates_hz: FloatArray, window_ms: float) -> FloatArray:
    return rates_hz * (window_ms / 1000.0)
