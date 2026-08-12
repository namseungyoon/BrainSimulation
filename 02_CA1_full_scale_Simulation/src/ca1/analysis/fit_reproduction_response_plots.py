from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ca1.analysis.fit_reproduction_plots import MODEL_COLORS, PLOT_MODELS, save_figure
from ca1.analysis.fit_reproduction_schema import (
    CountStats,
    CurveName,
    ModelName,
    ReproductionDataset,
    ResponseTrace,
)
from ca1.analysis.fit_reproduction_stats import count_stats_for_dataset

_CURVES: tuple[CurveName, ...] = ("GT", "AEIF", "A-GLIF")


def create_response_figure(traces: dict[str, ResponseTrace]) -> Figure:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    cell_order = tuple(traces)
    fig = Figure(figsize=(10.8, 8.2))
    axes = tuple(fig.add_subplot(3, 3, idx + 1) for idx in range(9))
    for ax, cell_name in zip(axes, cell_order):
        _plot_response(ax, traces[cell_name])
    _hide_unused(axes, len(cell_order))
    handles, labels = axes[0].get_legend_handles_labels()
    _ = fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    _ = fig.supxlabel("Time from stimulus onset (ms)", y=0.03)
    _ = fig.supylabel("Membrane potential (mV)", x=0.02)
    _ = fig.suptitle("Stimulus response voltage traces", y=1.04, fontsize=13)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.95))
    return fig


def create_count_statistics_figure(dataset: ReproductionDataset) -> Figure:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    stats = count_stats_for_dataset(dataset)
    fig = Figure(figsize=(11.8, 3.8))
    axes = tuple(fig.add_subplot(1, 3, idx + 1) for idx in range(3))
    _heatmap(fig, axes[0], _count_matrix(dataset, stats, "rmse"), dataset.cell_order, "Spike-count RMSE (z)", 4.0)
    _heatmap(fig, axes[1], _count_matrix(dataset, stats, "p"), dataset.cell_order, "Count chi-square p", 1.0)
    _bias_bars(axes[2], dataset, stats)
    _ = fig.suptitle("Spike-count statistical validation", y=1.03, fontsize=13)
    fig.tight_layout()
    return fig


def save_response_outputs(
    traces: dict[str, ResponseTrace] | None,
    dataset: ReproductionDataset,
    out_dir: Path,
    prefix: str,
    formats: tuple[str, ...],
    dpi: int,
) -> tuple[Path, ...]:
    saved: list[Path] = []
    count_fig = create_count_statistics_figure(dataset)
    try:
        saved.extend(save_figure(count_fig, out_dir / f"{prefix}_spike_count_stats", formats, dpi))
    finally:
        plt.close(count_fig)
    if traces is not None:
        response_fig = create_response_figure(traces)
        try:
            saved.extend(save_figure(response_fig, out_dir / f"{prefix}_stimulus_response", formats, dpi))
        finally:
            plt.close(response_fig)
    return tuple(saved)


def _plot_response(ax: Axes, trace: ResponseTrace) -> None:
    rel_time = trace.time_ms - 200.0
    _ = ax.axvspan(0.0, 600.0, color="#eeeeee", alpha=0.75)
    for curve in _CURVES:
        voltage = trace.voltages_mV[curve]
        _ = ax.plot(rel_time[: voltage.size], voltage, color=MODEL_COLORS[curve], linewidth=1.0, label=curve)
    _ = ax.set_title(f"{trace.cell_name}  {trace.current_ratio:.1f}x rheobase", fontsize=9)
    _ = ax.set_xlim(-25.0, 650.0)
    _ = ax.set_ylim(-90.0, 45.0)
    ax.grid(alpha=0.18, linewidth=0.6)


def _heatmap(
    fig: Figure,
    ax: Axes,
    values: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    cell_order: tuple[str, ...],
    title: str,
    vmax: float,
) -> None:
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
    _ = ax.set_yticks([0, 1])
    _ = ax.set_yticklabels(["AEIF", "A-GLIF"])
    _ = ax.set_xticks(np.arange(len(cell_order)))
    _ = ax.set_xticklabels(cell_order, rotation=65, ha="right")
    _ = ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            label = "NA" if not np.isfinite(value) else f"{value:.2g}"
            color = "#111111" if not np.isfinite(value) or value < 2.3 else "white"
            _ = ax.text(col, row, label, ha="center", va="center", fontsize=6.5, color=color)
    _ = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def _count_matrix(
    dataset: ReproductionDataset,
    stats: dict[tuple[ModelName, str], CountStats],
    metric: str,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    matrix = np.full((2, len(dataset.cell_order)), np.nan)
    for row, model in enumerate(PLOT_MODELS):
        for col, cell_name in enumerate(dataset.cell_order):
            stat = stats[(model, cell_name)]
            matrix[row, col] = stat.count_rmse_z if metric == "rmse" else stat.chi_square_p
    return matrix


def _bias_bars(
    ax: Axes,
    dataset: ReproductionDataset,
    stats: dict[tuple[ModelName, str], CountStats],
) -> None:
    x_values = np.arange(len(dataset.cell_order))
    width = 0.38
    model_offsets: tuple[tuple[float, ModelName], ...] = (
        (-width / 2, "AEIF"),
        (width / 2, "A-GLIF"),
    )
    for offset, model in model_offsets:
        values = [stats[(model, cell)].signed_count_bias for cell in dataset.cell_order]
        _ = ax.bar(x_values + offset, values, width, label=model, color=MODEL_COLORS[model], alpha=0.85)
    _ = ax.axhline(0.0, color="#222222", linewidth=0.8)
    _ = ax.set_ylabel("Mean count bias")
    _ = ax.set_title("Signed count bias")
    _ = ax.set_xticks(x_values)
    _ = ax.set_xticklabels(dataset.cell_order, rotation=65, ha="right")
    _ = ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)


def _hide_unused(axes: tuple[Axes, ...], used: int) -> None:
    for idx, ax in enumerate(axes):
        if idx >= used:
            ax.set_visible(False)
