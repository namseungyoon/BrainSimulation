from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

if matplotlib.get_backend().lower() in ("", "agg", "module://matplotlib_inline.backend_inline"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ca1.analysis.fit_reproduction_data import fit_metrics
from ca1.analysis.fit_reproduction_schema import (
    FitCell,
    FloatArray,
    ModelName,
    ReproductionDataset,
    TargetCell,
)

MODEL_COLORS: Final[dict[ModelName | str, str]] = {
    "GT": "#171717",
    "AEIF": "#2878B5",
    "A-GLIF": "#D55E00",
}
MODEL_MARKERS: Final[dict[ModelName, str]] = {"AEIF": "s", "A-GLIF": "^"}
PLOT_MODELS: Final[tuple[ModelName, ...]] = ("AEIF", "A-GLIF")


def create_fi_grid(dataset: ReproductionDataset) -> Figure:
    _set_style()
    fig = Figure(figsize=(10.8, 8.2))
    axes = tuple(fig.add_subplot(3, 3, idx + 1) for idx in range(9))
    for ax, cell_name in zip(axes, dataset.cell_order):
        target = dataset.targets[cell_name]
        _plot_cell_fi(ax, target, dataset.fits)
    _hide_unused_axes(axes, len(dataset.cell_order))
    handles, labels = axes[0].get_legend_handles_labels()
    _ = fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    _ = fig.supxlabel("Injected current / rheobase", y=0.03)
    _ = fig.supylabel("Firing rate (Hz)", x=0.02)
    _ = fig.suptitle("Morphology-to-point-neuron f-I reproduction", y=1.04, fontsize=13)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.95))
    return fig


def create_summary(dataset: ReproductionDataset) -> Figure:
    _set_style()
    fig = Figure(figsize=(11.8, 3.8))
    axes = tuple(fig.add_subplot(1, 3, idx + 1) for idx in range(3))
    _plot_metric_heatmap(fig, dataset, axes[0], "rate")
    _plot_metric_heatmap(fig, dataset, axes[1], "passive")
    _plot_loss_bars(dataset, axes[2])
    _ = fig.suptitle("Single-cell fit error summary", y=1.03, fontsize=13)
    fig.tight_layout()
    return fig


def save_figure(
    fig: Figure,
    base_path: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> tuple[Path, ...]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = base_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return tuple(saved)


def save_reproduction_figures(
    dataset: ReproductionDataset,
    out_dir: Path,
    prefix: str,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    fi_fig = create_fi_grid(dataset)
    summary_fig = create_summary(dataset)
    try:
        return (
            *save_figure(fi_fig, out_dir / f"{prefix}_fi_grid", formats, dpi),
            *save_figure(summary_fig, out_dir / f"{prefix}_summary", formats, dpi),
        )
    finally:
        plt.close(fi_fig)
        plt.close(summary_fig)


def aglif_rates_available(dataset: ReproductionDataset) -> bool:
    return all(dataset.fits["A-GLIF"][cell].rates_hz is not None for cell in dataset.cell_order)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_cell_fi(
    ax: Axes,
    target: TargetCell,
    fits: dict[ModelName, dict[str, FitCell]],
) -> None:
    x_values = target.currents_nA / target.rheobase_nA
    lower = np.maximum(0.0, target.rates_hz - target.rate_sigma_hz)
    upper = target.rates_hz + target.rate_sigma_hz
    if target.peak_index < x_values.size - 1:
        _ = ax.axvspan(float(x_values[target.peak_index]), float(x_values[-1]), color="#eeeeee", alpha=0.8)
    _ = ax.fill_between(x_values, lower, upper, color="#a3a3a3", alpha=0.25, linewidth=0)
    _ = ax.plot(
        x_values,
        target.rates_hz,
        color=MODEL_COLORS["GT"],
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="NEURON GT",
    )
    for model in PLOT_MODELS:
        fit = fits[model][target.name]
        if fit.rates_hz is not None:
            _ = ax.plot(
                x_values[: fit.rates_hz.size],
                fit.rates_hz,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markersize=3,
                linewidth=1.1,
                label=model,
            )
    _ = ax.set_title(_cell_title(target.name, fits), fontsize=9)
    _ = ax.set_xlim(float(np.min(x_values)) * 0.95, float(np.max(x_values)) * 1.05)
    _ = ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.18, linewidth=0.6)


def _cell_title(cell_name: str, fits: dict[ModelName, dict[str, FitCell]]) -> str:
    aeif = fits["AEIF"][cell_name]
    aglif = fits["A-GLIF"][cell_name]
    aeif_label = _status_token(aeif)
    aglif_label = _status_token(aglif)
    return f"{cell_name}\nAEIF {aeif_label}  A-GLIF {aglif_label}"


def _status_token(fit: FitCell) -> str:
    if fit.passed is True:
        return "pass"
    if fit.passed is False:
        return "fail"
    if fit.rates_hz is None:
        return "no replay"
    return "n/a"


def _hide_unused_axes(axes: tuple[Axes, ...], used: int) -> None:
    for idx, ax in enumerate(axes):
        if idx >= used:
            ax.set_visible(False)


def _plot_metric_heatmap(fig: Figure, dataset: ReproductionDataset, ax: Axes, metric: str) -> None:
    values = _metric_matrix(dataset, metric)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=4.0)
    _ = ax.set_yticks([0, 1])
    _ = ax.set_yticklabels(["AEIF", "A-GLIF"])
    _ = ax.set_xticks(np.arange(len(dataset.cell_order)))
    _ = ax.set_xticklabels(dataset.cell_order, rotation=65, ha="right")
    _ = ax.set_title("f-I RMSE (z)" if metric == "rate" else "Passive RMSE (z)")
    _annotate_matrix(ax, values)
    _ = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def _metric_matrix(dataset: ReproductionDataset, metric: str) -> FloatArray:
    matrix = np.full((2, len(dataset.cell_order)), np.nan)
    for row, model in enumerate(PLOT_MODELS):
        for col, cell_name in enumerate(dataset.cell_order):
            metrics = fit_metrics(dataset.targets[cell_name], dataset.fits[model][cell_name])
            matrix[row, col] = metrics.rate_rmse_z if metric == "rate" else metrics.passive_rmse_z
    return matrix


def _annotate_matrix(ax: Axes, values: FloatArray) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            label = "NA" if not np.isfinite(value) else f"{value:.1f}"
            color = "#111111" if not np.isfinite(value) or value < 2.3 else "white"
            _ = ax.text(col, row, label, ha="center", va="center", fontsize=6.5, color=color)


def _plot_loss_bars(dataset: ReproductionDataset, ax: Axes) -> None:
    x_values = np.arange(len(dataset.cell_order))
    width = 0.38
    model_offsets: tuple[tuple[float, ModelName], ...] = (
        (-width / 2, "AEIF"),
        (width / 2, "A-GLIF"),
    )
    for offset, model in model_offsets:
        losses = [
            fit_metrics(dataset.targets[cell], dataset.fits[model][cell]).loss
            for cell in dataset.cell_order
        ]
        _ = ax.bar(x_values + offset, losses, width, label=model, color=MODEL_COLORS[model], alpha=0.85)
    ax.set_yscale("log")
    _ = ax.set_ylabel("Optimizer loss")
    _ = ax.set_title("Stored fit objective")
    _ = ax.set_xticks(x_values)
    _ = ax.set_xticklabels(dataset.cell_order, rotation=65, ha="right")
    _ = ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
