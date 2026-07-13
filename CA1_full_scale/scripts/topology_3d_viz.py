#!/usr/bin/env python3
"""Create paper-ready before/after views of ModelDB recurrent topology.

The interactive output uses Plotly and embeds Plotly.js in every HTML file.
Static PNGs use Matplotlib's CPU renderer, which also works on headless hosts
where Kaleido/Chrome sandboxing is unavailable.

Example
-------
source env.sh
python scripts/topology_3d_viz.py \
    --config configs/full_scale.yaml \
    --scale 1.0 \
    --output-dir docs/generated/topology_viz
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from ca1.config import build_network_spec  # noqa: E402
from ca1.sim.modeldb_fastconn import fastconn_axon_distribution  # noqa: E402
from ca1.sim.modeldb_positions import (  # noqa: E402
    ModelDbGeometry,
    modeldb_connectivity_positions,
)
from ca1.sim.modeldb_topology import (  # noqa: E402
    ModelDbFastconn3D,
    binned_fixed_indegree_connections,
    recurrent_projection_plans,
)


CELL_TYPE_ORDER = (
    "Pyramidal",
    "PV_Basket",
    "Bistratified",
    "O_LM",
    "Axo",
    "CCK_Basket",
    "Ivy",
    "Neurogliaform",
    "SCA",
    "CA3",
    "ECIII",
)
POST_TYPES = ("Pyramidal", "Neurogliaform", "PV_Basket", "Bistratified")

# Tol bright / Tableau-inspired colors, separated further for adjacent types.
TYPE_COLORS: dict[str, str] = {
    "Pyramidal": "#4E79A7",
    "PV_Basket": "#E15759",
    "Bistratified": "#EDC948",
    "O_LM": "#59A14F",
    "Axo": "#B07AA1",
    "CCK_Basket": "#FF9DA7",
    "Ivy": "#9C755F",
    "Neurogliaform": "#76B7B2",
    "SCA": "#79706E",
    "CA3": "#F28E2B",
    "ECIII": "#2F9ED1",
}

# Marker area communicates broad morphology/population role, not cell count.
TYPE_SIZES: dict[str, float] = {
    "Pyramidal": 6.0,
    "PV_Basket": 9.0,
    "Bistratified": 8.0,
    "O_LM": 8.0,
    "Axo": 8.5,
    "CCK_Basket": 8.5,
    "Ivy": 7.5,
    "Neurogliaform": 8.0,
    "SCA": 7.5,
    "CA3": 5.0,
    "ECIII": 5.0,
}

AXIS_RANGES = {"x": (0.0, 4000.0), "y": (0.0, 1000.0), "z": (0.0, 354.0)}
PLOTLY_CAMERAS: dict[str, dict[str, Any]] = {
    "oblique": {"eye": {"x": 1.60, "y": 1.40, "z": 0.72}},
    "longitudinal": {"eye": {"x": 0.12, "y": 2.15, "z": 0.58}},
    "top": {
        "eye": {"x": 0.0, "y": 0.0, "z": 2.45},
        "up": {"x": 0.0, "y": 1.0, "z": 0.0},
        "projection": {"type": "orthographic"},
    },
}
MPL_CAMERAS = {
    "oblique": (22.0, -60.0),
    "longitudinal": (17.0, -90.0),
    "top": (90.0, -90.0),
}


@dataclass(frozen=True)
class IncomingEdges:
    positions_um: npt.NDArray[np.float64]
    source_types: tuple[str, ...]
    ring_indices: npt.NDArray[np.int8]


@dataclass(frozen=True)
class Comparison:
    post_type: str
    post_index: int
    post_position_um: npt.NDArray[np.float64]
    uniform: IncomingEdges
    gaussian_3d: IncomingEdges
    uniform_inner_fraction: float
    gaussian_inner_fraction: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize uniform-x ModelDB fastconn versus source-faithful "
            "3-D Gaussian fastconn using real repository generators."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/full_scale.yaml"))
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale applied to CA1 and context-only CA3/ECIII counts (default: full).",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--context-per-type",
        type=int,
        default=800,
        help="Maximum displayed context cells per neuron type.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/generated/topology_viz"),
    )
    parser.add_argument(
        "--post-types",
        nargs="+",
        choices=POST_TYPES,
        default=list(POST_TYPES),
    )
    parser.add_argument("--png-dpi", type=int, default=220)
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Generate only the self-contained interactive Plotly HTML files.",
    )
    args = parser.parse_args()
    if not 0.0 < args.scale <= 1.0:
        parser.error("--scale must be in (0, 1]")
    if args.context_per_type < 1:
        parser.error("--context-per-type must be positive")
    if args.png_dpi < 72:
        parser.error("--png-dpi must be at least 72")
    return args


def _plotly_modules() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise SystemExit(
            "Plotly is required. Install it in the active environment with "
            "`python -m pip install plotly`, then rerun this script."
        ) from exc
    return go, make_subplots


def _external_counts(spec: Any, scale: float) -> dict[str, int]:
    full: dict[str, int] = {}
    for afferent in spec.afferents:
        source = afferent.name.split("_to_", maxsplit=1)[0]
        if source in {"CA3", "ECIII"}:
            full[source] = max(full.get(source, 0), int(afferent.n_source))
    return {name: max(1, int(round(count * scale))) for name, count in full.items()}


def _representative_post_index(
    post_type: str, positions: npt.NDArray[np.float64]
) -> int:
    # NGF spans three SLM depth planes. The shallow plane is the representative
    # one because it is adjacent to the Ivy/O-LM source layers and avoids making
    # the topology comparison mostly a sheet-boundary feasibility effect.
    target_z = (
        float(np.min(positions[:, 2]))
        if post_type == "Neurogliaform"
        else float(np.median(positions[:, 2]))
    )
    center = np.asarray(
        [2000.0, 500.0, target_z], dtype=np.float64
    )
    scale = np.asarray([4000.0, 1000.0, 354.0], dtype=np.float64)
    distance = np.sum(((positions - center) / scale) ** 2, axis=1)
    return int(np.argmin(distance))


def _post_rng_seed(seed: int, projection: str) -> int:
    """Match the 3-D generator's stable per-projection seed for singleton post 0."""
    payload = f"{int(seed)}\0{projection}\0{0}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"ca1-3dtopo").digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _uniform_sources_for_post(
    *,
    pre_type: str,
    post_type: str,
    pre_count: int,
    post_count: int,
    indegree: int,
    post_index: int,
    seed: int,
) -> npt.NDArray[np.int64]:
    calls = binned_fixed_indegree_connections(
        pre_type=pre_type,
        post_type=post_type,
        pre_count=pre_count,
        post_count=post_count,
        indegree=indegree,
    )
    call = next(
        item for item in calls if item.target_start <= post_index < item.target_stop
    )
    projection = f"recurrent:{pre_type}->{post_type}"
    rng = np.random.default_rng(_post_rng_seed(seed, projection))
    return np.asarray(
        rng.choice(
            np.arange(call.source_start, call.source_stop, dtype=np.int64),
            size=call.indegree,
            replace=False,
        ),
        dtype=np.int64,
    )


def _comparison_for_post(
    *,
    post_type: str,
    positions: Mapping[str, npt.NDArray[np.float64]],
    counts: Mapping[str, int],
    plans: Sequence[Any],
    seed: int,
) -> Comparison:
    post_positions = positions[post_type]
    post_index = _representative_post_index(post_type, post_positions)
    post_position = np.asarray(post_positions[post_index], dtype=np.float64)

    uniform_positions: list[npt.NDArray[np.float64]] = []
    gaussian_positions: list[npt.NDArray[np.float64]] = []
    uniform_types: list[str] = []
    gaussian_types: list[str] = []
    uniform_rings: list[npt.NDArray[np.int8]] = []
    gaussian_rings: list[npt.NDArray[np.int8]] = []

    for plan in plans:
        if plan.post != post_type:
            continue
        pre_positions = positions[plan.pre]
        uniform_indices = _uniform_sources_for_post(
            pre_type=plan.pre,
            post_type=post_type,
            pre_count=counts[plan.pre],
            post_count=counts[post_type],
            indegree=plan.indegree,
            post_index=post_index,
            seed=seed,
        )
        selected_uniform = pre_positions[uniform_indices]
        inner_boundary = 4.0 * fastconn_axon_distribution(plan.pre).c_um / 5.0
        distances = np.linalg.norm(selected_uniform - post_position, axis=1)
        old_rings = np.where(distances <= inner_boundary, 1, 0).astype(np.int8)

        # A singleton target view preserves the exact selected cell position and
        # makes edge extraction cheap while still executing ModelDbFastconn3D's
        # production KD-tree, feasibility, Gaussian-ring, and seed code paths.
        target_alias = f"__viz_target_{post_type}"
        topology = ModelDbFastconn3D(
            {plan.pre: pre_positions, target_alias: post_position.reshape(1, 3)}
        )
        post_edges = next(
            topology.iter_post_edges(
                pre_type=plan.pre,
                post_type=target_alias,
                indegree=plan.indegree,
                seed=seed,
                projection=f"recurrent:{plan.pre}->{post_type}",
            )
        )
        selected_gaussian = pre_positions[post_edges.source_indices]

        uniform_positions.append(selected_uniform)
        gaussian_positions.append(selected_gaussian)
        uniform_types.extend([plan.pre] * len(selected_uniform))
        gaussian_types.extend([plan.pre] * len(selected_gaussian))
        uniform_rings.append(old_rings)
        gaussian_rings.append(post_edges.ring_indices)

    old_positions = np.concatenate(uniform_positions, axis=0)
    new_positions = np.concatenate(gaussian_positions, axis=0)
    old_ring_array = np.concatenate(uniform_rings)
    new_ring_array = np.concatenate(gaussian_rings)
    return Comparison(
        post_type=post_type,
        post_index=post_index,
        post_position_um=post_position,
        uniform=IncomingEdges(old_positions, tuple(uniform_types), old_ring_array),
        gaussian_3d=IncomingEdges(new_positions, tuple(gaussian_types), new_ring_array),
        uniform_inner_fraction=float(np.mean(old_ring_array == 1)),
        gaussian_inner_fraction=float(np.mean(new_ring_array == 1)),
    )


def _axis_spec(camera: Mapping[str, Any]) -> dict[str, Any]:
    axis_common = {
        "showbackground": True,
        "backgroundcolor": "#F7F9FC",
        "gridcolor": "#D9E0E8",
        "zerolinecolor": "#BAC5D1",
        "showspikes": False,
    }
    return {
        "xaxis": {**axis_common, "title": "longitudinal x (µm)", "range": AXIS_RANGES["x"]},
        "yaxis": {**axis_common, "title": "transverse y (µm)", "range": AXIS_RANGES["y"]},
        "zaxis": {**axis_common, "title": "depth z (µm)", "range": AXIS_RANGES["z"]},
        "aspectmode": "manual",
        "aspectratio": {"x": 4.0, "y": 1.0, "z": 0.58},
        "camera": dict(camera),
    }


def _edge_coordinates(
    source_positions: npt.NDArray[np.float64], post: npt.NDArray[np.float64]
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    values: list[list[float | None]] = [[], [], []]
    for source in source_positions:
        for axis in range(3):
            values[axis].extend([float(source[axis]), float(post[axis]), None])
    return values[0], values[1], values[2]


def _add_edge_set_plotly(
    fig: Any,
    go: Any,
    edges: IncomingEdges,
    post: npt.NDArray[np.float64],
    *,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    types_array = np.asarray(edges.source_types, dtype=object)
    for cell_type in CELL_TYPE_ORDER:
        mask = types_array == cell_type
        if not bool(np.any(mask)):
            continue
        selected = edges.positions_um[mask]
        edge_x, edge_y, edge_z = _edge_coordinates(selected, post)
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": TYPE_COLORS[cell_type], "width": 1.2},
                opacity=0.20,
                hoverinfo="skip",
                showlegend=False,
                legendgroup=cell_type,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=selected[:, 0],
                y=selected[:, 1],
                z=selected[:, 2],
                mode="markers",
                name=cell_type,
                legendgroup=cell_type,
                showlegend=show_legend,
                marker={
                    "size": TYPE_SIZES[cell_type],
                    "color": TYPE_COLORS[cell_type],
                    "opacity": 0.91,
                    "line": {"color": "white", "width": 0.45},
                },
                customdata=np.full((len(selected), 1), cell_type),
                hovertemplate=(
                    "%{customdata[0]} source<br>x=%{x:.0f} µm<br>"
                    "y=%{y:.0f} µm<br>z=%{z:.0f} µm<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
    fig.add_trace(
        go.Scatter3d(
            x=[post[0]],
            y=[post[1]],
            z=[post[2]],
            mode="markers",
            name="highlighted post",
            showlegend=show_legend,
            legendgroup="post",
            marker={
                "size": 14,
                "symbol": "diamond",
                "color": "#111827",
                "line": {"color": "#FBBF24", "width": 2.5},
            },
            hovertemplate=(
                "highlighted post cell<br>x=%{x:.0f} µm<br>"
                "y=%{y:.0f} µm<br>z=%{z:.0f} µm<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def _comparison_title(comparison: Comparison) -> str:
    old = 100.0 * comparison.uniform_inner_fraction
    new = 100.0 * comparison.gaussian_inner_fraction
    return (
        f"{comparison.post_type} post cell · innermost ring: {old:.1f}% → {new:.1f}% "
        "(design ≈8% → ≈87%) · nearby-cell shared input: 10–61× higher after"
    )


def _plotly_comparison(comparison: Comparison, go: Any, make_subplots: Any) -> Any:
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.015,
        subplot_titles=(
            f"BEFORE · uniform-x ({len(comparison.uniform.positions_um):,} edges)",
            f"AFTER · source-faithful 3-D Gaussian ({len(comparison.gaussian_3d.positions_um):,} edges)",
        ),
    )
    _add_edge_set_plotly(
        fig,
        go,
        comparison.uniform,
        comparison.post_position_um,
        row=1,
        col=1,
        show_legend=True,
    )
    _add_edge_set_plotly(
        fig,
        go,
        comparison.gaussian_3d,
        comparison.post_position_um,
        row=1,
        col=2,
        show_legend=False,
    )
    scene = _axis_spec(PLOTLY_CAMERAS["oblique"])
    fig.update_layout(
        title={"text": _comparison_title(comparison), "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 14, "color": "#172033"},
        scene=scene,
        scene2=scene,
        legend={
            "title": {"text": "presynaptic type"},
            "orientation": "h",
            "yanchor": "top",
            "y": -0.07,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.88)",
        },
        margin={"l": 15, "r": 15, "t": 92, "b": 95},
        width=1800,
        height=960,
        hoverlabel={"font_size": 13},
    )
    return fig


def _sample_context(
    positions: Mapping[str, npt.NDArray[np.float64]],
    per_type: int,
    seed: int,
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, int]]:
    sampled: dict[str, npt.NDArray[np.float64]] = {}
    displayed: dict[str, int] = {}
    for type_index, cell_type in enumerate(CELL_TYPE_ORDER):
        points = positions[cell_type]
        count = min(per_type, len(points))
        if count == len(points):
            indices = np.arange(len(points), dtype=np.int64)
        else:
            rng = np.random.default_rng(seed + 1009 * (type_index + 1))
            indices = np.sort(rng.choice(len(points), size=count, replace=False))
        sampled[cell_type] = points[indices]
        displayed[cell_type] = count
    return sampled, displayed


def _plotly_context(sampled: Mapping[str, npt.NDArray[np.float64]], go: Any) -> Any:
    fig = go.Figure()
    for cell_type in CELL_TYPE_ORDER:
        points = sampled[cell_type]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=cell_type,
                marker={
                    "size": TYPE_SIZES[cell_type] * 0.72,
                    "color": TYPE_COLORS[cell_type],
                    "opacity": 0.74,
                    "line": {"width": 0},
                },
                hovertemplate=(
                    f"{cell_type}<br>x=%{{x:.0f}} µm<br>y=%{{y:.0f}} µm<br>"
                    "z=%{z:.0f} µm<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title={
            "text": "CA1 population context · nine intrinsic types + CA3/ECIII sources",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        scene=_axis_spec(PLOTLY_CAMERAS["oblique"]),
        font={"family": "Arial, sans-serif", "size": 14, "color": "#172033"},
        legend={"title": {"text": "neuron type"}, "itemsizing": "constant"},
        margin={"l": 20, "r": 180, "t": 78, "b": 30},
        width=1500,
        height=900,
    )
    return fig


def _configure_mpl_axis(ax: Any, *, elev: float, azim: float) -> None:
    ax.set_xlim(*AXIS_RANGES["x"])
    ax.set_ylim(*AXIS_RANGES["y"])
    ax.set_zlim(*AXIS_RANGES["z"])
    ax.set_xlabel("longitudinal x (µm)", labelpad=8)
    ax.set_ylabel("transverse y (µm)", labelpad=8)
    ax.set_zlabel("depth z (µm)", labelpad=8)
    # Modest depth exaggeration keeps the 354-µm laminae readable beside the
    # 4,000-µm longitudinal axis without changing any numeric axis limits.
    ax.set_box_aspect((3.5, 1.15, 0.78))
    ax.view_init(elev=elev, azim=azim)
    if elev >= 89.0:
        # The z axis is viewed exactly end-on in the top snapshot; hiding that
        # collapsed spine avoids an unreadable stack of z tick labels.
        ax.set_zticks([])
        ax.set_zlabel("")
        ax.zaxis.line.set_linewidth(0.0)
        ax.zaxis.pane.set_visible(False)
    ax.grid(True, alpha=0.28)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#F7F9FC")
        pane.set_edgecolor("#D7DEE8")
        pane.set_alpha(1.0)


def _draw_mpl_edges(
    ax: Any,
    edges: IncomingEdges,
    post: npt.NDArray[np.float64],
) -> None:
    types_array = np.asarray(edges.source_types, dtype=object)
    for cell_type in CELL_TYPE_ORDER:
        mask = types_array == cell_type
        if not bool(np.any(mask)):
            continue
        selected = edges.positions_um[mask]
        segments = np.stack(
            (selected, np.broadcast_to(post, selected.shape)), axis=1
        )
        ax.add_collection3d(
            Line3DCollection(
                segments,
                colors=TYPE_COLORS[cell_type],
                linewidths=0.62,
                alpha=0.26,
                rasterized=True,
            )
        )
        ax.scatter(
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            s=TYPE_SIZES[cell_type] ** 2 * 0.66,
            c=TYPE_COLORS[cell_type],
            alpha=0.92,
            edgecolors="white",
            linewidths=0.28,
            depthshade=False,
            rasterized=True,
        )
    ax.scatter(
        [post[0]],
        [post[1]],
        [post[2]],
        s=190,
        c="#111827",
        marker="*",
        edgecolors="#FBBF24",
        linewidths=1.6,
        depthshade=False,
        zorder=20,
    )


def _legend_handles(types: Sequence[str], *, include_post: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=cell_type,
            markerfacecolor=TYPE_COLORS[cell_type],
            markeredgecolor="white",
            markersize=TYPE_SIZES[cell_type],
        )
        for cell_type in CELL_TYPE_ORDER
        if cell_type in types
    ]
    if include_post:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                label="highlighted post",
                markerfacecolor="#111827",
                markeredgecolor="#FBBF24",
                markersize=12,
            )
        )
    return handles


def _write_comparison_pngs(
    comparison: Comparison, output_dir: Path, dpi: int
    ) -> list[Path]:
    outputs: list[Path] = []
    types = sorted(
        set(comparison.uniform.source_types), key=lambda name: CELL_TYPE_ORDER.index(name)
    )
    for angle_name, (elev, azim) in MPL_CAMERAS.items():
        fig = plt.figure(figsize=(16.2, 6.6), facecolor="white", constrained_layout=False)
        left = fig.add_subplot(1, 2, 1, projection="3d")
        right = fig.add_subplot(1, 2, 2, projection="3d")
        _draw_mpl_edges(left, comparison.uniform, comparison.post_position_um)
        _draw_mpl_edges(right, comparison.gaussian_3d, comparison.post_position_um)
        for ax, label in (
            (left, "BEFORE · uniform-x"),
            (right, "AFTER · 3-D Gaussian"),
        ):
            _configure_mpl_axis(ax, elev=elev, azim=azim)
            ax.set_title(label, fontsize=13, fontweight="bold", pad=12)
        fig.suptitle(_comparison_title(comparison), fontsize=15, fontweight="bold", y=0.982)
        fig.legend(
            handles=_legend_handles(types, include_post=True),
            loc="lower center",
            ncol=min(6, len(types) + 1),
            frameon=False,
            bbox_to_anchor=(0.5, 0.015),
        )
        fig.subplots_adjust(left=0.025, right=0.985, top=0.88, bottom=0.15, wspace=0.02)
        path = output_dir / f"{comparison.post_type.lower()}_{angle_name}.png"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def _write_context_pngs(
    sampled: Mapping[str, npt.NDArray[np.float64]], output_dir: Path, dpi: int
) -> list[Path]:
    outputs: list[Path] = []
    for angle_name in ("oblique", "top"):
        elev, azim = MPL_CAMERAS[angle_name]
        fig = plt.figure(figsize=(13.8, 6.7), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        for cell_type in CELL_TYPE_ORDER:
            points = sampled[cell_type]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=TYPE_SIZES[cell_type] ** 2 * 0.34,
                c=TYPE_COLORS[cell_type],
                alpha=0.68,
                edgecolors="none",
                depthshade=False,
                rasterized=True,
            )
        _configure_mpl_axis(ax, elev=elev, azim=azim)
        ax.set_title(
            "CA1 population context · nine intrinsic types + CA3/ECIII sources",
            fontsize=15,
            fontweight="bold",
            pad=14,
        )
        fig.legend(
            handles=_legend_handles(CELL_TYPE_ORDER, include_post=False),
            loc="center right",
            bbox_to_anchor=(0.985, 0.5),
            frameon=False,
            title="neuron type",
        )
        fig.subplots_adjust(left=0.03, right=0.84, top=0.92, bottom=0.06)
        path = output_dir / f"population_context_{angle_name}.png"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def _write_readme(
    *,
    output_dir: Path,
    config: Path,
    scale: float,
    seed: int,
    full_counts: Mapping[str, int],
    generated_counts: Mapping[str, int],
    displayed_counts: Mapping[str, int],
    comparisons: Sequence[Comparison],
) -> None:
    comparison_rows = "\n".join(
        "| {post} | {idx:,} | ({x:.1f}, {y:.1f}, {z:.1f}) | {n:,} | "
        "{old:.1f}% | {new:.1f}% |".format(
            post=item.post_type,
            idx=item.post_index,
            x=item.post_position_um[0],
            y=item.post_position_um[1],
            z=item.post_position_um[2],
            n=len(item.uniform.positions_um),
            old=100.0 * item.uniform_inner_fraction,
            new=100.0 * item.gaussian_inner_fraction,
        )
        for item in comparisons
    )
    count_rows = "\n".join(
        f"| {name} | {full_counts[name]:,} | {generated_counts[name]:,} | "
        f"{displayed_counts[name]:,} | {TYPE_COLORS[name]} | {TYPE_SIZES[name]:.1f} |"
        for name in CELL_TYPE_ORDER
    )
    text = f"""# ModelDB topology visualization

Generated from `{config}` with seed `{seed}` on the CPU. The representative
geometry uses scale `{scale:g}` for all nine CA1 counts and for the context-only
CA3/ECIII counts. Positions and topology are not spatially cropped: every paired
panel uses the full 4000 × 1000 × 354 µm sheet and every incoming recurrent
biological projection for its highlighted post cell.

The population context displays a deterministic maximum of
`{max(displayed_counts.values()):,}` points per type. The paired panels do not
subsample edges. Split receptor ports and co-release rows are grouped into one
biological pre→post plan before sampling.

Posts are selected nearest the sheet center (x=2000, y=500 µm) and the median
depth plane. Neurogliaform uses its shallow SLM plane (z=187 µm), adjacent to
its Ivy/O-LM source layers, so the comparison reflects topology rather than a
deep-sheet feasibility boundary effect.

## Representative posts and measured locality

| post type | network index | position µm (x, y, z) | incoming edges/topology | uniform inner ring | 3-D inner ring |
|---|---:|---:|---:|---:|---:|
{comparison_rows}

The old fixed-indegree edge identities are deterministic CPU samples without
replacement from the exact interval returned by
`binned_fixed_indegree_connections`; NEST-GPU normally samples that interval at
runtime. The new identities come directly from `ModelDbFastconn3D.iter_post_edges`.
For fast extraction, the exact representative position is presented to that
generator as a singleton target view (post index 0), while the old x-bin is
selected using the representative cell's network index.

## Color and size convention

Marker size is a categorical visual convention and does not encode abundance.

| type | full count | generated count | context shown | color | marker size |
|---|---:|---:|---:|---|---:|
{count_rows}

## Generator provenance

- `src/ca1/config.py::build_network_spec` — canonical full-scale counts/projections.
- `src/ca1/sim/modeldb_positions.py::modeldb_connectivity_positions` — positions,
  including external source populations.
- `src/ca1/sim/modeldb_positions.py::_positions_for_count` — ModelDB 3-D grid.
- `src/ca1/sim/modeldb_topology.py::binned_fixed_indegree_connections` — old
  uniform-x source-window topology.
- `src/ca1/sim/modeldb_topology.py::ModelDbFastconn3D.iter_post_edges` — new
  source-faithful 3-D Gaussian topology.

## Reading the figure

The topology change leaves the full sheet and indegree budgets intact but moves
presynaptic sources from a broad longitudinal window into a tight 3-D
neighborhood around each post cell (the innermost Gaussian ring is approximately
87% after the change).

Each HTML is self-contained Plotly. PNGs are high-resolution Matplotlib CPU
renders of the same data and camera views, used because this headless host blocks
the Kaleido/Chrome crash-handler sandbox.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    go, make_subplots = _plotly_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec = build_network_spec(args.config, scale=args.scale, seed=args.seed)
    internal_counts = spec.scaled_counts()
    external_counts = _external_counts(spec, args.scale)
    generated_counts = {**internal_counts, **external_counts}
    missing = set(CELL_TYPE_ORDER) - set(generated_counts)
    if missing:
        raise ValueError(f"position counts missing for: {sorted(missing)}")
    positions = modeldb_connectivity_positions(generated_counts)
    plans = recurrent_projection_plans(spec.projections, internal_counts)

    comparisons = [
        _comparison_for_post(
            post_type=post_type,
            positions=positions,
            counts=internal_counts,
            plans=plans,
            seed=args.seed,
        )
        for post_type in args.post_types
    ]
    sampled_context, displayed_counts = _sample_context(
        positions, args.context_per_type, args.seed
    )

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "modeldb_topology_3d",
            "height": 1080,
            "width": 1920,
            "scale": 2,
        },
    }
    html_paths: list[Path] = []
    for comparison in comparisons:
        figure = _plotly_comparison(comparison, go, make_subplots)
        path = args.output_dir / f"{comparison.post_type.lower()}_comparison.html"
        figure.write_html(path, include_plotlyjs=True, full_html=True, config=config)
        html_paths.append(path)
    context_figure = _plotly_context(sampled_context, go)
    context_path = args.output_dir / "population_context.html"
    context_figure.write_html(
        context_path, include_plotlyjs=True, full_html=True, config=config
    )
    html_paths.append(context_path)

    png_paths: list[Path] = []
    if not args.skip_png:
        for comparison in comparisons:
            png_paths.extend(
                _write_comparison_pngs(comparison, args.output_dir, args.png_dpi)
            )
        png_paths.extend(
            _write_context_pngs(
                sampled_context, args.output_dir, args.png_dpi
            )
        )

    full_internal_counts = {
        name: int(cell_type.count) for name, cell_type in spec.cell_types.items()
    }
    full_external_counts = {
        name: int(round(count / args.scale)) for name, count in external_counts.items()
    }
    full_counts = {**full_internal_counts, **full_external_counts}
    _write_readme(
        output_dir=args.output_dir,
        config=args.config,
        scale=args.scale,
        seed=args.seed,
        full_counts=full_counts,
        generated_counts=generated_counts,
        displayed_counts=displayed_counts,
        comparisons=comparisons,
    )
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "scale": args.scale,
        "seed": args.seed,
        "geometry_um": asdict(ModelDbGeometry()),
        "full_counts": full_counts,
        "generated_counts": generated_counts,
        "context_displayed_counts": displayed_counts,
        "palette": TYPE_COLORS,
        "marker_sizes": TYPE_SIZES,
        "comparisons": [
            {
                "post_type": item.post_type,
                "post_index": item.post_index,
                "post_position_um": item.post_position_um.tolist(),
                "incoming_edges_per_topology": len(item.uniform.positions_um),
                "uniform_inner_fraction": item.uniform_inner_fraction,
                "gaussian_3d_inner_fraction": item.gaussian_inner_fraction,
            }
            for item in comparisons
        ],
        "html": [str(path) for path in html_paths],
        "png": [str(path) for path in png_paths],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(html_paths)} self-contained Plotly HTML files")
    print(f"Wrote {len(png_paths)} high-resolution PNG files")
    for item in comparisons:
        print(
            f"{item.post_type}: {len(item.uniform.positions_um)} edges, "
            f"inner ring {100 * item.uniform_inner_fraction:.1f}% -> "
            f"{100 * item.gaussian_inner_fraction:.1f}%"
        )
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
