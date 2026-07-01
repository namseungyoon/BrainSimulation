# -*- coding: utf-8 -*-
"""
04_layers/layers.py  —  단계 4: 층 구분 (V1c)

목적:
  CA1 의 4개 층(SO/SP/SR/SLM) 경계와 두께를 "정규화 깊이" 좌표로 정의한다.
    정규화깊이 nd = ([PH]y - [PH]SO.lower) / ([PH]SLM.upper - [PH]SO.lower)
    -> nd=0 (SO 바닥) ~ nd=1 (SLM 천장), SO->SLM 단조증가.
  brain_regions 라벨과 nd 가 층별로 겹치지 않고 순서대로 분리됨을 확인.

검증 (V1c):
  - 층별 nd 분포가 순서대로(SO<SP<SR<SLM) 비겹침
  - 누적경계(정규화)가 두께기반 예측과 일치: SO0.259 / SP0.349 / SR0.776 / SLM1.0

산출 그림:
  figures/V1c_1_depth_by_layer.png   : 층별 정규화깊이 분포(box) + 경계선
  figures/V1c_2_boundaries_cross.png : 슬라이스 단면 층 라벨 + 경계
  figures/V1c_3_thickness_map.png    : 층별 두께 공간분포(top view)
  figures/V1c_4_column_profiles.png  : 방사 컬럼 따라 층 시퀀스(nd)
출력:
  layer_model.json

실행:
  python 04_layers/layers.py
"""
import os
import json
import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_ID = {"SO": 1, "SP": 2, "SR": 3, "SLM": 4}
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}
THICK = {"SO": 230.0, "SP": 80.0, "SR": 380.0, "SLM": 199.0}  # step1 atlas 실측


def load():
    br, h = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    phy, _ = nrrd.read(os.path.join(ATLAS, "[PH]y.nrrd"))
    so, _ = nrrd.read(os.path.join(ATLAS, "[PH]SO.nrrd"))     # (2,X,Y,Z)
    slm, _ = nrrd.read(os.path.join(ATLAS, "[PH]SLM.nrrd"))
    base = so[0]
    total = slm[1] - so[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = (phy - base) / total                            # 정규화깊이
    return br, origin, vsize, nd, total


def main():
    br, origin, vsize, nd, total = load()
    ca1 = br > 0
    print(f"[load] CA1 voxels {int(ca1.sum()):,}")

    # 층별 nd 분포 + 경계 계산
    stats = {}
    for L in LAYER_ORDER:
        m = ca1 & (br == LAYER_ID[L]) & np.isfinite(nd)
        v = nd[m]
        stats[L] = {"mean": float(v.mean()),
                    "p5": float(np.percentile(v, 5)),
                    "p95": float(np.percentile(v, 95)),
                    "n": int(m.sum())}
    # 두께기반 누적경계
    cum = np.cumsum([THICK[L] for L in LAYER_ORDER])
    cum_norm = (cum / cum[-1]).tolist()
    boundaries = {LAYER_ORDER[i]: cum_norm[i] for i in range(4)}
    print("[V1c] 층별 nd:",
          {L: round(stats[L]["mean"], 3) for L in LAYER_ORDER})
    print("[V1c] 누적경계(정규화):",
          {L: round(boundaries[L], 3) for L in LAYER_ORDER})

    _fig_depth_by_layer(br, nd, ca1, boundaries)
    _fig_boundaries_cross(br, origin, vsize)
    _fig_thickness_map(br, origin, vsize, total)
    _fig_column_profiles(br, origin, vsize, nd)

    model = {
        "step": "4 layers (V1c)",
        "normalized_depth_def":
            "([PH]y - [PH]SO.lower) / ([PH]SLM.upper - [PH]SO.lower), 0=SO base..1=SLM top",
        "thickness_um": THICK,
        "cumulative_boundary_norm": boundaries,
        "layer_nd_stats": stats,
    }
    with open(os.path.join(HERE, "layer_model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f"[OK] layer_model.json + figures -> {FIG}")


def _fig_depth_by_layer(br, nd, ca1, boundaries):
    data = []
    for L in LAYER_ORDER:
        m = ca1 & (br == LAYER_ID[L]) & np.isfinite(nd)
        data.append(nd[m])
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, vert=False, labels=LAYER_ORDER, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, L in zip(bp["boxes"], LAYER_ORDER):
        patch.set_facecolor(LAYER_COLOR[L])
    prev = 0.0
    for L in LAYER_ORDER:
        b = boundaries[L]
        ax.axvline(b, color="gray", ls="--", lw=1)
        ax.text(b, 4.6, f"{b:.3f}", ha="center", fontsize=8, color="gray")
        prev = b
    ax.set_xlabel("normalized depth  (0 = SO base, 1 = SLM top)")
    ax.set_title("V1c-1  layer separation by normalized depth\n"
                 "(dashed = cumulative boundary from thickness)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1c_1_depth_by_layer.png"), dpi=130)
    plt.close(fig)


def _fig_boundaries_cross(br, origin, vsize):
    cx = int((2642.7 - origin[0]) / vsize)      # cylinder300 부근
    plane = br[cx]
    ys, zs = np.where(plane > 0)
    y0, y1 = ys.min(), ys.max()
    z0, z1 = zs.min(), zs.max()
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = matplotlib.colors.ListedColormap(
        ["white"] + [LAYER_COLOR[L] for L in LAYER_ORDER])
    ax.imshow(plane.T, origin="lower", cmap=cmap, vmin=0, vmax=4, aspect="equal")
    ax.set_xlim(y0 - 2, y1 + 2); ax.set_ylim(z0 - 2, z1 + 2)
    handles = [Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel("y voxel"); ax.set_ylabel("z voxel")
    ax.set_title(f"V1c-2  layer boundaries — cross-section @ x-index {cx}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1c_2_boundaries_cross.png"), dpi=130)
    plt.close(fig)


def _fig_thickness_map(br, origin, vsize, total):
    """층 전체두께(total) 의 공간분포 top-view (x-z, 각 컬럼 대표값)."""
    ca1 = br > 0
    idx = np.argwhere(ca1)
    tv = total[idx[:, 0], idx[:, 1], idx[:, 2]]
    ok = np.isfinite(tv)
    idx, tv = idx[ok], tv[ok]
    xw = idx[:, 0] * vsize + origin[0]
    zw = idx[:, 2] * vsize + origin[2]
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(xw, zw, c=tv, s=2, cmap="magma", vmin=np.percentile(tv, 2),
                    vmax=np.percentile(tv, 98))
    fig.colorbar(sc, ax=ax, label="full-column thickness (um)")
    ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    ax.set_title("V1c-3  total layer-column thickness across CA1 (top view)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1c_3_thickness_map.png"), dpi=130)
    plt.close(fig)


def _fig_column_profiles(br, origin, vsize, nd):
    """방사 컬럼 몇 개를 골라 깊이(nd) 따라 층 라벨 시퀀스 확인."""
    cx = int((2642.7 - origin[0]) / vsize)
    plane = br[cx]
    ndp = nd[cx]
    # z 몇 개 골라 y축(깊이 방향 대체)으로 라벨/ nd
    zs = np.where(plane.any(axis=0))[0]
    picks = zs[np.linspace(0, len(zs) - 1, 4).astype(int)]
    fig, axes = plt.subplots(1, len(picks), figsize=(15, 5), sharey=True)
    for ax, z in zip(axes, picks):
        col = plane[:, z]
        ndc = ndp[:, z]
        ys = np.where(col > 0)[0]
        for L in LAYER_ORDER:
            m = col[ys] == LAYER_ID[L]
            ax.scatter(ndc[ys][m], ys[m], s=10, c=LAYER_COLOR[L], label=L)
        ax.set_title(f"z={z}")
        ax.set_xlabel("normalized depth")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("y voxel (along column)")
    handles = [Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER]
    axes[-1].legend(handles=handles, loc="best", fontsize=8)
    fig.suptitle("V1c-4  radial column profiles: layer sequence vs normalized depth")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1c_4_column_profiles.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
