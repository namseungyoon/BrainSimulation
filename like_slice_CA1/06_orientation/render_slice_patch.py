# -*- coding: utf-8 -*-
"""
06_orientation/render_slice_patch.py  —  슬라이스 조직도式 단면 그림

slice400에서 얇은 직사각형 패치(가로=횡방향, 세로=깊이)를 잘라, 그 안 세포들의
형태를 변환·배치하고 가지(수상)를 실제 선으로 그린다. 층은 두께기반 띠로 음영.
  - V2d_6_slice_patch.png : 패치 여러 세포 (기저수상=파랑 SO쪽, 정점수상=초록 SR/SLM쪽)
  - V2d_7_zoom.png        : 추체 1개 확대 (가지 상세)

층(정점축=깊이) 경계(µm, SO바닥=0): SO 0–233 · SP 233–315 · SR 315–702 · SLM 702–905.

실행: python 06_orientation/render_slice_patch.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "lib"))
import morph_transform as mt                   # noqa: E402

MORPH_DIR = os.path.join(ROOT, "data", "morphology_library")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
ASSIGN = os.path.join(ROOT, "05b_memap", "model_assignment.npz")
FIG = os.path.join(HERE, "figures")

# 두께 기반 층 경계 (SO바닥=0), slice400 실측 233/82/387/203
LAYER_BOUND = [("SO", 0, 233), ("SP", 233, 315), ("SR", 315, 702), ("SLM", 702, 905)]
LAYER_BG = {"SO": "#eaf0f7", "SP": "#fdece0", "SR": "#eaf5ee", "SLM": "#f9eaea"}
T_TOTAL = 905.0
COMP_COLOR = {3: "#2b6cb0", 4: "#2f8f4e", 1: "#000000"}  # 기저/정점/소마
COMP_NAME = {3: "기저수상", 4: "정점수상", 1: "소마"}
rng = np.random.default_rng(3)


def local_frame(quat):
    R = mt.quat_to_R(quat)
    return (R.apply([0., 1., 0.]),   # radial(깊이)
            R.apply([1., 0., 0.]),   # t1(가로)
            R.apply([0., 0., 1.]))   # t2(두께)


def draw_cell(ax, world, swc, c0_pos, nd_c0, radial, t1, include_axon=False):
    """세포 형태를 (가로=t1투영, 세로=깊이) 평면에 선분으로."""
    id2idx = {int(i): k for k, i in enumerate(swc["id"])}
    horiz = (world - c0_pos) @ t1
    depth = nd_c0 * T_TOTAL + (world - c0_pos) @ radial
    segs, cols = [], []
    for k, par in enumerate(swc["parent"]):
        t = swc["type"][k]
        if t == 2 and not include_axon:
            continue
        j = id2idx.get(int(par))
        if j is None:
            continue
        segs.append([(horiz[j], depth[j]), (horiz[k], depth[k])])
        cols.append(COMP_COLOR.get(t, "#bbbbbb"))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.4, alpha=0.8))
    # 소마 점
    sm = swc["type"] == 1
    ax.scatter(horiz[sm].mean(), depth[sm].mean(), s=18, c="k", zorder=5)


def layer_bands(ax, xmin, xmax):
    for L, lo, hi in LAYER_BOUND:
        ax.axhspan(lo, hi, color=LAYER_BG[L], zorder=0)
        ax.text(xmax, (lo + hi) / 2, f" {L}", va="center", fontsize=10,
                color="#555", fontweight="bold")
        ax.axhline(hi, color="#ccc", lw=0.6, zorder=1)


def main():
    c = np.load(CELLS, allow_pickle=True)
    a = np.load(ASSIGN, allow_pickle=True)
    xyz = c["xyz"].astype(float); quat = c["quat_wxyz"].astype(float)
    layer = c["layer"].astype(str); nd = c["nd"].astype(float)
    mtype = c["mtype"].astype(str); morph = a["morphology"].astype(str)

    # 중심 추체세포 c0 = slice400 무게중심에 가장 가까운 SP_PC
    pc = np.where(mtype == "SP_PC")[0]
    ctr = xyz[pc].mean(0)
    c0 = pc[np.argmin(np.linalg.norm(xyz[pc] - ctr, axis=1))]
    radial, t1, t2 = local_frame(quat[c0])
    nd_c0 = nd[c0]

    # 패치 선택: 가로 |t1|<250, 두께 |t2|<35 (얇은 절편)
    rel = xyz - xyz[c0]
    h = rel @ t1; th = rel @ t2
    inpatch = (np.abs(h) < 250) & (np.abs(th) < 35)
    idx = np.where(inpatch)[0]
    if len(idx) > 40:
        idx = idx[rng.choice(len(idx), 40, replace=False)]
    print(f"[patch] 중심세포 {c0}, 패치 세포 {len(idx)} "
          f"(층: {dict(zip(*np.unique(layer[idx], return_counts=True)))})")

    # ---- 그림 1: 패치 단면 ----
    fig, ax = plt.subplots(figsize=(8, 9))
    hmin, hmax = -260, 260
    layer_bands(ax, hmin, hmax)
    for k in idx:
        try:
            s = mt.load_swc(os.path.join(MORPH_DIR, morph[k] + ".swc"))
        except FileNotFoundError:
            continue
        w, _ = mt.transform(s["xyz"], mt.soma_center(s), quat[k], xyz[k])
        draw_cell(ax, w, s, xyz[c0], nd_c0, radial, t1)
    ax.set_xlim(hmin, hmax); ax.set_ylim(-30, T_TOTAL + 30)
    ax.set_xlabel("횡방향 (µm)"); ax.set_ylabel("깊이 (µm, SO바닥=0 → SLM천장)")
    ax.set_title(f"V2d-6  slice400 얇은 절편 단면 ({len(idx)}세포)\n"
                 "기저수상(파랑)=SO쪽 · 정점수상(초록)=SR/SLM쪽 · 검정=소마 · 배경=층")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=COMP_COLOR[t], label=COMP_NAME[t])
                       for t in (3, 4, 1)], loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2d_6_slice_patch.png"), dpi=140)
    plt.close(fig)

    # ---- 그림 2: 추체 1개 확대 ----
    fig, ax = plt.subplots(figsize=(7, 9))
    s = mt.load_swc(os.path.join(MORPH_DIR, morph[c0] + ".swc"))
    w, _ = mt.transform(s["xyz"], mt.soma_center(s), quat[c0], xyz[c0])
    hh = (w - xyz[c0]) @ t1
    layer_bands(ax, hh.min(), hh.max())
    draw_cell(ax, w, s, xyz[c0], nd_c0, radial, t1, include_axon=False)
    ax.set_xlim(hh.min() - 20, hh.max() + 40)
    ax.set_ylim(-30, T_TOTAL + 30)
    ax.set_xlabel("횡방향 (µm)"); ax.set_ylabel("깊이 (µm)")
    ax.set_title("V2d-7  추체세포 1개 확대\n소마(SP)에서 기저수상↓(SO)·정점수상↑(SR→SLM)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2d_7_zoom.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] -> {FIG}")


if __name__ == "__main__":
    main()
