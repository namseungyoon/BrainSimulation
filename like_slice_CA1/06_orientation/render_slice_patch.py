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
COMP_COLOR = {1: "#000000", 2: "#e08214", 3: "#2b6cb0", 4: "#2f8f4e"}  # 소마/축삭/기저/정점
COMP_NAME = {1: "소마", 2: "축삭", 3: "기저수상", 4: "정점수상"}
rng = np.random.default_rng(3)


def local_frame(quat):
    R = mt.quat_to_R(quat)
    return (R.apply([0., 1., 0.]),   # radial(깊이)
            R.apply([1., 0., 0.]),   # t1(가로)
            R.apply([0., 0., 1.]))   # t2(두께)


def draw_cell(ax, world, swc, c0_pos, nd_c0, radial, t1, include_axon=True):
    """세포 형태를 (가로=t1투영, 세로=깊이) 평면에 구획별 색 선분으로.
    축삭은 옅고 얇게(배경 haze), 수상돌기는 진하게 위에 그림."""
    id2idx = {int(i): k for k, i in enumerate(swc["id"])}
    horiz = (world - c0_pos) @ t1
    depth = nd_c0 * T_TOTAL + (world - c0_pos) @ radial
    ax_segs, dn_segs, dn_cols = [], [], []
    for k, par in enumerate(swc["parent"]):
        t = swc["type"][k]
        j = id2idx.get(int(par))
        if j is None:
            continue
        seg = [(horiz[j], depth[j]), (horiz[k], depth[k])]
        if t == 2:
            if include_axon:
                ax_segs.append(seg)
        else:
            dn_segs.append(seg); dn_cols.append(COMP_COLOR.get(t, "#bbbbbb"))
    if ax_segs:   # 축삭: 옅은 배경
        ax.add_collection(LineCollection(ax_segs, colors=COMP_COLOR[2],
                                         linewidths=0.2, alpha=0.18, zorder=2))
    ax.add_collection(LineCollection(dn_segs, colors=dn_cols,
                                     linewidths=0.5, alpha=0.85, zorder=3))
    sm = swc["type"] == 1
    ax.scatter(horiz[sm].mean(), depth[sm].mean(), s=16, c="k", zorder=5)


def comp_legend(ax, with_axon=True):
    from matplotlib.lines import Line2D
    order = [1, 2, 3, 4] if with_axon else [1, 3, 4]
    ax.legend(handles=[Line2D([0], [0], color=COMP_COLOR[t], lw=2,
                              label=COMP_NAME[t]) for t in order],
              loc="upper left", fontsize=9, framealpha=0.9)


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

    # 고정 직사각형 박스 (중심 c0): 가로(t1) ±HW, 두께(t2) ±TH, 깊이 전층.
    # 박스 안 세포는 전부 표시 (랜덤 추출 아님).
    HW, TH = 75.0, 18.0     # 가로 150µm × 두께 36µm
    rel = xyz - xyz[c0]
    h = rel @ t1; th = rel @ t2
    inpatch = (np.abs(h) < HW) & (np.abs(th) < TH)
    idx = np.where(inpatch)[0]
    print(f"[patch] 박스 중심세포 {c0}, 가로±{HW} 두께±{TH}µm 안 세포 {len(idx)}개 "
          f"(층: {dict(zip(*np.unique(layer[idx], return_counts=True)))})")

    # ---- 위치 지도: 전체 slice400 top-view + 박스 세포 강조 ----
    fig, axm = plt.subplots(figsize=(8, 6.5))
    axm.scatter(xyz[:, 0], xyz[:, 2], s=1, c="#dddddd", alpha=0.4, label="slice400 전체")
    axm.scatter(xyz[idx, 0], xyz[idx, 2], s=14, c="red", label=f"박스 내 {len(idx)}세포")
    axm.scatter(xyz[c0, 0], xyz[c0, 2], s=60, marker="*", c="k", label="박스 중심")
    axm.set_aspect("equal"); axm.set_xlabel("x (µm)"); axm.set_ylabel("z (µm)")
    axm.legend(fontsize=9); axm.set_title("V2d-8  찍은 직사각형 부위 위치 (위에서 본 slice400)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2d_8_patch_location.png"), dpi=130)
    plt.close(fig)

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
    ax.set_title(f"V2d-6  찍은 직사각형 박스 내 전체 {len(idx)}세포 (가로150×두께36µm×전층)\n"
                 "구획별 색: 소마/축삭/기저수상/정점수상 · 배경=층")
    comp_legend(ax, with_axon=True)
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
    draw_cell(ax, w, s, xyz[c0], nd_c0, radial, t1, include_axon=True)
    ax.set_xlim(hh.min() - 40, hh.max() + 60)
    ax.set_ylim(-30, T_TOTAL + 30)
    ax.set_xlabel("횡방향 (µm)"); ax.set_ylabel("깊이 (µm)")
    ax.set_title("V2d-7  추체세포 1개 확대 (구획별 색)\n"
                 "소마(SP)·기저수상↓(SO)·정점수상↑(SR→SLM)·축삭(주황)")
    comp_legend(ax, with_axon=True)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2d_7_zoom.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] -> {FIG}")


if __name__ == "__main__":
    main()
