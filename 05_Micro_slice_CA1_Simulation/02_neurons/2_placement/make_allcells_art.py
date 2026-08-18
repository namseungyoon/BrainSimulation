# -*- coding: utf-8 -*-
"""
02_neurons/2_placement/make_allcells_art.py  —  2-2 보조: 전세포 아트 렌더

5,610 전세포의 실제 수상돌기를 **선(line)** 으로 검은 배경에 옅은 파란빛으로
렌더(Blue Brain 스타일). 세포당 세그먼트 상한으로 다운샘플.
결과: figures/2-2_allcells_art.png · (옵션 --gif) 회전 GIF

실행: python 02_neurons/2_placement/make_allcells_art.py [--gif] [--seg N]
"""
import os
import sys
import json

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEGCAP = int(sys.argv[sys.argv.index("--seg") + 1]) if "--seg" in sys.argv else 220


def morph_segments(path, cap=SEGCAP):
    rows = np.loadtxt(path, comments="#")
    idx = rows[:, 0].astype(int); typ = rows[:, 1].astype(int)
    xyz = rows[:, 2:5].astype(np.float32); par = rows[:, 6].astype(int)
    id2row = {i: k for k, i in enumerate(idx)}
    seg = [(id2row[par[k]], k) for k in range(len(idx))
           if par[k] in id2row and typ[k] in (1, 3, 4)]
    seg = np.array(seg, int)
    if len(seg) > cap:
        seg = seg[np.linspace(0, len(seg) - 1, cap).astype(int)]
    return xyz, seg


def build():
    d = np.load(NPZ, allow_pickle=True)
    xyz = d["xyz"]; Q = d["orientation_wxyz"]; morph = d["morphology"].astype(str)
    cfg = json.load(open(CFG, encoding="utf-8")); fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); Mrows = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    cache = {}
    segs = []
    for i in range(len(xyz)):
        mm = morph[i]
        if mm not in cache:
            cache[mm] = morph_segments(os.path.join(LIB, mm + ".swc"))
        pts, sg = cache[mm]
        if len(sg) == 0:
            continue
        loc = (xyz[i] + Rot.from_quat(Q[i][[1, 2, 3, 0]]).apply(pts) - seed) @ Mrows
        segs.append(loc[sg])          # (nseg,2,3)
    S = np.concatenate(segs).astype(np.float32)
    print(f"[아트] {len(xyz):,}세포 · 세그먼트 {len(S):,} (세포당≤{SEGCAP})")
    return S, cfg


def style(ax, w):
    ax.set_facecolor("black")
    ax.set_box_aspect((w["long"], w["radial"], w["thick"]))
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0)); a.line.set_color((0, 0, 0, 0))
        a.set_ticks([])
    ax.set_axis_off()


def add_lines(ax, S, set_lims=True):
    # 글로우: 굵고 흐린 선 + 가늘고 밝은 선 2겹
    ax.add_collection3d(Line3DCollection(S, colors=[(0.40, 0.68, 1.0, 0.10)], linewidths=0.9))
    ax.add_collection3d(Line3DCollection(S, colors=[(0.80, 0.93, 1.0, 0.30)], linewidths=0.3))
    if set_lims:
        m = S.reshape(-1, 3)
        ax.set_xlim(m[:, 0].min(), m[:, 0].max())
        ax.set_ylim(m[:, 1].min(), m[:, 1].max())
        ax.set_zlim(m[:, 2].min(), m[:, 2].max())


def zoom_limits(S, cfg, t):
    """t: 0(전체)→1(근접). 창 중심으로 서서히 확대."""
    w = cfg["window_um"]; c = w["center_local"]
    m = S.reshape(-1, 3)
    full = np.array([[m[:, 0].min(), m[:, 0].max()],
                     [m[:, 1].min(), m[:, 1].max()],
                     [m[:, 2].min(), m[:, 2].max()]])
    ctr = np.array([c["u"], c["r"], c["w"]])
    # 근접 시 반경(µm): 창의 약 18%
    near_half = np.array([w["long"], w["radial"], w["thick"]]) * 0.18
    e = t * t * (3 - 2 * t)   # smoothstep
    lims = []
    for k in range(3):
        lo0, hi0 = full[k]
        lo1, hi1 = ctr[k] - near_half[k], ctr[k] + near_half[k]
        lims.append((lo0 + (lo1 - lo0) * e, hi0 + (hi1 - hi0) * e))
    return lims


def main():
    S, cfg = build(); w = cfg["window_um"]
    if "--gif-only" not in sys.argv:
        fig = plt.figure(figsize=(13, 12), facecolor="black")
        ax = fig.add_axes([0, 0.03, 1, 0.97], projection="3d"); style(ax, w)
        add_lines(ax, S, set_lims=False); ax.view_init(elev=10, azim=-70)
        xl, yl, zl = zoom_limits(S, cfg, 0.55)   # 클로즈업(프레임 채움)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        fig.text(0.5, 0.02, "5,610 neurons · CA1 micro-slice  (soma window 800x500x400um)",
                 color="0.65", ha="center", fontsize=11)
        out = os.path.join(FIG, "2-2_allcells_art.png")
        fig.savefig(out, dpi=170, facecolor="black"); plt.close(fig)
        print(f"[2-2] 저장 -> {out}")

    if "--gif" in sys.argv:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig = plt.figure(figsize=(9, 9), facecolor="black")
        ax = fig.add_subplot(111, projection="3d"); style(ax, w)
        add_lines(ax, S, set_lims=False); ax.view_init(elev=12, azim=-70)
        N = 80

        def upd(i):
            t = i / (N - 1)
            (xl, yl, zl) = zoom_limits(S, cfg, t)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
            return ()

        ani = FuncAnimation(fig, upd, frames=N, interval=90)
        gout = os.path.join(FIG, "2-2_allcells_zoom.gif")
        ani.save(gout, writer=PillowWriter(fps=11), dpi=85, savefig_kwargs={"facecolor": "black"})
        plt.close(fig); print(f"[2-2] 확대 GIF -> {gout}")


if __name__ == "__main__":
    main()
