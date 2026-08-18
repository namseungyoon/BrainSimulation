# -*- coding: utf-8 -*-
"""
02_neurons/2_placement/make_placement_gif.py  —  2-2 보조: 창 세포 3D 회전 GIF

세포 원장(window_cells.npz)의 5,610개 체세포를 **하나도 빠짐없이** 국소 프레임
(u,r,w)에서 3D로 렌더하고 회전시킨다. 층별 색 + 창 상자 + MEA 전극.
결과: figures/2-2_placement_3d.gif

재료: data/derived/window_cells.npz · config/window_layout.json
실행: python 02_neurons/2_placement/make_placement_gif.py
"""
import os
import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def box_edges(cx, cy, cz, hx, hy, hz):
    c = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    E = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b]) for a, b in E]


def main():
    d = np.load(NPZ, allow_pickle=True)
    uvw = d["uvw"]; layer = d["layer"]
    cfg = json.load(open(CFG, encoding="utf-8"))
    w = cfg["window_um"]; c = w["center_local"]
    ecfg = cfg["electrodes"]; elecs = ecfg["list"]; face_w = ecfg["mea_face_w_um"]
    hx, hy, hz = w["long"] / 2, w["radial"] / 2, w["thick"] / 2

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    n = 0
    for Ln in LAYER_ORDER:
        m = layer == Ln
        if m.any():
            ax.scatter(uvw[m, 0], uvw[m, 1], uvw[m, 2], s=4, c=LC[Ln], alpha=0.55,
                       depthshade=True, linewidths=0, label=f"{Ln} ({int(m.sum())})")
            n += int(m.sum())
    for p, q in box_edges(c["u"], c["r"], c["w"], hx, hy, hz):
        ax.plot(*zip(p, q), color="black", lw=0.8, alpha=0.6)
    for e in elecs:
        mk = "*" if e.get("role") == "stim" else "s"
        ax.scatter([e["u"]], [e["r"]], [face_w], s=220, marker=mk, c="red",
                   edgecolors="black", zorder=10)
        ax.text(e["u"], e["r"], face_w, "  " + e["id"], fontsize=8)
    ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm)"); ax.set_zlabel("두께 w (µm)")
    ax.set_box_aspect((w["long"], w["radial"], w["thick"]))
    ax.legend(loc="upper left", fontsize=8, title=f"층 (총 {n:,}개)")
    fig.suptitle(f"2-2  창 세포 3D 배치 — {n:,}개 전세포 (층관통_v1)", fontsize=12)

    def upd(i):
        ax.view_init(elev=18, azim=i * 4)
        return ()

    ani = FuncAnimation(fig, upd, frames=90, interval=80, blit=False)
    out = os.path.join(FIG, "2-2_placement_3d.gif")
    ani.save(out, writer=PillowWriter(fps=12), dpi=90)
    plt.close(fig)
    print(f"[2-2] 회전 GIF 저장 ({n:,}개 전세포) -> {out}")


if __name__ == "__main__":
    main()
