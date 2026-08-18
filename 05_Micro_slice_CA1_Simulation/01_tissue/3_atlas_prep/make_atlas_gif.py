# -*- coding: utf-8 -*-
"""
01_tissue/3_atlas_prep/make_atlas_gif.py  —  국소(크롭) atlas 3D 회전 GIF

data/derived/atlas_crop.npz 의 층 복셀(SO/SP/SR/SLM)을 물리좌표 점구름으로 3D 렌더 +
확정 전극(자극/기록) 표시 → 방위각 회전 GIF(figures/1-3_atlas_3d.gif).

실행: python 01_tissue/3_atlas_prep/make_atlas_gif.py
"""
import os
import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "atlas_crop.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")
LAYERS = {1: "SO", 2: "SP", 3: "SR", 4: "SLM"}
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}
PER_LAYER = 2500


def main():
    d = np.load(NPZ, allow_pickle=True)
    regions = d["regions"]; origin = np.asarray(d["origin"], float); vs = float(d["vsize"])
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    allpts = []
    for lab, nm in LAYERS.items():
        idx = np.argwhere(regions == lab)
        if len(idx) == 0:
            continue
        if len(idx) > PER_LAYER:
            idx = idx[rng.choice(len(idx), PER_LAYER, replace=False)]
        xyz = origin + idx * vs
        allpts.append(xyz)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=3, c=LC[nm], label=nm, alpha=0.45,
                   depthshade=True, edgecolors="none")
    allpts = np.vstack(allpts)

    # 전극
    cfg = json.load(open(CFG, encoding="utf-8"))
    stim = cfg["electrodes"]["stim_id"]
    for e in cfg["electrodes"]["list"]:
        p = np.array(e["xyz_um"]); isS = e["id"] == stim
        ax.scatter([p[0]], [p[1]], [p[2]], s=200 if isS else 120,
                   marker="P" if isS else "o", c="#e23b3b" if isS else "#111111",
                   edgecolors="white", linewidths=1.5, depthshade=False, zorder=6)
        ax.text(p[0], p[1], p[2] + 25, f'{e["id"]}', fontsize=9, fontweight="bold",
                color="#e23b3b" if isS else "#111111")

    rng_xyz = allpts.max(0) - allpts.min(0)
    ax.set_box_aspect(tuple(rng_xyz))
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    ax.legend(loc="upper right", fontsize=9, title="층")
    ax.set_title(f'국소 atlas 3D — 층 + 전극 (자극 {stim}/SR)  「{cfg["name"]}」', fontsize=12)

    def update(f):
        ax.view_init(elev=18, azim=f * 10)
        return []

    anim = FuncAnimation(fig, update, frames=36, interval=110, blit=False)
    out = os.path.join(FIG, "1-3_atlas_3d.gif")
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    sz = os.path.getsize(out) / 1e6
    print(f"[GIF] {out}  ({sz:.1f} MB, 36프레임)")


if __name__ == "__main__":
    main()
