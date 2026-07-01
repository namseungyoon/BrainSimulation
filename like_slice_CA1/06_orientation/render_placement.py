# -*- coding: utf-8 -*-
"""
06_orientation/render_placement.py  —  배치된 세포 다수를 회전 GIF 로 렌더

slice400 세포를 실제 구성비대로 표본 추출(기본 300개) → 각 세포의 변이형태(.swc)를
평행이동+회전 배치 → 수상(축삭 제외) 점을 층별 색으로 3D 회전 GIF.

산출: figures/V2d_5_placement_dense.gif , figures/V2d_5_placement_top.png

실행: python 06_orientation/render_placement.py [n_cells]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))
from gif_util import save_rotate_gif           # noqa: E402
import morph_transform as mt                   # noqa: E402

MORPH_DIR = os.path.join(ROOT, "data", "morphology_library")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
ASSIGN = os.path.join(ROOT, "05b_memap", "model_assignment.npz")
FIG = os.path.join(HERE, "figures")
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}
rng = np.random.default_rng(1)


def main():
    n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    c = np.load(CELLS, allow_pickle=True)
    a = np.load(ASSIGN, allow_pickle=True)
    xyz = c["xyz"].astype(float); quat = c["quat_wxyz"].astype(float)
    layer = c["layer"].astype(str); morph = a["morphology"].astype(str)
    N = len(xyz)
    pick = rng.choice(N, min(n_cells, N), replace=False)
    print(f"[render] 표본 {len(pick)} / {N:,} 세포 (실제 구성비)")

    allpts, allcol = [], []
    for j, k in enumerate(pick):
        try:
            s = mt.load_swc(os.path.join(MORPH_DIR, morph[k] + ".swc"))
        except FileNotFoundError:
            continue
        w, _ = mt.transform(s["xyz"], mt.soma_center(s), quat[k], xyz[k])
        m = s["type"] != 2                      # 축삭 제외(가독)
        pts = w[m]
        if len(pts) > 350:
            pts = pts[rng.choice(len(pts), 350, replace=False)]
        allpts.append(pts)
        allcol.append(np.full(len(pts), LAYER_ORDER.index(layer[k])))
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(pick)}")
    P = np.vstack(allpts); C = np.concatenate(allcol)
    print(f"[render] 총 점 {len(P):,}")

    # 3D 회전 GIF
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for li, L in enumerate(LAYER_ORDER):
        m = C == li
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=1, c=LAYER_COLOR[L],
                       alpha=0.45)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    ax.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER],
              fontsize=8, loc="upper right")
    save_rotate_gif(fig, ax, os.path.join(FIG, "V2d_5_placement_dense.gif"),
                    title=f"V2d-5  slice400 세포 배치 ({len(pick)}개 표본, 수상, 층별 색) 회전")
    plt.close(fig)

    # top-view PNG
    fig, ax = plt.subplots(figsize=(9, 7))
    for li, L in enumerate(LAYER_ORDER):
        m = C == li
        if m.any():
            ax.scatter(P[m, 0], P[m, 2], s=1, c=LAYER_COLOR[L], alpha=0.4)
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
    ax.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER],
              fontsize=8)
    ax.set_title(f"V2d-5  slice400 세포 배치 위에서 본 그림 ({len(pick)}개 표본)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2d_5_placement_top.png"), dpi=130)
    plt.close(fig)
    print(f"[OK] -> {FIG}")


if __name__ == "__main__":
    main()
