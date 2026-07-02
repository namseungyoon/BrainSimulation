# -*- coding: utf-8 -*-
"""
07_connectivity/render_box.py  —  touch 커넥텀 하위부피 박스 시각화

touch_connectome.py 와 동일한 박스(중심 추체 기준 가로500×두께100µm×전층) 세포를
slice400 안에서 표시. 어느 부위인지 확인용.
  - V3b_4_box_location.png : 위에서 본 slice400(회색)+박스 세포(층별 색)+박스 외곽
  - V3b_5_box_3d.gif       : 박스 세포 소마 3D (층별 색) 회전
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
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "lib"))
from gif_util import save_rotate_gif            # noqa: E402
import morph_transform as mt                    # noqa: E402

CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
FIG = os.path.join(HERE, "figures")
HALF_T1, HALF_T2 = 250.0, 50.0
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def main():
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); quat = c["quat_wxyz"].astype(float)
    mt_us = c["mtype"].astype(str); layer = c["layer"].astype(str)
    pc = np.where(mt_us == "SP_PC")[0]
    c0 = pc[np.argmin(np.linalg.norm(xyz[pc] - xyz[pc].mean(0), axis=1))]
    R = mt.quat_to_R(quat[c0])
    radial = R.apply([0., 1., 0.]); t1 = R.apply([1., 0., 0.]); t2 = R.apply([0., 0., 1.])
    rel = xyz - xyz[c0]
    h = rel @ t1; th = rel @ t2; rd = rel @ radial
    inbox = (np.abs(h) < HALF_T1) & (np.abs(th) < HALF_T2)
    sub = np.where(inbox)[0]
    print(f"[box] 박스 세포 {len(sub)} (층 {dict(zip(*np.unique(layer[sub], return_counts=True)))})")

    # 박스 8꼭짓점(월드) = c0 + a*t1 + b*t2 + d*radial
    rmin, rmax = rd[sub].min(), rd[sub].max()
    corners = []
    for sa in (-HALF_T1, HALF_T1):
        for sb in (-HALF_T2, HALF_T2):
            for sd in (rmin, rmax):
                corners.append(xyz[c0] + sa * t1 + sb * t2 + sd * radial)
    corners = np.array(corners)

    # ---- 위치 지도 (top-view x-z) ----
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xyz[:, 0], xyz[:, 2], s=1, c="#dddddd", alpha=0.4, label="slice400 전체")
    for L in LAYER_ORDER:
        m = inbox & (layer == L)
        if m.any():
            ax.scatter(xyz[m, 0], xyz[m, 2], s=8, c=LAYER_COLOR[L], label=f"{L}({m.sum()})")
    # 박스 외곽(꼭짓점 투영 convex hull 간단히 min-max 사각형 대신 점만)
    ax.scatter(corners[:, 0], corners[:, 2], s=40, marker="s",
               edgecolor="k", facecolor="none", label="박스 꼭짓점")
    ax.scatter(xyz[c0, 0], xyz[c0, 2], s=80, marker="*", c="k", label="박스 중심")
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"V3b-4  touch 커넥텀 하위부피 박스 위치 ({len(sub)}세포)\n"
                 "가로500×두께100µm×전층 · 위에서 본 slice400")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3b_4_box_location.png"), dpi=130)
    plt.close(fig)

    # ---- 박스 세포 소마 3D 회전 GIF ----
    fig = plt.figure(figsize=(8, 7))
    axg = fig.add_subplot(111, projection="3d")
    for L in LAYER_ORDER:
        m = inbox & (layer == L)
        if m.any():
            axg.scatter(xyz[m, 0], xyz[m, 1], xyz[m, 2], s=6, c=LAYER_COLOR[L], label=L)
    # 박스 모서리 선
    axg.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=30, c="k", marker="s")
    axg.set_xlabel("x"); axg.set_ylabel("y"); axg.set_zlabel("z")
    axg.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER], fontsize=8)
    save_rotate_gif(fig, axg, os.path.join(FIG, "V3b_5_box_3d.gif"),
                    title=f"V3b-5  하위부피 박스 세포 소마 3D ({len(sub)}개, 층별 색) 회전")
    plt.close(fig)
    print(f"[OK] -> {FIG}")


if __name__ == "__main__":
    main()
