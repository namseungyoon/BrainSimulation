# -*- coding: utf-8 -*-
"""
02_neurons/2_placement/make_allcells_3d.py  —  2-2 보조: 5,610 전세포 실제형태 3D

세포 원장 5,610개 **전부**의 실제 형태(.swc, 수상돌기)를 배치+회전해 soma 창
800(층관통)×500(종축)×400(두께) 안에 3D 렌더한다. 세그먼트 총량이 6,600만이라
세포당 ~250점으로 다운샘플한 점구름으로 표현(빨강=추체 / 파랑=억제).
결과: figures/2-2_allcells_3d.png (2각도) · (옵션) 회전 GIF

재료: data/derived/window_cells.npz · config/window_layout.json · data/morphology_library
실행: python 02_neurons/2_placement/make_allcells_3d.py [--gif]
"""
import os
import sys
import json
import logging

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
KPTS = 250   # 세포당 다운샘플 점수


def load_dend_pts(path, k=KPTS):
    rows = np.loadtxt(path, comments="#")
    typ = rows[:, 1].astype(int)
    m = (typ == 1) | (typ == 3) | (typ == 4)   # soma+basal+apical (axon 제외)
    pts = rows[m, 2:5]
    if len(pts) > k:
        pts = pts[np.linspace(0, len(pts) - 1, k).astype(int)]
    return pts


def box_edges(cx, cy, cz, hx, hy, hz):
    c = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    E = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b]) for a, b in E]


def build_cloud():
    d = np.load(NPZ, allow_pickle=True)
    xyz = d["xyz"]; Q = d["orientation_wxyz"]; mt = d["mtype"]; morph = d["morphology"].astype(str)
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])
    Mrows = np.column_stack([L, R, Tk])

    cache = {}
    US, RS, WS, ISPC = [], [], [], []
    is_pc = mt == "SP_PC"
    for i in range(len(xyz)):
        mm = morph[i]
        if mm not in cache:
            cache[mm] = load_dend_pts(os.path.join(LIB, mm + ".swc"))
        pts = cache[mm]
        world = xyz[i] + Rot.from_quat(Q[i][[1, 2, 3, 0]]).apply(pts)
        loc = (world - seed) @ Mrows
        US.append(loc[:, 0]); RS.append(loc[:, 1]); WS.append(loc[:, 2])
        ISPC.append(np.full(len(pts), is_pc[i]))
    U = np.concatenate(US); Rr = np.concatenate(RS); W = np.concatenate(WS)
    pc = np.concatenate(ISPC).astype(bool)
    print(f"[전세포 점구름] {len(xyz):,}세포 · 점 {len(U):,}개 (세포당 ~{KPTS})")
    return U, Rr, W, pc, cfg


def main():
    U, Rr, W, pc, cfg = build_cloud()
    w = cfg["window_um"]; c = w["center_local"]
    ecfg = cfg["electrodes"]; elecs = ecfg["list"]; face_w = ecfg["mea_face_w_um"]
    hx, hy, hz = w["long"] / 2, w["radial"] / 2, w["thick"] / 2

    def render(ax):
        ax.scatter(U[pc], Rr[pc], W[pc], s=0.25, c="#C44E52", alpha=0.10,
                   linewidths=0, rasterized=True)
        ax.scatter(U[~pc], Rr[~pc], W[~pc], s=0.25, c="#3B75AF", alpha=0.14,
                   linewidths=0, rasterized=True)
        for p, q in box_edges(c["u"], c["r"], c["w"], hx, hy, hz):
            ax.plot(*zip(p, q), color="black", lw=1.0, alpha=0.7)
        for e in elecs:
            mk = "*" if e.get("role") == "stim" else "s"
            ax.scatter([e["u"]], [e["r"]], [face_w], s=160, marker=mk, c="lime",
                       edgecolors="black", zorder=10)
        ax.set_xlabel("종축 u (500)"); ax.set_ylabel("층관통 r (800)"); ax.set_zlabel("두께 w (400)")
        ax.set_box_aspect((w["long"], w["radial"], w["thick"]))

    fig = plt.figure(figsize=(20, 9))
    for k, (el, az) in enumerate([(18, -60), (18, 30)], 1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        render(ax); ax.view_init(elev=el, azim=az)
    fig.legend(handles=[Patch(color="#C44E52", label="추체 SP_PC (5,040)"),
                        Patch(color="#3B75AF", label="억제뉴런 (570)"),
                        plt.Line2D([], [], marker="*", ls="", mfc="lime", mec="black", ms=12, label="자극전극"),
                        plt.Line2D([], [], marker="s", ls="", mfc="lime", mec="black", ms=9, label="기록전극")],
               loc="lower center", ncol=4, fontsize=10)
    fig.suptitle("2-2  전세포 실제형태 3D — 5,610개 전부 (수상돌기 점구름) · soma 창 800×500×400µm", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIG, "2-2_allcells_3d.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[2-2] 저장 -> {out}")

    if "--gif" in sys.argv:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(111, projection="3d")
        render(ax)
        fig.suptitle("2-2  전세포 실제형태 3D — 5,610개 전부", fontsize=12)
        ani = FuncAnimation(fig, lambda i: ax.view_init(18, i * 4), frames=90, interval=80)
        gout = os.path.join(FIG, "2-2_allcells_3d.gif")
        ani.save(gout, writer=PillowWriter(fps=12), dpi=80); plt.close(fig)
        print(f"[2-2] GIF -> {gout}")


if __name__ == "__main__":
    main()
