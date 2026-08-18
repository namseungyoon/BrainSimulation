# -*- coding: utf-8 -*-
"""
01_tissue/4_vectorize/vectorize.py  —  1-4: 좌표·방향장 벡터화 (1-4)

확정 창(층관통_v1) 안 세포의 방향(orientation quaternion)에서 **방사 방향장**
(정단 = R·[0,1,0], SP→SR/SLM 방향)을 계산·시각화·검증한다.
검증(1-4): 방사벡터가 (1) 종축에 수직, (2) 층관통축(SO→SLM)과 정렬.

재료: config/window_layout.json · circuit nodes.h5 · slice400.nrrd
실행: python 01_tissue/4_vectorize/vectorize.py
"""
import os
import glob
import json
import logging

import numpy as np
import h5py
import nrrd
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find(p):
    return sorted(glob.glob(os.path.join(DATA, "**", glob.escape(p)), recursive=True), key=len)[0]


def decode(g, name):
    lib = [s.decode() if isinstance(s, bytes) else s for s in g["@library"][name][:]]
    return np.array(lib, dtype=object)[g[name][:]]


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; w = cfg["window_um"]; c = w["center_local"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])

    mask, h = nrrd.read(find(os.path.join("slices", "slice400.nrrd")))
    origin = np.asarray(h["space origin"], float); vs = float(h["space directions"][0][0])
    nx, ny, nz = mask.shape
    with h5py.File(find(os.path.join("hippocampus_neurons", "nodes.h5")), "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        quat = np.stack([g[f"orientation_{a}"][:] for a in "wxyz"], 1)
        layer = decode(g, "layer")

    idx = np.floor((xyz - origin) / vs).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    ins = np.zeros(len(xyz), bool); ii = idx[ok]
    ins[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sl = np.where(ins)[0]
    d = xyz[sl] - seed
    u = d @ L; r = d @ R; ww = d @ Tk
    inwin = ((np.abs(u - c["u"]) <= w["long"] / 2) & (np.abs(r - c["r"]) <= w["radial"] / 2) &
             (np.abs(ww - c["w"]) <= w["thick"] / 2))
    win = sl[inwin]
    Cu, Cr, Lw = u[inwin], r[inwin], layer[win]
    Q = quat[win]

    # 방사(정단) 방향 = R(quat)·[0,1,0]
    radial = Rot.from_quat(Q[:, [1, 2, 3, 0]]).apply([0.0, 1.0, 0.0])
    rad_u = radial @ L; rad_r = radial @ R    # 국소 프레임 성분
    print(f"[창 세포] {len(win):,}개")
    print(f"[1-4] 방사·종축 정렬 |rad·û| 중앙 = {np.median(np.abs(rad_u)):.3f} (0에 가까울수록 종축 수직)")
    print(f"[1-4] 방사·층관통 rad·r̂ 중앙 = {np.median(rad_r):.3f} (+1에 가까울수록 SO→SLM 정렬)")
    print(f"[1-4] 층관통 정렬 세포 비율(rad·r̂>0.7) = {100*np.mean(rad_r>0.7):.1f}%")

    fig_vectors(Cu, Cr, rad_u, rad_r, Lw, c, w, len(win))
    print(f"\n[1-4] 그림 저장 -> {FIG}/1-4_radial_field.png")


def fig_vectors(u, r, rad_u, rad_r, Lw, c, w, n, n_arrow=1200):
    rng = np.random.default_rng(0)
    s = rng.choice(len(u), min(n_arrow, len(u)), replace=False)
    fig, ax = plt.subplots(figsize=(11, 9))
    for Ln in LAYER_ORDER:
        m = Lw == Ln
        if m.any():
            ax.scatter(u[m], r[m], s=6, c=LC[Ln], alpha=0.35, label=Ln)
    ax.quiver(u[s], r[s], rad_u[s], rad_r[s], color="black", alpha=0.55,
              scale=32, width=0.003, headwidth=3)
    ax.add_patch(Rectangle((c["u"] - w["long"] / 2, c["r"] - w["radial"] / 2),
                           w["long"], w["radial"], fill=False, ec="black", lw=2, ls="--"))
    ax.set_aspect("equal"); ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm, SP=0)")
    ax.set_title(f"1-4  방사(정단) 방향장 — 창 세포 {n:,}개 · 화살표=R·[0,1,0] (SO→SLM)")
    ax.legend(handles=[Patch(facecolor=LC[v], label=v) for v in LAYER_ORDER], title="층", loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1-4_radial_field.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
