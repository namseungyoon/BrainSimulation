# -*- coding: utf-8 -*-
"""
01_tissue/2_bbox/electrode_layout.py  —  Stage 2 보조: MEA 자극/기록 전극 위치 (후보1)

축 배정: 종축(long, SC섬유·전극라인)=800 · 층관통(radial, SP+SR)=500 · 두께=400.
전극을 긴 축(종축 800)에 배치 → 스팬 400 + 사방 여유 200µm(바깥 전극 기여반경 ~200µm 확보).
  - 자극(Stim): SR
  - 기록(Rec) : SR, 종축 200µm 간격. 자극에 가까울수록 응답 큼. 깊이 mid-SR(sink 최대).
※ 정량 응답은 E4(구동 후) forward 계산. 여기선 생리 기반 배치.

실행: python 01_tissue/2_bbox/electrode_layout.py
"""
import os
import glob
import logging

import numpy as np
import h5py
import nrrd
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(HERE, "figures")

W_LONG, W_RADIAL, W_THICK = 800.0, 500.0, 400.0
R_CENTER = 175.0
PITCH = 200.0
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find(p):
    return sorted(glob.glob(os.path.join(DATA, "**", p), recursive=True), key=len)[0]


def decode(g, name):
    lib = [s.decode() if isinstance(s, bytes) else s for s in g["@library"][name][:]]
    return np.array(lib, dtype=object)[g[name][:]]


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v


def main():
    mask, h = nrrd.read(find(os.path.join("slices", "slice400.nrrd")))
    origin = np.asarray(h["space origin"], float); vsize = float(h["space directions"][0][0])
    nx, ny, nz = mask.shape
    with h5py.File(find(os.path.join("hippocampus_neurons", "nodes.h5")), "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        quat = np.stack([g[f"orientation_{c}"][:] for c in "wxyz"], 1)
        layer = decode(g, "layer")

    idx = np.floor((xyz - origin) / vsize).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    inside = np.zeros(len(xyz), bool); ii = idx[ok]
    inside[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sl = np.where(inside)[0]
    Cs, Ls, Qs = xyz[sl], layer[sl], quat[sl]
    c0 = Cs.mean(0)
    _, _, Vt = np.linalg.svd(Cs - c0, full_matrices=False)
    spans = (Cs - c0) @ Vt.T
    long_dir = Vt[int(np.argmax(spans.max(0) - spans.min(0)))]
    u = (Cs - c0) @ long_dir; u_c = np.median(u)
    band = np.abs(u - u_c) <= W_LONG / 2
    sp_band = band & (Ls == "SP")
    radial_dir = unit(Rot.from_quat(Qs[sp_band][:, [1, 2, 3, 0]]).apply([0, 1, 0]).mean(0))
    radial_dir = unit(radial_dir - (radial_dir @ long_dir) * long_dir)
    seed = Cs[sp_band].mean(0)
    d = Cs - seed
    l = d @ long_dir; r = d @ radial_dir
    inwin = ((np.abs(l) <= W_LONG / 2) & (np.abs(r - R_CENTER) <= W_RADIAL / 2) &
             (np.abs(d @ unit(np.cross(long_dir, radial_dir))) <= W_THICK / 2))
    lw, rw, Lw = l[inwin], r[inwin], Ls[inwin]

    m = Lw == "SP"
    sp_hi = np.percentile(rw[m], 95) if m.sum() > 3 else 46.0
    r_top = R_CENTER + W_RADIAL / 2
    sr_mid = (sp_hi + r_top) / 2                     # SP상단 ~ 창상단 중점 = mid-SR
    margin = (W_LONG - 2 * PITCH) / 2                # 바깥 전극 ~ 창 가장자리 여유
    print(f"[층 반경(µm)] SP 상단≈{sp_hi:.0f}  창 상단={r_top:.0f}  → mid-SR r≈{sr_mid:.0f}")

    electrodes = [
        ("Stim", -PITCH, sr_mid, "자극(SR)"),
        ("Rec1", 0.0,     sr_mid, "기록1(SR, +200µm) ← 최강응답"),
        ("Rec2", +PITCH,  sr_mid, "기록2(SR, +400µm)"),
    ]
    print(f"\n[전극] 종축(800) 위 배치 · 스팬 {2*PITCH:.0f}µm · 사방 여유 {margin:.0f}µm (기여반경 확보)")
    for name, el, er, desc in electrodes:
        print(f"   {name:5s} 종축 l={el:+5.0f}  층관통 r={er:+5.0f} µm   {desc}")

    fig_layout(lw, rw, Lw, electrodes, sp_hi, sr_mid, margin)
    print(f"\n[그림] 1-2_electrode_layout.png 저장 -> {FIG_DIR}")


def fig_layout(lw, rw, Lw, electrodes, sp_hi, sr_mid, margin):
    fig, ax = plt.subplots(figsize=(12, 7))
    for L in LAYER_ORDER:
        m = Lw == L
        if m.any():
            ax.scatter(lw[m], rw[m], s=14, c=LAYER_COLOR[L], label=L, alpha=0.8)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axhline(sr_mid, color=LAYER_COLOR["SR"], ls=":", lw=1)
    ax.text(-W_LONG / 2, sr_mid + 10, "mid-SR (fEPSP sink)", color="#2f6b45", fontsize=9)
    # 여유 표시(바깥 전극 바깥쪽)
    ax.axvspan(-W_LONG / 2, -W_LONG / 2 + margin, color="gray", alpha=0.08)
    ax.axvspan(W_LONG / 2 - margin, W_LONG / 2, color="gray", alpha=0.08)
    for name, el, er, desc in electrodes:
        mk = "P" if name == "Stim" else "o"
        col = "red" if name == "Stim" else "black"
        ax.scatter([el], [er], s=280, marker=mk, c=col, edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(name, (el, er), textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=11, fontweight="bold", color=col)
    ax.set_xlabel("종축 l (µm, SC섬유·전극라인, 800축)")
    ax.set_ylabel("층관통 r (µm, SP=0, +는 SR)")
    ax.set_title(f"후보1  MEA 전극 — 자극(SR)+기록2(SR,200µm간격)  ·  사방 여유 {margin:.0f}µm(회색)\n"
                 "긴 축(종축)에 배치 → 바깥 전극 기여반경 확보")
    ax.legend(title="층", loc="lower right"); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "1-2_electrode_layout.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
