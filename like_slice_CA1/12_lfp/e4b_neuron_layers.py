# -*- coding: utf-8 -*-
"""12_lfp/e4b_neuron_layers.py  —  세포 층 + 뉴런 타입(12종) 정확 색상 지도

slice_cells.npz(17,647세포)를 두 뷰로:
 (A) 층 단면 뷰  — x=밴드 축, y=정규화 깊이 nd(SO→SP→SR→SLM 층 띠 음영) · 12 m-type 색
 (B) 전극면 뷰   — PC PCA 면투영 + MEA 3x8 전극 · 12 m-type 색
PC(89%)는 옅은 회색(작게), 인터뉴런 11종은 뚜렷한 색(크게). 층 라벨·개수 범례.
실행: <ca1sim>/python.exe 12_lfp/e4b_neuron_layers.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
PITCH, R_ON, NCOL, NROW = 200.0, 100.0, 8, 3

# m-type 색(층별로 계열 묶음; PC는 옅은 회색). 순서 = 범례 순서(층별)
MTYPE_COL = {
    "SO_OLM": "#1f4e79", "SO_Tri": "#2e86c1", "SO_BS": "#5dade2", "SO_BP": "#48c9b0",   # SO 청록계
    "SP_PC": "#d8d2c4",                                                                  # PC 옅은 회색
    "SP_PVBC": "#c0392b", "SP_CCKBC": "#e67e22", "SP_Ivy": "#b9770e",
    "SP_BS": "#8e44ad", "SP_AA": "#e84393",                                              # SP 억제 난색·보라
    "SR_SCA": "#27ae60",                                                                 # SR 초록
    "SLM_PPA": "#6e2c00",                                                                # SLM 갈색
}
# 층 띠 경계(nd) 및 색
LAYER_BANDS = [("SO", 0.00, 0.22, "#eaf2f8"), ("SP", 0.22, 0.42, "#fdf2e9"),
               ("SR", 0.42, 0.68, "#eafaf1"), ("SLM", 0.68, 0.85, "#f4ecf7")]


def place_mea(face):
    gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH
    gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
    G0 = np.column_stack([np.meshgrid(gx, gy)[0].ravel(), np.meshgrid(gx, gy)[1].ravel()])
    fc = face.mean(0); tree = cKDTree(face); best = (-1, None)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Grot = G0 @ Rm.T
        for dx in np.linspace(-400, 400, 9):
            for dy in np.linspace(-200, 200, 9):
                E = Grot + fc + [dx, dy]; on = int(np.sum(tree.query(E)[0] < R_ON))
                if on > best[0]:
                    best = (on, E.copy())
    return best[1]


def main():
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    xyz = d["xyz"].astype(float); mt = d["mtype"]; ly = d["layer"]; nd = d["nd"].astype(float)
    N = len(mt)
    Ppc = xyz[mt == "SP_PC"]; c0 = Ppc.mean(0)
    Vt = np.linalg.svd(Ppc - c0, full_matrices=False)[2]
    aband = (xyz - c0) @ Vt[0]                       # 밴드 축
    face = (xyz - c0) @ Vt[:2].T                     # 전극면
    facepc = (Ppc - c0) @ Vt[:2].T
    E = place_mea(facepc)
    counts = {m: int((mt == m).sum()) for m in MTYPE_COL}

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.32)

    # ---------- (A) 층 단면 뷰 (밴드축 × 깊이 nd) ----------
    axA = fig.add_subplot(gs[0])
    x0, x1 = aband.min(), aband.max()
    for name, lo, hi, col in LAYER_BANDS:
        axA.axhspan(lo, hi, color=col, zorder=0)
        axA.text(x1 * 0.995, (lo + hi) / 2, name, ha="right", va="center", fontsize=11,
                 fontweight="bold", color="0.35", zorder=1)
    # PC 먼저(옅게), 인터뉴런 위에(뚜렷)
    mpc = mt == "SP_PC"
    axA.scatter(aband[mpc][::6], nd[mpc][::6], s=2, color=MTYPE_COL["SP_PC"], alpha=0.5, zorder=2)
    for m in MTYPE_COL:
        if m == "SP_PC":
            continue
        mm = mt == m
        axA.scatter(aband[mm], nd[mm], s=16, color=MTYPE_COL[m], edgecolors="0.25",
                    linewidths=0.3, zorder=4, label=m)
    axA.set_ylim(0.85, 0.0)                           # SO 위, SLM 아래(해부 방향)
    axA.set_xlim(x0, x1)
    axA.set_xlabel("CA1 밴드 축 (µm)"); axA.set_ylabel("정규화 깊이 nd")
    axA.set_title("(A) 층 단면 뷰 — 깊이 nd로 SO→SP→SR→SLM 층 띠 · 12 m-type 정확 색상\n"
                  "PC(옅은 회색)는 SP에 조밀 · 인터뉴런은 층별로 분포", fontsize=11)

    # ---------- (B) 전극면 뷰 ----------
    axB = fig.add_subplot(gs[1])
    axB.scatter(face[mpc][::6, 0], face[mpc][::6, 1], s=2, color=MTYPE_COL["SP_PC"], alpha=0.4, zorder=2)
    for m in MTYPE_COL:
        if m == "SP_PC":
            continue
        mm = mt == m
        axB.scatter(face[mm, 0], face[mm, 1], s=16, color=MTYPE_COL[m], edgecolors="0.25",
                    linewidths=0.3, zorder=4)
    axB.scatter(E[:, 0], E[:, 1], s=90, marker="s", facecolor="none", edgecolors="k",
                linewidths=1.5, zorder=6)
    axB.set_aspect("equal"); axB.set_xlabel("면 가로 (µm)"); axB.set_ylabel("면 세로 (µm)")
    axB.set_title("(B) 전극면 뷰 (PC PCA 투영) + MEA 3×8(검정 사각) — 같은 12 m-type 색\n"
                  "이 면이 CSD·fEPSP를 계산하는 전극 평면", fontsize=11)

    # ---------- 범례(층별 그룹, 개수) ----------
    handles = []
    for m in MTYPE_COL:
        ei = "EXC" if m == "SP_PC" else "INH"
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=MTYPE_COL[m],
                              markeredgecolor="0.3", markersize=9,
                              label=f"{m}  {counts[m]:,} ({ei})"))
    fig.legend(handles=handles, loc="center right", fontsize=8.5, framealpha=0.95,
               title="뉴런 타입 (12 m-type)", bbox_to_anchor=(1.0, 0.5))

    fig.suptitle(f"뉴런 층·타입 지도 — CA1 like-slice {N:,}세포 (PC 15,723 + 인터뉴런 11종 1,924)  ·  slice_cells.npz",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    out = os.path.join(FIG, "E4b_neuron_layers.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)
    for name, lo, hi, _ in LAYER_BANDS:
        inb = (nd >= lo) & (nd < hi)
        print(f"  {name}: {int(inb.sum())}세포 (nd {lo}-{hi})", flush=True)


if __name__ == "__main__":
    main()
