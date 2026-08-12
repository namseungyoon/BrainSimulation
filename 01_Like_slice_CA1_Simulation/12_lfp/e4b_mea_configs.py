# -*- coding: utf-8 -*-
"""12_lfp/e4b_mea_configs.py  —  여러 MEA 격자 구성 vs CA1 밴드 적합성 비교 (가능성 탐색)

실제 PC(SP) 밴드 footprint에 대해 여러 격자(8x8, 3x8, 4x8, 6x6 @200um)를 각각
최적 배치(밴드 장축 정렬·중심/회전 탐색)해서 조직 위 전극 수를 비교.
레이아웃(적합성)만 계산 — fEPSP 맵은 별도(e4b_band.py). cKDTree로 빠르게.

주의: 가능성 탐색용. 실제 MEA 실험의 슬라이스 크기·측정부위 확정 시 그 사양으로 대체.
실행: <ca1sim>/python.exe 12_lfp/e4b_mea_configs.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
os.makedirs(FIG, exist_ok=True)
PITCH, D_ELEC, R_ON = 200.0, 10.0, 100.0
# (이름, 장축 열 수, 단축 행 수)
CONFIGS = [("8×8", 8, 8), ("3×8", 8, 3), ("4×8", 8, 4), ("6×6", 6, 6)]


def grid(ncol, nrow):
    gx = (np.arange(ncol) - (ncol - 1) / 2) * PITCH        # 장축
    gy = (np.arange(nrow) - (nrow - 1) / 2) * PITCH        # 단축
    X, Y = np.meshgrid(gx, gy)
    return np.column_stack([X.ravel(), Y.ravel()])


def place_optim(G0, face, tree, fc):
    """중심·회전 탐색으로 조직 위 전극 최대 배치."""
    best = (-1, None, 0.0)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Gr = G0 @ Rm.T
        for dx in np.linspace(-400, 400, 9):
            for dy in np.linspace(-200, 200, 9):
                E = Gr + fc + [dx, dy]
                dist, _ = tree.query(E)
                on = int(np.sum(dist < R_ON))
                if on > best[0]:
                    best = (on, E.copy(), th)
    return best


def main():
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    xyz = d["xyz"].astype(float); mtype = d["mtype"]
    P = xyz[mtype == "SP_PC"]
    c0 = P.mean(axis=0); Vt = np.linalg.svd(P - c0, full_matrices=False)[2]
    face = (P - c0) @ Vt[:2].T                              # 밴드 장축=x
    tree = cKDTree(face)
    fc = face.mean(axis=0)
    W1, W2 = np.ptp(face[:, 0]), np.ptp(face[:, 1])
    print(f"[CA1 밴드] SP_PC {len(P)}개 · 밴드 {W1:.0f}(장축) x {W2:.0f}(단축) um", flush=True)

    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4.6 * len(CONFIGS), 5.2))
    for ax, (name, ncol, nrow) in zip(axes, CONFIGS):
        G0 = grid(ncol, nrow)
        span_l, span_s = (ncol - 1) * PITCH, (nrow - 1) * PITCH
        n_on, E, th = place_optim(G0, face, tree, fc)
        ntot = ncol * nrow
        print(f"[{name}] {ncol}×{nrow}={ntot}전극 · footprint {span_l:.0f}×{span_s:.0f}um · 회전{np.rad2deg(th):.0f}° "
              f"-> 조직 위 {n_on}/{ntot} ({100*n_on/ntot:.0f}%)", flush=True)
        over = tree.query(E)[0] < R_ON
        ax.scatter(face[::6, 0], face[::6, 1], s=1, color="0.75", alpha=0.4)
        ax.scatter(E[over, 0], E[over, 1], s=60, color="#c0392b", edgecolors="k", linewidths=0.5, zorder=5, label=f"조직 위 {n_on}")
        ax.scatter(E[~over, 0], E[~over, 1], s=45, color="0.6", edgecolors="k", linewidths=0.3, zorder=4, label=f"밖 {ntot-n_on}")
        ax.set_aspect("equal"); ax.set_title(f"{name} ({ntot}전극)\nfootprint {span_l:.0f}×{span_s:.0f}µm · {n_on}/{ntot} 조직 위")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("장축 (µm)")
    axes[0].set_ylabel("단축 (µm)")
    fig.suptitle(f"E4b 가능성 탐색 — MEA 격자 구성별 CA1 밴드({W1:.0f}×{W2:.0f}µm, 간격{PITCH:.0f}µm) 적합성 비교\n"
                 f"밴드가 얇아(단축 {W2:.0f}µm) 단축 행 적을수록 유리 — 3×8이 좁은 밴드에 최적",
                 fontsize=11, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG, "E4b_mea_configs.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
