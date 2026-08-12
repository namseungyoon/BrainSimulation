# -*- coding: utf-8 -*-
"""12_lfp/e4b_mea_layout.py  —  슬라이스 면적 vs MEA(8x8, 간격200um, 직경10um) 적합성

슬라이스(slice400, 17,647세포)의 실제 cut-face(절단면) 크기를 PCA로 구하고,
가상 MEA(전극 직경 10um · 간격 200um · 8x8 정사각 = 1400um span)를 겹쳐
슬라이스에 들어가는지(64전극 중 몇 개가 조직 위에 오는지) 확인·시각화.

실행: <ca1sim>/python.exe 12_lfp/e4b_mea_layout.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# MEA 사양
D_ELEC = 10.0        # 전극 직경 um
PITCH = 200.0        # 간격 um
NGRID = 8            # 8x8


def main():
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    xyz = d["xyz"].astype(float)                 # (N,3) um
    layer = d["layer"]
    N = xyz.shape[0]

    # --- PCA: cut-face(큰 2축) + 두께(작은 축) ---
    c = xyz.mean(axis=0)
    X = xyz - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = Vt                                      # (3,3) 주축
    proj = X @ pcs.T                              # (N,3) 주축 좌표
    ext = proj.max(axis=0) - proj.min(axis=0)     # 각 주축 범위
    order = np.argsort(-ext)                       # 큰 축부터
    a1, a2, a3 = order                             # a1,a2=면, a3=두께
    face = proj[:, [a1, a2]]                        # (N,2) 면 좌표
    W1, W2, THK = ext[a1], ext[a2], ext[a3]
    print(f"[슬라이스] 세포 {N} · cut-face {W1:.0f} x {W2:.0f} um · 두께 {THK:.0f} um", flush=True)
    print(f"[슬라이스] 면 대략 면적(외접사각) {W1*W2/1e6:.2f} mm^2", flush=True)

    # --- MEA 격자 (면 중심에 배치) ---
    span = (NGRID - 1) * PITCH                     # 1400 um
    fc = face.mean(axis=0)
    gs = np.arange(NGRID) * PITCH - span / 2
    gx, gy = np.meshgrid(fc[0] + gs, fc[1] + gs)
    ex = np.column_stack([gx.ravel(), gy.ravel()])  # (64,2) 전극 중심
    print(f"[MEA] 8x8 · 간격 {PITCH:.0f}um · 전극 {D_ELEC:.0f}um · 전체 span {span:.0f}x{span:.0f} um ({span/1000:.1f}x{span/1000:.1f} mm)", flush=True)

    # --- 각 전극이 조직 위인가: 반경 R 내 세포 존재 ---
    R_NEAR = PITCH / 2                              # 100um
    over = np.zeros(len(ex), bool)
    for i, e in enumerate(ex):
        over[i] = np.any(np.sum((face - e) ** 2, axis=1) < R_NEAR ** 2)
    n_over = int(over.sum())
    # 면 외접사각 대비 MEA span
    fits_bbox = (span <= W1) and (span <= W2)
    print(f"[적합성] 조직 위 전극 {n_over}/64 ({100*n_over/64:.0f}%) · MEA span {span:.0f} vs 면 {W1:.0f}x{W2:.0f}"
          f" -> 외접사각 {'들어감' if fits_bbox else '초과(한 변 부족)'}", flush=True)

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))

    # (좌) 면 위 세포 + MEA 격자
    a = ax[0]
    colmap = {"SP": "#e67e22", "SO": "#3498db", "SR": "#2ecc71", "SLM": "#9b59b6"}
    for lay, col in colmap.items():
        mm = layer == lay
        a.scatter(face[mm, 0], face[mm, 1], s=2, color=col, alpha=0.35, label=f"{lay}({mm.sum()})")
    for i, e in enumerate(ex):
        a.add_patch(plt.Circle((e[0], e[1]), max(D_ELEC / 2, 6), color="k" if over[i] else "red", zorder=5))
    a.add_patch(Rectangle((ex[:, 0].min() - D_ELEC, ex[:, 1].min() - D_ELEC),
                          span + 2 * D_ELEC, span + 2 * D_ELEC, fill=False, ec="k", lw=1.2, ls="--"))
    a.set_aspect("equal")
    a.set_xlabel("면 가로 (µm)"); a.set_ylabel("면 세로 (µm)")
    a.set_title(f"(좌) 슬라이스 cut-face {W1:.0f}×{W2:.0f}µm + MEA 8×8(간격{PITCH:.0f}µm)\n"
                f"검정=조직 위 전극 {n_over}/64 · 빨강=조직 밖")
    a.legend(fontsize=8, loc="upper right", markerscale=3)

    # (우) 축척 비교 막대
    b = ax[1]
    b.barh(["슬라이스 면 가로", "슬라이스 면 세로", "MEA span(1400µm)", "슬라이스 두께"],
           [W1, W2, span, THK], color=["#e67e22", "#e67e22", "#c0392b", "#7f8c8d"])
    for i, v in enumerate([W1, W2, span, THK]):
        b.text(v + 30, i, f"{v:.0f}µm", va="center", fontsize=9)
    b.axvline(span, color="#c0392b", ls=":", lw=1)
    b.set_xlabel("길이 (µm)")
    b.set_title("(우) 크기 비교 — MEA span vs 슬라이스 면/두께")

    verdict = ("MEA가 슬라이스 면 안에 들어감" if fits_bbox and n_over >= 60 else
               ("일부만 조직 위(면이 좁거나 CA1 곡선)" if n_over >= 20 else "MEA가 슬라이스보다 큼(안 들어감)"))
    fig.suptitle(f"E4b 계획 — 슬라이스 면적 vs MEA(8×8·간격200µm·직경10µm) 적합성:  {verdict}",
                 fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "E4b_mea_layout.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
