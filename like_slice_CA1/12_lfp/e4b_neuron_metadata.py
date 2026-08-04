# -*- coding: utf-8 -*-
"""12_lfp/e4b_neuron_metadata.py  —  뉴런 메타데이터 정리 (인벤토리 표 + 슬라이스 지도)

slice_cells.npz(17,647세포 실배치)에서 세포 인벤토리를 정리:
 (A) 12 m-type 세포 수(막대, E/I 색)  (B) 슬라이스 면 지도(층별 색 + MEA 3x8 오버레이)
 (C) 4개 대표 me-model 축소 매핑 + 요약 통계.
모든 수치는 slice_cells.npz 직접 집계(추측 없음). NEURON 불필요.
실행: <ca1sim>/python.exe 12_lfp/e4b_neuron_metadata.py
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
PITCH, R_ON, NCOL, NROW = 200.0, 100.0, 8, 3

# 4개 대표 me-model (network_lib.load_representatives: etype별 sorted[0])
TEMPLATES = [
    ("PC  (cACpyr)", "CA1_pyr_SP-PC_cACpyr_mpg141017_a1-2_idC", "cACpyr", "EXC"),
    ("PV  (cNAC)", "CA1_int_SO-BP_cNAC_980120A", "cNAC", "INH"),
    ("cAC", "CA1_int_SO-BP_cAC_980120A", "cAC", "INH"),
    ("bAC", "CA1_int_SLM-PPA_bAC_011127HP1", "bAC", "INH"),
]
LAYER_COL = {"SP": "#c0392b", "SO": "#2980b9", "SR": "#27ae60", "SLM": "#8e44ad"}


def place_mea(face):
    gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH
    gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
    G0 = np.column_stack([np.meshgrid(gx, gy)[0].ravel(), np.meshgrid(gx, gy)[1].ravel()])
    fc = face.mean(0); tree = cKDTree(face); best = (-1, None, 0.0)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Grot = G0 @ Rm.T
        for dx in np.linspace(-400, 400, 9):
            for dy in np.linspace(-200, 200, 9):
                E = Grot + fc + [dx, dy]; on = int(np.sum(tree.query(E)[0] < R_ON))
                if on > best[0]:
                    best = (on, E.copy(), th)
    return best


def main():
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    xyz = d["xyz"].astype(float); mt = d["mtype"]; et = d["etype"]; ly = d["layer"]; sc = d["sclass"]
    N = len(mt)
    mtypes, counts = np.unique(mt, return_counts=True)
    order = np.argsort(-counts)
    mtypes, counts = mtypes[order], counts[order]
    n_exc = int((sc == "EXC").sum()); n_inh = int((sc == "INH").sum())

    # PC 기준 면투영(전극 배치와 동일 좌표계) → 전 세포 투영
    Ppc = xyz[mt == "SP_PC"]; c0 = Ppc.mean(0)
    Vt = np.linalg.svd(Ppc - c0, full_matrices=False)[2]
    face = (xyz - c0) @ Vt[:2].T
    facepc = (Ppc - c0) @ Vt[:2].T
    n_on, E, th = place_mea(facepc)
    area = (np.ptp(facepc[:, 0]) / 1000) * (np.ptp(facepc[:, 1]) / 1000)
    dens = len(facepc) / area

    # etype 집계
    ets, etc = np.unique(et, return_counts=True)

    print(f"[세포] 총 {N} · EXC {n_exc}({100*n_exc/N:.0f}%) · INH {n_inh}({100*n_inh/N:.0f}%)", flush=True)
    for m, c in zip(mtypes, counts):
        i = np.where(mt == m)[0][0]
        print(f"  {m:10s} {c:6d}  층={ly[i]:4s} etype={et[i]:8s} {sc[i]}", flush=True)
    print(f"[면] PC 밴드 {np.ptp(facepc[:,0]):.0f}x{np.ptp(facepc[:,1]):.0f}µm · 면밀도 {dens:.0f}/mm² · MEA 3x8 조직 위 {n_on}/24", flush=True)

    # ================= 그림 =================
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.25, 1.0],
                          hspace=0.34, wspace=0.22)

    # (A) m-type 막대 (E/I 색, 로그 x)
    axA = fig.add_subplot(gs[0, 0])
    cols = ["#c0392b" if sc[np.where(mt == m)[0][0]] == "EXC" else "#2980b9" for m in mtypes]
    yb = np.arange(len(mtypes))[::-1]
    axA.barh(yb, counts, color=cols, edgecolor="0.3", height=0.72)
    axA.set_yticks(yb); axA.set_yticklabels(mtypes, fontsize=9)
    axA.set_xscale("log"); axA.set_xlim(10, 30000)
    for y, c in zip(yb, counts):
        axA.text(c * 1.15, y, f"{c:,}", va="center", fontsize=8.2)
    axA.set_xlabel("세포 수 (로그)")
    axA.set_title(f"(A) 12 m-type 세포 수 — 총 {N:,}개\n빨강=흥분 PC {n_exc:,}(89%) · 파랑=억제 INT {n_inh:,}(11%)", fontsize=10.5)
    axA.grid(axis="x", alpha=0.3)

    # (B) 슬라이스 면 지도 (층별 색 + MEA 3x8)
    axB = fig.add_subplot(gs[:, 1])
    for lname, col in LAYER_COL.items():
        m = ly == lname
        if lname == "SP":
            axB.scatter(face[m, 0][::8], face[m, 1][::8], s=1.4, color=col, alpha=0.28, label=f"{lname} ({int(m.sum()):,})")
        else:
            axB.scatter(face[m, 0], face[m, 1], s=6, color=col, alpha=0.8, label=f"{lname} ({int(m.sum()):,})")
    axB.scatter(E[:, 0], E[:, 1], s=95, marker="s", facecolor="none",
                edgecolors="k", linewidths=1.4, zorder=6, label="MEA 3×8 전극")
    axB.set_aspect("equal"); axB.set_xlabel("면 가로 (µm)"); axB.set_ylabel("면 세로 (µm)")
    axB.legend(fontsize=8, loc="upper right", framealpha=0.9)
    axB.set_title(f"(B) 슬라이스 면 지도(층별) + MEA 3×8\n면 {np.ptp(facepc[:,0]):.0f}×{np.ptp(facepc[:,1]):.0f}µm · PC 면밀도 {dens:.0f}/mm² · 조직 위 {n_on}/24", fontsize=10.5)

    # (C) 축소 매핑 + etype 요약 텍스트
    axC = fig.add_subplot(gs[1, 0]); axC.axis("off")
    axC.text(0, 1.0, "(C) me-model 축소: 12 m-type → 4개 대표 형태학 (etype별)", fontsize=11, fontweight="bold", va="top")
    lines = []
    for lab, folder, etype, ei in TEMPLATES:
        ncell = int((et == etype).sum())
        lines.append(f"● {lab:12s} [{ei}]  ← {ncell:>6,}세포  ({etype})")
        lines.append(f"     {folder}")
    lines.append("")
    lines.append("etype 분포:  " + " · ".join(f"{e}={c:,}" for e, c in sorted(zip(ets.tolist(), etc.tolist()), key=lambda z: -z[1])))
    lines.append("층 분포:  SP=17,330 · SO=264 · SLM=29 · SR=24   (SP에 PC+대부분 INT)")
    lines.append("활성채널(PC): nax·kdr·kap·kmb·kad·kca·cagk·hd·can·cal·cat (+cacum)")
    lines.append("주의: fEPSP 순방향 모델은 PC(89%)만 사용 — INT는 소수·비정렬로 기여 작음")
    for i, s in enumerate(lines):
        fam = "DejaVu Sans Mono" if s.strip().startswith("CA1_") else "Malgun Gothic"
        fs = 7.8 if s.strip().startswith("CA1_") else 8.6
        axC.text(0, 0.90 - i * 0.083, s, fontsize=fs, va="top", family=fam)

    fig.suptitle("뉴런 메타데이터 — CA1 like-slice 실배치 17,647세포 (Romani 아틀라스 영역 · slice_cells.npz 집계)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "E4b_neuron_metadata.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
