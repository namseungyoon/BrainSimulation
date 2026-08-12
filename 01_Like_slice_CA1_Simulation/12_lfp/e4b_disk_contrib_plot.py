# -*- coding: utf-8 -*-
"""12_lfp/e4b_disk_contrib_plot.py  —  원형 전극 + 24전극 뉴런 기여 분포 그림

_e4b_disk_contrib.npz 로드:
 (A) 24전극별 국소 인터뉴런 조성(11종 스택 막대) — '어떤 뉴런'
 (B) 24전극별 PC 수(150µm 내) vs 신호기여 유효 Neff — '몇 개'
 (C) 전극 공간맵(3×8) — Neff 색  (D) 디스크 vs 점 진폭(원형 전극 효과 정량)
실행: <ca1sim>/python.exe 12_lfp/e4b_disk_contrib_plot.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
D = np.load(os.path.join(FIG, "_e4b_disk_contrib.npz"), allow_pickle=True)
E = D["E"]; over = D["over"]; amp_pt = D["amp_pt"]; amp_disk = D["amp_disk"]
Neff = D["Neff"]; r90 = D["r90"]; census = D["census"]; mtypes = list(D["mtypes"])
R_CEN = float(D["r_census"]); D_ELEC = float(D["d_elec"]); NELEC = len(E)

MTYPE_COL = {
    "SO_OLM": "#1f4e79", "SO_Tri": "#2e86c1", "SO_BS": "#5dade2", "SO_BP": "#48c9b0",
    "SP_PC": "#d8d2c4", "SP_PVBC": "#c0392b", "SP_CCKBC": "#e67e22", "SP_Ivy": "#b9770e",
    "SP_BS": "#8e44ad", "SP_AA": "#e84393", "SR_SCA": "#27ae60", "SLM_PPA": "#6e2c00",
}
INT = [m for m in MTYPE_COL if m != "SP_PC"]
ipc = mtypes.index("SP_PC")
pc_cnt = census[:, ipc]
int_cnt = np.array([[census[j, mtypes.index(m)] for m in INT] for j in range(NELEC)])  # (24,11)
dd = 100 * np.abs(amp_disk - amp_pt) / np.abs(amp_pt)

fig = plt.figure(figsize=(15, 9.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.36, wspace=0.30)
x = np.arange(NELEC)

# (A) 인터뉴런 조성 스택
axA = fig.add_subplot(gs[0, :])
bottom = np.zeros(NELEC)
for k, m in enumerate(INT):
    axA.bar(x, int_cnt[:, k], bottom=bottom, color=MTYPE_COL[m], label=m, edgecolor="0.35", linewidth=0.3)
    bottom += int_cnt[:, k]
axA.set_xticks(x); axA.set_xticklabels([f"#{i}" for i in range(NELEC)], fontsize=7, rotation=90)
axA.set_ylabel(f"인터뉴런 수 ({R_CEN:.0f}µm 내)")
axA.set_title(f"(A) 24전극별 국소 뉴런 조성 — '어떤 뉴런' (반경 {R_CEN:.0f}µm 내 인터뉴런 11종 스택)\n"
              f"※ fEPSP 신호원은 PC만(각 전극 PC 중앙 {int(np.median(pc_cnt))}개) — 인터뉴런은 해부적 근접(신호 미기여)", fontsize=10.5)
axA.legend(ncol=6, fontsize=7.5, loc="upper center", framealpha=0.9)
axA.set_ylim(0, int_cnt.sum(1).max() * 1.35)
axA.grid(axis="y", alpha=0.3)

# (B) PC 수 vs 유효 Neff
axB = fig.add_subplot(gs[1, 0])
axB.bar(x - 0.2, pc_cnt, 0.4, color="#d8d2c4", edgecolor="0.4", label=f"PC 수({R_CEN:.0f}µm내)")
axB.bar(x + 0.2, Neff, 0.4, color="#c0392b", alpha=0.8, label="유효 Neff(신호기여)")
axB.set_xticks(x[::2]); axB.set_xticklabels([f"#{i}" for i in range(0, NELEC, 2)], fontsize=7)
axB.set_ylabel("세포 수"); axB.legend(fontsize=8)
axB.set_title(f"(B) 전극별 '몇 개' — 국소 PC(중앙 {int(np.median(pc_cnt))}) vs 유효 Neff(중앙 {int(np.median(Neff[over]))})\n"
              "Neff≫국소 PC: 신호는 멀리서도 광역 적분", fontsize=10)
axB.grid(axis="y", alpha=0.3)

# (C) 전극 공간맵 (Neff 색)
axC = fig.add_subplot(gs[1, 1])
sc = axC.scatter(E[:, 0], E[:, 1], c=Neff, s=240, cmap="viridis",
                 edgecolors=["k" if o else "0.6" for o in over], linewidths=1.3)
for j in range(NELEC):
    axC.text(E[j, 0], E[j, 1], str(j), fontsize=6, ha="center", va="center", color="w")
fig.colorbar(sc, ax=axC, label="유효 Neff")
axC.set_aspect("equal"); axC.set_xlabel("면 가로 (µm)"); axC.set_ylabel("세로 (µm)")
axC.set_title("(C) 전극 공간맵(3×8) — 신호기여 유효세포 Neff", fontsize=10)

# (D) 디스크 vs 점
axD = fig.add_subplot(gs[1, 2])
axD.scatter(np.abs(amp_pt), np.abs(amp_disk), s=40, c="#2980b9", edgecolors="0.3", zorder=3)
lim = [0, np.abs(amp_pt).max() * 1.08]
axD.plot(lim, lim, "k--", lw=1, alpha=0.6)
axD.set_xlim(lim); axD.set_ylim(lim)
axD.set_xlabel("점 전극 |진폭| (µV)"); axD.set_ylabel("원형(10µm) 전극 |진폭| (µV)")
axD.set_title(f"(D) 원형 전극 재구현 vs 점\n지름 {D_ELEC:.0f}µm · 차이 중앙 {np.median(dd):.3f}%·최대 {dd.max():.3f}%\n(원거리 소스라 사실상 동일)", fontsize=10)
axD.grid(alpha=0.3)

fig.suptitle("E4b — 원형(디스크) 전극 재구현 + 24전극 뉴런 기여 분포 (몇 개·어떤 뉴런)  ·  fEPSP 소스=PC만",
             fontsize=12.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(FIG, "E4b_disk_contrib.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
