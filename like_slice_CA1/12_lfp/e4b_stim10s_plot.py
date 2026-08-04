# -*- coding: utf-8 -*-
"""12_lfp/e4b_stim10s_plot.py  —  10초 자극-반응 fEPSP 결과 그림 (_e4b_stim10s.npz 로드)

패널: (A) 10초 전체 개관(대표 전극) + 자극 표시  (B) baseline 유발 fEPSP 확대(24전극)
      (C) paired-pulse + 트레인 확대(단기가소성)  (D) 자극별 유발 진폭(STP)  (E) baseline 공간맵.
실행: <ca1sim>/python.exe 12_lfp/e4b_stim10s_plot.py
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
D = np.load(os.path.join(FIG, "_e4b_stim10s.npz"), allow_pickle=True)
t = D["t"]; Ve = D["Ve"]; E = D["E"]; over = D["over"]; stim = D["stim"]
labels = D["stim_labels"]; amp = D["amp_stim"]; j_max = int(D["j_max"]); Npc = int(D["npc"])
NELEC = Ve.shape[0]
vmax = float(np.abs(Ve).max())
vm = Ve[j_max]
ts = t / 1000.0                                       # s
base_med = np.median(np.abs(amp[over, 0]))
ppr = abs(amp[j_max, 4]) / abs(amp[j_max, 3])
trn = abs(amp[j_max, 9]) / abs(amp[j_max, 5])


def zoom(ax, t0, t1, elecs, rel=None, colors=None):
    m = (t >= t0) & (t <= t1)
    x = t[m] - (rel if rel is not None else 0)
    for k, e in enumerate(elecs):
        c = colors[k] if colors else ("#c0392b" if e == j_max else "0.7")
        lw = 1.9 if e == j_max else 0.8
        ax.plot(x, Ve[e, m], color=c, lw=lw, zorder=5 if e == j_max else 2)
    ax.axhline(0, color="0.7", lw=0.5)


fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.42, wspace=0.2)

# (A) 10초 개관
axA = fig.add_subplot(gs[0, :])
axA.plot(ts, vm, color="#c0392b", lw=0.8)
for s in stim:
    axA.axvline(s / 1000, color="0.6", lw=0.6, ls=":")
axA.axvspan(0.5, 5.3, color="#ecf0f1", alpha=0.6, zorder=0)
axA.axvspan(6.8, 7.3, color="#d6eaf8", alpha=0.7, zorder=0)
axA.axvspan(8.9, 9.4, color="#fdebd0", alpha=0.7, zorder=0)
axA.text(2.9, vmax * 0.9, "baseline\n(1·3·5s 단일)", ha="center", fontsize=8.5, color="0.35")
axA.text(7.05, vmax * 0.9, "paired-pulse\n(50ms)", ha="center", fontsize=8.5, color="#1f6fb2")
axA.text(9.15, vmax * 0.9, "20Hz 트레인×5", ha="center", fontsize=8.5, color="#b9722e")
axA.set_xlim(0, 10); axA.set_ylim(-vmax * 1.05, vmax * 0.55)
axA.set_xlabel("시간 (s)"); axA.set_ylabel("전극 fEPSP (µV)")
axA.set_title(f"(A) 10초 자극-반응 개관 — 대표 전극 #{j_max} (밴드 중심) · 자극 10회의 유발 fEPSP", fontsize=11)

# (B) baseline 유발 확대(24전극)
axB = fig.add_subplot(gs[1, 0])
zoom(axB, 995, 1045, list(np.where(over)[0]), rel=1000)
axB.set_xlim(-5, 45); axB.set_xlabel("자극 후 시간 (ms)"); axB.set_ylabel("fEPSP (µV)")
axB.set_title(f"(B) baseline #1 유발 fEPSP — 24전극 동시\n중앙 |{base_med:.0f}|µV · 최대(빨강) |{abs(amp[j_max,0]):.0f}|µV", fontsize=10.5)

# (C) PP + 트레인 확대(대표 전극)
axC = fig.add_subplot(gs[1, 1])
m2 = (t >= 6900) & (t <= 9300)
axC.plot(t[m2] / 1000, Ve[j_max, m2], color="#7d3c98", lw=1.4)
for s in stim[3:]:
    axC.axvline(s / 1000, color="0.6", lw=0.5, ls=":")
axC.axhline(0, color="0.7", lw=0.5)
axC.set_xlabel("시간 (s)"); axC.set_ylabel("fEPSP (µV)")
axC.set_title(f"(C) 단기가소성 — paired-pulse + 20Hz 트레인 (전극#{j_max})\nPPR={ppr:.2f} · 트레인 말/초={trn:.2f}", fontsize=10.5)

# (D) 자극별 유발 진폭 (STP)
axD = fig.add_subplot(gs[2, 0])
a = np.abs(amp[j_max])
cols = ["#7f8c8d"] * 3 + ["#2980b9"] * 2 + ["#e67e22"] * 5
axD.bar(np.arange(len(a)), a, color=cols, edgecolor="0.3")
axD.set_xticks(np.arange(len(a))); axD.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
axD.set_ylabel("유발 |fEPSP| (µV)")
axD.set_title(f"(D) 자극별 유발 진폭(전극#{j_max}) — 단기가소성\n회색=baseline · 파랑=PP · 주황=트레인", fontsize=10.5)
axD.grid(axis="y", alpha=0.3)

# (E) baseline 공간맵
axE = fig.add_subplot(gs[2, 1])
amax = np.abs(amp[:, 0]).max()
sc = axE.scatter(E[:, 0], E[:, 1], c=amp[:, 0], cmap="RdBu_r", vmin=-amax, vmax=amax,
                 s=160, edgecolors=["k" if o else "0.6" for o in over], linewidths=1.2)
fig.colorbar(sc, ax=axE, label="baseline 유발 fEPSP (µV)")
axE.set_aspect("equal"); axE.set_xlabel("면 가로 (µm)"); axE.set_ylabel("세로 (µm)")
axE.set_title("(E) baseline 유발 fEPSP 공간맵 (24전극)", fontsize=10.5)

fig.suptitle(f"E4b — 전극 배치 후 10초 SC 자극-반응 유발 fEPSP  (실제 CA1 밴드 PC {Npc:,}개·MoI·정렬+동기 상한값)\n"
             f"baseline 유발 중앙 |{base_med:.0f}|µV(실측 0.1~1mV 저역대) · PPR={ppr:.2f} · 20Hz 트레인 억압 말/초={trn:.2f}",
             fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(FIG, "E4b_stim10s.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
