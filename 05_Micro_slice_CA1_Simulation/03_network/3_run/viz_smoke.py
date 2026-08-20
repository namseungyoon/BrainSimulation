# -*- coding: utf-8 -*-
"""
03_network/3_run/viz_smoke.py  —  스모크 구동 시각화

scratch/mpi_smoke_viz.npz 를 읽어 "자극을 어디에 줬고 어떻게 발화했나"를 그린다.
(a) 공간지도: 자극(E3·구동 시냅스·반경) + 발화세포(빨강)/침묵(회색)
(b) 래스터: 스파이크 시각 vs E3거리 (발화 전파)
(c) 대표세포 소마전압 (안정화→자극→발화)
실행: python 03_network/3_run/viz_smoke.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "figures")
d = np.load(os.path.join(ROOT, "scratch", "mpi_smoke_viz.npz"), allow_pickle=True)

cu = d["cell_uvw"]; fired = d["fired"]; e3 = d["e3"]; su = d["stim_uvw"]
R = float(d["radius"]); STIM = float(d["stim_t"]); SETTLE = float(d["settle"])
spk_t = d["spk_t"]; spk_id = d["spk_id"]; gid = d["cell_gid"]
tt = d["trace_t"]; tv = d["trace_v"]; target = int(d["target"])
cU, cR = cu[:, 0], cu[:, 1]

fig = plt.figure(figsize=(16, 7.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1])
axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, 1])

# (a) 공간지도
if len(su):
    s = np.random.default_rng(0).choice(len(su), min(8000, len(su)), replace=False)
    axA.scatter(su[s, 0], su[s, 1], s=3, c="#55A868", alpha=0.18, linewidths=0, label="자극 시냅스(구동섬유)")
axA.scatter(cU[~fired], cR[~fired], s=55, c="#cccccc", edgecolors="gray", linewidths=0.5, label="침묵 세포")
axA.scatter(cU[fired], cR[fired], s=70, c="#C44E52", edgecolors="black", linewidths=0.6, label="발화 세포")
axA.scatter([e3[0]], [e3[1]], s=420, marker="*", c="red", edgecolors="black", zorder=8)
axA.annotate("E3 자극", (e3[0], e3[1]), fontsize=11, ha="center", va="bottom", xytext=(0, 14), textcoords="offset points")
axA.add_patch(Circle((e3[0], e3[1]), R, fill=False, ec="#2222aa", lw=1.8, ls="--"))
axA.set_xlabel("종축 u (µm)"); axA.set_ylabel("층관통 r (µm)")
axA.set_title(f"(a) 자극 위치 + 발화 공간지도\n발화 {int(fired.sum())}/{len(gid)}세포 · 자극반경 {R:.0f}µm")
axA.legend(loc="upper right", fontsize=9); axA.set_aspect("equal")

# (b) 래스터 (E3 거리순)
dist = np.hypot(cU - e3[0], cR - e3[1])
order = {int(g): dist[i] for i, g in enumerate(gid)}
if len(spk_t):
    sd = np.array([order.get(int(i), 0) for i in spk_id])
    axB.scatter(spk_t - SETTLE, sd, s=8, c="#C44E52", alpha=0.7, linewidths=0)
axB.axvline(STIM - SETTLE, ls=":", color="blue", lw=1.5, label="자극")
axB.set_xlim(-5, 80); axB.set_xlabel("자극기준 시간 (ms)"); axB.set_ylabel("E3 거리 (µm)")
axB.set_title("(b) 발화 래스터 (E3 거리순)"); axB.legend(fontsize=8); axB.grid(alpha=0.3)

# (c) 대표세포 전압
if len(tt):
    axC.plot(tt - SETTLE, tv, color="#4C72B0", lw=0.8)
axC.axvline(STIM - SETTLE, ls=":", color="blue", lw=1.5)
axC.set_xlim(-20, 80); axC.set_xlabel("자극기준 시간 (ms)"); axC.set_ylabel("소마 전압 (mV)")
axC.set_title(f"(c) 대표세포(gid {target}) 소마전압 — 안정화→자극→응답"); axC.grid(alpha=0.3)

fig.suptitle("스모크 구동 시각화 — SC 국소자극(E3)이 만든 발화", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "smoke_drive.png"), dpi=130)
print(f"[그림] -> {FIG}/smoke_drive.png")
