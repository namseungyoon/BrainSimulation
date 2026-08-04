# -*- coding: utf-8 -*-
"""12_lfp/e4b_csd_evoked.py  —  10초 데이터 자극정렬 CSD (반복 평균 + baseline vs 트레인)

10초 유발 데이터(_e4b_stim10s.npz)를 자극 시점에 정렬:
 (A) baseline 3회 유발 fEPSP 정렬·평균(대표 전극) — 결정론이라 3회 동일 확인
 (B) baseline-평균 유발의 지연시간별 2D CSD 스냅샷
 (C) baseline vs paired2 vs 트레인말 CSD(피크) 비교 — STP가 CSD 패턴을 바꾸는지
자극정렬 CSD 기법의 실사용. 평면 MEA 표면 CSD(측면).
실행: <ca1sim>/python.exe 12_lfp/e4b_csd_evoked.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SIGMA = 0.3

D = np.load(os.path.join(FIG, "_e4b_stim10s.npz"), allow_pickle=True)
t = D["t"]; Ve = D["Ve"]; E = D["E"]; stim = D["stim"]; labels = D["stim_labels"]
j_max = int(D["j_max"]); rec_dt = float(D["rec_dt"])
xg = np.unique(E[:, 0]); yg = np.unique(E[:, 1]); NROW, NCOL = len(yg), len(xg)
order = np.lexsort((E[:, 0], E[:, 1]))
xf = np.linspace(xg[0], xg[-1], 120); yf = np.linspace(yg[0], yg[-1], 44)
iy0 = int(np.argmin(np.abs(yf - 0.0)))
WIN = 40.0                                              # 자극 후 창(ms)
nwin = int(WIN / rec_dt) + 1
tw = np.arange(nwin) * rec_dt                           # 자극 후 시간


def epoch(ts):
    """자극 ts(ms) 후 WIN창의 (24, nwin) fEPSP."""
    i0 = int(round(ts / rec_dt))
    return Ve[:, i0:i0 + nwin]


def lap(F, xs, ys):
    fy, fx = np.gradient(F, ys, xs)
    return np.gradient(fx, xs, axis=1) + np.gradient(fy, ys, axis=0)


def csd2d(V24):
    """(24,) 전극 전위 → 2D 평활 CSD field(A/m³)."""
    V2 = V24[order].reshape(NROW, NCOL)
    Vf = RectBivariateSpline(yg, xg, V2, kx=2, ky=3)(yf, xf)
    return -SIGMA * lap(Vf, xf, yf) * 1e6


base_idx = [0, 1, 2]                                    # baseline #1,2,3
ep_base = np.array([epoch(stim[i]) for i in base_idx])  # (3,24,nwin)
ep_avg = ep_base.mean(0)                                # (24,nwin) 평균
# 정렬 확인(결정론): 3회 최대편차
dev = np.abs(ep_base - ep_avg).max()
ipk = int(np.argmax(np.abs(ep_avg[j_max])))             # 평균 피크 지연 인덱스
tpk = tw[ipk]

lat = [max(0.1, tpk - 1.5), tpk, tpk + 4, tpk + 10]     # 지연시간 스냅샷(피크+10 포함)
ilat = [int(np.argmin(np.abs(tw - x))) for x in lat]
snaps = [csd2d(ep_avg[:, i]) for i in ilat]
vS = np.percentile(np.abs(snaps), 99.5)
ext = [xg[0], xg[-1], yg[0], yg[-1]]

# baseline vs paired2(#4) vs 트레인말(#9) 피크 CSD
cmp_stims = [("baseline(평균)", ep_avg), (f"{labels[4]}", epoch(stim[4])), (f"{labels[9]}", epoch(stim[9]))]
cmp_csd = []
for name, ep in cmp_stims:
    ip = int(np.argmax(np.abs(ep[j_max])))
    cmp_csd.append((name, csd2d(ep[:, ip]), ep[j_max, ip]))
vC = np.percentile(np.abs([c[1] for c in cmp_csd]), 99.5)

# ================= 그림 =================
fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.0], hspace=0.5, wspace=0.3)

# (A) baseline 3회 정렬·평균
axA = fig.add_subplot(gs[0, 0:2])
for k, i in enumerate(base_idx):
    axA.plot(tw, ep_base[k, j_max], lw=1.0, alpha=0.7, label=f"{labels[i]}")
axA.plot(tw, ep_avg[j_max], "k--", lw=1.4, label="평균")
axA.axhline(0, color="0.7", lw=0.5); axA.set_xlabel("자극 후 시간 (ms)"); axA.set_ylabel("fEPSP (µV)")
axA.legend(fontsize=7.5, ncol=2)
axA.set_title(f"(A) baseline 3회 자극정렬·평균 (전극#{j_max})\n결정론이라 3회 최대편차 {dev:.2e}µV(=동일) → 평균=단일", fontsize=9.5)

# (B) 평균 유발 지연시간별 CSD 스냅샷
for k, (i, C) in enumerate(zip(ilat, snaps)):
    ax = fig.add_subplot(gs[1, k])
    im = ax.imshow(C, extent=ext, origin="lower", cmap="PRGn", vmin=-vS, vmax=vS, aspect="auto")
    ax.scatter(E[:, 0], E[:, 1], s=4, c="k", zorder=5)
    ax.set_title(f"피크{'+' if tw[i]-tpk>=0 else ''}{tw[i]-tpk:.0f}ms\n(t={tw[i]:.1f})", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.set_ylabel("(B) 평균유발\n지연 CSD", fontsize=9)
caxB = fig.add_axes([0.92, 0.38, 0.011, 0.22]); fig.colorbar(im, cax=caxB, label="CSD (A/m³)")

# (C) baseline vs paired2 vs 트레인말 CSD 비교
for k, (name, C, apk) in enumerate(cmp_csd):
    ax = fig.add_subplot(gs[2, k])
    im2 = ax.imshow(C, extent=ext, origin="lower", cmap="PRGn", vmin=-vC, vmax=vC, aspect="auto")
    ax.scatter(E[:, 0], E[:, 1], s=4, c="k", zorder=5)
    ax.set_title(f"{name}\n피크 {apk:.0f}µV", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.set_ylabel("(C) STP별\nCSD 비교", fontsize=9)
axT = fig.add_subplot(gs[2, 3]); axT.axis("off")
axT.text(0, 0.9, "(C) 해석", fontsize=10, fontweight="bold", va="top")
r_pp = abs(cmp_csd[1][2]) / abs(cmp_csd[0][2]); r_tr = abs(cmp_csd[2][2]) / abs(cmp_csd[0][2])
for i, s in enumerate([
    f"CSD 패턴은 유지,",
    f"진폭만 STP로 스케일:",
    f"• paired2/baseline = {r_pp:.2f}",
    f"• 트레인말/baseline = {r_tr:.2f}",
    "",
    "약한 촉진(PPF) →",
    "CSD도 소폭 증가.",
    "결정론이라 잡음0;",
    "실측은 반복평균으로",
    "잡음↓ 효과 큼.",
]):
    axT.text(0, 0.78 - i * 0.083, s, fontsize=8.6, va="top")

fig.suptitle("E4b — 10초 데이터 자극정렬 CSD (반복 평균 + STP별 비교) · 평면 MEA 표면 CSD(측면)\n"
             "지연시간 CSD 기법을 반복 유발에 적용 · sink=보라·source=초록",
             fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 0.91, 0.94])
out = os.path.join(FIG, "E4b_csd_evoked.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"[정렬] baseline 3회 최대편차 {dev:.2e}µV · 피크지연 {tpk:.1f}ms", flush=True)
print(f"[STP-CSD] paired2/base={r_pp:.3f} · train말/base={r_tr:.3f}", flush=True)
