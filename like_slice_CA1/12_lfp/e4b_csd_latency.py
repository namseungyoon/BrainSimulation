# -*- coding: utf-8 -*-
"""12_lfp/e4b_csd_latency.py  —  지연시간(latency)별 CSD 스냅샷 + 시간×위치 CSD 이미지

자극-정렬 CSD 기법: 유발 반응의 특정 지연시간(예: 피크·피크+10ms)의 전극값으로 CSD를 그림.
단일 유발 반응(_e4b_band_3x8.npz)에도 시간축이 있어 바로 가능(반복평균은 10초 데이터로 추후).
 (상단) 지연시간 6개의 2D 평활 CSD 스냅샷  (하단) 밴드중앙 행의 fEPSP(x,t)·CSD(x,t) 시간이미지.
실행: <ca1sim>/python.exe 12_lfp/e4b_csd_latency.py
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

D = np.load(os.path.join(FIG, "_e4b_band_3x8.npz"), allow_pickle=True)
t = D["t"]; Ve = D["Ve"]; E = D["E"]
xg = np.unique(E[:, 0]); yg = np.unique(E[:, 1])
NROW, NCOL = len(yg), len(xg)
order = np.lexsort((E[:, 0], E[:, 1]))
Vg = Ve[order].reshape(NROW, NCOL, -1)
xf = np.linspace(xg[0], xg[-1], 120)
yf = np.linspace(yg[0], yg[-1], 44)
iy0 = int(np.argmin(np.abs(yf - 0.0)))                  # 밴드 중앙 행


def lap(F, xs, ys):
    fy, fx = np.gradient(F, ys, xs)
    return np.gradient(fx, xs, axis=1) + np.gradient(fy, ys, axis=0)


def csd2d(f):
    spl = RectBivariateSpline(yg, xg, Vg[:, :, f], kx=2, ky=3)
    Vf = spl(yf, xf)
    return Vf, -SIGMA * lap(Vf, xf, yf) * 1e6           # µV field, A/m³


tp = float(t[int(np.argmax(np.abs(Ve).max(axis=0)))])   # 피크 지연
lat = [tp - 1.5, tp, tp + 3, tp + 6, tp + 11, tp + 18]
lat = [max(3.05, min(31, x)) for x in lat]
ilat = [int(np.argmin(np.abs(t - x))) for x in lat]

# 스냅샷 CSD·V
snaps = [csd2d(f) for f in ilat]
vS = np.percentile(np.abs([s[1] for s in snaps]), 99.5)
ext = [xg[0], xg[-1], yg[0], yg[-1]]

# 시간×위치 이미지 (밴드중앙 행)
win = (t >= 3.0) & (t <= 30.0)
iw = np.where(win)[0][::4]                               # 시간 다운샘플
tw = t[iw]
Vc = np.zeros((len(iw), len(xf))); Cc = np.zeros((len(iw), len(xf)))
for i, f in enumerate(iw):
    Vf, C = csd2d(f)
    Vc[i] = Vf[iy0]; Cc[i] = C[iy0]
vV = np.percentile(np.abs(Vc), 99.5); vC = np.percentile(np.abs(Cc), 99.5)

# ================= 그림 =================
fig = plt.figure(figsize=(15, 7.4))
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.28)

# (상단) 지연시간 스냅샷 6개
for k, (f, (Vf, C)) in enumerate(zip(ilat, snaps)):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(C, extent=ext, origin="lower", cmap="PRGn", vmin=-vS, vmax=vS, aspect="auto")
    ax.scatter(E[:, 0], E[:, 1], s=5, c="k", zorder=5)
    dl = t[f] - tp
    ax.set_title(f"t={t[f]:.1f}ms\n(피크{'+' if dl>=0 else ''}{dl:.1f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.set_ylabel("2D CSD\n스냅샷", fontsize=9)
cax = fig.add_axes([0.92, 0.60, 0.011, 0.28])
fig.colorbar(im, cax=cax, label="CSD (A/m³)")

# (하단 좌) fEPSP 시간×위치
axV = fig.add_subplot(gs[1, 0:3])
imV = axV.imshow(Vc.T, extent=[tw[0], tw[-1], xg[0], xg[-1]], origin="lower", cmap="RdBu_r",
                 vmin=-vV, vmax=vV, aspect="auto")
for x in lat:
    axV.axvline(x, color="0.35", lw=0.7, ls=":")
fig.colorbar(imV, ax=axV, label="fEPSP (µV)")
axV.set_xlabel("시간 (ms)"); axV.set_ylabel("밴드 위치 (µm)")
axV.set_title("(하단좌) 밴드중앙 행 fEPSP(위치×시간) · 점선=스냅샷 지연", fontsize=10)

# (하단 우) CSD 시간×위치
axC = fig.add_subplot(gs[1, 3:6])
imC = axC.imshow(Cc.T, extent=[tw[0], tw[-1], xg[0], xg[-1]], origin="lower", cmap="PRGn",
                 vmin=-vC, vmax=vC, aspect="auto")
for x in lat:
    axC.axvline(x, color="0.35", lw=0.7, ls=":")
fig.colorbar(imC, ax=axC, label="CSD (A/m³)")
axC.set_xlabel("시간 (ms)"); axC.set_ylabel("밴드 위치 (µm)")
axC.set_title("(하단우) 밴드중앙 행 CSD(위치×시간) · sink=보라 · 표준 laminar-style", fontsize=10)

fig.suptitle(f"E4b — 자극정렬 CSD: 지연시간별 스냅샷 + 시간×위치 이미지 (단일 유발, 피크 t={tp:.1f}ms)\n"
             f"'특정 지연시간(예: 피크+10ms)의 전극값으로 CSD' 기법 · 평면 MEA 표면 CSD · 10초 데이터로 반복평균 예정",
             fontsize=11, y=1.01)
fig.tight_layout(rect=[0, 0, 0.91, 0.93])
out = os.path.join(FIG, "E4b_csd_latency.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"[지연 스냅샷 ms] " + " ".join(f"{t[f]:.1f}" for f in ilat) + f" · 피크 {tp:.1f}", flush=True)
print(f"[스케일] |fEPSP|~{vV:.0f}µV · |CSD|~{vC:.0f}A/m³", flush=True)


if __name__ == "__main__":
    pass
