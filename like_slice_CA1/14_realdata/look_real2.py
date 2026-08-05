# -*- coding: utf-8 -*-
"""14_realdata/look_real2.py — 실측 fEPSP: 자극 아티팩트 블랭킹 후 확대

아티팩트(±1000µV·약 2ms)를 제외하고 fEPSP 성분만 y축 자동조정해 확인.
실행: <ca1sim>/py 14_realdata/look_real2.py
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"D:/Project_2025_2026_HIPPO/Workspace/HippocampalSignalProcessing/DATASET/ETRI"
NAMES = ["E17", "E45", "E55"]
BLANK = 2.0        # 자극 후 블랭킹(ms) — 아티팩트 제외
WIN = 20.0         # 분석창(ms)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
summary = []
for k, n in enumerate(NAMES):
    d = loadmat(os.path.join(SRC, f"fEPSP_{n}.mat"), squeeze_me=True)[f"fEPSP_{n}"]
    t = d[:, 0]; W = d[:, 1:]
    ia = int(np.argmax(np.abs(W[:, 0]))); t0 = t[ia]
    pre = t < t0 - 1.0
    Wb = W - W[pre].mean(axis=0)
    m = (t >= t0 + BLANK) & (t <= t0 + WIN)
    tt = t[m] - t0
    ax = axes[k]
    for j in range(Wb.shape[1]):
        ax.plot(tt, Wb[m, j], lw=0.7, alpha=0.5, color="0.55")
    mu = Wb[m].mean(axis=1); sd = Wb[m].std(axis=1)
    ax.fill_between(tt, mu - sd, mu + sd, color="#1f6fb2", alpha=0.18)
    ax.plot(tt, mu, lw=2.2, color="#1f6fb2", label=f"평균±SD (n={Wb.shape[1]})")
    ip = int(np.argmin(mu)); amp = mu[ip]; tpk = tt[ip]
    ax.plot(tpk, amp, "v", color="#c0392b", ms=10)
    # 20-80% 하강 slope (평균파형)
    lo, hi = 0.2 * amp, 0.8 * amp
    idx = np.where((mu[:ip + 1] <= lo) & (mu[:ip + 1] >= hi))[0]
    sl = np.polyfit(tt[idx], mu[idx], 1)[0] if len(idx) >= 2 else np.nan
    if len(idx) >= 2:
        ax.plot(tt[idx], np.polyval(np.polyfit(tt[idx], mu[idx], 1), tt[idx]), "r-", lw=2.5, alpha=0.85)
    ax.axhline(0, color="0.6", lw=0.6); ax.grid(alpha=0.3)
    ax.set_xlabel(f"자극 후 시간 (ms)  ·  0~{BLANK}ms 블랭킹")
    ax.set_ylabel("전위 (µV)" if k == 0 else "")
    ax.set_title(f"{n} — 진폭 {amp:.0f}µV @ +{tpk:.1f}ms\nslope {sl:.1f} µV/ms · 잡음 {W[pre].std():.1f}µV", fontsize=11)
    ax.legend(fontsize=8)
    summary.append((n, amp, tpk, sl, W[pre].std(), Wb.shape[1]))
    print(f"[{n}] 진폭 {amp:>8.1f}µV @ +{tpk:.1f}ms · slope {sl:>8.2f}µV/ms · 잡음 {W[pre].std():.1f}µV · sweep {Wb.shape[1]}", flush=True)

fig.suptitle(f"실측 MEA fEPSP (ETRI) — 자극 아티팩트 {BLANK}ms 블랭킹 후 · 10 kHz 샘플링 · 자극 t=10ms\n"
             "회색=개별 sweep · 파랑=평균±SD · 빨강=20–80% 하강 회귀(초기 slope)", fontsize=12.5, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(HERE, "figures", "REAL_fepsp_blanked.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
np.savez(os.path.join(HERE, "figures", "_real_summary.npz"),
         names=np.array([s[0] for s in summary]), amp=np.array([s[1] for s in summary]),
         tpk=np.array([s[2] for s in summary]), slope=np.array([s[3] for s in summary]),
         noise=np.array([s[4] for s in summary]), nsweep=np.array([s[5] for s in summary]))
