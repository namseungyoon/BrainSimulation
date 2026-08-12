# -*- coding: utf-8 -*-
"""14_realdata/look_real2.py — 실측 fEPSP: 자극 아티팩트 블랭킹 후 확대

아티팩트(±1000µV·약 2ms)를 제외하고 fEPSP 성분만 y축 자동조정해 확인.
실행: <ca1sim>/py 14_realdata/look_real2.py

★기울기는 **시뮬과 같은 자**로 잰다 — `13_net_fepsp/mea_postproc.measure_fepsp`.
  실측과 시뮬을 나란히 놓고 비교할 것이므로 두 쪽이 다른 정의를 쓰면 비교 자체가 성립하지 않는다.
  기준선은 이미 자극 전 구간 평균으로 빼 놓았으므로 `base=0.0`으로 고정한다
  (자극 아티팩트 꼬리가 +2 ms에서도 E45 기준 +286 µV 남아 있어, 창 앞쪽을 기준선으로 쓰면 안 된다).
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "13_net_fepsp"))
from mea_postproc import measure_fepsp, SLOPE_METHOD    # noqa: E402  — 공용 자
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
    fe = measure_fepsp(tt, mu, float(tt[0]), WIN - BLANK, 0.0, base=0.0)
    amp = fe["amp"]; tpk = fe["tpk"]; sl = fe["slope"]
    ax.plot(tpk, amp, "v", color="#c0392b", ms=10)
    # 초기 기울기 = 0.6·진폭 / (t80 − t20).  두 교차점을 잇는 선으로 그린다.
    t20, t80 = fe["t20"], fe["t80"]
    ax.plot([t20, t80], [0.2 * amp, 0.8 * amp], "r-", lw=2.5, alpha=0.9)
    ax.plot([t20, t80], [0.2 * amp, 0.8 * amp], "o", color="#c0392b", ms=4)
    ax.axhline(0, color="0.6", lw=0.6); ax.grid(alpha=0.3)
    ax.set_xlabel(f"자극 후 시간 (ms)  ·  0~{BLANK}ms 블랭킹")
    ax.set_ylabel("전위 (µV)" if k == 0 else "")
    ax.set_title(f"{n} — 진폭 {amp:.0f}µV @ +{tpk:.1f}ms\n"
                 f"기울기 {sl:.1f} µV/ms ({SLOPE_METHOD}) · 잡음 {W[pre].std():.1f}µV", fontsize=11)
    ax.legend(fontsize=8)
    summary.append((n, amp, tpk, sl, W[pre].std(), Wb.shape[1], fe["slope_legacy"], fe["n_band"]))
    print(f"[{n}] 진폭 {amp:>8.1f}µV @ +{tpk:.1f}ms · 기울기 {sl:>8.2f}µV/ms({SLOPE_METHOD}) "
          f"[옛 표본회귀 {fe['slope_legacy']:>8.2f} · 띠표본 {fe['n_band']}개] · "
          f"잡음 {W[pre].std():.1f}µV · sweep {Wb.shape[1]}", flush=True)

fig.suptitle(f"실측 MEA fEPSP (ETRI) — 자극 아티팩트 {BLANK}ms 블랭킹 후 · 10 kHz 샘플링 · 자극 t=10ms\n"
             "회색=개별 sweep · 파랑=평균±SD · 빨강=20→80% 교차시각 초기 기울기 (시뮬과 동일한 자)",
             fontsize=12.5, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = os.path.join(HERE, "figures", "REAL_fepsp_blanked.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
np.savez(os.path.join(HERE, "figures", "_real_summary.npz"),
         names=np.array([s[0] for s in summary]), amp=np.array([s[1] for s in summary]),
         tpk=np.array([s[2] for s in summary]), slope=np.array([s[3] for s in summary]),
         noise=np.array([s[4] for s in summary]), nsweep=np.array([s[5] for s in summary]),
         slope_legacy=np.array([s[6] for s in summary]), n_band=np.array([s[7] for s in summary]),
         slope_method=SLOPE_METHOD)
