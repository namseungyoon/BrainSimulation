# -*- coding: utf-8 -*-
"""14_realdata/look_real.py — 실측 MEA fEPSP 원파형 첫 확인 (E9 시작)

ETRI 실측 데이터(DATASET/ETRI/fEPSP_E*.mat)를 있는 그대로 그려서
자극 아티팩트 구간·fEPSP 성분·sweep 변동을 눈으로 확인한다.
자동 측정 전에 반드시 이 단계를 거친다(아티팩트 오염 방지).
실행: <ca1sim>/py 14_realdata/look_real.py
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

fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
for k, n in enumerate(NAMES):
    d = loadmat(os.path.join(SRC, f"fEPSP_{n}.mat"), squeeze_me=True)[f"fEPSP_{n}"]
    t = d[:, 0]; W = d[:, 1:]
    ia = int(np.argmax(np.abs(W[:, 0])))          # 자극 아티팩트
    pre = t < t[ia] - 1.0
    base = W[pre].mean(axis=0)
    Wb = W - base                                  # sweep별 기준선 제거
    # (상단) 전체 30ms
    ax = axes[0, k]
    for j in range(Wb.shape[1]):
        ax.plot(t, Wb[:, j], lw=0.6, alpha=0.5, color="0.5")
    ax.plot(t, Wb.mean(axis=1), lw=2, color="#c0392b", label="평균")
    ax.axvspan(t[ia] - 0.3, t[ia] + 1.5, color="#f5b7b1", alpha=0.6, zorder=0)
    ax.text(t[ia] + 0.6, ax.get_ylim()[1] * 0.85, "자극\n아티팩트", fontsize=8, ha="center", color="#922b21")
    ax.set_title(f"{n} — 전체 30ms · sweep {Wb.shape[1]}개", fontsize=11)
    ax.set_xlabel("시간 (ms)"); ax.set_ylabel("전위 (µV)" if k == 0 else "")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (하단) 자극 후 확대
    ax2 = axes[1, k]
    m = (t >= t[ia] - 1) & (t <= t[ia] + 22)
    for j in range(Wb.shape[1]):
        ax2.plot(t[m] - t[ia], Wb[m, j], lw=0.7, alpha=0.55, color="0.55")
    mu = Wb[m].mean(axis=1)
    ax2.plot(t[m] - t[ia], mu, lw=2.2, color="#1f6fb2", label="평균")
    ax2.axvspan(-0.3, 1.5, color="#f5b7b1", alpha=0.6, zorder=0)
    ax2.axhline(0, color="0.6", lw=0.6)
    ip = int(np.argmin(mu)); tt = (t[m] - t[ia])
    ax2.plot(tt[ip], mu[ip], "v", color="#c0392b", ms=9)
    ax2.annotate(f"평균 최저 {mu[ip]:.0f}µV\n@ +{tt[ip]:.1f}ms", (tt[ip], mu[ip]),
                 textcoords="offset points", xytext=(8, -14), fontsize=8.5, color="#c0392b")
    ax2.set_xlabel("자극 후 시간 (ms)"); ax2.set_ylabel("전위 (µV)" if k == 0 else "")
    ax2.set_title(f"{n} — 자극 후 확대 (기준선 제거)", fontsize=10.5)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    print(f"[{n}] 자극 t={t[ia]:.1f}ms · 자극전 잡음 {W[pre].std():.1f}µV · "
          f"평균파형 최저 {mu[ip]:.1f}µV @ +{tt[ip]:.1f}ms", flush=True)

fig.suptitle("실측 MEA fEPSP 원파형 (ETRI) — 자동 측정 전 육안 확인\n"
             "회색=개별 sweep · 굵은선=평균 · 분홍=자극 아티팩트 구간(측정에서 제외 필요)",
             fontsize=12.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "figures", "REAL_raw_traces.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
