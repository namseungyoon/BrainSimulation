# -*- coding: utf-8 -*-
"""[학습 Step 2] 왜 SR은 음성, 세포체는 양성인가 — sink + source 다이폴

논문 근거: Colbert & Levy 1992(SC->SR 음성 실험) · Taube 1988(CA1 sink/source)
배우는 것:
  - 흥분성 시냅스 = 전류가 세포 안으로(sink, I<0) -> 그 근처 전극 음성(-)
  - 그 전류는 어딘가로 나와야 함(source, I>0, 소마) -> 그 근처 양성(+)
  - 두 전류가 짝(다이폴)을 이루면 '극성반전'이 생긴다 (SR 음성 <-> 소마 양성)
  - sink만 있으면(전류보존 위반) 물리적으로 불가능 -> 반드시 sink=source 짝

가장 단순한 모형: 점전류 2개(sink 위, source 아래)로 fEPSP 부호를 이해한다.
실행: <ca1sim>/python.exe 12_lfp/study/s2_dipole.py
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
os.makedirs(FIG, exist_ok=True)

SIGMA = 0.3


def V_of_points(sources, X, Y):
    """sources=[(I_nA, x0, y0), ...] 가 만드는 세포외 전압(uV)을 (X,Y) 격자에서.
    3D 점전류, z=0 평면. r는 반경 1um로 하한(발산 방지)."""
    V = np.zeros_like(X, dtype=float)
    for I, x0, y0 in sources:
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        r = np.maximum(r, 1.0)
        V += I / (4.0 * np.pi * SIGMA * r)          # mV
    return V * 1e3                                   # uV


def main():
    # 다이폴: sink(흥분 시냅스, SR) 위 y=+120, source(귀환, 소마) 아래 y=-120
    y_sink, y_src = 120.0, -120.0
    dipole = [(-1.0, 0.0, y_sink), (+1.0, 0.0, y_src)]   # 전류 합 = 0 (전류보존)
    monopole = [(-1.0, 0.0, y_sink)]                     # sink만 (물리적으로 불가·비교용)

    # === 2D 필드맵 격자 ===
    gx = np.linspace(-250, 250, 220)
    gy = np.linspace(-350, 350, 300)
    X, Y = np.meshgrid(gx, gy)
    Vd = V_of_points(dipole, X, Y)

    # === 깊이 프로파일(전극을 x=40um 선 위에서 y로 훑기) ===
    xe = 40.0
    ye = np.linspace(-350, 350, 200)
    Vprof_d = V_of_points(dipole, np.full_like(ye, xe), ye)
    Vprof_m = V_of_points(monopole, np.full_like(ye, xe), ye)
    # 극성반전 깊이
    sgn = np.sign(Vprof_d)
    revs = [ye[k] for k in range(1, len(ye)) if sgn[k] != sgn[k - 1]]
    print(f"[다이폴] 전극선 x={xe}um: sink(y={y_sink}) 근처 최음 {Vprof_d.min():.2f}uV, "
          f"source(y={y_src}) 근처 최양 {Vprof_d.max():.2f}uV, 극성반전 y~{revs[0]:.0f}um" if revs else "")

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))

    # (A) 다이폴 필드맵
    a = ax[0]
    vmax = np.percentile(np.abs(Vd), 99)
    im = a.pcolormesh(X, Y, Vd, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    a.contour(X, Y, Vd, levels=[0], colors="k", linewidths=1.0, linestyles="--")
    a.plot(0, y_sink, "v", color="#1a5276", ms=13, markeredgecolor="w")
    a.text(12, y_sink, " sink (흥분 시냅스, SR)\n 전류 안으로 -> 음성", fontsize=8, va="center")
    a.plot(0, y_src, "^", color="#922b21", ms=13, markeredgecolor="w")
    a.text(12, y_src, " source (귀환, 소마)\n 전류 밖으로 -> 양성", fontsize=8, va="center")
    a.axhline(0, color="0.5", lw=0.5)
    fig.colorbar(im, ax=a, label="V (uV)  파랑=음(sink) / 빨강=양(source)")
    a.set_xlabel("x (um)"); a.set_ylabel("깊이 y (um)")
    a.set_title("(A) 다이폴 필드맵 — 점선=극성반전(V=0)")

    # (B) 깊이 프로파일 = fEPSP 극성반전
    b = ax[1]
    b.plot(Vprof_d, ye, "o-", color="#2c3e50", ms=3, lw=1.4)
    b.axvline(0, color="0.5", lw=0.8)
    b.fill_betweenx(ye, 0, Vprof_d, where=(Vprof_d < 0), color="#3498db", alpha=0.25)
    b.fill_betweenx(ye, 0, Vprof_d, where=(Vprof_d > 0), color="#e74c3c", alpha=0.25)
    b.axhline(y_sink, color="#1a5276", ls=":", lw=0.8); b.text(Vprof_d.min(), y_sink + 8, "SR(sink)", fontsize=8, color="#1a5276")
    b.axhline(y_src, color="#922b21", ls=":", lw=0.8); b.text(Vprof_d.max(), y_src - 18, "소마(source)", fontsize=8, color="#922b21", ha="right")
    if revs:
        b.axhline(revs[0], color="orange", lw=1); b.text(0.2, revs[0] + 6, f"반전 y~{revs[0]:.0f}", fontsize=8, color="#a04000")
    b.set_xlabel("세포외 전압 V (uV)"); b.set_ylabel("깊이 y (um)")
    b.set_title("(B) 깊이 프로파일 = fEPSP\nSR 음성 <-> 소마 양성 (극성반전)")

    # (C) sink만 vs 다이폴: 전류보존의 중요성
    c = ax[2]
    c.plot(Vprof_m, ye, color="#e67e22", lw=2, label="sink만 (전류보존 위반)")
    c.plot(Vprof_d, ye, color="#2c3e50", lw=2, label="sink+source 다이폴")
    c.axvline(0, color="0.5", lw=0.8)
    c.set_xlabel("V (uV)"); c.set_ylabel("깊이 y (um)")
    c.set_title("(C) 왜 짝이 필요한가\nsink만이면 전부 음성(반전 없음)")
    c.legend(fontsize=8, loc="lower right")

    fig.suptitle("학습 Step 2 — 왜 SR 음성 / 소마 양성인가: sink+source 다이폴 (Colbert&Levy 1992)",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "S2_dipole.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
