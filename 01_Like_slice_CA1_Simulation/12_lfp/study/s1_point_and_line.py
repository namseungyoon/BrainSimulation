# -*- coding: utf-8 -*-
"""[학습 Step 1] 전류 하나가 만드는 세포외 전압 — 점전류원(PSA) & 선전류원(LSA)

논문: Ness 2020 리뷰(원리) · Holt & Koch 1999(LSA 공식)
배우는 것:
  (A) 점전류원 V(r) = I / (4*pi*sigma*r)  -> 거리에 반비례(1/r)로 약해진다
  (B) sigma(조직 전도도)가 V를 어떻게 스케일하나
  (C) 선전류원(LSA) vs 점전류원(PSA): 멀면 같고, 가까우면 다르다(LSA가 더 정확)

이 두 공식이 우리 계산기(lfp_calc.py)의 심장이다. 여기선 '전류 1개'로 감을 잡는다.
실행: <ca1sim>/python.exe 12_lfp/study/s1_point_and_line.py
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
LFP = os.path.dirname(HERE)                 # 12_lfp/
sys.path.insert(0, LFP)
from lfp_calc import psa_matrix, lsa_matrix  # 우리가 만든 계산기 재사용

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def point_V(I_nA, r_um, sigma):
    """점전류원 세포외 전압(mV). 단위 nA, um, S/m -> mV (환산 1e-3 내장)."""
    return I_nA / (4.0 * np.pi * sigma * r_um)


def main():
    sigma = 0.3          # S/m (표준 조직 전도도)
    I = 1.0              # nA (전류 꼭지 하나)

    # === (A) 거리에 따른 감쇠: 1/r ===
    r = np.linspace(5, 1000, 400)          # um
    Va = point_V(I, r, sigma) * 1e3        # uV
    v100 = point_V(I, 100.0, sigma) * 1e3
    print(f"[A] 점전류원 I=1nA, sigma=0.3: r=100um -> V={v100:.4f} uV (해석식 1/(4*pi*0.3*100))")

    # === (B) sigma(조직 전도도)의 영향 ===
    sigmas = [(0.15, "묽은 조직 0.15"), (0.3, "표준 0.3"), (1.5, "식염수 1.5")]

    # === (C) 선전류원(LSA) vs 점전류원(PSA): 20um 세그먼트, 전극을 옆으로 ===
    # 세그먼트: (-10,0,0)~(10,0,0), 길이 20um, 반경 0.5um
    seg = dict(p0=np.array([[-10.0, 0, 0]]), p1=np.array([[10.0, 0, 0]]),
               mid=np.array([[0.0, 0, 0]]), length=np.array([20.0]),
               radius=np.array([0.5]))
    d = np.linspace(1.0, 200.0, 200)                       # 전극 수직거리 um
    elec = np.column_stack([np.zeros_like(d), d, np.zeros_like(d)])
    Vpsa = np.array([psa_matrix(seg, [e], sigma)[0, 0] for e in elec]) * I * 1e3  # uV
    Vlsa = np.array([lsa_matrix(seg, [e], sigma)[0, 0] for e in elec]) * I * 1e3
    # 근/원거리 대표값
    for dd in [5.0, 20.0, 100.0]:
        j = int(np.argmin(np.abs(d - dd)))
        rel = abs(Vpsa[j] - Vlsa[j]) / abs(Vpsa[j]) * 100
        print(f"[C] 전극 {dd:>5.0f}um: PSA={Vpsa[j]:7.3f} vs LSA={Vlsa[j]:7.3f} uV  차이 {rel:5.1f}%")

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    a.plot(r, Va, color="#c0392b", lw=2)
    a.plot([100], [v100], "o", color="k", ms=7)
    a.annotate(f"r=100um\nV={v100:.2f} uV", (100, v100), textcoords="offset points",
               xytext=(30, 20), fontsize=9, arrowprops=dict(arrowstyle="->"))
    a.set_xlabel("전극까지 거리 r (um)"); a.set_ylabel("세포외 전압 V (uV)")
    a.set_title("(A) 점전류원: V = I / (4*pi*sigma*r)\n거리에 반비례(1/r)로 약해진다")
    a.grid(alpha=0.3)

    b = ax[1]
    for s, lab in sigmas:
        b.plot(r, point_V(I, r, s) * 1e3, lw=2, label=lab)
    b.set_xlabel("거리 r (um)"); b.set_ylabel("V (uV)")
    b.set_title("(B) 조직 전도도 sigma의 영향\nsigma 클수록(잘 흐를수록) V 작아짐")
    b.legend(fontsize=8); b.grid(alpha=0.3); b.set_ylim(0, 6)

    c = ax[2]
    c.plot(d, Vpsa, color="#2980b9", lw=2, label="점전류원 PSA")
    c.plot(d, Vlsa, color="#27ae60", lw=2, ls="--", label="선전류원 LSA (정확)")
    c.axvspan(1, 10, color="orange", alpha=0.12)
    c.text(11, c.get_ylim()[1] * 0.6 if False else 30, "가까이(<길이)\nPSA 과대·발산", fontsize=8, color="#a04000")
    c.set_xlabel("전극 수직거리 (um, 20um 세그먼트)"); c.set_ylabel("V (uV)")
    c.set_title("(C) LSA vs PSA: 멀면 같고, 가까우면 다르다\n(가까울수록 LSA가 정확)")
    c.legend(fontsize=8); c.grid(alpha=0.3); c.set_ylim(0, 60)

    fig.suptitle("학습 Step 1 — 전류 하나가 만드는 세포외 전압 (Ness 2020 원리 · Holt&Koch 1999 LSA)",
                 fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "S1_point_and_line.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
