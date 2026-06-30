"""
5_stochastic_mvr.py — 단계 5: §2.5 확률적 다소포 방출(MVR)
[8개 파라미터 추적] 이 단계가 찾음: N_RRP  (1/8)
============================================================================
Source: Ecker et al. (2020) §2.5 Eq.(7)-(10); 다소포 방출(MVR) 이항 모델.

한 발화에서 방출되는 소포 수 k 는 **이항분포** B(N_RRP, U_SE):
    P(k) = C(N,k) U^k (1-U)^(N-k),   평균 = N·U,   CV = sqrt((1-U)/(N·U))
→ 시행마다 PSC 진폭이 변동(확률적). **N_RRP(방출 가능 소포 수)가 클수록 CV↓**.
모델 보정: **첫 PSC 의 실험 CV** 에 맞춰 N_RRP 를 정한다(= CV 피팅).

3 패널:
  (A) 방출 소포 수 이항분포 (N_RRP=1·2·6)
  (B) 시행 간 PSC 진폭 변동 (N_RRP 작으면 큼, 실패 많음)
  (C) CV vs N_RRP + 실험 CV 타깃에 맞춘 N_RRP 피팅

해석식(이항분포) — NEURON 불필요. 실제 EMS 시냅스 트레이스 비교는 5-1_ 참조.
실행: <ca1sim py> .../03_synapses/5_stochastic_mvr.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
from common.plotstyle import set_korean_font           # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")

U = 0.5                       # 대표 방출확률 U_SE (PC->PC)
EXP_CV = 0.76                 # 첫 PSC 실험 CV 타깃 (SC->PC, Ecker 2020 Fig.3F)
NRRP_SHOW = [1, 2, 6]
COL = {1: "tab:red", 2: "tab:orange", 6: "tab:green"}


def cv_of(N, u=U):
    return float(np.sqrt((1.0 - u) / (N * u)))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.suptitle("단계 5 — 확률적 다소포 방출(MVR): 방출 소포 수 = 이항분포 · "
                 "N_RRP 클수록 변동성↓ · 첫 PSC CV 피팅",
                 fontsize=12.5, fontweight="bold")

    # ── (A) 이항분포 B(N_RRP, U) ──
    width = 0.26
    for i, N in enumerate(NRRP_SHOW):
        ks = np.arange(0, N + 1)
        pmf = binom.pmf(ks, N, U)
        axA.bar(ks + (i - 1) * width, pmf, width, color=COL[N], alpha=0.9,
                label=f"N_RRP={N}  (CV={cv_of(N):.2f}, 실패={(1-U)**N*100:.0f}%)")
    axA.set_xlabel("방출된 소포 수 k"); axA.set_ylabel("확률 P(k)")
    axA.set_title(f"(A) 방출 소포 수 = 이항분포 B(N_RRP, U={U})", fontsize=10)
    axA.set_xticks(range(0, max(NRRP_SHOW) + 1))
    axA.legend(fontsize=8); axA.grid(axis="y", alpha=0.3)

    # ── (B) 시행 간 PSC 진폭 변동 (정규화 진폭 = k/(N·U)) ──
    rng = np.random.RandomState(7)
    for N in (1, 6):
        k = rng.binomial(N, U, 4000)
        amp = k / (N * U)                       # 평균=1 로 정규화
        axB.hist(amp, bins=np.linspace(-0.1, 3.1, 33), density=True, alpha=0.55,
                 color=COL[N], label=f"N_RRP={N} (CV={cv_of(N):.2f})")
    axB.axvline(1.0, color="0.5", ls="--", lw=1)
    axB.set_xlabel("정규화 PSC 진폭 (평균=1)"); axB.set_ylabel("시행 밀도")
    axB.set_title("(B) 시행 간 변동성 — N_RRP 작으면 큼(실패↑)", fontsize=10)
    axB.legend(fontsize=8.5); axB.grid(alpha=0.3)

    # ── (C) CV vs N_RRP + 실험 CV 피팅 ──
    Ns = np.arange(1, 11)
    cvs = np.array([cv_of(N) for N in Ns])
    axC.plot(Ns, cvs, "o-", color="tab:purple", lw=2, ms=6, label=f"모델 CV (U={U})")
    axC.axhline(EXP_CV, color="k", ls="--", lw=1.5, label=f"실험 CV 타깃 ~{EXP_CV}")
    n_fit = (1 - U) / (U * EXP_CV ** 2)         # CV=sqrt((1-U)/(N·U)) → N
    axC.axvline(n_fit, color="tab:green", ls=":", lw=1.5)
    axC.plot([round(n_fit)], [cv_of(round(n_fit))], "*", color="tab:green", ms=18,
             mec="k", zorder=6)
    axC.annotate(f"CV 피팅 → N_RRP ~ {n_fit:.1f}\n(채택 {round(n_fit)})",
                 xy=(n_fit, EXP_CV), xytext=(n_fit + 1.5, EXP_CV + 0.18),
                 fontsize=9, color="tab:green", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:green"))
    axC.set_xlabel("N_RRP (방출 가능 소포 수)"); axC.set_ylabel("첫 PSC 진폭 CV")
    axC.set_title("(C) CV vs N_RRP — 실험 CV 에 맞춰 N_RRP 결정", fontsize=10)
    axC.set_xticks(Ns); axC.legend(fontsize=8.5); axC.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "5_stochastic_mvr.png")
    fig.savefig(out, dpi=125)
    print(f"[그림] {out}", flush=True)
    print(f"[MVR] U={U}, 실험 CV~{EXP_CV} → 피팅 N_RRP~{n_fit:.2f} (채택 {round(n_fit)}). "
          f"CV(N=1)={cv_of(1):.2f}, CV(N=6)={cv_of(6):.2f}", flush=True)


if __name__ == "__main__":
    main()
