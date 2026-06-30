"""
5-2_binomial_concept.py — 다소포 방출 이항분포 P(k)=C(N,k)U^k(1-U)^(N-k) 시각화 (실제 경로)
============================================================================
Source: Ecker(2020) §2.5 MVR 이항 모델 + Table 3 경로별 (N_RRP, U).

식의 '뜻'을 그림으로, **실제 시냅스 경로 값**으로:
  스파이크 시 N_RRP개 소포가 각각 확률 U로 독립 방출 → 방출 수 k.
  P(k) = C(N_RRP,k) · U^k · (1-U)^(N_RRP-k)
- (좌) 식 풀이: **PC->PC (N_RRP=2, U=0.50)** 의 모든 패턴 나열 → C(2,k)=1·2·1
- (우) 경로별 분포: N_RRP·U 가 경로마다 달라 P(k)도 다름
    PV+->PC(N=1,U=0.13: 거의 실패) · PC->PC(N=2,U=0.50) · CCK-->CCK-(N=6,U=0.26)

값은 params_table3.CLASSES 에서 직접 로드. 해석식 — NEURON 불필요.
실행: <ca1sim py> .../03_synapses/5-2_binomial_concept.py
"""
import os
import sys
from math import comb
from itertools import combinations

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
from params_table3 import CLASSES                       # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
REL_C, FAIL_C = "crimson", "white"

# 식 풀이용 대표 경로 (작은 N_RRP 라 모든 패턴 나열 가능)
EX_KEY = "PC->PC (E2)"
# 경로별 비교 (N_RRP·U 가 다름)
COMPARE = [("PV+->PC", "PV+->PC (I2)", "tab:orange"),
           ("PC->PC", "PC->PC (E2)", "tab:red"),
           ("CCK-->CCK-", "CCK-->CCK- (I2)", "tab:green")]


def cv_of(N, u):
    return float(np.sqrt((1.0 - u) / (N * u)))


def draw_pattern(ax, x, y, released, n):
    for v in range(n):
        ax.scatter(x + v * 0.034, y, s=130,
                   facecolor=(REL_C if v in released else FAIL_C),
                   edgecolor="k", linewidths=0.9, zorder=3)


def main():
    os.makedirs(OUT, exist_ok=True)
    exU = CLASSES[EX_KEY]["Use"]; exN = int(CLASSES[EX_KEY]["Nrrp"])

    fig = plt.figure(figsize=(15.5, 8.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.5, 1], width_ratios=[1.15, 1],
                          hspace=0.16, wspace=0.18)
    axEq = fig.add_subplot(gs[0, :])
    axEn = fig.add_subplot(gs[1, 0])
    axPk = fig.add_subplot(gs[1, 1])
    fig.suptitle("다소포 방출 = 이항분포  P(k)=C(N_RRP,k)·U^k·(1-U)^(N_RRP-k)  "
                 "— 경로마다 N_RRP·U 다름", fontsize=13.5, fontweight="bold")

    # ── (상) 식 + 항별 의미 ──
    axEq.axis("off"); axEq.set_xlim(0, 1); axEq.set_ylim(0, 1)
    axEq.text(0.5, 0.88, "스파이크 시 N_RRP개 소포가 '각각 확률 U로 독립' 방출 → 방출 수 k",
              ha="center", fontsize=11.5, fontweight="bold")
    axEq.text(0.5, 0.55, "P(k)  =  C(N_RRP, k)   x   U^k   x   (1-U)^(N_RRP - k)",
              ha="center", fontsize=15, fontweight="bold")
    chips = [(0.18, "C(N_RRP, k)", "tab:blue", "k개를 고르는\n경우의 수(배열)"),
             (0.50, "U^k", "crimson", "고른 k개가\n방출할 확률"),
             (0.82, "(1-U)^(N-k)", "0.35", "나머지 N-k개가\n실패할 확률")]
    for x, term, col, desc in chips:
        axEq.text(x, 0.16, f"{term}\n{desc}", ha="center", va="center", fontsize=9.2,
                  color=col, fontweight="bold",
                  bbox=dict(fc="white", ec=col, lw=1.3, boxstyle="round,pad=0.4"))

    # ── (좌하) 실제 경로 식 풀이: PC->PC (N_RRP=2) ──
    axEn.axis("off"); axEn.set_xlim(0, 1); axEn.set_ylim(0, 1)
    axEn.set_title(f"식 풀이 예시: {EX_KEY}  (N_RRP={exN}, U={exU})  — 빨강=방출, 흰=실패",
                   fontsize=10)
    rows_y = np.linspace(0.80, 0.16, exN + 1)
    for k in range(exN + 1):
        y = rows_y[k]
        axEn.text(0.02, y, f"k={k}", fontsize=11, fontweight="bold", va="center")
        for pi, rel in enumerate(combinations(range(exN), k)):
            draw_pattern(axEn, 0.16 + pi * 0.20, y, set(rel), exN)
        pp = exU ** k * (1 - exU) ** (exN - k)
        C = comb(exN, k); Pk = C * pp
        axEn.text(0.60, y, f"C({exN},{k})={C}  x  {pp:.3f}  =  P({k})={Pk:.3f}",
                  fontsize=9, va="center", color="tab:blue")
    axEn.scatter(0.16, 0.97, s=120, facecolor=REL_C, edgecolor="k"); axEn.text(0.19, 0.97, "방출", va="center", fontsize=8.5)
    axEn.scatter(0.30, 0.97, s=120, facecolor=FAIL_C, edgecolor="k"); axEn.text(0.33, 0.97, "실패", va="center", fontsize=8.5)

    # ── (우하) 경로별 P(k) 비교 (N_RRP·U 다름) ──
    kmax = max(int(CLASSES[key]["Nrrp"]) for _, key, _ in COMPARE)
    ks = np.arange(0, kmax + 1)
    w = 0.26
    for i, (lab, key, col) in enumerate(COMPARE):
        U = CLASSES[key]["Use"]; N = int(CLASSES[key]["Nrrp"])
        pmf = binom.pmf(ks, N, U)
        axPk.bar(ks + (i - 1) * w, pmf, w, color=col, alpha=0.9,
                 label=f"{lab}: N={N}, U={U}, 평균={N*U:.2f}, CV={cv_of(N,U):.2f}")
    axPk.set_xlabel("방출 소포 수 k"); axPk.set_ylabel("P(k)")
    axPk.set_title("경로별 P(k) — N_RRP·U 가 분포(변동성)를 결정", fontsize=10)
    axPk.set_xticks(ks); axPk.legend(fontsize=8.2); axPk.grid(axis="y", alpha=0.3)
    axPk.text(0.97, 0.62, "대부분 경로 N_RRP=1\n(=베르누이: 방출/실패)\nPVBC계열만 N_RRP=6",
              transform=axPk.transAxes, ha="right", va="top", fontsize=7.8,
              bbox=dict(fc="#FFF6D5", ec="0.6", alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "5-2_binomial_concept.png")
    fig.savefig(out, dpi=130)
    print(f"[그림] {out}", flush=True)
    for lab, key, _ in COMPARE:
        U = CLASSES[key]["Use"]; N = int(CLASSES[key]["Nrrp"])
        print(f"  {lab}: N_RRP={N}, U={U}, 평균={N*U:.2f}, CV={cv_of(N,U):.2f}, "
              f"실패율(1-U)^N={(1-U)**N:.2f}", flush=True)


if __name__ == "__main__":
    main()
