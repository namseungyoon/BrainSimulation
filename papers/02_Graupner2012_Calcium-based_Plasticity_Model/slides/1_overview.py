"""
1_overview.py — [슬라이드용] Graupner-Brunel 모델 전체 개요 한 장
============================================================================
논문을 처음 읽을 때 길을 잃지 않도록, 모델의 신호 흐름을 한 장으로 정리한다.

    ① 스파이크(pre·post)  →  ② 칼슘 c(t)  →  ③ 효능 ρ(이중우물)  →  ④ 시냅스 세기 w

각 칸은 실제 계산 결과를 담고, 아래에 핵심 방정식 Eq.1 을 항별 색으로 표시.

실행:
    conda activate ca1sim
    python papers/02_Graupner2012_Calcium-based_Plasticity_Model/slides/1_overview.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, PAPER)
from common.plotstyle import set_korean_font          # noqa: E402
from plasticity_model import PARAM_SETS, calcium_trace, potential  # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
os.makedirs(OUT, exist_ok=True)

C_CUBIC, C_POT, C_DEP = "#6a3d9a", "#e8710a", "#0f9e75"   # 기억/강화/약화 색
C_CA = "#1f77b4"


def arrow(fig, x0, x1, y, text):
    a = FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=18,
                        lw=2.0, color="0.35", zorder=10)
    fig.add_artist(a)
    fig.text((x0 + x1) / 2, y + 0.045, text, ha="center", va="bottom",
             fontsize=10, color="0.25")


def main():
    p = PARAM_SETS["DP"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.28, wspace=0.55)
    ax1, ax2, ax3, ax4 = axes

    # ① 스파이크 입력 --------------------------------------------------------
    ax1.axvline(0, ymin=0.55, ymax=0.9, color=C_CA, lw=2.5)
    ax1.axvline(10, ymin=0.15, ymax=0.5, color="#d62728", lw=2.5)
    ax1.text(0, 0.95, "pre", ha="center", color=C_CA, fontsize=11, transform=ax1.get_xaxis_transform())
    ax1.text(10, 0.05, "post", ha="center", va="bottom", color="#d62728", fontsize=11, transform=ax1.get_xaxis_transform())
    ax1.set_xlim(-20, 40); ax1.set_ylim(0, 1); ax1.set_yticks([])
    ax1.set_xlabel("시간 [ms]")
    ax1.set_title("① 스파이크 입력\npre·post 발화", fontsize=12)

    # ② 칼슘 트레이스 --------------------------------------------------------
    t = np.arange(-20.0, 100.0, 0.05)
    c = calcium_trace(t, [0.0], [10.0], p)
    ax2.plot(t, c, color=C_CA, lw=2.2)
    ax2.axhline(p.theta_p, color=C_POT, ls="--", lw=1.5)
    ax2.axhline(p.theta_d, color=C_DEP, ls="--", lw=1.5)
    ax2.fill_between(t, p.theta_p, c, where=c > p.theta_p, color=C_POT, alpha=0.25)
    ax2.text(99, p.theta_p + 0.05, "θ_p (강화 문턱)", ha="right", color=C_POT, fontsize=9)
    ax2.text(99, p.theta_d - 0.18, "θ_d (약화 문턱)", ha="right", color=C_DEP, fontsize=9)
    ax2.set_xlim(-20, 100); ax2.set_ylim(0, 3.0)
    ax2.set_xlabel("시간 [ms]"); ax2.set_ylabel("칼슘 c")
    ax2.set_title("② 칼슘 c(t)\n스파이크→칼슘, 문턱과 비교", fontsize=12)

    # ③ 효능 ρ: 이중우물 -----------------------------------------------------
    rho = np.linspace(0, 1, 300)
    U = potential(rho, p, calcium=0.0)
    ax3.plot(rho, U, color=C_CUBIC, lw=2.4)
    yb = potential(0.03, p, 0.0)
    ax3.plot(0.03, yb, "o", ms=13, color=C_CA, mec="#0C447C")
    a = FancyArrowPatch((0.18, yb - 0.002), (0.9, potential(0.95, p, 0.0)),
                        connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                        mutation_scale=16, lw=2.0, color=C_POT)
    ax3.add_patch(a)
    ax3.text(0.03, yb, "DOWN", ha="center", va="top", fontsize=9, color="0.3")
    ax3.text(0.97, potential(0.97, p, 0.0), "UP", ha="center", va="top", fontsize=9, color="0.3")
    ax3.text(0.5, yb * 0.4, "강화 시\nUP으로", ha="center", color=C_POT, fontsize=9)
    ax3.set_xlim(-0.05, 1.05); ax3.set_xlabel("효능 ρ"); ax3.set_ylabel("포텐셜 U(ρ)")
    ax3.set_title("③ 효능 ρ = 이중우물\n두 안정상태(기억) + 장벽", fontsize=12)

    # ④ 시냅스 세기 ----------------------------------------------------------
    w0, w1 = 1.0, p.b
    ax4.bar([0, 1], [w0, w1], color=["#9aa0a6", C_POT], width=0.55)
    ax4.text(0, w0 + 0.15, "w0\n(DOWN)", ha="center", fontsize=9)
    ax4.text(1, w1 + 0.15, f"w1 = b·w0\n(UP, b={p.b:g})", ha="center", fontsize=9)
    ax4.set_xticks([0, 1]); ax4.set_xticklabels(["ρ~0", "ρ~1"])
    ax4.set_ylim(0, w1 * 1.25); ax4.set_ylabel("시냅스 세기 w")
    ax4.set_title("④ 시냅스 세기 w\nw = w0 + ρ(w1-w0)", fontsize=12)

    # 단계 간 화살표 (figure 좌표) -------------------------------------------
    arrow(fig, 0.247, 0.302, 0.55, "칼슘 유입")
    arrow(fig, 0.492, 0.548, 0.55, "θ 초과→구동")
    arrow(fig, 0.737, 0.792, 0.55, "ρ 변화")

    # 하단 방정식 (항별 색) --------------------------------------------------
    fig.text(0.5, 0.135,
             r"$\tau\,\dot{\rho} \;=\; \mathbf{-\rho(1-\rho)(\rho^*-\rho)} "
             r"\;+\; \mathbf{\gamma_p(1-\rho)\,\Theta[c-\theta_p]} "
             r"\;-\; \mathbf{\gamma_d\,\rho\,\Theta[c-\theta_d]} \;+\; Noise$",
             ha="center", fontsize=15)
    fig.text(0.5, 0.045,
             "cubic 항 = 기억(이중우물)      "
             "γ_p 항 = 강화(칼슘>θ_p)      "
             "γ_d 항 = 약화(칼슘>θ_d)",
             ha="center", fontsize=10.5, color="0.3")
    # 색 강조 라벨
    fig.text(0.235, 0.045, "■", ha="center", fontsize=11, color=C_CUBIC)
    fig.text(0.505, 0.045, "■", ha="center", fontsize=11, color=C_POT)
    fig.text(0.735, 0.045, "■", ha="center", fontsize=11, color=C_DEP)

    fig.suptitle("Graupner & Brunel (2012) 칼슘 기반 가소성 모델 — 한 장 개요",
                 fontsize=14, y=0.96)
    out = os.path.join(OUT, "1_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[슬라이드] 개요 그림 저장: {out}")


if __name__ == "__main__":
    main()
