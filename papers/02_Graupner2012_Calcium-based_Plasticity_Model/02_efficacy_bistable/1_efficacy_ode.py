"""
1_efficacy_ode.py — 단계 2: 시냅스 효능 ODE (Eq.1) 항별 분해
============================================================================
Source: Graupner & Brunel (2012) Eq.1, Fig 1D.
    τ dρ/dt = -ρ(1-ρ)(ρ*-ρ)          # (1) 이중우물(cubic): 활동 없을 때 두 안정상태
              + γ_p (1-ρ) Θ[c-θ_p]    # (2) 강화: 칼슘이 θ_p 초과 시 ρ→UP 으로 밈
              - γ_d ρ Θ[c-θ_d]        # (3) 약화: 칼슘이 θ_d 초과 시 ρ→DOWN 으로 당김
              + Noise(t)              # (4) 활동 의존 잡음 (여기선 결정론적으로 끔)

확인 목표 (4패널):
  A. 활동 없을 때(c=0) 우변 = cubic 항뿐 → 고정점 ρ=0,0.5,1 (bistable)
  B. 그 cubic 항의 포텐셜 U(ρ) → 두 우물(DOWN=0, UP=1) + 장벽(ρ*=0.5)
  C. 칼슘이 문턱을 넘으면 (2)(3)항이 켜져 ρ 가 목표점으로 끌려감
       c<θ_d: 안 움직임 / θ_d<c<θ_p: DOWN 으로 / c>θ_p: 0.5 위로(→UP 운명)
  D. 활동 없을 때 여러 초기값 ρ0 의 완화 → ρ*=0.5 를 경계로 0 또는 1 로 (기억 latching)

실행:
    conda activate ca1sim
    python papers/02_Graupner2012_Calcium-based_Plasticity_Model/02_efficacy_bistable/1_efficacy_ode.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, PAPER)
from common.plotstyle import set_korean_font          # noqa: E402
from plasticity_model import PARAM_SETS, drift, potential, integrate_rho  # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
os.makedirs(OUT, exist_ok=True)

ORANGE, TURQ = "orange", "#1abc9c"
GREEN, RED = "#2ca02c", "#d62728"


def equilibrium(level, p, rho0, T_ms, dt=1.0):
    """고정 칼슘 level 에서 rho0 부터 결정론적으로 적분해 정착값을 얻는다."""
    t = np.arange(0.0, T_ms, dt)
    c = np.full_like(t, float(level))
    rho = integrate_rho(t, c, p, rho0=rho0, noise=False)
    return t, rho


def main():
    p = PARAM_SETS["DP"]     # θ_d=1.0, θ_p=1.3, γ_d=200, γ_p=321.808, ρ*=0.5
    rho = np.linspace(0.0, 1.0, 400)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    axA, axB, axC, axD = axes.ravel()

    # --- A. 활동 없을 때 우변(=cubic 항). τ 를 곱해 대괄호 안 값을 본다 ------------
    rhs0 = drift(rho, 0.0, p) * p.tau            # = -ρ(1-ρ)(ρ*-ρ)
    axA.axhline(0, color="0.6", lw=0.8)
    axA.plot(rho, rhs0, color="#1f77b4", lw=2.0)
    for fp, kind in [(0.0, "안정"), (p.rho_star, "불안정"), (1.0, "안정")]:
        col = RED if kind == "불안정" else GREEN
        axA.plot(fp, 0.0, "o", color=col, ms=9,
                 mfc=col if kind != "불안정" else "white", mec=col, zorder=5)
        axA.annotate(f"ρ={fp:g}\n({kind})", (fp, 0.0),
                     textcoords="offset points", xytext=(0, -34 if fp == p.rho_star else 12),
                     ha="center", fontsize=8, color=col)
    axA.set_title("A. 활동 없음(c=0): 우변 = cubic 항만\n"
                  "→ 고정점 3개 (0·1 안정, 0.5 불안정) = 쌍안정", fontsize=10)
    axA.set_xlabel("효능 ρ"); axA.set_ylabel("τ·dρ/dt  =  -ρ(1-ρ)(ρ*-ρ)")
    axA.grid(alpha=0.25)

    # --- B. 그 항의 포텐셜 U(ρ) (이중우물) --------------------------------------
    U0 = potential(rho, p, calcium=0.0)
    axB.plot(rho, U0, color="#6a3d9a", lw=2.2)
    axB.plot([0, 1], [potential(0.0, p, 0.0), potential(1.0, p, 0.0)], "o",
             color=GREEN, ms=9, zorder=5)
    axB.plot(p.rho_star, potential(p.rho_star, p, 0.0), "o", mfc="white",
             mec=RED, ms=9, zorder=5)
    axB.annotate("DOWN\n우물", (0.0, potential(0.0, p, 0.0)), textcoords="offset points",
                 xytext=(18, 6), fontsize=8, color=GREEN)
    axB.annotate("UP\n우물", (1.0, potential(1.0, p, 0.0)), textcoords="offset points",
                 xytext=(-34, 6), fontsize=8, color=GREEN)
    axB.annotate("장벽 ρ*=0.5", (p.rho_star, potential(p.rho_star, p, 0.0)),
                 textcoords="offset points", xytext=(0, 10), ha="center",
                 fontsize=8, color=RED)
    axB.set_title("B. 활동 없을 때의 포텐셜 U(ρ)  (τ dρ/dt = -dU/dρ)\n"
                  "→ 두 우물(DOWN·UP) 사이 장벽 = 안정적 기억", fontsize=10)
    axB.set_xlabel("효능 ρ"); axB.set_ylabel("포텐셜 U(ρ)")
    axB.grid(alpha=0.25)

    # --- C. 칼슘 문턱이 켜지면 ρ 가 끌려감 (고정 칼슘 3 수준) --------------------
    levels = [(0.5,  "c=0.5 < θ_d : 구동 없음"),
              (1.15, "θ_d < c=1.15 < θ_p : 약화→DOWN"),
              (1.5,  "c=1.5 > θ_p : 강화→0.5 위로")]
    cols = ["#7f7f7f", TURQ, ORANGE]
    for (lv, lab), col in zip(levels, cols):
        for r0 in (0.2, 0.8):
            t, r = equilibrium(lv, p, r0, T_ms=3000.0)
            axC.plot(t, r, color=col, lw=1.8,
                     label=lab if r0 == 0.2 else None)
    axC.axhline(p.rho_star, color=RED, lw=1.0, ls="--")
    axC.text(3000, p.rho_star + 0.02, "ρ*=0.5 (UP/DOWN 경계)", ha="right",
             fontsize=8, color=RED)
    axC.set_ylim(-0.05, 1.05)
    axC.set_title("C. 고정 칼슘에서 ρ(t) (초기 0.2·0.8, 결정론적)\n"
                  "→ 칼슘 수준이 ρ 의 목표점을 정한다", fontsize=10)
    axC.set_xlabel("시간 [ms]"); axC.set_ylabel("효능 ρ")
    axC.legend(fontsize=8, loc="center right"); axC.grid(alpha=0.25)

    # --- D. 활동 없을 때 완화 (여러 ρ0) → 두 안정상태로 latching -----------------
    T = 900_000.0                              # 900 s (τ=150 s 규모)
    t = np.arange(0.0, T, 200.0)
    c0 = np.zeros_like(t)
    for r0 in [0.1, 0.3, 0.45, 0.5, 0.55, 0.7, 0.9]:
        r = integrate_rho(t, c0, p, rho0=r0, noise=False)
        col = GREEN if r0 > p.rho_star else (RED if r0 < p.rho_star else "0.5")
        axD.plot(t / 1000.0, r, color=col, lw=1.6)
        axD.text(-8, r0, f"{r0:g}", fontsize=7, ha="right", va="center", color=col)
    axD.axhline(p.rho_star, color="0.5", lw=1.0, ls="--")
    axD.set_ylim(-0.05, 1.05)
    axD.set_title("D. 활동 없을 때 완화 (초기값 7개, c=0)\n"
                  "→ ρ*=0.5 위는 UP(1), 아래는 DOWN(0) 으로 수렴", fontsize=10)
    axD.set_xlabel("시간 [s]"); axD.set_ylabel("효능 ρ")
    axD.grid(alpha=0.25)

    fig.suptitle("단계 2 — 시냅스 효능 ODE (Eq.1) 항별 분해   "
                 f"[DP 세트: θ_d={p.theta_d}, θ_p={p.theta_p}, "
                 f"γ_d={p.gamma_d}, γ_p={p.gamma_p}, ρ*={p.rho_star}]",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "1_efficacy_ode.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 콘솔 요약 (수치 확정)
    print("[단계2] 효능 ODE 항별 분해 그림 생성 완료")
    print(f"  cubic 항 고정점: ρ=0(안정), ρ={p.rho_star}(불안정), ρ=1(안정)")
    _, r_pot = equilibrium(1.5, p, 0.2, T_ms=3000.0)
    _, r_dep = equilibrium(1.15, p, 0.8, T_ms=3000.0)
    print(f"  강화 구동(c=1.5>θ_p) 정착 ρ ~ {r_pot[-1]:.3f}  "
          f"(이론 γ_p/(γ_p+γ_d)={p.gamma_p/(p.gamma_p+p.gamma_d):.3f}, >0.5 → UP 운명)")
    print(f"  약화 구동(θ_d<c=1.15<θ_p) 정착 ρ ~ {r_dep[-1]:.3f}  (<0.5 → DOWN 운명)")
    print(f"  → 저장: {out}")


if __name__ == "__main__":
    main()
