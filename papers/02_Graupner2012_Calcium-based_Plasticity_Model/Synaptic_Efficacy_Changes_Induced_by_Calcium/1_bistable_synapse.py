"""
1_bistable_synapse.py — [섹션 1-①] 이중안정 시냅스 (Eq.1 의 cubic 항)
============================================================================
Source: Graupner & Brunel (2012), "Synaptic Efficacy Changes Induced by Calcium",
        Eq.1 첫 항 + Fig 1D(활동 없을 때).

핵심 메시지:
    활동이 없을 때 Eq.1 은 τ dρ/dt = -ρ(1-ρ)(ρ*-ρ) 로 줄고,
    이 cubic 함수가 ρ 에 **두 안정상태**를 준다:
        ρ=0 (DOWN, 낮은 효능)  ·  ρ=1 (UP, 높은 효능)
    경계(불안정 고정점)는 ρ*=0.5. → 시냅스가 두 값 중 하나로 '잠기는' 쌍안정 기억.

패널:
    A. 위상선 dρ/dt vs ρ : 세 고정점 + 흐름 방향(화살표)
    B. 포텐셜 U(ρ) : 두 우물(DOWN/UP) + 장벽 (τ dρ/dt = -dU/dρ)
    C. ρ(t) : 여러 초기값이 경계 ρ*=0.5 를 기준으로 0 또는 1 로 완화

실행:
    conda activate ca1sim
    python papers/02_Graupner2012_Calcium-based_Plasticity_Model/Synaptic_Efficacy_Changes_Induced_by_Calcium/1_bistable_synapse.py
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

GREEN, RED, PURPLE, BLUE = "#2ca02c", "#d62728", "#6a3d9a", "#1f77b4"


def main():
    p = PARAM_SETS["DP"]
    rho = np.linspace(0.0, 1.0, 400)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.8))

    # --- A. 위상선 + 흐름 방향 ------------------------------------------------
    rhs = drift(rho, 0.0, p) * p.tau          # = -ρ(1-ρ)(ρ*-ρ)
    axA.axhline(0, color="0.6", lw=0.8)
    axA.plot(rho, rhs, color=BLUE, lw=2.4)
    # 고정점
    for fp, kind in [(0.0, "안정 (DOWN)"), (0.5, "불안정 (경계)"), (1.0, "안정 (UP)")]:
        stable = kind.startswith("안정")
        col = GREEN if stable else RED
        axA.plot(fp, 0.0, "o", ms=12, color=col,
                 mfc=col if stable else "white", mec=col, zorder=5)
    axA.annotate("ρ*=0.5", (0.5, 0.0), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=10, color=RED)
    # 흐름 방향 화살표 (dρ/dt 부호 → ρ 가 어디로 흐르나)
    ymin = rhs.min()
    for x in [0.15, 0.30, 0.70, 0.85]:
        d = drift(x, 0.0, p) * p.tau
        direction = 0.09 if d > 0 else -0.09
        axA.annotate("", xy=(x + direction, ymin * 0.5), xytext=(x, ymin * 0.5),
                     arrowprops=dict(arrowstyle="-|>", color="0.35", lw=2))
    axA.text(0.25, ymin * 0.72, "→0 으로", ha="center", fontsize=9, color="0.35")
    axA.text(0.78, ymin * 0.72, "→1 로", ha="center", fontsize=9, color="0.35")
    axA.set_title("A. 위상선  dρ/dt vs ρ  (활동 없음)\n"
                  "고정점 3개 · 화살표=흐름 방향", fontsize=11)
    axA.set_xlabel("효능 ρ"); axA.set_ylabel(r"$\tau\,d\rho/dt = -\rho(1-\rho)(\rho^*-\rho)$")
    axA.set_xlim(-0.03, 1.03); axA.grid(alpha=0.25)

    # --- B. 포텐셜 이중우물 ---------------------------------------------------
    U = potential(rho, p, calcium=0.0)
    axB.plot(rho, U, color=PURPLE, lw=2.6)
    axB.plot([0, 1], [potential(0.0, p, 0.0), potential(1.0, p, 0.0)], "o",
             ms=12, color=GREEN, zorder=5)
    axB.plot(0.5, potential(0.5, p, 0.0), "o", ms=12, mfc="white", mec=RED, zorder=5)
    axB.plot(0.06, potential(0.06, p, 0.0), "o", ms=15, color=BLUE, mec="#0C447C", zorder=6)
    axB.annotate("DOWN\n우물", (0, potential(0.0, p, 0.0)), textcoords="offset points",
                 xytext=(20, 4), fontsize=9, color=GREEN)
    axB.annotate("UP\n우물", (1, potential(1.0, p, 0.0)), textcoords="offset points",
                 xytext=(-38, 4), fontsize=9, color=GREEN)
    axB.annotate("장벽\nρ*=0.5", (0.5, potential(0.5, p, 0.0)), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9, color=RED)
    axB.set_title("B. 포텐셜 U(ρ)  (τ dρ/dt = -dU/dρ)\n"
                  "두 우물 사이 장벽 = 안정적 기억", fontsize=11)
    axB.set_xlabel("효능 ρ"); axB.set_ylabel("포텐셜 U(ρ)")
    axB.set_xlim(-0.03, 1.03); axB.grid(alpha=0.25)

    # --- C. 완화 ρ(t) ---------------------------------------------------------
    T = 900_000.0
    t = np.arange(0.0, T, 200.0)
    c0 = np.zeros_like(t)
    for r0 in [0.1, 0.3, 0.45, 0.5, 0.55, 0.7, 0.9]:
        r = integrate_rho(t, c0, p, rho0=r0, noise=False)
        col = GREEN if r0 > 0.5 else (RED if r0 < 0.5 else "0.5")
        axC.plot(t / 1000.0, r, color=col, lw=1.8)
        axC.text(-10, r0, f"{r0:g}", fontsize=8, ha="right", va="center", color=col)
    axC.axhline(0.5, color="0.5", ls="--", lw=1.0)
    axC.text(t[-1] / 1000.0, 0.52, "ρ*=0.5 경계", ha="right", fontsize=9, color="0.4")
    axC.set_ylim(-0.05, 1.05)
    axC.set_title("C. 활동 없을 때 완화 ρ(t)\n"
                  "경계 위→UP(1), 아래→DOWN(0)", fontsize=11)
    axC.set_xlabel("시간 [s]"); axC.set_ylabel("효능 ρ"); axC.grid(alpha=0.25)

    fig.suptitle("섹션 1-① 이중안정 시냅스 — Eq.1 의 cubic 항이 두 안정상태(DOWN·UP)를 만든다  "
                 f"[DP: ρ*={p.rho_star}]", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT, "1_bistable_synapse.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[섹션1-①] 이중안정 시냅스 그림 저장")
    print(f"  고정점: ρ=0(안정), ρ*={p.rho_star}(불안정), ρ=1(안정)")
    print(f"  → 저장: {out}")


if __name__ == "__main__":
    main()
