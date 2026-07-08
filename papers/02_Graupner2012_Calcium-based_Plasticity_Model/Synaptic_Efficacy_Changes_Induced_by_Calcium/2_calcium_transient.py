"""
2_calcium_transient.py — [섹션 1-②] 칼슘 트랜지언트 c(t)
============================================================================
Source: Graupner & Brunel (2012), "Synaptic Efficacy Changes Induced by Calcium",
        Eq.2(칼슘 동역학) + Fig 1A.

핵심 메시지:
    시냅스 가소성의 '입력 신호'는 후시냅스 칼슘 c(t) 하나뿐이다.
        - pre 스파이크 → 지연 D 후 C_pre 만큼 점프 (NMDAR 활성 지연)
        - post 스파이크 → 즉시 C_post 만큼 점프 (역전파 활동전위)
        - 각 점프는 τ_Ca 로 지수 감쇠하고, 여러 기여가 '선형 합산'된다.
    이 합산 봉우리가 강화문턱 θ_p / 약화문턱 θ_d 를 얼마나·얼마 동안 넘느냐가
    강화(LTP)/약화(LTD)의 양을 정한다. (다음 항목 3에서 정량화)

패널:
    A. 분해 : pre 기여(지연 D)·post 기여(즉시)·합 c(t) 를 함께
    B. 문턱 비교 : 합 c(t) 와 θ_p·θ_d, 초과 구간(그늘)·초과 시간 비율 α_p·α_d
    C. Δt 스캔 : 스파이크 순서에 따라 봉우리·겹침이 어떻게 달라지나

실행:
    conda activate ca1sim
    python papers/02_Graupner2012_Calcium-based_Plasticity_Model/Synaptic_Efficacy_Changes_Induced_by_Calcium/2_calcium_transient.py
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
from plasticity_model import PARAM_SETS, calcium_trace, time_above_thresholds  # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, C_PRE, C_POST = "#1f77b4", "#8c564b", "#d62728"
C_POT, C_DEP = "#e8710a", "#0f9e75"


def main():
    p = PARAM_SETS["DP"]
    t = np.arange(-40.0, 200.0, 0.05)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.8))

    # --- A. 분해: pre 기여 + post 기여 + 합 -----------------------------------
    dt_demo = 10.0                         # Δt = t_post - t_pre = +10 ms (pre 먼저)
    t_pre, t_post = 0.0, dt_demo
    c_pre = calcium_trace(t, [t_pre], [], p)      # pre 만
    c_post = calcium_trace(t, [], [t_post], p)    # post 만
    c_sum = calcium_trace(t, [t_pre], [t_post], p)

    axA.plot(t, c_pre, color=C_PRE, lw=1.6, ls="--",
             label=f"pre 기여 (지연 D={p.D}ms, C_pre={p.C_pre:g})")
    axA.plot(t, c_post, color=C_POST, lw=1.6, ls="--",
             label=f"post 기여 (즉시, C_post={p.C_post:g})")
    axA.plot(t, c_sum, color=BLUE, lw=2.6, label="합 c(t)")
    axA.axvline(t_pre, color=C_PRE, lw=1.0, alpha=0.5)
    axA.axvline(t_post, color=C_POST, lw=1.0, alpha=0.5)
    axA.annotate("pre", (t_pre, 2.55), color=C_PRE, ha="center", fontsize=9)
    axA.annotate("post", (t_post, 2.55), color=C_POST, ha="center", fontsize=9)
    # 지연 D 화살표
    axA.annotate("", xy=(t_pre + p.D, 0.35), xytext=(t_pre, 0.35),
                 arrowprops=dict(arrowstyle="<|-|>", color=C_PRE, lw=1.4))
    axA.text(t_pre + p.D / 2, 0.45, f"D={p.D}", ha="center", color=C_PRE, fontsize=8)
    axA.set_xlim(-30, 120); axA.set_ylim(0, 2.8)
    axA.set_xlabel("시간 [ms]"); axA.set_ylabel("칼슘 c")
    axA.set_title("A. 칼슘 = pre·post 기여의 선형 합산\n"
                  "각 스파이크 → 점프 후 τ_Ca 지수 감쇠", fontsize=11)
    axA.legend(fontsize=8.5, loc="upper right"); axA.grid(alpha=0.25)

    # --- B. 문턱 비교 + 초과 시간 --------------------------------------------
    axB.plot(t, c_sum, color=BLUE, lw=2.6)
    axB.axhline(p.theta_p, color=C_POT, ls="--", lw=1.4)
    axB.axhline(p.theta_d, color=C_DEP, ls="--", lw=1.4)
    axB.fill_between(t, p.theta_d, c_sum, where=c_sum > p.theta_d,
                     color=C_DEP, alpha=0.20)
    axB.fill_between(t, p.theta_p, c_sum, where=c_sum > p.theta_p,
                     color=C_POT, alpha=0.35)
    ap, ad = time_above_thresholds(t, c_sum, p)
    axB.text(118, p.theta_p + 0.05, f"θ_p={p.theta_p:g}", ha="right",
             color=C_POT, fontsize=9)
    axB.text(118, p.theta_d - 0.14, f"θ_d={p.theta_d:g}", ha="right",
             color=C_DEP, fontsize=9)
    axB.text(0.97, 0.80, f"강화(θ_p 초과): α_p={ap*100:.1f}%",
             transform=axB.transAxes, ha="right", color=C_POT, fontsize=9.5)
    axB.text(0.97, 0.71, f"약화(θ_d 초과): α_d={ad*100:.1f}%",
             transform=axB.transAxes, ha="right", color=C_DEP, fontsize=9.5)
    axB.set_xlim(-30, 120); axB.set_ylim(0, 2.8)
    axB.set_xlabel("시간 [ms]"); axB.set_ylabel("칼슘 c")
    axB.set_title("B. 문턱과 비교 → 초과 시간(그늘)\n"
                  "이 시간이 강화·약화의 '양'을 정한다", fontsize=11)
    axB.grid(alpha=0.25)

    # --- C. Δt 스캔: 순서가 봉우리를 바꾼다 ----------------------------------
    for dt_val, col in [(+30.0, "#4c9be8"), (+10.0, BLUE),
                        (-10.0, "#9467bd"), (-30.0, "#d62728")]:
        c = calcium_trace(t, [0.0], [dt_val], p)
        axC.plot(t, c, color=col, lw=2.0, label=f"Δt={dt_val:+.0f} ms")
    axC.axhline(p.theta_p, color=C_POT, ls="--", lw=1.2)
    axC.axhline(p.theta_d, color=C_DEP, ls="--", lw=1.2)
    axC.text(118, p.theta_p + 0.05, "θ_p", ha="right", color=C_POT, fontsize=9)
    axC.text(118, p.theta_d - 0.14, "θ_d", ha="right", color=C_DEP, fontsize=9)
    axC.set_xlim(-30, 120); axC.set_ylim(0, 3.0)
    axC.set_xlabel("시간 [ms]"); axC.set_ylabel("칼슘 c")
    axC.set_title("C. 스파이크 순서 Δt → 봉우리 변화\n"
                  "pre 먼저(Δt>0)면 겹쳐서 봉우리↑", fontsize=11)
    axC.legend(fontsize=8.5, loc="upper right"); axC.grid(alpha=0.25)

    fig.suptitle("섹션 1-② 칼슘 트랜지언트 c(t) — 가소성의 유일한 입력 신호  "
                 f"[DP: C_pre={p.C_pre:g}, C_post={p.C_post:g}, τ_Ca={p.tau_ca:g}ms, D={p.D:g}ms]",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT, "2_calcium_transient.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[섹션1-②] 칼슘 트랜지언트 그림 저장")
    print(f"  Δt=+10ms 합 봉우리 max c = {c_sum.max():.3f}")
    print(f"  θ_p 초과 α_p={ap*100:.2f}%, θ_d 초과 α_d={ad*100:.2f}%")
    print(f"  → 저장: {out}")


if __name__ == "__main__":
    main()
