"""
2_stdp_from_calcium.py — [슬라이드용] 왜 스파이크 타이밍이 가소성을 바꾸나
============================================================================
핵심 직관: 스파이크 순서 Δt = t_post - t_pre 가 바뀌면
    → 칼슘 봉우리 모양이 바뀌고
    → 칼슘이 강화문턱 θ_p / 약화문턱 θ_d 위에 머무는 '시간'이 바뀌고
    → 강화항(γ_p·α_p) 과 약화항(γ_d·α_d) 의 균형이 바뀌어 LTP/LTD 가 갈린다.

여기서 α_p, α_d = 칼슘이 각 문턱 위에 머무는 시간 비율 (논문 Fig 2B).
경계 ρ*=0.5 에서 순 구동 부호 = sign(γ_p·α_p - γ_d·α_d) 가 방향을 예측한다.

실행:
    conda activate ca1sim
    python papers/02_Graupner2012_Calcium-based_Plasticity_Model/slides/2_stdp_from_calcium.py
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

C_POT, C_DEP = "#e8710a", "#0f9e75"


def main():
    p = PARAM_SETS["DP"]
    t = np.arange(-100.0, 250.0, 0.05)

    # Δt 스윕 → α_p, α_d, 순 구동
    dts = np.arange(-60.0, 60.0 + 0.5, 1.0)
    ap = np.zeros_like(dts); ad = np.zeros_like(dts)
    for i, dt in enumerate(dts):
        c = calcium_trace(t, [0.0], [dt], p)
        ap[i], ad[i] = time_above_thresholds(t, c, p)
    net = p.gamma_p * ap - p.gamma_d * ad          # 경계 ρ*=0.5 에서 부호가 방향 결정

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axA, axB, axC = axes

    # A. 대표 Δt 두 개의 칼슘 트레이스 --------------------------------------
    demos = [(+10.0, C_POT, "Δt=+10 ms (pre 먼저)"),
             (-20.0, C_DEP, "Δt=-20 ms (post 먼저)")]
    for dt, col, lab in demos:
        c = calcium_trace(t, [0.0], [dt], p)
        axA.plot(t, c, color=col, lw=2.2, label=lab)
    axA.axhline(p.theta_p, color=C_POT, ls="--", lw=1.3)
    axA.axhline(p.theta_d, color=C_DEP, ls="--", lw=1.3)
    axA.text(248, p.theta_p + 0.05, "θ_p", ha="right", color=C_POT, fontsize=10)
    axA.text(248, p.theta_d + 0.05, "θ_d", ha="right", color=C_DEP, fontsize=10)
    axA.set_xlim(-60, 150); axA.set_ylim(0, 3.0)
    axA.set_xlabel("시간 [ms]"); axA.set_ylabel("칼슘 c")
    axA.set_title("A. Δt 에 따라 칼슘 봉우리가 달라진다\n"
                  "pre 먼저→겹쳐서 봉우리↑ · post 먼저→낮음", fontsize=11)
    axA.legend(fontsize=9, loc="upper right"); axA.grid(alpha=0.25)

    # B. α_p, α_d vs Δt -----------------------------------------------------
    axB.plot(dts, ap * 100, color=C_POT, lw=2.2, label="α_p (θ_p 초과 시간)")
    axB.plot(dts, ad * 100, color=C_DEP, lw=2.2, label="α_d (θ_d 초과 시간)")
    axB.axvline(0, color="0.6", lw=0.8, ls=":")
    axB.set_xlabel("Δt = t_post - t_pre [ms]"); axB.set_ylabel("문턱 초과 시간 비율 [%]")
    axB.set_title("B. 문턱 위에 머무는 시간 (Fig 2B)\n"
                  "이 시간이 강화·약화의 '양'을 정한다", fontsize=11)
    axB.legend(fontsize=9); axB.grid(alpha=0.25)

    # C. 순 구동 = STDP 곡선 형태 -------------------------------------------
    axC.axhline(0, color="0.5", lw=1.0)
    axC.plot(dts, net, color="k", lw=2.2)
    axC.fill_between(dts, 0, net, where=net > 0, color=C_POT, alpha=0.3)
    axC.fill_between(dts, 0, net, where=net < 0, color=C_DEP, alpha=0.3)
    # LTP/LTD 라벨
    axC.text(0.60, 0.90, "LTP (강화)", transform=axC.transAxes, color=C_POT,
             fontsize=11, ha="center")
    axC.text(0.22, 0.12, "LTD (약화)", transform=axC.transAxes, color=C_DEP,
             fontsize=11, ha="center")
    axC.axvline(0, color="0.6", lw=0.8, ls=":")
    axC.set_xlabel("Δt = t_post - t_pre [ms]")
    axC.set_ylabel("순 구동  γ_p·α_p - γ_d·α_d")
    axC.set_title("C. 순 구동의 부호 = 가소성 방향\n"
                  "(경계 ρ*=0.5 에서 STDP 곡선 형태)", fontsize=11)
    axC.grid(alpha=0.25)

    fig.suptitle("칼슘으로 STDP 설명 — 스파이크 타이밍 → 칼슘 → 강화/약화 균형  "
                 f"[DP 세트: θ_d={p.theta_d}, θ_p={p.theta_p}, D={p.D} ms]",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "2_stdp_from_calcium.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 콘솔 요약
    zero_cross = dts[np.where(np.diff(np.sign(net)))[0]]
    print("[슬라이드] STDP-from-calcium 그림 저장")
    print(f"  순 구동 부호전환(LTD↔LTP) Δt ~ {zero_cross}")
    print(f"  Δt=+10: α_p={np.interp(10,dts,ap)*100:.2f}% "
          f"Δt=-20: α_p={np.interp(-20,dts,ap)*100:.2f}%")
    print(f"  → 저장: {out}")


if __name__ == "__main__":
    main()
