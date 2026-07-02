"""
1_calcium_trace.py — 단계 1: 칼슘 트레이스 c(t)
============================================================================
Source: Graupner & Brunel (2012) Results, Fig 1A / Fig 2A.
    c(t) = Σ C_pre·exp(-(t-t_pre-D)/τ_Ca)·Θ  +  Σ C_post·exp(-(t-t_post)/τ_Ca)·Θ

확인 목표:
  (A) pre-post 스파이크 쌍이 만드는 칼슘 트레이스의 모양 (지연 D, 감쇠 τ_Ca, 선형 합산)
  (B) 스파이크 시간차 Δt 부호에 따라 봉우리와 θ_p/θ_d 초과 정도가 어떻게 바뀌는가
      → Δt>0 (pre 먼저): 두 전이가 겹쳐 봉우리↑
      → Δt<0 (post 먼저): pre 지연 D 때문에 덜 겹침

실행:
    conda activate ca1sim
    python papers/02_Graupner2012_Calcium-based_Plasticity_Model/01_calcium_dynamics/1_calcium_trace.py
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


def main():
    p = PARAM_SETS["demo_fig2"]        # C_pre=1, C_post=2, D=13.7, tau_ca=20, θ_d=1.0, θ_p=1.3
    t = np.arange(-60.0, 150.0, 0.05)

    # 두 시나리오: Δt = t_post - t_pre (pre@0 고정)
    scenarios = [
        ("Δt = +20 ms  (pre 먼저)", 0.0, 20.0),
        ("Δt = -20 ms  (post 먼저)", 0.0, -20.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, t_pre, dt) in zip(axes, scenarios):
        t_post = t_pre + dt
        c = calcium_trace(t, [t_pre], [t_post], p)
        c_pre_only = calcium_trace(t, [t_pre], [], p)
        c_post_only = calcium_trace(t, [], [t_post], p)
        ap, ad = time_above_thresholds(t, c, p)

        ax.plot(t, c, color="k", lw=2.0, label="c(t) 합산", zorder=3)
        ax.plot(t, c_pre_only, color="#1f77b4", lw=1.0, ls="--", label="pre 기여 (C_pre)")
        ax.plot(t, c_post_only, color="#d62728", lw=1.0, ls="--", label="post 기여 (C_post)")

        ax.axhline(p.theta_p, color="orange", lw=1.2, ls=":", label=f"θ_p = {p.theta_p}")
        ax.axhline(p.theta_d, color="turquoise", lw=1.2, ls=":", label=f"θ_d = {p.theta_d}")
        ax.axvline(t_pre, color="#1f77b4", lw=0.8, alpha=0.5)
        ax.axvline(t_post, color="#d62728", lw=0.8, alpha=0.5)

        ax.set_title(f"{title}\nθ_p 초과 {ap*100:.1f}% · θ_d 초과 {ad*100:.1f}%", fontsize=10)
        ax.set_xlabel("시간 [ms]")
        ax.set_xlim(-60, 150)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("칼슘 농도 c  (θ 기준, 무차원)")
    axes[0].legend(fontsize=8, loc="upper right")

    fig.suptitle("단계 1 — 칼슘 트레이스 c(t): pre·post 스파이크 쌍 "
                 f"(C_pre={p.C_pre}, C_post={p.C_post}, D={p.D} ms, τ_Ca={p.tau_ca} ms)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "1_calcium_trace.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 콘솔 요약 (실행 로그로 수치 확정)
    print("[단계1] 칼슘 트레이스 생성 완료")
    for title, t_pre, dt in scenarios:
        c = calcium_trace(t, [t_pre], [t_pre + dt], p)
        ap, ad = time_above_thresholds(t, c, p)
        print(f"  {title:26s}: max c = {c.max():.3f}, "
              f"θ_p 초과 {ap*100:5.2f}%, θ_d 초과 {ad*100:5.2f}%")
    print(f"  → 저장: {out}")


if __name__ == "__main__":
    main()
