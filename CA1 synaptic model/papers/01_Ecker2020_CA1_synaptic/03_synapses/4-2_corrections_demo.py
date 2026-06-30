"""
s4_corrections.py — 스텝 5: §2.7 식(12)~(13) 칼슘/온도 보정
============================================================================
Source: Ecker et al. (2020) §2.7, Eq.(12) Hill, Eq.(13) Q10.

왜 필요한가: 실험마다 조건([Ca2+]o, 온도)이 달라 파라미터를 한 기준으로 통일해야 한다.
  - 식(12) 칼슘 보정: 세포외 칼슘 [Ca2+]o 에 따라 방출확률 U_SE 가 Hill 곡선으로 변함.
  - 식(13) 온도 보정: 기록 온도가 다르면 시정수 τ 를 Q10 로 환산.

실행:
    conda activate ca1sim
    python SourceCode/02_synapse_model/s4_corrections.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.plotstyle import set_korean_font          # noqa: E402
from common.corrections import hill_ca, q10_scale     # noqa: E402

set_korean_font()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("스텝 5 — §2.7 식(12)~(13): 칼슘·온도 보정 / Ca & temperature correction",
                 fontsize=13, fontweight="bold")

    # (A) 식(12) Hill: 칼슘 → 방출확률
    ca = np.linspace(0, 4, 400)
    axA.plot(ca, hill_ca(ca, 1.0, 2.79), color="tab:red", lw=2.2,
             label="가파름 steep (K½=2.79)")
    axA.plot(ca, hill_ca(ca, 1.0, 1.09), color="tab:blue", lw=2.2,
             label="완만 shallow (K½=1.09)")
    for x, txt in [(2.0, "시험관 in vitro\n2.0 mM"), (1.2, "생체 in vivo\n1.2 mM")]:
        axA.axvline(x, color="0.6", ls=":", lw=1)
        axA.text(x + 0.05, 0.05, txt, fontsize=8, color="0.3")
    axA.set_title("(A) 식(12) 칼슘→방출확률 (Hill) / Ca→U_SE", fontsize=10)
    axA.set_xlabel("세포외 칼슘 [Ca2+]o (mM)")
    axA.set_ylabel("방출확률 U_SE (정규화 / release prob.)")
    axA.legend(fontsize=9)

    # (B) 식(13) Q10: 온도 → 시정수
    T = np.linspace(20, 38, 200)
    tau_exp, T_exp, Q10 = 10.0, 25.0, 2.2     # 예: 25°C에서 측정한 τ=10 ms
    axB.plot(T, q10_scale(tau_exp, Q10, T_exp, T), color="tab:green", lw=2.4)
    for x, txt, c in [(25, "실험 온도 25°C", "0.3"), (34, "시뮬 온도 34°C", "tab:red")]:
        axB.axvline(x, color=c, ls=":", lw=1.2)
    tau34 = q10_scale(tau_exp, Q10, T_exp, 34.0)
    axB.plot([25, 34], [tau_exp, tau34], "ko", ms=6)
    axB.annotate(f"τ {tau_exp:.1f}→{tau34:.1f} ms\n(더 빨라짐 / faster)",
                 xy=(34, tau34), xytext=(29, tau_exp * 0.75), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="0.4"))
    axB.set_title("(B) 식(13) 온도 보정 (Q10=2.2) / temperature scaling", fontsize=10)
    axB.set_xlabel("온도 temperature (°C)")
    axB.set_ylabel("시정수 τ (ms)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "4-2_corrections_demo.png")
    plt.savefig(out, dpi=120)

    u_vitro = hill_ca(2.0, 1.0, 2.79); u_vivo = hill_ca(1.2, 1.0, 2.79)
    print(f"[그림] {out}  (한글폰트={plt.rcParams['font.family']})")
    print(f"[검증A] 가파른 칼슘의존: U_SE(2.0mM)={u_vitro:.2f} vs U_SE(1.2mM)={u_vivo:.2f} "
          f"→ 생체(저칼슘)에서 방출확률 {u_vitro/u_vivo:.1f}배 낮음")
    print(f"[검증B] τ: 25°C 10.0 ms → 34°C {tau34:.2f} ms (Q10=2.2 로 빨라짐)")


if __name__ == "__main__":
    main()
