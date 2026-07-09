# -*- coding: utf-8 -*-
"""
11_schaffer/e3_concept.py  —  E3 개념도 (피드포워드 억제 회로 + I-O 곡선)

교육용 모식도(시뮬 데이터 아님):
  (A) 회로: SC→PC 직접 흥분(1단계·빠름) + SC→인터뉴런→PC 억제(2단계·늦음) = 피드포워드 억제
  (B) I-O 곡선 개념: 정상(억제 있음)=완만·선형 vs 억제 차단(억제 없음)=급포화
주의: 이건 Romani Fig.4 '기대형' 개념도. 우리 현재 예비결과는 정상~억제차단 곡선 겹침(피드포워드 억제 미작동)→E3' 재작업.
실행: python 11_schaffer/e3_concept.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # ── (A) 회로 모식 ─────────────────────────────────────────────
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    nodes = {
        "SC": (0.13, 0.58, "#C0392B", "SC 입력\n(CA3 대용)"),
        "PC": (0.83, 0.70, "#DD8452", "추체세포\nPC"),
        "INT": (0.46, 0.20, "#4C72B0", "인터뉴런\n(억제)"),
    }
    for x, y, c, lab in nodes.values():
        axA.add_patch(Circle((x, y), 0.095, fc=c, ec="k", lw=1.2, alpha=0.9, zorder=3))
        axA.text(x, y, lab, ha="center", va="center", color="white",
                 fontsize=9.5, fontweight="bold", zorder=4)

    def arrow(p, q, color, label, rad, dy=0.03):
        axA.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22,
                      color=color, lw=2.4, connectionstyle=f"arc3,rad={rad}",
                      shrinkA=20, shrinkB=20, zorder=2))
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        axA.text(mx, my + dy, label, ha="center", va="center", fontsize=9,
                 color=color, fontweight="bold")

    arrow((0.13, 0.58), (0.83, 0.70), "#2E8B57", "직접 흥분(+)\n1단계·빠름", 0.18, 0.06)
    arrow((0.13, 0.58), (0.46, 0.20), "#2E8B57", "흥분(+)", -0.12, -0.02)
    arrow((0.46, 0.20), (0.83, 0.70), "#C0392B", "억제(-)\n2단계·늦음", -0.18, -0.05)
    axA.text(0.5, 0.96, "(A) 피드포워드 억제 회로", ha="center",
             fontsize=12, fontweight="bold")
    axA.text(0.5, 0.03,
             "[비유] 가속페달(SC→PC) + 브레이크(SC→인터뉴런→PC)\n"
             "브레이크가 살짝 늦음 → PC 발화 '시간창'이 좁음 (window of opportunity)",
             ha="center", va="bottom", fontsize=9, color="#333",
             bbox=dict(boxstyle="round", fc="#f5f5f2", ec="#bbb"))

    # ── (B) I-O 곡선 개념 ─────────────────────────────────────────
    x = np.linspace(0, 100, 200)
    y_ctrl = x                                   # 정상: 완만·선형(Romani R=0.992)
    y_gaba = 100.0 / (1.0 + np.exp(-(x - 15) / 4.0))   # 억제 차단: 급포화
    axB.plot(x, y_ctrl, color="#2f6fb0", lw=2.8, label="정상 (억제 있음) — 완만·선형")
    axB.plot(x, y_gaba, color="#C0392B", lw=2.8, label="억제 차단 (억제 없음) — 급포화")
    axB.fill_between(x, y_ctrl, y_gaba, where=(y_gaba > y_ctrl), color="#C0392B", alpha=0.08)
    axB.annotate("억제가 반응을\n완만하게 조절", (70, 70), (40, 88),
                 fontsize=9, color="#2f6fb0",
                 arrowprops=dict(arrowstyle="->", color="#2f6fb0"))
    axB.annotate("억제 차단 시\n조금만 자극해도 포화", (25, 96), (33, 55),
                 fontsize=9, color="#C0392B",
                 arrowprops=dict(arrowstyle="->", color="#C0392B"))
    axB.set_xlabel("활성 SC 축삭 (%)  = 입력 세기"); axB.set_ylabel("발화 PC (%)  = 출력")
    axB.set_title("(B) 입출력(I-O) 곡선 — 개념도", fontsize=12, fontweight="bold")
    axB.legend(fontsize=9, loc="lower right"); axB.grid(alpha=0.3)
    axB.set_xlim(0, 100); axB.set_ylim(0, 105)

    fig.suptitle("E3 개념 — Schaffer collateral 자극에 대한 피드포워드 억제와 I-O 곡선\n"
                 "개념도(Romani Fig.4 기대형) · 우리 현재 예비: 정상 ~ 억제차단 곡선 겹침(피드포워드 억제 미작동) → E3' 재작업 필요",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(FIG, "E3_concept.png")
    fig.savefig(out, dpi=135); plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
