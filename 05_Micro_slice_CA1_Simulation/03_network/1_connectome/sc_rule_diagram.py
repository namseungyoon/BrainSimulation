# -*- coding: utf-8 -*-
"""
03_network/1_connectome/sc_rule_diagram.py  —  3-1 설계 모식도

Schaffer(SC) 연결 규칙을 그림으로: 소마 위치가 아니라 '수상돌기가 지나는 층'
(SR/SO)에 SC가 붙는다는 원칙 + 세포별 SC 수용 표.
결과: figures/3-1_sc_rule.png

실행: python 03_network/1_connectome/sc_rule_diagram.py
"""
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# 층 경계 (1-5 결과, 중앙라인 r-범위)
LAYERS = [("SO", -330, -65, "#4C72B0"), ("SP", -65, 25, "#DD8452"),
          ("SR", 25, 450, "#55A868"), ("SLM", 450, 650, "#C44E52")]


def pyramidal(ax, x0):
    """추체 모식: 소마 SP, 정단↑(SR→SLM tuft), 기저↓(SO)."""
    rng = np.random.default_rng(1)
    soma_r = -20
    ax.plot(x0, soma_r, "o", ms=16, color="black", zorder=6)
    # 정단 줄기 SP→SLM
    ax.plot([x0, x0], [soma_r, 560], color="black", lw=2.2, zorder=5)
    # 정단 사선가지 (SR)
    for rr in np.linspace(60, 430, 7):
        s = rng.choice([-1, 1]); ln = rng.uniform(35, 70)
        ax.plot([x0, x0 + s * ln], [rr, rr + rng.uniform(20, 45)], color="0.25", lw=1.0, zorder=4)
    # SLM tuft
    for ang in np.linspace(-1.0, 1.0, 7):
        ax.plot([x0, x0 + ang * 95], [560, 630], color="0.25", lw=1.0, zorder=4)
    # 기저 (SO)
    for ang in np.linspace(-1.1, 1.1, 7):
        ax.plot([x0, x0 + ang * 80], [soma_r, -300], color="0.25", lw=1.1, zorder=4)


def synapses(ax, x0):
    rng = np.random.default_rng(3)
    # SC: SR(정단 하부) + SO(기저)  = 빨강
    sr = rng.uniform(40, 440, 60); sr_x = x0 + rng.uniform(-70, 70, 60)
    so = rng.uniform(-300, -70, 34); so_x = x0 + rng.uniform(-70, 70, 34)
    ax.scatter(np.r_[sr_x, so_x], np.r_[sr, so], s=26, c="#D6202B", edgecolors="white",
               linewidths=0.4, zorder=7, label="Schaffer 시냅스 (SR+SO)")
    # 내후각피질(perforant path): SLM = 초록 (대비용, SC 아님)
    pp = rng.uniform(470, 630, 16); pp_x = x0 + rng.uniform(-90, 90, 16)
    ax.scatter(pp_x, pp, s=26, c="#2E8B57", edgecolors="white", linewidths=0.4, zorder=7,
               label="Perforant path (SLM · SC 아님)")


def main():
    fig, (axd, axt) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.15, 1]})

    # ── 왼쪽: 층 + 추체 모식 + 시냅스
    for name, lo, hi, col in LAYERS:
        axd.add_patch(Rectangle((0, lo), 10, hi - lo, color=col, alpha=0.16, zorder=0))
        axd.text(0.15, (lo + hi) / 2, name, fontsize=13, fontweight="bold", va="center", color=col)
    pyramidal(axd, 5.2)
    synapses(axd, 5.2)
    axd.annotate("소마 (SP)", (5.2, -20), (7.4, -160), fontsize=11,
                 arrowprops=dict(arrowstyle="->", color="black"))
    axd.text(5.0, 700, "추체세포: 소마는 SP,\n수상돌기는 SR·SO·SLM로 뻗음", ha="center",
             fontsize=11, fontweight="bold")
    axd.set_xlim(0, 10); axd.set_ylim(-360, 760); axd.set_xticks([])
    axd.set_ylabel("층관통 r (µm, SP=0 · +r=SR/SLM)")
    axd.legend(loc="lower right", fontsize=9, framealpha=0.9)
    axd.set_title("SC는 '소마 위치'가 아니라 '수상돌기가 지나는 층(SR·SO)'에 붙는다")

    # ── 오른쪽: 표
    axt.axis("off")
    rows = [["세포", "소마 위치", "SC 받나?", "어디에"],
            ["추체 SP_PC", "SP", "O  대량", "SR 정단 + SO 기저"],
            ["SR 억제뉴런\n(SR_SCA 등)", "SR", "O", "SR 수상돌기"],
            ["SO 억제뉴런\n(일부)", "SO", "O  일부", "SO 수상돌기"],
            ["OLM (SO)", "SO", "X  거의 안받음", "주로 추체 축삭에서"]]
    tb = axt.table(cellText=rows, cellLoc="center", loc="center", bbox=[0.0, 0.30, 1.0, 0.60])
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 2.0)
    for j in range(4):
        tb[0, j].set_facecolor("#333333"); tb[0, j].set_text_props(color="white", fontweight="bold")
    tb[1, 2].set_facecolor("#E8F5E9"); tb[1, 0].set_text_props(fontweight="bold")
    axt.text(0.5, 0.94, "세포별 Schaffer 수용 규칙", ha="center", fontsize=13, fontweight="bold",
             transform=axt.transAxes)
    axt.text(0.5, 0.24,
             "판정 = 세포의 '수상돌기 세그먼트'가 SR/SO 층 안에 있는가\n"
             "(소마 위치 아님 → 추체는 소마가 SP여도 SC 대량 수용)",
             ha="center", va="top", fontsize=10.5, transform=axt.transAxes, color="#333")
    axt.text(0.5, 0.10,
             "SC 수렴도: 추체당 ~20,878 시냅스 · 억제뉴런 ~12,714\n"
             "SLM tuft·축삭 제외 · fEPSP는 이 SC 전류가 SR–SP 쌍극자로 생성",
             ha="center", va="top", fontsize=9.5, transform=axt.transAxes, color="#666")

    fig.suptitle("3-1  Schaffer(SC) 연결 규칙 — 마이크로슬라이스 CA1", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "3-1_sc_rule.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[3-1] 저장 -> {out}")


if __name__ == "__main__":
    main()
