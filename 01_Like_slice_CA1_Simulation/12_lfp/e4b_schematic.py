# -*- coding: utf-8 -*-
"""12_lfp/e4b_schematic.py  —  한 전극에서 fEPSP가 만들어지는 순서 도식 (수식 적용 위치)

직관 도식: 뉴런 발화(스파이크/EPSP) -> 막전류 -> [공식: 거리·MEA 영상법] -> 세포별 합 -> 전극 fEPSP.
수식이 '어디서' 적용되고 값이 '어떤 순서로' 만들어지는지 화살표·번호로 표현.
실행: <ca1sim>/python.exe 12_lfp/e4b_schematic.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def box(ax, x, y, w, hh, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="0.3", lw=1.3, zorder=3))
    ax.text(x + w / 2, y + hh - 1.8, title, ha="center", va="top", fontsize=10.5, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + hh - 4.6, sub, ha="center", va="top", fontsize=8.2, zorder=4)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 color="#c0392b", lw=2.0, zorder=2))


def curve(ax, x0, x1, y0, y1, xs, ys, color):
    xs = np.asarray(xs); ys = np.asarray(ys)
    X = x0 + (xs - xs.min()) / (np.ptp(xs)) * (x1 - x0)
    Y = y0 + (ys - ys.min()) / (max(np.ptp(ys), 1e-9)) * (y1 - y0)
    ax.plot(X, Y, color=color, lw=1.6, zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    tt = np.linspace(0, 1, 200)
    spike = np.exp(-((tt - 0.4) / 0.03) ** 2) * 1.0 - 0.05                       # 스파이크
    imem = -np.exp(-((tt - 0.4) / 0.04) ** 2) + 0.5 * np.exp(-((tt - 0.5) / 0.08) ** 2)  # 막전류 이중파형
    fepsp = -(np.exp(-((tt - 0.45) / 0.09) ** 2)) + 0.06                          # 음성 fEPSP

    # ── 상단 파이프라인 5단계 ──
    y0 = 74; w = 16; hh = 18
    xs_box = [3, 23, 43, 63, 83]
    box(ax, xs_box[0], y0, w, hh, "① 뉴런 발화", "각 세포 막전위 V_m(t)\n(스파이크·시냅스 EPSP)", "#fdebd0")
    curve(ax, xs_box[0] + 2, xs_box[0] + w - 2, y0 + 1.5, y0 + 8, tt, spike, "#c0392b")
    box(ax, xs_box[1], y0, w, hh, "② 막전류 추출", "세그먼트별 I_i(t) [nA]\n(use_fast_imem)\nsink(-)/source(+)", "#eafaf1")
    curve(ax, xs_box[1] + 2, xs_box[1] + w - 2, y0 + 1.5, y0 + 7, tt, imem, "#16a085")
    box(ax, xs_box[2], y0, w, hh, "③ 공식 적용", "각 전류 × 전달가중치\n거리 1/r + MEA 영상법\n(아래 수식)", "#eaf2fb")
    ax.text(xs_box[2] + w / 2, y0 + 4.5, r"$\frac{I_i}{4\pi\sigma\,r_i}\times$영상법", ha="center", fontsize=10, color="#21618c", zorder=5)
    box(ax, xs_box[3], y0, w, hh, "④ 세포별 합", "세그먼트 Σ = 한 세포 기여\n$V_{cell}=\\Sigma_i (\\cdots)$", "#f4ecf7")
    box(ax, xs_box[4], y0, w, hh, "⑤ 전극 fEPSP", "전체 세포 Σ = 측정값\n$V_e(t)=\\Sigma_{cell}V_{cell}$", "#fdedec")
    curve(ax, xs_box[4] + 2, xs_box[4] + w - 2, y0 + 1.5, y0 + 8, tt, fepsp, "#c0392b")
    for i in range(4):
        arrow(ax, xs_box[i] + w, y0 + hh / 2, xs_box[i + 1], y0 + hh / 2)

    # ── 중앙: 핵심 수식 (③에서 적용) ──
    ax.add_patch(FancyBboxPatch((8, 46), 84, 20, boxstyle="round,pad=0.6,rounding_size=1.5",
                                fc="#f8f9fb", ec="#21618c", lw=1.6, zorder=3))
    ax.text(50, 63.5, "③ 공식이 적용되는 곳 — 값이 만들어지는 순서", ha="center", fontsize=11, fontweight="bold", color="#21618c")
    ax.text(50, 57.8,
            r"$V_e(\mathbf{r}_j)\;=\;\sum_{cell}\;\sum_{i\,\in\,seg}\;\frac{I_i(t)}{4\pi\sigma_T}\;\cdot\;"
            r"\left[\,\frac{2}{r_i}\,+\,2\sum_{n}W_{TS}^{\,n}(\cdots)\,\right]$",
            ha="center", va="center", fontsize=17, color="#154360")
    ax.text(50, 51.8,
            "②  $I_i$=세그먼트 막전류    ③  [ ]=거리 $1/r_i$ + MEA 영상법(유리×2·식염수 $W_{TS}$)    "
            "④  $\\Sigma_{seg}$=한 세포    ⑤  $\\Sigma_{cell}$=전극 fEPSP",
            ha="center", fontsize=8.3, color="#5d4037")
    ax.text(50, 48.5, "단위: I[nA]·거리[µm]·σ[S/m] → 곧바로 mV   ·   유리 절연=×2   ·   식염수 감쇠 $W_{TS}=-2/3$",
            ha="center", fontsize=8.5, color="0.4")

    # ── 하단: 물리 도식 (전극 + 세포 + 거리) ──
    gx0, gx1, gy = 8, 92, 6
    ax.add_patch(Rectangle((gx0, gy - 2.5), gx1 - gx0, 2.5, fc="#34495e", zorder=2))   # 유리 MEA
    ax.text(gx0 + 1, gy - 1.3, "유리 MEA (전극, z=0)", fontsize=8, color="w", va="center")
    ax.add_patch(Rectangle((gx0, gy), gx1 - gx0, 30, fc="#fdf6e3", alpha=0.5, zorder=1))  # 조직
    ax.text(gx1 - 1, gy + 28, "조직 슬라이스", fontsize=8, ha="right")
    ex = 50
    ax.plot([ex], [gy], "s", color="red", ms=14, zorder=6)
    ax.text(ex, gy - 4, "전극 j", ha="center", fontsize=9, color="red", fontweight="bold")
    # 세포 3개(다이폴): soma 아래(유리쪽 아님) / SR 위 — 정단이 유리쪽
    for cx, r_lab in [(30, "r₁"), (52, "r₂"), (74, "r₃")]:
        ax.plot([cx, cx], [gy + 4, gy + 26], color="0.45", lw=4, solid_capstyle="round", zorder=4)  # 세포체
        ax.plot(cx, gy + 6, "v", color="#1a5276", ms=9, zorder=5)      # SR sink(유리쪽 아래)
        ax.plot(cx, gy + 25, "^", color="#922b21", ms=9, zorder=5)     # soma source(위)
        ax.add_patch(FancyArrowPatch((cx, gy + 6), (ex, gy + 0.4), arrowstyle="-|>", mutation_scale=12,
                                     color="#2980b9", lw=1.3, ls="--", zorder=3, alpha=0.8))
        ax.text((cx + ex) / 2, gy + 3.3, r_lab, fontsize=9, color="#1f618d")
    ax.text(30, gy + 28.5, "sink(-)=SR 흥분 시냅스", fontsize=7.5, color="#1a5276", ha="center")
    ax.text(74, gy + 28.5, "source(+)=소마 귀환", fontsize=7.5, color="#922b21", ha="center")
    ax.text(50, gy + 32.5, "각 세포의 각 세그먼트 전류가 거리(r)만큼 약해져 유리면 전극 j에 합산 → fEPSP",
            ha="center", fontsize=9, style="italic", color="0.3")

    fig.suptitle("한 전극에서 fEPSP가 만들어지는 순서 —  ① 발화 → ② 막전류 → ③ 공식(거리·MEA 영상법) → ④ 세포별 합 → ⑤ 전극 fEPSP",
                 fontsize=12.5, fontweight="bold", y=0.98)
    out = os.path.join(FIG, "E4b_schematic.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
