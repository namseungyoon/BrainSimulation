"""
1_innervation.py — 단계 1: 축삭-수상돌기 분포 (axo-dendritic innervation profile)
[8개 파라미터 추적] 이 단계: 0/8 — 해부 검증(파라미터 아님)
============================================================================
Source: Ecker et al. (2020) Fig.2 단계1, §2.1, Fig.3a/4a.

논문 단계1 = "어떤 presynaptic 유형이 postsynaptic 수상돌기의 **어느 깊이**에 시냅스를 놓나".
예: Schaffer(SC)→PC 는 중간 apical(SR), O-LM→PC 는 먼 apical(SLM), PC→PC(E-E)는 기저(basal)+근위.
실제 PC 형태에서 소마로부터의 경로거리(path distance)별로 세 유형의 분포를 보여준다.

실행: conda activate ca1sim
      python papers/01_Ecker2020_CA1_synaptic/03_synapses/1_innervation.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.nrn_env import h            # noqa: E402
from common.plotstyle import set_korean_font   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_recording import load_pc    # noqa: E402

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# presynaptic 유형별 표적 수상돌기 깊이(소마 경로거리 µm) 규칙 — 논문 정성 패턴
# (라벨, 표적구역, (lo,hi), 색, 짧은 레이어표기)
RULES = [
    ("PC->PC (E-E): 기저+근위 / basal+proximal", "dend", (0, 150), "tab:red",   "기저·근위 basal"),
    ("SC->PC: 중간 apical(SR) / mid apical",     "apic", (100, 350), "tab:green", "중간 apical SR"),
    ("O-LM->PC: 먼 apical(SLM) / distal apical", "apic", (350, 9999), "tab:blue", "먼 apical SLM"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    cell, _ = load_pc()
    soma = cell.soma[0]
    h.distance(0, soma(0.5))   # 소마를 원점으로

    # 수상돌기 구역별 (거리, 섹션) 수집
    domains = {"apic": [], "dend": []}
    for key, seclist in (("apic", cell.apic), ("dend", cell.dend)):
        for sec in seclist:
            domains[key].append((h.distance(sec(0.5)), sec))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("단계 1 — 축삭-수상돌기 분포 / axo-dendritic innervation profile (PC)",
                 fontsize=12, fontweight="bold")

    # 전체 거리 최댓값(표적 윈도우 음영 클램프용)
    xmax = max(d for pts in domains.values() for d, _ in pts)

    ymax = 0          # 막대 최고 높이(여백 확보용)
    stats = []        # (라벨, 표적, 색, 레이어, n, dmin, dmax, lo, hi_clamped)
    for label, dom, (lo, hi), c, tag in RULES:
        ds = [d for (d, s) in domains[dom] if lo <= d < hi]
        if not ds:
            continue
        # 표적 거리 윈도우를 옅은 음영 밴드로 표시
        hi_c = min(hi, xmax)
        ax.axvspan(lo, hi_c, color=c, alpha=0.07, zorder=0)
        short = label.split(":")[0].strip().replace("->", "→")
        counts, _, _ = ax.hist(ds, bins=20, alpha=0.55, color=c,
                               label=short, zorder=2)
        ymax = max(ymax, counts.max())
        stats.append((label, dom, c, tag, len(ds), min(ds), max(ds), lo, hi_c))

    # 막대 위에 여백을 만들고, 각 클러스터 위에 표적/구획수/거리범위 정보 박스 표시
    ax.set_ylim(0, ymax * 1.32)
    ytext = ymax * 1.16
    for label, dom, c, tag, n, dmin, dmax, lo, hi_c in stats:
        xc = 0.5 * (dmin + dmax)            # 클러스터 중심(실제 데이터 기준)
        box = (f"{tag}\n"
               f"표적: {dom}\n"
               f"n = {n} 구획\n"
               f"{dmin:.0f}–{dmax:.0f} µm")
        ax.annotate(box, xy=(xc, ytext), ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="black", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.4", fc=c, ec="black",
                              lw=0.8, alpha=0.30))

    ax.set_xlabel("소마로부터 경로거리 path distance (µm)")
    ax.set_ylabel("시냅스 가능 구획 수 / # dendritic compartments")
    # 범례는 그래프 아래로 빼서 본문 박스와 겹치지 않게
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "1_innervation.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[그림] {out}")
    for label, dom, _c, _tag, n, dmin, dmax, _lo, _hi in stats:
        print(f"  {label:42s}: {n}개 구획, 거리 {dmin:.0f}~{dmax:.0f}µm")


if __name__ == "__main__":
    main()
