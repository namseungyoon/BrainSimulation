# -*- coding: utf-8 -*-
"""
03_network/3_run/assemble_figure.py  —  물리적 구조 완성: 조립 메타데이터 그림

assemble_scaling.py 가 저장한 scratch/assemble_scaling.json(누적 세포수별 section·
segment·RSS·시간)을 읽어 (a) 메모리 확장 (b) segment 확장 그래프를 그리고,
전체 조립 요약(총 section/segment·메모리·시간)을 콘솔에 낸다.

실행: python 03_network/3_run/assemble_figure.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    d = json.load(open(os.path.join(ROOT, "scratch", "assemble_scaling.json")))
    rows = np.array(d["rows"], float)   # [built, nsec, nseg, rss_MB, elapsed_s]
    cells, nsec, nseg, rss, el = rows.T
    n = int(d["n"]); tot = d["total_s"]
    base = rss[0] - (rss[1] - rss[0]) / (cells[1] - cells[0]) * cells[0] if len(rss) > 1 else 45
    per_cell_mb = (rss[-1] - 45) / cells[-1]
    per_cell_ms = tot / cells[-1] * 1000

    print(f"=== 물리적 구조 완성 · 전체 {n:,} 세포 조립 ===")
    print(f"[총 구획] section {int(nsec[-1]):,} · segment {int(nseg[-1]):,}")
    print(f"[메모리]  최종 RSS {rss[-1]:,.0f}MB · 세포당 ~{per_cell_mb:.1f}MB")
    print(f"[시간]    총 {tot:,.0f}s ({tot/60:.1f}분) · 세포당 ~{per_cell_ms:.0f}ms")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].plot(cells, rss / 1024.0, "o-", color="#4C72B0", lw=2, ms=6)
    ax[0].set_xlabel("누적 세포 수"); ax[0].set_ylabel("RSS 메모리 (GB)")
    ax[0].set_title(f"(a) 메모리 확장 — 최종 {rss[-1]/1024:.1f}GB (세포당 ~{per_cell_mb:.1f}MB)")
    ax[0].grid(alpha=0.3)
    for x, y in zip(cells, rss / 1024.0):
        ax[0].annotate(f"{y:.1f}", (x, y), fontsize=8, ha="center", va="bottom")

    ax[1].plot(cells, nseg / 1e3, "s-", color="#C44E52", lw=2, ms=6, label="segment")
    ax[1].plot(cells, nsec / 1e3, "^-", color="#55A868", lw=2, ms=6, label="section")
    ax[1].set_xlabel("누적 세포 수"); ax[1].set_ylabel("구획 수 (×10³)")
    ax[1].set_title(f"(b) 구획 확장 — 총 seg {int(nseg[-1]):,} · sec {int(nsec[-1]):,}")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f"물리적 구조 완성 — 전체 {n:,} 완전형태 세포 NEURON 조립 "
                 f"(총 {tot/60:.1f}분·{rss[-1]/1024:.1f}GB)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phys_assembly.png"), dpi=130)
    plt.close(fig)
    print(f"[그림] -> {FIG}/phys_assembly.png")


if __name__ == "__main__":
    main()
