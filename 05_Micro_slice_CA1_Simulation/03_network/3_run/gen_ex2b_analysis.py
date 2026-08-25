# -*- coding: utf-8 -*-
"""Ex2b 분석 그래프 — 매트릭스 데이터(예시 또는 실측)에서 4패널 분석 PNG.
① PPR@50 vs U (파라미터→거동 검증)  ② 클래스별 PPR 분포(STP 분류)
③ 표적특이 STP (같은 pre, 다른 post)  ④ STP 곡선 오버레이(PPR vs ISI)
실행: python gen_ex2b_analysis.py [--src scratch/ex2b_example.json] [--out ...png]
라벨은 영문(matplotlib 한글 결자 회피).
"""
import os, io, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def arg(f, d):
    return sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d


SRC = arg("--src", os.path.join(ROOT, "scratch", "ex2b_example.json"))
OUT = arg("--out", os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "figures", "ex2b_analysis_예시.png"))
CCOL = {"E1": "#f59e0b", "E2": "#ef4444", "I1": "#14b8a6", "I2": "#3b82f6", "I3": "#8b5cf6"}


def main():
    d = json.load(io.open(SRC, encoding="utf-8"))
    P = d["pairs"]; example = d.get("example", False)
    isis = d.get("isis", [20, 50, 100, 200])
    j50 = isis.index(50) if 50 in isis else 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9)); ax = ax.ravel()
    ttl = "Ex2b connection bench" + (" — EXAMPLE (Tsodyks-Markram prediction)" if example else " — measured")
    fig.suptitle(ttl, fontsize=13, weight="bold")

    # ① PPR@50 vs U
    for c in CCOL:
        xs = [p["U"] for p in P if p["cls"] == c]; ys = [p["pprs"][j50] for p in P if p["cls"] == c]
        if xs: ax[0].scatter(xs, ys, s=28, c=CCOL[c], label=c, alpha=.8, edgecolor="white", linewidth=.5)
    ax[0].axhline(1, color="#888", ls="--", lw=.8); ax[0].set_xlabel("U (release prob)"); ax[0].set_ylabel("PPR @ ISI 50ms")
    ax[0].set_title("(1) PPR vs U — high U -> depression", fontsize=11); ax[0].legend(fontsize=8, ncol=5)
    ax[0].annotate("facilitation", (0.02, 1.6), fontsize=8, color="#c0392b"); ax[0].annotate("depression", (0.6, 0.4), fontsize=8, color="#2471a3")

    # ② 클래스별 PPR 분포
    cls_order = ["E1", "E2", "I1", "I2", "I3"]
    data = [[p["pprs"][j50] for p in P if p["cls"] == c] for c in cls_order]
    bp = ax[1].boxplot([x if x else [np.nan] for x in data], patch_artist=True, widths=.6)
    ax[1].set_xticks(range(1, len(cls_order) + 1)); ax[1].set_xticklabels(cls_order)
    for patch, c in zip(bp["boxes"], cls_order): patch.set_facecolor(CCOL[c]); patch.set_alpha(.6)
    for c, xs in zip(cls_order, data):                      # 개별 점 오버레이
        if xs: ax[1].scatter([cls_order.index(c) + 1] * len(xs), xs, s=10, c=CCOL[c], alpha=.5, zorder=3)
    ax[1].axhline(1, color="#888", ls="--", lw=.8); ax[1].set_ylabel("PPR @ ISI 50ms")
    ax[1].set_title("(2) PPR by synapse class (STP taxonomy)", fontsize=11)

    # ③ 표적특이 STP: pre=SP_PC 가 다른 post로
    pcp = [p for p in P if p["pre"] == "SP_PC"]
    pcp.sort(key=lambda p: p["pprs"][j50])
    if pcp:
        cols = [CCOL[p["cls"]] for p in pcp]
        ax[2].bar(range(len(pcp)), [p["pprs"][j50] for p in pcp], color=cols, alpha=.8)
        ax[2].set_xticks(range(len(pcp))); ax[2].set_xticklabels([p["post"].replace("SP_", "").replace("SO_", "").replace("SR_", "").replace("SLM_", "") for p in pcp], rotation=45, ha="right", fontsize=8)
        ax[2].axhline(1, color="#888", ls="--", lw=.8); ax[2].set_ylabel("PPR @ 50ms")
        ax[2].set_title("(3) Target-specific STP: PC -> different targets", fontsize=11)

    # ④ STP 곡선 오버레이 (대표 경로)
    key = ["SP_PC->SP_PC", "SC->SP_PC", "SP_PVBC->SP_PC", "SP_PC->SO_OLM", "SP_CCKBC->SP_PC", "SP_Ivy->SP_PC"]
    bk = {p["pre"] + "->" + p["post"]: p for p in P}
    for k in key:
        p = bk.get(k)
        if p: ax[3].plot(isis, p["pprs"], "o-", color=CCOL[p["cls"]], label=f"{k} ({p['cls']})", lw=1.6, ms=5)
    ax[3].axhline(1, color="#888", ls="--", lw=.8); ax[3].set_xlabel("ISI (ms)"); ax[3].set_ylabel("PPR")
    ax[3].set_title("(4) STP curves (PPR vs ISI) — key pathways", fontsize=11); ax[3].legend(fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=130); print(f"[analysis] -> {OUT}")


if __name__ == "__main__":
    main()
