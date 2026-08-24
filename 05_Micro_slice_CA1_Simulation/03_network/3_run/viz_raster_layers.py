# -*- coding: utf-8 -*-
"""Ex1 volley 응답 raster + PSTH — 층(SO/SP/SR/SLM) 색분리 + 추체/억제 구분.

Brunel식 raster처럼 (위) 뉴런별 스파이크 점, (아래) 집단 발화율(PSTH).
우리 데이터는 자극 후 ~30ms 창의 volley 응답. y축은 층 순으로 묶고 층별 색.
추체(PC, 흥분)는 작은 점, 억제뉴런은 굵은 점으로 이중부호화(층 색 + E/I).
입력: scratch/mpi_baseline.npz (Ex1) + data/derived/window_cells.npz
출력: 03_network/3_run/figures/ex1_raster_layers.png
실행: python viz_raster_layers.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")

# 층 순서(해부 깊이: alveus쪽 SO → SP → SR → SLM) + 색 (구분 잘 되는 정성 팔레트)
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COL = {"SO": "#DD8452", "SP": "#4C72B0", "SR": "#55A868", "SLM": "#8172B3"}


def main():
    b = np.load(os.path.join(ROOT, "scratch", "mpi_baseline.npz"), allow_pickle=True)
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    lay = wc["layer"].astype(str); mt = wc["mtype"].astype(str)
    N = len(lay); is_pc = (mt == "SP_PC")
    st = b["spk_t"].astype(float); sid = b["spk_id"].astype(int)
    stim_t = float(b["stim_t"])
    m = sid < N; st = st[m]; sid = sid[m]
    t = st - stim_t                                   # 자극 기준 ms

    # y축 재배열: 층 순 → 각 층 내 gid 순으로 연속 행 배정
    order_key = np.array([LAYER_ORDER.index(l) if l in LAYER_ORDER else 99 for l in lay])
    row_of = np.empty(N, int)
    rows = np.lexsort((np.arange(N), order_key))       # 층→gid 정렬된 gid 목록
    row_of[rows] = np.arange(N)                        # gid → y행
    # 층 경계(밴드)
    sorted_layers = lay[rows]
    bands = {}
    for L in LAYER_ORDER:
        idx = np.where(sorted_layers == L)[0]
        if len(idx):
            bands[L] = (idx.min(), idx.max())

    y = row_of[sid]
    cols = np.array([LAYER_COL.get(lay[g], "#999999") for g in sid])
    pc_mask = is_pc[sid]

    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)

    # 층 배경 밴드(옅게)
    for L, (lo, hi) in bands.items():
        ax.axhspan(lo - 0.5, hi + 0.5, color=LAYER_COL[L], alpha=0.05, zorder=0)

    # raster: 추체(작은 점) + 억제(굵은 점)
    ax.scatter(t[pc_mask], y[pc_mask], s=3, c=cols[pc_mask], marker="o",
               linewidths=0, alpha=0.85, zorder=2)
    ax.scatter(t[~pc_mask], y[~pc_mask], s=26, c=cols[~pc_mask], marker="D",
               edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3)
    ax.axvline(0, color="crimson", lw=1.2, ls="--", zorder=1)
    ax.text(0.2, N * 0.99, "stim", color="crimson", fontsize=9, va="top")

    # 층 라벨
    for L, (lo, hi) in bands.items():
        ax.text(-1.8, (lo + hi) / 2, L, color=LAYER_COL[L], fontsize=11,
                fontweight="bold", va="center", ha="right")

    ax.set_ylabel("neuron  (grouped by layer)")
    ax.set_ylim(-0.5, N + 0.5); ax.set_xlim(-2, 31)
    ax.set_title("Ex1 volley response — raster by layer (SO/SP/SR/SLM)\n"
                 f"R=150 um · {int(b['fired'].sum())}/{N} fired · PC=dot, interneuron=diamond",
                 fontsize=11)
    ax.tick_params(labelbottom=False)

    # 범례
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=LAYER_COL[L],
                  markersize=8, label=L) for L in LAYER_ORDER]
    leg += [Line2D([0], [0], marker='o', color='w', markerfacecolor='#555', markersize=6, label='PC (exc)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#555', markeredgecolor='k', markersize=8, label='interneuron (inh)')]
    ax.legend(handles=leg, loc='upper right', fontsize=8, framealpha=0.9, ncol=2)

    # PSTH: 층별 집단 발화율 (Hz/세포, 0.5ms bin)
    bw = 0.5
    edges = np.arange(-2, 31 + bw, bw)
    for L in LAYER_ORDER:
        sel = np.array([lay[g] == L for g in sid])
        nL = int(np.sum(lay == L))
        if sel.sum() == 0 or nL == 0:
            continue
        hcnt, _ = np.histogram(t[sel], bins=edges)
        rate = hcnt / (nL * bw / 1000.0)              # Hz/세포
        axr.plot(edges[:-1] + bw / 2, rate, color=LAYER_COL[L], lw=1.4, label=L)
    axr.axvline(0, color="crimson", lw=1.0, ls="--")
    axr.set_xlabel("time from stim [ms]")
    axr.set_ylabel("rate [Hz]")
    axr.set_xlim(-2, 31)
    axr.legend(fontsize=7, ncol=4, loc='upper right')

    FIG = os.path.join(ROOT, "04_experiments", "Ex1_baseline", "figures")
    os.makedirs(FIG, exist_ok=True)
    outp = os.path.join(FIG, "ex1_raster_layers.png")
    fig.savefig(outp, dpi=140, bbox_inches="tight")
    print(f"[raster] 저장 {outp}", flush=True)
    # 요약
    for L in LAYER_ORDER:
        nsp = int(np.sum(np.array([lay[g] == L for g in sid]))) if len(sid) else 0
        print(f"   {L}: 세포 {int(np.sum(lay==L)):4d} · 스파이크 {nsp}", flush=True)


if __name__ == "__main__":
    main()
