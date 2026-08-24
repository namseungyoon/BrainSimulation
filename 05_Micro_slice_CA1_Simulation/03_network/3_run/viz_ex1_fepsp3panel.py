# -*- coding: utf-8 -*-
"""Ex1 volley 3패널 — (위) 층별 raster · (중) PSTH · (아래) 3전극 fEPSP.
입력: scratch/mpi_baseline.npz(스파이크) + scratch/mpi_fepsp.npz(fEPSP) + window_cells.
출력: 04_experiments/Ex1_baseline/figures/ex1_fepsp_3panel.png
실행: python 03_network/3_run/viz_ex1_fepsp3panel.py
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
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COL = {"SO": "#DD8452", "SP": "#4C72B0", "SR": "#55A868", "SLM": "#8172B3"}
ECOL = {"E1": "#e0a800", "E2": "#9b5de5", "E3": "#1fb2a6"}
XMAX = 30.0


def main():
    b = np.load(os.path.join(ROOT, "scratch", "mpi_baseline.npz"), allow_pickle=True)
    fe = np.load(os.path.join(ROOT, "scratch", "mpi_fepsp.npz"), allow_pickle=True)
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    lay = wc["layer"].astype(str); mt = wc["mtype"].astype(str)
    N = len(lay); is_pc = (mt == "SP_PC")
    st = b["spk_t"].astype(float); sid = b["spk_id"].astype(int); stim = float(b["stim_t"])
    m = sid < N; st = st[m]; sid = sid[m]; t = st - stim

    # y 재배열: 층 순
    order_key = np.array([LAYER_ORDER.index(l) if l in LAYER_ORDER else 99 for l in lay])
    rows = np.lexsort((np.arange(N), order_key)); row_of = np.empty(N, int); row_of[rows] = np.arange(N)
    sorted_layers = lay[rows]; bands = {}
    for L in LAYER_ORDER:
        idx = np.where(sorted_layers == L)[0]
        if len(idx):
            bands[L] = (idx.min(), idx.max())
    y = row_of[sid]; cols = np.array([LAYER_COL.get(lay[g], "#999") for g in sid]); pcm = is_pc[sid]

    fig = plt.figure(figsize=(9, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.2, 1.8], hspace=0.12)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax); axf = fig.add_subplot(gs[2], sharex=ax)

    # (위) raster
    for L, (lo, hi) in bands.items():
        ax.axhspan(lo - .5, hi + .5, color=LAYER_COL[L], alpha=.05, zorder=0)
    ax.scatter(t[pcm], y[pcm], s=3, c=cols[pcm], linewidths=0, alpha=.85, zorder=2)
    ax.scatter(t[~pcm], y[~pcm], s=26, c=cols[~pcm], marker="D", edgecolors="k", linewidths=.4, alpha=.95, zorder=3)
    ax.axvline(0, color="crimson", ls="--", lw=1.1)
    for L, (lo, hi) in bands.items():
        ax.text(-1.6, (lo + hi) / 2, L, color=LAYER_COL[L], fontsize=11, fontweight="bold", va="center", ha="right")
    ax.set_ylabel("neuron (by layer)"); ax.set_ylim(-.5, N + .5)
    ax.set_title(f"Ex1 volley — raster + PSTH + fEPSP (전체망 {N}세포 · R=150)\n"
                 f"발화 {int(b['fired'].sum())}/{N} · PC=점, 억제=마름모", fontsize=11)
    ax.tick_params(labelbottom=False)
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=LAYER_COL[L], markersize=8, label=L) for L in LAYER_ORDER]
    ax.legend(handles=leg, loc='upper right', fontsize=8, ncol=4)

    # (중) PSTH 층별
    bw = 0.5; edges = np.arange(-2, XMAX + bw, bw)
    for L in LAYER_ORDER:
        sel = np.array([lay[g] == L for g in sid]); nL = int(np.sum(lay == L))
        if sel.sum() == 0 or nL == 0:
            continue
        hc, _ = np.histogram(t[sel], bins=edges)
        axr.plot(edges[:-1] + bw / 2, hc / (nL * bw / 1000.0), color=LAYER_COL[L], lw=1.4, label=L)
    axr.axvline(0, color="crimson", ls="--", lw=1.0)
    axr.set_ylabel("rate [Hz]"); axr.tick_params(labelbottom=False)

    # (아래) fEPSP 3전극
    tf = fe["t"].astype(float) - stim; V = fe["V"]; enames = [str(x) for x in fe["enames"]]
    base_mask = tf < 0
    for i, nm in enumerate(enames):
        eid = nm.split("(")[0]; base = V[i][base_mask].mean() if base_mask.any() else V[i][0]
        axf.plot(tf, (V[i] - base) * 1000.0, color=ECOL.get(eid, "#888"), lw=1.8, label=nm)  # µV
    axf.axvline(0, color="crimson", ls="--", lw=1.0); axf.axhline(0, color="#bbb", lw=.6)
    axf.set_xlabel("time from stim [ms]"); axf.set_ylabel("fEPSP [µV]")
    axf.set_xlim(-2, XMAX); axf.legend(fontsize=8, ncol=3, loc='lower right')

    outd = os.path.join(ROOT, "04_experiments", "Ex1_baseline", "figures"); os.makedirs(outd, exist_ok=True)
    p = os.path.join(outd, "ex1_fepsp_3panel.png"); fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"[3panel] 저장 {p}", flush=True)
    # fEPSP 피크 요약
    for i, nm in enumerate(enames):
        eid = nm.split("(")[0]; base = V[i][base_mask].mean() if base_mask.any() else V[i][0]
        seg = (V[i] - base) * 1000.0; mm = tf >= 0; k = int(np.argmax(np.abs(seg[mm])))
        print(f"   {nm}: 피크 {seg[mm][k]:+.2f} µV @ {tf[mm][k]:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
