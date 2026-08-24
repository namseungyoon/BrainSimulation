# -*- coding: utf-8 -*-
"""Ex1 volley — Brunel식 그림: (위) 50개 예시 추체뉴런 raster · (아래) 전체 뉴런 PSTH.
사용자가 보여준 Brunel raster 스타일. 입력: scratch/mpi_baseline.npz + window_cells.
출력: 04_experiments/Ex1_baseline/figures/ex1_raster50_psth.png
실행: python 03_network/3_run/viz_raster50_psth.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
NSAMP = 50
XMAX = 30.0
CDOT = "#3aa66f"   # raster 점(초록)
CBAR = "#d96ec9"   # PSTH(자홍)


def main():
    b = np.load(os.path.join(ROOT, "scratch", "mpi_baseline.npz"), allow_pickle=True)
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    mt = wc["mtype"].astype(str); N = len(mt); is_pc = (mt == "SP_PC")
    st = b["spk_t"].astype(float); sid = b["spk_id"].astype(int); stim = float(b["stim_t"])
    m = sid < N; st = st[m]; sid = sid[m]; t = st - stim
    totPC = int(is_pc.sum())

    # 50개 예시 추체뉴런: 발화한 PC 중 gid 균등 샘플
    pc_fired = np.array(sorted(set(sid[is_pc[sid]].tolist())))
    if len(pc_fired) >= NSAMP:
        sample = pc_fired[np.linspace(0, len(pc_fired) - 1, NSAMP).astype(int)]
    else:
        sample = pc_fired
    row_of = {int(g): i for i, g in enumerate(sample)}

    fig = plt.figure(figsize=(8, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.3], hspace=0.08)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)

    # (위) 50개 예시 뉴런 raster
    sel = np.array([g in row_of for g in sid])
    ys = np.array([row_of[int(g)] for g in sid[sel]])
    ax.scatter(t[sel], ys, s=10, c=CDOT, marker="o", linewidths=0)
    ax.axvline(0, color="crimson", ls="--", lw=1.0)
    ax.set_ylabel("neuron ID (example)"); ax.set_ylim(-1, len(sample))
    ax.set_title(f"Ex1 volley — {NSAMP} example pyramidal neurons (of {totPC:,}) + all-neuron PSTH\n"
                 f"R=150 um · fired {int(b['fired'].sum())}/{N}", fontsize=11)
    ax.tick_params(labelbottom=False)

    # (아래) 전체 뉴런 PSTH (Hz/뉴런, 0.5ms bin)
    bw = 0.5; edges = np.arange(-2, XMAX + bw, bw)
    hc, _ = np.histogram(t, bins=edges)
    rate = hc / (N * bw / 1000.0)
    axr.bar(edges[:-1], rate, width=bw, align="edge", color=CBAR, linewidth=0)
    axr.axvline(0, color="crimson", ls="--", lw=1.0)
    axr.set_xlabel("time from stim [ms]"); axr.set_ylabel("firing rate\n[Hz]")
    axr.set_xlim(-2, XMAX)

    outd = os.path.join(ROOT, "04_experiments", "Ex1_baseline", "figures"); os.makedirs(outd, exist_ok=True)
    p = os.path.join(outd, "ex1_raster50_psth.png"); fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"[raster50] 저장 {p} · 예시 {len(sample)}뉴런 · PSTH 피크 {rate.max():.0f}Hz", flush=True)


if __name__ == "__main__":
    main()
