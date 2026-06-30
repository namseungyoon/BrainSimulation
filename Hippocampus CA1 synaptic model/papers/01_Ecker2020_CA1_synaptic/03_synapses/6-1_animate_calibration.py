"""
s2_animate_calibration.py — 스텝 6 시각화: g_hat 보정 과정을 애니메이션(GIF)으로
============================================================================
Source: Ecker et al. (2020) §2.6 식(11).
g_hat(peak conductance)를 0.2→1.4 nS 로 키우며:
  (왼쪽) 실제 PC 소마 EPSP 파형이 커지는 모습,
  (오른쪽) "PSP vs g_hat" 곡선이 그려지며 실험 목표선(0.5mV)을 만나는 지점 = 보정된 g_hat.
를 프레임별로 보여준다(GIF).

실행:
    conda activate ca1sim
    python SourceCode/03_paired_calibration/s2_animate_calibration.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
SOURCECODE = os.path.dirname(THIS)
ROOT = os.path.dirname(SOURCECODE)
sys.path.insert(0, SOURCECODE)
sys.path.insert(0, os.path.join(SOURCECODE, "03_synapses"))

from common.nrn_env import h                       # noqa: E402
from common.cell_loader import load_cell           # noqa: E402
from common.plotstyle import set_korean_font       # noqa: E402
from common.corrections import q10_scale           # noqa: E402
from params_table3 import CLASSES                  # noqa: E402
from paired_recording import place_synapses, load_pc, V_HOLD, T_SPIKE, TSTOP, N_SYN  # noqa: E402

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(THIS, "figures")
PSP_EXP = 0.5


def main():
    os.makedirs(OUT, exist_ok=True)
    p = dict(CLASSES["PC->PC (E2)"])
    p["tau_d_AMPA"] = q10_scale(p["tau_d_AMPA"], 2.2, 25.0, 34.0)   # 온도 보정

    cell, _ = load_pc()
    syns, ncs, _keep = place_synapses(cell, p, N_SYN)
    vsoma = h.Vector().record(cell.soma[0](0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0
    h.dt = 0.1

    def mean_trace(g, n=4):
        for nc in ncs:
            nc.weight[0] = g
        vs = []
        for k in range(n):
            for j, syn in enumerate(syns):
                syn.setRNG(7, k + 1, j + 1)
            h.finitialize(V_HOLD)
            h.continuerun(TSTOP)
            vs.append(np.array(vsoma).copy())
        t = np.array(tvec)
        mv = np.mean(vs, axis=0)
        i0 = np.searchsorted(t, T_SPIKE - 1.0)
        return t, mv - mv[i0], float(mv[i0:].max() - mv[i0])

    # --- g_hat 스윕: 프레임 데이터 사전계산 ---
    print("[계산] g_hat 스윕 시뮬 중 ...")
    gs = np.linspace(0.2, 1.4, 18)
    traces, peaks = [], []
    for g in gs:
        t, dv, pk = mean_trace(g)
        traces.append((t, dv)); peaks.append(pk)
    peaks = np.array(peaks)
    # 목표를 만나는 보정 g_hat (선형 보간)
    g_cal = float(np.interp(PSP_EXP, peaks, gs))
    print(f"[보정] PSP=0.5mV 가 되는 g_hat ≈ {g_cal:.3f} nS")

    # --- 애니메이션 ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("스텝 6 — g_hat 보정 과정 / watching peak-conductance calibration (Eq.11)",
                 fontsize=12, fontweight="bold")
    tt = traces[0][0]
    lineL, = axL.plot([], [], color="tab:purple", lw=2)
    axL.axhline(PSP_EXP, color="tab:red", ls="--", lw=1.2, label="실험 목표 0.5 mV")
    axL.set_xlim(T_SPIKE - 5, T_SPIKE + 40); axL.set_ylim(-0.1, 0.75)
    axL.set_title("(A) 소마 EPSP 파형 / somatic EPSP", fontsize=10)
    axL.set_xlabel("시간 t (ms)"); axL.set_ylabel("PSP (mV)"); axL.legend(fontsize=9, loc="upper right")
    txtL = axL.text(0.04, 0.92, "", transform=axL.transAxes, fontsize=11, va="top")

    axR.axhline(PSP_EXP, color="tab:red", ls="--", lw=1.2, label="목표 0.5 mV")
    axR.plot(gs, peaks, color="0.85", lw=1, zorder=1)
    curveR, = axR.plot([], [], "o-", color="tab:blue", lw=2, ms=4, zorder=2)
    markR, = axR.plot([], [], "*", color="tab:green", ms=18, zorder=3)
    axR.set_xlim(gs[0], gs[-1]); axR.set_ylim(0, max(peaks) * 1.1)
    axR.set_title("(B) PSP vs g_hat → 목표 만나는 점이 보정값", fontsize=10)
    axR.set_xlabel("peak conductance g_hat (nS)"); axR.set_ylabel("PSP 진폭 (mV)")
    axR.legend(fontsize=9, loc="lower right")

    n_hold = 6                      # 마지막에 보정점 강조 프레임
    n_frames = len(gs) + n_hold

    def update(i):
        k = min(i, len(gs) - 1)
        t, dv = traces[k]
        lineL.set_data(t, dv)
        txtL.set_text(f"g_hat = {gs[k]:.2f} nS\nPSP = {peaks[k]:.3f} mV")
        curveR.set_data(gs[:k + 1], peaks[:k + 1])
        if i >= len(gs):            # 보정점 표시
            markR.set_data([g_cal], [PSP_EXP])
            txtL.set_text(f"보정 완료!\ng_hat* = {g_cal:.2f} nS\nPSP = 0.50 mV")
        return lineL, curveR, markR, txtL

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=350, blit=False)
    out = os.path.join(OUT, "6-1_calibration.gif")
    ani.save(out, writer=animation.PillowWriter(fps=3))
    plt.close(fig)
    print(f"[GIF] {out}")


if __name__ == "__main__":
    main()
