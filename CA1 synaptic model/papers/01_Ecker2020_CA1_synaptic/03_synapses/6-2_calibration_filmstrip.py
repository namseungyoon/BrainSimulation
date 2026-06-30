"""
s2b_calibration_filmstrip.py — 스텝 6 시각화(정적 PNG): ĝ 보정 과정 한 장에
============================================================================
GIF 가 안 보이는 환경을 위해, ĝ 를 키울 때 EPSP 가 커지는 모습과 PSP-vs-ĝ 곡선을
한 장 PNG 로 보여준다.
실행: python SourceCode/03_paired_calibration/s2b_calibration_filmstrip.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    p["tau_d_AMPA"] = q10_scale(p["tau_d_AMPA"], 2.2, 25.0, 34.0)

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
            h.finitialize(V_HOLD); h.continuerun(TSTOP)
            vs.append(np.array(vsoma).copy())
        t = np.array(tvec); mv = np.mean(vs, axis=0)
        i0 = np.searchsorted(t, T_SPIKE - 1.0)
        return t, mv - mv[i0], float(mv[i0:].max() - mv[i0])

    gs = np.linspace(0.2, 1.4, 9)
    print("[계산] ĝ 스윕 ...")
    traces, peaks = [], []
    for g in gs:
        t, dv, pk = mean_trace(g)
        traces.append((t, dv)); peaks.append(pk)
    peaks = np.array(peaks)
    g_cal = float(np.interp(PSP_EXP, peaks, gs))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("스텝 6 — g_hat 보정 과정 (정적) / peak-conductance calibration",
                 fontsize=12, fontweight="bold")
    cmap = plt.cm.viridis
    for i, ((t, dv), g) in enumerate(zip(traces, gs)):
        axL.plot(t, dv, color=cmap(i / (len(gs) - 1)), lw=1.8,
                 label=f"g_hat={g:.2f} nS" if i % 2 == 0 else None)
    axL.axhline(PSP_EXP, color="tab:red", ls="--", lw=1.4, label="목표 0.5 mV")
    axL.set_xlim(T_SPIKE - 5, T_SPIKE + 40); axL.set_ylim(-0.1, 0.8)
    axL.set_title("(A) g_hat ↑ → EPSP 봉우리 ↑ / EPSP grows with g_hat", fontsize=10)
    axL.set_xlabel("시간 t (ms)"); axL.set_ylabel("PSP (mV)")
    axL.legend(fontsize=8, loc="upper right")

    axR.plot(gs, peaks, "o-", color="tab:blue", lw=2, ms=5)
    axR.axhline(PSP_EXP, color="tab:red", ls="--", lw=1.4, label="실험 목표 0.5 mV")
    axR.axvline(g_cal, color="tab:green", ls=":", lw=1.5)
    axR.plot([g_cal], [PSP_EXP], "*", color="tab:green", ms=20,
             label=f"보정값 g_hat* ≈ {g_cal:.2f} nS")
    axR.set_title("(B) PSP vs g_hat → 목표 만나는 점 = 보정값", fontsize=10)
    axR.set_xlabel("peak conductance g_hat (nS)"); axR.set_ylabel("PSP 진폭 (mV)")
    axR.legend(fontsize=9, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "6-2_calibration_filmstrip.png")
    plt.savefig(out, dpi=120)
    print(f"[그림] {out}")
    print(f"[보정] PSP=0.5mV 가 되는 g_hat ≈ {g_cal:.3f} nS")


if __name__ == "__main__":
    main()
