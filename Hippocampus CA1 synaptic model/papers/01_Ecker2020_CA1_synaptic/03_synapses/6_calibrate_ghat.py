"""
6_calibrate_ghat.py — 단계 6: §2.6 paired recording + ĝ 보정
[8개 파라미터 추적] 이 단계가 찾음: ĝ (peak conductance)  (1/8)
============================================================================
Source: Ecker et al. (2020) §2.6, Eq.(11); 보정 §2.7.

흐름:
  1) 파라미터에 보정 적용(온도 Q10, 칼슘 Hill) — common/corrections.py
  2) 실제 PC 세포(02_models)에 AMPA 시냅스 N개 배치
  3) pre 발화 1번 → 소마 EPSP(PSP) 평균 측정(여러 trial)
  4) 식(11)로 g_hat 반복 보정 → 모델 PSP ≈ 실험 목표 PSP 수렴
     g_hat ← g_hat · PSP_exp(1 − PSP_model/df) / (PSP_model(1 − PSP_exp/df)),  df = |E_rev − V_SS|

실행:
    conda activate ca1sim
    python SourceCode/03_paired_calibration/s1_calibrate_ghat.py
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

from common.nrn_env import h, MODELS_DIR            # noqa: E402
from common.cell_loader import load_cell           # noqa: E402
from common.plotstyle import set_korean_font       # noqa: E402
from common.corrections import q10_scale, hill_ca  # noqa: E402
from params_table3 import CLASSES                  # noqa: E402

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(THIS, "figures")

V_HOLD = -70.0
E_REV = 0.0                 # mod 의 AMPA/NMDA reversal (e=0)
PSP_EXP = 0.5               # 실험 목표 PSP 진폭(mV) — 대표값
N_SYN = 5                   # 연결당 시냅스 수
T_SPIKE = 50.0
TSTOP = 90.0


def place_synapses(cell, p, n_syn):
    """PC 첨단수상돌기(apical)에 AMPA 시냅스 n개 배치."""
    apics = list(cell.apic)
    idxs = np.linspace(len(apics) * 0.2, len(apics) * 0.8, n_syn).astype(int)
    syns, ncs, keep = [], [], []
    for j, i in enumerate(idxs):
        sec = apics[int(i)]
        syn = h.ProbAMPANMDA_EMS(sec(0.5))
        syn.tau_r_AMPA = p["tau_r_AMPA"]; syn.tau_d_AMPA = p["tau_d_AMPA"]
        syn.NMDA_ratio = p["NMDA_ratio"]
        syn.Use, syn.Dep, syn.Fac, syn.Nrrp = p["Use"], p["Dep"], p["Fac"], int(p["Nrrp"])
        vs = h.VecStim(); tv = h.Vector([T_SPIKE]); vs.play(tv)
        nc = h.NetCon(vs, syn); nc.delay = 0.0
        syns.append(syn); ncs.append(nc); keep += [vs, tv]
    return syns, ncs, keep


def measure_psp(syns, ncs, vsoma, tvec, g_nS, n_trials=6):
    """g_hat 에서 소마 EPSP 평균(mV) 측정 (trial 평균)."""
    for nc in ncs:
        nc.weight[0] = g_nS
    psps = []
    for k in range(n_trials):
        for j, syn in enumerate(syns):
            syn.setRNG(7, k + 1, j + 1)
        h.finitialize(V_HOLD)
        h.continuerun(TSTOP)
        t, v = np.array(tvec), np.array(vsoma)
        i0 = np.searchsorted(t, T_SPIKE - 1.0)
        base = v[i0]
        peak = v[i0:].max()
        psps.append(peak - base)
    return float(np.mean(psps)), (np.array(tvec).copy(), np.array(vsoma).copy())


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- 1) 보정 적용 (스텝 5) : 원자료가 25°C 기록이라 가정 → 34°C 환산 ---
    p = dict(CLASSES["PC->PC (E2)"])
    tau_raw = p["tau_d_AMPA"]
    p["tau_d_AMPA"] = q10_scale(tau_raw, Q10=2.2, T_exp=25.0, T_sim=34.0)
    use_ref = hill_ca(2.0, p["Use"] / hill_ca(2.0, 1.0, 2.79), 2.79)  # 2mM 기준 유지(예시)
    print(f"[보정] τ_decay {tau_raw:.2f}→{p['tau_d_AMPA']:.2f} ms (Q10), "
          f"Use(2mM 기준)={p['Use']:.2f}")

    # --- 2) 실제 PC 로드 + 시냅스 배치 ---
    pc_root = os.path.join(MODELS_DIR, "pyramidal")
    pc_dir = os.path.join(pc_root, sorted(os.listdir(pc_root))[0])
    cell, tname = load_cell(pc_dir)
    syns, ncs, _keep = place_synapses(cell, p, N_SYN)
    vsoma = h.Vector().record(cell.soma[0](0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0
    # EMS mod 는 고정 dt 지수업데이트(A_*_step)라 cvode 비호환 → 고정 dt 사용.
    # 역치하 EPSP 라 dt=0.1ms 로 키워 가속(정확도 충분).
    h.dt = 0.1

    # --- 3) 식(11) 반복 보정 ---
    df = abs(E_REV - V_HOLD)
    g = p["g_nS"]                      # 초기 g_hat
    g0 = g
    hist_g, hist_psp = [], []
    trace_first, trace_last = None, None
    for it in range(4):
        psp, trace = measure_psp(syns, ncs, vsoma, tvec, g)
        hist_g.append(g); hist_psp.append(psp)
        if it == 0:
            trace_first = trace
        trace_last = trace
        # Eq.(11)
        g = g * PSP_EXP * (1 - psp / df) / (psp * (1 - PSP_EXP / df))
        print(f"  iter {it}: g_hat={hist_g[-1]:.3f} nS → PSP_model={psp:.3f} mV (target {PSP_EXP})")

    g_final = hist_g[-1]

    # --- 4) 그림 ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("스텝 6 — §2.6 식(11): paired recording 으로 g_hat 보정 / peak conductance calibration",
                 fontsize=12, fontweight="bold")
    it = np.arange(len(hist_psp))
    axA.axhline(PSP_EXP, color="tab:red", ls="--", lw=1.5, label="실험 목표 PSP_exp")
    axA.plot(it, hist_psp, "o-", color="tab:blue", lw=2, label="모델 PSP_model")
    axA.set_title("(A) 식(11) 반복 → PSP 수렴 / convergence", fontsize=10)
    axA.set_xlabel("반복 iteration"); axA.set_ylabel("PSP 진폭 (mV)")
    axA.legend(fontsize=9)
    ax2 = axA.twinx()
    ax2.plot(it, hist_g, "s:", color="tab:green", lw=1.5, label="g_hat (nS)")
    ax2.set_ylabel("g_hat (nS)", color="tab:green")

    tf, vf = trace_first; tl, vl = trace_last
    b_f = vf[np.searchsorted(tf, T_SPIKE - 1)]; b_l = vl[np.searchsorted(tl, T_SPIKE - 1)]
    axB.plot(tf, vf - b_f, color="0.6", lw=1.8, label=f"초기 g_hat={g0:.2f} nS")
    axB.plot(tl, vl - b_l, color="tab:purple", lw=2, label=f"보정 g_hat={g_final:.2f} nS")
    axB.axhline(PSP_EXP, color="tab:red", ls="--", lw=1, label="목표 0.5 mV")
    axB.set_xlim(T_SPIKE - 5, T_SPIKE + 50)
    axB.set_title("(B) 소마 EPSP: 보정 전 vs 후 / somatic EPSP", fontsize=10)
    axB.set_xlabel("시간 t (ms)"); axB.set_ylabel("PSP (mV, 기저선 대비)")
    axB.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "6_calibrate_ghat.png")
    plt.savefig(out, dpi=120)
    print(f"[그림] {out}")
    print(f"[검증] 초기 g_hat={g0:.2f} nS (PSP {hist_psp[0]:.3f}) → 보정 g_hat={g_final:.3f} nS "
          f"(PSP {hist_psp[-1]:.3f} ≈ 목표 {PSP_EXP} mV)")


if __name__ == "__main__":
    main()
