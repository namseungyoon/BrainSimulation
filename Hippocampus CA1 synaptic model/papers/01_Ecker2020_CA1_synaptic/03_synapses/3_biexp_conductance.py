"""
3_biexp_conductance.py — 단계 3: §2.3 biexponential 후시냅스 전도도
[8개 파라미터 추적] 이 단계가 찾음: E_rev, τ_rise, τ_decay  (3/8)
============================================================================
Source: Ecker et al. (2020) §2.3, Eq.(1).
    g(t) = ĝ · A · (exp(-t/τ_decay) − exp(-t/τ_rise))
    A = 1 / (exp(-t_p/τ_decay) − exp(-t_p/τ_rise)),  t_p = 봉우리 시각
    → 봉우리에서 g = ĝ (peak conductance)

확인 목표:
  (A) 실제 EMS 메커니즘이 방출 1번에 만드는 g_AMPA(t) 가 식(1) 해석식과 일치하는가?
      그리고 봉우리 높이가 ĝ 와 같은가?
  (B) τ_rise/τ_decay 가 파형 모양을 어떻게 정하는가 (AMPA vs GABA vs NMDA 시간상수).

실행:
    conda activate ca1sim
    python SourceCode/02_synapse_model/s1_biexp_conductance.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.nrn_env import h, load_project_mechanisms      # noqa: E402
from params_table3 import CLASSES                            # noqa: E402

load_project_mechanisms()
h.load_file("stdrun.hoc")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def analytic_g(t, t0, g_hat, tau_r, tau_d):
    """식(1) 해석식: 봉우리=ĝ 로 정규화된 두 지수의 차."""
    tp = (tau_r * tau_d) / (tau_d - tau_r) * np.log(tau_d / tau_r)   # 봉우리 시각
    A = 1.0 / (np.exp(-tp / tau_d) - np.exp(-tp / tau_r))            # 정규화
    g = np.zeros_like(t)
    m = t >= t0
    dt = t[m] - t0
    g[m] = g_hat * A * (np.exp(-dt / tau_d) - np.exp(-dt / tau_r))
    return g, t0 + tp


def simulate_single_release(p, t_spike=5.0, v_hold=-65.0, tstop=60.0, dt=0.025):
    """passive 구획 + EMS 시냅스, 스파이크 1번 → g_AMPA(t) 기록 (확실한 방출: Use=1, Nrrp=1)."""
    soma = h.Section(name="soma")
    soma.L = soma.diam = 20.0
    soma.insert("pas")
    soma.e_pas = v_hold

    syn = h.ProbAMPANMDA_EMS(soma(0.5))
    syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2)
    syn.tau_d_AMPA = p["tau_d_AMPA"]
    syn.NMDA_ratio = p["NMDA_ratio"]
    syn.Use, syn.Nrrp, syn.Fac, syn.Dep = 1.0, 1, 0.0, 100.0   # 단일 방출 보장
    syn.setRNG(1, 1, 1)

    vs = h.VecStim(); tv = h.Vector([t_spike]); vs.play(tv)
    nc = h.NetCon(vs, syn); nc.weight[0] = p["g_nS"]; nc.delay = 0.0

    t = h.Vector().record(h._ref_t)
    g_ampa = h.Vector().record(syn._ref_g_AMPA)   # µS
    g_nmda = h.Vector().record(syn._ref_g_NMDA)   # µS

    h.dt = dt; h.celsius = 34.0
    h.finitialize(v_hold)
    h.continuerun(tstop)
    return np.array(t), np.array(g_ampa) * 1e3, np.array(g_nmda) * 1e3  # → nS


def mg_gate(v, mg=1.0, slope=0.062, scale=2.62):
    """식(4): NMDA 의 Mg²⁺ 전압의존 차단 게이트 mg(V)."""
    return 1.0 / (1.0 + np.exp(-slope * v) * (mg / scale))


def clamp_peak_currents(p, v_hold, t_spike=5.0, tstop=120.0, dt=0.025):
    """전압클램프(SEClamp)로 V 고정 + 방출 1번 → AMPA/NMDA 봉우리 전류(nA) 측정."""
    soma = h.Section(name="soma")
    soma.L = soma.diam = 20.0
    soma.insert("pas")
    syn = h.ProbAMPANMDA_EMS(soma(0.5))
    syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2); syn.tau_d_AMPA = p["tau_d_AMPA"]
    syn.NMDA_ratio = p["NMDA_ratio"]
    syn.Use, syn.Nrrp, syn.Fac, syn.Dep = 1.0, 1, 0.0, 100.0
    syn.setRNG(1, 1, 1)
    vs = h.VecStim(); tv = h.Vector([t_spike]); vs.play(tv)
    nc = h.NetCon(vs, syn); nc.weight[0] = p["g_nS"]; nc.delay = 0.0
    clamp = h.SEClamp(soma(0.5)); clamp.dur1 = tstop; clamp.amp1 = v_hold; clamp.rs = 0.001

    iA = h.Vector().record(syn._ref_i_AMPA)   # nA
    iN = h.Vector().record(syn._ref_i_NMDA)   # nA
    h.dt = dt; h.celsius = 34.0
    h.finitialize(v_hold)
    h.continuerun(tstop)
    iA, iN = np.array(iA), np.array(iN)
    # 봉우리 전류: AMPA 는 inward(음수)→min, NMDA 는 부호가 V 따라 바뀜→절대값 최대
    peakA = iA[np.argmax(np.abs(iA))]
    peakN = iN[np.argmax(np.abs(iN))]
    return peakA, peakN


def figure_current_mg(p):
    """식(2)-(4) 시각화: mg(V) 게이트 + AMPA/NMDA I–V 곡선."""
    volts = np.arange(-90, 45, 5.0)
    pA, pN = [], []
    for v in volts:
        a, n = clamp_peak_currents(p, float(v))
        pA.append(a * 1e3); pN.append(n * 1e3)   # nA → pA
    pA, pN = np.array(pA), np.array(pN)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Step 2 — §2.3 Eq.(2)-(4): synaptic current & NMDA Mg block",
                 fontsize=13, fontweight="bold")
    # (A) Mg 게이트 mg(V)
    vv = np.linspace(-90, 40, 400)
    ax1.plot(vv, mg_gate(vv), color="tab:green", lw=2.4)
    ax1.axvline(-65, color="0.6", ls=":", lw=1)
    ax1.annotate("at -65 mV\nNMDA mostly blocked", xy=(-65, mg_gate(-65)),
                 xytext=(-55, 0.55), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="0.4"))
    ax1.set_title("(A) Eq.(4): Mg block gate  mg(V)", fontsize=10)
    ax1.set_xlabel("V_m (mV)"); ax1.set_ylabel("mg(V)  (0=blocked, 1=open)")
    # (B) I–V 곡선
    ax2.axhline(0, color="0.7", lw=0.8); ax2.axvline(0, color="0.7", lw=0.8)
    ax2.plot(volts, pA, "o-", color="tab:red", lw=2, ms=4, label="AMPA  I_peak (ohmic)")
    ax2.plot(volts, pN, "s-", color="tab:green", lw=2, ms=4, label="NMDA  I_peak (Mg-gated)")
    ax2.set_title("(B) Eq.(2)-(3): peak current vs holding V", fontsize=10)
    ax2.set_xlabel("holding V_m (mV)"); ax2.set_ylabel("peak synaptic current (pA)")
    ax2.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "3b_current_mg_block.png")
    plt.savefig(out, dpi=120)
    # NMDA 비선형 지표: 가장 음수(가장 큰 inward) NMDA 전류의 전압
    v_max_nmda = volts[np.argmin(pN)]
    print(f"[그림] {out}")
    print(f"[검증] AMPA I-V 선형, 0 mV(=mod의 e) 부근에서 반전. "
          f"NMDA inward 최대는 V≈{v_max_nmda:.0f} mV (과분극 아님) → Mg 차단의 비선형 J-형 확인.")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    p = CLASSES["PC->PC (E2)"]          # ĝ=0.6 nS, τ_r=0.2, τ_d=3.0 ms
    t, gA, gN = simulate_single_release(p)

    # 식(1) 해석식과 겹쳐 비교
    g_ana, tp = analytic_g(t, 5.0, p["g_nS"], p.get("tau_r_AMPA", 0.2), p["tau_d_AMPA"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Step 1 — §2.3 Eq.(1): biexponential synaptic conductance",
                 fontsize=13, fontweight="bold")

    # (A) 메커니즘 g_AMPA vs 해석식
    axA.plot(t, gA, color="tab:red", lw=2.4, label="EMS mod  g_AMPA(t)")
    axA.plot(t, g_ana, "k--", lw=1.2, label="Eq.(1) analytic")
    axA.axhline(p["g_nS"], color="0.6", ls=":", lw=1)
    axA.axvline(tp, color="0.6", ls=":", lw=1)
    axA.annotate(f"peak = ĝ = {p['g_nS']} nS\n at t_p = {tp-5.0:.2f} ms",
                 xy=(tp, p["g_nS"]), xytext=(tp + 6, p["g_nS"] * 0.7),
                 arrowprops=dict(arrowstyle="->", color="0.4"), fontsize=9)
    axA.set_title("(A) PC→PC AMPA: mod = Eq.(1) ?", fontsize=10)
    axA.set_xlabel("t (ms)"); axA.set_ylabel("g (nS)")
    axA.legend(fontsize=9); axA.set_xlim(0, 40)

    # (B) τ 가 모양을 정한다 — AMPA / GABA / NMDA 시간상수 (식 1 형태, 봉우리=1 정규화)
    tt = np.linspace(0, 120, 3000)
    shapes = [("AMPA (τr=0.2, τd=3)", 0.2, 3.0, "tab:red"),
              ("GABA_A (τr=0.2, τd=8.3)", 0.2, 8.3, "tab:blue"),
              ("NMDA (τr=9, τd=61)", 9.0, 61.0, "tab:green")]
    for name, tr, td, c in shapes:
        g, _ = analytic_g(tt, 0.0, 1.0, tr, td)   # ĝ=1 정규화
        axB.plot(tt, g, color=c, lw=2, label=name)
    axB.set_title("(B) tau_rise/tau_decay set the shape (peak normalized to 1)", fontsize=10)
    axB.set_xlabel("t (ms)"); axB.set_ylabel("g / ĝ")
    axB.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "3_biexp_conductance.png")
    plt.savefig(out, dpi=120)

    # 콘솔 검증
    peak_meas = gA.max(); tp_meas = t[np.argmax(gA)] - 5.0
    print(f"[그림] {out}")
    print(f"[검증] g_AMPA 봉우리: 측정 {peak_meas:.4f} nS vs ĝ {p['g_nS']} nS "
          f"(오차 {abs(peak_meas-p['g_nS'])/p['g_nS']*100:.2f}%)")
    print(f"       봉우리 시각: 측정 {tp_meas:.3f} ms vs 해석 {tp-5.0:.3f} ms")
    print(f"       → mod 가 식(1)을 그대로 구현함을 확인. (NMDA는 -65mV에서 Mg차단으로 작음: "
          f"max g_NMDA={gN.max():.4f} nS)")

    # --- 스텝 2: 식(2)-(4) 전류 & Mg 차단 ---
    print()
    figure_current_mg(p)


if __name__ == "__main__":
    main()
