"""
stochastic_vs_deterministic.py — 확률 O vs 확률 X 비교 (9개 클래스)
============================================================================
Source: Ecker et al. (2020) §2.4(결정론 TM) vs §2.5(확률 MVR).
각 연결 클래스에서 동일 프로토콜(8발@20Hz+회복)을 10번씩:
  - 확률 O (stochastic, ProbAMPANMDA_EMS/ProbGABAAB_EMS): 시행마다 다름
  - 확률 X (deterministic, DetAMPANMDA/DetGABAAB): 10번 모두 동일(겹침)
→ "확률을 넣으면 매 시행 다르고, 빼면(결정론) 매번 같다"를 눈으로 비교.

실행:
    conda activate ca1sim
    python papers/01_Ecker2020_CA1_synaptic/03_synapses/stochastic_vs_deterministic.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.nrn_env import h           # noqa: E402
from common.plotstyle import set_korean_font   # noqa: E402
import synapse_pair as sp              # noqa: E402
from params_table3 import CLASSES      # noqa: E402

set_korean_font()

N_REPEAT = 35   # 논문 Fig.5 프로토콜(Kohus 2016) = 35 반복
N_PULSES = 8
FREQ_HZ = 20.0
T_START = 100.0
RECOVERY = 500.0
V_HOLD = -65.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def build_syn(seg, p, deterministic, seed):
    """확률(EMS) 또는 결정론(Det) 시냅스 생성. 같은 TM 파라미터(Use/Dep/Fac)."""
    if p["receptor"] == "AMPANMDA":
        syn = h.DetAMPANMDA(seg) if deterministic else h.ProbAMPANMDA_EMS(seg)
        syn.tau_r_AMPA = p.get("tau_r_AMPA", 0.2); syn.tau_d_AMPA = p["tau_d_AMPA"]
        syn.NMDA_ratio = p["NMDA_ratio"]
    else:
        syn = h.DetGABAAB(seg) if deterministic else h.ProbGABAAB_EMS(seg)
        syn.tau_r_GABAA = p.get("tau_r_GABAA", 0.2); syn.tau_d_GABAA = p["tau_d_GABAA"]
        if hasattr(syn, "GABAB_ratio"):
            syn.GABAB_ratio = 0.0
    syn.Use, syn.Dep, syn.Fac = p["Use"], p["Dep"], p["Fac"]
    if not deterministic:
        syn.Nrrp = int(p["Nrrp"])
        syn.setRNG(7, seed, 3)           # 시행마다 다른 시드 → 확률적 변동
    return syn


def run(p, deterministic, seed, dt=0.025):
    post = sp.make_passive_post(e_pas=V_HOLD)
    syn = build_syn(post(0.5), p, deterministic, seed)
    spikes = sp.spike_train(N_PULSES, FREQ_HZ, T_START, RECOVERY)
    nc, vs, tv = sp.make_netcon(syn, p["g_nS"], spikes)
    t = h.Vector().record(h._ref_t)
    v = h.Vector().record(post(0.5)._ref_v)
    h.dt = dt; h.celsius = 34.0
    h.finitialize(V_HOLD)
    h.continuerun(spikes[-1] + 150.0)
    return np.array(t), np.array(v)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle("확률 포함 vs 확률 미포함 — 각 35시행 / stochastic vs deterministic (35 trials each)",
                 fontsize=13, fontweight="bold")

    for ax, (name, p) in zip(axes.flat, CLASSES.items()):
        # 확률 미포함 (결정론) — 35번 (모두 동일하게 겹침)
        for k in range(N_REPEAT):
            t, v = run(p, deterministic=True, seed=k + 1)
            ax.plot(t, v, color="black", lw=1.6, alpha=0.9, zorder=3,
                    label="확률 미포함 (35시행, 겹침)" if k == 0 else None)
        # 확률 포함 (stochastic) — 35번 (시행마다 다름)
        col = "tab:red" if p["ei"] == "E" else "tab:blue"
        for k in range(N_REPEAT):
            t, v = run(p, deterministic=False, seed=k + 1)
            ax.plot(t, v, color=col, lw=0.7, alpha=0.5, zorder=2,
                    label="확률 포함 (35시행, 제각각)" if k == 0 else None)
        ax.set_title(f"{name}  [{p['stp']}]", fontsize=9)
        ax.set_xlabel("시간 t (ms)"); ax.set_ylabel("Vm (mV)")
        if name == list(CLASSES)[0]:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "5-1_stochastic_vs_deterministic.png")
    plt.savefig(out, dpi=110)
    print(f"[그림] {out}")
    print("확률 미포함(검정): 35시행이 한 줄로 겹침 / 확률 포함(색): 35시행이 제각각 퍼짐")


if __name__ == "__main__":
    main()
