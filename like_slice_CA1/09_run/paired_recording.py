# -*- coding: utf-8 -*-
"""
09_run/paired_recording.py  —  대표 시뮬레이션: in silico paired recording (Romani S18 / Ecker Fig.5 방식)

단일 연결(대표: PC→PC)에 9클래스 EMS 시냅스를 놓고, 전세포 스파이크열(8발 20Hz + 회복)을
주어 후세포 PSP 파형을 측정. 확률 EMS 이므로 여러 시행 → 평균 PSP + 시행간 변동(CV) + STP.

- 흥분(PC→PC E2, 억압) 대표 + 대조로 PC→SOM+ (E1, 촉진) · PV+→PC (I2, 억제) 3종.
실행: python 09_run/paired_recording.py
"""
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import params_table3 as P3                        # noqa: E402
from synapse_pair import run_trial, spike_train    # noqa: E402
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
N_TRIALS = 25
CASES = ["PC->PC (E2)", "PC->SOM+ (E1)", "PV+->PC (I2)"]


def pulse_amps(t, v, pulses, inh):
    """각 펄스 직후 PSP 진폭(기저 대비). inh면 음의 편향."""
    amps = []
    for tp in pulses:
        base = v[(t >= tp - 2) & (t < tp)].mean()
        win = v[(t >= tp) & (t < tp + 20)]
        pk = win.min() if inh else win.max()
        amps.append(abs(pk - base))
    return np.array(amps)


def main():
    pulses = spike_train(8, 20.0, 100.0, 500.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, cls in zip(axes, CASES):
        p = P3.CLASSES[cls]; inh = p["ei"] == "I"
        traces = []
        for tr in range(N_TRIALS):
            t, v, _ = run_trial(p, seeds=(tr + 1, tr + 7, 3), deterministic=False)
            traces.append(v)
        L = min(len(x) for x in traces)
        V = np.stack([x[:L] for x in traces]); t = t[:L]
        mean = V.mean(0)
        for v in V:
            ax.plot(t, v, color="#bbb", lw=0.4, alpha=0.5)
        ax.plot(t, mean, color=("#4C72B0" if inh else "#DD8452"), lw=1.8, label="평균")
        for tp in pulses:
            ax.axvline(tp, color="k", ls=":", lw=0.4, alpha=0.4)
        amps = pulse_amps(t, mean, pulses, inh)
        ratio = amps[7] / amps[0] if amps[0] > 0 else 0     # 8번째/1번째 (STP)
        # 1번 펄스 진폭들로 CV
        a1 = pulse_amps(t, V.mean(0), [pulses[0]], inh)  # placeholder
        a1_trials = np.array([pulse_amps(t, V[i], [pulses[0]], inh)[0] for i in range(len(V))])
        cv = a1_trials.std() / a1_trials.mean() if a1_trials.mean() > 0 else 0
        ax.set_title(f"{cls}\n{'IPSP' if inh else 'EPSP'} · STP비(8/1)={ratio:.2f} · CV={cv:.2f}",
                     fontsize=10)
        ax.set_xlabel("시간 (ms)"); ax.set_ylabel("후세포 막전위 (mV)")
        ax.legend(fontsize=8)
        print(f"[{cls}] 1번PSP={a1_trials.mean():.3f}mV STP비(8/1)={ratio:.2f} CV={cv:.2f}")
    fig.suptitle("V5b  대표 in silico paired recording — 단일 연결 PSP·단기가소성·시행간변동(CV)\n"
                 f"8발 20Hz + 회복, {N_TRIALS}시행(회색)+평균(색) · 확률 EMS 시냅스",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT, "V5b_paired_recording.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
