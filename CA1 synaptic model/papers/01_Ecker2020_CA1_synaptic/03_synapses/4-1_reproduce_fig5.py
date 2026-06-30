"""
reproduce_fig5.py — 시냅스 모델 검증: 9개 클래스 STP 재현 (Ecker 2020 Fig.5)
============================================================================
Source: Ecker et al. (2020) Fig.5, §3.4.
완성된 통합 시냅스 모델(전도도+TM+확률방출, 단계3~5 + Table3 파라미터)이
**9개 연결 클래스에서 논문대로의 STP 다양성**(억압/촉진/유사선형)을 내는지 검증한다.

프로토콜: 8발 @20Hz + 회복 스파이크, N_TRIALS 회 확률 시행 → 평균 PSP.
검증: 각 클래스의 후기/초기 응답비로 방향을 판정하고 기대값(E1/I1=촉진, E2/I2=억압,
      I3=유사선형)과 PASS/FAIL 비교 → 9개 중 몇 개 일치하는지.

실행:
    conda activate ca1sim
    python papers/01_Ecker2020_CA1_synaptic/03_synapses/reproduce_fig5.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.plotstyle import set_korean_font   # noqa: E402
import synapse_pair as sp                       # noqa: E402
from params_table3 import CLASSES               # noqa: E402

set_korean_font()

N_TRIALS = 35
N_PULSES = 8
FREQ_HZ = 20.0
T_START = 100.0
RECOVERY_DELAY = 500.0
V_HOLD = -65.0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# 기대 STP 방향(한글) + 판정 규칙(후기/초기 응답비 r) — 논문 분류
EXPECT = {
    "E1": ("촉진",   lambda r: r > 1.1),
    "I1": ("촉진",   lambda r: r > 1.1),
    "E2": ("억압",   lambda r: r < 0.9),
    "I2": ("억압",   lambda r: r < 0.9),
    "I3": ("유사선형", lambda r: 0.3 <= r <= 1.3),
}


def stable_seed(name):
    """클래스명 → 재현성 있는 정수 시드(파이썬 hash 솔팅 회피)."""
    return (sum(ord(c) for c in name) % 9000) + 1


def pulse_amplitudes(t, v_mean, spikes, ei, win=None):
    """평균 트레이스에서 펄스별 PSP 진폭(흥분 +, 억제 +크기) 추출."""
    if win is None:
        win = 0.9 * (1000.0 / FREQ_HZ)
    amps = []
    for tp in spikes:
        i0 = np.searchsorted(t, tp)
        i1 = np.searchsorted(t, tp + win)
        base = v_mean[max(i0 - 1, 0)]
        seg = v_mean[i0:i1]
        if len(seg) == 0:
            amps.append(0.0); continue
        amps.append(float(seg.max() - base) if ei == "E" else float(base - seg.min()))
    return np.array(amps)


def run_class(name, p):
    cseed = stable_seed(name)
    traces, t_ref = [], None
    for k in range(N_TRIALS):
        t, v, spikes = sp.run_trial(p, seeds=(12345, cseed, k + 1), n_pulses=N_PULSES,
                                    freq_hz=FREQ_HZ, t_start=T_START,
                                    recovery_delay=RECOVERY_DELAY, v_hold=V_HOLD)
        if t_ref is None:
            t_ref = t
        traces.append(v[:len(t_ref)])
    V = np.vstack([tr[:len(t_ref)] for tr in traces])
    v_mean = V.mean(axis=0)
    amps = pulse_amplitudes(t_ref, v_mean, spikes, p["ei"])
    return t_ref, V, v_mean, amps, spikes


def stp_ratio(amps):
    """후기(6~8발 평균)/초기(1~2발 평균) 응답비 — 단일펄스 잡음에 강건."""
    early = np.mean(amps[0:2]) if np.mean(amps[0:2]) > 1e-6 else 1e-6
    late = np.mean(amps[5:8])
    return late / early


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle("시냅스 모델 검증 — 9개 클래스 STP 재현 / Fig.5  "
                 f"({N_PULSES}발 @{FREQ_HZ:.0f}Hz + 회복, {N_TRIALS}시행)\n"
                 "오른쪽축=정규화(1st PSP=1)  |  범례 참고",
                 fontsize=11, fontweight="bold")

    rows = []
    for ax, (name, p) in zip(axes.flat, CLASSES.items()):
        t, V, v_mean, amps, spikes = run_class(name, p)
        color = "tab:red" if p["ei"] == "E" else "tab:blue"
        for tr in V:
            ax.plot(t, tr, color="0.82", lw=0.4, zorder=1)
        ax.plot(t, v_mean, color=color, lw=1.9, zorder=2)        # 확률 포함 평균
        td, vd, _ = sp.run_trial(p, (0, 0, 0), n_pulses=N_PULSES, freq_hz=FREQ_HZ,
                                 t_start=T_START, recovery_delay=RECOVERY_DELAY,
                                 v_hold=V_HOLD, deterministic=True)  # 확률 미포함(이론 평균)
        ax.plot(td, vd, color="black", lw=1.1, ls="--", zorder=3)
        for s in spikes:
            ax.axvline(s, color="0.6", ls=":", lw=0.5, zorder=0)

        r = stp_ratio(amps)
        exp_kor, rule = EXPECT[p["stp"]]
        ok = rule(r)
        meas = "촉진" if r > 1.1 else ("억압" if r < 0.9 else "유사선형")
        verdict = "일치" if ok else "불일치"
        # () = 기대값, [] = 결과
        ax.set_title(f"{name}  후기/초기={r:.2f}\n(기대: {exp_kor})  [결과: {meas} · {verdict}]",
                     fontsize=9, color=("green" if ok else "red"))
        ax.set_xlabel("시간 t (ms)"); ax.set_ylabel("Vm (mV)")
        # 오른쪽 y축: 1번째 PSP 진폭을 1.0 으로 정규화한 값 (affine 눈금)
        i0 = np.searchsorted(t, T_START)
        base = v_mean[max(i0 - 1, 0)]
        amp1 = amps[0] if abs(amps[0]) > 1e-3 else 1e-3
        norm = (lambda vm: (vm - base) / amp1) if p["ei"] == "E" else (lambda vm: (base - vm) / amp1)
        lo, hi = ax.get_ylim()
        ax2 = ax.twinx()
        ax2.set_ylim(norm(lo), norm(hi))
        ax2.set_ylabel("정규화 normalized (1st PSP=1)", fontsize=8, color="0.45")
        ax2.tick_params(labelsize=7, colors="0.45")
        # 첫 패널에만 범례
        if name == list(CLASSES)[0]:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([0], [0], color="0.82", lw=1.2),
                               Line2D([0], [0], color=color, lw=2),
                               Line2D([0], [0], color="black", lw=1.2, ls="--")],
                      labels=["개별 35시행 (확률 포함)", "확률 포함 — 평균",
                              "확률 미포함 (이론 평균)"],
                      fontsize=6.5, loc="upper right")
        rows.append((name, p["stp"], exp_kor, meas, r, ok))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "4-1_reproduce_fig5.png")
    plt.savefig(out, dpi=110)
    print(f"[그림] {out}\n")

    # 검증표
    print(f"{'경로':18s} {'stp':4s} {'후기/초기':>9s}  {'(기대)':7s} {'[결과]':7s} 판정")
    print("-" * 62)
    npass = 0
    for name, stp, exp_kor, meas, r, ok in rows:
        npass += ok
        print(f"{name:18s} {stp:4s} {r:>9.2f}  ({exp_kor:5s}) [{meas:5s}] {'일치' if ok else '불일치'}")
    print(f"\n[검증 결과] 9개 클래스 중 {npass}/9 일치 "
          f"(억압/촉진/유사선형 방향이 논문 Fig.5 와 일치)")


if __name__ == "__main__":
    main()
