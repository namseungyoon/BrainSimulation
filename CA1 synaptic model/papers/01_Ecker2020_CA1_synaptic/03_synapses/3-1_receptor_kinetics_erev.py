"""
3-1_receptor_kinetics_erev.py — 시냅스 모델 설명 그림 2종 (수용체 동역학 + 반전전위)
============================================================================
Source: Ecker et al. (2020) §2.3 식(1)-(4), Table 3. (해석식 기반 — NEURON 불필요)

그림1 (3-1_biexp_3receptors.png): **이중지수 전도도 g(t)=g_hat·A·(e^-t/τd − e^-t/τr)** 를
  수용체 3종(AMPA·NMDA·GABA_A)의 실제 시간상수로 그려 **모양 차이**를 보인다.
그림2 (3-1_erev_receptor.png): **반전전위 E_rev 가 수용체마다 달라(흥분 −8.5 / 억제 −73mV)**
  같은 막전위에서도 전류 방향·크기가 어떻게 달라지는지 — I-V·휴지 PSC·반전 시연.

I_syn(t) = g(t)·(V − E_rev)   [nS·mV = pA].  음수=내향(탈분극), 양수=외향(과분극).
실행: <ca1sim py> .../03_synapses/3-1_receptor_kinetics_erev.py
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
from common.plotstyle import set_korean_font           # noqa: E402
from params_table3 import (CLASSES, E_REV_EXC, E_REV_INH,   # noqa: E402
                           PC_PC_NMDA_TAU_R, PC_PC_NMDA_TAU_D)

set_korean_font()
OUT = os.path.join(THIS, "figures")

_A = CLASSES["PC->PC (E2)"]      # AMPA: τr 0.2, τd 3.0, g_hat 0.6
_G = CLASSES["PV+->PC (I2)"]     # GABA_A: τr 0.2, τd 11.1, g_hat 2.0

# (이름, τr, τd, g_hat[nS], E_rev[mV], 색)
RECEPTORS = [
    ("AMPA",   _A.get("tau_r_AMPA", 0.2), _A["tau_d_AMPA"], _A["g_nS"], E_REV_EXC, "tab:red"),
    ("NMDA",   PC_PC_NMDA_TAU_R, PC_PC_NMDA_TAU_D, _A["g_nS"] * _A["NMDA_ratio"], E_REV_EXC, "tab:green"),
    ("GABA_A", _G.get("tau_r_GABAA", 0.2), _G["tau_d_GABAA"], _G["g_nS"], E_REV_INH, "tab:blue"),
]


def biexp(t, t0, ghat, tr, td):
    """식(1): 봉우리=g_hat 로 정규화된 두 지수의 차."""
    tp = (tr * td) / (td - tr) * np.log(td / tr)
    A = 1.0 / (np.exp(-tp / td) - np.exp(-tp / tr))
    g = np.zeros_like(t)
    m = t >= t0
    dt = t[m] - t0
    g[m] = ghat * A * (np.exp(-dt / td) - np.exp(-dt / tr))
    return g


# ───────────────── 그림1: 이중지수 3수용체 형태 ─────────────────
def fig_kinetics():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle("시냅스 모델 — 이중지수 전도도 g(t) : 수용체 3종의 모양 차이 (식 1)",
                 fontsize=13, fontweight="bold")
    tt = np.linspace(0, 200, 6000)
    for name, tr, td, ghat, _, c in RECEPTORS:
        g = biexp(tt, 0.0, 1.0, tr, td)                 # 봉우리=1 정규화
        lab = f"{name}  (τr={tr:g}, τd={td:g} ms)"
        axL.plot(tt, g, color=c, lw=2.2, label=lab)
        axR.plot(tt, g, color=c, lw=2.2, label=lab)
    axL.set_xlim(0, 25); axL.set_title("(A) 빠른 구간 (0–25ms): AMPA·GABA_A 차이", fontsize=10)
    axR.set_xlim(0, 200); axR.set_title("(B) 전체 (0–200ms): NMDA 긴 꼬리", fontsize=10)
    for ax in (axL, axR):
        ax.set_xlabel("시냅스전 스파이크 이후 시간 (ms)")
        ax.set_ylabel("정규화 전도도 g / g_hat"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    box_txt = "AMPA: 빠른 흥분\nGABA_A: 중간 억제\nNMDA: 매우 느림(τd~149ms)"
    axL.text(0.97, 0.95, box_txt, transform=axL.transAxes, ha="right", va="top",
             fontsize=8.5, bbox=dict(fc="#FFF6D5", ec="0.6", alpha=0.9))
    axR.text(0.45, 0.27, box_txt, transform=axR.transAxes, ha="left", va="top",
             fontsize=8.5, bbox=dict(fc="#FFF6D5", ec="0.6", alpha=0.9))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "3-1_biexp_3receptors.png")
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f"[그림1] {out}", flush=True)


# ───────────────── 그림2: 반전전위 E_rev 효과 ─────────────────
def fig_erev():
    fig, (axIV, axPSC, axRev) = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("시냅스 모델 — 반전전위 E_rev (수용체별): 같은 전압에서도 전류 방향·크기가 다름",
                 fontsize=13, fontweight="bold")
    REST = -70.0
    amp = next(r for r in RECEPTORS if r[0] == "AMPA")
    gab = next(r for r in RECEPTORS if r[0] == "GABA_A")

    # (A) I-V 관계: I_peak = g_hat·(V − E_rev)  → x절편 = E_rev(반전전위)
    V = np.linspace(-100, 20, 240)
    for name, tr, td, ghat, E, c in (amp, gab):
        axIV.plot(V, ghat * (V - E), color=c, lw=2.4, label=f"{name} (g_hat={ghat:g}nS, E_rev={E:g}mV)")
        axIV.plot(E, 0, "o", color=c, ms=10, mec="k", zorder=5)               # E_rev (x절편; 값은 범례)
    axIV.axhline(0, color="0.6", lw=0.8); axIV.axvline(REST, color="0.4", ls="--", lw=1.2)
    axIV.set_xlim(-103, 23); axIV.set_ylim(-72, 205)
    axIV.text(REST - 2, -66, "휴지 -70mV", fontsize=8.2, color="0.3", va="bottom", ha="right")

    # ── 구동력(V-E_rev) 차이: 휴지 -70mV → 각 E_rev 까지의 V축 거리 ──
    for name, tr, td, ghat, E, c in (amp, gab):
        axIV.plot(REST, ghat * (REST - E), "o", color=c, ms=9, mec="k", zorder=6)   # 휴지 동작점
    # 구동력 가로 화살표 + 각 색(수용체)에 값 라벨 직접 표시
    axIV.annotate("", xy=(amp[4], -12), xytext=(REST, -12),
                  arrowprops=dict(arrowstyle="<->", color=amp[5], lw=2.4))
    axIV.text((REST + amp[4]) / 2, -5, f"구동력 {REST - amp[4]:+.1f}mV (큰 내향)",
              color=amp[5], ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    axIV.annotate("", xy=(gab[4], 10), xytext=(REST, 10),
                  arrowprops=dict(arrowstyle="<->", color=gab[5], lw=2.4))
    axIV.text(gab[4] - 3, 12, f"구동력 {REST - gab[4]:+.1f}mV (~0, 션팅)",
              color=gab[5], ha="right", va="bottom", fontsize=8.4, fontweight="bold")
    axIV.set_title("(A) I-V: x절편=E_rev(반전전위) · 휴지에서 구동력 차이\n"
                   "음수=내향(탈분극)·양수=외향", fontsize=10)
    axIV.set_xlabel("막전위 V (mV)"); axIV.set_ylabel("시냅스 전류 I (pA)")
    axIV.legend(fontsize=8.0, loc="upper left"); axIV.grid(alpha=0.3)

    # (B) 휴지(-70mV)에서 PSC(t): AMPA(큰 내향) vs GABA_A(거의 0 = 션팅)
    tt = np.linspace(0, 60, 3000)
    for name, tr, td, ghat, E, c in (amp, gab):
        I = biexp(tt, 2.0, ghat, tr, td) * (REST - E)
        df = REST - E
        axPSC.plot(tt, I, color=c, lw=2.2, label=f"{name}: 구동력 V-E_rev={df:+.1f}mV")
    axPSC.axhline(0, color="0.6", lw=0.8)
    axPSC.set_title("(B) 휴지 -70mV 에서 PSC\nAMPA 큰 내향(흥분) · GABA_A ~0(반전 근처=션팅)", fontsize=10)
    axPSC.set_xlabel("시간 (ms)"); axPSC.set_ylabel("시냅스 전류 I (pA)")
    axPSC.legend(fontsize=8.5); axPSC.grid(alpha=0.3)

    # (C) 반전 시연(GABA_A): 전압 따라 PSC 가 E_rev(−73mV)에서 방향 반전
    name, tr, td, ghat, E, _ = gab
    holds = [-90, -80, -73, -60, -45]
    cmap = plt.get_cmap("coolwarm")
    for vh in holds:
        I = biexp(tt, 2.0, ghat, tr, td) * (vh - E)
        col = cmap((vh + 90) / 45.0)
        axRev.plot(tt, I, color=col, lw=2.0, label=f"V={vh}mV")
    axRev.axhline(0, color="k", lw=1.0)
    axRev.set_title(f"(C) GABA_A 반전 시연 — E_rev={E:g}mV 에서 부호 반전\n(위=외향, 아래=내향)",
                    fontsize=10)
    axRev.set_xlabel("시간 (ms)"); axRev.set_ylabel("시냅스 전류 I (pA)")
    axRev.legend(fontsize=8, title="고정 전압", ncol=2)
    axRev.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT, "3-1_erev_receptor.png")
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f"[그림2] {out}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[수용체] " + " · ".join(f"{n}(τr{tr:g}/τd{td:g}, E_rev{E:g})"
                                   for n, tr, td, g, E, c in RECEPTORS), flush=True)
    fig_kinetics()
    fig_erev()
    print("[설명] 흥분(AMPA·NMDA) E_rev=-8.5mV → 휴지에서 큰 내향(탈분극). "
          "억제(GABA_A) E_rev=-73mV ~ 휴지(-70)와 가까워 전류 작음(주로 션팅).", flush=True)


if __name__ == "__main__":
    main()
