"""
3-2_erev_concept.py — 반전전위(E_rev) 개념 설명 그림 (인과 흐름 + 시간축 + I-V)
============================================================================
"결합 → 열림 → 전류 → 전압변화" 인과 순서와 반전전위 개념을 한 장에:
  (상) 흐름도: ① 전달물질 결합 → ② 채널 열림(g↑) → ③ 전류 I=g·(V-E_rev) → ④ 막전위 변화
  (좌) 시간축: g(t) → I(t) → V(t) (각 단계의 실제 파형, 화살표로 인과)
  (우) I-V: 반전전위 = 전류가 0이 되는 막전위. V<E_rev 내향(탈분극)·V>E_rev 외향.

핵심: 수용체는 '전달물질 결합'으로 열린다(막전위 무관). E_rev 는 '열린 채널에서
전류가 0이 되고 방향이 바뀌는 막전위'. AMPA(흥분, E_rev=-8.5) 예시로 설명.
해석식 + 단순 passive 적분(NEURON 불필요).
실행: <ca1sim py> .../03_synapses/3-2_erev_concept.py
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
from matplotlib.patches import FancyBboxPatch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.dirname(THIS))
from common.plotstyle import set_korean_font           # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")

# AMPA(흥분성) 예시 파라미터
GHAT, TAU_R, TAU_D = 3.0, 0.2, 3.0     # nS, ms (가시적 EPSP 위해 ĝ 다소 크게)
E_REV = -8.5                            # 흥분성 반전전위 (mV)
V_REST = -70.0
RIN, CM = 100.0, 0.2                    # MΩ, nF → τm = 20 ms
T0 = 5.0                                # 전달물질 결합 시각


def biexp(t, t0, ghat, tr, td):
    tp = (tr * td) / (td - tr) * np.log(td / tr)
    A = 1.0 / (np.exp(-tp / td) - np.exp(-tp / tr))
    g = np.zeros_like(t)
    m = t >= t0
    dt = t[m] - t0
    g[m] = ghat * A * (np.exp(-dt / td) - np.exp(-dt / tr))
    return g


def integrate():
    """passive 1구획: Cm dV/dt = -(V-Vrest)/Rin - g(t)(V-E_rev). → g, I(pA), V(mV)."""
    dt = 0.05
    t = np.arange(0, 80, dt)
    g = biexp(t, T0, GHAT, TAU_R, TAU_D)          # nS
    V = np.empty_like(t); V[0] = V_REST
    I = np.empty_like(t)
    for i in range(len(t)):
        I[i] = g[i] * (V[i] - E_REV)              # pA  (nS·mV)
        if i + 1 < len(t):
            i_leak = (V[i] - V_REST) / RIN        # nA
            dV = -(i_leak + I[i] / 1000.0) / CM * dt
            V[i + 1] = V[i] + dV
    return t, g, I, V


def flow_diagram(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [("①\n전달물질 결합\n(전세포 발화)", 0.13),
             ("②\n채널 열림\n(전도도 g↑)", 0.38),
             ("③\n전류 흐름\nI = g·(V-E_rev)", 0.63),
             ("④\n막전위 변화\n(PSP)", 0.88)]
    for txt, x in steps:
        ax.add_patch(FancyBboxPatch((x - 0.10, 0.18), 0.20, 0.64,
                                    boxstyle="round,pad=0.02", fc="#DCE6F7", ec="0.3", lw=1.4))
        ax.text(x, 0.5, txt, ha="center", va="center", fontsize=9.5, fontweight="bold")
    for x0, x1 in [(0.23, 0.28), (0.48, 0.53), (0.73, 0.78)]:
        ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="0.3", lw=2.4))
    ax.text(0.63, 0.04, "수용체는 '막전위'가 아니라 '전달물질 결합'으로 열린다",
            ha="center", fontsize=8.5, color="tab:red", style="italic")
    ax.set_title("인과 순서:  결합 → 열림 → 전류 → 전압 변화   (AMPA 흥분성 예시)",
                 fontsize=12.5, fontweight="bold")


def main():
    os.makedirs(OUT, exist_ok=True)
    t, g, I, V = integrate()
    vpk = float(V.max())

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.85, 1, 1, 1], width_ratios=[1, 1],
                          hspace=0.5, wspace=0.22)
    axf = fig.add_subplot(gs[0, :])
    axg = fig.add_subplot(gs[1, 0])
    axi = fig.add_subplot(gs[2, 0], sharex=axg)
    axv = fig.add_subplot(gs[3, 0], sharex=axg)
    axIV = fig.add_subplot(gs[1:, 1])
    fig.suptitle("반전전위(E_rev) 이해 — 인과 흐름 · 시간축(g→I→V) · I-V",
                 fontsize=14, fontweight="bold")

    flow_diagram(axf)

    # ── 좌: 시간축 g → I → V ──
    axg.plot(t, g, color="tab:purple", lw=2.2)
    axg.axvline(T0, color="0.6", ls=":", lw=1)
    axg.annotate("① 결합 시각", xy=(T0, g.max() * 0.5), xytext=(T0 + 8, g.max() * 0.7),
                 fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.5"))
    axg.set_ylabel("g (nS)"); axg.set_title("② 채널 열림 — 전도도 g(t)", fontsize=10)
    axg.grid(alpha=0.3)

    axi.plot(t, I, color="tab:red", lw=2.2)
    axi.axhline(0, color="0.6", lw=0.8)
    axi.set_ylabel("I (pA)")
    axi.set_title("③ 전류 I = g·(V-E_rev)  — 내향(-)=탈분극성", fontsize=10)
    axi.grid(alpha=0.3)
    axi.annotate("음수=내향 전류\n(세포 안으로 +전하)", xy=(t[np.argmin(I)], I.min()),
                 xytext=(25, I.min() * 0.6), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.5"))

    axv.plot(t, V, color="navy", lw=2.2)
    axv.axhline(V_REST, color="0.6", ls=":", lw=1)
    axv.text(t[-1], V_REST, " 휴지 -70", fontsize=8, va="center", color="0.4")
    axv.annotate(f"EPSP 정점 {vpk:.1f}mV\n(E_rev=-8.5 쪽으로 상승)",
                 xy=(t[np.argmax(V)], vpk), xytext=(28, vpk + 1.5), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="0.5"))
    axv.set_ylabel("V (mV)"); axv.set_xlabel("시간 (ms)")
    axv.set_title("④ 막전위 변화 — EPSP (E_rev 쪽으로 탈분극)", fontsize=10)
    axv.grid(alpha=0.3)

    # 단계 사이 인과 화살표(그림 좌표)
    for y0, y1, lab in [(0.625, 0.60, "× (V-E_rev)\n옴 법칙"),
                        (0.395, 0.37, "막 충전\n(전류→전압)")]:
        fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
            (0.265, y0), (0.265, y1), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=18, color="0.4", lw=2.0))
        fig.text(0.285, (y0 + y1) / 2, lab, fontsize=7.6, color="0.4", va="center")

    # ── 우: I-V (반전전위) ──
    Vx = np.linspace(-95, 10, 240)
    Iline = GHAT * (Vx - E_REV)
    axIV.axhspan(axIV.get_ylim()[0] if False else -260, 0, xmin=0, xmax=1, color="none")
    axIV.fill_between(Vx, GHAT * (Vx - E_REV), 0, where=(Vx < E_REV),
                      color="tab:blue", alpha=0.08)
    axIV.fill_between(Vx, GHAT * (Vx - E_REV), 0, where=(Vx > E_REV),
                      color="tab:red", alpha=0.08)
    axIV.plot(Vx, Iline, color="k", lw=2.4)
    axIV.axhline(0, color="0.6", lw=0.8)
    # E_rev 점
    axIV.plot(E_REV, 0, "o", color="tab:green", ms=13, mec="k", zorder=6)
    axIV.annotate("반전전위 E_rev=-8.5mV\n전류=0 (구동력 0)\n채널은 열려있어도 안 흐름",
                  xy=(E_REV, 0), xytext=(E_REV - 2, GHAT * 30), fontsize=9,
                  ha="center", arrowprops=dict(arrowstyle="->", color="tab:green"),
                  color="tab:green", fontweight="bold")
    # 휴지점 + 구동력
    I_rest = GHAT * (V_REST - E_REV)
    axIV.axvline(V_REST, color="0.45", ls="--", lw=1.2)
    axIV.plot(V_REST, I_rest, "o", color="navy", ms=11, mec="k", zorder=6)
    axIV.annotate("", xy=(V_REST, I_rest), xytext=(V_REST, 0),
                  arrowprops=dict(arrowstyle="<->", color="navy", lw=1.8))
    axIV.text(V_REST - 1.5, I_rest * 0.5, "구동력\nV-E_rev\n=-61.5mV", ha="right",
              fontsize=8.5, color="navy", fontweight="bold")
    # EPSP 궤적(휴지→정점, E_rev 쪽)
    axIV.annotate("", xy=(vpk, GHAT * (vpk - E_REV)), xytext=(V_REST, I_rest),
                  arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=2.4))
    axIV.text((V_REST + vpk) / 2 - 2, GHAT * ((V_REST + vpk) / 2 - E_REV) - 18,
              "EPSP: V↑ → 구동력↓ → I↓", color="tab:orange", fontsize=8.5, fontweight="bold")
    # 영역 라벨
    axIV.text(-88, GHAT * (-88 - E_REV) - 18, "V < E_rev\n내향(탈분극)", color="tab:blue",
              fontsize=8.5, fontweight="bold")
    axIV.text(2, GHAT * (2 - E_REV) + 5, "V > E_rev\n외향", color="tab:red",
              fontsize=8.5, fontweight="bold")
    axIV.set_xlabel("막전위 V (mV)"); axIV.set_ylabel("시냅스 전류 I (pA)")
    axIV.set_title("I-V 관계 — 반전전위 = 전류 방향이 바뀌는 막전위\n"
                   "(왼쪽 시간축의 각 순간 (V,I)가 이 직선 위의 점)", fontsize=10.5)
    axIV.grid(alpha=0.3)

    out = os.path.join(OUT, "3-2_erev_concept.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[그림] {out}", flush=True)
    print(f"[요약] 결합→열림→전류(내향 최대 {I.min():.0f}pA)→EPSP(정점 {vpk:.1f}mV). "
          f"E_rev=-8.5 에서 전류 0.", flush=True)


if __name__ == "__main__":
    main()
