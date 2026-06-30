"""
4-2_tm_dynamics.py — TM 단기가소성 '시정수' 설명 (시간축 연속 u(t)·R(t))
============================================================================
Source: Ecker(2020) §2.4 Eq.(5)-(6) Tsodyks-Markram.

4_tm_stp.py 는 발화순번(이산점)으로 그려 D·F 시정수가 안 보인다. 이 그림은
**시간축 연속 곡선**으로 u(t)·R(t)·반응(u·R)을 그려, 다음을 눈으로 보이게 한다:
  - R(가용자원): 스파이크마다 u·R 만큼 **급감** → 시정수 **D 로 느리게 회복** → 고갈
  - u(방출확률): 스파이크마다 +U(1-u) **상승** → 시정수 **F 로 이완(decay)**
  - 반응 amp = u·R

대비(F 역할):
  (좌) PC->PC (억압, E2): U=0.50, D=671, F=17  → R 고갈 지배, u 거의 일정 → 억압
  (우) PC->SOM+ (촉진, E1): U=0.09, D=138, F=670 → u 축적 지배 → 촉진

실행: <ca1sim py> .../03_synapses/4-2_tm_dynamics.py   (NEURON 불필요, 해석식)
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
from params_table3 import CLASSES                       # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")


def tm_traces(spikes, U, D, F, t_end, dt=0.25):
    """연속 u(t), R(t) + 스파이크별 (t, amp, u_release, R_release)."""
    t = np.arange(0, t_end + dt, dt)
    ut = np.zeros_like(t); Rt = np.ones_like(t)
    u, R, prev = 0.0, 1.0, 0.0
    spk_info = []
    for k, spk in enumerate(spikes):
        seg = (t >= prev) & (t < spk)
        if k == 0:
            ut[seg] = 0.0; Rt[seg] = 1.0
            u_ev, R_ev = 0.0, 1.0
        else:
            ut[seg] = u * np.exp(-(t[seg] - prev) / F)              # u 이완 (τ=F)
            Rt[seg] = 1.0 - (1.0 - R) * np.exp(-(t[seg] - prev) / D)  # R 회복 (τ=D)
            u_ev = u * np.exp(-(spk - prev) / F)
            R_ev = 1.0 - (1.0 - R) * np.exp(-(spk - prev) / D)
        u_rel = u_ev + U * (1.0 - u_ev)                            # 촉진: 스파이크에서 u 상승
        amp = u_rel * R_ev                                          # 반응 = u·R
        spk_info.append((spk, amp, u_rel, R_ev))
        u, R, prev = u_rel, R_ev - u_rel * R_ev, spk               # R 소모
    seg = t >= prev
    ut[seg] = u * np.exp(-(t[seg] - prev) / F)
    Rt[seg] = 1.0 - (1.0 - R) * np.exp(-(t[seg] - prev) / D)
    return t, ut, Rt, spk_info


def panel(ax, title, p, color, note):
    U, D, F = p["Use"], p["Dep"], p["Fac"]
    isi = 1000.0 / 20.0                              # 20 Hz
    spikes = 50.0 + np.arange(8) * isi
    t, ut, Rt, info = tm_traces(spikes, U, D, F, t_end=spikes[-1] + 150)

    # 연속 곡선
    ax.plot(t, Rt, "--", color="0.45", lw=1.8, label="R (가용자원)")
    ax.plot(t, ut, ":", color="tab:purple", lw=1.8, label="u (방출확률)")
    ax.axhline(U, color="tab:purple", ls="-", lw=1.0, alpha=0.6)
    # 반응 amp = u·R (스파이크 stem)
    sp_t = [s[0] for s in info]; sp_a = [s[1] for s in info]
    ax.stem(sp_t, sp_a, linefmt=color, markerfmt="o", basefmt=" ")
    for s in info:
        ax.plot([s[0]], [s[1]], "o", color=color, ms=5)

    # 시정수 주석: R 회복(D) 한 구간 + u 이완(F)
    k = 2                                            # 3번째 스파이크 직후 구간에 표기
    t0 = sp_t[k]; t1 = sp_t[k + 1]
    ax.annotate(f"R 회복 τ=D={D:.0f}ms", xy=((t0 + t1) / 2, np.interp((t0 + t1) / 2, t, Rt)),
                xytext=((t0 + t1) / 2 + 25, np.interp((t0 + t1) / 2, t, Rt) + 0.18),
                fontsize=8.3, color="0.3", arrowprops=dict(arrowstyle="->", color="0.5"))
    ax.annotate(f"u 이완 τ=F={F:.0f}ms", xy=(t0 + 6, np.interp(t0 + 6, t, ut)),
                xytext=(t0 + 30, min(0.9, U + 0.35)),
                fontsize=8.3, color="tab:purple", arrowprops=dict(arrowstyle="->", color="tab:purple"))
    # 첫 스파이크 소모 표시
    ax.annotate("스파이크마다\nu·R 소모", xy=(sp_t[0], sp_a[0]),
                xytext=(sp_t[0] - 5, sp_a[0] + 0.22), fontsize=8.0, color=color,
                ha="center", arrowprops=dict(arrowstyle="->", color=color))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("시간 (ms)"); ax.set_ylim(0, 1.18); ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="0.45", ls="--", lw=1.8, label="R (가용자원, τ=D)"),
        Line2D([0], [0], color="tab:purple", ls=":", lw=1.8, label="u (방출확률, τ=F)"),
        Line2D([0], [0], color="tab:purple", ls="-", lw=1.0, label=f"U_SE={U:.2f} (첫 방출확률)"),
        Line2D([0], [0], color=color, marker="o", lw=1.4, label="반응 (u·R)"),
    ]
    ax.legend(handles=handles, fontsize=7.8, loc="upper right", framealpha=0.95)
    ax.text(0.985, 0.72, note, transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(fc="#FFF6D5", ec="0.6", alpha=0.95))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    fig.suptitle("TM 단기가소성 — 시간축에서 본 u·R 동역학 (D·F·U_SE 의 역할, 20 Hz)",
                 fontsize=13.5, fontweight="bold")

    pc = CLASSES["PC->PC (E2)"]
    som = CLASSES["PC->SOM+ (E1)"]
    panel(axL, "PC->PC (E2) — 억압", pc, "tab:red",
          f"U_SE={pc['Use']:.2f}, D={pc['Dep']:.0f}ms, F={pc['Fac']:.0f}ms\n"
          "U 큼+회복 D 느림 -> R 고갈이 지배\n"
          "F 작음 -> u 거의 일정 => 반응 감소(억압)")
    panel(axR, "PC->SOM+ (E1) — 촉진", som, "tab:green",
          f"U_SE={som['Use']:.2f}, D={som['Dep']:.0f}ms, F={som['Fac']:.0f}ms\n"
          "U 작음 -> R 소모 적음(유지)\n"
          "F 큼 -> u 축적이 지배 => 반응 증가(촉진)")
    axL.set_ylabel("u, R, 반응(u·R)  [0~1]")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "4-2_tm_dynamics.png")
    fig.savefig(out, dpi=130)
    print(f"[그림] {out}", flush=True)
    for nm, p in (("PC->PC", pc), ("PC->SOM+", som)):
        _, _, _, info = tm_traces(50 + np.arange(8) * 50.0, p["Use"], p["Dep"], p["Fac"], 550)
        a = [s[1] for s in info]
        print(f"  {nm}: amp 1->8 = {a[0]:.3f}->{a[-1]:.3f} ({a[-1]/a[0]:.2f}배), "
              f"U={p['Use']}, D={p['Dep']}, F={p['Fac']}", flush=True)


if __name__ == "__main__":
    main()
