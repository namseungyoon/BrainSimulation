"""
2_num_synapses.py — 단계 2: 연결당 시냅스 수 (number of synapses per connection)
[8개 파라미터 추적] 이 단계: 0/8 — 해부 검증(파라미터 아님)
============================================================================
Source: Ecker et al. (2020) Fig.2 단계2, Fig.3b, §3.3.

논문 단계2 = "한 연결(connection)은 여러 개의 시냅스 접촉(contact)으로 이뤄진다".
연결당 시냅스 수(Nsyn/conn)가 많을수록 소마 PSP 가 커진다. 실제 PC 에서 Nsyn 을 바꿔 측정.
참고(논문): E-E≈1.3, E-I≈8.2, PC→O-LM≈2.8, I-I≈2.8 등 유형마다 다름.

실행: conda activate ca1sim
      python papers/01_Ecker2020_CA1_synaptic/03_synapses/2_num_synapses.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.nrn_env import h                # noqa: E402
from common.plotstyle import set_korean_font        # noqa: E402
from common.corrections import q10_scale            # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params_table3 import CLASSES, NSYN_PER_CONNECTION   # noqa: E402
from paired_recording import load_pc, place_synapses, measure_psp, V_HOLD, T_SPIKE, TSTOP, DT  # noqa: E402

set_korean_font()
h.load_file("stdrun.hoc")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    os.makedirs(OUT, exist_ok=True)
    p = dict(CLASSES["PC->PC (E2)"])
    p["tau_d_AMPA"] = q10_scale(p["tau_d_AMPA"], 2.2, 25.0, 34.0)
    g = p["g_nS"]

    cell, _ = load_pc()
    vsoma = h.Vector().record(cell.soma[0](0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0
    h.dt = DT

    n_list = [1, 2, 3, 5, 8]
    psps = []
    for n in n_list:
        syns, ncs, _keep = place_synapses(cell, p, n_syn=n)
        psp, _ = measure_psp(syns, ncs, vsoma, tvec, g, n_trials=6)
        psps.append(psp)
        print(f"  Nsyn={n}: 소마 EPSP = {psp:.3f} mV")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                  gridspec_kw=dict(width_ratios=[1, 1.25]))
    fig.suptitle("단계 2 — 연결당 시냅스 수(Nsyn/conn) / synapses per connection",
                 fontsize=13, fontweight="bold")

    # (좌) PC→PC 에서 Nsyn 을 바꿔 측정 → PSP 단조 증가 (모델 시연)
    ax.plot(n_list, psps, "o-", color="tab:purple", lw=2, ms=7)
    nref = NSYN_PER_CONNECTION["PC->PC (E2)"][0]
    ax.axvline(nref, color="tab:purple", ls=":", lw=1.4)
    ax.text(nref, ax.get_ylim()[1] * 0.96, f" 문헌 {nref}", fontsize=8,
            color="tab:purple", va="top")
    ax.set_xlabel("연결당 시냅스 수 Nsyn/conn")
    ax.set_ylabel("소마 EPSP 진폭 (mV)")
    ax.set_title("(좌) 모델: 시냅스↑ → PSP↑ (PC→PC, g_hat 고정)", fontsize=10)
    ax.set_xticks(n_list); ax.grid(alpha=0.3)

    # (우) 경로별 Nsyn 참고값 (문서값=진하게, 추정값=빗금)
    names = list(NSYN_PER_CONNECTION.keys())
    vals = [NSYN_PER_CONNECTION[k][0] for k in names]
    conf = [NSYN_PER_CONNECTION[k][1] for k in names]
    y = np.arange(len(names))[::-1]
    for yi, k, v, c in zip(y, names, vals, conf):
        doc = c == "documented"
        ax2.barh(yi, v, color=("#3B6BA5" if doc else "#9FB8D4"),
                 hatch=(None if doc else "///"), edgecolor="k", lw=0.6)
        ax2.text(v + 0.15, yi, f"{v:g}", va="center", fontsize=8.5)
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8.5)
    ax2.set_xlabel("연결당 시냅스 수 (대표값)")
    ax2.set_title("(우) 9 클래스 Nsyn 참고값  ■문서값 ▨추정값", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(fc="#3B6BA5", ec="k", label="문서값(Ecker/Megías)"),
                        Patch(fc="#9FB8D4", ec="k", hatch="///", label="추정값(Bezaire&Soltesz)")],
               fontsize=7.5, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "2_num_synapses.png")
    plt.savefig(out, dpi=120)
    print(f"[그림] {out}")
    print(f"[검증] Nsyn 1→8 에서 EPSP {psps[0]:.3f}→{psps[-1]:.3f} mV (단조 증가)")
    print("[Nsyn 표] " + ", ".join(f"{k}={v:g}({c[:3]})"
                                   for k, (v, c, _) in NSYN_PER_CONNECTION.items()))


if __name__ == "__main__":
    main()
