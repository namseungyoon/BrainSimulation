# -*- coding: utf-8 -*-
"""
11_schaffer/sc_epsp_test.py  —  E2 1단계: 단일 Schaffer collateral(SC)→PC EPSP 검증

전체 슬라이스 배선 전에, **대표 추체세포 1개**의 apical(SR) 수상돌기에 흥분성 시냅스를 놓고
단일 전세포 스파이크로 유발되는 **소마 EPSP 진폭**을 측정 → Romani SC-PC 0.15±0.12mV와 대조.
정점거리별(근위 SR / 원위 SR / SLM) 감쇠도 함께 확인.

⚠️ 현재는 Ecker PC->PC(E2) AMPA/NMDA 파라미터 재사용(SC 전용 최적화는 이후). 결정적(deterministic) 방출.
실행: python 11_schaffer/sc_epsp_test.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT); SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED); sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, os.path.join(PAPER, "04_network"))
from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
import params_table3 as P3
from synapse_pair import build_synapse
MODELS = os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
ROMANI_EPSP = (0.15, 0.12)   # Romani SC-PC EPSP 평균±표준편차 (mV)


def main():
    type_dir = net.load_representatives(MODELS)
    cell, tname = load_cell(type_dir["PC"], gid=0)
    print(f"[세포] 대표 추체 {tname}", flush=True)
    soma = cell.soma[0]
    h.distance(0, soma(0.5))
    # apical 섹션(이름에 apic) 중 경로거리별로 3곳 선택: 근위SR/원위SR/SLM
    apic = [s for s in cell.all if ".apic" in s.name()]
    if not apic:
        apic = [s for s in cell.all if ".dend" in s.name()]
    seg_by_dist = []
    for s in apic:
        d = h.distance(s(0.5))
        seg_by_dist.append((d, s))
    seg_by_dist.sort()
    dmax = seg_by_dist[-1][0]
    targets = []
    for frac, lab in [(0.25, "근위 SR"), (0.55, "원위 SR"), (0.85, "SLM 부근")]:
        want = frac * dmax
        d, s = min(seg_by_dist, key=lambda x: abs(x[0] - want))
        targets.append((lab, d, s(0.5)))
    print("[표적] " + ", ".join(f"{l}({d:.0f}um)" for l, d, _ in targets), flush=True)

    p = P3.CLASSES["PC->PC (E2)"]   # 흥분성 AMPA/NMDA (SC 전용 최적화 전, 재사용)
    print(f"[시냅스] Ecker PC->PC(E2): g={p['g_nS']}nS, NMDA비={p['NMDA_ratio']}", flush=True)
    keep = []
    fire_t = [50.0, 120.0, 190.0]   # 각 표적을 다른 시각에 1회 발화 → 한 번 실행서 3 EPSP
    for (lab, d, seg), tf in zip(targets, fire_t):
        syn = build_synapse(seg, p, seeds=(1, 1, 1), deterministic=True)
        ns = h.NetStim(); ns.number = 1; ns.start = tf; ns.interval = 1; ns.noise = 0
        nc = h.NetCon(ns, syn); nc.weight[0] = p["g_nS"]; nc.delay = 1.0
        keep += [syn, ns, nc]

    vsoma = h.Vector().record(soma(0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025
    h.finitialize(-70.0); h.continuerun(250.0)
    t = np.array(tvec); v = np.array(vsoma)

    # 각 발화 후 EPSP 진폭
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, v, color="#C0392B", lw=1.0)
    results = []
    for (lab, d, seg), tf in zip(targets, fire_t):
        base = v[(t >= tf - 5) & (t < tf)].mean()
        win = v[(t >= tf) & (t < tf + 40)]
        pk = win.max(); amp = pk - base
        results.append((lab, d, amp))
        ax.axvline(tf, color="gray", ls=":", lw=0.6)
        ax.annotate(f"{lab}\n{amp:.3f}mV", (tf + 5, pk), fontsize=8)
        print(f"  [{lab} {d:.0f}um] EPSP {amp:.3f}mV", flush=True)
    ax.axhspan(-70 + ROMANI_EPSP[0] - ROMANI_EPSP[1], -70 + ROMANI_EPSP[0] + ROMANI_EPSP[1],
               color="green", alpha=0.12)
    ax.text(250, -70 + ROMANI_EPSP[0], f" Romani SC-PC\n {ROMANI_EPSP[0]}±{ROMANI_EPSP[1]}mV",
            fontsize=8, va="center", color="green")
    ax.set_xlabel("시간 (ms)"); ax.set_ylabel("소마 막전위 (mV)")
    ax.set_title("E2-a  단일 SC→PC EPSP 검증 (apical 위치별) — 소마에서 측정\n"
                 "정점거리 멀수록 소마 EPSP 감쇠 (Ecker E2 AMPA/NMDA 재사용, 결정적)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG, "E2a_sc_epsp.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[그림] {out}", flush=True)
    prox = results[0][2]
    print(f"[대조] 근위 SR EPSP {prox:.3f}mV vs Romani {ROMANI_EPSP[0]}±{ROMANI_EPSP[1]}mV → "
          f"{'범위 내' if abs(prox-ROMANI_EPSP[0])<=ROMANI_EPSP[1] else '조정 필요(g 스케일)'}", flush=True)


if __name__ == "__main__":
    main()
