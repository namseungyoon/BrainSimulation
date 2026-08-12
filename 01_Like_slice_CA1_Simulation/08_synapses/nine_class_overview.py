# -*- coding: utf-8 -*-
"""
08_synapses/nine_class_overview.py  —  9 pathway 클래스 개요 (3×3 카드)

각 클래스(=Ecker paired-recording으로 특성화한 연결 유형)를 한 패널로:
  - pre→post 세포유형 · 흥분/억제 · STP 라벨
  - STP 프로파일(Tsodyks-Markram, 20Hz) 곡선
  - 핵심 파라미터(g_nS·U·D·F·Nrrp) + 우리 커넥텀 시냅스 수
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "papers",
                                "01_Ecker2020_CA1_synaptic", "03_synapses"))
from params_table3 import CLASSES                # noqa: E402
FIG = os.path.join(HERE, "figures")


def tm(U, D, F, spikes):
    u = U; R = 1.0; out = []; prev = None; up = U
    for t in spikes:
        if prev is None:
            u = U; R = 1.0
        else:
            dt = t - prev
            u = U + up * (1 - U) * np.exp(-dt / F)
            R = 1 + (R - up * R - 1) * np.exp(-dt / D)
        out.append(u * R); up = u; prev = t
    return np.array(out)


def main():
    summ = json.load(open(os.path.join(HERE, "synapse_assignment_summary.json"),
                          encoding="utf-8"))
    our = {r["class"]: r for r in summ["by_class"]}
    spikes = list(np.arange(8) * 50.0)
    names = list(CLASSES.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, name in zip(axes.ravel(), names):
        p = CLASSES[name]
        ei = p["ei"]; col = "#DD8452" if ei == "E" else "#4C72B0"
        psp = tm(p["Use"], p["Dep"], p["Fac"], spikes); psp = psp / psp[0]
        ax.plot(range(1, 9), psp, "o-", color=col, lw=2)
        ax.axhline(1, color="gray", ls=":", lw=0.7)
        ax.set_ylim(0, max(2.9, psp.max() * 1.1))
        ax.set_xlabel("스파이크 순번(20Hz)", fontsize=8)
        ax.set_ylabel("정규화 PSP", fontsize=8)
        pre, post = p["pre"], p["post"]
        oc = our.get(name, {})
        prof = {"E1": "촉진", "E2": "억압", "I1": "촉진", "I2": "억압",
                "I3": "pseudo-linear"}[p["stp"]]
        ax.set_title(f"{name}\n{pre} → {post}  ({'흥분' if ei=='E' else '억제'}, {p['stp']}={prof})",
                     fontsize=10, color=col)
        txt = (f"g={p['g_nS']}nS  U={p['Use']}\nDep={p['Dep']}ms  Fac={p['Fac']}ms\n"
               f"Nrrp={p['Nrrp']}  NMDA비={p['NMDA_ratio']}\n"
               f"우리 시냅스: {oc.get('our_synapses',0):,}개")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, bbox=dict(boxstyle="round", fc="#f7f7f7", ec=col, alpha=0.9))
    fig.suptitle("9 pathway 클래스 개요 — 각 연결 유형이 어떻게 특성화됐나 (Ecker 2020 paired recording)\n"
                 "pre→post · 흥분(주황)/억제(파랑) · STP 프로파일(반복자극 시 PSP 변화) · 파라미터 · 우리 커넥텀 시냅스수",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "V4_4_nine_class_overview.png"), dpi=130)
    plt.close(fig)
    print(f"[OK] -> {FIG}/V4_4_nine_class_overview.png")


if __name__ == "__main__":
    main()
