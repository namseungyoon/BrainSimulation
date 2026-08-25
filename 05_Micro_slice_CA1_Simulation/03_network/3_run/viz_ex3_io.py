# -*- coding: utf-8 -*-
"""Ex3 I-O 곡선 — 발화% + fEPSP(peak) vs fiber volley, 정상 vs 억제차단.
입력: scratch/ex3_io.npz (또는 ex3_io.json). 출력: Ex3_io_inhibition/figures/ex3_io_curve.png
실행: python 03_network/3_run/viz_ex3_io.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CN = {"normal": "#4C72B0", "block": "#C44E52"}   # 정상 파랑, 억제차단 빨강
LB = {"normal": "normal (inhibition on)", "block": "block (GABA off)"}


def main():
    d = json.load(open(os.path.join(ROOT, "scratch", "ex3_io.json")))
    R = d["results"]
    conds = {}
    for r in R:
        conds.setdefault(r["cond"], []).append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for cn, rows in conds.items():
        rows = sorted(rows, key=lambda r: r["volley_pct"])
        x = [r["volley_pct"] for r in rows]
        fe = [r["pctE"] for r in rows]
        pk = [-r["fepsp_peak_uV"][2] / 1000.0 for r in rows]   # E3 |peak| mV (sink 크기)
        ax1.plot(x, fe, "o-", color=CN.get(cn, "#888"), lw=2, ms=7, label=LB.get(cn, cn))
        ax2.plot(x, pk, "s-", color=CN.get(cn, "#888"), lw=2, ms=7, label=LB.get(cn, cn))
    ax1.set_xlabel("fiber volley (recruited %)"); ax1.set_ylabel("pyramidal firing [%]")
    ax1.set_title("Firing I-O"); ax1.set_ylim(0, 105); ax1.legend(fontsize=9); ax1.grid(alpha=.2)
    ax2.set_xlabel("fiber volley (recruited %)"); ax2.set_ylabel("fEPSP E3(SR) |peak| [mV]")
    ax2.set_title("fEPSP I-O"); ax2.legend(fontsize=9); ax2.grid(alpha=.2)
    fig.suptitle("Ex3 SC I-O (w=0, R-fraction) — normal vs inhibition block\n"
                 "note: strong stim → firing saturates by 50% (limited dynamic range)", fontsize=11)
    fig.tight_layout()
    outd = os.path.join(ROOT, "04_experiments", "Ex3_io_inhibition", "figures"); os.makedirs(outd, exist_ok=True)
    p = os.path.join(outd, "ex3_io_curve.png"); fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"[ex3-io] 저장 {p}", flush=True)


if __name__ == "__main__":
    main()
