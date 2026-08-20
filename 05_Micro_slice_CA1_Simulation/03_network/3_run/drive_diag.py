# -*- coding: utf-8 -*-
"""
03_network/3_run/drive_diag.py  —  구동 진단 (단일세포, non-MPI)

E3 근처 추체 1개를 짓고 그 세포의 SC 시냅스(GBPlasticityStpProbSyn, 가소성 동결)를
부착한 뒤: (1) 안정화 후 정지막전위 확인 (2) 단일 volley·버스트 자극에 발화하나 확인.
이슈1(정지막전위 −86mV) 원인이 안정화 부족인지 판별.

실행: python 03_network/3_run/drive_diag.py
"""
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
import net_build as nb
DERIVED = os.path.join(ROOT, "data", "derived")
FIG = os.path.join(HERE, "figures")


def seg_kdtree(cell):
    P, ref = [], []
    for sec in cell.all:
        n = int(sec.n3d())
        if n < 2:
            continue
        Lt = sec.arc3d(n - 1) or 1.0
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            ref.append((sec, min(max(sec.arc3d(i) / Lt, 0.0), 1.0)))
    return cKDTree(np.array(P)), ref


def main():
    B = nb.NetBuilder(); hh = B.h
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]; mt = B.mt
    d = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    post = d["post_gid"]; wxyz = d["xyz"].astype(float); dist_e3 = d["dist_e3"].astype(float)

    # E3 근처 SC 많은 추체 선택
    pc_mask = (mt[post] == "SP_PC") & (dist_e3 < 150)
    gid = int(np.bincount(post[pc_mask]).argmax())
    sidx = np.where(post == gid)[0]
    print(f"[대상] gid {gid} · {mt[gid]} · SC 시냅스 {len(sidx)}개 (E3<150 {int(np.sum(dist_e3[sidx]<150))})")

    cell = B.build_cell(gid)
    tree, ref = seg_kdtree(cell)
    rot = Rot.from_quat(Q[gid][[1, 2, 3, 0]])
    soma = cell.soma[0]

    # --- 안정화 진단: 정지막전위가 시간에 따라 어떻게 잡히나 ---
    tv = hh.Vector(); tv.record(hh._ref_t)
    vv = hh.Vector(); vv.record(soma(0.5)._ref_v)
    hh.dt = 0.025
    hh.finitialize(-70)
    hh.continuerun(400)
    t = np.array(tv); v = np.array(vv)
    for tt in [50, 100, 200, 300, 400]:
        print(f"  t={tt:>3}ms  Vsoma {v[t<=tt][-1]:.2f} mV")
    vrest = v[-1]

    # --- 자극: 시냅스 부착 후 안정화 → 단일 volley + 버스트 ---
    syns, ncs, vstims = [], [], []
    for si in sidx:
        mp = rot.inv().apply(wxyz[si] - XYZ[gid])
        _, k = tree.query(mp, k=1); sec, x = ref[k]
        syn = hh.GBPlasticityStpProbSyn(sec(x))
        syn.Use = 0.14; syn.Dep = 186; syn.Fac = 129; syn.Nrrp = 12
        syn.gamma_p = 0.0; syn.gamma_d = 0.0
        syn.setRNG(gid + 1, int(si) + 1, 3)
        vs = hh.VecStim(); syns.append(syn); vstims.append(vs)
        nc = hh.NetCon(vs, syn); nc.weight[0] = 0.8 / 1000.0; nc.delay = 1.0
        nc2 = hh.NetCon(soma(0.5)._ref_v, syn, sec=soma); nc2.weight[0] = -1.0
        ncs.append((nc, nc2))

    apc = hh.APCount(soma(0.5)); apc.thresh = -10
    tv2 = hh.Vector(); tv2.record(hh._ref_t)
    vv2 = hh.Vector(); vv2.record(soma(0.5)._ref_v)
    results = {}
    SETTLE = 300.0
    playvecs = [hh.Vector() for _ in vstims]
    for vs, pv in zip(vstims, playvecs):
        vs.play(pv)
    for label, times, gscale in [("단일 0.8nS", [SETTLE + 10], 1.0),
                                 ("단일 2nS", [SETTLE + 10], 2.5),
                                 ("버스트 4x100Hz 2nS", [SETTLE + 10 + 10 * i for i in range(4)], 2.5)]:
        for (nc, _), pv in zip(ncs, playvecs):
            nc.weight[0] = 0.8 / 1000.0 * gscale
            pv.resize(0); pv.append(*times) if len(times) > 1 else pv.append(times[0])
        hh.finitialize(-70)
        hh.continuerun(SETTLE + 80)
        t2 = np.array(tv2); v2 = np.array(vv2)
        post_stim = v2[t2 >= SETTLE]
        results[label] = (t2.copy(), v2.copy(), int(apc.n), float(post_stim.max()))
        print(f"  [{label}] 발화 {int(apc.n)} · 자극후 최대 {post_stim.max():.1f}mV")

    # 그림
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].plot(t, v, color="#4C72B0"); ax[0].axhline(vrest, ls="--", color="gray", lw=1)
    ax[0].set_title(f"(a) 안정화 — gid {gid} {mt[gid]}\n정지막전위 {vrest:.1f}mV (400ms)")
    ax[0].set_xlabel("시간 (ms)"); ax[0].set_ylabel("소마 전압 (mV)"); ax[0].grid(alpha=0.3)
    for label, (t2, v2, nsp, vmax) in results.items():
        ax[1].plot(t2 - SETTLE, v2, lw=0.9, label=f"{label} ({nsp}sp)")
    ax[1].set_xlim(-10, 80); ax[1].axvline(10, ls=":", color="red", lw=1)
    ax[1].set_title("(b) 자극 응답 (안정화 후)"); ax[1].set_xlabel("자극기준 시간 (ms)")
    ax[1].set_ylabel("소마 전압 (mV)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"구동 진단 — {mt[gid]} SC {len(sidx)}시냅스 · 안정화+자극", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "drive_diag.png"), dpi=130)
    print(f"[그림] -> {FIG}/drive_diag.png")


if __name__ == "__main__":
    main()
