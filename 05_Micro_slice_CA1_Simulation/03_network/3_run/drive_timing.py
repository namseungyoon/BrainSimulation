# -*- coding: utf-8 -*-
"""
03_network/3_run/drive_timing.py  —  자극 타이밍 직관 그림 (단일세포)

"E3 자극"과 "SC 시냅스 전류"가 하나의 사슬임을 시간축에서 보여준다.
E3 근처 추체 1개 + 그 세포 SC 시냅스(GBPlasticityStpProbSyn) → 안정화 후
t=STIM에 섬유 발화(단일 volley) → 시냅스 전도도 → 소마 전압을 한 시간축에 3단으로.
실행: python 03_network/3_run/drive_timing.py
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
SETTLE, STIM = 300.0, 310.0


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

    pc_mask = (mt[post] == "SP_PC") & (dist_e3 < 150)
    gid = int(np.bincount(post[pc_mask]).argmax())
    sidx = np.where(post == gid)[0]
    print(f"[대상] gid {gid} {mt[gid]} · SC {len(sidx)}시냅스")

    cell = B.build_cell(gid); tree, ref = seg_kdtree(cell)
    rot = Rot.from_quat(Q[gid][[1, 2, 3, 0]]); soma = cell.soma[0]
    syns, vstims, gvecs = [], [], []
    for si in sidx:
        mp = rot.inv().apply(wxyz[si] - XYZ[gid])
        _, k = tree.query(mp, k=1); sec, x = ref[k]
        syn = hh.GBPlasticityStpProbSyn(sec(x))
        syn.Use = 0.14; syn.Dep = 186; syn.Fac = 129; syn.Nrrp = 12
        syn.gmax = 0.8 / 1000.0; syn.gamma_p = 0.0; syn.gamma_d = 0.0
        syn.setRNG(gid + 1, int(si) + 1, 3)
        vs = hh.VecStim(); nc = hh.NetCon(vs, syn); nc.weight[0] = 1.0; nc.delay = 1.0
        nc2 = hh.NetCon(soma(0.5)._ref_v, syn, sec=soma); nc2.weight[0] = -1.0
        gv = hh.Vector(); gv.record(syn._ref_g); gvecs.append(gv)
        syns.append(syn); vstims.append((vs, nc))

    # 자극: 단일 volley (모든 섬유 t=STIM 발화) — play 벡터 참조 유지(GC 방지)
    playv = hh.Vector([STIM])
    for vs, nc in vstims:
        vs.play(playv)
    tvec = hh.Vector(); tvec.record(hh._ref_t)
    vv = hh.Vector(); vv.record(soma(0.5)._ref_v)
    apc = hh.APCount(soma(0.5)); apc.thresh = -10
    hh.dt = 0.025; hh.finitialize(-70); hh.continuerun(STIM + 60)

    t = np.array(tvec); v = np.array(vv)
    gtot = np.sum([np.array(g) for g in gvecs], axis=0) * 1000.0   # µS→nS
    tt = t - STIM
    print(f"[결과] 발화 {int(apc.n)} · EPSP 최대 {v[t>=STIM].max():.1f}mV · g 최대 {gtot.max():.1f}nS")

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [0.5, 1, 1.3]})
    # (1) E3 자극 이벤트
    ax[0].axvline(0, color="red", lw=2)
    ax[0].plot([0], [1], "v", color="red", ms=14)
    ax[0].text(0.5, 1, " E3 자극 = 섬유 발화 이벤트 1개 (단일 volley)", va="center", fontsize=11)
    ax[0].set_ylim(0, 2); ax[0].set_yticks([]); ax[0].set_title("① 자극 (E3 → SC 섬유 발화)")
    # (2) SC 시냅스 전도도
    ax[1].plot(tt, gtot, color="#55A868", lw=1.6)
    ax[1].axvline(0, color="red", ls=":", lw=1)
    ax[1].set_ylabel("SC 시냅스\n총 전도도 (nS)"); ax[1].set_title("② SC 시냅스 열림 (전류 통로)")
    ax[1].grid(alpha=0.3)
    # (3) 후세포 전압
    ax[2].plot(tt, v, color="#4C72B0", lw=1.0)
    ax[2].axvline(0, color="red", ls=":", lw=1)
    ax[2].axhline(-50, color="gray", ls="--", lw=0.8); ax[2].text(45, -49, "역치≈−50mV", fontsize=8, color="gray")
    ax[2].set_ylabel("후세포 소마\n전압 (mV)"); ax[2].set_xlabel("자극기준 시간 (ms)")
    ax[2].set_title(f"③ 후세포 응답 (발화 {int(apc.n)}회)"); ax[2].grid(alpha=0.3)
    ax[2].set_xlim(-10, 55)
    fig.suptitle(f"자극 타이밍 사슬 — E3 자극이 SC 시냅스로 전류를 흘려 발화 (gid {gid} {mt[gid]}, SC {len(sidx)}시냅스)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "drive_timing.png"), dpi=130)
    print(f"[그림] -> {FIG}/drive_timing.png")


if __name__ == "__main__":
    main()
