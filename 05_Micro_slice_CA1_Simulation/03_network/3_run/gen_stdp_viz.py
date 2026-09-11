# -*- coding: utf-8 -*-
"""나-1-② STDP 전파 시각화 — SC→PC 단일세포에서 세그먼트 Vm(t)을 기록하고,
유도 30회 창을 평균해 EPSP 전파(시냅스→소마) + bAP 역전파(소마→시냅스) 3D 데이터 저장.
gen_stdp_pair.py와 동일 셋업(gid·시냅스·gamma·ca_stp). 실행: gen_stdp_viz.py [--dt 10] [--npair 30] [--ca_stp 0]
출력: scratch/stdp_viz_dt{DT}_ca{CA}.npz (segpos·soma_pos·synpos·Vavg·twin)."""
import os, sys, time, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib")); import net_build as nb
DERIVED = os.path.join(ROOT, "data", "derived")
def arg(f, d): return type(d)(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
DT = int(arg("--dt", 10)); NPAIR = int(arg("--npair", 30)); CA = int(arg("--ca_stp", 0))
INTERVAL = 200.0; NPOST = 4; POSTISI = 10.0; SETTLE = 300.0; RECDT = 0.5
GAMMA_P, GAMMA_D = 1645.59, 313.0965
WPRE, WPOST = 20.0, 190.0   # 평균 창: pre 스파이크 20ms 전 ~ 190ms 후

def seg_kdtree(cell):
    P, ref = [], []
    for sec in cell.all:
        n = int(sec.n3d())
        if n < 2: continue
        Lt = sec.arc3d(n-1) or 1.0
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i))); ref.append((sec, min(max(sec.arc3d(i)/Lt, 0.0), 1.0)))
    return cKDTree(np.array(P)), ref

def seg_positions(cell, xyz, rot):
    """세그먼트 중심 3D 위치 + 세그먼트 참조 + 소마 위치."""
    pos, ref = [], []
    for sec in cell.all:
        n = int(sec.n3d())
        if n < 2: continue
        arc = np.array([sec.arc3d(i) for i in range(n)]); Lt = arc[-1] or 1.0
        xs = np.array([sec.x3d(i) for i in range(n)]); ys = np.array([sec.y3d(i) for i in range(n)]); zs = np.array([sec.z3d(i) for i in range(n)])
        for seg in sec:
            a = seg.x * Lt
            loc = np.array([np.interp(a, arc, xs), np.interp(a, arc, ys), np.interp(a, arc, zs)])
            pos.append(xyz + rot.apply(loc)); ref.append(seg)
    soma = cell.soma[0]; ns = int(soma.n3d())
    sc = np.array([soma.x3d(ns//2), soma.y3d(ns//2), soma.z3d(ns//2)]) if ns >= 1 else np.zeros(3)
    return np.array(pos), ref, xyz + rot.apply(sc)

B = nb.NetBuilder(); hh = B.h
wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
XYZ, Q, mt = wc["xyz"], wc["orientation_wxyz"], B.mt
sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
post, wxyz, dist_e3 = sc["post_gid"], sc["xyz"].astype(float), sc["dist_e3"].astype(float)
pc_mask = (mt[post] == "SP_PC") & (dist_e3 < 150); gid = int(np.bincount(post[pc_mask]).argmax())
sidx = np.where(post == gid)[0]
cell = B.build_cell(gid); rot = Rot.from_quat(Q[gid][[1, 2, 3, 0]]); soma = cell.soma[0]
tree, ref = seg_kdtree(cell)
si = int(sidx[0]); mp = rot.inv().apply(wxyz[si] - XYZ[gid]); _, k = tree.query(mp, k=1); sec, x = ref[k]
syn = hh.GBPlasticityStpProbSyn(sec(x))
syn.Use = 0.14; syn.Dep = 186; syn.Fac = 129; syn.Nrrp = 12; syn.gmax = 0.8/1000.0
syn.gamma_p = GAMMA_P; syn.gamma_d = GAMMA_D; syn.ca_stp = CA; syn.rho0 = 0.0
syn.setRNG(gid+1, si+1, 3)
synpos = XYZ[gid] + rot.apply(wxyz[si] - XYZ[gid] + rot.inv().apply(np.zeros(3)))  # 시냅스 3D 위치
synpos = wxyz[si]

# 세그먼트 위치·Vm 기록
segpos, segref, soma_pos = seg_positions(cell, XYZ[gid], rot)
tv = hh.Vector(); tv.record(hh._ref_t, RECDT)
vseg = [hh.Vector() for _ in segref]; [v.record(s._ref_v, RECDT) for v, s in zip(vseg, segref)]

# 자극: pre VecStim + post IClamp 버스트, Δt 간격, NPAIR회
vs = hh.VecStim(); ncpre = hh.NetCon(vs, syn); ncpre.weight[0] = 1.0; ncpre.delay = 0.5
ncpost = hh.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncpost.weight[0] = -1.0; ncpost.threshold = -10.0
pre_times = [SETTLE + kk*INTERVAL for kk in range(NPAIR)]
playv = hh.Vector(pre_times); vs.play(playv)
ics = [hh.IClamp(soma(0.5)) for _ in range(NPAIR*NPOST)]
for ic in ics: ic.dur = 2.0; ic.amp = 2.0
for kk, pt in enumerate(pre_times):
    for j in range(NPOST): ics[kk*NPOST + j].delay = pt + DT + j*POSTISI
tstop = pre_times[-1] + DT + (NPOST-1)*POSTISI + WPOST
print(f"[stdp-viz] gid {gid} · 세그 {len(segref)} · Δt{DT:+d} · ca_stp {CA} · {NPAIR}회 · tstop {tstop:.0f}ms", flush=True)
t0 = time.time(); hh.dt = 0.025; hh.finitialize(-70); hh.continuerun(tstop)
t = np.array(tv); V = np.array([np.array(v) for v in vseg])   # (nseg, nframes)
print(f"[stdp-viz] 구동 {time.time()-t0:.0f}s · V {V.shape}", flush=True)

# 30회 창 평균 (pre 스파이크 기준 정렬)
wf = int(round((WPRE + WPOST) / RECDT))
acc = np.zeros((V.shape[0], wf)); cnt = 0
for pt in pre_times:
    i0 = int(np.searchsorted(t, pt - WPRE))
    if i0 + wf <= V.shape[1]:
        acc += V[:, i0:i0+wf]; cnt += 1
Vavg = acc / max(cnt, 1)
twin = np.arange(wf) * RECDT - WPRE   # 상대시각, 0 = pre 스파이크
out = os.path.join(ROOT, "scratch", f"stdp_viz_dt{DT}_ca{CA}.npz")
np.savez_compressed(out, segpos=segpos, soma_pos=soma_pos, synpos=synpos,
                    Vavg=Vavg.astype(np.float32), twin=twin, dt=DT, ca_stp=CA, npair=cnt)
print(f"[stdp-viz] 저장 {out} · 평균 {cnt}회 · 창 {wf}프레임({-WPRE:.0f}~{WPOST:.0f}ms)", flush=True)
