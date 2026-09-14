# -*- coding: utf-8 -*-
"""나-1-② STDP 유형 재현 — Graupner & Brunel 2012 Fig.2 "하나의 규칙, 6가지 STDP".
칼슘 파라미터(C_pre·C_post·D·theta_p) 위치를 바꿔 6가지 STDP 곡선 유형을 SC→PC 쌍에서 재현.
단일 스파이크쌍(버스트 없음)·ca_stp=0(Graupner 원본)·rho0=0.5(불안정점 → LTD/LTP 둘 다 가시).
실행: gen_stdp_types.py [--npair 20]  → scratch/stdp_types.json"""
import os, sys, json, time, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib")); import net_build as nb
DERIVED = os.path.join(ROOT, "data", "derived")
def arg(f, d): return type(d)(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
NPAIR = int(arg("--npair", 20)); INTERVAL = 200.0; SETTLE = 300.0; RHO0 = 0.5
DTS = [-50, -30, -20, -10, -5, 5, 10, 20, 30, 50]
# (이름, 라벨, C_pre, C_post, D(ms), theta_p, theta_d) — Graupner&Brunel 2012 Fig.2
TYPES = [
    ("D",    "억압만 LTD-only",      0.6, 0.6, 0.0,  1.3, 1.0),
    ("DP",   "고전 헤비안 Hebbian",  1.0, 2.0, 13.7, 1.3, 1.0),
    ("P",    "강화만 LTP-only",      2.0, 2.0, 0.0,  1.3, 1.0),
    ("DPD",  "억압-강화-억압",       0.9, 0.9, 4.6,  1.3, 1.0),
    ("DPDp", "DPD 변형(θp↑)",        1.0, 2.0, 2.2,  2.5, 1.0),
    ("Dp",   "억압만 변형(θp매우높음)",1.0, 2.0, 0.0,  3.5, 1.0),
]

def seg_kdtree(cell):
    P, ref = [], []
    for sec in cell.all:
        n = int(sec.n3d())
        if n < 2: continue
        Lt = sec.arc3d(n-1) or 1.0
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i))); ref.append((sec, min(max(sec.arc3d(i)/Lt, 0.0), 1.0)))
    return cKDTree(np.array(P)), ref

B = nb.NetBuilder(); hh = B.h
wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
XYZ, Q, mt = wc["xyz"], wc["orientation_wxyz"], B.mt
sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
post, wxyz, dist_e3 = sc["post_gid"], sc["xyz"].astype(float), sc["dist_e3"].astype(float)
pc_mask = (mt[post] == "SP_PC") & (dist_e3 < 150); gid = int(np.bincount(post[pc_mask]).argmax())
sidx = np.where(post == gid)[0]
cell = B.build_cell(gid); tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[gid][[1, 2, 3, 0]]); soma = cell.soma[0]
si = int(sidx[0]); mp = rot.inv().apply(wxyz[si] - XYZ[gid]); _, k = tree.query(mp, k=1); sec, x = ref[k]
syn = hh.GBPlasticityStpProbSyn(sec(x))
syn.Use = 0.14; syn.Dep = 186; syn.Fac = 129; syn.Nrrp = 12; syn.gmax = 0.8/1000.0
syn.gamma_p = 1645.59; syn.gamma_d = 313.0965; syn.ca_stp = 0
syn.setRNG(gid+1, si+1, 3)
vs = hh.VecStim(); ncpre = hh.NetCon(vs, syn); ncpre.weight[0] = 1.0; ncpre.delay = 0.5
ncpost = hh.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncpost.weight[0] = -1.0; ncpost.threshold = -10.0
pre_times = [SETTLE + kk*INTERVAL for kk in range(NPAIR)]
playv = hh.Vector(pre_times); vs.play(playv)
ics = [hh.IClamp(soma(0.5)) for _ in range(NPAIR)]   # 단일 post 스파이크(버스트 없음)
for ic in ics: ic.dur = 2.0; ic.amp = 2.0
rho_v = hh.Vector(); rho_v.record(syn._ref_rho)
print(f"[STDP-types] gid {gid} · {NPAIR}짝@5Hz · 단일 post · ca_stp0 · rho0={RHO0} · 6유형", flush=True)

out = {"npair": NPAIR, "rho0": RHO0, "dts": DTS, "types": []}
for name, label, cpre, cpost, dcal, thp, thd in TYPES:
    syn.C_pre = cpre; syn.C_post = cpost; syn.D = dcal; syn.theta_p = thp; syn.theta_d = thd
    print(f"[{name}] {label} · C_pre {cpre} C_post {cpost} D {dcal} θp {thp}", flush=True)
    res = []
    for dt in DTS:
        for kk, pt in enumerate(pre_times):
            ics[kk].delay = pt + dt          # 단일 post 스파이크 @ pre+Δt
        tstop = pre_times[-1] + dt + 200.0
        syn.rho0 = RHO0
        hh.dt = 0.025; hh.finitialize(-70); hh.continuerun(tstop)
        rho = np.array(rho_v)
        res.append(dict(dt=dt, rho0=float(rho[0]), rho_end=float(rho[-1]), drho=float(rho[-1]-rho[0])))
        print(f"  Δt={dt:+4d} → rho {rho[0]:.3f}→{rho[-1]:.3f} (Δ{rho[-1]-rho[0]:+.3f})", flush=True)
    out["types"].append(dict(name=name, label=label, cpre=cpre, cpost=cpost, D=dcal, theta_p=thp, theta_d=thd, res=res))

p = os.path.join(ROOT, "scratch", "stdp_types.json")
json.dump(out, open(p, "w"))
print("저장:", p, flush=True)
