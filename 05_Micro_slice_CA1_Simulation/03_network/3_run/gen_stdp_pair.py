# -*- coding: utf-8 -*-
"""나-1-② STDP — SC→PC 쌍에 Graupner-Brunel 칼슘 모델 켜고 Δt 스윕 → ρ 변화(STDP 곡선).
칼슘/효능 기본값 = Wittenberg 2006 fit(mod 기본). gamma_p/gamma_d 켬(STP는 0으로 동결했던 것),
ca_stp=0(보수적, Graupner 원본). pre=VecStim, post=IClamp 강제발화 → sentinel NetCon(weight<0).
실행: gen_stdp_pair.py [--npair 60] [--interval 1000] [--test]
"""
import os, sys, json, time, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib")); import net_build as nb
DERIVED = os.path.join(ROOT, "data", "derived")
def arg(f, d): return type(d)(sys.argv[sys.argv.index(f)+1]) if f in sys.argv else d
NPAIR = int(arg("--npair", 30)); INTERVAL = float(arg("--interval", 200.0)); TEST = "--test" in sys.argv   # 유도 반복 30회 @ 5Hz
NPOST = int(arg("--npost", 4)); POSTISI = float(arg("--postisi", 10.0))   # post-burst: 4발@100Hz (Wittenberg: burst 필요)
TAG = arg("--tag", "")                    # 출력파일 구분 접미사 (실험2/3 다회 실행 시 덮어쓰기 방지)
SETTLE = 300.0
GAMMA_P, GAMMA_D = 1645.59, 313.0965      # mod 기본값(Wittenberg fit) — 켬
# ca_stp 비교: 0=칼슘 고정(Graupner 원본·보수적), 1=칼슘이 단기방출 추종. --ca_stp 0|1 로 단일 지정 가능.
CA_STPS = [int(arg("--ca_stp", -1))] if "--ca_stp" in sys.argv else [0, 1]
# --dt 로 Δt 하나 고정(실험2 burst수·실험3 주파수 스윕용) / 없으면 전체 Δt 훑음(실험1 타이밍곡선).
if "--dt" in sys.argv:
    DTS = [int(arg("--dt", 10))]
elif TEST:
    DTS = [10, -10]
else:
    DTS = [-100, -50, -30, -20, -10, -5, 5, 10, 20, 30, 50, 100]

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
cell = B.build_cell(gid); tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[gid][[1,2,3,0]]); soma = cell.soma[0]
si = int(sidx[0]); mp = rot.inv().apply(wxyz[si] - XYZ[gid]); _, k = tree.query(mp, k=1); sec, x = ref[k]
syn = hh.GBPlasticityStpProbSyn(sec(x))
syn.Use = 0.14; syn.Dep = 186; syn.Fac = 129; syn.Nrrp = 12; syn.gmax = 0.8/1000.0
syn.gamma_p = GAMMA_P; syn.gamma_d = GAMMA_D; syn.rho0 = 0.0   # ca_stp는 아래 비교 루프에서 설정
syn.setRNG(gid+1, si+1, 3)
vs = hh.VecStim(); ncpre = hh.NetCon(vs, syn); ncpre.weight[0] = 1.0; ncpre.delay = 0.5
ncpost = hh.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncpost.weight[0] = -1.0; ncpost.threshold = -10.0
pre_times = [SETTLE + kk*INTERVAL for kk in range(NPAIR)]
playv = hh.Vector(pre_times); vs.play(playv)
ics = [hh.IClamp(soma(0.5)) for _ in range(NPAIR*NPOST)]
for ic in ics: ic.dur = 2.0; ic.amp = 2.0
rho_v = hh.Vector(); rho_v.record(syn._ref_rho)
apc = hh.APCount(soma(0.5)); apc.thresh = -10.0
print(f"[STDP] gid {gid} · pairings {NPAIR} @ {1000/INTERVAL:.1f}Hz · post-burst {NPOST}발@{1000/POSTISI:.0f}Hz · gamma_p={GAMMA_P} · ca_stp 비교 {CA_STPS}", flush=True)

res = []
for cs in CA_STPS:                                             # ca_stp 0(고정) vs 1(방출추종) 비교
    syn.ca_stp = cs
    print(f"[ca_stp={cs}] {'칼슘 고정(Graupner 원본)' if cs==0 else '칼슘이 단기방출 추종'}", flush=True)
    for dt in DTS:
        for kk, pt in enumerate(pre_times):                    # post-burst = pre + Δt, NPOST발
            for j in range(NPOST):
                ics[kk*NPOST + j].delay = pt + dt + j*POSTISI
        tstop = pre_times[-1] + dt + (NPOST-1)*POSTISI + 200.0
        t0 = time.time(); hh.dt = 0.025; hh.finitialize(-70); hh.continuerun(tstop)
        rho = np.array(rho_v)
        res.append(dict(ca_stp=cs, dt=dt, rho0=float(rho[0]), rho_end=float(rho[-1]), drho=float(rho[-1]-rho[0]), napc=int(apc.n)))
        print(f"  Δt={dt:+4d}ms → rho {rho[0]:.3f}→{rho[-1]:.3f} (Δ{rho[-1]-rho[0]:+.3f}) · post발화 {int(apc.n)}/{NPAIR} · {time.time()-t0:.0f}s", flush=True)

suffix = TAG if TAG else ("_test" if TEST else "")
out = os.path.join(ROOT, "scratch", "stdp_pair" + suffix + ".json")
json.dump({"gid": gid, "npair": NPAIR, "interval": INTERVAL, "npost": NPOST, "postisi": POSTISI,
           "gamma_p": GAMMA_P, "gamma_d": GAMMA_D, "ca_stps": CA_STPS, "dts": DTS, "res": res}, open(out, "w"))
print("저장:", out, flush=True)
