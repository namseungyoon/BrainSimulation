# -*- coding: utf-8 -*-
"""
10_analysis/drive_sweep.py  —  E1 튜닝: 외부 Poisson 구동 강도 스윕 → 생리적 PC 발화율 찾기

세포를 **한 번만 구축**(비싼 단계)하고, 구동 weight 배율(drive_scale)만 바꿔가며
짧은 창을 재실행해 PC 발화율을 측정 → PC ~1Hz(in vivo)를 주는 배율 탐색.

⚠️ 속도 위해 공간 국소 부분집합 사용 → 전체보다 억제가 약함(같은 구동서 PC 다소 높게 나옴).
   따라서 여기서 찾은 배율은 전체 슬라이스의 근사 하한. 이후 전체서 재확인.

실행: mpiexec -n 10 <python> 10_analysis/drive_sweep.py --counts 600,70,45,45 --win 100 --t0 40
"""
import os
import sys
import numpy as np
from neuron import h

h.nrnmpi_init()
pc = h.ParallelContext(); RANK = int(pc.id()); NHOST = int(pc.nhost())
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT); SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED); sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, os.path.join(PAPER, "04_network"))
import network_lib as net
from common.cell_loader import load_cell
from synapse_pair import build_synapse
import params_table3 as P3
MODELS = os.path.join(SHARED, "models")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
SCALES = [0.30, 0.15, 0.08, 0.04, 0.02]


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def log(m):
    if RANK == 0:
        print(m, flush=True)


def main():
    counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, argval("--counts", "600,70,45,45").split(","))))
    win = float(argval("--win", "100")); t0 = float(argval("--t0", "40"))
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str); mtype = c["mtype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])
    ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
    keep = []
    for tn, k in counts.items():
        ids = np.where(t4 == tn)[0]; keep.extend(ids[np.argsort(dist[ids])[:k]].tolist())
    keep = np.array(sorted(keep)); N = len(keep)
    orig2gid = {int(o): g for g, o in enumerate(keep)}
    gtype = [t4[o] for o in keep]; gmt = [mtype[o] for o in keep]
    log(f"[스윕] {N}세포 · 창 {win}ms(측정 {t0}-{win}) · 배율 {SCALES}")

    type_dir = net.load_representatives(MODELS)
    my = [g for g in range(N) if g % NHOST == RANK]
    cells = {}; keeph = []
    for g in my:
        cell, _ = load_cell(type_dir[gtype[g]], gid=g)
        for sec in cell.all:
            sec.nseg = 1
        cells[g] = cell
        s = cell.soma[0]; nc = h.NetCon(s(0.5)._ref_v, None, sec=s); nc.threshold = -20.0
        pc.set_gid2node(g, RANK); pc.cell(g, nc); keeph.append(nc)
    pc.barrier(); log("[구축] 완료")

    p = np.load(PRUNED, allow_pickle=True)
    pre = p["pre"]; post = p["post"]; cid = p["cls"]; classes = list(p["classes"].astype(str))
    rng = np.random.RandomState(1000 + RANK); n_syn = 0
    for i in range(len(pre)):
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:
            continue
        ga = orig2gid[a]; clsn = classes[int(cid[i])]
        try:
            pr = P3.CLASSES[clsn]; seg = net._placement(cells[gb], clsn, rng)
            syn = build_synapse(seg, pr, seeds=(i + 1, 1, 1), deterministic=False)
            ncc = pc.gid_connect(ga, syn); ncc.threshold = -20.0
            ncc.weight[0] = pr["g_nS"]; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_syn += 1
        except Exception:
            pass
    log(f"[연결] 랭크0 {n_syn} 시냅스")

    # 외부구동 — base weight 보관(스윕서 배율만 변경)
    drive = []  # (ncd, base_w)
    for g in my:
        n_stim, w = net.DRIVE[gtype[g]]
        for j in range(n_stim):
            ns = h.NetStim(); ns.interval = 1000.0 / net.DRIVE_RATE; ns.number = 1e9
            ns.start = 0; ns.noise = 1.0
            r = h.Random(); r.Random123(g, j, 0); r.negexp(1); ns.noiseFromRandom(r)
            syn = h.Exp2Syn(cells[g].soma[0](0.5)); syn.tau1 = 0.2; syn.tau2 = 2.0; syn.e = 0.0
            ncd = h.NetCon(ns, syn); ncd.delay = 0.0
            drive.append((ncd, w)); keeph += [ns, r, syn]

    tvec = h.Vector(); gidvec = h.Vector(); pc.spike_record(-1, tvec, gidvec)
    is_pc = np.array([gt == "PC" for gt in gtype])
    n_pc = int(pc.allreduce(sum(1 for g in my if gtype[g] == "PC"), 1))
    n_int = int(pc.allreduce(sum(1 for g in my if gtype[g] != "PC"), 1))
    h.celsius = 34.0; h.cvode_active(0); h.dt = net.DT; pc.set_maxstep(10)

    log("\n===== 구동 배율 스윕 =====")
    log(f"{'scale':>6} | {'PC(Hz)':>8} | {'INT(Hz)':>8}")
    results = []
    for sc in SCALES:
        for ncd, bw in drive:
            ncd.weight[0] = bw * sc
        tvec.resize(0); gidvec.resize(0)
        h.finitialize(-70.0); pc.psolve(win)
        tt = np.array(tvec.to_python()); gg = np.array(gidvec.to_python(), dtype=int)
        m = (tt >= t0) & (tt < win)
        dur = (win - t0) / 1000.0
        pc_sp = 0; int_sp = 0
        for gi in gg[m]:
            if is_pc[gi]:
                pc_sp += 1
            else:
                int_sp += 1
        pc_sp = int(pc.allreduce(pc_sp, 1)); int_sp = int(pc.allreduce(int_sp, 1))
        pc_r = pc_sp / max(1, n_pc) / dur; int_r = int_sp / max(1, n_int) / dur
        log(f"{sc:>6.2f} | {pc_r:>8.2f} | {int_r:>8.2f}")
        results.append((sc, pc_r, int_r))

    if RANK == 0:
        # PC~1Hz에 가장 가까운 배율
        best = min(results, key=lambda x: abs(x[1] - 1.0))
        print(f"\n[제안] PC~1Hz 근접 배율 = {best[0]:.2f} (PC {best[1]:.2f}Hz, INT {best[2]:.2f}Hz)", flush=True)
        print("[주의] 부분집합(억제 약함) 기준 → 전체 슬라이스는 이 배율서 PC가 더 낮을 수 있음. 전체서 재확인 권장.", flush=True)
        np.save(os.path.join(HERE, "figures", "_drive_sweep.npy"), np.array(results))
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
