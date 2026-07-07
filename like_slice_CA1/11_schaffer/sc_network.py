# -*- coding: utf-8 -*-
"""
11_schaffer/sc_network.py  —  E2-2: 슬라이스에 Schaffer collateral(SC) 경로 배선 + 볼리 자극

조용한 in vitro 슬라이스(외부 Poisson 구동 OFF) + CA3 SC 입력:
  - N_FIBER개의 CA3 축삭(NetStim fiber)을 만들고, 각 세포 수상돌기에 SC 흥분성 시냅스(Ecker E2)를
    무작위 fiber에 연결. PC는 apical(SR), 인터뉴런은 임의 수상돌기(피드포워드 억제 담당).
  - 볼리: 활성 fiber 비율(sc_active)만 t=stim_t 에 1회 동기 발화 → 유발 반응.
  - 측정: 자극 후 발화한 세포 수(=Romani Fig4 I-O 지표) + baseline 정적 확인.

이 스크립트가 E3(I-O + gabazine)의 토대. --no_inh 로 억제연결 차단(gabazine 모사) 지원.
실행: mpiexec -n 10 <python> 11_schaffer/sc_network.py --counts 1000,110,70,70 --sc_active 1.0 --stim_t 100 --tstop 200
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
N_FIBER = 100          # CA3 SC 축삭 수(볼리 단위)
SC_PER_CELL = 12       # 세포당 SC 시냅스 수
SC_CLASS = "PC->PC (E2)"   # SC 흥분성 시냅스로 Ecker E2 AMPA/NMDA 재사용


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def log(m):
    if RANK == 0:
        print(m, flush=True)


def sr_or_dend(cell, is_pc, rng):
    """PC면 apical(SR) 우선, 인터뉴런이면 임의 수상돌기."""
    if is_pc:
        segs = [s for s in cell.all if ".apic" in s.name()]
    else:
        segs = []
    if not segs:
        segs = [s for s in cell.all if (".dend" in s.name() or ".apic" in s.name())]
    if not segs:
        return cell.soma[0](0.5)
    return segs[rng.randint(len(segs))](0.5)


def main():
    counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, argval("--counts", "1000,110,70,70").split(","))))
    sc_active = float(argval("--sc_active", "1.0"))
    stim_t = float(argval("--stim_t", "100"))
    tstop = float(argval("--tstop", "200"))
    no_inh = "--no_inh" in sys.argv      # gabazine 모사(억제 연결 차단)
    sc_g = float(argval("--sc_g", str(P3.CLASSES[SC_CLASS]["g_nS"])))
    sc_per_cell = int(argval("--sc_per_cell", str(SC_PER_CELL)))  # 세포당 SC 시냅스 수(볼리 세기)

    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])
    ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
    keep = []
    for tn, k in counts.items():
        ids = np.where(t4 == tn)[0]; keep.extend(ids[np.argsort(dist[ids])[:k]].tolist())
    keep = np.array(sorted(keep)); N = len(keep)
    orig2gid = {int(o): g for g, o in enumerate(keep)}
    gtype = [t4[o] for o in keep]
    log(f"[E2-2] {N}세포 · SC fiber {N_FIBER}(활성 {sc_active:.0%}) · 자극 t={stim_t}ms · "
        f"no_inh(gabazine)={no_inh} · SC g={sc_g}nS")

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
    pc.barrier(); log("[구축] 완료 (외부 Poisson 구동 OFF = 조용한 슬라이스)")

    # 내재 커넥텀 (no_inh 면 억제 연결 skip = gabazine)
    p = np.load(PRUNED, allow_pickle=True)
    pre = p["pre"]; post = p["post"]; cid = p["cls"]; classes = list(p["classes"].astype(str))
    inh_cls = set(i for i, cl in enumerate(classes) if not cl.startswith("PC->"))
    rng = np.random.RandomState(1000 + RANK); n_syn = 0; n_skip_inh = 0
    for i in range(len(pre)):
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:
            continue
        if no_inh and int(cid[i]) in inh_cls:
            n_skip_inh += 1; continue
        ga = orig2gid[a]; clsn = classes[int(cid[i])]
        try:
            pr = P3.CLASSES[clsn]; seg = net._placement(cells[gb], clsn, rng)
            syn = build_synapse(seg, pr, seeds=(i + 1, 1, 1), deterministic=False)
            ncc = pc.gid_connect(ga, syn); ncc.threshold = -20.0
            ncc.weight[0] = pr["g_nS"]; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_syn += 1
        except Exception:
            pass
    log(f"[내재연결] 랭크0 {n_syn} 시냅스" + (f" (억제 {n_skip_inh} 차단=gabazine)" if no_inh else ""))

    # SC fiber(NetStim) — 전 랭크 동일 패턴(볼리 동기). 활성 fiber = 앞쪽 int(sc_active*N)
    n_act = int(round(sc_active * N_FIBER))
    fibers = []
    for k in range(N_FIBER):
        ns = h.NetStim(); ns.number = 1 if k < n_act else 0
        ns.start = stim_t; ns.interval = 1; ns.noise = 0
        fibers.append(ns); keeph.append(ns)

    # 각 세포에 SC 시냅스 SC_PER_CELL개 → 무작위 fiber 연결(PC는 SR)
    frng = np.random.RandomState(7)   # fiber 배정용(전 랭크 동일 시드 아님, 세포별 무관)
    prm = P3.CLASSES[SC_CLASS]; n_sc = 0
    for g in my:
        is_pc = gtype[g] == "PC"
        for _ in range(sc_per_cell):
            seg = sr_or_dend(cells[g], is_pc, rng)
            syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000, 1, 1), deterministic=False)
            fib = fibers[rng.randint(N_FIBER)]
            ncc = h.NetCon(fib, syn); ncc.weight[0] = sc_g; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_sc += 1
    n_sc_all = int(pc.allreduce(n_sc, 1))
    log(f"[SC배선] 총 {n_sc_all} SC 시냅스 ({sc_per_cell}/세포, {N_FIBER} fiber 중 {n_act} 활성)")

    tvec = h.Vector(); gidvec = h.Vector(); pc.spike_record(-1, tvec, gidvec)
    is_pc_arr = np.array([gt == "PC" for gt in gtype])
    n_pc = int(pc.allreduce(sum(1 for g in my if gtype[g] == "PC"), 1))
    h.celsius = 34.0; h.cvode_active(0); h.dt = net.DT; pc.set_maxstep(10)
    log(f"[실행] tstop={tstop}ms (자극 t={stim_t}) …")
    h.finitialize(-70.0); pc.psolve(tstop)

    tt = np.array(tvec.to_python()); gg = np.array(gidvec.to_python(), dtype=int)
    # baseline(자극 전) vs 유발(자극 후 50ms) PC 발화
    pre_m = tt < stim_t
    post_m = (tt >= stim_t) & (tt < stim_t + 50)
    pre_pc = sum(1 for gi in gg[pre_m] if is_pc_arr[gi])
    post_pc = sum(1 for gi in gg[post_m] if is_pc_arr[gi])
    fired_post = len(set(int(gi) for gi in gg[post_m] if is_pc_arr[gi]))
    pre_pc = int(pc.allreduce(pre_pc, 1)); post_pc = int(pc.allreduce(post_pc, 1))
    fired_post = int(pc.allreduce(fired_post, 1))
    if RANK == 0:
        print("\n===== E2-2 결과 =====", flush=True)
        print(f"  baseline(자극 전 {stim_t:.0f}ms): PC 스파이크 {pre_pc} → "
              f"{'조용함 OK' if pre_pc < n_pc*0.05 else '⚠️ baseline 활동 있음'}", flush=True)
        print(f"  유발(자극 후 50ms): PC 스파이크 {post_pc}, 발화 PC {fired_post}/{n_pc} "
              f"({100*fired_post/max(1,n_pc):.0f}%)", flush=True)
        print(f"  → SC 볼리가 조용한 슬라이스에서 PC 반응 유발: "
              f"{'성공' if fired_post > pre_pc else '반응 약함'}", flush=True)
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
