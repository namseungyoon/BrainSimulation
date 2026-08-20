# -*- coding: utf-8 -*-
"""
03_network/3_run/mpi_wire_full.py  —  전체 MPI 배선 (SC + 내부, 후시냅스 point 방식)

5,610 세포를 MPI 분배 조립 + SC(GBPlasticityStpProbSyn) + 내부(E=GBPlasticity, I=ProbGABAAB_EMS)
시냅스를 전부 부착하고 구조 메타데이터(시냅스수·메모리·시간)를 낸다. 물리회로 완성 검증.
- SC: 섬유 VecStim(gid_connect) · 후시냅스 point 직접자극 대비
- 내부: pre 소마 스파이크 pc.gid_connect · 타깃구획(pre mtype) 샘플
- 가소성 동결(gamma=0) — STP 거동. 자극/구동은 별도.

실행: LD_LIBRARY_PATH=$CONDA_PREFIX/lib mpirun -np 5 python 03_network/3_run/mpi_wire_full.py [--ncell 5610]
"""
import os
import sys
import time
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
DERIVED = os.path.join(ROOT, "data", "derived")
CFG = os.path.join(ROOT, "config", "synapse_rules.json")
FIBER_OFFSET = 10_000_000

NCELL = int(sys.argv[sys.argv.index("--ncell") + 1]) if "--ncell" in sys.argv else 5610

SC2 = {"SP_CCKBC", "SR_SCA", "SLM_PPA", "SP_Ivy"}
SC3 = {"SP_PVBC", "SP_AA", "SP_BS", "SO_OLM", "SO_Tri", "SO_BS", "SO_BP"}
PERI = {"SP_PVBC", "SP_CCKBC", "SP_AA"}; SLM = {"SO_OLM", "SO_BS", "SO_BP", "SLM_PPA"}; SR = {"SR_SCA"}
GSYN_SC = 0.8


def sc_stp(mt):
    if mt == "SP_PC":
        return 0.14, 186.0, 129.0, 12
    if mt in SC2:
        return 0.11, 307.0, 195.0, 4
    return 0.11, 307.0, 195.0, 8


def target_comp(pre_mt):
    if pre_mt in PERI:
        return "peri"
    if pre_mt in SLM:
        return "slm"
    if pre_mt in SR:
        return "sr"
    return "dend"


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


def compartments(cell, XYZg, rot, seed, radial):
    secs, xs, P, soma = [], [], [], []
    for sec in cell.all:
        nm = sec.name(); n = int(sec.n3d())
        if n < 2 or ("axon" in nm) or ("node" in nm) or ("myelin" in nm):
            continue
        Lt = sec.arc3d(n - 1) or 1.0
        so = "soma" in nm
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            secs.append(sec); xs.append(min(max(sec.arc3d(i) / Lt, 0.0), 1.0)); soma.append(so)
    P = np.asarray(P, float)
    r = (XYZg + rot.apply(P) - seed) @ radial
    soma = np.asarray(soma); idx = np.arange(len(P)); dend = idx[~soma]; rd = r[dend]
    comp = {"soma": idx[soma], "dend": dend,
            "peri": np.concatenate([idx[soma], dend[np.abs(rd) < 60]]),
            "sr": dend[(rd >= 25) & (rd <= 450)], "slm": dend[rd > 450]}
    return secs, xs, comp


def main():
    h.nrnmpi_init(); pc = h.ParallelContext()
    rank, nhost = int(pc.id()), int(pc.nhost())
    import net_build as nb
    B = nb.NetBuilder(); t0 = time.time()
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]; mt = B.mt
    cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
    fr = cfg["frame_um"]; seed = np.array(fr["seed"])
    Mrows = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    rules = {r["id"]: r for r in json.load(open(CFG, encoding="utf-8"))["internal_rules"]}
    sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    fib = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    scpost = sc["post_gid"]; scxyz = sc["xyz"].astype(float); fiber_id = fib["fiber_id"]
    di = np.load(os.path.join(DERIVED, "synapses_internal.npz"), allow_pickle=True)
    ipre = di["pre_gid"]; ipost = di["post_gid"]; insyn = di["n_syn"]
    p = np.load(os.path.join(DERIVED, "synapse_params.npz"), allow_pickle=True)
    irule = p["internal_rule"]; igsyn = p["internal_gsyn"]; imech = p["internal_mech"].astype(str)

    gids = np.arange(min(NCELL, len(mt)))
    mine = set(int(g) for g in gids if g % nhost == rank)
    keep = []

    # 1) 세포 조립 + gid 등록
    for g in mine:
        cell = B.build_cell(g); pc.set_gid2node(g, rank)
        nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0]); nc.threshold = -10
        pc.cell(g, nc); keep.append(nc)
    pc.barrier()
    if rank == 0:
        print(f"[조립] {len(gids)}세포 · {nhost}랭크 · {time.time()-t0:.0f}s · RSS {nb.rss_mb():.0f}MB", flush=True)

    # 2) 섬유 VecStim 등록 (fid%nhost)
    for fidv in set(int(f) for f in np.unique(fiber_id) if int(f) % nhost == rank):
        pc.set_gid2node(FIBER_OFFSET + fidv, rank)
        vs = h.VecStim(); nc = h.NetCon(vs, None); pc.cell(FIBER_OFFSET + fidv, nc); keep.append((vs, nc))

    # 3) SC 배선 (post 소유)
    scsel = np.where(np.isin(scpost, list(mine)))[0]
    n_sc = 0
    cache_tree = {}
    for g in mine:
        cell = B.cells[g]; tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[g][[1, 2, 3, 0]]); soma = cell.soma[0]
        for si in scsel[scpost[scsel] == g]:
            mp = rot.inv().apply(scxyz[si] - XYZ[g]); _, k = tree.query(mp, k=1); sec, x = ref[k]
            syn = h.GBPlasticityStpProbSyn(sec(x)); U, D, Fa, Nr = sc_stp(mt[g])
            syn.Use = U; syn.Dep = D; syn.Fac = Fa; syn.Nrrp = Nr; syn.gmax = GSYN_SC / 1000.0
            syn.gamma_p = 0.0; syn.gamma_d = 0.0; syn.setRNG(g + 1, int(si) + 1, 3)
            ncp = pc.gid_connect(FIBER_OFFSET + int(fiber_id[si]), syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
            ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0
            keep.append((syn, ncp, ncs)); n_sc += 1
    pc.barrier()
    tot_sc0 = int(pc.allreduce(n_sc, 1))   # ⚠️ 집합통신은 전 랭크 호출 필수 (if rank==0 안에 두면 데드락)
    if rank == 0:
        print(f"[SC 배선] {tot_sc0:,}개 · {time.time()-t0:.0f}s · RSS {nb.rss_mb():.0f}MB", flush=True)

    # 4) 내부 배선 (post 소유)
    isel = np.where(np.isin(ipost, list(mine)))[0]
    rng = np.random.default_rng(0); compcache = {}; n_int = 0
    NG = len(gids)
    for ci in isel:
        pg, qg, ns = int(ipre[ci]), int(ipost[ci]), int(insyn[ci])
        if pg >= NG:   # subset 테스트: 미빌드 pre 연결 방지(전체 실행 시 무영향)
            continue
        if qg not in compcache:
            compcache[qg] = compartments(B.cells[qg], XYZ[qg], Rot.from_quat(Q[qg][[1, 2, 3, 0]]), seed, Mrows[:, 1])
        secs, xs, comp = compcache[qg]
        pool = comp[target_comp(mt[pg])]
        if len(pool) == 0:
            pool = comp["dend"] if len(comp["dend"]) else comp["soma"]
        if len(pool) == 0:
            continue
        rl = rules.get(int(irule[ci])); gs = float(igsyn[ci]); mech = imech[ci]
        soma_q = B.cells[qg].soma[0]
        for k in pool[rng.integers(0, len(pool), ns)]:
            sec, x = secs[k], xs[k]
            if mech == "E":
                syn = h.GBPlasticityStpProbSyn(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
                    try: syn.NMDA_ratio = rl["nmda_ampa"]
                    except Exception: pass
                syn.gmax = gs / 1000.0; syn.gamma_p = 0.0; syn.gamma_d = 0.0
                syn.setRNG(qg + 1, 900000 + ci, 7)
                ncp = pc.gid_connect(pg, syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
                ncs = h.NetCon(soma_q(0.5)._ref_v, syn, sec=soma_q); ncs.weight[0] = -1.0
                keep.append((syn, ncp, ncs))
            else:
                syn = h.ProbGABAAB_EMS(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
                ncp = pc.gid_connect(pg, syn); ncp.weight[0] = gs / 1000.0; ncp.delay = 1.0
                keep.append((syn, ncp))
            n_int += 1
    pc.barrier()
    tot_sc = int(pc.allreduce(n_sc, 1)); tot_int = int(pc.allreduce(n_int, 1)); rss = pc.allreduce(nb.rss_mb(), 1)
    if rank == 0:
        print(f"\n[전체 배선 완료] SC {tot_sc:,} + 내부 {tot_int:,} = {tot_sc+tot_int:,} 시냅스", flush=True)
        print(f"[자원] 총 RSS {rss:,.0f}MB ({nhost}랭크 합) · 총 {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}분)", flush=True)
        json.dump({"ncell": len(gids), "sc": tot_sc, "internal": tot_int, "rss_mb": rss, "sec": time.time() - t0},
                  open(os.path.join(ROOT, "scratch", "mpi_wire_full.json"), "w"))
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
