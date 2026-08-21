# -*- coding: utf-8 -*-
"""
03_network/3_run/mpi_smoke.py  —  MPI 구동 확인 (E0 smoke test)

목적: MPI 분산 조립 + SC 배선(GBPlasticityStpProbSyn) + SC 자극 구동이 실제로
돌아가고 발화가 나오는지 확인(소수 세포). 배선 mechanism 전환 검증 포함.
- 세포: gid % nhost 로 랭크 분배, pc.cell로 gid 등록
- SC 시냅스: GBPlasticityStpProbSyn (STP+가소성 겸용) · setRNG(gid) 필수 · NetCon 2개
  (pre 섬유 gid_connect weight>0, post 스파이크 sentinel weight<0)
- 섬유 VecStim: fid%nhost 랭크에 gid 등록 → E3 반경 섬유만 t=STIM에 발화
- 가소성 동결(gamma=0) → E1~E6/E9식 STP 거동
실행: LD_LIBRARY_PATH=$CONDA_PREFIX/lib mpirun -np 6 python 03_network/3_run/mpi_smoke.py [--ncell 300] [--r 200]
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
FIBER_OFFSET = 10_000_000
SETTLE = 200.0     # 안정화 구간(거친 dt)
STIM_T = SETTLE + 10.0   # 자극 시각(ms)
WINDOW = 80.0      # 자극 후 관찰창
TSTOP = SETTLE + WINDOW


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d

NCELL = arg("--ncell", 300)
RADIUS = arg("--r", 200.0)
USE_GPU = "--gpu" in sys.argv   # 스모크를 CoreNEURON online GPU로 (소규모라 OOM 없음)

# post mtype → SC STP 규칙
SC2 = {"SP_CCKBC", "SR_SCA", "SLM_PPA", "SP_Ivy"}
SC3 = {"SP_PVBC", "SP_AA", "SP_BS", "SO_OLM", "SO_Tri", "SO_BS", "SO_BP"}


def sc_stp(mt):
    if mt == "SP_PC":
        return 0.14, 186.0, 129.0, 12
    if mt in SC2:
        return 0.11, 307.0, 195.0, 4
    return 0.11, 307.0, 195.0, 8


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
    h.nrnmpi_init()
    pc = h.ParallelContext()
    rank, nhost = int(pc.id()), int(pc.nhost())
    import net_build as nb
    B = nb.NetBuilder()            # h, stdrun, mechanism, window_cells, templates
    t0 = time.time()

    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]; mt = B.mt
    d = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    fib = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    post = d["post_gid"]; wxyz = d["xyz"].astype(float); dist_e3 = d["dist_e3"].astype(float)
    fiber_id = fib["fiber_id"]

    gids = np.arange(min(NCELL, len(mt)))
    mine = [int(g) for g in gids if g % nhost == rank]

    # 1) 세포 조립 + gid 등록
    keep = []
    for g in mine:
        cell = B.build_cell(g)
        pc.set_gid2node(g, rank)
        ncout = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0])
        ncout.threshold = -10
        pc.cell(g, ncout); keep.append(ncout)
    pc.barrier()
    if rank == 0:
        print(f"[조립] {len(gids)}세포 분배({nhost}랭크) · {time.time()-t0:.1f}s", flush=True)

    # 2) 섬유 VecStim 등록 (fid%nhost) — E3 반경 섬유만 자극
    driven = set(np.unique(fiber_id[dist_e3 < RADIUS]).tolist())
    my_fibers = set(int(f) for f in np.unique(fiber_id) if int(f) % nhost == rank)
    vstore = []
    for fid in my_fibers:
        fgid = FIBER_OFFSET + fid
        pc.set_gid2node(fgid, rank)
        vs = h.VecStim()
        tv = h.Vector([STIM_T] if fid in driven else [])
        vs.play(tv)
        nc = h.NetCon(vs, None)
        pc.cell(fgid, nc); vstore.append((vs, tv, nc))

    # 3) SC 시냅스 부착 (post 소유 랭크에서)
    sel = np.isin(post, mine)
    idx = np.where(sel)[0]
    nsyn = 0
    for g in mine:
        cell = B.cells[g]
        tree, ref = seg_kdtree(cell)
        rot = Rot.from_quat(Q[g][[1, 2, 3, 0]])
        soma = cell.soma[0]
        for si in idx[post[idx] == g]:
            mp = rot.inv().apply(wxyz[si] - XYZ[g])
            _, k = tree.query(mp, k=1)
            sec, x = ref[k]
            syn = h.GBPlasticityStpProbSyn(sec(x))
            U, De, Fa, Nr = sc_stp(mt[g])
            syn.Use = U; syn.Dep = De; syn.Fac = Fa; syn.Nrrp = Nr
            syn.gmax = 0.8 / 1000.0                          # SC gsyn 0.8nS = gmax(µS)
            syn.gamma_p = 0.0; syn.gamma_d = 0.0            # 가소성 동결(STP만)
            syn.setRNG(int(g) + 1, int(si) + 1, 3)          # gid 결정론적 시딩
            fid = int(fiber_id[si])
            nc_pre = pc.gid_connect(FIBER_OFFSET + fid, syn)
            nc_pre.weight[0] = 1.0; nc_pre.delay = 1.0       # weight=무차원 배율
            nc_post = h.NetCon(soma(0.5)._ref_v, syn, sec=soma)
            nc_post.weight[0] = -1.0; nc_post.delay = 0.0         # post sentinel
            keep.append((syn, nc_pre, nc_post)); nsyn += 1
    pc.barrier()
    tot_syn = int(pc.allreduce(nsyn, 1))
    if rank == 0:
        print(f"[SC배선] 총 {tot_syn:,}개(GBPlasticityStpProbSyn) · 자극섬유 {len(driven):,} · {time.time()-t0:.1f}s", flush=True)

    # 4) 발화 기록 — 전체 스파이크(gid,시각) + 대표세포 전압파형
    tspk = h.Vector(); idspk = h.Vector()
    pc.spike_record(-1, tspk, idspk)
    # 대표세포 = E3 근처 SC 많은 추체
    pc_near = (mt[post] == "SP_PC") & (dist_e3 < 150) & (post < len(gids))
    target = int(np.bincount(post[pc_near], minlength=len(gids)).argmax()) if pc_near.any() else int(mine[0])
    rec = None
    if target in mine:
        rec = (h.Vector(), h.Vector())
        rec[0].record(h._ref_t); rec[1].record(B.cells[target].soma[0](0.5)._ref_v)

    # 5) 구동 — 안정화(거친 dt) → 자극창(고정 dt 0.025)
    pc.set_maxstep(10)
    if USE_GPU:
        from neuron import coreneuron
        coreneuron.enable = True; coreneuron.gpu = True
        h.finitialize(-70); trun = time.time()
        h.dt = 0.025; pc.psolve(TSTOP)     # 균일 dt 단일 psolve (online GPU)
        psolve_s = time.time() - trun
    else:
        h.finitialize(-70)
        trun = time.time()
        h.dt = 0.25
        pc.psolve(SETTLE)
        h.dt = 0.025
        pc.psolve(TSTOP)
        psolve_s = time.time() - trun

    # 6) 스파이크 gather → rank0 저장·시각화 데이터
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    all_t = comm.gather(list(tspk), root=0)
    all_id = comm.gather(list(idspk), root=0)
    trace = None
    if rec is not None:
        trace = (np.array(rec[0]), np.array(rec[1]))
    traces = comm.gather(trace, root=0)

    if rank == 0:
        st = np.concatenate([np.array(x) for x in all_t]) if any(len(x) for x in all_t) else np.array([])
        sid = np.concatenate([np.array(x) for x in all_id]) if any(len(x) for x in all_id) else np.array([])
        cells = np.array([g for g in gids])
        fired = np.array([g in set(sid.astype(int)) for g in cells])
        # 세포 국소좌표 + E3 + 자극 시냅스
        cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
        fr = cfg["frame_um"]; seed = np.array(fr["seed"])
        Mrows = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
        cell_uvw = (XYZ[cells] - seed) @ Mrows
        e3loc = (d["e3_xyz"] - seed) @ Mrows
        # 자극받은 시냅스(구동섬유 소속 + post가 창 안) 위치
        driven_syn = np.isin(fiber_id, list(driven)) & np.isin(post, cells)
        suvw = d["uvw"].astype(float)[driven_syn]
        tr = next((x for x in traces if x is not None), None)
        np.savez_compressed(os.path.join(ROOT, "scratch", "mpi_smoke_viz.npz"),
                            cell_uvw=cell_uvw, fired=fired, cell_gid=cells,
                            spk_t=st, spk_id=sid, e3=e3loc, stim_uvw=suvw.astype(np.float32),
                            radius=RADIUS, stim_t=STIM_T, settle=SETTLE,
                            trace_t=tr[0] if tr is not None else np.array([]),
                            trace_v=tr[1] if tr is not None else np.array([]),
                            target=target)
        print(f"\n[구동 완료] psolve {psolve_s:.1f}s", flush=True)
        print(f"[결과] 총 스파이크 {len(st):,} · 발화세포 {int(fired.sum())}/{len(cells)} "
              f"({100*fired.mean():.0f}%)", flush=True)
        json.dump({"ncell": len(gids), "nsyn": tot_syn, "spikes": int(len(st)),
                   "active": int(fired.sum()), "radius": RADIUS},
                  open(os.path.join(ROOT, "scratch", "mpi_smoke.json"), "w"))
        print(f"[시각화 데이터] scratch/mpi_smoke_viz.npz · [총시간] {time.time()-t0:.1f}s", flush=True)
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
