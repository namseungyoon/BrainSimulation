# -*- coding: utf-8 -*-
"""13_net_fepsp/net_fepsp.py  —  현실 전체-네트워크 fEPSP 엔진 (복제 아님)

순방향 모델(대표 PC 1개 정렬·동기 복제)의 이상화를 벗어나, 실제 네트워크를 구동하고
**전세포의 진짜 막전류**로 MEA fEPSP를 계산한다:
  · 세포: slice_cells.npz 실배치(17,647 또는 서브셋), 4대표 me-model, sec.nseg=1
  · 연결: pruned_connectivity 내부 커넥텀(확률 EMS) + SC(촉진형 "SC->PC (E1s)")
  · 자극: 동기 SC 테스트 볼리(모든 섬유 동시 발화) → 유발 fEPSP (억제·이질·지터 창발)
  · 기하: 각 세포를 quat_wxyz+xyz로 실제 3D 배치(복제 아님). PC PCA로 밴드면(전극)·
          밴드법선(깊이)·SR을 유리면 z=0쪽.
  · fEPSP: rank별 M_rank(24전극 × rank세그) @ I_rank → allreduce (MPI로 메모리 분산)

실행(서브셋 단일):  <ca1sim>/python.exe 13_net_fepsp/net_fepsp.py --counts 300,80,60,60 --tstop 120
실행(전규모 MPI):   mpiexec -n 20 <py> 13_net_fepsp/net_fepsp.py --counts full --tstop 300
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
for p in (SHARED, os.path.join(PAPER, "03_synapses"), os.path.join(PAPER, "04_network"), HERE,
          os.path.join(ROOT, "12_lfp")):
    sys.path.insert(0, p)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
import params_table3 as P3
from synapse_pair import build_synapse
import lfp_calc as L

pc = h.ParallelContext()
RANK = int(pc.id()); NHOST = int(pc.nhost())
MODELS = os.environ.get("MODELS_DIR") or os.path.join(SHARED, "models")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True) if RANK == 0 else None
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
SIG_T, SIG_S, SIG_G, N_IMG = 0.3, 1.5, 0.0, 20
PITCH, R_ON, NCOL, NROW = 200.0, 100.0, 8, 3
Z_GLASS_MARGIN = 20.0                                   # 유리면-최근접소스 간격(µm)


def log(m):
    if RANK == 0:
        print(m, flush=True)


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def quat_to_R(q):
    """quat (w,x,y,z) → 3x3 회전행렬."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def sr_or_dend(cell, is_pc, rng):
    if is_pc:
        segs = [s for s in cell.all if ".apic" in s.name()]
    else:
        segs = []
    if not segs:
        segs = [s for s in cell.all if (".dend" in s.name() or ".apic" in s.name())]
    return (segs[rng.randint(len(segs))] if segs else cell.soma[0])(0.5)


def main():
    t_all = time.time()
    counts_s = argval("--counts", "300,80,60,60")
    tstop = float(argval("--tstop", "120"))
    dt = float(argval("--dt", "0.025")); rec_dt = float(argval("--rec_dt", "0.2"))
    det = "--det" in sys.argv
    sc_class = argval("--sc_class", "SC->PC (E1s)")     # 기본 촉진형 SC
    sc_pc = int(argval("--sc_pc", "60")); sc_int = int(argval("--sc_int", "40"))
    sc_g_pc = float(argval("--sc_g_pc", "1.0")); sc_g_int = float(argval("--sc_g_int", "1.0"))
    n_fiber = int(argval("--n_fiber", "800"))
    no_inh = "--no_inh" in sys.argv
    pc_only = "--pc_only" in sys.argv                   # 검증용(억제 off + PC만 소스)
    seed = int(argval("--seed", "1"))
    stim = [float(x) for x in argval("--stim", "20,70").split(",")]   # 동기 SC 볼리 시각(ms)
    tag = argval("--tag", "sub")

    # ---- 세포 선택 ----
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str); quat = c["quat_wxyz"].astype(float)
    mtype = c["mtype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype]); Ntot = len(xyz)
    if counts_s == "full":
        keep = np.arange(Ntot)
    else:
        counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, counts_s.split(","))))
        ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
        ks = []
        for tn, k in counts.items():
            ids = np.where(t4 == tn)[0]; ks.extend(ids[np.argsort(dist[ids])[:k]].tolist())
        keep = np.array(sorted(ks))
    N = len(keep); orig2gid = {int(o): g for g, o in enumerate(keep)}
    gtype = [t4[o] for o in keep]
    if pc_only:
        no_inh = True
    log(f"[net_fepsp] N={N} · det={det} · SC={sc_class} · stim={stim}ms · tstop={tstop} · no_inh={no_inh} · pc_only={pc_only} · nhost={NHOST}")

    # ---- 전극·기하 좌표계 (PC PCA: 밴드면 + 밴드법선 깊이) ----
    Ppc = xyz[t4 == "PC"]; c0 = Ppc.mean(0)
    U, S, Vt = np.linalg.svd(Ppc - c0, full_matrices=False)
    face_ax = Vt[:2]                                    # 밴드면 2축(전극면)
    depth_ax = Vt[2]                                    # 밴드법선(깊이/방사)
    nd = c["nd"].astype(float)
    if np.corrcoef((xyz - c0) @ depth_ax, nd)[0, 1] < 0:   # 깊이축을 nd 증가(SO→SLM)와 정렬
        depth_ax = -depth_ax
    # 전극: 밴드면에 3x8 배치(조직 위 최대) — PC 면좌표 기준
    facepc = (Ppc - c0) @ face_ax.T
    gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH; gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
    Gx, Gy = np.meshgrid(gx, gy); G0 = np.column_stack([Gx.ravel(), Gy.ravel()]); NELEC = G0.shape[0]
    from scipy.spatial import cKDTree
    tree = cKDTree(facepc); fc = facepc.mean(0); best = (-1, None, 0.0)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Grot = G0 @ Rm.T
        for dxx in np.linspace(-400, 400, 9):
            for dyy in np.linspace(-200, 200, 9):
                E2 = Grot + fc + [dxx, dyy]; on = int((tree.query(E2)[0] < R_ON).sum())
                if on > best[0]:
                    best = (on, E2.copy(), th)
    n_on, E2d, th = best
    over = tree.query(E2d)[0] < R_ON
    log(f"[전극] 3x8 회전{np.rad2deg(th):.0f}° 조직위 {n_on}/24")

    # ---- 네트워크 구축(rank 분산) ----
    type_dir = net.load_representatives(MODELS)
    my = [g for g in range(N) if g % NHOST == RANK]; cells = {}; keeph = []
    t0 = time.time()
    for g in my:
        cell, _ = load_cell(type_dir[gtype[g]], gid=g)
        for sec in cell.all:
            sec.nseg = 1
        cells[g] = cell
        s = cell.soma[0]; nc = h.NetCon(s(0.5)._ref_v, None, sec=s); nc.threshold = -20.0
        pc.set_gid2node(g, RANK); pc.cell(g, nc); keeph.append(nc)
    h.define_shape(); pc.barrier()
    spt = h.Vector(); spg = h.Vector(); pc.spike_record(-1, spt, spg)   # 스파이크 기록(역치하 판정)
    log(f"[1/4 구축] rank0 {len(my)}세포 · {time.time()-t0:.0f}s")

    # ---- 내부 커넥텀(확률/결정) ----
    pr = np.load(PRUNED, allow_pickle=True)
    pre = pr["pre"]; post = pr["post"]; cid = pr["cls"]; classes = list(pr["classes"].astype(str))
    inh_cls = set(i for i, cl in enumerate(classes) if not cl.startswith("PC->"))
    rng = np.random.RandomState(1000 + RANK + seed * 97); t0 = time.time(); n_syn = 0
    for i in range(len(pre)):
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:
            continue
        if no_inh and int(cid[i]) in inh_cls:
            continue
        ga = orig2gid[a]; cls = classes[int(cid[i])]
        try:
            prm = P3.CLASSES[cls]; seg = net._placement(cells[gb], cls, rng)
            syn = build_synapse(seg, prm, seeds=(i + 1 + seed * 100000, 1, 1), deterministic=det)
            nc = pc.gid_connect(ga, syn); nc.threshold = -20.0
            nc.weight[0] = prm["g_nS"]; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_syn += 1
        except Exception:
            pass
    n_syn_all = int(pc.allreduce(n_syn, 1)); pc.barrier()
    log(f"[2/4 내부연결] {n_syn_all:,} 시냅스 · {time.time()-t0:.0f}s" + (" (억제off)" if no_inh else ""))

    # ---- 동기 SC 볼리(모든 섬유 stim 시각 발화) + SC 시냅스(촉진, 확률) ----
    t0 = time.time()
    tv = h.Vector(stim); fibers = []
    for k in range(n_fiber):
        vs = h.VecStim(); vs.play(tv); fibers.append(vs); keeph.append(vs)
    keeph.append(tv)
    prm = P3.CLASSES[sc_class]; scrng = np.random.RandomState(7000 + RANK + seed * 131); n_sc = 0
    for g in my:
        is_pc = gtype[g] == "PC"; k_syn = sc_pc if is_pc else sc_int; gnS = sc_g_pc if is_pc else sc_g_int
        for _ in range(k_syn):
            seg = sr_or_dend(cells[g], is_pc, scrng)
            syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000 + seed * 7, 1, 1), deterministic=det)
            nc = h.NetCon(fibers[scrng.randint(n_fiber)], syn); nc.weight[0] = gnS; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_sc += 1
    pc.barrier()
    log(f"[3/4 SC배선] rank0 {n_sc} SC시냅스 · {time.time()-t0:.0f}s")

    # ---- 전세포 실제 기하 + 전달행렬 M_rank (rank 세그) ----
    t0 = time.time()
    mids = []; rads = []; vecs = []
    cv = h.CVode(); cv.use_fast_imem(1)
    apical_sign = []
    for g in my:
        geom = L.collect_segments(list(cells[g].all))
        Rc = quat_to_R(quat[keep[g]])
        real = geom["mid"] @ Rc.T + xyz[keep[g]]        # 실제 슬라이스 3D 좌표
        mids.append(real); rads.append(geom["radius"])
        for seg in geom["segs"]:
            v = h.Vector(); v.record(seg._ref_i_membrane_, rec_dt); vecs.append(v)
    mids = np.vstack(mids) if mids else np.zeros((0, 3)); rads = np.concatenate(rads) if rads else np.zeros(0)
    # 밴드면 좌표(전극과 동일계) + 깊이 z' (SR/apical쪽을 유리면 0으로)
    uv = (mids - c0) @ face_ax.T                         # (Nseg,2)
    dep = (mids - c0) @ depth_ax                         # 깊이(+=SR/SLM쪽)
    # 전역 깊이 범위(allreduce max/min) — 유리면·슬라이스두께 h 정의용
    dmax = pc.allreduce(float(dep.max()) if len(dep) else -1e18, 2)
    dmin = pc.allreduce(float(dep.min()) if len(dep) else 1e18, 3)
    zloc = (dmax - dep) + Z_GLASS_MARGIN                 # SR/apical(큰 dep) → 유리면 0쪽
    Hh = (dmax - dmin) + 2 * Z_GLASS_MARGIN
    geom_r = dict(mid=np.column_stack([uv[:, 0], uv[:, 1], zloc]), radius=rads)
    # 전극(밴드면 2D → 3D: z=0)
    E3 = np.column_stack([E2d[:, 0], E2d[:, 1], np.zeros(NELEC)])
    M_rank = L.moi_point_matrix(geom_r, E3, SIG_T, SIG_S, SIG_G, Hh, N_IMG) if len(rads) else np.zeros((NELEC, 0))
    log(f"[전달행렬] rank세그 {len(rads)} · Hh={Hh:.0f}µm · {time.time()-t0:.0f}s")

    # ---- 구동 ----
    t0 = time.time()
    h.celsius = 34.0; h.cvode_active(0); h.dt = dt
    tvec = h.Vector().record(h._ref_t)
    h.finitialize(-70.0); pc.set_maxstep(10); pc.psolve(tstop)
    log(f"[4/4 구동완료] {time.time()-t0:.0f}s")

    # ---- rank 부분 fEPSP → 전 rank 합 ----
    nt = int(round(tstop / rec_dt)) + 1
    I = np.array([np.asarray(v) for v in vecs]) if vecs else None
    if I is not None and I.size:
        Ve_local = (M_rank @ I) * 1e3                    # (NELEC, nt) µV
        if Ve_local.shape[1] != nt:                      # 기록 길이 방어
            nt = Ve_local.shape[1]
    else:
        Ve_local = np.zeros((NELEC, nt))
    if NHOST > 1:                                        # rank별 부분합 → allgather 후 합
        parts = pc.py_allgather(Ve_local.tolist())
        Ve = np.sum([np.array(p) for p in parts], axis=0)
    else:
        Ve = Ve_local
    t = np.arange(Ve.shape[1]) * rec_dt

    nspk_all = int(pc.allreduce(len(spt), 1)); npc_all = sum(1 for g in range(N) if gtype[g] == "PC")
    log(f"[스파이크] 총 {nspk_all}개 (PC {npc_all}·전체 {N}세포, tstop {tstop}ms) → {'역치하(스파이크 거의 없음)' if nspk_all < 0.05*N else '초과발화(집단스파이크 오염)'}")
    if RANK == 0:
        j_on = np.where(over)[0]
        amp = np.array([Ve[j, np.argmax(np.abs(Ve[j]))] for j in range(NELEC)])
        jmax = j_on[np.argmax(np.abs(amp[j_on]))] if len(j_on) else int(np.argmax(np.abs(amp)))
        print(f"[결과] 유발 fEPSP 조직위 중앙 |{np.median(np.abs(amp[over])):.1f}|µV · 최대 |{np.abs(amp).max():.1f}|µV(#{jmax})", flush=True)
        out = os.path.join(FIG, f"_net_fepsp_{tag}.npz")
        np.savez(out, t=t, Ve=Ve, E=E2d, over=over, amp=amp, N=N, sc_class=sc_class,
                 stim=np.array(stim), det=det, no_inh=no_inh, pc_only=pc_only, jmax=jmax, seed=seed)
        print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)
    pc.barrier(); pc.done()


if __name__ == "__main__":
    main()
