# -*- coding: utf-8 -*-
"""
03_network/3_run/mpi_plasticity.py — 나-2-② 네트워크 주파수의존 LTD/LTP (Dudek & Bear 1992)

mpi_baseline.py의 망 빌드를 재사용하되, SC→PC(SP_PC) 시냅스의 장기가소성을 켜고
(gamma_p·gamma_d 동결 해제·ca_stp 0·rho0 0.5), 조건화 자극열(빈도 f × N펄스)을 인가한 뒤
SC→PC 시냅스 효능 rho의 조건화 전후 변화를 집단으로 측정한다.

핵심(네트워크 규모): 조건화 중 후시냅스 PC 발화는 망 동역학에서 자연 발생 —
고빈도면 EPSP 합산→PC 발화→C_post 가세→칼슘이 theta_p 넘어 강화(LTP),
저빈도면 PC 침묵→C_pre만→칼슘이 theta_d만 스쳐 억압(LTD). 수천 시냅스 집단 평균이
결정론 모델의 노이즈 평균을 대체해 등급적 BCM 곡선을 준다.

판독: SC→PC 시냅스 rho 조건화 전후(rho_init 0.5 → rho_final) 분포·평균 → LTD/LTP.

실행:
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib mpirun -np 5 python 03_network/3_run/mpi_plasticity.py \
      --plastic --frac 0.08 --cond FREQ NPULSE [--rho0 0.5]
  예: --cond 1 900 (LFS 1Hz), --cond 100 900 (HFS 100Hz)
"""
import os, sys, time, json
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


def arg(flag, default, cast=float):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


PLASTIC = "--plastic" in sys.argv                 # SC→PC 가소성 ON
FRAC = arg("--frac", 0.08)                         # 자극 섬유 비율(표준 volley 8%)
RADIUS = arg("--r", 150.0)
SETTLE = arg("--settle", 300.0)
RHO0 = arg("--rho0", 0.5)                           # 초기 효능(불안정점)
GABAOFF = "--gabaoff" in sys.argv
# 조건화: --cond FREQ NPULSE
if "--cond" in sys.argv:
    ci = sys.argv.index("--cond"); COND_FREQ = float(sys.argv[ci + 1]); COND_NP = int(sys.argv[ci + 2])
else:
    COND_FREQ, COND_NP = 1.0, 900
STIM_T = SETTLE + 10.0
COND_DUR = (COND_NP - 1) * 1000.0 / COND_FREQ       # 첫→마지막 조건화 펄스
OBS = arg("--obs", 50.0)
TSTOP = STIM_T + COND_DUR + OBS
TAG = "_f%g_n%d_%s" % (COND_FREQ, COND_NP, "block" if GABAOFF else "norm")
CHUNK = arg("--chunk", 5000.0)                      # 진행 로그 청크(ms sim time)

SC2 = {"SP_CCKBC", "SR_SCA", "SLM_PPA", "SP_Ivy"}
PERI = {"SP_PVBC", "SP_CCKBC", "SP_AA"}
SLM = {"SO_OLM", "SO_BS", "SO_BP", "SLM_PPA"}
SR = {"SR_SCA"}


def sc_stp(mt):
    if mt == "SP_PC":
        return 0.14, 186.0, 129.0, 12
    if mt in SC2:
        return 0.11, 307.0, 195.0, 4
    return 0.11, 307.0, 195.0, 8


def target_comp(m):
    return "peri" if m in PERI else "slm" if m in SLM else "sr" if m in SR else "dend"


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
    P = np.asarray(P, float); r = (XYZg + rot.apply(P) - seed) @ radial
    soma = np.asarray(soma); idx = np.arange(len(P)); dend = idx[~soma]; rd = r[dend]
    return secs, xs, {"soma": idx[soma], "dend": dend,
                      "peri": np.concatenate([idx[soma], dend[np.abs(rd) < 60]]),
                      "sr": dend[(rd >= 25) & (rd <= 450)], "slm": dend[rd > 450]}


def main():
    h.nrnmpi_init(); pc = h.ParallelContext(); rank, nhost = int(pc.id()), int(pc.nhost())
    pc.timeout(3600)
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
    scpost = sc["post_gid"]; scxyz = sc["xyz"].astype(float); dist_e3 = sc["dist_e3"].astype(float)
    fiber_id = fib["fiber_id"]
    di = np.load(os.path.join(DERIVED, "synapses_internal.npz"), allow_pickle=True)
    ipre = di["pre_gid"]; ipost = di["post_gid"]; insyn = di["n_syn"]
    p = np.load(os.path.join(DERIVED, "synapse_params.npz"), allow_pickle=True)
    irule = p["internal_rule"]; igsyn = p["internal_gsyn"]; imech = p["internal_mech"].astype(str)

    N = len(mt); gids = np.arange(N); mine = set(int(g) for g in gids if g % nhost == rank)
    keep = []
    for g in mine:
        cell = B.build_cell(g); pc.set_gid2node(g, rank)
        nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0]); nc.threshold = -10
        pc.cell(g, nc); keep.append(nc)
    pc.barrier()
    if rank == 0:
        print(f"[조립] {N}세포 · {time.time()-t0:.0f}s", flush=True)

    # 조건화 자극: 가까운 순 FRAC 섬유(표준 volley), 각 섬유 COND_NP 펄스 @ COND_FREQ
    nfib = int(fiber_id.max()) + 1
    fmin = np.full(nfib, np.inf); np.minimum.at(fmin, fiber_id, dist_e3)
    frank = np.argsort(fmin); k = int(round(FRAC * nfib))
    driven = set(int(f) for f in frank[:k])
    if rank == 0:
        print(f"[자극] frac {FRAC:.2f} = {k}/{nfib} 섬유 · 조건화 {COND_FREQ:g}Hz × {COND_NP}펄스 "
              f"(지속 {COND_DUR/1000:.1f}s) · tstop {TSTOP/1000:.1f}s", flush=True)
    cond_spk = [STIM_T + kk * 1000.0 / COND_FREQ for kk in range(COND_NP)]
    for fidv in set(int(f) for f in np.unique(fiber_id) if int(f) % nhost == rank):
        pc.set_gid2node(FIBER_OFFSET + fidv, rank)
        spk = cond_spk if fidv in driven else []
        vs = h.VecStim(); tv = h.Vector(spk)
        vs.play(tv); nc = h.NetCon(vs, None); pc.cell(FIBER_OFFSET + fidv, nc)
        keep.append((vs, tv, nc))

    # SC 배선 — SP_PC 표적은 가소성 ON(--plastic), 나머지는 동결
    scsel = np.where(np.isin(scpost, list(mine)))[0]; n_sc = 0
    pc_syns = []                      # (gid, syn) — SC→PC 가소성 시냅스(rho 판독 대상)
    for g in mine:
        cell = B.cells[g]; tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[g][[1, 2, 3, 0]]); soma = cell.soma[0]
        is_pc = (mt[g] == "SP_PC")
        for si in scsel[scpost[scsel] == g]:
            mp = rot.inv().apply(scxyz[si] - XYZ[g]); _, kk = tree.query(mp, k=1); sec, x = ref[kk]
            syn = h.GBPlasticityStpProbSyn(sec(x)); U, D, Fa, Nr = sc_stp(mt[g])
            syn.Use = U; syn.Dep = D; syn.Fac = Fa; syn.Nrrp = Nr; syn.gmax = 0.8 / 1000.0
            if PLASTIC and is_pc:
                syn.ca_stp = 0; syn.rho0 = RHO0           # 가소성 ON: gamma는 .mod 기본값(1645·313) 유지
                pc_syns.append((g, syn))
            else:
                syn.gamma_p = 0.0; syn.gamma_d = 0.0      # 동결(단기만)
            syn.setRNG(g + 1, int(si) + 1, 3)
            ncp = pc.gid_connect(FIBER_OFFSET + int(fiber_id[si]), syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
            ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0
            keep.append((syn, ncp, ncs)); n_sc += 1
    pc.barrier(); tot_sc = int(pc.allreduce(n_sc, 1)); tot_pcsyn = int(pc.allreduce(len(pc_syns), 1))
    if rank == 0:
        print(f"[SC] {tot_sc:,} · 가소성 SC→PC {tot_pcsyn:,} · {time.time()-t0:.0f}s", flush=True)

    # 내부 배선 (전부 동결 — 단기만)
    isel = np.where(np.isin(ipost, list(mine)))[0]; rng = np.random.default_rng(0); cc = {}; n_int = 0
    for cix in isel:
        pg, qg, ns = int(ipre[cix]), int(ipost[cix]), int(insyn[cix])
        if qg not in cc:
            cc[qg] = compartments(B.cells[qg], XYZ[qg], Rot.from_quat(Q[qg][[1, 2, 3, 0]]), seed, Mrows[:, 1])
        secs, xs, comp = cc[qg]; pool = comp[target_comp(mt[pg])]
        if len(pool) == 0:
            pool = comp["dend"] if len(comp["dend"]) else comp["soma"]
        if len(pool) == 0:
            continue
        rl = rules.get(int(irule[cix])); gs = float(igsyn[cix]); mech = imech[cix]; soma_q = B.cells[qg].soma[0]
        for kk in pool[rng.integers(0, len(pool), ns)]:
            sec, x = secs[kk], xs[kk]
            if mech == "E":
                syn = h.GBPlasticityStpProbSyn(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
                syn.gmax = gs / 1000.0; syn.gamma_p = 0.0; syn.gamma_d = 0.0; syn.setRNG(qg + 1, 900000 + cix, 7)
                ncp = pc.gid_connect(pg, syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
                ncs = h.NetCon(soma_q(0.5)._ref_v, syn, sec=soma_q); ncs.weight[0] = -1.0
                keep.append((syn, ncp, ncs))
            else:
                syn = h.ProbGABAAB_EMS(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
                syn.setRNG(qg + 1, 800000 + cix, 4)
                ncp = pc.gid_connect(pg, syn); ncp.weight[0] = 0.0 if GABAOFF else gs / 1000.0; ncp.delay = 1.0
                keep.append((syn, ncp))
            n_int += 1
    pc.barrier(); tot_int = int(pc.allreduce(n_int, 1))
    if rank == 0:
        print(f"[내부] {tot_int:,} · 총 {tot_sc+tot_int:,} 시냅스 · {time.time()-t0:.0f}s", flush=True)

    # ── (선택) psolve 속도 실측 후 종료 ──
    if "--timing" in sys.argv:
        import time as _t
        pc.set_maxstep(10); h.finitialize(-70)
        tcur = 0.0
        for dtv, dur in [(0.25, 20.0), (0.025, 50.0)]:
            h.dt = dtv; tcur += dur
            w0 = _t.time(); pc.psolve(tcur); wall = _t.time() - w0
            rate = wall / (dur / 1000.0)
            if rank == 0:
                print(f"[타이밍] dt={dtv} · +{dur:.0f}ms 시뮬 · 벽시계 {wall:.1f}s · "
                      f"{rate:.0f} 벽초/시뮬초 · {rate/3600:.3f} 시간/시뮬-초", flush=True)
        if rank == 0:
            print(f"[환산] 20초 시뮬 → {20*rate/3600:.1f}시간 = {20*rate/86400:.2f}일 "
                  f"· 80분(4800초) 시뮬 → {4800*rate/86400:.0f}일 (dt=0.025 기준)", flush=True)
        pc.barrier(); pc.done(); h.quit(); return

    # ── 구동: settle → rho_init → 조건화 → rho_final ──
    tspk = h.Vector(); idspk = h.Vector(); pc.spike_record(-1, tspk, idspk)
    pc.set_maxstep(10); h.finitialize(-70); trun = time.time()
    h.dt = 0.25; pc.psolve(SETTLE)
    rho_init = np.array([syn.rho for _, syn in pc_syns], float)
    nsp0 = int(pc.allreduce(int(tspk.size()), 1))
    if rank == 0:
        print(f"[안정화] {time.time()-trun:.0f}s · 스파이크 {nsp0} · rho_init 평균 "
              f"{float(pc.allreduce(rho_init.sum(),1))/max(tot_pcsyn,1):.3f}", flush=True)
    h.dt = 0.025; t = SETTLE
    while t < TSTOP - 1e-6:
        t = min(t + CHUNK, TSTOP); pc.psolve(t)
        nsp = int(pc.allreduce(int(tspk.size()), 1))
        rmean = float(pc.allreduce(np.sum([syn.rho for _, syn in pc_syns]), 1)) / max(tot_pcsyn, 1)
        if rank == 0:
            print(f"  [조건화] t={t-STIM_T:+.0f}ms · 누적스파이크 {nsp:,} · rho평균 {rmean:.3f} · {time.time()-trun:.0f}s", flush=True)
    psolve_s = time.time() - trun
    rho_final = np.array([syn.rho for _, syn in pc_syns], float)

    # 집단 집계 (rank 0)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    gi = comm.gather(rho_init, root=0); gf = comm.gather(rho_final, root=0)
    ggid = comm.gather(np.array([g for g, _ in pc_syns], int), root=0)
    nsp = int(pc.allreduce(int(tspk.size()), 1))
    if rank == 0:
        ri = np.concatenate([a for a in gi if len(a)]) if any(len(a) for a in gi) else np.array([])
        rf = np.concatenate([a for a in gf if len(a)]) if any(len(a) for a in gf) else np.array([])
        gg = np.concatenate([a for a in ggid if len(a)]) if any(len(a) for a in ggid) else np.array([], int)
        d = rf - ri
        ltp = int(np.sum(d > 0.02)); ltd = int(np.sum(d < -0.02)); nz = int(np.sum(np.abs(d) <= 0.02))
        print(f"\n[결과] {COND_FREQ:g}Hz × {COND_NP}펄스 · psolve {psolve_s:.0f}s · 누적스파이크 {nsp:,}", flush=True)
        print(f"  rho 평균 {ri.mean():.3f} → {rf.mean():.3f} (Δ{rf.mean()-ri.mean():+.3f})", flush=True)
        print(f"  시냅스 {len(d):,}: LTP {ltp:,}({100*ltp/len(d):.0f}%) · LTD {ltd:,}({100*ltd/len(d):.0f}%) · 무변화 {nz:,}({100*nz/len(d):.0f}%)", flush=True)
        np.savez_compressed(os.path.join(ROOT, "scratch", f"mpi_plast{TAG}.npz"),
                            rho_init=ri, rho_final=rf, gid=gg, cond_freq=COND_FREQ, cond_np=COND_NP,
                            rho0=RHO0, psolve_s=psolve_s, nspike=nsp)
        json.dump({"cond_freq": COND_FREQ, "cond_np": COND_NP, "rho0": RHO0,
                   "rho_init_mean": float(ri.mean()), "rho_final_mean": float(rf.mean()),
                   "dmean": float(rf.mean() - ri.mean()), "n_syn": int(len(d)),
                   "ltp": ltp, "ltd": ltd, "nochange": nz, "nspike": nsp, "psolve_s": psolve_s},
                  open(os.path.join(ROOT, "scratch", f"mpi_plast{TAG}.json"), "w"))
        print(f"[저장] scratch/mpi_plast{TAG}.npz · 총 {time.time()-t0:.0f}s", flush=True)
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
