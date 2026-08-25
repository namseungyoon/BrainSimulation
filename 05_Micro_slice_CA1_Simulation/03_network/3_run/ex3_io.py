# -*- coding: utf-8 -*-
"""
03_network/3_run/ex3_io.py  —  Ex3: SC I-O + 억제 차단 (recruitment 스윕)

설계(2026-08-24 확정):
 - locus(자극 영역) 고정 = E3. 세기축 = locus에서 가까운 순 섬유의 **모집 비율**(=fiber volley).
   섬유는 min(dist_e3)로 순위 → 가까운 frac 비율을 발화(축삭 통째로 → 모든 시냅스 방출).
 - 전도도(gsyn) 0.8nS 고정. 세기는 오직 전시냅스 모집.
 - ×2 조건: 정상 / 억제차단(GABA weight=0, bicuculline 대응).
 - 전체망 **1회 조립** 후 조건마다 VecStim·GABA만 바꿔 재실행(재빌드 0).
읽음값(1차): 발화 세포%(추체/억제 분해)·총 스파이크. (fEPSP slope는 mea_forward 구축 후 2차)
결과: scratch/ex3_io.npz + ex3_io.json (진행 중 partial 저장).

실행: LD_LIBRARY_PATH=$CONDA_PREFIX/lib mpirun -np 5 python 03_network/3_run/ex3_io.py
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

FRACS = [0.005, 0.01, 0.02, 0.04, 0.08]          # 모집 비율 (fiber volley) — 역치~상승부 (50~800섬유)
CONDS = [("normal", False), ("block", True)]       # (이름, 억제차단?)
SETTLE = float(sys.argv[sys.argv.index("--settle") + 1]) if "--settle" in sys.argv else 30.0  # 무음망→짧게
STIM_T = SETTLE + 10.0
OBS = float(sys.argv[sys.argv.index("--obs") + 1]) if "--obs" in sys.argv else 30.0  # 자극 후 관측(ms)
TSTOP = STIM_T + OBS
FEPSP = "--fepsp" in sys.argv                       # 조건마다 전극 fEPSP 기록(fast_imem+mea_forward)
FSTRIDE = int(sys.argv[sys.argv.index("--fstride") + 1]) if "--fstride" in sys.argv else 4

SC2 = {"SP_CCKBC", "SR_SCA", "SLM_PPA", "SP_Ivy"}
PERI = {"SP_PVBC", "SP_CCKBC", "SP_AA"}; SLM = {"SO_OLM", "SO_BS", "SO_BP", "SLM_PPA"}; SR = {"SR_SCA"}


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
    fiber_id = fib["fiber_id"].astype(int)
    nfib = int(fib["n_fiber"]) if "n_fiber" in fib.files else int(fiber_id.max() + 1)
    di = np.load(os.path.join(DERIVED, "synapses_internal.npz"), allow_pickle=True)
    ipre = di["pre_gid"]; ipost = di["post_gid"]; insyn = di["n_syn"]
    p = np.load(os.path.join(DERIVED, "synapse_params.npz"), allow_pickle=True)
    irule = p["internal_rule"]; igsyn = p["internal_gsyn"]; imech = p["internal_mech"].astype(str)

    # ── 섬유 모집 순위: 각 섬유의 locus(E3) 최소 거리 (축삭이 locus를 지나가느냐) ──
    fmin = np.full(nfib, np.inf)
    np.minimum.at(fmin, fiber_id, dist_e3)
    frank = np.argsort(fmin)                          # 가까운 섬유부터
    def recruited(frac):
        k = int(round(frac * nfib))
        return set(int(f) for f in frank[:k])

    N = len(mt); gids = np.arange(N); mine = set(int(g) for g in gids if g % nhost == rank)
    keep = []
    for g in mine:
        cell = B.build_cell(g); pc.set_gid2node(g, rank)
        nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0]); nc.threshold = -10
        pc.cell(g, nc); keep.append(nc)
    pc.barrier()
    if rank == 0:
        print(f"[조립] {N}세포 · {time.time()-t0:.0f}s", flush=True)

    # VecStim (섬유별) — tv 내용은 조건마다 갱신
    vstim = {}
    for fidv in set(int(f) for f in np.unique(fiber_id) if int(f) % nhost == rank):
        pc.set_gid2node(FIBER_OFFSET + fidv, rank)
        vs = h.VecStim(); tv = h.Vector(); vs.play(tv); nc = h.NetCon(vs, None)
        pc.cell(FIBER_OFFSET + fidv, nc)
        vstim[fidv] = (vs, tv, nc); keep.append((vs, tv, nc))

    # SC 배선
    scsel = np.where(np.isin(scpost, list(mine)))[0]; n_sc = 0
    for g in mine:
        cell = B.cells[g]; tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[g][[1, 2, 3, 0]]); soma = cell.soma[0]
        for si in scsel[scpost[scsel] == g]:
            mp = rot.inv().apply(scxyz[si] - XYZ[g]); _, k = tree.query(mp, k=1); sec, x = ref[k]
            syn = h.GBPlasticityStpProbSyn(sec(x)); U, D, Fa, Nr = sc_stp(mt[g])
            syn.Use = U; syn.Dep = D; syn.Fac = Fa; syn.Nrrp = Nr; syn.gmax = 0.8 / 1000.0
            syn.gamma_p = 0.0; syn.gamma_d = 0.0; syn.setRNG(g + 1, int(si) + 1, 3)
            ncp = pc.gid_connect(FIBER_OFFSET + int(fiber_id[si]), syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
            ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0
            keep.append((syn, ncp, ncs)); n_sc += 1
    pc.barrier(); tot_sc = int(pc.allreduce(n_sc, 1))
    if rank == 0:
        print(f"[SC] {tot_sc:,} · {time.time()-t0:.0f}s", flush=True)

    # 내부 배선 (GABA NetCon은 억제차단 위해 별도 보관)
    isel = np.where(np.isin(ipost, list(mine)))[0]; rng = np.random.default_rng(0); cc = {}; n_int = 0
    gaba = []   # (ncp, w0) — block 조건에서 weight=0
    for ci in isel:
        pg, qg, ns = int(ipre[ci]), int(ipost[ci]), int(insyn[ci])
        if qg not in cc:
            cc[qg] = compartments(B.cells[qg], XYZ[qg], Rot.from_quat(Q[qg][[1, 2, 3, 0]]), seed, Mrows[:, 1])
        secs, xs, comp = cc[qg]; pool = comp[target_comp(mt[pg])]
        if len(pool) == 0:
            pool = comp["dend"] if len(comp["dend"]) else comp["soma"]
        if len(pool) == 0:
            continue
        rl = rules.get(int(irule[ci])); gs = float(igsyn[ci]); mech = imech[ci]; soma_q = B.cells[qg].soma[0]
        for k in pool[rng.integers(0, len(pool), ns)]:
            sec, x = secs[k], xs[k]
            if mech == "E":
                syn = h.GBPlasticityStpProbSyn(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
                syn.gmax = gs / 1000.0; syn.gamma_p = 0.0; syn.gamma_d = 0.0; syn.setRNG(qg + 1, 900000 + ci, 7)
                ncp = pc.gid_connect(pg, syn); ncp.weight[0] = 1.0; ncp.delay = 1.0
                ncs = h.NetCon(soma_q(0.5)._ref_v, syn, sec=soma_q); ncs.weight[0] = -1.0
                keep.append((syn, ncp, ncs))
            else:
                syn = h.ProbGABAAB_EMS(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
                syn.setRNG(qg + 1, 800000 + ci, 4)
                w0 = gs / 1000.0
                ncp = pc.gid_connect(pg, syn); ncp.weight[0] = w0; ncp.delay = 1.0
                gaba.append((ncp, w0)); keep.append((syn, ncp))
            n_int += 1
    pc.barrier(); tot_int = int(pc.allreduce(n_int, 1)); tot_gaba = int(pc.allreduce(len(gaba), 1))
    if rank == 0:
        print(f"[내부] {tot_int:,} (GABA {tot_gaba:,}) · 총 {tot_sc+tot_int:,} 시냅스 · {time.time()-t0:.0f}s", flush=True)
        print(f"[준비] 조건 {len(FRACS)}세기 × {len(CONDS)} = {len(FRACS)*len(CONDS)}회 · 전체망 재사용", flush=True)

    tspk = h.Vector(); idspk = h.Vector(); pc.spike_record(-1, tspk, idspk)
    is_pc = np.array([mt[g] == "SP_PC" for g in range(N)])
    totE = int(np.sum(is_pc)); totI = int(N - totE)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    results = []; fep_all = []; spk_all = []

    # ── fEPSP 기록기(조건 전체 재사용, 1회 설정) ──
    frec = None; elec = None; enames = None
    if FEPSP:
        import fepsp_record as fr
        elec = np.array([e["xyz_um"] for e in cfg["electrodes"]["list"]], float)
        enames = [e["id"] + "(" + e["layer"] + ")" for e in cfg["electrodes"]["list"]]
        frec = fr.FEPSPRecorder(elec, stride=FSTRIDE)
        for g in mine:
            frec.add_cell(B.cells[g], XYZ[g], Rot.from_quat(Q[g][[1, 2, 3, 0]]))
        frec.finalize(rec_dt=0.1)                                    # 매 조건 finitialize마다 재기록
        if rank == 0:
            print(f"[fEPSP] 기록기 준비(stride {FSTRIDE}) · 조건마다 E1/E2/E3 fEPSP 기록", flush=True)

    for cname, block in CONDS:
        # 억제 weight 설정
        for ncp, w0 in gaba:
            ncp.weight[0] = 0.0 if block else w0
        for frac in FRACS:
            drv = recruited(frac)
            for fidv, (vs, tv, nc) in vstim.items():
                tv.resize(0)
                if fidv in drv:
                    tv.append(STIM_T)
            tspk.resize(0); idspk.resize(0)
            pc.set_maxstep(10); h.finitialize(-70); tr = time.time()
            h.dt = 0.25; pc.psolve(SETTLE)          # 안정화(거친)
            h.dt = 0.025; pc.psolve(TSTOP)          # 자극+관측(고운)
            dt_run = time.time() - tr
            at = comm.gather(list(tspk), root=0); ai = comm.gather(list(idspk), root=0)
            Vtot = None; tfe = None
            if FEPSP:                                                # 전 랭크 집합통신
                Vtot = comm.allreduce(frec.potential_local(), op=MPI.SUM)
                tfe = np.array(frec.times())
            if rank == 0:
                st = np.concatenate([np.array(x) for x in at]) if any(len(x) for x in at) else np.array([])
                sid = np.concatenate([np.array(x) for x in ai]).astype(int) if any(len(x) for x in ai) else np.array([], int)
                m = sid < N
                st = st[m]; sid = sid[m]
                fired = set(sid.tolist())
                fm = np.array([g in fired for g in range(N)])
                nE = int(np.sum(fm & is_pc)); nI = int(np.sum(fm & ~is_pc))
                rec = {"cond": cname, "frac": frac, "volley_pct": int(round(frac * 100)),
                       "recruited_fibers": len(drv), "spikes": int(len(st)),
                       "fired": int(fm.sum()), "firedE": nE, "firedI": nI,
                       "pctE": 100.0 * nE / totE, "pctI": 100.0 * nI / max(totI, 1),
                       "run_s": dt_run}
                spk_all.append({"cond": cname, "frac": frac, "st": st.astype(np.float32), "sid": sid.astype(np.int32)})
                if FEPSP:
                    base = Vtot[:, tfe < STIM_T].mean(axis=1) if (tfe < STIM_T).any() else Vtot[:, 0]
                    mm = tfe >= STIM_T
                    slope = fr.FEPSPRecorder.fepsp_slope(Vtot - base[:, None], tfe, win=(0.3, 1.5), t0=STIM_T)
                    peak = np.array([(Vtot[i] - base[i])[mm].min() for i in range(len(elec))])  # sink 최저
                    rec["fepsp_slope_uV_ms"] = [round(float(s) * 1000, 3) for s in slope]        # µV/ms
                    rec["fepsp_peak_uV"] = [round(float(p) * 1000, 2) for p in peak]
                    fep_all.append({"cond": cname, "frac": frac, "V": (Vtot * 1000).astype(np.float32), "t": tfe.astype(np.float32)})  # µV
                results.append(rec)
                fpk = (" · fEPSP E3 " + f"{rec['fepsp_peak_uV'][2]:+.1f}µV") if FEPSP else ""
                print(f"  [{cname:6s}] volley {int(frac*100):3d}% (섬유 {len(drv):5d}) -> "
                      f"추체 {nE:4d}/{totE} ({rec['pctE']:4.1f}%) · 억제 {nI:3d} ({rec['pctI']:4.1f}%) · "
                      f"스파이크 {len(st):5d}{fpk} · {dt_run:.0f}s", flush=True)
                json.dump({"conds": [c for c, _ in CONDS], "fracs": FRACS, "settle": SETTLE,
                           "stim_t": STIM_T, "tstop": TSTOP, "totE": totE, "totI": totI,
                           "results": results},
                          open(os.path.join(ROOT, "scratch", "ex3_io.json"), "w"), ensure_ascii=False, indent=1)
            pc.barrier()

    if rank == 0:
        R = results
        arr = lambda k: np.array([r[k] for r in R])
        extra = {}
        if FEPSP:
            extra["fepsp_slope"] = np.array([r["fepsp_slope_uV_ms"] for r in R])   # (ncond, 3) µV/ms
            extra["fepsp_peak"] = np.array([r["fepsp_peak_uV"] for r in R])        # (ncond, 3) µV
            extra["enames"] = np.array(enames)
        np.savez_compressed(os.path.join(ROOT, "scratch", "ex3_io.npz"),
                            cond=np.array([r["cond"] for r in R]), frac=arr("frac"),
                            volley_pct=arr("volley_pct"), recruited=arr("recruited_fibers"),
                            firedE=arr("firedE"), firedI=arr("firedI"),
                            pctE=arr("pctE"), pctI=arr("pctI"), spikes=arr("spikes"),
                            totE=totE, totI=totI, stim_t=STIM_T, settle=SETTLE, **extra)
        # 트레이스(raster·fEPSP 파형)는 별도 파일(가변길이 → object)
        np.savez(os.path.join(ROOT, "scratch", "ex3_io_traces.npz"),
                 spk=np.array(spk_all, dtype=object),
                 fep=np.array(fep_all, dtype=object),
                 elec=(elec if elec is not None else np.zeros((0, 3))),
                 enames=(np.array(enames) if enames else np.array([])), stim_t=STIM_T)
        print(f"[Ex3] 트레이스 저장 scratch/ex3_io_traces.npz (raster {len(spk_all)}조건" +
              (f" · fEPSP {len(fep_all)}조건)" if FEPSP else ")"), flush=True)
        print(f"\n[Ex3 완료] {len(R)}회 · 총 {time.time()-t0:.0f}s · scratch/ex3_io.npz", flush=True)
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
