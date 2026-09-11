# -*- coding: utf-8 -*-
"""
03_network/3_run/mpi_baseline.py  —  A: baseline 발화 확인 (전체 회로, 억제 포함)

전체 5,610 세포 + 590만 시냅스(SC + 내부 E/I)를 MPI 빌드·배선한 뒤, 국소 SC 자극
(후시냅스 point: E3 반경 섬유 단일 volley)을 주고 발화패턴을 본다.
스모크(SC만·55% 과발화)와 달리 **내부 억제가 발화를 생리적 수준으로 잡는지** 검증.
결과: 총 스파이크·발화세포(E/I 분해)·대표 전압 → scratch/mpi_baseline.npz

실행: LD_LIBRARY_PATH=$CONDA_PREFIX/lib mpirun -np 5 python 03_network/3_run/mpi_baseline.py [--r 150]
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
RADIUS = float(sys.argv[sys.argv.index("--r") + 1]) if "--r" in sys.argv else 150.0
FRAC = float(sys.argv[sys.argv.index("--frac") + 1]) if "--frac" in sys.argv else 0.0  # >0이면 가까운 순 N% 섬유 모집(Ex3와 동일 volley 정의). 0이면 --r 반경.
USE_GPU = "--gpu" in sys.argv   # CoreNEURON GPU 실행(단일 psolve, 균일 dt)
NOSTIM = "--nostim" in sys.argv  # 자극 없음(자발 발화율 baseline)
FEPSP = "--fepsp" in sys.argv    # 막전류→전극 fEPSP 기록(fast_imem + mea_forward)
SETTLE = float(sys.argv[sys.argv.index("--settle") + 1]) if "--settle" in sys.argv else 300.0
ISI = float(sys.argv[sys.argv.index("--isi") + 1]) if "--isi" in sys.argv else 0.0  # >0이면 페어펄스 2번째 자극(Ex3b: fEPSP PPR). 0=단발
TRAIN_FREQ = float(sys.argv[sys.argv.index("--train") + 1]) if "--train" in sys.argv else 0.0  # >0이면 train(Hz). 다음 인자=펄스 수
TRAIN_NP = int(sys.argv[sys.argv.index("--train") + 2]) if "--train" in sys.argv else 8
GABAOFF = "--gabaoff" in sys.argv                # GABA_A off(bicuculline 대응, 탈억제 조건)
if NOSTIM:
    # 무자극 자발 baseline: settle 후 OBS(기본 1000ms) 관측, 자극 없음
    OBS = float(sys.argv[sys.argv.index("--obs") + 1]) if "--obs" in sys.argv else 1000.0
    STIM_T = SETTLE                 # 자극 없으니 기준시각 = 관측 시작
    TSTOP = SETTLE + OBS
    TAG = "_nostim"
else:
    OBS = float(sys.argv[sys.argv.index("--obs") + 1]) if "--obs" in sys.argv else 30.0  # 마지막 자극 후 관측창(ms)
    STIM_T = SETTLE + 10.0                          # 1번째 자극(자극 전 10ms baseline)
    STIMDUR = ((TRAIN_NP - 1) * 1000.0 / TRAIN_FREQ) if TRAIN_FREQ > 0 else ISI  # 1번째→마지막 자극 시간
    TSTOP = SETTLE + 10.0 + STIMDUR + OBS
    _cond = ("tr%d" % int(TRAIN_FREQ)) if TRAIN_FREQ > 0 else ("isi%d" % int(ISI) if ISI > 0 else "single")
    TAG = "_%s_%s" % (_cond, "block" if GABAOFF else "norm")   # 조건별 출력 구분
WRITE_DIR = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None  # 파일모드: 모델을 여기 덤프 후 종료

SC2 = {"SP_CCKBC", "SR_SCA", "SLM_PPA", "SP_Ivy"}
SC3 = {"SP_PVBC", "SP_AA", "SP_BS", "SO_OLM", "SO_Tri", "SO_BS", "SO_BP"}
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
    pc.timeout(3600)  # 무거운 스텝(고빈도 burst 후반 스파이크 급증)에서 nrn_timeout 오탐 abort 방지; 실제 데드락은 1h 후 감지
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

    # 자극 대상: 무자극=없음. --frac>0이면 가까운 순 N% 섬유(Ex3와 동일 volley). 아니면 --r 반경.
    if NOSTIM:
        driven = set()
    elif FRAC > 0:
        nfib = int(fiber_id.max()) + 1
        fmin = np.full(nfib, np.inf); np.minimum.at(fmin, fiber_id, dist_e3)
        frank = np.argsort(fmin); k = int(round(FRAC * nfib))
        driven = set(int(f) for f in frank[:k])
        if rank == 0:
            print(f"[자극] frac {FRAC:.2f} = 가까운 {k}/{nfib} 섬유 (locus=윈도우중심)", flush=True)
    else:
        driven = set(np.unique(fiber_id[dist_e3 < RADIUS]).tolist())
    vstim = {}
    for fidv in set(int(f) for f in np.unique(fiber_id) if int(f) % nhost == rank):
        pc.set_gid2node(FIBER_OFFSET + fidv, rank)
        if fidv not in driven:
            spk = []
        elif TRAIN_FREQ > 0:
            spk = [STIM_T + kk * 1000.0 / TRAIN_FREQ for kk in range(TRAIN_NP)]   # train N펄스
        elif ISI > 0:
            spk = [STIM_T, STIM_T + ISI]                                          # 페어펄스
        else:
            spk = [STIM_T]                                                        # 단발
        vs = h.VecStim(); tv = h.Vector(spk)
        vs.play(tv); nc = h.NetCon(vs, None); pc.cell(FIBER_OFFSET + fidv, nc)
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
        print(f"[SC] {tot_sc:,} · 자극섬유 {len(driven):,} · {time.time()-t0:.0f}s", flush=True)

    # 내부 배선
    isel = np.where(np.isin(ipost, list(mine)))[0]; rng = np.random.default_rng(0); cc = {}; n_int = 0
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
                syn.setRNG(qg + 1, 800000 + ci, 4)   # 확률 시냅스 RNG 필수(GPU 직렬화 위해서도) — stream 4
                ncp = pc.gid_connect(pg, syn); ncp.weight[0] = 0.0 if GABAOFF else gs / 1000.0; ncp.delay = 1.0  # --gabaoff=탈억제
                keep.append((syn, ncp))
            n_int += 1
    pc.barrier(); tot_int = int(pc.allreduce(n_int, 1))
    if rank == 0:
        print(f"[내부] {tot_int:,} · 총 {tot_sc+tot_int:,} 시냅스 · {time.time()-t0:.0f}s", flush=True)

    # ── 파일모드(B): 모델을 디스크로 덤프하고 종료. 이후 special-core가 GPU로 실행 ──
    if WRITE_DIR:
        pc.set_maxstep(10); h.dt = 0.025; h.finitialize(-70)
        if rank == 0:
            os.makedirs(WRITE_DIR, exist_ok=True)
        pc.barrier()
        pc.nrncore_write(WRITE_DIR)
        if rank == 0:
            meta = {"n": N, "sc": tot_sc, "internal": tot_int, "stim_t": STIM_T,
                    "settle": SETTLE, "tstop": TSTOP, "radius": RADIUS, "nostim": int(NOSTIM), "tag": TAG}
            json.dump(meta, open(os.path.join(WRITE_DIR, "meta.json"), "w"))
            print(f"[WRITE] CoreNEURON 모델 저장 -> {WRITE_DIR} · tstop {TSTOP}ms · {time.time()-t0:.0f}s", flush=True)
        pc.barrier(); pc.done(); h.quit(); return

    # ── fEPSP 기록 준비 (--fepsp): fast_imem + 세그먼트 막전류 → 전극 (finitialize 前) ──
    frec = None; elec = None; enames = None
    if FEPSP:
        import fepsp_record as fr
        elec_mea = np.array([e["xyz_um"] for e in cfg["electrodes"]["list"]], float)
        enames_mea = [e["id"] + "(" + e["layer"] + ")" for e in cfg["electrodes"]["list"]]
        # 라미나 프로브(Ex4c CSD용): 기록기둥 u 아래 SO→SLM r축 32전극, 두께 중앙 w=0
        NLAM = 32; u_col = cfg["electrodes"]["center_local"]["u"]
        r_lam = np.linspace(-150.0, 500.0, NLAM)
        elec_lam = np.array([seed + Mrows @ np.array([u_col, float(r), 0.0]) for r in r_lam])
        enames_lam = ["L%02d_r%d" % (i, int(r_lam[i])) for i in range(NLAM)]
        n_mea = len(elec_mea)
        elec = np.vstack([elec_mea, elec_lam])
        enames = enames_mea + enames_lam
        FSTRIDE = int(sys.argv[sys.argv.index("--fstride") + 1]) if "--fstride" in sys.argv else 4
        frec = fr.FEPSPRecorder(elec, stride=FSTRIDE)            # 세그먼트 1/stride 기록(대규모 setup 회피)
        for g in mine:
            frec.add_cell(B.cells[g], XYZ[g], Rot.from_quat(Q[g][[1, 2, 3, 0]]))
        FRECDT = float(sys.argv[sys.argv.index("--frecdt") + 1]) if "--frecdt" in sys.argv else 0.25
        n_rt = int(round((TSTOP - (STIM_T - 10.0)) / FRECDT)) + 1
        rt = h.Vector(np.linspace(STIM_T - 10.0, TSTOP, n_rt))  # ★관측창만 기록(settle 제외)·TSTOP 안 넘겨 tvec>기록 불일치 방지
        frec.finalize(rec_tvec=rt)
        nseg_tot = int(pc.allreduce(frec.n_seg(), 1))
        if rank == 0:
            print(f"[fEPSP] 기록 세그먼트 {nseg_tot:,} · 전극 {len(elec)} · rec_dt {FRECDT}ms · 관측창만({STIM_T-10:.0f}~{TSTOP:.0f}ms, {int(rt.size())}점)", flush=True)

    if rank == 0:
        print(f"      구동 시작", flush=True)

    # 발화 기록 + 구동
    tspk = h.Vector(); idspk = h.Vector(); pc.spike_record(-1, tspk, idspk)
    if USE_GPU:
        # CoreNEURON GPU: 모델을 GPU로 1회 전송 후 단일 psolve(균일 dt). volley는 STIM_T에
        # VecStim이 자동 발생. 청크 루프(랭크간 allreduce 반복)는 GPU 재진입 오버헤드라 미사용.
        from neuron import coreneuron
        coreneuron.enable = True; coreneuron.gpu = True
        pc.set_maxstep(10); h.finitialize(-70); trun = time.time()
        h.dt = 0.025; pc.psolve(TSTOP)
        psolve_s = time.time() - trun
        if rank == 0:
            print(f"[GPU 구동 완료] tstop {TSTOP}ms · dt {h.dt} · psolve {psolve_s:.1f}s", flush=True)
    else:
        # CPU: 안정화(거친 dt) → 구동(고운 dt) + 5ms 청크마다 누적 스파이크 계측(과발화 즉시 감지)
        pc.set_maxstep(10); h.finitialize(-70); trun = time.time()
        h.dt = 0.25; pc.psolve(SETTLE)
        nsp0 = int(pc.allreduce(int(tspk.size()), 1))
        if rank == 0:
            print(f"[안정화 완료] {time.time()-trun:.0f}s · 안정화중 스파이크 {nsp0} (0이면 정상)", flush=True)
        h.dt = 0.025; t = SETTLE
        while t < TSTOP - 1e-6:
            t = min(t + 5.0, TSTOP); pc.psolve(t)
            nsp = int(pc.allreduce(int(tspk.size()), 1))   # 전 랭크 호출(집합통신)
            if rank == 0:
                print(f"  [구동] t={t-STIM_T:+.0f}ms(자극기준) · 누적스파이크 {nsp:,} · {time.time()-trun:.0f}s", flush=True)
        psolve_s = time.time() - trun

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    at = comm.gather(list(tspk), root=0); ai = comm.gather(list(idspk), root=0)
    fep = None
    if FEPSP:
        Vlsa = comm.allreduce(frec.potential_local("lsa"), op=MPI.SUM)   # 선원(근거리 정확)
        Vpsa = comm.allreduce(frec.potential_local("psa"), op=MPI.SUM)   # 점원(비교용)
        tfe = np.array(frec.times()); nt = min(len(tfe), Vlsa.shape[1])  # tvec가 TSTOP 넘겨 실제 기록보다 길 수 있음 → 실제 기록 길이로 클램프
        tfe = tfe[:nt]; Vlsa = Vlsa[:, :nt]; Vpsa = Vpsa[:, :nt]         # tfe·V·Im 시간축 길이 통일(불일치 IndexError 방지)
        keep = np.where(tfe >= (STIM_T - 10.0))[0]                       # settle 잘라냄(자극 10ms전부터)
        pos = np.array(frec._pos); soma = np.array(frec._soma, dtype=int)
        Im = np.array([np.array(v)[:nt] for v in frec.vecs]) if frec.n_seg() else np.zeros((0, nt))
        sstride = max(1, frec.n_seg() // 2500)                           # segI 공간 서브샘플(뷰 2500세그)
        didx = keep[np.linspace(0, len(keep) - 1, min(150, len(keep))).astype(int)]  # segI 시간 다운샘플(뷰 150)
        gp = comm.gather(pos[::sstride], root=0); gs = comm.gather(soma[::sstride], root=0)
        gI = comm.gather(Im[::sstride][:, didx] if Im.size else Im, root=0)
        if rank == 0:
            SP = np.vstack([a for a in gp if len(a)]); SS = np.concatenate([a for a in gs if len(a)]); SI = np.vstack([a for a in gI if len(a)])
            fep = dict(Vlsa=Vlsa[:, keep], Vpsa=Vpsa[:, keep], t=tfe[keep], t_seg=tfe[didx], SP=SP, SS=SS, SI=SI)
    if rank == 0:
        st = np.concatenate([np.array(x) for x in at]) if any(len(x) for x in at) else np.array([])
        sid = np.concatenate([np.array(x) for x in ai]).astype(int) if any(len(x) for x in ai) else np.array([], int)
        cellmask = sid < N               # 섬유(VecStim) gid(FIBER_OFFSET+) 제거 → 실제 뉴런 스파이크만
        st = st[cellmask]; sid = sid[cellmask]
        fired = set(sid.tolist())
        is_pc = np.array([mt[g] == "SP_PC" for g in range(N)])
        firedmask = np.array([g in fired for g in range(N)])
        nE = int(np.sum(firedmask & is_pc)); nI = int(np.sum(firedmask & ~is_pc))
        totE = int(np.sum(is_pc)); totI = int(N - totE)
        print(f"\n[구동 완료] tstop {TSTOP}ms · psolve {psolve_s:.0f}s", flush=True)
        print(f"[발화] 총 스파이크 {len(st):,} · 발화세포 {int(firedmask.sum())}/{N} ({100*firedmask.mean():.0f}%)", flush=True)
        print(f"       추체 {nE}/{totE} ({100*nE/totE:.0f}%) · 억제 {nI}/{totI} ({100*nI/max(totI,1):.0f}%)", flush=True)
        cnt = np.bincount(sid, minlength=N)
        obs_ms = TSTOP - SETTLE                              # 관측 창(ms)
        rate = 1000.0 * len(st) / max(obs_ms, 1) / N        # 평균 발화율(Hz/세포)
        note = "무자극 자발" if NOSTIM else "자극 volley 1회"
        print(f"       발화세포 평균 {cnt[firedmask].mean() if firedmask.any() else 0:.1f} 스파이크/세포 ({note})", flush=True)
        print(f"       망 평균 발화율 {rate:.3f} Hz/세포 (관측 {obs_ms:.0f}ms)", flush=True)
        np.savez_compressed(os.path.join(ROOT, "scratch", f"mpi_baseline{TAG}.npz"),
                            spk_t=st, spk_id=sid, fired=firedmask, is_pc=is_pc,
                            radius=RADIUS, stim_t=STIM_T, settle=SETTLE, n=N, nostim=int(NOSTIM))
        json.dump({"n": N, "sc": tot_sc, "internal": tot_int, "spikes": int(len(st)),
                   "active": int(firedmask.sum()), "activeE": nE, "activeI": nI,
                   "psolve_s": psolve_s, "radius": RADIUS, "nostim": int(NOSTIM),
                   "obs_ms": obs_ms, "rate_hz": rate},
                  open(os.path.join(ROOT, "scratch", f"mpi_baseline{TAG}.json"), "w"))
        print(f"[저장] scratch/mpi_baseline{TAG}.npz · 총 {time.time()-t0:.0f}s", flush=True)
        if fep is not None:
            base = fep["Vlsa"][:, fep["t"] < STIM_T].mean(axis=1)
            m = fep["t"] >= STIM_T
            print("[fEPSP] MEA 전극별 자극후 피크(LSA):", flush=True)
            for i in range(n_mea):
                seg = fep["Vlsa"][i] - base[i]; k = int(np.argmax(np.abs(seg[m])))
                print(f"   {enames[i]}: {seg[m][k]:+.4f} mV @ {fep['t'][m][k]-STIM_T:.1f}ms", flush=True)
            np.savez_compressed(os.path.join(ROOT, "scratch", f"mpi_fepsp{TAG}.npz"),
                                V_lsa=fep["Vlsa"], V_psa=fep["Vpsa"], t=fep["t"], t_seg=fep["t_seg"],
                                elec=elec, enames=np.array(enames), n_mea=n_mea,
                                segpos=fep["SP"], segsoma=fep["SS"], segI=fep["SI"],
                                stim_t=STIM_T, settle=SETTLE, tag=TAG)
            print(f"[fEPSP] 저장 scratch/mpi_fepsp{TAG}.npz · MEA {n_mea}+라미나 {len(enames)-n_mea} · segI {fep['SI'].shape}", flush=True)
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
