# -*- coding: utf-8 -*-
"""
09_run/run_mpi.py  —  전체 슬라이스 MPI 병렬 구동 (ParallelContext gid 배선)

물리 10코어를 모두 쓰기 위해 NEURON ParallelContext 로 세포를 gid 라운드로빈 분산.
  - 세포: gid = 세포 인덱스. gid % NHOST == RANK 인 세포만 그 랭크가 소유·구축.
  - 연결: post 를 소유한 랭크가 시냅스 생성 → pc.gid_connect(pre_gid, syn) (랭크 넘나듦 OK).
  - 외부구동: 소유 세포에 로컬 Poisson.
  - 스파이크: pc.spike_record 로 (gid,t) 수집 → 랭크별 CSV → 랭크0가 병합/세그먼트화.

실행(중요: h.nrnmpi_init() 먼저 → mpiexec 로 런치):
  mpiexec -n 10 <conda python> 09_run/run_mpi.py --counts full --tstop 1000 --coarse --seg_ms 100

작은 검증:
  mpiexec -n 4 <conda python> 09_run/run_mpi.py --counts 300,80,60,60 --tstop 50 --coarse --seg_ms 50
"""
import os
import sys
import csv
import time
import numpy as np
from neuron import h

# ── MPI 초기화 (ParallelContext 이전에!) ──────────────────────────────────────
h.nrnmpi_init()
pc = h.ParallelContext()
RANK = int(pc.id())
NHOST = int(pc.nhost())

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, os.path.join(PAPER, "04_network"))
import network_lib as net                              # noqa: E402
from common.cell_loader import load_cell               # noqa: E402
from synapse_pair import build_synapse                 # noqa: E402
import params_table3 as P3                             # noqa: E402

MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(HERE, "figures"); CSVDIR = os.path.join(HERE, "spikes")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    global CSVDIR
    if "--outdir" in sys.argv:                    # 벤치/임시 출력 분리(실측 데이터 보호)
        CSVDIR = os.path.join(HERE, sys.argv[sys.argv.index("--outdir") + 1])
    if RANK == 0:
        os.makedirs(OUT, exist_ok=True); os.makedirs(CSVDIR, exist_ok=True)
    pc.barrier()

    counts_s = argval("--counts", "full")
    tstop = float(argval("--tstop", "1000"))
    seg_ms = float(argval("--seg_ms", "100"))
    coarse = "--coarse" in sys.argv
    use_cn = "--coreneuron" in sys.argv          # WSL CoreNEURON(CPU/GPU) 가속 엔진
    n_seg = int(round(tstop / seg_ms))
    vm_khz = float(argval("--vm_khz", "0"))       # 0=끔. 20 → 소마 막전위 20kHz 기록
    vm_cells_n = int(argval("--vm_cells", "-1"))  # 랭크별 기록 세포수(-1=소유 전부)
    drive_scale = float(argval("--drive_scale", "1.0"))  # 외부 Poisson 구동 weight 배율(튜닝용)

    # ── 세포 메타 (전 랭크 로드) ─────────────────────────────────────────────
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])
    Ntot = len(xyz)

    if counts_s == "full":
        keep = np.arange(Ntot)
    else:
        counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, counts_s.split(","))))
        ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
        ks = []
        for tn, k in counts.items():
            ids = np.where(t4 == tn)[0]; ks.extend(ids[np.argsort(dist[ids])[:k]].tolist())
        keep = np.array(sorted(ks))
    N = len(keep)
    # gid = keep 배열 내 위치(0..N-1). 원본→gid 매핑.
    orig2gid = {int(o): g for g, o in enumerate(keep)}
    gid_type = [t4[o] for o in keep]
    gid_pos = xyz[keep]
    log(f"[설정] counts={counts_s} → {N}세포 · tstop={tstop:.0f}ms seg={seg_ms}ms×{n_seg} "
        f"coarse={coarse} · NHOST={NHOST}")

    # ── 세포 구축 (소유분만) ─────────────────────────────────────────────────
    type_dir = net.load_representatives(MODELS)
    keeph = []
    my_gids = [g for g in range(N) if g % NHOST == RANK]
    cells = {}
    t0 = time.time()
    ncomp_local = 0
    for g in my_gids:
        cell, _ = load_cell(type_dir[gid_type[g]], gid=g)
        if coarse:
            for sec in cell.all:
                sec.nseg = 1
        cells[g] = cell
        soma = cell.soma[0]
        nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma); nc.threshold = -20.0
        pc.set_gid2node(g, RANK)
        pc.cell(g, nc)
        keeph.append(nc)
        ncomp_local += sum(s.nseg for s in cell.all)
    t_build = time.time() - t0
    pc.barrier()
    log(f"[1/4 구축] 각 랭크 소유 세포 구축 완료 (랭크0: {len(my_gids)}세포, {ncomp_local}구획)")

    # ── 연결 (post 소유 랭크가 시냅스 생성 + gid_connect) ────────────────────
    p = np.load(PRUNED, allow_pickle=True)
    pre = p["pre"]; post = p["post"]; cid = p["cls"]; classes = list(p["classes"].astype(str))
    # 원본 인덱스 → gid (subset 이면 일부만 유효)
    t0 = time.time(); n_syn = 0; n_fail = 0
    rng = np.random.RandomState(1000 + RANK)
    for i in range(len(pre)):
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:        # post 를 이 랭크가 소유할 때만
            continue
        ga = orig2gid[a]
        cls = classes[int(cid[i])]
        try:
            pr = P3.CLASSES[cls]
            seg = net._placement(cells[gb], cls, rng)
            syn = build_synapse(seg, pr, seeds=(i + 1, 1, 1), deterministic=False)
            nc = pc.gid_connect(ga, syn)
            nc.threshold = -20.0; nc.weight[0] = pr["g_nS"]; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_syn += 1
        except Exception as ex:
            n_fail += 1
            if n_fail <= 2:
                print(f"  [R{RANK} 시냅스 건너뜀] edge {i} {cls}: {ex}", flush=True)
    t_wire = time.time() - t0
    n_syn_all = int(pc.allreduce(n_syn, 1))    # 1 = sum
    pc.barrier()
    log(f"[2/4 연결] 총 시냅스 {n_syn_all:,} (랭크0 {n_syn:,}, 실패 {n_fail})")

    # ── 외부 구동 (소유 세포 로컬 Poisson) ───────────────────────────────────
    for g in my_gids:
        n_stim, w = net.DRIVE[gid_type[g]]
        for j in range(n_stim):
            ns = h.NetStim(); ns.interval = 1000.0 / net.DRIVE_RATE; ns.number = 1e9
            ns.start = 0; ns.noise = 1.0
            r = h.Random(); r.Random123(g, j, 0); r.negexp(1); ns.noiseFromRandom(r)
            syn = h.Exp2Syn(cells[g].soma[0](0.5)); syn.tau1 = 0.2; syn.tau2 = 2.0; syn.e = 0.0
            ncd = h.NetCon(ns, syn); ncd.weight[0] = w * drive_scale; ncd.delay = 0.0
            keeph += [ns, r, syn, ncd]

    # ── 스파이크 기록 (gid 기반, 이 랭크 소유분 전체) ────────────────────────
    tvec = h.Vector(); gidvec = h.Vector()
    pc.spike_record(-1, tvec, gidvec)

    # ── 막전위 20kHz 기록(옵션): 소마 Vm, 세그먼트마다 float32 npy 저장 ──────
    vm_vecs = []; vm_gids = []; trec = None
    if vm_khz > 0:
        dt_rec = 1000.0 / (vm_khz * 1000.0)         # kHz→ms (20kHz→0.05ms)
        rec_gids = my_gids if vm_cells_n < 0 else my_gids[:vm_cells_n]
        for g in rec_gids:
            v = h.Vector(); v.record(cells[g].soma[0](0.5)._ref_v, dt_rec)
            vm_vecs.append(v); vm_gids.append(g)
        trec = h.Vector(); trec.record(h._ref_t, dt_rec)
        np.save(os.path.join(CSVDIR, f"_rank{RANK}_vmgids.npy"), np.array(vm_gids, dtype=int))
        n_rec_all = int(pc.allreduce(len(vm_gids), 1))
        log(f"[Vm기록] {vm_khz:.0f}kHz(Δ{dt_rec}ms) 소마 막전위 · 총 {n_rec_all}세포 "
            f"(float32, 세그먼트마다 저장)")

    # 좌표 저장(그림/GIF용) — 실행 전에 미리 저장(크래시 대비)
    if RANK == 0:
        np.savez(os.path.join(CSVDIR, "FULL_positions.npz"),
                 gid=np.arange(N), xyz=gid_pos, type=np.array(gid_type))
        # 진행상황 헤더
        with open(os.path.join(CSVDIR, "PROGRESS.txt"), "w", encoding="utf-8") as f:
            f.write(f"N={N} tstop={tstop} seg_ms={seg_ms} n_seg={n_seg} NHOST={NHOST} "
                    f"build_rank0={t_build:.0f}s wire_rank0={t_wire:.0f}s\n")

    # ── 실행 (증분 세그먼트: 매 세그먼트 CSV 저장 → 중간 크래시 시 진행분 보존) ──
    h.celsius = 34.0; h.cvode_active(0); h.dt = net.DT
    pc.set_maxstep(10)
    if use_cn:
        from neuron import coreneuron
        coreneuron.enable = True; coreneuron.verbose = 0
        log("[CoreNEURON] 가속 엔진 활성화 (CPU 백엔드)")
    h.finitialize(-70.0)
    log(f"[3/4 실행] 증분 psolve {n_seg}세그먼트 (seg={seg_ms}ms, dt={net.DT}) …")
    t_run0 = time.time()
    for s in range(n_seg):
        a, b = s * seg_ms, (s + 1) * seg_ms
        pc.psolve(b)
        # 이 세그먼트 [a,b) 스파이크만 랭크별 CSV로 즉시 저장
        tt = np.array(tvec.to_python(), dtype=float)
        gg = np.array(gidvec.to_python(), dtype=int)
        m = (tt >= a) & (tt < b)
        segfn = os.path.join(CSVDIR, f"_rank{RANK}_seg{s:02d}.csv")
        with open(segfn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["gid", "t_ms"])
            for gg_i, tt_i in zip(gg[m], tt[m]):
                w.writerow([int(gg_i), round(float(tt_i), 3)])
        n_seg_local = int(m.sum())
        n_seg_all = int(pc.allreduce(n_seg_local, 1))
        # Vm 안전 저장(전체 누적본 overwrite, float32)
        if vm_khz > 0:
            arr = np.array([v.to_python() for v in vm_vecs], dtype=np.float32)
            np.save(os.path.join(CSVDIR, f"_rank{RANK}_vm.npy"), arr)
            if RANK == 0 and trec is not None:
                np.save(os.path.join(CSVDIR, "FULL_vm_time_ms.npy"),
                        np.array(trec.to_python(), dtype=np.float32))
        pc.barrier()
        if RANK == 0:
            el = time.time() - t_run0
            eta = el / (s + 1) * (n_seg - s - 1)
            msg = (f"  seg{s:02d} [{int(a)}-{int(b)}ms] 스파이크 {n_seg_all:,} · "
                   f"경과 {el/60:.1f}min · 세그당 {el/(s+1)/60:.1f}min · ETA {eta/3600:.1f}h")
            print(msg, flush=True)
            with open(os.path.join(CSVDIR, "PROGRESS.txt"), "a", encoding="utf-8") as f:
                f.write(msg + "\n")
    t_sim = time.time() - t_run0

    t_build_max = float(pc.allreduce(t_build, 2))   # 2 = max
    t_sim_max = float(pc.allreduce(t_sim, 2))
    t_wire_max = float(pc.allreduce(t_wire, 2))
    log(f"===== 시간(최대 랭크) =====")
    log(f"  구축 {t_build_max:.1f}s · 연결 {t_wire_max:.1f}s · 구동 {t_sim_max:.1f}s "
        f"({t_sim_max/3600:.2f}h)")

    # ── 랭크0: 세그먼트 CSV 병합(gid+type) + 합본 + 요약 ─────────────────────
    if RANK == 0:
        allrows = []
        for s in range(n_seg):
            a, b = s * seg_ms, (s + 1) * seg_ms
            seg_rows = []
            for rk in range(NHOST):
                fn = os.path.join(CSVDIR, f"_rank{rk}_seg{s:02d}.csv")
                if not os.path.exists(fn):
                    continue
                with open(fn, encoding="utf-8") as f:
                    rd = csv.reader(f); next(rd, None)
                    for gid_s, t_s in rd:
                        g = int(gid_s); seg_rows.append((g, gid_type[g], float(t_s)))
            seg_rows.sort(key=lambda r: (r[2], r[0]))
            outfn = os.path.join(CSVDIR, f"FULL_spikes_seg{s:02d}_{int(a)}-{int(b)}ms.csv")
            with open(outfn, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["gid", "type", "t_ms"]); w.writerows(seg_rows)
            allrows.extend(seg_rows)
        allfn = os.path.join(CSVDIR, "FULL_spikes_all.csv")
        with open(allfn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["gid", "type", "t_ms"]); w.writerows(allrows)
        fired = len({r[0] for r in allrows})
        by_type = {}
        for r in allrows:
            by_type.setdefault(r[1], 0); by_type[r[1]] += 1
        summary = (f"[완료] 스파이크 {len(allrows):,} · 발화세포 {fired}/{N} ({100*fired/N:.1f}%) "
                   f"· 타입별 {by_type} · 구동 {t_sim_max/3600:.2f}h")
        print(summary, flush=True)
        print(f"[CSV] 세그먼트 {n_seg}개 + 합본 + 좌표 → {CSVDIR}", flush=True)
        with open(os.path.join(CSVDIR, "PROGRESS.txt"), "a", encoding="utf-8") as f:
            f.write(summary + "\n[DONE]\n")

    pc.barrier()
    pc.done()
    h.quit()


if __name__ == "__main__":
    main()
