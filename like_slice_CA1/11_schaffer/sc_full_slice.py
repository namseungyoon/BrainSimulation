# -*- coding: utf-8 -*-
"""
11_schaffer/sc_full_slice.py  —  E2-c: 전 슬라이스 SC 경로 배치 + 포아송 SC 구동 + 증분 기록

run_mpi.py(전 슬라이스 gid 분산·증분 체크포인트·ETA·크래시안전) + sc_network.py(SC 배선)를 합침.
  - 세포: 전 17,647(또는 subset) gid 라운드로빈 분산, coarse nseg=1.
  - 내부 커넥텀(pruned) 배선 → 피드포워드 억제 포함(--no_inh 로 억제 차단).
  - SC 경로: 랭크별 독립 포아송 NetStim 섬유(각 CA3 축삭 대용) → 각 세포 SR/SO 수상돌기에 SC 시냅스(Ecker E2).
    (외부 Exp2Syn 임시구동 대신 '진짜 SC 경로'로 baseline 구동.)
  - 속도 레버: --dt(0.025→0.1)·--det(결정론 시냅스)·--sc_rate(섬유 포아송 Hz).
  - 기록: 스파이크(전 구간, 세그먼트마다 증분 CSV) + 옵션 20kHz Vm(대표 subset).

데드라인 안전: tstop 넉넉히 걸고, 세그먼트마다 CSV 저장 → 중간에 멈춰도 완료분 보존.

실행(전 슬라이스):
  mpiexec -n 10 <py> 11_schaffer/sc_full_slice.py --counts full --tstop 1000 --seg_ms 100 --dt 0.1 --det --sc_rate 3
스모크(소규모):
  mpiexec -n 4 <py> 11_schaffer/sc_full_slice.py --counts 300,80,60,60 --tstop 50 --seg_ms 25 --dt 0.1 --det --sc_rate 5
"""
import os
import sys
import csv
import time
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
import network_lib as net                              # noqa: E402
from common.cell_loader import load_cell               # noqa: E402
from synapse_pair import build_synapse                 # noqa: E402
import params_table3 as P3                             # noqa: E402

MODELS = os.path.join(SHARED, "models")
CSVDIR = os.path.join(HERE, "sc_full_spikes")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
SC_CLASS = "PC->PC (E2)"


def log(m):
    if RANK == 0:
        print(m, flush=True)


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def sr_or_dend(cell, is_pc, rng):
    """PC면 apical(SR) 우선, 인터뉴런이면 임의 수상돌기(SLM·소마 제외)."""
    if is_pc:
        segs = [s for s in cell.all if ".apic" in s.name()]
    else:
        segs = []
    if not segs:
        segs = [s for s in cell.all if (".dend" in s.name() or ".apic" in s.name())]
    return (segs[rng.randint(len(segs))] if segs else cell.soma[0])(0.5)


def main():
    counts_s = argval("--counts", "full")
    tstop = float(argval("--tstop", "1000")); seg_ms = float(argval("--seg_ms", "100"))
    n_seg = int(round(tstop / seg_ms))
    dt = float(argval("--dt", str(net.DT)))
    det = "--det" in sys.argv
    sc_rate = float(argval("--sc_rate", "3.0"))          # 섬유당 포아송 Hz(CA3 발화 대용)
    n_fiber = int(argval("--n_fiber", "800"))
    sc_pc = int(argval("--sc_pc", "60")); sc_int = int(argval("--sc_int", "40"))
    sc_g_pc = float(argval("--sc_g_pc", "1.0")); sc_g_int = float(argval("--sc_g_int", "1.0"))
    no_inh = "--no_inh" in sys.argv
    vm_khz = float(argval("--vm_khz", "0")); vm_cells_n = int(argval("--vm_cells", "5"))

    if RANK == 0:
        os.makedirs(CSVDIR, exist_ok=True)
    pc.barrier()

    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str)
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
    gtype = [t4[o] for o in keep]; gpos = xyz[keep]
    log(f"[E2-c] counts={counts_s}->{N}세포 · dt={dt} det={det} · SC {n_fiber}섬유@{sc_rate}Hz(포아송) "
        f"PC {sc_pc}syn/{sc_g_pc}nS INT {sc_int}syn/{sc_g_int}nS · no_inh={no_inh} · tstop={tstop:.0f} seg{seg_ms:.0f}x{n_seg}")

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
    t_build = time.time() - t0; pc.barrier()
    log(f"[1/4 구축] 랭크0 {len(my)}세포 · {t_build:.0f}s")

    # ── 내부 커넥텀 (--no_inh 면 억제 연결 skip) ──────────────────────────────
    p = np.load(PRUNED, allow_pickle=True)
    pre = p["pre"]; post = p["post"]; cid = p["cls"]; classes = list(p["classes"].astype(str))
    inh_cls = set(i for i, cl in enumerate(classes) if not cl.startswith("PC->"))
    rng = np.random.RandomState(1000 + RANK); t0 = time.time(); n_syn = 0
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
            pr = P3.CLASSES[cls]; seg = net._placement(cells[gb], cls, rng)
            syn = build_synapse(seg, pr, seeds=(i + 1, 1, 1), deterministic=det)
            nc = pc.gid_connect(ga, syn); nc.threshold = -20.0
            nc.weight[0] = pr["g_nS"]; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_syn += 1
        except Exception:
            pass
    t_wire = time.time() - t0; n_syn_all = int(pc.allreduce(n_syn, 1)); pc.barrier()
    log(f"[2/4 내부연결] 총 {n_syn_all:,} 시냅스 · {t_wire:.0f}s" + (" (억제 차단)" if no_inh else ""))

    # ── SC 포아송 섬유(랭크별 독립·재현) + SC 시냅스 ──────────────────────────
    t0 = time.time(); fibers = []
    for k in range(n_fiber):
        ns = h.NetStim(); ns.interval = 1000.0 / sc_rate; ns.number = 1e9; ns.start = 0; ns.noise = 1.0
        r = h.Random(); r.Random123(RANK * 100000 + k, 7, 0); r.negexp(1); ns.noiseFromRandom(r)
        fibers.append(ns); keeph += [ns, r]
    prm = P3.CLASSES[SC_CLASS]; scrng = np.random.RandomState(7000 + RANK); n_sc = 0
    for g in my:
        is_pc = gtype[g] == "PC"; k_syn = sc_pc if is_pc else sc_int; gnS = sc_g_pc if is_pc else sc_g_int
        for _ in range(k_syn):
            seg = sr_or_dend(cells[g], is_pc, scrng)
            syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000, 1, 1), deterministic=det)
            nc = h.NetCon(fibers[scrng.randint(n_fiber)], syn); nc.weight[0] = gnS; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_sc += 1
    t_sc = time.time() - t0; n_sc_all = int(pc.allreduce(n_sc, 1)); pc.barrier()
    log(f"[3/4 SC배선] 총 {n_sc_all:,} SC 시냅스 · {t_sc:.0f}s")

    # ── 기록 ──────────────────────────────────────────────────────────────────
    tvec = h.Vector(); gidvec = h.Vector(); pc.spike_record(-1, tvec, gidvec)
    n_pc = int(pc.allreduce(sum(1 for g in my if gtype[g] == "PC"), 1))
    vm_vecs = []; vm_gids = []; trec = None
    if vm_khz > 0:
        dt_rec = 1000.0 / (vm_khz * 1000.0)
        for g in my[:vm_cells_n]:
            v = h.Vector(); v.record(cells[g].soma[0](0.5)._ref_v, dt_rec); vm_vecs.append(v); vm_gids.append(g)
        trec = h.Vector(); trec.record(h._ref_t, dt_rec)
        np.save(os.path.join(CSVDIR, f"_rank{RANK}_vmgids.npy"), np.array(vm_gids, dtype=int))

    if RANK == 0:
        np.savez(os.path.join(CSVDIR, "SC_positions.npz"), gid=np.arange(N), xyz=gpos, type=np.array(gtype))
        with open(os.path.join(CSVDIR, "PROGRESS.txt"), "w", encoding="utf-8") as f:
            f.write(f"N={N} tstop={tstop} seg_ms={seg_ms} dt={dt} det={det} sc_rate={sc_rate} "
                    f"build={t_build:.0f}s wire={t_wire:.0f}s sc={t_sc:.0f}s\n")

    h.celsius = 34.0; h.cvode_active(0); h.dt = dt; pc.set_maxstep(10); h.finitialize(-70.0)
    log(f"[4/4 실행] 증분 psolve {n_seg}세그(seg={seg_ms:.0f}ms, dt={dt}) …")
    t_run0 = time.time()
    for s in range(n_seg):
        a, b = s * seg_ms, (s + 1) * seg_ms
        pc.psolve(b)
        tt = np.array(tvec.to_python(), dtype=float); gg = np.array(gidvec.to_python(), dtype=int)
        m = (tt >= a) & (tt < b)
        with open(os.path.join(CSVDIR, f"_rank{RANK}_seg{s:03d}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["gid", "t_ms"])
            for gi, ti in zip(gg[m], tt[m]):
                w.writerow([int(gi), round(float(ti), 3)])
        n_all = int(pc.allreduce(int(m.sum()), 1))
        if vm_khz > 0:
            arr = np.array([v.to_python() for v in vm_vecs], dtype=np.float32)
            np.save(os.path.join(CSVDIR, f"_rank{RANK}_vm.npy"), arr)
            if RANK == 0 and trec is not None:
                np.save(os.path.join(CSVDIR, "SC_vm_time_ms.npy"), np.array(trec.to_python(), dtype=np.float32))
        pc.barrier()
        if RANK == 0:
            el = time.time() - t_run0; eta = el / (s + 1) * (n_seg - s - 1)
            msg = (f"  seg{s:03d} [{int(a)}-{int(b)}ms] 스파이크 {n_all:,} · 경과 {el/60:.1f}min · "
                   f"세그당 {el/(s+1)/60:.2f}min · ETA {eta/3600:.2f}h")
            print(msg, flush=True)
            with open(os.path.join(CSVDIR, "PROGRESS.txt"), "a", encoding="utf-8") as f:
                f.write(msg + "\n")
    t_sim = time.time() - t_run0; t_sim_max = float(pc.allreduce(t_sim, 2))
    log(f"===== 구동 {t_sim_max/3600:.2f}h (dt={dt} det={det}) =====")

    # ── 랭크0 병합 + 발화율 요약 ────────────────────────────────────────────
    if RANK == 0:
        allrows = []
        for s in range(n_seg):
            for rk in range(NHOST):
                fn = os.path.join(CSVDIR, f"_rank{rk}_seg{s:03d}.csv")
                if not os.path.exists(fn):
                    continue
                with open(fn, encoding="utf-8") as f:
                    rd = csv.reader(f); next(rd, None)
                    for gid_s, t_s in rd:
                        g = int(gid_s); allrows.append((g, gtype[g], float(t_s)))
        allrows.sort(key=lambda r: (r[2], r[0]))
        with open(os.path.join(CSVDIR, "SC_spikes_all.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["gid", "type", "t_ms"]); w.writerows(allrows)
        fired = len({r[0] for r in allrows})
        dur_s = tstop / 1000.0
        npc = sum(1 for gt in gtype if gt == "PC"); nint = max(1, N - npc)
        pc_sp = sum(1 for r in allrows if r[1] == "PC"); int_sp = len(allrows) - pc_sp
        pc_rate = pc_sp / max(1, npc) / dur_s; int_rate = int_sp / nint / dur_s
        summary = (f"[완료] 스파이크 {len(allrows):,} · 발화세포 {fired}/{N}({100*fired/max(1,N):.0f}%) · "
                   f"PC {pc_rate:.2f}Hz · INT {int_rate:.2f}Hz · 구동 {t_sim_max/3600:.2f}h")
        print(summary, flush=True)
        with open(os.path.join(CSVDIR, "PROGRESS.txt"), "a", encoding="utf-8") as f:
            f.write(summary + "\n[DONE]\n")
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
