# -*- coding: utf-8 -*-
"""
11_schaffer/sc_gpu_io.py  —  E3c: 전슬라이스 GPU 결정론 SC 자극 I-O 곡선 + 억제 차단.

sc_io_curve.py(subset·확률·CPU)를 GPU 전슬라이스·결정론으로 이식.
  - 세포/시냅스 1회 구축(build-once) → SC 볼리 활성비율 스윕 × 억제{정상·차단} 루프 psolve.
  - 볼리 = NetStim(noise=0, number 0/1) → Random123 없음(GPU 세그폴트 무관).
  - 결정론 시냅스(deterministic=True) → GPU 실행 가능(E8.2 실증).
  - 억제 차단 = 내재 억제 NetCon weight 0 토글(재구축 불필요).
  - 측정: 자극 후 [stim_t, tstop) PC 발화 비율.

★ 검증 포인트: CoreNEURON GPU에서 build-once→다중 psolve + 조건간 변경(자극 number·억제 weight)이
  제대로 반영되는가. 소규모(--counts a,b,c,d)로 먼저 확인 후 전슬라이스(--counts full).

실행(소규모 GPU 검증):
  <special> -mpi -python sc_gpu_io.py --counts 300,80,60,60 --sweep 0.1,1.0 --stim_t 10 --tstop 60 \
     --sc_g_pc 1.0 --sc_g_int 6.0 --inh_scale 3.0 --coreneuron --gpu
실행(전슬라이스):
  mpiexec -n 4 <special> -mpi -python sc_gpu_io.py --counts full --sweep 0.1,0.2,0.4,0.6,0.8,1.0 ... --coreneuron --gpu
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
import network_lib as net                              # noqa: E402
from common.cell_loader import load_cell               # noqa: E402
from synapse_pair import build_synapse                 # noqa: E402
import params_table3 as P3                             # noqa: E402

MODELS = os.environ.get("MODELS_DIR") or os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
N_FIBER = 100
SC_CLASS = "PC->PC (E2)"


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def log(m):
    if RANK == 0:
        print(m, flush=True)


def sr_or_dend(cell, is_pc, rng):
    if is_pc:
        segs = [s for s in cell.all if ".apic" in s.name()]
    else:
        segs = []
    if not segs:
        segs = [s for s in cell.all if (".dend" in s.name() or ".apic" in s.name())]
    return (segs[rng.randint(len(segs))] if segs else cell.soma[0])(0.5)


def main():
    counts_s = argval("--counts", "300,80,60,60")
    stim_t = float(argval("--stim_t", "10")); tstop = float(argval("--tstop", "60"))
    sc_per_cell = int(argval("--sc_per_cell", "60"))
    sc_g_pc = float(argval("--sc_g_pc", "1.0")); sc_g_int = float(argval("--sc_g_int", "6.0"))
    inh_scale = float(argval("--inh_scale", "3.0"))
    sweep = [float(x) for x in argval("--sweep", "0.1,1.0").split(",")]
    n_fiber = int(argval("--n_fiber", str(N_FIBER)))
    use_cn = "--coreneuron" in sys.argv; use_gpu = "--gpu" in sys.argv

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
    gtype = [t4[o] for o in keep]
    log(f"[E3c] counts={counts_s}->{N}세포 · det=True · SC {sc_per_cell}/세포(PC {sc_g_pc}/INT {sc_g_int}nS) "
        f"fiber {n_fiber} · 자극t={stim_t} tstop={tstop} · 억제×{inh_scale} · 스윕 {sweep} · "
        f"엔진={'GPU' if use_gpu else ('CoreNEURON-CPU' if use_cn else 'plain')}")

    type_dir = net.load_representatives(MODELS)
    my = [g for g in range(N) if g % NHOST == RANK]; cells = {}; keeph = []
    for g in my:
        cell, _ = load_cell(type_dir[gtype[g]], gid=g)
        for sec in cell.all:
            sec.nseg = 1
        cells[g] = cell
        s = cell.soma[0]; nc = h.NetCon(s(0.5)._ref_v, None, sec=s); nc.threshold = -20.0
        pc.set_gid2node(g, RANK); pc.cell(g, nc); keeph.append(nc)
    pc.barrier(); log(f"[1/3 구축] 랭크0 {len(my)}세포")

    # 내재 커넥텀(결정론) — 억제 NetCon 보관(차단 토글용)
    p = np.load(PRUNED, allow_pickle=True)
    pre = p["pre"]; post = p["post"]; cid = p["cls"]; classes = list(p["classes"].astype(str))
    inh_cls = set(i for i, cl in enumerate(classes) if not cl.startswith("PC->"))
    rng = np.random.RandomState(1000 + RANK); inh_ncs = []; n_syn = 0
    for i in range(len(pre)):
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:
            continue
        ga = orig2gid[a]; ci = int(cid[i]); clsn = classes[ci]
        try:
            pr = P3.CLASSES[clsn]; seg = net._placement(cells[gb], clsn, rng)
            syn = build_synapse(seg, pr, seeds=(i + 1, 1, 1), deterministic=True)
            ncc = pc.gid_connect(ga, syn); ncc.threshold = -20.0
            ncc.weight[0] = pr["g_nS"]; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_syn += 1
            if ci in inh_cls:
                inh_ncs.append((ncc, pr["g_nS"]))
        except Exception:
            pass
    n_syn_all = int(pc.allreduce(n_syn, 1)); pc.barrier()
    log(f"[2/3 내재연결] 총 {n_syn_all:,} 시냅스(랭크0 억제 {len(inh_ncs)})")

    # SC fibers(볼리 NetStim, noise=0) + SC 시냅스(결정론)
    fibers = [h.NetStim() for _ in range(n_fiber)]
    for ns in fibers:
        ns.start = stim_t; ns.interval = 1; ns.noise = 0; ns.number = 0; keeph.append(ns)
    prm = P3.CLASSES[SC_CLASS]; scrng = np.random.RandomState(7000 + RANK); n_sc = 0
    for g in my:
        is_pc = gtype[g] == "PC"
        for _ in range(sc_per_cell):
            seg = sr_or_dend(cells[g], is_pc, scrng)
            syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000, 1, 1), deterministic=True)
            ncc = h.NetCon(fibers[scrng.randint(n_fiber)], syn)
            ncc.weight[0] = sc_g_pc if is_pc else sc_g_int; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_sc += 1
    n_sc_all = int(pc.allreduce(n_sc, 1)); pc.barrier()
    log(f"[3/3 SC배선] 총 {n_sc_all:,} SC 시냅스")

    tvec = h.Vector(); gidvec = h.Vector(); pc.spike_record(-1, tvec, gidvec)
    is_pc_arr = np.array([gt == "PC" for gt in gtype])
    n_pc = int(pc.allreduce(sum(1 for g in my if gtype[g] == "PC"), 1))
    h.celsius = 34.0; h.cvode_active(0); h.dt = net.DT; pc.set_maxstep(10)
    if use_cn:
        from neuron import coreneuron
        coreneuron.enable = True; coreneuron.verbose = 0
        if use_gpu:
            coreneuron.gpu = True
        log(f"[CoreNEURON] {'GPU' if use_gpu else 'CPU'} 백엔드 · build-once→다중 psolve")

    def run_point(sc_active):
        n_act = int(round(sc_active * n_fiber))
        for k, ns in enumerate(fibers):
            ns.number = 1 if k < n_act else 0
        tvec.resize(0); gidvec.resize(0)
        h.finitialize(-70.0); pc.psolve(tstop)
        tt = np.array(tvec.to_python()); gg = np.array(gidvec.to_python(), dtype=int)
        m = (tt >= stim_t) & (tt < tstop)
        fired = set(int(gi) for gi in gg[m] if is_pc_arr[gi])
        return int(pc.allreduce(len(fired), 1))

    results = {"control": [], "block": []}
    for cond in ["control", "block"]:
        for ncc, bw in inh_ncs:
            ncc.weight[0] = (bw * inh_scale) if cond == "control" else 0.0
        log(f"\n== {cond} (억제 {'ON x'+str(inh_scale) if cond=='control' else 'OFF=차단'}) ==")
        log(f"{'SC%':>6} | {'발화PC':>7} | {'비율%':>6}")
        for sa in sweep:
            fired = run_point(sa)
            frac = 100.0 * fired / max(1, n_pc)
            results[cond].append((sa, fired, frac))
            log(f"{sa*100:>5.0f}% | {fired:>7d} | {frac:>5.1f}")

    if RANK == 0:
        gap = max(abs(results["block"][i][2] - results["control"][i][2]) for i in range(len(sweep)))
        # 검증: 스윕점이 서로 다른가(자극 반영) + 조건이 다른가(억제 토글 반영)
        ctrl_vals = [r[2] for r in results["control"]]
        varies = (max(ctrl_vals) - min(ctrl_vals)) if len(ctrl_vals) > 1 else 0.0
        print(f"\n[E3c 요약] N={N} · 억제 gap 최대 {gap:.1f}%p · control I-O 변동 {varies:.1f}%p", flush=True)
        print(f"[검증] 자극반영={'OK' if varies>0.5 else 'FAIL(스윕점 동일)'} · "
              f"억제토글반영={'OK' if gap>0.5 else 'FAIL(조건 동일)'}", flush=True)
        outdir = os.path.join(HERE, "sc_det_gpu"); os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "e3c_io_results.npy"),
                np.array([(cond, r[0], r[1], r[2]) for cond in results for r in results[cond]], dtype=object))
        np.save(os.path.join(outdir, "e3c_io_meta.npy"),
                np.array({"N": N, "n_pc": n_pc, "sc_g_pc": sc_g_pc, "sc_g_int": sc_g_int,
                          "inh_scale": inh_scale, "stim_t": stim_t, "tstop": tstop, "sweep": sweep}, dtype=object))
        print(f"[저장] {outdir}/e3c_io_results.npy", flush=True)
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
