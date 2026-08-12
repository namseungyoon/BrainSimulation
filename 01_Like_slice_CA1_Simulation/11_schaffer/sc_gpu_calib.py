# -*- coding: utf-8 -*-
"""
11_schaffer/sc_gpu_calib.py  —  생리적 발화율 보정: SC 구동 세기(sc_g_pc) 스윕 → 정상상태 PC 발화율(Hz).

build-once(한 번 빌드) → SC→PC 전도도(sc_g_pc)를 바꿔가며 지속 구동(sustained) psolve →
정상상태 창([ss, tstop))에서 PC 평균 발화율(Hz) 측정. 목표: PC 0.3~2 Hz(생리적) 지점 찾기.
SC 구동 = NetStim noise=0 + 섬유별 위상·주기 무작위(빌드타임, Random123 없음 → GPU 안전).

실행(subset GPU 브래킷):
  mpiexec -n 4 <special> -mpi -python sc_gpu_calib.py --counts 1600,150,125,125 --tstop 500 --ss 300 \
     --sc_rate 150 --sc_g_int 3 --gpc_sweep 10,5,2,1,0.6,0.3 --coreneuron --gpu
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
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
SC_CLASS = "PC->PC (E2)"


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def log(m):
    if RANK == 0:
        print(m, flush=True)


def sr_or_dend(cell, is_pc, rng):
    segs = [s for s in cell.all if ".apic" in s.name()] if is_pc else []
    if not segs:
        segs = [s for s in cell.all if (".dend" in s.name() or ".apic" in s.name())]
    return (segs[rng.randint(len(segs))] if segs else cell.soma[0])(0.5)


def main():
    counts_s = argval("--counts", "1600,150,125,125")
    tstop = float(argval("--tstop", "500")); ss = float(argval("--ss", "300"))
    sc_rate = float(argval("--sc_rate", "150")); sc_g_int = float(argval("--sc_g_int", "3.0"))
    sc_pc = int(argval("--sc_pc", "60")); sc_int = int(argval("--sc_int", "40"))
    n_fiber = int(argval("--n_fiber", "800"))
    gpc_sweep = [float(x) for x in argval("--gpc_sweep", "10,5,2,1,0.6,0.3").split(",")]
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
    log(f"[보정] counts={counts_s}->{N}세포 · det · SC {sc_pc}/세포 fiber {n_fiber}@{sc_rate}Hz(지속) "
        f"INT {sc_g_int}nS · 창[{ss},{tstop})ms · sc_g_pc 스윕 {gpc_sweep} · {'GPU' if use_gpu else 'CPU'}")

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

    p = np.load(PRUNED, allow_pickle=True)
    pre = p["pre"]; post = p["post"]; cid = p["cls"]; classes = list(p["classes"].astype(str))
    rng = np.random.RandomState(1000 + RANK); n_syn = 0
    for i in range(len(pre)):
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:
            continue
        ga = orig2gid[a]; clsn = classes[int(cid[i])]
        try:
            pr = P3.CLASSES[clsn]; seg = net._placement(cells[gb], clsn, rng)
            syn = build_synapse(seg, pr, seeds=(i + 1, 1, 1), deterministic=True)
            ncc = pc.gid_connect(ga, syn); ncc.threshold = -20.0
            ncc.weight[0] = pr["g_nS"]; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_syn += 1
        except Exception:
            pass
    log(f"[2/3 내재연결] 랭크0 {n_syn} 시냅스")

    # 지속 SC 구동(noise=0 위상·주기 무작위) + SC→PC NetCon 보관(전도도 스윕용)
    frng = np.random.RandomState(9000 + RANK); ivl = 1000.0 / sc_rate
    fibers = []
    for k in range(n_fiber):
        ns = h.NetStim(); ns.number = 1e9; ns.noise = 0.0
        ns.interval = ivl * frng.uniform(0.7, 1.3); ns.start = frng.uniform(0.0, ivl)
        fibers.append(ns); keeph.append(ns)
    prm = P3.CLASSES[SC_CLASS]; scrng = np.random.RandomState(7000 + RANK)
    sc_pc_ncs = []; n_sc = 0
    for g in my:
        is_pc = gtype[g] == "PC"; k_syn = sc_pc if is_pc else sc_int
        for _ in range(k_syn):
            seg = sr_or_dend(cells[g], is_pc, scrng)
            syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000, 1, 1), deterministic=True)
            ncc = h.NetCon(fibers[scrng.randint(n_fiber)], syn)
            ncc.weight[0] = sc_g_int; ncc.delay = SYN_DELAY   # INT 고정, PC는 스윕서 설정
            keeph += [syn, ncc]; n_sc += 1
            if is_pc:
                sc_pc_ncs.append(ncc)
    log(f"[3/3 SC배선] 랭크0 {n_sc} SC 시냅스(PC {len(sc_pc_ncs)})")

    tvec = h.Vector(); gidvec = h.Vector(); pc.spike_record(-1, tvec, gidvec)
    is_pc_arr = np.array([gt == "PC" for gt in gtype])
    n_pc = int(pc.allreduce(sum(1 for g in my if gtype[g] == "PC"), 1))
    h.celsius = 34.0; h.cvode_active(0); h.dt = float(argval("--dt", str(net.DT))); pc.set_maxstep(10)
    if use_cn:
        from neuron import coreneuron
        coreneuron.enable = True; coreneuron.verbose = 0
        if use_gpu:
            coreneuron.gpu = True
        log(f"[CoreNEURON] {'GPU' if use_gpu else 'CPU'} · build-once→sc_g_pc 스윕")

    win = (tstop - ss) / 1000.0
    log(f"\n{'sc_g_pc':>8} | {'정상PC율(Hz)':>12} | {'발화PC%':>7}")
    rows = []
    for gpc in gpc_sweep:
        for ncc in sc_pc_ncs:
            ncc.weight[0] = gpc
        tvec.resize(0); gidvec.resize(0)
        h.finitialize(-70.0); pc.psolve(tstop)
        tt = np.array(tvec.to_python()); gg = np.array(gidvec.to_python(), dtype=int)
        m = (tt >= ss) & (tt < tstop)
        pc_spk = int(np.sum(is_pc_arr[gg[m]])) if len(gg[m]) else 0
        n_pc_spk = int(pc.allreduce(pc_spk, 1))
        fired = int(pc.allreduce(len(set(int(gi) for gi in gg[m] if is_pc_arr[gi])), 1))
        rate = n_pc_spk / max(1, n_pc) / win
        rows.append((gpc, rate, 100.0 * fired / max(1, n_pc)))
        log(f"{gpc:>8.2f} | {rate:>12.2f} | {100.0*fired/max(1,n_pc):>6.1f}")

    if RANK == 0:
        phys = [r for r in rows if 0.3 <= r[1] <= 2.0]
        print(f"\n[보정 결과] 생리적(0.3~2Hz) 구간 sc_g_pc: "
              + (", ".join(f"{r[0]}nS→{r[1]:.2f}Hz" for r in phys) if phys else "없음(침묵↔과활성, sparse 구간 부재 가능)"), flush=True)
        outdir = os.path.join(HERE, "sc_det_gpu"); os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "calib_gpc.npy"), np.array(rows, dtype=object))
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
