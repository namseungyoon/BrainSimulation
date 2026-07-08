# -*- coding: utf-8 -*-
"""
11_schaffer/sc_io_curve.py  —  E3: SC 자극 입출력(I-O) 곡선 + gabazine 대조 (Romani Fig.4 재현 *시도*)

세포를 **한 번만 구축**하고, 활성 SC fiber 비율(sc_active)을 5→100% 스윕하며 자극 후 발화한 PC 비율을 측정.
억제 시냅스 NetCon weight를 0으로 토글 = gabazine(GABA 차단) 모사(재구축 불필요).
  - 목표(Romani Fig.4): control은 FFI로 완만/선형(R≈0.992), gabazine은 급격 포화.
  - ⚠️ 현재 결과(예비): control ≈ gabazine (두 곡선 겹침) = **피드포워드 억제 미작동**(SC→PC가 억제 압도).
    FFI가 실제로 나올 때까지 "Fig4 재현"으로 보고 금지. SC→PC↓ / 억제↑ / disynaptic 타이밍 재작업 필요.
  - ⚠️ SC 시냅스는 Ecker "PC->PC (E2)" 대용(Romani SC-PC 전용 파라미터 아님). g는 튜닝값(측정 아님).

실행: mpiexec -n 10 <python> 11_schaffer/sc_io_curve.py --counts 900,110,95,95 --stim_t 10 --tstop 60
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
import network_lib as net
from common.cell_loader import load_cell
from synapse_pair import build_synapse
import params_table3 as P3
MODELS = os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures"); CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
N_FIBER = 100
SC_CLASS = "PC->PC (E2)"
SWEEP = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


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
    counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, argval("--counts", "900,110,95,95").split(","))))
    stim_t = float(argval("--stim_t", "10")); tstop = float(argval("--tstop", "60"))
    sc_per_cell = int(argval("--sc_per_cell", "80")); sc_g = float(argval("--sc_g", "3.0"))
    # 피드포워드 억제 게이팅: SC→INT 강하게(빨리 발화→억제 공급), SC→PC 중간(억제가 veto 가능)
    sc_g_pc = float(argval("--sc_g_pc", str(sc_g)))
    sc_g_int = float(argval("--sc_g_int", str(sc_g)))

    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])
    ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
    keep = []
    for tn, k in counts.items():
        ids = np.where(t4 == tn)[0]; keep.extend(ids[np.argsort(dist[ids])[:k]].tolist())
    keep = np.array(sorted(keep)); N = len(keep)
    orig2gid = {int(o): g for g, o in enumerate(keep)}
    gtype = [t4[o] for o in keep]
    log(f"[E3] {N}세포 · SC {sc_per_cell}/세포 (PC {sc_g_pc}nS / INT {sc_g_int}nS) · fiber {N_FIBER} · 자극t={stim_t} · 스윕 {SWEEP}")

    type_dir = net.load_representatives(MODELS)
    my = [g for g in range(N) if g % NHOST == RANK]
    cells = {}; keeph = []
    for g in my:
        cell, _ = load_cell(type_dir[gtype[g]], gid=g)
        for sec in cell.all:
            sec.nseg = 1
        cells[g] = cell
        s = cell.soma[0]; nc = h.NetCon(s(0.5)._ref_v, None, sec=s); nc.threshold = -20.0
        pc.set_gid2node(g, RANK); pc.cell(g, nc); keeph.append(nc)
    pc.barrier(); log("[구축] 완료(조용한 슬라이스: 배경 Poisson OFF)")

    # 내재연결 — 억제 NetCon은 (ncc, base_w)로 보관(gabazine 토글용)
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
            syn = build_synapse(seg, pr, seeds=(i + 1, 1, 1), deterministic=False)
            ncc = pc.gid_connect(ga, syn); ncc.threshold = -20.0
            ncc.weight[0] = pr["g_nS"]; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_syn += 1
            if ci in inh_cls:
                inh_ncs.append((ncc, pr["g_nS"]))
        except Exception:
            pass
    n_inh_all = int(pc.allreduce(len(inh_ncs), 1))
    log(f"[내재연결] 랭크0 {n_syn} 시냅스(억제 {len(inh_ncs)}) · 총 억제 {n_inh_all}")

    # SC fibers + 시냅스
    fibers = [h.NetStim() for _ in range(N_FIBER)]
    for ns in fibers:
        ns.start = stim_t; ns.interval = 1; ns.noise = 0; ns.number = 0
        keeph.append(ns)
    prm = P3.CLASSES[SC_CLASS]; n_sc = 0
    scrng = np.random.RandomState(7000 + RANK)   # SC 배선 전용 RNG(내재연결 rng 오염 방지·재현성)
    for g in my:
        is_pc = gtype[g] == "PC"
        for _ in range(sc_per_cell):
            seg = sr_or_dend(cells[g], is_pc, scrng)
            syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000, 1, 1), deterministic=False)
            ncc = h.NetCon(fibers[scrng.randint(N_FIBER)], syn)
            ncc.weight[0] = sc_g_pc if is_pc else sc_g_int; ncc.delay = SYN_DELAY
            keeph += [syn, ncc]; n_sc += 1
    log(f"[SC배선] 랭크0 {n_sc} SC 시냅스")

    tvec = h.Vector(); gidvec = h.Vector(); pc.spike_record(-1, tvec, gidvec)
    is_pc_arr = np.array([gt == "PC" for gt in gtype])
    n_pc = int(pc.allreduce(sum(1 for g in my if gtype[g] == "PC"), 1))
    h.celsius = 34.0; h.cvode_active(0); h.dt = net.DT; pc.set_maxstep(10)

    def run_point(sc_active):
        n_act = int(round(sc_active * N_FIBER))
        for k, ns in enumerate(fibers):
            ns.number = 1 if k < n_act else 0
        tvec.resize(0); gidvec.resize(0)
        h.finitialize(-70.0); pc.psolve(tstop)
        tt = np.array(tvec.to_python()); gg = np.array(gidvec.to_python(), dtype=int)
        m = (tt >= stim_t) & (tt < tstop)
        fired = set(int(gi) for gi in gg[m] if is_pc_arr[gi])
        return int(pc.allreduce(len(fired), 1))

    results = {"control": [], "gabazine": []}
    for cond in ["control", "gabazine"]:
        for ncc, bw in inh_ncs:
            ncc.weight[0] = bw if cond == "control" else 0.0
        log(f"\n== {cond} (억제 {'ON' if cond=='control' else 'OFF=gabazine'}) ==")
        log(f"{'SC%':>6} | {'발화PC':>7} | {'비율%':>6}")
        for sa in SWEEP:
            fired = run_point(sa)
            frac = 100.0 * fired / max(1, n_pc)
            results[cond].append((sa, fired, frac))
            log(f"{sa*100:>5.0f}% | {fired:>7d} | {frac:>5.1f}")

    if RANK == 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
        os.makedirs(FIG, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        for cond, col, lab in [("control", "#2f6fb0", "정상(억제 ON)"), ("gabazine", "#C0392B", "gabazine(억제 OFF)")]:
            xs = [r[0] * 100 for r in results[cond]]; ys = [r[2] for r in results[cond]]
            ax.plot(xs, ys, "o-", color=col, lw=2, label=lab)
        xc = np.array([r[0] * 100 for r in results["control"]]); yc = np.array([r[2] for r in results["control"]])
        yg = np.array([r[2] for r in results["gabazine"]])
        R = np.corrcoef(xc, yc)[0, 1] if yc.std() > 0 else float("nan")
        ffi_gap = float(np.max(np.abs(yg - yc)))   # gabazine vs control 최대 차이(%p) = FFI 효과 크기
        ffi_ok = ffi_gap >= 10.0                    # 10%p 미만이면 FFI 사실상 미작동
        note = (f"I-O 선형성 R={R:.3f}\ngabazine 대비(FFI) 최대 {ffi_gap:.1f}%p\n"
                + ("→ FFI 작동" if ffi_ok else "→ ⚠️FFI 미작동(곡선 겹침)\n= Fig4 재현 아님"))
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc=("#eef" if ffi_ok else "#fdecea"),
                          ec=("#2f6fb0" if ffi_ok else "#C0392B")))
        ax.set_xlabel("활성 SC 축삭 비율 (%)"); ax.set_ylabel("발화한 PC 비율 (%)")
        title2 = ("Fig.4 재현" if ffi_ok else "예비 — FFI 미작동, Fig.4 미재현")
        ax.set_title(f"E3  SC 자극 I-O 곡선 + gabazine ({title2})\n"
                     f"{N}세포 · 조용한 슬라이스 + SC 볼리 (예비)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        out = os.path.join(FIG, "E3_sc_io_curve.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        np.save(os.path.join(FIG, "_e3_io.npy"),
                np.array([(c_, r[0], r[1], r[2]) for c_ in results for r in results[c_]], dtype=object))
        print(f"\n[그림] {out} (N={N}, 예비)", flush=True)
        print(f"[판정] I-O 선형 R={R:.3f} · gabazine 대비(FFI) 최대 {ffi_gap:.1f}%p → "
              + ("FFI 작동" if ffi_ok else "⚠️FFI 미작동(control≈gabazine) = Fig4 재현 아님, 재작업 필요"), flush=True)
    pc.barrier(); pc.done(); h.quit()


if __name__ == "__main__":
    main()
