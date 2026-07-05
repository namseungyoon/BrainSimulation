# -*- coding: utf-8 -*-
"""
09_run/benchmark.py  —  NEURON 처리량 측정 + 전체 슬라이스 외삽

목적: "금요일 저녁~월요일 오전(~60h) 안에 전체 슬라이스(17,647세포)를 돌릴 수 있나?"
방법: 확실히 끝나는 작은 규모를 **벽시계 시간**으로 측정 →
       (구축시간/세포) 와 (구동시간 / (세포·ms)) 를 뽑아 전체로 외삽.

측정 항목
  1) 세포 인스턴스화(load) 시간/세포
  2) 시냅스 연결 시간
  3) 구동 시간 (coarse nseg=1, 짧은 창)  → 세포·ms 당 초
  4) 시냅스 스케일(국소 세포수↑ 시 연결수) → 별도 로그로 비선형성 확인

실행: python -u 09_run/benchmark.py --counts 20,6,4,4 --sim_ms 50 --coarse
"""
import os
import sys
import time
import numpy as np

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
from common.nrn_env import h                           # noqa: E402

MODELS = os.path.join(SHARED, "models")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
FULL_N = 17647


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    counts = dict(zip(["PC", "PV", "cAC", "bAC"],
                      map(int, argval("--counts", "20,6,4,4").split(","))))
    sim_ms = float(argval("--sim_ms", "50"))
    coarse = "--coarse" in sys.argv
    N = sum(counts.values())
    print(f"[벤치] counts={counts} (N={N}) sim={sim_ms}ms coarse={coarse}", flush=True)

    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])
    ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
    keep = []
    for tn, k in counts.items():
        ids = np.where(t4 == tn)[0]; keep.extend(ids[np.argsort(dist[ids])[:k]].tolist())
    keep = np.array(sorted(keep)); remap = {o: n for n, o in enumerate(keep)}
    cells_meta = [dict(id=remap[o], type=t4[o], pos=xyz[o].tolist()) for o in keep]

    p = np.load(PRUNED, allow_pickle=True)
    pre, post, cid = p["pre"], p["post"], p["cls"]; classes = list(p["classes"].astype(str))
    ks = set(keep.tolist())
    idxs = np.where([(a in ks) and (b in ks) for a, b in zip(pre, post)])[0]
    edges = [dict(pre=remap[int(pre[i])], post=remap[int(post[i])], cls=classes[int(cid[i])]) for i in idxs]
    print(f"[선택] {N}세포 · 연결 {len(edges):,} (평균차수 {len(edges)/N:.1f})", flush=True)

    rng = np.random.RandomState(7); keeph = []
    type_dir = net.load_representatives(MODELS)

    t0 = time.time()
    cells = net.build_cells(cells_meta, type_dir, verbose=False)
    t_build = time.time() - t0
    if coarse:
        ncomp = 0
        for cl in cells:
            for sec in cl.all:
                sec.nseg = 1; ncomp += 1
        print(f"[coarse] nseg=1 → 구획 {ncomp}", flush=True)

    t0 = time.time()
    if "--no_syn" in sys.argv:
        edges = []; print("[no_syn] 시냅스 연결 생략(구획 적분비용만 측정)", flush=True)
    net.wire_synapses(cells, edges, rng, keeph)
    t_wire = time.time() - t0

    net.add_external_drive(cells, cells_meta, 0, keeph)
    spikes = net.record_spikes(cells, keeph)

    h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025; h.finitialize(-70.0)
    t0 = time.time()
    h.continuerun(sim_ms)
    t_sim = time.time() - t0

    per_cell_build = t_build / N
    per_cellms_sim = t_sim / (N * sim_ms)
    print("\n===== 측정 결과 =====", flush=True)
    print(f"구축      : {t_build:8.1f}s  ({per_cell_build*1000:.1f} ms/세포)", flush=True)
    print(f"연결      : {t_wire:8.1f}s  ({len(edges)}시냅스)", flush=True)
    print(f"구동      : {t_sim:8.1f}s  ({sim_ms}ms, {N}세포)"
          f"  → {per_cellms_sim*1e3:.4f} ms(벽) / (세포·ms)", flush=True)

    # ===== 전체 슬라이스 외삽 =====
    # 연결수는 국소밀도에 ~2차. 여기선 구동시간을 (세포·ms) 선형으로 보수적 외삽하고
    # 시냅스 효과는 별도 상한 언급.
    print("\n===== 전체 슬라이스(17,647세포) 외삽 =====", flush=True)
    for target_ms in (300, 1000, 3000):
        build_full = per_cell_build * FULL_N
        sim_full = per_cellms_sim * FULL_N * target_ms
        tot_h = (build_full + sim_full) / 3600
        print(f"  {target_ms:5d}ms 구동: 구축 {build_full/3600:5.2f}h + 구동 {sim_full/3600:6.2f}h "
              f"= {tot_h:6.2f}h", flush=True)
    print(f"\n  (참고) 가용 창 ~60h. 위는 구동시간을 (세포·ms) 선형 가정한 하한치.", flush=True)
    print(f"  실제론 국소 시냅스수가 세포밀도에 ~2차로 늘어 구동시간이 더 큼(상한 별도).", flush=True)


if __name__ == "__main__":
    main()
