# -*- coding: utf-8 -*-
"""
09_run/run_long.py  —  장시간 구동 + 세그먼트 CSV 저장 (V5 확장)

run_slice.py 와 동일 세팅으로 세포 구축·연결·구동하되:
  - tstop 를 여러 300ms 세그먼트로 나눠 **증분 실행** (finitialize 1회 후 continuerun 누적)
  - 각 세그먼트 끝에 그 구간 스파이크를 **CSV로 즉시 저장**(중간 중단돼도 보존)
  - 발화 세포수(전체 대비) 명시 → "250개 다 찍혔나" 검증
  - 마지막에 전체 raster 그림

CSV: figures/../spikes/V5_spikes_seg{i}_{a}-{b}ms.csv (cell_id,type,t_ms) + 합본.

실행: python -u 09_run/run_long.py --counts 150,40,30,30 --seg_ms 300 --n_seg 10
"""
import os
import sys
import csv
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
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
CSVDIR = os.path.join(HERE, "spikes"); os.makedirs(CSVDIR, exist_ok=True)
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    counts = dict(zip(["PC", "PV", "cAC", "bAC"],
                      map(int, argval("--counts", "150,40,30,30").split(","))))
    seg_ms = float(argval("--seg_ms", "300"))
    n_seg = int(argval("--n_seg", "10"))
    tstop = seg_ms * n_seg
    print(f"[설정] counts={counts} seg={seg_ms}ms×{n_seg} = {tstop:.0f}ms", flush=True)

    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])
    ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
    keep = []
    for tn, k in counts.items():
        ids = np.where(t4 == tn)[0]; keep.extend(ids[np.argsort(dist[ids])[:k]].tolist())
    keep = np.array(sorted(keep)); remap = {o: n for n, o in enumerate(keep)}
    cells_meta = [dict(id=remap[o], type=t4[o], pos=xyz[o].tolist()) for o in keep]
    types = [m["type"] for m in cells_meta]
    N = len(keep)

    p = np.load(PRUNED, allow_pickle=True)
    pre, post, cid = p["pre"], p["post"], p["cls"]; classes = list(p["classes"].astype(str))
    ks = set(keep.tolist())
    idxs = np.where([(a in ks) and (b in ks) for a, b in zip(pre, post)])[0]
    edges = [dict(pre=remap[int(pre[i])], post=remap[int(post[i])], cls=classes[int(cid[i])]) for i in idxs]
    print(f"[선택] {N}세포 · 연결 {len(edges):,} (평균차수 {len(edges)/N:.1f})", flush=True)

    rng = np.random.RandomState(7); keeph = []
    type_dir = net.load_representatives(MODELS)
    print(f"[1/4 구축] {N}세포 …", flush=True)
    cells = net.build_cells(cells_meta, type_dir, verbose=True)
    # coarse 모드: 모든 섹션 nseg=1 → 구획 수 급감(속도 5~10배, 정확도 소폭↓, 데모용)
    if "--coarse" in sys.argv:
        ncomp = 0
        for cl in cells:
            for sec in cl.all:
                sec.nseg = 1; ncomp += 1
        print(f"[coarse] nseg=1 적용 → 총 구획 {ncomp} (속도 우선)", flush=True)
    print(f"[2/4 연결] {len(edges)} EMS 시냅스 …", flush=True)
    net.wire_synapses(cells, edges, rng, keeph)
    print("[3/4 구동준비] Poisson 입력 …", flush=True)
    net.add_external_drive(cells, cells_meta, 0, keeph)
    spikes = net.record_spikes(cells, keeph)

    # 증분 실행 + 세그먼트 CSV
    print(f"[4/4 실행] 증분 {n_seg}세그먼트 …", flush=True)
    h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025; h.finitialize(-70.0)
    all_rows = []
    for s in range(n_seg):
        a, b = s * seg_ms, (s + 1) * seg_ms
        h.continuerun(b)
        rows = []
        for cid_, v in enumerate(spikes):
            tt = np.array(v.to_python(), dtype=float)
            tt = tt[(tt >= a) & (tt < b)]
            for t in tt:
                rows.append((cid_, types[cid_], round(float(t), 3)))
        fn = os.path.join(CSVDIR, f"V5_spikes_seg{s:02d}_{int(a)}-{int(b)}ms.csv")
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["cell_id", "type", "t_ms"]); w.writerows(rows)
        all_rows.extend(rows)
        fired = len({r[0] for r in rows})
        print(f"  seg{s:02d} [{int(a)}-{int(b)}ms] 스파이크 {len(rows):,} · 발화세포 {fired}/{N} → {os.path.basename(fn)}", flush=True)

    # 합본 CSV
    allfn = os.path.join(CSVDIR, "V5_spikes_all.csv")
    with open(allfn, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["cell_id", "type", "t_ms"]); w.writerows(all_rows)
    fired_all = len({r[0] for r in all_rows})
    print(f"[완료] 총 스파이크 {len(all_rows):,} · 발화세포 {fired_all}/{N} "
          f"({100*fired_all/N:.0f}%) · CSV {n_seg+1}개 → {CSVDIR}", flush=True)

    # 전체 raster
    spk = net.spikes_to_arrays(spikes)
    out_png = os.path.join(OUT, f"V5_raster_{N}cells_{int(tstop)}ms.png")
    net.analyze_activity(cells_meta, types, spk, tstop, out_png, demo=True)


if __name__ == "__main__":
    main()
