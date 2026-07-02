# -*- coding: utf-8 -*-
"""
09_run/run_slice.py  —  단계 9: 축소 데모 구동/시뮬레이션 (V5)

우리 slice400 배치·pruned 커넥텀·9클래스 시냅스를 Ecker network_lib 로 축소 구동.
  세포→4대표타입(e-type) 매핑 → 공간적으로 뭉친 부분집합 선택 → build→wire→drive→run→analyze.

⚠️ 축소·단순화(정성 데모):
  - 약 250세포(PC150·PV40·cAC30·bAC30), 4개 대표 형태(우리 변이형태 아님)
  - 공간 국소 부분집합(연결 보존 위해) → 전체 동역학과 동일 보장 없음(정성 검증)
  - 연결당 시냅스 다중성은 1개로 단순화(network_lib 규약)

실행: python 09_run/run_slice.py  [--counts PC,PV,cAC,bAC] [--tstop ms]
"""
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED)                              # common.*
sys.path.insert(0, os.path.join(PAPER, "03_synapses")) # params_table3, synapse_pair
sys.path.insert(0, os.path.join(PAPER, "04_network"))  # network_lib
import network_lib as net                              # noqa: E402

MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}


def main():
    counts = {"PC": 150, "PV": 40, "cAC": 30, "bAC": 30}
    if "--counts" in sys.argv:
        v = sys.argv[sys.argv.index("--counts") + 1].split(",")
        counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, v)))
    tstop = float(sys.argv[sys.argv.index("--tstop") + 1]) if "--tstop" in sys.argv else 500.0

    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])
    N = len(xyz)

    # 공간 국소 부분집합: 추체 무게중심에서 가까운 순으로 타입별 k개
    ctr = xyz[t4 == "PC"].mean(0)
    dist = np.linalg.norm(xyz - ctr, axis=1)
    keep = []
    for tn, k in counts.items():
        ids = np.where(t4 == tn)[0]
        ids = ids[np.argsort(dist[ids])[:k]]
        keep.extend(ids.tolist())
    keep = np.array(sorted(keep))
    remap = {old: new for new, old in enumerate(keep)}
    print(f"[선택] {len(keep)}세포 (국소): " +
          ", ".join(f"{tn} {int((t4[keep]==tn).sum())}" for tn in counts))

    cells_meta = [dict(id=remap[o], type=t4[o], pos=xyz[o].tolist()) for o in keep]

    # pruned 커넥텀에서 부분집합 내 edge만
    p = np.load(PRUNED, allow_pickle=True)
    pre, post, cid = p["pre"], p["post"], p["cls"]; classes = list(p["classes"].astype(str))
    keepset = set(keep.tolist())
    kept_e = np.array([(a in keepset) and (b in keepset) for a, b in zip(pre, post)])
    idxs = np.where(kept_e)[0]
    edges = [dict(pre=remap[int(pre[i])], post=remap[int(post[i])], cls=classes[int(cid[i])])
             for i in idxs]
    print(f"[연결] 부분집합 내 edge {len(edges):,} (평균 차수 {len(edges)/len(keep):.1f})")

    rng = np.random.RandomState(7)
    types = [m["type"] for m in cells_meta]
    keeph = []
    type_dir = net.load_representatives(MODELS)
    print("[로드] 대표 e-type: " + ", ".join(type_dir.keys()), flush=True)
    print(f"[1/5 구축] 세포 {len(cells_meta)} …", flush=True)
    cells = net.build_cells(cells_meta, type_dir, verbose=True)
    print(f"[2/5 연결] {len(edges)} EMS 시냅스 …", flush=True)
    n_ok, n_fail = net.wire_synapses(cells, edges, rng, keeph)
    if n_fail:
        print(f"  [경고] {n_fail} 실패", flush=True)
    print("[3/5 구동] Poisson 외부입력 …", flush=True)
    net.add_external_drive(cells, cells_meta, 0, keeph)
    spikes = net.record_spikes(cells, keeph)
    print(f"[4/5 실행] TSTOP={tstop:.0f}ms dt=0.025 …", flush=True)
    net.run_network(tstop)
    spk = net.spikes_to_arrays(spikes)
    out_png = os.path.join(OUT, "V5_network_activity.png")
    print("[5/5 분석] raster + 발화율 …", flush=True)
    net.analyze_activity(cells_meta, types, spk, tstop, out_png, demo=True)


if __name__ == "__main__":
    main()
