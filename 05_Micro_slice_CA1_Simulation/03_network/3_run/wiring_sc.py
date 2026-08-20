# -*- coding: utf-8 -*-
"""
03_network/3_run/wiring_sc.py  —  시냅스 배선 ①: SC 시냅스 부착 (실현가능성 실측)

조립된 세포(subset)에 SC 시냅스(ProbAMPANMDA_EMS, STP)를 실제 부착한다.
- 부착 위치: 시냅스 world xyz → 세포 형태좌표로 역변환 → NEURON 3D점 최근접 세그먼트
- STP: post mtype으로 SC1/2/3 규칙(U/D/F/NRRP) 배정 · gsyn 기본 0.8nS(미확보→보정예정)
- 섬유 정체성: 같은 fiber_id 시냅스는 VecStim 1개 공유(축삭 단위 발화)
소수 세포로 개당 시간·메모리를 실측해 전체 107만 부착 예산을 예측.

실행: python 03_network/3_run/wiring_sc.py [--ncell 200]
"""
import os
import sys
import time
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
import net_build as nb

DERIVED = os.path.join(ROOT, "data", "derived")
CFG = os.path.join(ROOT, "config", "synapse_rules.json")

NCELL = int(sys.argv[sys.argv.index("--ncell") + 1]) if "--ncell" in sys.argv else 200

# post mtype → SC 규칙(U/D/F/NRRP)
SC2 = {"SP_CCKBC", "SR_SCA", "SLM_PPA", "SP_Ivy"}
SC3 = {"SP_PVBC", "SP_AA", "SP_BS", "SO_OLM", "SO_Tri", "SO_BS", "SO_BP"}
GSYN_SC = 0.8  # nS 기본(미확보)


def sc_rule(mt):
    if mt == "SP_PC":
        return dict(Use=0.14, Dep=186.0, Fac=129.0, Nrrp=12)
    if mt in SC2:
        return dict(Use=0.11, Dep=307.0, Fac=195.0, Nrrp=4)
    return dict(Use=0.11, Dep=307.0, Fac=195.0, Nrrp=8)


def seg_kdtree(h, cell):
    """세포의 모든 3D점 좌표 + (section, 정규화 위치) 목록 → KDTree."""
    P, ref = [], []
    for sec in cell.all:
        n = int(sec.n3d())
        if n < 2:
            continue
        Ltot = sec.arc3d(n - 1) or 1.0
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            ref.append((sec, min(max(sec.arc3d(i) / Ltot, 0.0), 1.0)))
    return cKDTree(np.array(P)), ref


def main():
    t0 = time.time()
    B = nb.NetBuilder()
    h = B.h
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]
    d = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    fib = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    post = d["post_gid"]; wxyz = d["xyz"].astype(float); fiber_id = fib["fiber_id"]
    mt = B.mt

    gids = np.arange(min(NCELL, len(mt)))
    sel = np.isin(post, gids)
    idx = np.where(sel)[0]
    print(f"=== SC 배선 부착 실측 (세포 {len(gids):,} · SC 시냅스 {len(idx):,}) ===", flush=True)

    tb = time.time()
    B.build_cells(gids)
    print(f"[조립] {len(gids):,}세포 {time.time()-tb:.1f}s · RSS {nb.rss_mb():.0f}MB", flush=True)

    # 부착
    ta = time.time()
    keep = []           # GC 방지용 참조 보관
    vecstim = {}        # fiber_id -> VecStim
    n_syn = 0
    per_cell = {}
    for g in gids:
        cell = B.cells[int(g)]
        tree, ref = seg_kdtree(h, cell)
        rot = Rot.from_quat(Q[g][[1, 2, 3, 0]])
        sidx = idx[post[idx] == g]
        for si in sidx:
            morph_pt = rot.inv().apply(wxyz[si] - XYZ[g])   # world → 형태좌표
            _, k = tree.query(morph_pt, k=1)
            sec, x = ref[k]
            syn = h.ProbAMPANMDA_EMS(sec(x))
            r = sc_rule(mt[g])
            syn.Use = r["Use"]; syn.Dep = r["Dep"]; syn.Fac = r["Fac"]; syn.Nrrp = r["Nrrp"]
            fid = int(fiber_id[si])
            if fid not in vecstim:
                vs = h.VecStim(); vecstim[fid] = vs
            nc = h.NetCon(vecstim[fid], syn)
            nc.weight[0] = GSYN_SC / 1000.0
            keep.append((syn, nc)); n_syn += 1
        per_cell[int(g)] = len(sidx)

    dt = time.time() - ta
    rss = nb.rss_mb()
    print(f"[부착] SC {n_syn:,}개 · 섬유(VecStim) {len(vecstim):,}개 · {dt:.1f}s "
          f"(개당 {dt/max(n_syn,1)*1000:.2f}ms) · RSS {rss:,.0f}MB", flush=True)

    # 전체 예측
    NTOT = len(post)
    per_syn_ms = dt / max(n_syn, 1) * 1000
    # 시냅스만의 메모리 증가분(조립 후 대비)
    print(f"\n[전체 예측] SC {NTOT:,}개 부착 ≈ {per_syn_ms*NTOT/1000/60:.1f}분(부착만, 조립 별도)", flush=True)
    print(f"[요약] 세포당 SC 평균 {n_syn/len(gids):.0f}개 · 총 소요 {time.time()-t0:.0f}s", flush=True)
    json.dump({"ncell": int(len(gids)), "nsyn": n_syn, "nfiber": len(vecstim),
               "attach_s": dt, "per_syn_ms": per_syn_ms, "rss_mb": rss, "ntot": int(NTOT)},
              open(os.path.join(ROOT, "scratch", "wiring_sc_test.json"), "w"))


if __name__ == "__main__":
    main()
