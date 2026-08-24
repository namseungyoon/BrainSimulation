# -*- coding: utf-8 -*-
"""Ex2 형태 전파 데이터 — 대표 추체 1개의 전 세그먼트 Vm + 3D 위치 기록.
SC 시냅스(SR)에서 EPSP가 켜져 수상돌기→소마로 감쇠·전파하는 것을 형태 위에서 보기 위함.
release는 시각화용으로 Use=1(확정방출). 결과: scratch/ex2_morph.json (UI가 임베드).
실행: cd ~/mechbuild_gpu && python ex2_morph.py [--gid G] [--fiber 200]
"""
import os, sys, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
DERIVED = os.path.join(ROOT, "data", "derived")


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d

FIBER = arg("--fiber", 200)
GID = arg("--gid", -1)
SC_GMAX = 0.8 / 1000.0
SETTLE = 40.0; STIM = 50.0; OBS = 55.0


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


def main():
    import net_build as nb
    B = nb.NetBuilder(); h.load_file("stdrun.hoc")
    pc = h.ParallelContext(); h.cvode_active(0)
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"].astype(float); Q = wc["orientation_wxyz"]; mt = B.mt
    sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    fib = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    scpost = sc["post_gid"]; scxyz = sc["xyz"].astype(float); fid = fib["fiber_id"]
    is_pc = (mt == "SP_PC")

    sel = np.where((fid == FIBER) & is_pc[scpost])[0]
    if GID >= 0:
        gid = GID
    else:
        targs, cnts = np.unique(scpost[sel], return_counts=True)   # 단일시냅스(전형적 단위연결) 타깃
        one = targs[cnts == 1]
        gid = int(one[len(one) // 2]) if len(one) else int(targs[0])
    sel_g = sel[scpost[sel] == gid]
    print(f"[Ex2-morph] 세포 gid {gid} · SC 시냅스 {len(sel_g)}개", flush=True)

    cell = B.build_cell(gid); soma = cell.soma[0]
    tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[gid][[1, 2, 3, 0]])

    # 전 세그먼트: 3D 위치(형태 로컬, 소마중심) + Vm 기록
    segpos = []; segrec = []; seg_ref = []; seg_diam = []
    for secx in cell.all:
        nm = secx.name(); n = int(secx.n3d())
        if n < 2 or ("axon" in nm) or ("node" in nm) or ("myelin" in nm):
            continue
        arc = [secx.arc3d(i) for i in range(n)]; Lt = arc[-1] or 1.0
        xs = [secx.x3d(i) for i in range(n)]; ys = [secx.y3d(i) for i in range(n)]; zs = [secx.z3d(i) for i in range(n)]
        for seg in secx:
            a = seg.x * Lt
            segpos.append([float(np.interp(a, arc, xs)), float(np.interp(a, arc, ys)), float(np.interp(a, arc, zs))])
            seg_diam.append(float(seg.diam))
            v = h.Vector(); v.record(seg._ref_v); segrec.append(v)
            seg_ref.append((secx, round(seg.x, 4)))
    soma_idx = [i for i, (s, x) in enumerate(seg_ref) if "soma" in s.name()]
    P = np.array(segpos); soma_c = P[soma_idx].mean(axis=0) if soma_idx else P.mean(axis=0)
    P = P - soma_c                                   # 소마 중심

    # 시냅스 배치(Use=1 확정방출) + 세그먼트 인덱스
    fiber = h.VecStim(); keep = []; syn_idx = []
    for si in sel_g:
        mp = rot.inv().apply(scxyz[si] - XYZ[gid]); _, k = tree.query(mp, k=1); sec, x = ref[k]
        syn = h.GBPlasticityStpProbSyn(sec(x))
        syn.Use = 1.0; syn.Dep = 186.0; syn.Fac = 129.0; syn.Nrrp = 1; syn.gmax = SC_GMAX  # 단일소포 확정(역치하 uEPSP 시각화)
        syn.gamma_p = 0.0; syn.gamma_d = 0.0; syn.setRNG(gid + 1, int(si) + 1, 3)
        nc = h.NetCon(fiber, syn); nc.weight[0] = 1.0; nc.delay = 1.0
        ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0
        keep.append((syn, nc, ncs))
        # 가장 가까운 기록 세그먼트
        best = min(range(len(seg_ref)), key=lambda i: (seg_ref[i][0] == sec and abs(seg_ref[i][1] - x)) or 1e9)
        cand = [i for i in range(len(seg_ref)) if seg_ref[i][0] == sec]
        if cand:
            best = min(cand, key=lambda i: abs(seg_ref[i][1] - x))
        syn_idx.append(int(best))

    trec = h.Vector(); trec.record(h._ref_t)
    h.celsius = 34; h.dt = 0.025
    tv = h.Vector([STIM]); fiber.play(tv)
    pc.set_maxstep(10); h.finitialize(-70); pc.psolve(STIM + OBS)

    t = np.array(trec)
    # 다운샘플: ~150 프레임
    step = max(1, len(t) // 150)
    tt = t[::step] - STIM
    V = np.array([np.array(v)[::step] for v in segrec])   # [nseg × nframe]
    print(f"[Ex2-morph] 세그먼트 {len(segrec)} · 프레임 {len(tt)} · Vpeak {V.max():.1f}mV", flush=True)

    out = {
        "gid": int(gid), "n_seg": len(segrec),
        "pos": [[round(float(P[i, 0]), 1), round(float(P[i, 1]), 1), round(float(P[i, 2]), 1)] for i in range(len(P))],
        "diam": [round(d, 2) for d in seg_diam],
        "soma_idx": [int(i) for i in soma_idx], "syn_idx": syn_idx,
        "t": [round(float(x), 2) for x in tt],
        "vmin": -75.0, "vmax": float(round(V.max() + 2, 1)),
        # Vm: 프레임별 세그먼트 전압(정수 반올림으로 용량↓)
        "V": [[int(round(V[i, f])) for i in range(V.shape[0])] for f in range(V.shape[1])],
        "stim": STIM,
    }
    p = os.path.join(ROOT, "scratch", "ex2_morph.json")
    data = json.dumps(out, separators=(",", ":")); open(p, "w").write(data)
    print(f"[Ex2-morph] 저장 {p} ({os.path.getsize(p)//1024}KB)", flush=True)
    # UI 자동 생성: 템플릿 __INJECT__ 치환 → ex2_morph.html (자립형)
    tpl_p = os.path.join(HERE, "ex2_morph_tpl.html")
    if os.path.exists(tpl_p):
        html = open(tpl_p, encoding="utf-8").read().replace("__INJECT__", data)
        _uidir = os.path.join(ROOT, "04_experiments", "Ex2_schaffer", "ui"); os.makedirs(_uidir, exist_ok=True)
        open(os.path.join(_uidir, "ex2_morph.html"), "w", encoding="utf-8").write(html)
        print(f"[Ex2-morph] UI -> ex2_morph.html ({len(html)//1024}KB)", flush=True)


if __name__ == "__main__":
    main()
