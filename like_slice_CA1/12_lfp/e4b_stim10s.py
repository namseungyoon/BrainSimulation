# -*- coding: utf-8 -*-
"""12_lfp/e4b_stim10s.py  —  전극 배치 후 10초 SC 자극-반응 fEPSP 시뮬레이션 (계산·캐시)

E4b 확정 3x8 전극 배치를 그대로 두고, SC(샤퍼 곁가지) 자극 프로토콜을 10초간 전달해
24전극 유발(evoked) 세포외 fEPSP를 기록한다. 실제 MEA 유발전위 실험의 in silico 짝.

프로토콜(자극=동기 SC 볼리, 모든 SR 시냅스 동시 활성):
  · baseline 단일 test pulse : t = 1, 3, 5 s
  · paired-pulse (ISI 50ms)  : t = 7.00, 7.05 s   → 단기가소성(PPR)
  · 20Hz 트레인 ×5           : t = 9.00~9.20 s     → 트레인 억압/회복
세포·시냅스는 E4b와 동일(대표 PC + 40개 SR DetAMPANMDA, PC->PC E2, g=0.6nS) →
E4b-9(120µV) 결과와 직접 연속. 막전류는 rec_dt로 다운샘플 기록(메모리 절약).
전극전위 = 모든 PC 복제본 기여 합(검증된 MoI). 정렬·동기 이상화 상한값(지터 미포함).

실행: <ca1sim>/python.exe 12_lfp/e4b_stim10s.py     -> figures/_e4b_stim10s.npz
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
for p in (SHARED, os.path.join(PAPER, "03_synapses"), os.path.join(PAPER, "04_network"), HERE):
    sys.path.insert(0, p)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
import params_table3 as P3
from synapse_pair import build_synapse
import lfp_calc as L

MODELS = os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SIG_T, SIG_S, SIG_G, N_IMG = 0.3, 1.5, 0.0, 20
Z_SOMA = 60.0
PITCH, D_ELEC, R_ON = 200.0, 10.0, 100.0
NCOL, NROW = 8, 3                                  # 확정 3x8
TSTOP, DT, REC_DT = 10000.0, 0.025, 0.1           # ms
N_SYN = 40                                         # SR SC 시냅스 수 (E4b 동일)
# 자극 프로토콜 (ms)
STIM = [1000.0, 3000.0, 5000.0,                    # baseline 단일
        7000.0, 7050.0,                            # paired-pulse (ISI 50ms)
        9000.0, 9050.0, 9100.0, 9150.0, 9200.0]    # 20Hz train x5
STIM_LABELS = ["baseline #1", "baseline #2", "baseline #3", "PP-1", "PP-2",
               "train-1", "train-2", "train-3", "train-4", "train-5"]


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v


def rot_to_z(axis):
    a = unit(axis); z = np.array([0, 0, 1.0])
    v = np.cross(a, z); c = float(np.dot(a, z)); s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def main():
    t_start = time.time()
    # ---- 1) 대표 PC + 40 SR 시냅스 (E4b 동일) ----
    type_dir = net.load_representatives(MODELS)
    cell, tname = load_cell(type_dir["PC"], gid=0)
    h.define_shape()
    soma = cell.soma[0]; soma_c = L.seg_point(soma, 0.5)
    h.distance(0, soma(0.5))
    apic = [s for s in cell.all if ".apic" in s.name()]
    dmax = max(h.distance(s(0.5)) for s in apic)
    depth_axis = unit(np.mean([L.seg_point(s, 0.5) for s in apic if h.distance(s(0.5)) > 0.9 * dmax], axis=0) - soma_c)
    lo, hi = 0.30 * dmax, 0.68 * dmax
    sr = sorted([s for s in apic if lo <= h.distance(s(0.5)) <= hi], key=lambda s: h.distance(s(0.5)))
    chosen = [sr[i] for i in np.linspace(0, len(sr) - 1, N_SYN).round().astype(int)]
    p = P3.CLASSES["PC->PC (E2)"]
    syns = []
    for s in chosen:
        syn = build_synapse(s(0.5), p, seeds=(1, 1, 1), deterministic=True)
        syns.append(syn)
    # 자극 프로토콜: 이벤트마다 NetStim -> 모든 시냅스 동시(동기 SC 볼리)
    keep = []
    for t_ev in STIM:
        ns = h.NetStim(); ns.number = 1; ns.start = t_ev; ns.noise = 0
        keep.append(ns)
        for syn in syns:
            nc = h.NetCon(ns, syn); nc.weight[0] = p["g_nS"]; nc.delay = 1.0
            keep.append(nc)

    # ---- 2) 막전류 기록(다운샘플 rec_dt) ----
    geom = L.collect_segments(list(cell.all))
    nseg = len(geom["segs"]); nrec = int(TSTOP / REC_DT) + 1
    mb = nseg * nrec * 8 / 1e6
    print(f"[셋업] PC 세그먼트 {nseg} · 기록 {nrec}점(rec_dt={REC_DT}ms) · I배열 ~{mb:.0f}MB · 시냅스 {N_SYN} · 자극 {len(STIM)}회", flush=True)
    cv = h.CVode(); cv.use_fast_imem(1)
    vecs = [h.Vector() for _ in range(nseg)]
    for v, seg in zip(vecs, geom["segs"]):
        v.record(seg._ref_i_membrane_, REC_DT)

    # MEA 프레임: 깊이축 -> -z (SR 유리쪽)
    R = rot_to_z(-depth_axis)
    loc = (geom["mid"] - soma_c) @ R.T
    loc[:, 2] += Z_SOMA - loc[:, 2].min()
    Hh = loc[:, 2].max() + Z_SOMA
    geom_loc = dict(mid=loc, radius=geom["radius"])

    # ---- 3) 실제 PC 위치 면투영 + MEA 3x8 배치(회전·중심 탐색) ----
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    xyz = d["xyz"].astype(float); mtype = d["mtype"]
    P = xyz[mtype == "SP_PC"]
    c0 = P.mean(axis=0); U, S, Vt = np.linalg.svd(P - c0, full_matrices=False)
    face = (P - c0) @ Vt[:2].T
    Npc = face.shape[0]
    gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH
    gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
    G0 = np.column_stack([np.meshgrid(gx, gy)[0].ravel(), np.meshgrid(gx, gy)[1].ravel()])
    NELEC = G0.shape[0]
    fc = face.mean(axis=0)
    from scipy.spatial import cKDTree
    tree = cKDTree(face)
    best = (-1, None, 0.0)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Grot = G0 @ Rm.T
        for dx in np.linspace(-400, 400, 9):
            for dy in np.linspace(-200, 200, 9):
                E = Grot + fc + [dx, dy]
                on = int(np.sum(tree.query(E)[0] < R_ON))
                if on > best[0]:
                    best = (on, E.copy(), th)
    n_on, E, th_best = best
    over = tree.query(E)[0] < R_ON
    print(f"[MEA] 3x8 · 회전 {np.rad2deg(th_best):.0f}deg · 조직 위 {n_on}/{NELEC}", flush=True)

    # ---- 4) 전극별 집단 전달벡터 Msum (기하만; 시간무관) ----
    Msum = np.zeros((NELEC, nseg))
    for j, e in enumerate(E):
        virt = np.column_stack([e[0] - face[:, 0], e[1] - face[:, 1], np.zeros(Npc)])
        M = L.moi_point_matrix(geom_loc, virt, SIG_T, SIG_S, SIG_G, Hh, N_IMG)
        Msum[j] = M.sum(axis=0)
        if j % 8 == 0:
            print(f"  전달벡터 {j+1}/{NELEC}...", flush=True)

    # ---- 5) 10초 구동 ----
    print(f"[구동] tstop={TSTOP/1000:.0f}s dt={DT}ms ... (수 분 소요)", flush=True)
    h.celsius = 34.0; h.cvode_active(0); h.dt = DT
    h.finitialize(-70.0)
    t_run = time.time(); h.continuerun(TSTOP)
    print(f"[구동완료] {time.time()-t_run:.0f}s", flush=True)

    I = np.array([np.asarray(v) for v in vecs])            # (nseg, nrec) nA
    t = np.arange(I.shape[1]) * REC_DT
    Ve = (Msum @ I) * 1e3                                   # (NELEC, nrec) uV

    # 자극별 유발 피크(전극별): 각 자극 후 [0,40]ms 창의 최대 |Ve|
    amp_stim = np.zeros((NELEC, len(STIM)))
    for si, ts in enumerate(STIM):
        m = (t >= ts) & (t <= ts + 40.0)
        seg = Ve[:, m]
        idx = np.argmax(np.abs(seg), axis=1)
        amp_stim[:, si] = seg[np.arange(NELEC), idx]
    j_on = np.where(over)[0]
    amp1 = np.abs(amp_stim[:, 0])                            # baseline#1 진폭
    j_max = j_on[np.argmax(amp1[j_on])] if len(j_on) else int(np.argmax(amp1))
    print(f"[결과] baseline 유발 fEPSP 중앙 |{np.median(np.abs(amp_stim[over,0])):.1f}|µV · 최대 |{amp1.max():.1f}|µV(전극#{j_max})", flush=True)
    a = np.abs(amp_stim[j_max])
    print(f"[전극#{j_max} 유발 진폭µV] " + " ".join(f"{lab}:{a[i]:.0f}" for i, lab in enumerate(STIM_LABELS)), flush=True)
    if a[0] > 0:
        print(f"  PPR(PP-2/PP-1)={a[4]/a[3]:.2f} · 트레인말/초(train5/train1)={a[9]/a[5]:.2f}", flush=True)

    out = os.path.join(FIG, "_e4b_stim10s.npz")
    np.savez(out, t=t.astype(np.float32), Ve=Ve.astype(np.float32), E=E, over=over,
             stim=np.array(STIM), stim_labels=np.array(STIM_LABELS), amp_stim=amp_stim,
             ncol=NCOL, nrow=NROW, th=float(th_best), n_on=n_on, npc=Npc, j_max=int(j_max),
             tstop=TSTOP, rec_dt=REC_DT, n_syn=N_SYN)
    print(f"saved: {out} · 총 {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
