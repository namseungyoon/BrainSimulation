# -*- coding: utf-8 -*-
"""12_lfp/e4b_disk_contrib.py  —  원형(디스크) 전극 재구현 + 24전극 뉴런 기여 분포

(1) 전극을 점(point)이 아닌 **10µm 디스크**로: 디스크 표면 여러 점의 MoI 전위를 평균.
    점 대비 차이를 정량(원거리 소스라 미미할 것으로 예상 → 정직히 수치화).
(2) 24개 전극 각각에 대해: 신호를 만드는 PC가 **몇 개**(유효 Neff·유효반경)인지 +
    전극 주변 **어떤 뉴런**(12 m-type)이 있는지 국소 조성. fEPSP 소스는 PC만(정직).
실행: <ca1sim>/python.exe 12_lfp/e4b_disk_contrib.py   -> figures/_e4b_disk_contrib.npz
"""
import os
import sys
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
from scipy.spatial import cKDTree

MODELS = os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures")
SIG_T, SIG_S, SIG_G, N_IMG, Z_SOMA = 0.3, 1.5, 0.0, 20, 60.0
PITCH, D_ELEC, R_ON, NCOL, NROW = 200.0, 10.0, 100.0, 8, 3
R_ELEC = D_ELEC / 2.0                                  # 5µm
R_CENSUS = 150.0                                       # 국소 조성 반경
N_DISK = 9                                             # 디스크 샘플점(중심+8링)


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v


def rot_to_z(a):
    a = unit(a); z = np.array([0, 0, 1.0]); v = np.cross(a, z); c = float(np.dot(a, z)); s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s / s)


def disk_offsets():
    """디스크 표면 샘플점(전극면 z=0 평면): 중심 + 8링(r=R_ELEC)."""
    offs = [(0.0, 0.0)]
    for k in range(8):
        a = 2 * np.pi * k / 8
        offs.append((R_ELEC * np.cos(a), R_ELEC * np.sin(a)))
    return np.array(offs)


def main():
    # ---- 단일 대표 PC + 40 SR 시냅스 (E4b 동일) → 막전류 ----
    td = net.load_representatives(MODELS)
    cell, tn = load_cell(td["PC"], gid=0); h.define_shape()
    soma = cell.soma[0]; sc = L.seg_point(soma, 0.5); h.distance(0, soma(0.5))
    apic = [s for s in cell.all if ".apic" in s.name()]
    dmax = max(h.distance(s(0.5)) for s in apic)
    da = unit(np.mean([L.seg_point(s, 0.5) for s in apic if h.distance(s(0.5)) > 0.9 * dmax], 0) - sc)
    lo, hi = 0.30 * dmax, 0.68 * dmax
    srr = sorted([s for s in apic if lo <= h.distance(s(0.5)) <= hi], key=lambda s: h.distance(s(0.5)))
    ch = [srr[i] for i in np.linspace(0, len(srr) - 1, 40).round().astype(int)]
    p = P3.CLASSES["PC->PC (E2)"]; ns = h.NetStim(); ns.number = 1; ns.start = 5; ns.noise = 0
    keep = []
    for s in ch:
        syn = build_synapse(s(0.5), p, seeds=(1, 1, 1), deterministic=True)
        nc = h.NetCon(ns, syn); nc.weight[0] = p["g_nS"]; nc.delay = 1; keep += [syn, nc]
    geom = L.collect_segments(list(cell.all)); vecs, cv = L.setup_imem(geom["segs"])
    h.celsius = 34; h.cvode_active(0); h.dt = 0.025; h.finitialize(-70); h.continuerun(50)
    I = np.array([np.array(v) for v in vecs])
    R = rot_to_z(-da); loc = (geom["mid"] - sc) @ R.T; loc[:, 2] += Z_SOMA - loc[:, 2].min()
    Hh = loc[:, 2].max() + Z_SOMA; rad = geom["radius"]
    geom_loc = dict(mid=loc, radius=rad)

    # ---- 실제 세포 위치(전 타입) 면투영 ----
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    xyz = d["xyz"].astype(float); mt = d["mtype"]
    is_pc = mt == "SP_PC"
    c0 = xyz[is_pc].mean(0); Vt = np.linalg.svd(xyz[is_pc] - c0, full_matrices=False)[2]
    face_all = (xyz - c0) @ Vt[:2].T
    face_pc = face_all[is_pc]; Npc = len(face_pc)
    mtypes = sorted(set(mt.tolist()))

    # ---- MEA 3x8 배치 ----
    gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH; gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
    G0 = np.column_stack([np.meshgrid(gx, gy)[0].ravel(), np.meshgrid(gx, gy)[1].ravel()])
    NELEC = G0.shape[0]; fc = face_pc.mean(0); tree = cKDTree(face_pc); best = (-1, None, 0.0)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Grot = G0 @ Rm.T
        for dx in np.linspace(-400, 400, 9):
            for dy in np.linspace(-200, 200, 9):
                E = Grot + fc + [dx, dy]; on = int((tree.query(E)[0] < R_ON).sum())
                if on > best[0]:
                    best = (on, E.copy(), th)
    n_on, E, th = best; over = tree.query(E)[0] < R_ON
    treeA = cKDTree(face_all)

    offs = disk_offsets()
    print(f"[준비] PC {Npc} · 세그 {len(rad)} · 전극 {NELEC} · 디스크 {N_DISK}점(지름 {D_ELEC}µm) · 조성반경 {R_CENSUS}µm", flush=True)

    # ---- 전극별: 점/디스크 진폭 + PC 기여분포 + 국소 조성 ----
    amp_pt = np.zeros(NELEC); amp_disk = np.zeros(NELEC)
    Neff = np.zeros(NELEC); r50 = np.zeros(NELEC); r90 = np.zeros(NELEC)
    census = np.zeros((NELEC, len(mtypes)), int)
    for j, e in enumerate(E):
        # 점 전극: 세포별 전달행렬 (peak 시각은 점 기준으로 결정)
        virt = np.column_stack([e[0] - face_pc[:, 0], e[1] - face_pc[:, 1], np.zeros(Npc)])
        M = L.moi_point_matrix(geom_loc, virt, SIG_T, SIG_S, SIG_G, Hh, N_IMG)   # (Npc, Nseg)
        ve_t = M.sum(0) @ I
        ip = int(np.argmax(np.abs(ve_t)))
        amp_pt[j] = ve_t[ip] * 1e3
        cpk = np.abs(M @ I[:, ip]); tot = cpk.sum()
        Neff[j] = (tot ** 2) / (cpk ** 2).sum()
        r = np.sqrt(((face_pc - e) ** 2).sum(1)); o = np.argsort(r); cum = np.cumsum(cpk[o]) / tot
        r50[j] = r[o][np.searchsorted(cum, 0.5)]; r90[j] = r[o][np.searchsorted(cum, 0.9)]
        # 디스크 전극: 표면 N_DISK점의 Msum 평균으로 진폭
        acc = np.zeros(I.shape[1])
        for ox, oy in offs:
            vd = np.column_stack([e[0] + ox - face_pc[:, 0], e[1] + oy - face_pc[:, 1], np.zeros(Npc)])
            Md = L.moi_point_matrix(geom_loc, vd, SIG_T, SIG_S, SIG_G, Hh, N_IMG)
            acc += Md.sum(0) @ I
        amp_disk[j] = (acc / N_DISK)[ip] * 1e3
        # 국소 조성(전 타입, R_CENSUS 내)
        idx = treeA.query_ball_point(e, R_CENSUS)
        for cidx in idx:
            census[j, mtypes.index(mt[cidx])] += 1
        if j % 6 == 0:
            print(f"  전극 {j+1}/{NELEC} · Neff {Neff[j]:.0f} · 점 {amp_pt[j]:.0f}µV 디스크 {amp_disk[j]:.0f}µV", flush=True)

    dd = np.where(np.abs(amp_pt) > 1e-6, 100 * np.abs(amp_disk - amp_pt) / np.maximum(np.abs(amp_pt), 1e-9), 0.0)
    print(f"[디스크 vs 점] 진폭 차이 중앙 {np.median(dd):.3f}% · 최대 {dd.max():.3f}% (10µm 디스크·원거리 소스라 미미)", flush=True)
    print(f"[24전극] Neff 중앙 {np.median(Neff[over]):.0f} · 유효반경90 중앙 {np.median(r90[over]):.0f}µm", flush=True)
    tot_int = census[:, [i for i, m in enumerate(mtypes) if m != 'SP_PC']].sum(1)
    print(f"[국소조성 {R_CENSUS}µm] PC 중앙 {int(np.median(census[:, mtypes.index('SP_PC')]))} · 인터뉴런 중앙 {int(np.median(tot_int))} · 신호원=PC만(정직)", flush=True)

    np.savez(os.path.join(FIG, "_e4b_disk_contrib.npz"),
             E=E, over=over, amp_pt=amp_pt, amp_disk=amp_disk, Neff=Neff, r50=r50, r90=r90,
             census=census, mtypes=np.array(mtypes), face_pc=face_pc, face_all=face_all,
             mt_all=mt, ncol=NCOL, nrow=NROW, r_census=R_CENSUS, d_elec=D_ELEC, npc=Npc)
    print("saved: figures/_e4b_disk_contrib.npz", flush=True)


if __name__ == "__main__":
    main()
