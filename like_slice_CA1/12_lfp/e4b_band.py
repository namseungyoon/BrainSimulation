# -*- coding: utf-8 -*-
"""12_lfp/e4b_band.py  —  E4b(확정): CA1 밴드 따라 MEA 8x8 배치 + 64전극 집단 fEPSP 맵

E4b v1(원반 앙상블)을 실제 슬라이스로: 실제 PC 세포 (x,y) 위치(곡선 CA1 밴드)를 footprint로,
MEA 8x8(간격200um, 직경10um)을 밴드에 맞춰 최적 배치(중심·회전 탐색) -> 각 전극의 집단 fEPSP.
MoI(검증된 lfp_calc.moi_point_matrix)로 각 전극 전위 = 모든 PC 복제본 기여 합.

실행: <ca1sim>/python.exe 12_lfp/e4b_band.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
for p in (SHARED, os.path.join(PAPER, "03_synapses"), os.path.join(PAPER, "04_network"), HERE):
    sys.path.insert(0, p)

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
PITCH, NGRID, D_ELEC = 200.0, 8, 10.0
R_ON = 100.0                       # 전극이 '조직 위'로 치는 반경


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
    # ---- 1) 단일 상세 PC 막전류 (E4b식) ----
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
    chosen = [sr[i] for i in np.linspace(0, len(sr) - 1, 40).round().astype(int)]
    p = P3.CLASSES["PC->PC (E2)"]
    ns = h.NetStim(); ns.number = 1; ns.start = 5.0; ns.noise = 0
    keep = []
    for s in chosen:
        syn = build_synapse(s(0.5), p, seeds=(1, 1, 1), deterministic=True)
        nc = h.NetCon(ns, syn); nc.weight[0] = p["g_nS"]; nc.delay = 1.0
        keep += [syn, nc]
    geom = L.collect_segments(list(cell.all))
    vecs, cv = L.setup_imem(geom["segs"])
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025
    h.finitialize(-70.0); h.continuerun(50.0)
    t = np.array(tvec)
    I = np.array([np.array(v) for v in vecs])            # (N_seg, N_t) nA
    # MEA 프레임: 깊이축 -> -z (SR 유리쪽), 유리 위 Z_SOMA
    R = rot_to_z(-depth_axis)
    loc = (geom["mid"] - soma_c) @ R.T
    loc[:, 2] += Z_SOMA - loc[:, 2].min()                 # z를 [Z_SOMA, ...]로
    Hh = loc[:, 2].max() + Z_SOMA
    geom_loc = dict(mid=loc, radius=geom["radius"])       # 셀 로컬(소마 x,y=0 근처)

    # ---- 2) 실제 PC 위치 -> 슬라이스 면(PCA) 투영 ----
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    xyz = d["xyz"].astype(float); mtype = d["mtype"]
    pc = mtype == "SP_PC"
    P = xyz[pc]                                            # PC 위치 (Npc,3)
    c0 = P.mean(axis=0); U, S, Vt = np.linalg.svd(P - c0, full_matrices=False)
    face = (P - c0) @ Vt[:2].T                             # (Npc,2) 면 좌표
    Npc = face.shape[0]
    print(f"[슬라이스] PC {Npc}개 · 면 {np.ptp(face[:,0]):.0f}x{np.ptp(face[:,1]):.0f}um", flush=True)

    # ---- 3) MEA 8x8 밴드 최적 배치 (중심·회전 탐색: 조직 위 전극 최대) ----
    span = (NGRID - 1) * PITCH
    gs = np.arange(NGRID) * PITCH - span / 2
    G0 = np.array(np.meshgrid(gs, gs)).reshape(2, -1).T   # (64,2) 기본 격자
    fc = face.mean(axis=0)
    best = (-1, fc, 0.0)
    # 대략 밀도맵으로 빠르게: 각 후보 격자점 근처 PC 존재 여부
    for th in np.deg2rad(np.arange(0, 90, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Grot = G0 @ Rm.T
        for dx in np.linspace(-300, 300, 7):
            for dy in np.linspace(-150, 150, 7):
                E = Grot + fc + [dx, dy]
                # 조직 위 전극 수 (KDTree 없이 근사: 최근접 PC 거리)
                on = 0
                for e in E:
                    if np.min(np.sum((face - e) ** 2, axis=1)) < R_ON ** 2:
                        on += 1
                if on > best[0]:
                    best = (on, E.copy(), th)
    n_on, E, th_best = best
    print(f"[MEA 배치] 최적: 회전 {np.rad2deg(th_best):.0f}deg · 조직 위 전극 {n_on}/64 ({100*n_on/64:.0f}%)", flush=True)
    over = np.array([np.min(np.sum((face - e) ** 2, axis=1)) < R_ON ** 2 for e in E])

    # ---- 4) 각 전극의 집단 fEPSP (모든 PC 복제본 기여 합, MoI) ----
    ipk_ref = None
    Ve = np.zeros((64, len(t)))
    for j, e in enumerate(E):
        # 각 PC 세포(면좌표 f)에 대해 가상전극 = (e - f, z=0); 셀 로컬 geom로 MoI
        virt = np.column_stack([e[0] - face[:, 0], e[1] - face[:, 1], np.zeros(Npc)])
        M = L.moi_point_matrix(geom_loc, virt, SIG_T, SIG_S, SIG_G, Hh, N_IMG)  # (Npc, Nseg)
        Msum = M.sum(axis=0)
        Ve[j] = (Msum @ I) * 1e3                           # uV
        if j % 16 == 0:
            print(f"  전극 {j+1}/64 계산...", flush=True)
    # 전극별 피크 진폭
    ipk = np.argmax(np.abs(Ve), axis=1)
    amp = np.array([Ve[j, ipk[j]] for j in range(64)])     # uV (부호포함)
    j_on = np.where(over)[0]
    j_max = j_on[np.argmax(np.abs(amp[j_on]))] if len(j_on) else int(np.argmax(np.abs(amp)))
    print(f"[결과] 조직 위 전극 fEPSP 피크 |중앙값| {np.median(np.abs(amp[over])):.1f}uV · 최대 {np.abs(amp).max():.1f}uV(전극#{j_max})", flush=True)

    # ---------------- 그림 ----------------
    fig = plt.figure(figsize=(15, 6.2))
    # (A) 밴드 + MEA 진폭맵
    a = fig.add_subplot(1, 2, 1)
    a.scatter(face[::5, 0], face[::5, 1], s=1, color="0.7", alpha=0.4)
    amax = np.abs(amp).max()
    sc = a.scatter(E[:, 0], E[:, 1], c=amp, s=180, cmap="RdBu_r", vmin=-amax, vmax=amax,
                   edgecolors=["k" if o else "0.6" for o in over], linewidths=1.2, zorder=5)
    fig.colorbar(sc, ax=a, label="전극 fEPSP 피크 (µV)")
    a.set_aspect("equal"); a.set_xlabel("면 가로 (µm)"); a.set_ylabel("면 세로 (µm)")
    a.set_title(f"(A) CA1 밴드 + MEA 8×8(회전{np.rad2deg(th_best):.0f}°)\n조직 위 {n_on}/64 · 전극별 집단 fEPSP 진폭")

    # (B) 대표 파형: 최대 전극 vs 조직 밖 전극
    b = fig.add_subplot(1, 2, 2)
    m = (t >= 3) & (t <= 30)
    b.plot(t[m], Ve[j_max][m], color="#c0392b", lw=1.9, label=f"밴드 중심 전극#{j_max} ({amp[j_max]:.1f}µV)")
    j_off = np.where(~over)[0]
    if len(j_off):
        jo = j_off[np.argmin(np.abs(amp[j_off]))]
        b.plot(t[m], Ve[jo][m], color="#7f8c8d", lw=1.4, ls="--", label=f"조직 밖 전극#{jo} ({amp[jo]:.2f}µV)")
    b.axhline(0, color="0.7", lw=0.5)
    b.set_xlabel("시간 (ms)"); b.set_ylabel("세포외 전위 (µV)")
    b.set_title("(B) 전극별 fEPSP 파형\n밴드 위=큰 신호 · 밖=작음")
    b.legend(fontsize=8)

    fig.suptitle(f"E4b(확정) — CA1 밴드 따라 MEA 8×8 배치 + 64전극 집단 fEPSP  "
                 f"(PC {Npc}개·MoI 검증엔진·정렬+동기 이상화)\n"
                 f"조직 위 {n_on}/64 전극이 fEPSP 포착(중앙 |{np.median(np.abs(amp[over])):.0f}|µV). "
                 f"절대크기는 세포당 진폭 이상화라 sub-mV(E4b v1과 동일 한계)",
                 fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG, "E4b_band_mea.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
