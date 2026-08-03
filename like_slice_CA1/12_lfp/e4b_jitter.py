# -*- coding: utf-8 -*-
"""12_lfp/e4b_jitter.py  —  E4b 현실성: 생물학적 지터가 집단 fEPSP를 얼마나 낮추나

E4b 밴드 결과(120µV)는 완전 정렬·동기·동일깊이의 '상한값'. 실제 세포는 제각각(지터):
  - 타이밍 지터  : 세포마다 발화 시각이 흩어짐(sigma_t)
  - 깊이 지터    : 세포마다 z 위치가 다름(sigma_z)
  - 방향 지터    : 세포마다 정단-기저 축이 기울어짐(sigma_ang)
이 지터를 넣어 완벽(상한) vs 현실 fEPSP를 한 전극(밴드 중심)에서 비교.

실행: <ca1sim>/python.exe 12_lfp/e4b_jitter.py
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
NSUB = 4000                 # 지터 루프 표본(전체 15,723 대비 축소, 비율로 스케일)
FULL_AMP = None             # 전체밀도 완벽 진폭(120µV, 스케일 기준) — 런타임 계산
SIG_T_MS, SIG_Z_UM, SIG_ANG_DEG = 1.5, 50.0, 15.0   # 현실적 지터 크기
RNG = np.random.RandomState(3)


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v


def rot_to_z(axis):
    a = unit(axis); z = np.array([0, 0, 1.0])
    v = np.cross(a, z); c = float(np.dot(a, z)); s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def tilt_rot(theta_deg, rng):
    """정단축(z)을 임의 방위로 theta만큼 기울이는 회전."""
    th = np.deg2rad(theta_deg)
    az = rng.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(az), np.sin(az), 0.0])          # xy 평면 축
    v = axis; c = np.cos(th); s = np.sin(th)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + s * vx + (1 - c) * (vx @ vx)


def main():
    # --- 단일 상세 PC 막전류 ---
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
    t = np.array(tvec); dt = t[1] - t[0]
    I = np.array([np.array(v) for v in vecs])                # (Nseg, Nt) nA
    R = rot_to_z(-depth_axis)
    loc = (geom["mid"] - soma_c) @ R.T
    loc[:, 2] += Z_SOMA - loc[:, 2].min()
    Hh = loc[:, 2].max() + Z_SOMA
    rad = geom["radius"]

    # --- 실제 PC 위치(면) + 밴드중심 전극 ---
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    P = d["xyz"].astype(float)[d["mtype"] == "SP_PC"]
    c0 = P.mean(axis=0); Vt = np.linalg.svd(P - c0, full_matrices=False)[2]
    face = (P - c0) @ Vt[:2].T
    Npc = face.shape[0]
    elec = np.array([face.mean(axis=0)[0], face.mean(axis=0)[1], 0.0])   # 밴드 중심, 유리면

    # --- (A) 전체밀도 완벽 진폭(120µV 기준) ---
    virt_all = np.column_stack([elec[0] - face[:, 0], elec[1] - face[:, 1], np.zeros(Npc)])
    M_all = L.moi_point_matrix(dict(mid=loc, radius=rad), virt_all, SIG_T, SIG_S, SIG_G, Hh, N_IMG)
    V_perf_full = (M_all.sum(axis=0) @ I) * 1e3               # uV
    ipk = int(np.argmax(np.abs(V_perf_full)))
    amp_full = V_perf_full[ipk]
    print(f"[완벽·전체 {Npc}] 밴드중심 fEPSP {amp_full:.1f}µV (상한값)", flush=True)

    # --- 표본(NSUB)으로 완벽 vs 지터 비율 ---
    idx = RNG.choice(Npc, min(NSUB, Npc), replace=False)
    fsub = face[idx]
    nt = len(t)

    def ensemble(scale):
        """지터 scale(0=완벽,1=full) 로 표본 앙상블 fEPSP(t)."""
        Vsum = np.zeros(nt)
        dts = RNG.normal(0, SIG_T_MS * scale, len(idx))
        dzs = RNG.normal(0, SIG_Z_UM * scale, len(idx))
        angs = np.abs(RNG.normal(0, SIG_ANG_DEG * scale, len(idx)))
        for k, f in enumerate(fsub):
            g = loc.copy()
            if scale > 0:
                g = g @ tilt_rot(angs[k], RNG).T
                g[:, 2] += dzs[k]
            ve = np.array([[elec[0] - f[0], elec[1] - f[1], 0.0]])
            Mk = L.moi_point_matrix(dict(mid=g, radius=rad), ve, SIG_T, SIG_S, SIG_G, Hh, N_IMG)
            Vk = (Mk[0] @ I)
            sh = int(round(dts[k] / dt))
            Vsum += np.roll(Vk, sh)
        return Vsum * 1e3                                    # uV

    V_perf_sub = ensemble(0.0)
    V_jit_sub = ensemble(1.0)
    a_perf = V_perf_sub[np.argmax(np.abs(V_perf_sub))]
    a_jit = V_jit_sub[np.argmax(np.abs(V_jit_sub))]
    ratio = abs(a_jit) / abs(a_perf)
    real_full = amp_full * ratio
    print(f"[표본 {len(idx)}] 완벽 {a_perf:.1f}µV -> 지터 {a_jit:.1f}µV (비율 {ratio:.2f})", flush=True)
    print(f"[현실 추정] 전체밀도 상한 {amp_full:.0f}µV × {ratio:.2f} = {real_full:.0f}µV (지터 반영)", flush=True)

    # --- 지터 크기 스윕 ---
    scales = np.linspace(0, 1.5, 7)
    amps = []
    for sc in scales:
        Vs = ensemble(sc)
        amps.append(abs(Vs[np.argmax(np.abs(Vs))]) / abs(a_perf))    # 완벽 대비 비율
    amps = np.array(amps)

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    m = (t >= 3) & (t <= 30)
    # (A) 완벽 vs 지터 파형(표본, 완벽=120µV로 정규화 표시)
    a = ax[0]
    sc_full = amp_full / a_perf                              # 표본->전체 스케일
    a.plot(t[m], V_perf_sub[m] * sc_full, color="#7f8c8d", lw=1.6, ls="--", label=f"완벽(상한) {amp_full:.0f}µV")
    a.plot(t[m], V_jit_sub[m] * sc_full, color="#c0392b", lw=2.0, label=f"지터 반영 {real_full:.0f}µV")
    a.axhline(0, color="0.8", lw=0.5)
    a.set_xlabel("시간 (ms)"); a.set_ylabel("fEPSP (µV)")
    a.set_title(f"(A) 완벽(상한) vs 지터(현실)\n지터로 진폭 {100*(1-ratio):.0f}% 감소·완만해짐")
    a.legend(fontsize=8)
    # (B) 지터 크기 vs 진폭비
    b = ax[1]
    b.plot(scales, amps * 100, "o-", color="#2980b9", lw=1.8)
    b.axvline(1.0, color="0.6", ls=":"); b.text(1.02, amps[np.argmin(abs(scales-1))]*100, "현실적\n지터", fontsize=8)
    b.set_xlabel(f"지터 크기 (1 = σ_t{SIG_T_MS}ms·σ_z{SIG_Z_UM:.0f}µm·σ_ang{SIG_ANG_DEG:.0f}°)")
    b.set_ylabel("완벽 대비 진폭 (%)")
    b.set_title("(B) 지터↑ → 진폭↓\n제각각일수록 덜 더해짐"); b.grid(alpha=0.3); b.set_ylim(0, 105)

    fig.suptitle(f"E4b 현실성 — 생물학적 지터가 상한값 fEPSP를 낮춘다  "
                 f"(밴드중심 전극, PC {Npc}개)\n"
                 f"완벽 상한 {amp_full:.0f}µV → 지터 반영 ~{real_full:.0f}µV ({100*(1-ratio):.0f}% 감소) — "
                 f"여전히 realistic-order(실측 0.1~1mV 저역대)",
                 fontsize=10.5, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "E4b_jitter.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    np.savez(os.path.join(FIG, "_e4b_jitter.npz"), amp_full=amp_full, real_full=real_full,
             ratio=ratio, scales=scales, amps=amps, singles=singles)
    print("saved:", out)


if __name__ == "__main__":
    main()
