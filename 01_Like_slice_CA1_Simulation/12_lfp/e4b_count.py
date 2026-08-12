# -*- coding: utf-8 -*-
"""12_lfp/e4b_count.py  —  전극 하나에 뉴런 몇 개가 들어오나 (기여 세포 수·유효반경)

한 MEA 전극의 fEPSP에 실제로 몇 개의 PC가 기여하는지: 반경별 세포 수 + 기여 누적 +
유효 세포 수(participation ratio). 우리 slice400 SP 밴드(실측 위치) 기준.
실행: <ca1sim>/python.exe 12_lfp/e4b_count.py
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
SIG_T, SIG_S, SIG_G, N_IMG, Z_SOMA = 0.3, 1.5, 0.0, 20, 60.0


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v


def rot_to_z(a):
    a = unit(a); z = np.array([0, 0, 1.0]); v = np.cross(a, z); c = float(np.dot(a, z)); s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s / s)


def main():
    # 단일 세포 막전류 + 로컬 기하(간단화: E4b식)
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

    # 실제 PC 위치(면)
    d = np.load(os.path.join(ROOT, "05_placement", "slice_cells.npz"), allow_pickle=True)
    Pp = d["xyz"].astype(float)[d["mtype"] == "SP_PC"]
    c0 = Pp.mean(0); Vt = np.linalg.svd(Pp - c0, full_matrices=False)[2]
    face = (Pp - c0) @ Vt[:2].T
    Npc = len(face)
    area = (np.ptp(face[:, 0]) / 1000) * (np.ptp(face[:, 1]) / 1000)
    dens = Npc / area
    elec = face.mean(0)
    r = np.sqrt(((face - elec) ** 2).sum(1))

    # 전극 기준 각 세포의 피크 기여 |cpk|
    virt = np.column_stack([elec[0] - face[:, 0], elec[1] - face[:, 1], np.zeros(Npc)])
    M = L.moi_point_matrix(dict(mid=loc, radius=rad), virt, SIG_T, SIG_S, SIG_G, Hh, N_IMG)
    ipk = int(np.argmax(np.abs((M.sum(0) @ I))))
    cpk = np.abs(M @ I[:, ipk])                     # 세포별 |기여| (uV)
    total = cpk.sum()
    Neff = (cpk.sum() ** 2) / (cpk ** 2).sum()      # 유효 세포 수(참여비)

    print(f"[밴드] SP_PC {Npc}개 · 면적 {area:.2f}mm² · 밀도 {dens:.0f}/mm²", flush=True)
    for RR in [50, 100, 150, 200, 300, 500]:
        n = int((r < RR).sum()); frac = cpk[r < RR].sum() / total
        print(f"  반경 {RR:>4}µm 내: {n:>5}세포 · 기여 {100*frac:4.0f}%", flush=True)
    print(f"[유효 세포 수] participation ratio Neff = {Neff:.0f} (한 전극 신호를 '실질적으로' 만드는 세포 수)", flush=True)
    # 기여 50%/90% 반경
    order = np.argsort(r); cum = np.cumsum(cpk[order]) / total
    r50 = r[order][np.searchsorted(cum, 0.5)]; r90 = r[order][np.searchsorted(cum, 0.9)]
    print(f"[유효 반경] 신호 50%는 반경 {r50:.0f}µm·90%는 {r90:.0f}µm 내 세포에서", flush=True)

    # ---- 그림 ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    a = ax[0]
    sc2 = a.scatter(face[:, 0] - elec[0], face[:, 1] - elec[1], c=cpk, s=6, cmap="inferno_r",
                    vmax=np.percentile(cpk, 99))
    a.plot(0, 0, "s", color="#2980b9", ms=13, markeredgecolor="k", zorder=6)
    for RR in [100, 200, 300]:
        a.add_patch(plt.Circle((0, 0), RR, fill=False, ec="0.3", ls="--", lw=1))
        a.text(0, RR, f"{RR}µm", fontsize=8, ha="center", va="bottom", color="0.3")
    fig.colorbar(sc2, ax=a, label="세포별 기여 |V| (µV)")
    a.set_aspect("equal"); a.set_xlim(-600, 600); a.set_ylim(-600, 600)
    a.set_xlabel("전극 기준 가로 (µm)"); a.set_ylabel("세로 (µm)")
    a.set_title(f"(A) 한 전극(파랑) 주변 PC 기여\n가까운 세포일수록 크게 기여 (전극 중심)")

    b = ax[1]
    rs = np.linspace(0, 600, 120)
    ncum = [int((r < x).sum()) for x in rs]
    fcum = [cpk[r < x].sum() / total for x in rs]
    b.plot(rs, ncum, color="#c0392b", lw=2, label="세포 수(누적)")
    b.set_xlabel("전극으로부터 반경 (µm)"); b.set_ylabel("세포 수", color="#c0392b")
    b.tick_params(axis="y", labelcolor="#c0392b")
    b2 = b.twinx()
    b2.plot(rs, np.array(fcum) * 100, color="#2980b9", lw=2, ls="--", label="fEPSP 기여(누적 %)")
    b2.set_ylabel("fEPSP 기여 누적 (%)", color="#2980b9"); b2.tick_params(axis="y", labelcolor="#2980b9")
    b2.axhline(90, color="0.6", ls=":", lw=1); b2.axvline(r90, color="0.6", ls=":", lw=1)
    b.set_title(f"(B) 반경별 세포 수 vs 신호 기여\n유효세포 Neff≈{Neff:.0f} · 신호 90%는 {r90:.0f}µm 내")
    b.grid(alpha=0.3)

    fig.suptitle(f"전극 하나에 뉴런 몇 개? — SP 밴드 밀도 {dens:.0f}/mm² · 200µm 타일당 ~{int(dens*0.04)}세포 · "
                 f"유효 기여 세포 Neff≈{Neff:.0f} (신호 90%는 반경 {r90:.0f}µm 내)",
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "E4b_count.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
