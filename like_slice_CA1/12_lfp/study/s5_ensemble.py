# -*- coding: utf-8 -*-
"""[학습 Step 5] 다수 세포 앙상블 — 왜 mV 크기가 나오나 (단일세포 µV의 합)

핵심 질문: 세포 1개는 µV인데, 실측 fEPSP는 mV. 어떻게?
답: 같은 방향으로 정렬된 세포 수천 개가 '동시에' 반응 → 작은 신호들이 더해짐(앙상블).

증명(시뮬):
  1) E4a 상세 PC 1개로 세포외 fEPSP 파형(단위 신호) 계산.
  2) 이 세포를 N개 '복제'해 전극 주변에 수평으로 흩뿌리되 '같은 방향(정렬)'으로 둠.
     각 복제본의 기여 = 전극을 그 세포 위치만큼 옮겨 본 것(기하 평행이동) -> 합산.
  3) (A) 정렬+동기 -> 신호가 N에 따라 커짐   vs   (B) 방향 무작위 -> 상쇄(안 커짐).
  4) 실측 mV엔 ~10^3-10^4 세포가 필요함을 외삽.

실행: <ca1sim>/python.exe 12_lfp/study/s5_ensemble.py
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
LFP = os.path.dirname(HERE)
ROOT = os.path.dirname(LFP)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
for p in (SHARED, os.path.join(PAPER, "03_synapses"), os.path.join(PAPER, "04_network"), LFP):
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
SIGMA = 0.3
N_SYN = 40
SR_BAND = (0.30, 0.68)
NMAX = 2000
RNG = np.random.RandomState(7)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def main():
    # ---- 1) E4a 단일 상세 PC + SR 시냅스 + 단일볼리 -> 막전류 ----
    type_dir = net.load_representatives(MODELS)
    cell, tname = load_cell(type_dir["PC"], gid=0)
    h.define_shape()
    soma = cell.soma[0]
    soma_c = L.seg_point(soma, 0.5)
    h.distance(0, soma(0.5))
    apic = [s for s in cell.all if ".apic" in s.name()]
    dmax = max(h.distance(s(0.5)) for s in apic)
    depth_axis = unit(np.mean([L.seg_point(s, 0.5) for s in apic
                               if h.distance(s(0.5)) > 0.9 * dmax], axis=0) - soma_c)
    lo, hi = SR_BAND[0] * dmax, SR_BAND[1] * dmax
    sr = sorted([s for s in apic if lo <= h.distance(s(0.5)) <= hi], key=lambda s: h.distance(s(0.5)))
    chosen = [sr[i] for i in np.linspace(0, len(sr) - 1, N_SYN).round().astype(int)]
    p = P3.CLASSES["PC->PC (E2)"]
    ns = h.NetStim(); ns.number = 1; ns.start = 5.0; ns.noise = 0
    keep = []
    syn_pos = []
    for s in chosen:
        syn = build_synapse(s(0.5), p, seeds=(1, 1, 1), deterministic=True)
        nc = h.NetCon(ns, syn); nc.weight[0] = p["g_nS"]; nc.delay = 1.0
        keep += [syn, nc]; syn_pos.append(L.seg_point(s, 0.5))
    syn_c = np.mean(syn_pos, axis=0)

    geom = L.collect_segments(list(cell.all))
    vecs, cv = L.setup_imem(geom["segs"])
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025
    h.finitialize(-70.0); h.continuerun(50.0)
    t = np.array(tvec)
    I = np.array([np.array(v) for v in vecs])            # (N_seg, N_t) nA

    # ---- 2) 전극: SR 시냅스 무게중심 옆 30um (인구 중심) ----
    lateral = unit(np.cross(depth_axis, [0, 0, 1.0]))
    elec = syn_c + 30.0 * lateral
    M0 = L.lsa_matrix(geom, [elec], SIGMA)               # 단일세포 전달행렬 (1,N_seg)
    V1 = (M0 @ I)[0] * 1e3                                # 단일세포 fEPSP (uV)
    ipk = int(np.argmax(np.abs(V1)))
    print(f"[단일세포] fEPSP 음성피크 {V1[ipk]:.3f} uV @t={t[ipk]:.1f}ms", flush=True)

    # ---- 3) 앵상블: N개 복제본을 수평 원반(반경 R)에 정렬 배치 ----
    # 세포를 offset 만큼 옮김 = 전극을 -offset 만큼 옮겨 본 것
    d1 = lateral
    d2 = unit(np.cross(depth_axis, lateral))
    rr = 400.0 * np.sqrt(RNG.uniform(0, 1, NMAX))         # 반경 400um 균일밀도
    th = RNG.uniform(0, 2 * np.pi, NMAX)
    offs = np.outer(rr * np.cos(th), d1) + np.outer(rr * np.sin(th), d2)
    virt = elec[None, :] - offs                          # (NMAX,3) 가상 전극
    Mall = L.lsa_matrix(geom, virt, SIGMA)               # (NMAX, N_seg) 각 복제본 전달행렬
    cpk = Mall @ I[:, ipk] * 1e3                          # 각 복제본의 피크기여 (uV)

    Ns = np.unique(np.round(np.logspace(0, np.log10(NMAX), 30)).astype(int))
    align = np.array([cpk[:n].sum() for n in Ns])        # 정렬+동기: 그냥 합
    signs = RNG.choice([-1, 1], NMAX)                    # 방향 무작위(예시: 부호 무작위)
    rand = np.array([np.abs((signs[:n] * cpk[:n]).sum()) for n in Ns])

    # 대표 파형(N=1,10,100,1000): V_N(t) = (ΣM_k) @ I
    waves = {}
    for n in [1, 10, 100, 1000]:
        Msum = Mall[:n].sum(axis=0)
        waves[n] = (Msum @ I) * 1e3                       # (N_t,) uV

    a1 = abs(V1[ipk])
    print(f"[앵상블] 정렬+동기 N=100 -> {abs(align[np.argmin(abs(Ns-100))]):.1f}uV, "
          f"N=1000 -> {abs(align[-2] if Ns[-1]!=1000 else align[-1]):.1f}uV", flush=True)
    n_for_mV = 1000.0 / (abs(align[np.argmin(abs(Ns-1000))]) / 1000.0 / 1000) if False else None
    print(f"[외삽] 단일 {a1:.2f}uV -> 실측 ~1mV엔 대략 {int(1000/max(a1,1e-9))}배 규모 세포 동기 필요", flush=True)

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (A) 단일세포 = 벽돌 한 장
    a = ax[0, 0]
    m = (t >= 3) & (t <= 30)
    a.plot(t[m], V1[m], color="#c0392b", lw=1.8)
    a.axhline(0, color="0.7", lw=0.5)
    a.set_title(f"(A) 세포 1개 = 벽돌 한 장 (음성 {V1[ipk]:.2f} µV)\n아주 작다")
    a.set_xlabel("시간 (ms)"); a.set_ylabel("세포외 전위 (µV)")

    # (B) N개 정렬+동기 합산 파형
    b = ax[0, 1]
    for n, col in zip([1, 10, 100, 1000], ["#95a5a6", "#f39c12", "#2980b9", "#c0392b"]):
        b.plot(t[m], waves[n][m], lw=1.8, color=col, label=f"N={n}")
    b.axhline(0, color="0.7", lw=0.5)
    b.set_title("(B) 정렬+동기 세포 N개 합산\n세포가 많을수록 신호가 커진다")
    b.set_xlabel("시간 (ms)"); b.set_ylabel("세포외 전위 (µV)")
    b.legend(fontsize=8)

    # (C) 피크 크기 vs N: 정렬 vs 무작위
    c = ax[1, 0]
    c.loglog(Ns, np.abs(align), "o-", color="#c0392b", lw=1.8, label="정렬+동기 (더해짐)")
    c.loglog(Ns, np.abs(rand), "s--", color="#7f8c8d", lw=1.5, label="방향 무작위 (상쇄)")
    c.axhline(1000, color="green", ls=":", lw=1.2); c.text(1.2, 1200, "실측 fEPSP ~1 mV(=1000µV)", fontsize=8, color="green")
    c.set_title("(C) 신호 크기 vs 세포 수\n정렬하면 커지고, 무작위면 상쇄")
    c.set_xlabel("세포 수 N"); c.set_ylabel("피크 |V| (µV)")
    c.legend(fontsize=8); c.grid(alpha=0.3, which="both")

    # (D) 공간 배치(정렬 화살표) — 위에서 본 그림
    d = ax[1, 1]
    sub = offs[:400]
    d.scatter(sub @ d1, sub @ d2, s=6, color="#2980b9", alpha=0.5)
    d.scatter([0], [0], marker="s", s=90, color="k", zorder=5, label="전극")
    # 정렬 방향 표시(모두 같은 방향 화살표 몇 개)
    for k in range(0, 400, 40):
        d.arrow((sub @ d1)[k], (sub @ d2)[k], 0, 25, head_width=12, color="#c0392b", alpha=0.7)
    d.set_title("(D) 위에서 본 세포 배치 (반경 400µm)\n화살표=세포 방향(모두 정렬). 전극=검정")
    d.set_xlabel("가로 (µm)"); d.set_ylabel("세로 (µm)"); d.set_aspect("equal")
    d.legend(fontsize=8, loc="upper right")

    fig.suptitle("학습 Step 5 — 다수 세포 앙상블: 단일세포 µV가 '정렬+동기'로 더해져 집단 mV fEPSP가 된다",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "S5_ensemble.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
