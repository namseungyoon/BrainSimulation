# -*- coding: utf-8 -*-
"""[학습 Step 3] 진짜 뉴런은 다이폴을 저절로 만든다 — ball-and-stick + 시냅스 -> LFP

논문: Lindén 2014(LFPy) — 세포외전위 = 모든 세그먼트 막전류의 가중합.
배우는 것:
  - Step 2에선 sink/source를 '손으로' 2개 놓았다. 여기선 진짜 뉴런(공+막대)에
    수상돌기 시냅스 하나만 넣으면, sink(시냅스)와 source(소마+막대)가 '저절로' 생긴다.
  - use_fast_imem으로 각 세그먼트 막전류(nA)를 뽑아 우리 계산기(lfp_calc)로 세포외 V 계산.
  - 결과: 시냅스 근처 음성 / 소마 근처 양성 = fEPSP 다이폴 (E4a 상세세포의 축소판).

모형: soma(공, 20um) + dendrite(막대, 400um, 수동 pas). 시냅스 = dend 중간(y~210um).
실행: <ca1sim>/python.exe 12_lfp/study/s3_ball_stick.py
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
sys.path.insert(0, LFP)
from neuron import h
h.load_file("stdrun.hoc")   # run/continuerun/cvode_active 정의(직접 import 시 필요)
import lfp_calc as L

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SIGMA = 0.3


def build_cell():
    """공(soma)+막대(dend) 수동 뉴런. 명시적 3D 좌표(soma y[-10,10], dend y[10,410])."""
    soma = h.Section(name="soma")
    soma.nseg = 1
    soma.pt3dclear(); soma.pt3dadd(0, -10, 0, 20); soma.pt3dadd(0, 10, 0, 20)
    dend = h.Section(name="dend")
    dend.nseg = 21
    dend.pt3dclear(); dend.pt3dadd(0, 10, 0, 2); dend.pt3dadd(0, 410, 0, 2)
    dend.connect(soma(1), 0)
    for sec in (soma, dend):
        sec.insert("pas")
        sec.Ra = 150.0
        sec.cm = 1.0
        for seg in sec:
            seg.pas.e = -65.0
            seg.pas.g = 1e-4
    return soma, dend


def main():
    soma, dend = build_cell()
    syn_seg = dend(0.5)                       # 막대 중간 = y ~ 210 um
    syn = h.Exp2Syn(syn_seg)
    syn.tau1, syn.tau2, syn.e = 0.5, 5.0, 0.0   # 흥분성(반전전위 0mV)
    ns = h.NetStim(); ns.number = 1; ns.start = 5.0; ns.noise = 0
    nc = h.NetCon(ns, syn); nc.weight[0] = 0.006; nc.delay = 0.0   # 6 nS

    seclist = [soma, dend]
    geom = L.collect_segments(seclist)
    syn_xyz = L.seg_point(dend, 0.5)
    soma_xyz = L.seg_point(soma, 0.5)
    print(f"[세포] soma {np.round(soma_xyz,0)}  시냅스(dend 중간) {np.round(syn_xyz,0)}  세그먼트 {geom['mid'].shape[0]}개", flush=True)

    # 전극: 막대 축(y) 옆 x=40um 선, y -100~450 (소마 아래 ~ 막대 끝 위)
    yy = np.linspace(-100, 450, 40)
    elec = np.column_stack([np.full_like(yy, 40.0), yy, np.zeros_like(yy)])
    M = L.lsa_matrix(geom, elec, SIGMA)
    # 2D 필드맵 격자
    gx = np.linspace(-200, 200, 180); gy = np.linspace(-150, 500, 260)
    GX, GY = np.meshgrid(gx, gy)
    gelec = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
    Mg = L.lsa_matrix(geom, gelec, SIGMA)

    vecs, cv = L.setup_imem(geom["segs"])
    tvec = h.Vector().record(h._ref_t)
    vsoma = h.Vector().record(soma(0.5)._ref_v)
    h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025
    h.finitialize(-65.0); h.continuerun(30.0)

    t = np.array(tvec)
    I = np.array([np.array(v) for v in vecs])            # (N_seg, N_t) nA
    cons, imax = L.current_conservation(I)
    print(f"[전류보존] max|sumI|/max|I| = {cons/max(imax,1e-12):.2e}", flush=True)

    V = L.compute_lfp(M, I) * 1e3                          # (40, N_t) uV
    # 피크 시각(시냅스 근처 전극 = syn_xyz에 가장 가까운 전극)
    jsyn = int(np.argmin(np.abs(yy - syn_xyz[1])))
    jsoma = int(np.argmin(np.abs(yy - soma_xyz[1])))
    ipk = int(np.argmax(np.abs(V[jsyn])))
    tpk = t[ipk]
    Vg = (Mg @ I[:, ipk]) * 1e3                            # 피크 순간 필드맵 uV
    Vg = Vg.reshape(GX.shape)
    prof = V[:, ipk]                                       # 피크 순간 깊이 프로파일
    revs = [yy[k] for k in range(1, len(yy)) if np.sign(prof[k]) != np.sign(prof[k - 1])]
    print(f"[결과] 피크 t={tpk:.1f}ms | 시냅스전극(y={yy[jsyn]:.0f}) {prof[jsyn]:.2f}uV(음?) | "
          f"소마전극(y={yy[jsoma]:.0f}) {prof[jsoma]:.2f}uV(양?) | 극성반전 y~{revs[0]:.0f}um" if revs else "", flush=True)

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

    # (A) 형태 + 시냅스 + 전극선
    a = ax[0]
    a.plot([soma_xyz[0]], [soma_xyz[1]], "o", color="#f39c12", ms=22, label="soma(공)")
    a.plot([0, 0], [10, 410], "-", color="0.4", lw=6, solid_capstyle="round", label="dendrite(막대)")
    a.plot([syn_xyz[0]], [syn_xyz[1]], "^", color="#27ae60", ms=13, markeredgecolor="k", label="흥분 시냅스")
    a.plot(elec[:, 0], elec[:, 1], ".-", color="#2980b9", ms=4, lw=1, label="전극선(x=40um)")
    a.set_xlim(-120, 120); a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
    a.set_title("(A) ball-and-stick 뉴런\n공(soma)+막대(dend)+시냅스+전극")
    a.legend(fontsize=7, loc="upper right"); a.set_aspect("equal", adjustable="datalim")

    # (B) 피크 순간 필드맵
    b = ax[1]
    vmax = np.percentile(np.abs(Vg), 99)
    im = b.pcolormesh(GX, GY, Vg, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    b.contour(GX, GY, Vg, levels=[0], colors="k", linewidths=1.0, linestyles="--")
    b.plot([0, 0], [10, 410], "-", color="0.3", lw=3)
    b.plot([syn_xyz[0]], [syn_xyz[1]], "^", color="#145a32", ms=10)
    b.plot([soma_xyz[0]], [soma_xyz[1]], "o", color="#7d6608", ms=10)
    fig.colorbar(im, ax=b, label="V (uV)  파랑=음(sink)/빨강=양(source)")
    b.set_xlabel("x (um)"); b.set_ylabel("y (um)")
    b.set_title(f"(B) 피크 순간 세포외 필드맵 (t={tpk:.1f}ms)\n시냅스 sink→음 / 소마 source→양")

    # (C) 깊이 프로파일 = fEPSP 극성반전
    c = ax[2]
    c.plot(prof, yy, "o-", color="#2c3e50", ms=3, lw=1.4)
    c.axvline(0, color="0.5", lw=0.8)
    c.fill_betweenx(yy, 0, prof, where=(prof < 0), color="#3498db", alpha=0.25)
    c.fill_betweenx(yy, 0, prof, where=(prof > 0), color="#e74c3c", alpha=0.25)
    c.axhline(syn_xyz[1], color="#27ae60", ls=":", lw=0.8); c.text(prof.min(), syn_xyz[1] + 6, "시냅스", fontsize=8, color="#1e8449")
    c.axhline(soma_xyz[1], color="#f39c12", ls=":", lw=0.8); c.text(prof.max(), soma_xyz[1] - 16, "소마", fontsize=8, color="#b9770e", ha="right")
    if revs:
        c.axhline(revs[0], color="orange", lw=1); c.text(0.1, revs[0] + 8, f"반전 y~{revs[0]:.0f}", fontsize=8, color="#a04000")
    c.set_xlabel("피크 순간 V (uV)"); c.set_ylabel("깊이 y (um)")
    c.set_title("(C) 깊이 프로파일 = fEPSP\n시냅스 음성 <-> 소마 양성")

    # (D) 시간파형: 시냅스전극 vs 소마전극
    d = ax[3]
    d.plot(t, V[jsyn], color="#c0392b", lw=1.8, label=f"시냅스전극 (y={yy[jsyn]:.0f})")
    d.plot(t, V[jsoma], color="#2980b9", lw=1.8, label=f"소마전극 (y={yy[jsoma]:.0f})")
    d.axhline(0, color="0.5", lw=0.6); d.axvline(tpk, color="0.7", ls=":", lw=0.8)
    d.set_xlabel("시간 (ms)"); d.set_ylabel("V (uV)")
    d.set_title("(D) 시간파형\n같은 자극, 위치 따라 음/양 반대")
    d.legend(fontsize=8)

    fig.suptitle("학습 Step 3 — 진짜 뉴런(ball-and-stick)은 fEPSP 다이폴을 저절로 만든다 (Lindén 2014 LFPy 방식)",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "S3_ball_stick.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
