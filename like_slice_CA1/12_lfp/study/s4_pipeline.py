# -*- coding: utf-8 -*-
"""[학습 Step 4-보조] 데이터 변환 파이프라인 — 스파이크가 어떻게 세포외 신호가 되나

핵심 질문(사용자): "시뮬레이션에서 스파이크가 나오는데, 그 데이터가 어떻게 변환되는가?"

변환 3단계 (같은 공식이 스파이크·시냅스 둘 다 변환):
  ① 시뮬 원출력 : 각 세그먼트의 막전위 V_m(t)  (스파이크·EPSP)
  ② 막전류 추출 : use_fast_imem -> 세그먼트별 총 막전류 I_m(t) [nA]
                  (스파이크 = 소마 Na 유입 sink + K 유출 source / 시냅스 = 수상돌기 sink)
  ③ 변환(공식)  : V_e(전극) = (1/4*pi*sigma) * Σ_i I_i / r_i   -> 세포외 전위(측정 데이터)

모형: 활성 soma(hh, 스파이크) + 수동 dend. 수상돌기 시냅스가 소마 스파이크를 유발.
실행: <ca1sim>/python.exe 12_lfp/study/s4_pipeline.py
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
h.load_file("stdrun.hoc")
import lfp_calc as L

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SIGMA = 0.3


def build_cell():
    soma = h.Section(name="soma")
    soma.nseg = 1
    soma.pt3dclear(); soma.pt3dadd(0, -10, 0, 20); soma.pt3dadd(0, 10, 0, 20)
    soma.insert("hh")                                    # 활성 -> 스파이크
    dend = h.Section(name="dend")
    dend.nseg = 21
    dend.pt3dclear(); dend.pt3dadd(0, 10, 0, 2); dend.pt3dadd(0, 410, 0, 2)
    dend.connect(soma(1), 0)
    dend.insert("pas")
    for seg in dend:
        seg.pas.e = -65.0; seg.pas.g = 1e-4
    for sec in (soma, dend):
        sec.Ra = 150.0; sec.cm = 1.0
    return soma, dend


def main():
    soma, dend = build_cell()
    # 수상돌기 근위 시냅스: 강하게 -> 소마 스파이크 유발
    syn = h.Exp2Syn(dend(0.2)); syn.tau1, syn.tau2, syn.e = 0.5, 3.0, 0.0
    ns = h.NetStim(); ns.number = 1; ns.start = 5.0; ns.noise = 0
    nc = h.NetCon(ns, syn); nc.weight[0] = 0.05; nc.delay = 0.0   # 50 nS(역치 초과)

    seclist = [soma, dend]
    geom = L.collect_segments(seclist)
    soma_xyz = L.seg_point(soma, 0.5)
    syn_xyz = L.seg_point(dend, 0.2)

    # 전극: 소마 옆(스파이크 보기) + 수상돌기 옆(시냅스 보기)
    e_soma = soma_xyz + np.array([40.0, 0, 0])
    e_dend = syn_xyz + np.array([40.0, 0, 0])
    elec = np.array([e_soma, e_dend])
    M = L.lsa_matrix(geom, elec, SIGMA)
    # 필드맵 격자(피크 순간)
    gx = np.linspace(-150, 150, 160); gy = np.linspace(-120, 450, 240)
    GX, GY = np.meshgrid(gx, gy)
    Mg = L.lsa_matrix(geom, np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)]), SIGMA)

    # 기록: V_m(소마·시냅스세그), I_m(전 세그)
    vsoma = h.Vector().record(soma(0.5)._ref_v)
    vdend = h.Vector().record(dend(0.2)._ref_v)
    vecs, cv = L.setup_imem(geom["segs"])
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 6.3; h.dt = 0.025          # hh 표준온도
    h.finitialize(-65.0); h.continuerun(20.0)

    t = np.array(tvec)
    Vm_s = np.array(vsoma); Vm_d = np.array(vdend)
    I = np.array([np.array(v) for v in vecs])           # (N_seg, N_t) nA
    Ve = L.compute_lfp(M, I) * 1e3                        # (2, N_t) uV
    # 세그먼트 인덱스: 소마(0), 시냅스 세그먼트(dend 0.2에 가장 가까운)
    seg_mid = geom["mid"]
    isoma = int(np.argmin(np.abs(seg_mid[:, 1] - soma_xyz[1])))
    isyn = int(np.argmin(np.abs(seg_mid[:, 1] - syn_xyz[1])))
    spk = Vm_s.max()
    ipk = int(np.argmax(np.abs(Ve[0])))                  # 소마전극 피크(스파이크)
    print(f"[세포] soma Vm 최대 {spk:.1f}mV ({'스파이크 발생' if spk > 0 else '역치하'})", flush=True)
    print(f"[변환] 소마전극 세포외 피크 {Ve[0][ipk]:.2f}uV @t={t[ipk]:.2f}ms (스파이크 파형)", flush=True)
    print(f"[변환] 수상돌기전극 세포외 최음 {Ve[1].min():.2f}uV (시냅스 sink)", flush=True)

    # ---------------- 그림 (2x2 파이프라인) ----------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    # (A) ① 시뮬 원출력: 막전위
    a = ax[0, 0]
    a.plot(t, Vm_s, color="#c0392b", lw=1.6, label="소마 V_m (스파이크)")
    a.plot(t, Vm_d, color="#e67e22", lw=1.2, label="수상돌기 V_m (EPSP)")
    a.axhline(0, color="0.7", lw=0.5)
    a.set_xlabel("시간 (ms)"); a.set_ylabel("막전위 V_m (mV)")
    a.set_title("① 시뮬레이션 원출력 — 막전위 V_m(t)\n(우리가 계산하는 세포 '안'의 값)")
    a.legend(fontsize=8)

    # (B) ② 막전류 I_m (use_fast_imem)
    b = ax[0, 1]
    b.plot(t, I[isoma], color="#8e44ad", lw=1.6, label=f"소마 세그 I_m (스파이크: Na sink→K source)")
    b.plot(t, I[isyn], color="#16a085", lw=1.4, label=f"시냅스 세그 I_m (수상돌기 sink)")
    b.axhline(0, color="0.7", lw=0.5)
    b.text(0.02, 0.05, "음(-)=안으로(sink)\n양(+)=밖으로(source)", transform=b.transAxes, fontsize=8,
           va="bottom", bbox=dict(fc="white", ec="0.7", alpha=0.9))
    b.set_xlabel("시간 (ms)"); b.set_ylabel("막전류 I_m (nA)")
    b.set_title("② 막전류 추출 — I_m(t) [use_fast_imem]\n(변환의 '입력': 각 조각의 전류 꼭지)")
    b.legend(fontsize=7.5)

    # (C) ③ 변환 결과: 세포외 전위
    c = ax[1, 0]
    c.plot(t, Ve[0], color="#c0392b", lw=1.6, label=f"소마전극 (스파이크 파형)")
    c.plot(t, Ve[1], color="#2980b9", lw=1.4, label=f"수상돌기전극 (fEPSP)")
    c.axhline(0, color="0.7", lw=0.5)
    c.set_xlabel("시간 (ms)"); c.set_ylabel("세포외 전위 V_e (uV)")
    c.set_title("③ 변환 결과 = 측정 데이터 (세포외 V)\nV_e = (1/4πσ)·Σ I_i / r_i")

    c.legend(fontsize=8)

    # (D) 피크 순간 필드맵 + 공식
    d = ax[1, 1]
    Vg = (Mg @ I[:, ipk]) * 1e3
    Vg = Vg.reshape(GX.shape)
    vmax = np.percentile(np.abs(Vg), 99)
    im = d.pcolormesh(GX, GY, Vg, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    d.plot([0, 0], [10, 410], "-", color="0.3", lw=3)
    d.plot([soma_xyz[0]], [soma_xyz[1]], "o", color="#f39c12", ms=11, markeredgecolor="k")
    d.plot([e_soma[0]], [e_soma[1]], "s", color="k", ms=7)
    fig.colorbar(im, ax=d, label="V_e (uV)")
    d.set_xlabel("x (um)"); d.set_ylabel("y (um)")
    d.set_title(f"④ 스파이크 순간 세포외 필드 (t={t[ipk]:.1f}ms)\n소마 주변 강한 dipole")

    fig.suptitle("학습: 데이터 변환 파이프라인 — 스파이크·시냅스(막전위) → 막전류 → 세포외 전위(측정)\n"
                 "같은 공식 V_e=(1/4πσ)Σ I_i/r_i 이 스파이크(빠름)와 fEPSP(느림)를 모두 변환",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "S4_pipeline.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
