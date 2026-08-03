# -*- coding: utf-8 -*-
"""12_lfp/e4b_anim_data.py  —  인터랙티브 애니메이션용 시계열 데이터 추출

활성 뉴런(hh soma + dend 시냅스)이 발화 -> 한 전극이 기록하는 세포외 신호까지의
시계열(막전위·막전류·세포외전위)을 JSON으로 내보냄. HTML 위젯이 이 데이터로 재생.
실행: <ca1sim>/python.exe 12_lfp/e4b_anim_data.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from neuron import h
h.load_file("stdrun.hoc")
import lfp_calc as L

SIGMA = 0.3


def main():
    soma = h.Section(name="soma"); soma.nseg = 1
    soma.pt3dclear(); soma.pt3dadd(0, -10, 0, 20); soma.pt3dadd(0, 10, 0, 20)
    soma.insert("hh")
    dend = h.Section(name="dend"); dend.nseg = 15
    dend.pt3dclear(); dend.pt3dadd(0, 10, 0, 2); dend.pt3dadd(0, 310, 0, 2)
    dend.connect(soma(1), 0)
    dend.insert("pas")
    for seg in dend:
        seg.pas.e = -65.0; seg.pas.g = 1e-4
    for sec in (soma, dend):
        sec.Ra = 150.0; sec.cm = 1.0

    syn = h.Exp2Syn(dend(0.4)); syn.tau1, syn.tau2, syn.e = 0.5, 3.0, 0.0
    ns = h.NetStim(); ns.number = 1; ns.start = 5.0; ns.noise = 0
    nc = h.NetCon(ns, syn); nc.weight[0] = 0.05; nc.delay = 0.0

    geom = L.collect_segments([soma, dend])
    soma_c = L.seg_point(soma, 0.5)
    syn_c = L.seg_point(dend, 0.4)
    elec = np.array([soma_c + [45.0, 20.0, 0.0]])           # 소마 근처 전극
    M = L.lsa_matrix(geom, elec, SIGMA)

    vsoma = h.Vector().record(soma(0.5)._ref_v)
    vdend = h.Vector().record(dend(0.4)._ref_v)
    vecs, cv = L.setup_imem(geom["segs"])
    tvec = h.Vector().record(h._ref_t)
    h.celsius = 6.3; h.dt = 0.025
    h.finitialize(-65.0); h.continuerun(25.0)

    t = np.array(tvec)
    I = np.array([np.array(v) for v in vecs])               # (Nseg, Nt) nA
    Ve = (M @ I)[0] * 1e3                                    # uV
    # 대표 세그먼트: 소마(스파이크), 시냅스 세그
    isoma = int(np.argmin(np.abs(geom["mid"][:, 1] - soma_c[1])))
    isyn = int(np.argmin(np.abs(geom["mid"][:, 1] - syn_c[1])))
    im_soma = I[isoma]; im_syn = I[isyn]
    vm_s = np.array(vsoma); vm_d = np.array(vdend)

    # 다운샘플 ~220점
    N = 220
    idx = np.linspace(0, len(t) - 1, N).round().astype(int)
    data = dict(
        t=[round(float(x), 3) for x in t[idx]],
        vm_soma=[round(float(x), 2) for x in vm_s[idx]],
        vm_dend=[round(float(x), 2) for x in vm_d[idx]],
        im_soma=[round(float(x), 4) for x in im_soma[idx]],
        im_syn=[round(float(x), 4) for x in im_syn[idx]],
        ve=[round(float(x), 3) for x in Ve[idx]],
        meta=dict(spike_mV=round(float(vm_s.max()), 1),
                  ve_peak_uV=round(float(Ve[np.argmax(np.abs(Ve))]), 2),
                  syn_t=5.0),
    )
    out = os.path.join(HERE, "figures", "_e4b_anim.json")
    json.dump(data, open(out, "w"), ensure_ascii=False)
    print("saved:", out, "| points:", N, "| spike:", data["meta"]["spike_mV"], "mV | Ve peak:", data["meta"]["ve_peak_uV"], "uV")


if __name__ == "__main__":
    main()
