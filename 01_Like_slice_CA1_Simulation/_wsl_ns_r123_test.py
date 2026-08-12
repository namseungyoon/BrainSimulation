# -*- coding: utf-8 -*-
"""세그폴트 범인 판별: NetStim의 Random123 포아송 노이즈가 GPU 세그폴트 원인인가?
Det 시냅스(RNG 없음)는 확정 OK. NetStim noise 스트림만 토글(r123 인자)해서 GPU 실행 비교."""
import sys
from neuron import h, coreneuron
h.load_file("stdrun.hoc")
USE_GPU = "gpu" in sys.argv
R123 = "r123" in sys.argv                     # NetStim에 Random123 포아송 노이즈 켜기
soma = h.Section(name="soma"); soma.L = soma.diam = 20.0
soma.insert("pas"); soma.g_pas = 1e-4; soma.e_pas = -70.0
keep = []
for i in range(20):
    syn = h.DetAMPANMDA(soma(0.5)); syn.Use = 0.5; syn.Dep = 671.0; syn.Fac = 17.0
    syn.tau_d_AMPA = 3.0; syn.NMDA_ratio = 1.22
    ns = h.NetStim(); ns.interval = 10.0; ns.number = 1e9; ns.start = 5
    if R123:
        ns.noise = 1.0; r = h.Random(); r.Random123(i, 1, 0); r.negexp(1); ns.noiseFromRandom(r); keep.append(r)
    else:
        ns.noise = 0.0
    nc = h.NetCon(ns, syn); nc.weight[0] = 10.0; nc.delay = 1.0
    keep += [syn, ns, nc]
pc = h.ParallelContext()
ncrec = h.NetCon(soma(0.5)._ref_v, None, sec=soma); ncrec.threshold = -20.0
pc.set_gid2node(0, 0); pc.cell(0, ncrec)
tv = h.Vector(); gv = h.Vector(); pc.spike_record(-1, tv, gv)
h.dt = 0.025; h.celsius = 34.0; h.cvode_active(0)
coreneuron.enable = True; coreneuron.verbose = 1; coreneuron.gpu = USE_GPU
pc.set_maxstep(10); h.finitialize(-70.0); pc.psolve(100.0)
print(f"NS_TEST_OK backend={'GPU' if USE_GPU else 'CPU'} R123={R123} - 완주", flush=True)
