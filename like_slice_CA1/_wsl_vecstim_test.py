# -*- coding: utf-8 -*-
"""VecStim이 CoreNEURON GPU에서 이벤트를 전달하는가(세그폴트 없이)? passive+Det+VecStim 격리."""
import sys, numpy as np
from neuron import h, coreneuron
h.load_file("stdrun.hoc")
USE_GPU = "gpu" in sys.argv
soma = h.Section(name="soma"); soma.L = soma.diam = 20.0
soma.insert("pas"); soma.g_pas = 1e-4; soma.e_pas = -70.0
keep = []; rng = np.random.RandomState(1)
for i in range(20):
    syn = h.DetAMPANMDA(soma(0.5)); syn.Use = 0.5; syn.Dep = 671.0; syn.Fac = 17.0
    syn.tau_d_AMPA = 3.0; syn.NMDA_ratio = 1.22
    tsp = np.cumsum(rng.exponential(100.0, size=20)); tsp = tsp[tsp < 100.0]   # ~10Hz 포아송
    vs = h.VecStim(); vec = h.Vector(tsp); vs.play(vec)
    nc = h.NetCon(vs, syn); nc.weight[0] = 10.0; nc.delay = 1.0
    keep += [syn, vs, vec, nc]
pc = h.ParallelContext()
ncrec = h.NetCon(soma(0.5)._ref_v, None, sec=soma); ncrec.threshold = -20.0
pc.set_gid2node(0, 0); pc.cell(0, ncrec)
tv = h.Vector(); gv = h.Vector(); pc.spike_record(-1, tv, gv)
h.dt = 0.025; h.celsius = 34.0; h.cvode_active(0)
coreneuron.enable = True; coreneuron.verbose = 1; coreneuron.gpu = USE_GPU
pc.set_maxstep(10); h.finitialize(-70.0); pc.psolve(100.0)
print(f"VECSTIM_TEST_OK backend={'GPU' if USE_GPU else 'CPU'} - 완주", flush=True)
