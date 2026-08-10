# -*- coding: utf-8 -*-
"""GPU 스모크: 대표 PC + 확률 SC 시냅스(ProbAMPANMDA_EMS, Random123) + NetStim 구동을
CoreNEURON GPU로 실행 → SIGABRT 없이 완주 + 스파이크 = 확률 시냅스 GPU 런타임 검증.
실행: <gpu special> -python _wsl_gpu_smoke.py [gpu]   (인자 'gpu' 있으면 coreneuron.gpu=True)"""
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BRAIN = "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator"
SHARED = BRAIN + "/shared"; PAPER = BRAIN + "/papers/01_Ecker2020_CA1_synaptic"
sys.path.insert(0, SHARED); sys.path.insert(0, PAPER + "/03_synapses"); sys.path.insert(0, PAPER + "/04_network")
import network_lib as net                       # noqa
from common.cell_loader import load_cell          # noqa
from synapse_pair import build_synapse            # noqa
import params_table3 as P3                         # noqa
from neuron import h, coreneuron                   # noqa
h.load_file("stdrun.hoc")

USE_GPU = "gpu" in sys.argv
NOSYN = "nosyn" in sys.argv          # 시냅스 없이 채널만(GPU 격리용) + IClamp 구동
type_dir = net.load_representatives(SHARED + "/models")
cell, _ = load_cell(type_dir["PC"], gid=0)
for sec in cell.all:
    sec.nseg = 1

keep = []
if NOSYN:
    ic = h.IClamp(cell.soma[0](0.5)); ic.delay = 10; ic.dur = 80; ic.amp = 0.5
    keep.append(ic); segs = []
else:
    # 확률 SC 시냅스(PC->PC E2) 여러 개 + 각자 NetStim 포아송 구동
    pr = P3.CLASSES["PC->PC (E2)"]
    apics = [s for s in cell.all if ".apic" in s.name()]
    segs = (apics if apics else [cell.soma[0]])[:20]
    for i, sec in enumerate(segs):
        syn = build_synapse(sec(0.5), pr, seeds=(i + 1, 1, 1), deterministic=False)  # Prob = Random123
        ns = h.NetStim(); ns.interval = 10.0; ns.number = 1e9; ns.start = 5; ns.noise = 1.0
        r = h.Random(); r.Random123(i, 1, 0); r.negexp(1); ns.noiseFromRandom(r)
        nc = h.NetCon(ns, syn); nc.weight[0] = 10.0; nc.delay = 1.0
        keep += [syn, ns, r, nc]

pc = h.ParallelContext()
soma = cell.soma[0]
nc_rec = h.NetCon(soma(0.5)._ref_v, None, sec=soma); nc_rec.threshold = -20.0
pc.set_gid2node(0, 0); pc.cell(0, nc_rec)          # CoreNEURON: 스파이크 소스 gid 필수
tvec = h.Vector(); gidvec = h.Vector(); pc.spike_record(-1, tvec, gidvec)

h.celsius = 34.0; h.cvode_active(0); h.dt = 0.025
coreneuron.enable = True; coreneuron.verbose = 0
coreneuron.gpu = USE_GPU
pc.set_maxstep(10)
h.finitialize(-70.0)
pc.psolve(100.0)
print(f"[SMOKE] backend={'GPU' if USE_GPU else 'CPU'} · SC확률시냅스 {len(segs)}개 · "
      f"소마 스파이크 {len(tvec)}개 · t[:8]={[round(x,2) for x in list(tvec)[:8]]}", flush=True)
print("SMOKE_OK", flush=True)
