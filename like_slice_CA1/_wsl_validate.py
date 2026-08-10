# -*- coding: utf-8 -*-
"""검증: 대표 PC 1개 IClamp 스파이크를 일반 NEURON vs CoreNEURON CPU 대조.
   인자: 'cn'이면 CoreNEURON, 없으면 일반 NEURON. special 바이너리로 실행."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
BRAIN = "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator"
SHARED = BRAIN + "/shared"
PAPER = BRAIN + "/papers/01_Ecker2020_CA1_synaptic"
sys.path.insert(0, SHARED)
sys.path.insert(0, PAPER + "/03_synapses")
sys.path.insert(0, PAPER + "/04_network")
import numpy as np
import network_lib as net
from common.cell_loader import load_cell
from neuron import h, coreneuron
h.load_file("stdrun.hoc")

USE_CN = "cn" in sys.argv
MODELS = SHARED + "/models"
type_dir = net.load_representatives(MODELS)
cell, _ = load_cell(type_dir["PC"], gid=0)
for sec in cell.all:
    sec.nseg = 1
ic = h.IClamp(cell.soma[0](0.5)); ic.delay = 50; ic.dur = 400; ic.amp = 0.4
tvec = h.Vector()
nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0]); nc.threshold = -20.0
nc.record(tvec)
vsoma = h.Vector(); vsoma.record(cell.soma[0](0.5)._ref_v)
h.celsius = 34.0; h.dt = 0.025; h.cvode_active(0)
coreneuron.enable = USE_CN
h.finitialize(-70.0)
h.continuerun(500.0)
t = np.array(tvec.to_python())
v = np.array(vsoma.to_python())
mode = "CoreNEURON" if USE_CN else "NEURON"
print(f"MODE={mode} SPIKES={len(t)} VMAX={v.max():.3f} VMIN={v.min():.3f}")
print("SPIKETIMES=" + ",".join(f"{x:.3f}" for x in t[:20]))
