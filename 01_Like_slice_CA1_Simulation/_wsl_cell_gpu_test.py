# -*- coding: utf-8 -*-
"""실제 PC me-model(13채널·다구획, 시냅스 없음) CoreNEURON GPU 실행 판별.
격리 Det 시냅스는 GPU RC=0인데 전체 모델은 세그폴트 → 채널+형태 문제인지 시냅스 통합인지 구분."""
import os, sys
from neuron import h, coreneuron
h.nrnmpi_init()
pc = h.ParallelContext()
BRAIN = "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator"   # 절대경로 고정(이 파일 위치 무관)
SHARED = os.path.join(BRAIN, "shared"); PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED); sys.path.insert(0, os.path.join(PAPER, "03_synapses")); sys.path.insert(0, os.path.join(PAPER, "04_network"))
import network_lib as net
from common.cell_loader import load_cell
MODELS = os.path.join(SHARED, "models")
USE_GPU = "gpu" in sys.argv

type_dir = net.load_representatives(MODELS)
cell, _ = load_cell(type_dir["PC"], gid=0)
for sec in cell.all:
    sec.nseg = 1
ic = h.IClamp(cell.soma[0](0.5)); ic.delay = 2.0; ic.dur = 40.0; ic.amp = 0.5
s = cell.soma[0]; nc = h.NetCon(s(0.5)._ref_v, None, sec=s); nc.threshold = -20.0
pc.set_gid2node(0, 0); pc.cell(0, nc)
tv = h.Vector(); gv = h.Vector(); pc.spike_record(-1, tv, gv)
h.dt = 0.025; h.celsius = 34.0; h.cvode_active(0)
coreneuron.enable = True; coreneuron.verbose = 1; coreneuron.gpu = USE_GPU
pc.set_maxstep(10); h.finitialize(-70.0)
pc.psolve(50.0)
print(f"CELLTEST_OK backend={'GPU' if USE_GPU else 'CPU'} - PC me-model 13채널 - spikes={len(tv)} - 완주", flush=True)
