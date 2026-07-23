# -*- coding: utf-8 -*-
"""
자체완결 repro: BBP 확률 시냅스(ProbAMPANMDA_EMS, Random123)가 CoreNEURON GPU에서 도는가?
아틀라스/me-model 의존 없음 — passive 단일구획 + 확률 시냅스 + NetStim 포아송 구동.

목적: 로컬 WSL(HPC SDK 26.5·NEURON 9.0.1)에서 GPU 실행이 SEGFAULT(RC=139) → 이게
      로컬 툴체인 문제인지 Random123-GPU 근본 문제인지 다른 환경(클라우드 A6000)에서 판별.

빌드:  nrnivmodl -coreneuron .          # 이 폴더에서 (GPU 설치 nvc++ 환경)
실행:  x86_64/special -python gpu_repro_test.py gpu    # GPU 백엔드
       x86_64/special -python gpu_repro_test.py cpu    # CPU 백엔드(대조·같은 빌드)
판정:  "REPRO_OK ..." 출력 + RC=0 이면 성공. 세그폴트(RC=139)면 재현됨.
"""
import sys
from neuron import h, coreneuron
h.load_file("stdrun.hoc")

USE_GPU = "gpu" in sys.argv

# passive 단일구획 (채널 없음 — Random123 시냅스만 격리)
soma = h.Section(name="soma")
soma.L = soma.diam = 20.0
soma.insert("pas"); soma.g_pas = 1e-4; soma.e_pas = -70.0

# 확률 시냅스 20개(Ecker PC->PC E2 유사 파라미터) + 각자 포아송 NetStim
keep = []
for i in range(20):
    syn = h.ProbAMPANMDA_EMS(soma(0.5))
    syn.Use = 0.5; syn.Dep = 671.0; syn.Fac = 17.0; syn.Nrrp = 1
    syn.tau_d_AMPA = 3.0; syn.NMDA_ratio = 1.22
    syn.setRNG(i + 1, 2, 3)                       # ★ Random123 확률 방출 스트림
    ns = h.NetStim(); ns.interval = 10.0; ns.number = 1e9; ns.start = 5; ns.noise = 1.0
    r = h.Random(); r.Random123(i, 1, 0); r.negexp(1); ns.noiseFromRandom(r)
    nc = h.NetCon(ns, syn); nc.weight[0] = 10.0; nc.delay = 1.0
    keep += [syn, ns, r, nc]

# CoreNEURON: 스파이크 소스 gid 필수(패시브라 스파이크는 없지만 규약상 등록)
pc = h.ParallelContext()
ncrec = h.NetCon(soma(0.5)._ref_v, None, sec=soma); ncrec.threshold = -20.0
pc.set_gid2node(0, 0); pc.cell(0, ncrec)
tv = h.Vector(); gv = h.Vector(); pc.spike_record(-1, tv, gv)
vm = h.Vector(); vm.record(soma(0.5)._ref_v)

h.dt = 0.025; h.celsius = 34.0; h.cvode_active(0)
coreneuron.enable = True; coreneuron.verbose = 1
coreneuron.gpu = USE_GPU
pc.set_maxstep(10)
h.finitialize(-70.0)
pc.psolve(100.0)

vmax = max(vm) if len(vm) else -70.0
print(f"REPRO_OK backend={'GPU' if USE_GPU else 'CPU'} · Prob시냅스20(Random123) · "
      f"Vm_max={vmax:.3f}mV (>-70이면 시냅스 방출 발생) · 완주", flush=True)
