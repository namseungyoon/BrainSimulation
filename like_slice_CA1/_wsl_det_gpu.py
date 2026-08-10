# -*- coding: utf-8 -*-
"""
결정론 시냅스(DetAMPANMDA, DetGABAAB) GPU 포팅용 복사+변환.
원본 shared/mechanisms 는 무변경 → WSL ~/mods_det_gpu/ 에만 적용.
변환 2종:
  (1) 지연연결 self-event 가드 (#638, Prob*와 동일 R1~R4) — #ifndef CORENEURON_BUILD
  (2) RANGE 리팩터 (#1067): NET_RECEIVE 인자 R/Pr/u/tsyn → RANGE+ASSIGNED (Prob* 방식)
      우리 모델은 시냅스 인스턴스당 netcon 1개 → per-netcon→per-instance 동치.
실행(WSL): python3 _wsl_det_gpu.py
"""
import os
SRC = "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms"
DST = os.path.expanduser("~/mods_det_gpu")
os.makedirs(DST, exist_ok=True)

# 지연연결 가드 (_wsl_gpu_guard.py 와 동일 R1~R4)
GUARD = [
    ("    VERBATIM\n            // setup self events for delayed connections to change weights\n",
     "    VERBATIM\n#ifndef CORENEURON_BUILD\n            // setup self events for delayed connections to change weights\n"),
    ("    ENDVERBATIM\n        }\n    }",
     "#endif\n    ENDVERBATIM\n        }\n    }"),
    ("\n        IvocVect *vv_delay_weights = *((IvocVect**)(&_p_delay_weights));",
     "\n#ifndef CORENEURON_BUILD\n        IvocVect *vv_delay_weights = *((IvocVect**)(&_p_delay_weights));"),
    ("        }\n        return;\n    ENDVERBATIM",
     "        }\n#endif\n        return;\n    ENDVERBATIM"),
]

# ASSIGNED 에 R/Pr/u/tsyn 추가 (두 mod 공통 앵커: 지연연결 주석 직전)
ASSIGN = [
    ("\n\n    : stuff for delayed connections\n",
     "\n\n    R\n    Pr\n    u\n    tsyn (ms)\n\n    : stuff for delayed connections\n"),
]

# per-mod: RANGE 줄에 R,Pr,tsyn 추가 (u 이미 RANGE) + NET_RECEIVE 인자에서 R,Pr,u,tsyn 제거
PERMOD = {
 "DetAMPANMDA.mod": [
   ("    RANGE Use, u, Dep, Fac, u0, mg, NMDA_ratio\n",
    "    RANGE Use, u, Dep, Fac, u0, mg, NMDA_ratio, R, Pr, tsyn\n"),
   ("NET_RECEIVE (weight,weight_AMPA, weight_NMDA, R, Pr, u, tsyn (ms), nc_type){",
    "NET_RECEIVE (weight,weight_AMPA, weight_NMDA, nc_type){"),
 ],
 "DetGABAAB.mod": [
   ("    RANGE Use, u, Dep, Fac, u0, GABAB_ratio\n",
    "    RANGE Use, u, Dep, Fac, u0, GABAB_ratio, R, Pr, tsyn\n"),
   ("NET_RECEIVE (weight, weight_GABAA, weight_GABAB, R, Pr, u, tsyn (ms), nc_type){",
    "NET_RECEIVE (weight, weight_GABAA, weight_GABAB, nc_type){"),
 ],
}

for name in ("DetAMPANMDA.mod", "DetGABAAB.mod"):
    s = open(os.path.join(SRC, name), encoding="utf-8", errors="ignore").read().replace("\r\n", "\n").replace("\r", "\n")
    for old, new in GUARD + ASSIGN + PERMOD[name]:
        c = s.count(old)
        assert c == 1, f"[{name}] 앵커 {c}회(!=1): {old[:50]!r}"
        s = s.replace(old, new, 1)
    open(os.path.join(DST, name), "w", encoding="utf-8", newline="\n").write(s)
    print(f"[det-gpu] {name}  (가드4 + ASSIGN1 + RANGE/NET_RECEIVE2 = 7개 적용)")

# 결정론 repro 테스트 (RNG 전혀 없음 — 결정론 시냅스 GPU 실행 격리)
REPRO = r'''# -*- coding: utf-8 -*-
"""결정론 시냅스(DetAMPANMDA) CoreNEURON GPU 실행 판별. RNG 전혀 없음(NetStim noise=0)."""
import sys
from neuron import h, coreneuron
h.load_file("stdrun.hoc")
USE_GPU = "gpu" in sys.argv
soma = h.Section(name="soma")
soma.L = soma.diam = 20.0
soma.insert("pas"); soma.g_pas = 1e-4; soma.e_pas = -70.0
keep = []
for i in range(20):
    syn = h.DetAMPANMDA(soma(0.5))
    syn.Use = 0.5; syn.Dep = 671.0; syn.Fac = 17.0
    syn.tau_d_AMPA = 3.0; syn.NMDA_ratio = 1.22
    ns = h.NetStim(); ns.interval = 10.0; ns.number = 1e9; ns.start = 5; ns.noise = 0.0
    nc = h.NetCon(ns, syn); nc.weight[0] = 10.0; nc.delay = 1.0
    keep += [syn, ns, nc]
pc = h.ParallelContext()
ncrec = h.NetCon(soma(0.5)._ref_v, None, sec=soma); ncrec.threshold = -20.0
pc.set_gid2node(0, 0); pc.cell(0, ncrec)
tv = h.Vector(); gv = h.Vector(); pc.spike_record(-1, tv, gv)
h.dt = 0.025; h.celsius = 34.0; h.cvode_active(0)
coreneuron.enable = True; coreneuron.verbose = 1
coreneuron.gpu = USE_GPU
pc.set_maxstep(10)
h.finitialize(-70.0)
pc.psolve(100.0)
print(f"REPRO_OK backend={'GPU' if USE_GPU else 'CPU'} - Det시냅스20(결정론,RNG없음) - 완주", flush=True)
'''
open(os.path.join(DST, "det_repro_test.py"), "w", encoding="utf-8", newline="\n").write(REPRO)
print(f"DONE -> {DST}")
