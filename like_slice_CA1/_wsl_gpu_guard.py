# -*- coding: utf-8 -*-
"""
GPU용 mod 복사본 생성 + 시냅스 4종 지연연결(self-event) #ifndef CORENEURON_BUILD 가드.
원본 shared/mechanisms 는 안 건드림 → WSL ~/mods_gpu_src/ 에만 적용.
(VecStim 이 쓰는 검증된 패턴: VERBATIM 사이 NMODL net_send/flag==1 를 CoreNEURON 빌드에서 제외 → nvc++ #1067 회피.)
실행(WSL): python3 _wsl_gpu_guard.py
"""
import os, glob
SRC = "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/shared/mechanisms"
DST = os.path.expanduser("~/mods_gpu_src")
os.makedirs(DST, exist_ok=True)
SYN = {"ProbAMPANMDA_EMS.mod", "ProbGABAAB_EMS.mod", "DetAMPANMDA.mod", "DetGABAAB.mod"}

# (old, new) — 4개 시냅스 mod 공통 앵커. 각 시냅스 mod에서 정확히 1회씩 매칭돼야 함(assert).
REPL = [
    # R1 INITIAL 지연연결 열기 (12칸 들여쓴 주석 = INITIAL 고유)
    ("    VERBATIM\n            // setup self events for delayed connections to change weights\n",
     "    VERBATIM\n#ifndef CORENEURON_BUILD\n            // setup self events for delayed connections to change weights\n"),
    # R2 INITIAL 지연연결 닫기 (INITIAL 고유: 닫는 ENDVERBATIM + 8칸} + 4칸} — flag==1/t<0 닫기와 구분)
    ("    ENDVERBATIM\n        }\n    }",
     "#endif\n    ENDVERBATIM\n        }\n    }"),
    # R3 flag==1 self-event 열기 (8칸 들여쓴 vv_delay_weights = flag==1 고유)
    ("\n        IvocVect *vv_delay_weights = *((IvocVect**)(&_p_delay_weights));",
     "\n#ifndef CORENEURON_BUILD\n        IvocVect *vv_delay_weights = *((IvocVect**)(&_p_delay_weights));"),
    # R4 flag==1 self-event 닫기 (닫는 } + return + ENDVERBATIM)
    ("        }\n        return;\n    ENDVERBATIM",
     "        }\n#endif\n        return;\n    ENDVERBATIM"),
]

for f in sorted(glob.glob(SRC + "/*.mod")):
    name = os.path.basename(f)
    s = open(f, encoding="utf-8", errors="ignore").read().replace("\r\n", "\n").replace("\r", "\n")
    if name in SYN:
        for old, new in REPL:
            c = s.count(old)
            assert c == 1, f"[{name}] 앵커 {c}회(≠1): {old[:45]!r}"
            s = s.replace(old, new, 1)
        print(f"[guard] {name}  (4개 가드 적용)")
    else:
        print(f"[copy]  {name}")
    open(os.path.join(DST, name), "w", encoding="utf-8", newline="\n").write(s)

print(f"DONE -> {DST}  ({len(glob.glob(DST+'/*.mod'))} mods)")
