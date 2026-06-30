"""
verify_setup.py — Phase 0 환경 검증
------------------------------------------------------------
Source: Ecker et al. (2020) §2.5 stochastic TM 시냅스(ProbAMPANMDA_EMS/ProbGABAAB_EMS)
목적: NEURON 로드 + 컴파일된 BBP EMS 시냅스 메커니즘을 실제 인스턴스화하여
      Table 3 파라미터(Use/Dep/Fac/Nrrp)를 주입할 수 있는지 확인한다.

실행 (ca1sim 환경):
    conda activate ca1sim
    python SourceCode/00_setup/verify_setup.py
"""
import os
import sys

# common 패키지 import 경로 (SourceCode 를 sys.path 에 추가)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.nrn_env import h, neuron, load_project_mechanisms, have_mechanism  # noqa: E402


def main() -> int:
    print(f"[1] NEURON version : {neuron.__version__}")

    load_project_mechanisms()
    print("[2] nrnmech.dll 로드 완료")

    # 핵심: 논문에서 쓴 stochastic TM 시냅스 메커니즘 가용성
    required = ["ProbAMPANMDA_EMS", "ProbGABAAB_EMS", "DetAMPANMDA", "DetGABAAB", "VecStim"]
    missing = [m for m in required if not have_mechanism(m)]
    for m in required:
        print(f"    - {m:18s}: {'OK' if have_mechanism(m) else 'MISSING'}")
    if missing:
        print(f"[FAIL] 누락 메커니즘: {missing}")
        return 1

    # 실제 인스턴스화 + Table 3 파라미터 주입 검증 (PC->PC E2: Use~0.5, Dep~671, Fac~17, Nrrp=2)
    soma = h.Section(name="soma")
    soma.L = soma.diam = 20.0
    soma.insert("pas")

    syn = h.ProbAMPANMDA_EMS(soma(0.5))
    syn.Use = 0.5
    syn.Dep = 671.0
    syn.Fac = 17.0
    syn.Nrrp = 2
    syn.tau_r_AMPA = 0.2
    syn.tau_d_AMPA = 3.0
    print(f"[3] ProbAMPANMDA_EMS 인스턴스화 OK "
          f"(Use={syn.Use}, Dep={syn.Dep}, Fac={syn.Fac}, Nrrp={syn.Nrrp})")

    gaba = h.ProbGABAAB_EMS(soma(0.5))
    gaba.Use = 0.16
    gaba.Dep = 1122.0
    gaba.Fac = 9.3
    gaba.Nrrp = 1
    print(f"[4] ProbGABAAB_EMS 인스턴스화 OK "
          f"(Use={gaba.Use}, Dep={gaba.Dep}, Fac={gaba.Fac}, Nrrp={gaba.Nrrp})")

    print("\n[SUCCESS] Phase 0 환경 검증 통과 — 단일연결 STP 단계로 진행 가능.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
