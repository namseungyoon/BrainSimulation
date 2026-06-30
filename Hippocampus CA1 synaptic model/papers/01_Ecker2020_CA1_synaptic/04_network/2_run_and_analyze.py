"""
2_run_and_analyze.py — 축소 CA1 마이크로서킷: 시뮬레이션 (파이프라인)
============================================================================
Source: Ecker(2020) §3.5; EMS 확률 시냅스(ProbAMPANMDA/ProbGABAAB_EMS).

`1_build_connectivity.py` 가 만든 connectivity.json 을 읽어, **network_lib 의 단계
함수**로 마이크로서킷을 구성·실행한다. main() 에 전체 흐름이 그대로 드러난다:

    대표모델 → 세포 구축 → 9클래스 시냅스 연결 → 외부 구동 → 실행 → 분석

실행:
  <ca1sim python> .../04_network/2_run_and_analyze.py           # 전체 122세포(수 분)
  <ca1sim python> .../04_network/2_run_and_analyze.py --demo    # 42세포 축소(시간 한도 내)
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)                                  # common.*
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))      # params_table3, synapse_pair (network_lib 의존)
sys.path.insert(0, THIS)                                    # network_lib
import network_lib as net                                   # noqa: E402  ← 단계 함수 모음

MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(THIS, "figures")
TSTOP_FULL, TSTOP_DEMO = 500.0, 300.0
DEMO_COUNTS = {"PC": 30, "PV": 6, "cAC": 3, "bAC": 3}        # 축소 데모 타입별 세포 수


def load_connectivity():
    """1_build_connectivity.py 산출물(connectivity.json) 로드."""
    with open(os.path.join(THIS, "connectivity.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    demo = "--demo" in sys.argv
    tstop = TSTOP_DEMO if demo else TSTOP_FULL
    os.makedirs(OUT, exist_ok=True)

    # 연결도 로드 (+ 데모면 타입 균형 서브샘플)
    conn = load_connectivity()
    cells_meta, edges = conn["cells"], conn["edges"]
    if demo:
        cells_meta, edges = net.subsample(cells_meta, edges, DEMO_COUNTS)
        print(f"[축소데모] {len(cells_meta)}세포 · {len(edges)}연결 (원본 연결도 보존)", flush=True)
    rng = np.random.RandomState(7)
    types = [c["type"] for c in cells_meta]
    keep = []                                                # GC 방지: 생성 객체 보관

    # ===================== 파이프라인 (흐름이 그대로 보임) =====================
    type_dir = net.load_representatives(MODELS)                                  # 0) 대표 모델 4종
    print("[로드] 대표 e-type: " + ", ".join(type_dir.keys()), flush=True)

    print(f"[1/5 구축] 세포 {len(cells_meta)}개 인스턴스화 …", flush=True)
    cells = net.build_cells(cells_meta, type_dir)                                # 1) 세포 구축

    print(f"[2/5 연결] {len(edges)}개 9클래스 EMS 시냅스 배선 …", flush=True)
    n_ok, n_fail = net.wire_synapses(cells, edges, rng, keep)                    # 2) 시냅스 연결
    if n_fail:
        print(f"  [경고] {n_fail}/{len(edges)} 실패(건너뜀)", flush=True)

    print("[3/5 구동] 전 세포 소마에 외부 Poisson 입력 …", flush=True)
    net.add_external_drive(cells, cells_meta, 0, keep)                           # 3) 외부 구동

    spikes = net.record_spikes(cells, keep)                                      # 4) 스파이크 기록 준비

    print(f"[4/5 실행] TSTOP={tstop:.0f}ms, 고정 dt …", flush=True)
    net.run_network(tstop)                                                       # 5) 실행

    spk = net.spikes_to_arrays(spikes)
    out_png = os.path.join(OUT, "2_network_activity_demo.png" if demo else "2_network_activity.png")
    print("[5/5 분석] raster + 발화율 …", flush=True)
    net.analyze_activity(cells_meta, types, spk, tstop, out_png, demo)           # 6) 분석


if __name__ == "__main__":
    main()
