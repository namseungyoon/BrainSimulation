# -*- coding: utf-8 -*-
"""12_lfp/test_lfp_sanity.py  —  LFP 계산기 단위·부호 리트머스 (NEURON 불필요, 빠름)

E4a 실제 시뮬 전에 lfp_calc 의 수식·상수·부호를 합성 기하로 검증하는 게이트.
검증 항목:
  (1) 단위 상수: 점전류원 I=1nA, r=100um, sigma=0.3 -> V=1/(4pi.0.3.100)=2.653e-3 mV=2.653uV
  (2) PSA vs LSA 원거리 수렴: 전극이 세그먼트길이의 수배 이상 멀면 상대차 < 1%
  (3) 부호: 내향전류(sink, I<0) 근처 전극에서 V<0 / 외향(source, I>0)에서 V>0
  (4) 선형성: V(I1+I2) = V(I1)+V(I2)
실행: <ca1sim>/python.exe 12_lfp/test_lfp_sanity.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lfp_calc import lsa_matrix, psa_matrix, compute_lfp


def make_geom(p0, p1, radius=1.0):
    p0 = np.atleast_2d(np.asarray(p0, float))
    p1 = np.atleast_2d(np.asarray(p1, float))
    mid = 0.5 * (p0 + p1)
    length = np.linalg.norm(p1 - p0, axis=1)
    rad = np.full(p0.shape[0], radius, float)
    return dict(p0=p0, p1=p1, mid=mid, length=length, radius=rad, segs=None)


def main():
    sigma = 0.3
    ok = True

    # (1) 점전류원 단위 리트머스: 아주 짧은 세그먼트를 원점에, 전극 (100,0,0)
    geom = make_geom([[0, 0, 0]], [[1e-9, 0, 0]], radius=0.5)
    elec = [[100.0, 0.0, 0.0]]
    Mp = psa_matrix(geom, elec, sigma)
    Ml = lsa_matrix(geom, elec, sigma)
    analytic = 1.0 / (4.0 * np.pi * sigma * 100.0)   # mV per nA
    print(f"(1) 단위 리트머스: 해석값={analytic*1e3:.4f} uV/nA | PSA={Mp[0,0]*1e3:.4f} | LSA={Ml[0,0]*1e3:.4f}")
    e1 = abs(Mp[0, 0] - analytic) / analytic
    print(f"    PSA 상대오차={e1:.2e}  -> {'OK' if e1 < 1e-6 else 'FAIL'}")
    ok &= e1 < 1e-6

    # (2) PSA vs LSA 원거리 수렴: 길이 20um 세그먼트, 전극 500um 옆
    g2 = make_geom([[-10, 0, 0]], [[10, 0, 0]], radius=0.5)   # 길이 20um
    e2 = [[0.0, 500.0, 0.0]]
    Mp2 = psa_matrix(g2, e2, sigma)[0, 0]
    Ml2 = lsa_matrix(g2, e2, sigma)[0, 0]
    rel = abs(Mp2 - Ml2) / abs(Mp2)
    print(f"(2) 원거리 PSA={Mp2*1e3:.4f} vs LSA={Ml2*1e3:.4f} uV/nA  상대차={rel:.2e} -> {'OK' if rel < 0.01 else 'FAIL'}")
    ok &= rel < 0.01

    # (2b) 근거리에서는 PSA/LSA 유의차(LSA가 정확) — 차이가 존재해야 정상
    e2b = [[0.0, 5.0, 0.0]]   # 5um 옆(세그먼트 반길이 10um보다 가까움)
    Mp2b = psa_matrix(g2, e2b, sigma)[0, 0]
    Ml2b = lsa_matrix(g2, e2b, sigma)[0, 0]
    relb = abs(Mp2b - Ml2b) / abs(Mp2b)
    print(f"(2b) 근거리 PSA={Mp2b*1e3:.3f} vs LSA={Ml2b*1e3:.3f} uV/nA 상대차={relb:.2%} -> {'OK(유의차 존재)' if relb > 0.05 else 'WARN'}")

    # (3) 부호: I<0(sink) -> V<0
    I_sink = np.array([[-1.0]])    # (N=1, T=1) nA, 내향
    I_src = np.array([[+1.0]])
    Vsink = compute_lfp(Mp, I_sink)[0, 0]
    Vsrc = compute_lfp(Mp, I_src)[0, 0]
    print(f"(3) 부호: sink(I=-1nA) V={Vsink*1e3:.3f}uV(<0?)  source(I=+1nA) V={Vsrc*1e3:.3f}uV(>0?)"
          f" -> {'OK' if (Vsink < 0 and Vsrc > 0) else 'FAIL'}")
    ok &= (Vsink < 0 and Vsrc > 0)

    # (4) 선형성: 세그먼트 2개, 전류 합의 전위 = 전위 합
    g4 = make_geom([[0, 0, 0], [50, 0, 0]], [[1e-9, 0, 0], [50 + 1e-9, 0, 0]], radius=0.5)
    e4 = [[25.0, 80.0, 0.0]]
    M4 = psa_matrix(g4, e4, sigma)
    Ia = np.array([[2.0], [0.0]]); Ib = np.array([[0.0], [-3.0]])
    lin = abs(compute_lfp(M4, Ia + Ib)[0, 0] - (compute_lfp(M4, Ia)[0, 0] + compute_lfp(M4, Ib)[0, 0]))
    print(f"(4) 선형성 잔차={lin:.2e} -> {'OK' if lin < 1e-12 else 'FAIL'}")
    ok &= lin < 1e-12

    # (5) 등방 무한매질: 전극 매우 멀면 V -> 0 (1/r 감쇠)
    e5 = [[0.0, 1e5, 0.0]]
    Vfar = psa_matrix(geom, e5, sigma)[0, 0]
    print(f"(5) 원거리 감쇠 V(r=1e5um)={Vfar*1e6:.4f} nV -> {'OK' if abs(Vfar) < 1e-4 else 'FAIL'}")
    ok &= abs(Vfar) < 1e-4

    print("=" * 48)
    print("SANITY:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
