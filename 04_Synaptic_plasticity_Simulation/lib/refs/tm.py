# -*- coding: utf-8 -*-
"""lib/refs/tm.py — Tsodyks-Markram 단기가소성 순수 numpy 참조 (번호 없음)

결정론 TM(Fuhrmann et al. 2002 / Ecker 2020 §2.4 Eq.5-6). NEURON 없이 스파이크열로부터
각 발화의 방출확률 u·가용자원 R·정규화 방출량(u*R)을 계산. 5-9 에서 NEURON mod 와 대조하는
기준이 된다. 04 독립 원칙에 따라 자체 구현(등가 수식, papers/ 파일에 의존하지 않음).

  u : 방출확률(utilization). Fac(촉진) 시상수로 이완, 발화마다 U 만큼 증가
  R : 가용자원(0~1). Dep(회복) 시상수로 1 로 회복, 발화마다 u*R 소모
  방출량 amp = u*R   (첫 발화 대비 정규화하면 촉진/억압 프로파일)
"""
import numpy as np


def simulate(spike_times, U, Dep, Fac):
    """스파이크 시각열 → (u, R, amp) 배열. amp[n]=u[n]*R[n] (소모 전 R 기준).

    Fac>Dep 이면 촉진(뒤 펄스가 큼), Fac<Dep 이면 억압.
    """
    st = np.asarray(spike_times, dtype=float)
    us, Rs, amps = [], [], []
    u, R, last = 0.0, 1.0, None
    for t in st:
        if last is None:
            u = U
        else:
            dt = t - last
            u = u * np.exp(-dt / Fac)
            R = 1.0 - (1.0 - R) * np.exp(-dt / Dep)
            u = u + U * (1.0 - u)
        amp = u * R
        us.append(u); Rs.append(R); amps.append(amp)
        R = R - u * R                 # 방출 후 자원 소모
        last = t
    return np.array(us), np.array(Rs), np.array(amps)


def train(n, freq_hz, U, Dep, Fac, t0=0.0):
    """등간격 n펄스 트레인의 정규화 방출량(첫 펄스=1)."""
    isi = 1000.0 / freq_hz
    st = t0 + np.arange(n) * isi
    _, _, amp = simulate(st, U, Dep, Fac)
    return st, amp / amp[0]
