# Source: Ecker(2020) §2.4 Eq.(5)-(6) — Tsodyks-Markram 결정론 STP (라이브러리 모듈)
# 번호 파일(4_tm_stp 등)이 공유하는 함수. 번호 파일은 서로 import 불가(파이썬 규칙)라 여기로 분리.
import numpy as np


def simulate_tm(spike_times, U, D, F):
    """결정론 TM: 각 발화에서의 (u, R, 반응크기 amp=u·R) 반환.
    R=가용자원(회복 시정수 D), u=방출확률(촉진 시정수 F)."""
    u, R = 0.0, 1.0
    us, Rs, amps = [], [], []
    last = None
    for spk in spike_times:
        if last is None:
            u = U
        else:
            dt = spk - last
            u = u * np.exp(-dt / F)
            R = 1.0 - (1.0 - R) * np.exp(-dt / D)
            u = u + U * (1.0 - u)
        amp = u * R
        R = R - u * R
        us.append(u); Rs.append(R + u * R); amps.append(amp)   # R 은 소모 전 값
        last = spk
    return np.array(us), np.array(Rs), np.array(amps)
