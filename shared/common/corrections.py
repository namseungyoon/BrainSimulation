# Source: Ecker(2020) §2.7, Eq.(12)-(13) — 실험조건 통일을 위한 보정 함수
# 칼슘([Ca2+]o) / 온도(Q10) / LJP 보정. 시냅스 파라미터를 한 기준으로 맞출 때 사용.
import numpy as np


def hill_ca(ca, USE_max, K_half, n=4):
    """식(12): 세포외 칼슘에 따른 방출확률 U_SE (Hill, n=4).
    U_SE = USE_max * ca^n / (K_half^n + ca^n)."""
    ca = np.asarray(ca, dtype=float)
    return USE_max * ca ** n / (K_half ** n + ca ** n)


def q10_scale(tau_exp, Q10, T_exp, T_sim):
    """식(13): 온도 보정. τ_sim = τ_exp / Q10^((T_sim - T_exp)/10)."""
    return tau_exp / Q10 ** ((T_sim - T_exp) / 10.0)


def ljp_correct(v_measured, ljp):
    """LJP(액간접합전위) 보정: 보고된 전위에서 LJP 만큼 뺀다(부호는 규약 따름)."""
    return v_measured - ljp
