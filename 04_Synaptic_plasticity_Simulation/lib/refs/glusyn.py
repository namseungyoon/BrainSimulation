# -*- coding: utf-8 -*-
"""lib/refs/glusyn.py — 국소 전압 기반 스파인 칼슘 순수 numpy 참조 (번호 없음)

`mechanisms/GluSynapseCa.mod` 의 '정답'. 같은 수식을 두 번 독립으로 구현해 한쪽 오타를
드러낸다(5-1 과 같은 원칙). 효능 적분은 `lib.refs.gb.integrate_rho` 를 그대로 쓴다 —
GluSynapseCa 는 **칼슘 출처만** 바꾼 엔진이므로 효능 방정식이 같아야 한다.

  dc/dt = -c/tau_ca
          + k_nmda * gN_norm(t) * (e_ca - v(t))/norm_mV
          + k_vdcc * m_vdcc(v)  * (e_ca - v(t))/norm_mV

  gN_norm(t) = (B_NMDA - A_NMDA) * mggate(v)        <- gmax 로 나눈 정규화 전도도
  m_vdcc(v)  = 1 / (1 + exp(-(v - vh)/slope))
  mggate(v)  = 1 / (1 + exp(0.062*-v) * mg/2.62)

★ 칼슘 방정식은 **구동항에 대해 선형**이다. 그래서
      c(t; k_nmda, k_vdcc) = k_nmda * c_n(t) + k_vdcc * c_v(t)
  가 정확히 성립한다(각 갈래를 1 로 두고 따로 푼 것의 선형결합). 5-7 의 교정이 이 성질을 쓴다.
"""
import numpy as np

DEFAULTS = dict(
    tau_ca=48.8373, k_nmda=1.0, k_vdcc=0.0, e_ca=40.0,
    vh_vdcc=-30.0, slope_vdcc=7.0, norm_mV=100.0,
    tau_r_NMDA=9.0, tau_d_NMDA=61.0, NMDA_ratio=0.71, mg=1.0,
)


def mggate(v, mg=1.0):
    return 1.0 / (1.0 + np.exp(0.062 * -np.asarray(v, dtype=float)) * (mg / 2.62))


def m_vdcc(v, vh=-30.0, slope=7.0):
    return 1.0 / (1.0 + np.exp(-(np.asarray(v, dtype=float) - vh) / slope))


def nmda_kernel(t, pre_times, w, p):
    """정규화 NMDA 전도도 (B-A). mod 의 factor 정규화까지 같게 맞춘다."""
    tr, td = p["tau_r_NMDA"], p["tau_d_NMDA"]
    tp = (tr * td) / (td - tr) * np.log(td / tr)
    factor = 1.0 / (-np.exp(-tp / tr) + np.exp(-tp / td))
    amp = w * p["NMDA_ratio"] * factor
    out = np.zeros_like(np.asarray(t, dtype=float))
    for ts in np.atleast_1d(pre_times):
        m = t >= ts
        dtm = t[m] - ts
        out[m] += amp * (np.exp(-dtm / td) - np.exp(-dtm / tr))
    return out


def calcium(t, v, pre_times, w=1.0, p=None, split=False):
    """국소 전압 v(t) 와 방출 시각으로부터 칼슘 궤적. split=True 면 두 갈래를 따로.

    t 는 등간격(ms). 사다리꼴 적분 대신 지수 적분자(exact for piecewise-constant forcing)를
    써서 mod 의 cnexp 와 같은 정확도를 낸다.
    """
    p = dict(DEFAULTS if p is None else {**DEFAULTS, **p})
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    dt = float(t[1] - t[0])
    drive = (p["e_ca"] - v) / p["norm_mV"]
    gN = nmda_kernel(t, pre_times, w, p) * mggate(v, p["mg"])
    f_n = gN * drive
    f_v = m_vdcc(v, p["vh_vdcc"], p["slope_vdcc"]) * drive
    dec = np.exp(-dt / p["tau_ca"])
    coef = p["tau_ca"] * (1.0 - dec)          # exact for constant forcing over dt

    def integrate(f):
        c = np.zeros_like(t)
        for i in range(1, t.size):
            c[i] = c[i - 1] * dec + coef * f[i - 1]
        return c

    c_n, c_v = integrate(f_n), integrate(f_v)
    if split:
        return c_n, c_v
    return p["k_nmda"] * c_n + p["k_vdcc"] * c_v


def bap_waveform(t, t_peak, amp_mV, v_rest=-69.55, tau_r=0.4, tau_d=3.0):
    """합성 bAP 파형 — 진폭만 3-9 실측으로 맞춘다.

    ★우리 선택: 3-9 는 국소 전압의 **봉우리 진폭**만 저장했고 파형 자체는 저장하지 않았다.
    그래서 이중지수(상승 tau_r · 감쇠 tau_d)로 합성하고 진폭을 실측값에 맞춘다.
    수상돌기 bAP 는 소마 AP 보다 넓으므로 tau_d 를 3ms 로 뒀다(우리 선택 · 확인요).
    파형 모양이 아니라 **진폭의 위치 의존성**이 5-7 의 검사 대상이므로 이 근사가 결론을
    바꾸지 않는다 — 다만 절대 칼슘값은 파형에 의존하므로 그 사실을 함께 보고한다.
    """
    t = np.asarray(t, dtype=float)
    d = t - t_peak
    k = np.zeros_like(t)
    m = d >= -5.0
    x = d[m] + 5.0
    y = np.exp(-x / tau_d) - np.exp(-x / tau_r)
    tp = (tau_r * tau_d) / (tau_d - tau_r) * np.log(tau_d / tau_r)
    ynorm = np.exp(-tp / tau_d) - np.exp(-tp / tau_r)
    k[m] = y / ynorm
    return v_rest + amp_mV * np.clip(k, 0.0, None)
