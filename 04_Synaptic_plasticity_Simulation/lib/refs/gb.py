# -*- coding: utf-8 -*-
"""lib/refs/gb.py — Graupner-Brunel 칼슘 기반 장기가소성 순수 numpy 참조 (번호 없음)

Graupner & Brunel (2012) PNAS 109(10):3991-3996. NEURON 없이 스파이크열로부터
칼슘 궤적 c(t) 와 효능 rho(t) 를 적분한다. 5-3~5-5 에서 mod 와 절대차로 대조하는 기준.

  칼슘: dc/dt = -c/tau_ca
        pre 스파이크  -> 지연 D 뒤에  c += C_pre
        post 스파이크 -> 즉시         c += C_post
  효능: tau * drho/dt = -rho(1-rho)(rho_star-rho)
                        + gamma_p (1-rho) H[c-theta_p]
                        - gamma_d  rho    H[c-theta_d]
        rho in [0,1], 이중안정(0=DOWN, 1=UP, rho_star=0.5 가 불안정 경계)
  전달: w = w0 + rho*(w1-w0),  w1 = b*w0

★ 노이즈(sigma) 는 뺐다 — mod 도 결정론(sigma=0)이므로 대조가 가능해야 한다.
★ 파라미터 기본값은 mechanisms/_build/GBPlasticitySyn.mod 와 **같은 값**이어야 한다.
  5-1 이 mod PARAMETER 블록을 파싱해 실제로 같은지 단언한다(드리프트 방지).
"""
import numpy as np

# Wittenberg & Wang 2006 해마 슬라이스 적합 (G&B 2012 Table S2) = mod 기본값
WITTENBERG2006 = dict(
    tau_ca=48.8373, C_pre=1.0, C_post=0.275865, D=18.8008,
    theta_d=1.0, theta_p=1.3, gamma_d=313.0965, gamma_p=1645.59,
    tau=688355.0, rho_star=0.5, b=5.28145, w0=1.0,
)


def calcium(t, pre_times, post_times, p=None):
    """칼슘 궤적. t: 시간축(ms, 등간격). 해석해(지수 합)로 계산 — 적분오차 없음."""
    p = dict(WITTENBERG2006 if p is None else p)
    t = np.asarray(t, dtype=float)
    c = np.zeros_like(t)
    for ts in np.atleast_1d(pre_times):
        te = ts + p["D"]                       # pre 는 지연 D 뒤에 기여
        m = t >= te
        c[m] += p["C_pre"] * np.exp(-(t[m] - te) / p["tau_ca"])
    for ts in np.atleast_1d(post_times):
        m = t >= ts                            # post 는 즉시
        c[m] += p["C_post"] * np.exp(-(t[m] - ts) / p["tau_ca"])
    return c


def calcium_amp(t, pre_times, pre_amps, post_times, post_amp=None, p=None):
    """스파이크마다 **다른 크기**의 칼슘 기여 (엔진 B/C 용).

    엔진 B 는 방출량에 따라 전시냅스 칼슘이 달라진다(ca_stp=1):
        C_pre_eff = C_pre * (1 + ca_stp*(prn - 1))
    그 값을 pre_amps 로 받는다. post 는 크기가 고정(C_post)이다.
    """
    p = dict(WITTENBERG2006 if p is None else p)
    t = np.asarray(t, dtype=float)
    c = np.zeros_like(t)
    pre_times = np.atleast_1d(pre_times)
    pre_amps = np.atleast_1d(pre_amps)
    if pre_times.size != pre_amps.size:
        raise ValueError("pre_times 와 pre_amps 의 길이가 같아야 한다")
    for ts, amp in zip(pre_times, pre_amps):
        te = ts + p["D"]
        m = t >= te
        c[m] += float(amp) * np.exp(-(t[m] - te) / p["tau_ca"])
    ca_post = p["C_post"] if post_amp is None else post_amp
    for ts in np.atleast_1d(post_times) if len(np.atleast_1d(post_times)) else []:
        m = t >= ts
        c[m] += ca_post * np.exp(-(t[m] - ts) / p["tau_ca"])
    return c


def drho(rho, c, p):
    """drho/dt (1/ms). Heaviside 는 c > theta 로 (mod 와 동일한 강부등호)."""
    Hp = 1.0 if c > p["theta_p"] else 0.0
    Hd = 1.0 if c > p["theta_d"] else 0.0
    return (-rho * (1.0 - rho) * (p["rho_star"] - rho)
            + p["gamma_p"] * (1.0 - rho) * Hp
            - p["gamma_d"] * rho * Hd) / p["tau"]


def integrate_rho(t, c, rho0=0.0, p=None):
    """RK4 로 rho(t) 적분. t 는 등간격 가정. mod 의 derivimplicit 와 절대차로 비교한다."""
    p = dict(WITTENBERG2006 if p is None else p)
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])
    rho = np.empty_like(t)
    r = float(rho0)
    for i in range(t.size):
        rho[i] = r
        if i == t.size - 1:
            break
        ci, cn = c[i], c[i + 1]
        cm = 0.5 * (ci + cn)
        k1 = drho(r, ci, p)
        k2 = drho(r + 0.5 * dt * k1, cm, p)
        k3 = drho(r + 0.5 * dt * k2, cm, p)
        k4 = drho(r + dt * k3, cn, p)
        r = r + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        r = min(max(r, 0.0), 1.0)
    return rho


def simulate(pre_times, post_times, tstop, dt=0.025, rho0=0.0, p=None):
    """편의 함수 -> (t, c, rho)."""
    p = dict(WITTENBERG2006 if p is None else p)
    t = np.arange(0.0, tstop + 0.5 * dt, dt)
    c = calcium(t, pre_times, post_times, p)
    return t, c, integrate_rho(t, c, rho0=rho0, p=p)


def weight(rho, p=None):
    """효능 rho -> 전달 가중치 w = w0 + rho*(b*w0 - w0)."""
    p = dict(WITTENBERG2006 if p is None else p)
    return p["w0"] + np.asarray(rho) * (p["b"] * p["w0"] - p["w0"])


def potential(rho, p=None):
    """자율항의 포텐셜 U(rho) (자극 없을 때). drho/dt = -dU/drho * (1/tau).

    -rho(1-rho)(rho_star-rho) 를 적분하면
    U(rho) = rho^4/4 - (1+rho_star) rho^3/3 + rho_star rho^2/2   (부호 정리 후)
    이중우물이며 극소는 0·1, 극대는 rho_star.
    """
    p = dict(WITTENBERG2006 if p is None else p)
    rs = p["rho_star"]
    r = np.asarray(rho, dtype=float)
    return r ** 4 / 4.0 - (1.0 + rs) * r ** 3 / 3.0 + rs * r ** 2 / 2.0


def fixed_points(p=None):
    """자율항의 고정점과 안정성. 반환 [(rho, 'stable'|'unstable'), ...]."""
    p = dict(WITTENBERG2006 if p is None else p)
    rs = p["rho_star"]
    out = []
    for r in (0.0, rs, 1.0):
        h = 1e-4
        slope = (drho(r + h, 0.0, p) - drho(r - h, 0.0, p)) / (2 * h)
        out.append((r, "stable" if slope < 0 else "unstable"))
    return out
