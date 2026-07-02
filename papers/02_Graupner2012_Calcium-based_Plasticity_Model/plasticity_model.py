"""
plasticity_model.py — Graupner & Brunel (2012) 칼슘 기반 가소성 모델 핵심 라이브러리
============================================================================
Source: Graupner & Brunel (2012) PNAS 109(10):3991-3996,
        "Calcium-based plasticity model explains sensitivity of synaptic
         changes to spike pattern, rate, and dendritic location."

이 파일은 논문 전체 구현의 **단일 진실 소스(single source of truth)** 이다.
모든 단계 스크립트(01_..~09_..)는 여기서 파라미터와 적분기를 import 해서 쓴다.

핵심 방정식 (Eq.1):
    τ dρ/dt = -ρ(1-ρ)(ρ* - ρ)                         # 이중우물(cubic) — 활동 없을 때
              + γ_p (1-ρ) Θ[c(t) - θ_p]                # 강화(potentiation)
              - γ_d ρ Θ[c(t) - θ_d]                    # 약화(depression)
              + Noise(t)
    Noise(t) = σ √τ · √(Θ[c-θ_p] + Θ[c-θ_d]) · η(t)    # 활동 의존 백색잡음

칼슘 동역학 (§Results, Fig 1A):
    c(t) = Σ_pre  C_pre  · exp(-(t - t_pre  - D)/τ_Ca) · Θ(t - t_pre - D)
         + Σ_post C_post · exp(-(t - t_post   )/τ_Ca) · Θ(t - t_post)
    - pre 스파이크는 지연 D 후에 C_pre 만큼 점프, post 는 즉시 C_post 만큼 점프
    - 각 전이는 τ_Ca 로 지수 감쇠하며 선형 합산된다.

시냅스 세기 readout (Discussion):
    w = w0 + ρ (w1 - w0),   b = w1/w0 (UP/DOWN 세기 비)

★ 파라미터 출처 (CLAUDE.md 원칙: 수치는 소스 대조 확정)
   - 모든 값은 SI Appendix (Corrected Nov 28, 2012) Table S1/S2/S3 에서 그대로 옮겼다.
   - Table S1: Fig 2C,D STDP 곡선 7세트 + Fig S5 BCM 예시
   - Table S2: 실험 데이터 피팅 3세트 (hippocampal slices / cultures / cortical slices)
   - Table S3: Fig S1 예시 2세트
   - 주의: 표의 τ 단위는 **초(sec)**. 이 코드는 ms 로 통일하므로 τ[ms] = τ[s]×1000.

설계: 오프라인(그림용 벡터화 함수) + 온라인(Synapse.step, 향후 NEURON 결합용) 분리.

실행 환경: conda env `ca1sim` (numpy/scipy). NEURON 불필요.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


# ---------------------------------------------------------------------------
# 파라미터 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class Params:
    """Graupner-Brunel 모델 파라미터. 시간 단위는 ms 로 통일."""
    # --- 칼슘 동역학 ---
    C_pre: float          # pre 스파이크가 만드는 칼슘 점프 (무차원, θ 기준 정규화)
    C_post: float         # post 스파이크가 만드는 칼슘 점프
    tau_ca: float         # 칼슘 감쇠 시간상수 [ms]
    D: float              # pre 유발 칼슘 전이의 지연 [ms]
    # --- 효능 동역학 ---
    theta_d: float        # 약화(depression) 문턱
    theta_p: float        # 강화(potentiation) 문턱
    gamma_d: float        # 약화율
    gamma_p: float        # 강화율
    tau: float            # 효능 시간상수 [ms] (초 단위 → ×1000)
    sigma: float          # 잡음 진폭
    rho_star: float = 0.5 # 불안정 고정점(두 안정상태 경계). 본문: ρ*=0.5
    # --- 시냅스 세기 readout ---
    b: float = 5.0        # w1/w0 (UP/DOWN 비). # TODO(SI): 데이터셋별 확정
    beta: float = 0.5     # 자극 전 DOWN 상태 시냅스 비율 (분석해용)

    def with_(self, **kw) -> "Params":
        """일부 필드만 바꾼 새 Params 반환 (예: p.with_(C_post=0.3))."""
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# 파라미터 세트 (명명 사전) — SI Appendix Table S1/S2/S3 확정값
# ---------------------------------------------------------------------------
_S = 1000.0   # 초 → ms 환산 (표의 τ 는 초 단위)

PARAM_SETS: dict[str, Params] = {
    # === Table S1: Fig 2C,D 의 STDP 곡선 7세트 + Fig S5 BCM 예시 ==========
    #  칼슘진폭(C_pre,C_post)·문턱(θ_d,θ_p)이 곡선 유형을 결정. γ_d,γ_p,σ 는
    #  LTP/LTD 크기가 비슷하도록 조정됨. D 는 LTD→LTP 전이가 Δt=0 에서 나도록 선택.
    "DP":       Params(C_pre=1.0,  C_post=2.0,  tau_ca=20.0, D=13.7,
                       theta_d=1.0, theta_p=1.3, gamma_d=200.0, gamma_p=321.808,
                       sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
    "DPD":      Params(C_pre=0.9,  C_post=0.9,  tau_ca=20.0, D=4.6,
                       theta_d=1.0, theta_p=1.3, gamma_d=250.0, gamma_p=550.0,
                       sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
    "DPD_prime": Params(C_pre=1.0, C_post=2.0,  tau_ca=20.0, D=2.2,
                       theta_d=1.0, theta_p=2.5, gamma_d=50.0,  gamma_p=600.0,
                       sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
    "P":        Params(C_pre=2.0,  C_post=2.0,  tau_ca=20.0, D=0.0,
                       theta_d=1.0, theta_p=1.3, gamma_d=160.0, gamma_p=257.447,
                       sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
    "D":        Params(C_pre=0.6,  C_post=0.6,  tau_ca=20.0, D=0.0,
                       theta_d=1.0, theta_p=1.3, gamma_d=500.0, gamma_p=550.0,
                       sigma=5.6568, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
    "D_prime":  Params(C_pre=1.0,  C_post=2.0,  tau_ca=20.0, D=0.0,
                       theta_d=1.0, theta_p=3.5, gamma_d=60.0,  gamma_p=600.0,
                       sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
    "BCM":      Params(C_pre=1.0,  C_post=1.0,  tau_ca=20.0, D=0.0,   # C_pre=C_post 는 varied
                       theta_d=1.0, theta_p=1.3, gamma_d=200.0, gamma_p=400.0,
                       sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),

    # === Table S2: 실험 데이터 피팅 3세트 (bold 값은 고정, 나머지는 최적화됨) ===
    "hippo_slice_Wittenberg2006": Params(   # Fig 3, S10
        C_pre=1.0, C_post=0.275865, tau_ca=48.8373, D=18.8008,
        theta_d=1.0, theta_p=1.3, gamma_d=313.0965, gamma_p=1645.59,
        sigma=9.1844, tau=688.355*_S, rho_star=0.5, b=5.28145, beta=0.7),
    "hippo_culture_Wang2005": Params(       # Fig S3, S10
        C_pre=0.58156, C_post=1.76444, tau_ca=11.9536, D=10.0,
        theta_d=1.0, theta_p=1.3, gamma_d=61.141, gamma_p=113.6545,
        sigma=2.5654, tau=33.7596*_S, rho_star=0.5, b=36.0263, beta=0.5),
    "cortex_Sjostrom2001": Params(          # Fig 4, 5, S4
        C_pre=0.5617539, C_post=1.23964, tau_ca=22.6936, D=4.6098,
        theta_d=1.0, theta_p=1.3, gamma_d=331.909, gamma_p=725.085,
        sigma=3.3501, tau=346.3615*_S, rho_star=0.5, b=5.40988, beta=0.5),

    # === Table S3: Fig S1 예시 2세트 ====================================
    "figS1_DPprime": Params(C_pre=1.0, C_post=1.3, tau_ca=20.0, D=4.3,
        theta_d=1.0, theta_p=1.3, gamma_d=150.0, gamma_p=310.0,
        sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
    "figS1_DP": Params(C_pre=1.0, C_post=2.0, tau_ca=20.0, D=13.8,
        theta_d=1.0, theta_p=1.3, gamma_d=150.0, gamma_p=241.356,
        sigma=2.8284, tau=150*_S, rho_star=0.5, b=5.0, beta=0.5),
}

# 편의 별칭: Step 1(칼슘 트레이스 데모)은 DP-curve 세트를 사용한다.
PARAM_SETS["demo_fig2"] = PARAM_SETS["DP"]


# ---------------------------------------------------------------------------
# 1) 칼슘 동역학  (Step 01)
# ---------------------------------------------------------------------------
def calcium_trace(t, pre_spikes, post_spikes, p: Params):
    """
    pre/post 스파이크 시각으로부터 칼슘 트레이스 c(t) 를 계산.

    Parameters
    ----------
    t : (N,) array   시간 격자 [ms]
    pre_spikes  : iterable   pre 스파이크 시각 [ms]
    post_spikes : iterable   post 스파이크 시각 [ms]
    p : Params

    Returns
    -------
    c : (N,) array   칼슘 농도 (무차원, θ 기준)
    """
    t = np.asarray(t, dtype=float)
    c = np.zeros_like(t)
    for ts in pre_spikes:
        onset = ts + p.D                       # pre 전이는 지연 D 후
        m = t >= onset
        c[m] += p.C_pre * np.exp(-(t[m] - onset) / p.tau_ca)
    for ts in post_spikes:
        m = t >= ts                            # post 전이는 즉시
        c[m] += p.C_post * np.exp(-(t[m] - ts) / p.tau_ca)
    return c


# ---------------------------------------------------------------------------
# 2) 효능 동역학 — 결정론적 drift + 이중우물 포텐셜  (Step 02)
# ---------------------------------------------------------------------------
def drift(rho, c, p: Params):
    """
    Eq.1 의 결정론적 우변 (노이즈 제외), 단위 [1/ms].
        dρ/dt = drift(ρ, c)
    c 는 스칼라 또는 rho 와 브로드캐스트 가능한 배열.
    """
    Hp = (c > p.theta_p).astype(float) if np.ndim(c) else float(c > p.theta_p)
    Hd = (c > p.theta_d).astype(float) if np.ndim(c) else float(c > p.theta_d)
    cubic = -rho * (1.0 - rho) * (p.rho_star - rho)
    pot = p.gamma_p * (1.0 - rho) * Hp
    dep = p.gamma_d * rho * Hd
    return (cubic + pot - dep) / p.tau


def potential(rho, p: Params, calcium: float = 0.0):
    """
    고정 칼슘 값에서의 포텐셜 U(ρ) (τ dρ/dt = -dU/dρ 를 만족, 상수항 무시).
    Fig 1D 의 이중우물/이차 포텐셜 그리기용.

    U(ρ) = ρ^4/4 - (1+ρ*)ρ^3/3 + ρ* ρ^2/2
           - γ_p Hp (ρ - ρ^2/2) + γ_d Hd ρ^2/2
    """
    Hp = float(calcium > p.theta_p)
    Hd = float(calcium > p.theta_d)
    rho = np.asarray(rho, dtype=float)
    U = (rho**4 / 4.0
         - (1.0 + p.rho_star) * rho**3 / 3.0
         + p.rho_star * rho**2 / 2.0
         - p.gamma_p * Hp * (rho - rho**2 / 2.0)
         + p.gamma_d * Hd * rho**2 / 2.0)
    return U


# ---------------------------------------------------------------------------
# 3) 문턱 초과 시간  (Step 03)
# ---------------------------------------------------------------------------
def time_above_thresholds(t, c, p: Params):
    """
    칼슘이 각 문턱 위에 머무는 시간 비율 (α_p, α_d).
    Fig 1B / 2B 의 'fraction of time above threshold'.
    """
    t = np.asarray(t, dtype=float)
    total = t[-1] - t[0]
    dt = np.gradient(t)
    alpha_p = np.sum(dt[c > p.theta_p]) / total
    alpha_d = np.sum(dt[c > p.theta_d]) / total
    return alpha_p, alpha_d


# ---------------------------------------------------------------------------
# 4) 효능 적분기 — Euler–Maruyama (노이즈 포함)  (Step 02/04)
# ---------------------------------------------------------------------------
def integrate_rho(t, c, p: Params, rho0=0.0, noise=True, rng=None):
    """
    주어진 칼슘 트레이스 c(t) 에 대해 효능 ρ(t) 를 적분.
    Euler–Maruyama: dρ = drift·dt + σ√τ·√(Hp+Hd)·dW  (dW ~ N(0, dt))
    잡음 항은 τ dρ = ... 를 dρ 로 정규화하면 (σ/√τ)√(Hp+Hd) dW 형태가 된다.

    Returns
    -------
    rho : (N,) array
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    rho = np.empty_like(t)
    rho[0] = rho0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        r = rho[i]
        ci = c[i]
        r = r + drift(r, ci, p) * dt
        if noise:
            Hp = 1.0 if ci > p.theta_p else 0.0
            Hd = 1.0 if ci > p.theta_d else 0.0
            amp = p.sigma * np.sqrt(Hp + Hd) / np.sqrt(p.tau)
            r = r + amp * rng.normal(0.0, np.sqrt(dt))
        rho[i + 1] = np.clip(r, 0.0, 1.0)
    return rho


# ---------------------------------------------------------------------------
# 5) 시냅스 세기 readout  (Discussion)
# ---------------------------------------------------------------------------
def synaptic_strength(rho, p: Params, w0=1.0):
    """w = w0 + ρ(w1 - w0), w1 = b·w0."""
    w1 = p.b * w0
    return w0 + np.asarray(rho) * (w1 - w0)


def strength_change_ratio(rho_before, rho_after, p: Params):
    """
    자극 전후 평균 시냅스 세기 비 (Fig 2~5 세로축 'change in synaptic strength').
        ((1-β)·w(ρ_after,UP) + β·w(ρ_after,DOWN)) / w_before  형태의 단순화.
    여기서는 개별 시냅스의 ρ 변화로부터 세기비를 직접 계산.
    """
    w_before = synaptic_strength(rho_before, p)
    w_after = synaptic_strength(rho_after, p)
    return np.mean(w_after) / np.mean(w_before)


# ---------------------------------------------------------------------------
# 6) 온라인 스텝 인터페이스 (향후 NEURON per-timestep 결합용)
# ---------------------------------------------------------------------------
class Synapse:
    """
    이벤트 기반 시냅스 상태(칼슘 + 효능). NEURON 루프에서 매 dt 마다 step() 호출하고
    pre/post 스파이크 발생 시 플래그를 넘기면 그대로 결합 가능하도록 설계.
    """
    def __init__(self, p: Params, rho0=0.0, rng=None):
        self.p = p
        self.c = 0.0            # 현재 칼슘
        self.rho = rho0         # 현재 효능
        self._pre_pending = []  # (발효시각) 지연 D 큐
        self.rng = rng or np.random.default_rng()
        self.t = 0.0

    def _apply_pending_pre(self):
        due = [ts for ts in self._pre_pending if ts <= self.t]
        for _ in due:
            self.c += self.p.C_pre
        self._pre_pending = [ts for ts in self._pre_pending if ts > self.t]

    def step(self, dt, pre_spike=False, post_spike=False):
        """한 시간스텝 전진. pre/post_spike: 이번 스텝에 스파이크가 났는지."""
        p = self.p
        self.t += dt
        if pre_spike:
            self._pre_pending.append(self.t + p.D)  # 지연 후 발효
        if post_spike:
            self.c += p.C_post
        self._apply_pending_pre()
        # 칼슘 감쇠
        self.c *= np.exp(-dt / p.tau_ca)
        # 효능 갱신
        self.rho += drift(self.rho, self.c, p) * dt
        Hp = 1.0 if self.c > p.theta_p else 0.0
        Hd = 1.0 if self.c > p.theta_d else 0.0
        if Hp + Hd > 0.0:
            amp = p.sigma * np.sqrt(Hp + Hd) / np.sqrt(p.tau)
            self.rho += amp * self.rng.normal(0.0, np.sqrt(dt))
        self.rho = float(np.clip(self.rho, 0.0, 1.0))
        return self.rho

    def weight(self, w0=1.0):
        return synaptic_strength(self.rho, self.p, w0)


if __name__ == "__main__":
    # 최소 자기점검: demo 세트로 pre-post 쌍 칼슘 트레이스 요약 출력
    p = PARAM_SETS["demo_fig2"]
    t = np.arange(-50.0, 150.0, 0.1)
    c = calcium_trace(t, pre_spikes=[0.0], post_spikes=[20.0], p=p)
    ap, ad = time_above_thresholds(t, c, p)
    print(f"[self-check] max c = {c.max():.3f}, "
          f"time above theta_p = {ap*100:.2f}%, theta_d = {ad*100:.2f}%")
