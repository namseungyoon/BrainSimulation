"""
Hodgkin-Huxley 단일 뉴런 시뮬레이션 (기본 예제)
================================================

NEURON 라이브러리 없이 numpy/matplotlib 만으로 동작하는 생물물리 뉴런 모델.
soma 한 구획(single compartment)에 계단형 전류(step current)를 주입하고,
막전위(Vm) vs 시간 그래프를 저장한다.

- 모델: 고전적 Hodgkin-Huxley (1952), Na + K + leak 채널
- 적분: 4차 Runge-Kutta (RK4)
- 출력: voltage_plot.png (Vm, 게이팅 변수, 주입 전류)

실행:
    python hh_neuron.py
"""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로 저장 (headless 환경 대응)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. 모델 파라미터 (전형적인 HH 값, 단위: mV, mS/cm^2, uF/cm^2, uA/cm^2)
# ---------------------------------------------------------------------------
class HHParams:
    Cm = 1.0        # 막 용량 (uF/cm^2)

    gNa = 120.0     # 최대 Na 전도도 (mS/cm^2)
    gK = 36.0       # 최대 K  전도도 (mS/cm^2)
    gL = 0.3        # 누설 전도도   (mS/cm^2)

    ENa = 50.0      # Na 평형전위 (mV)
    EK = -77.0      # K  평형전위 (mV)
    EL = -54.387    # 누설 평형전위 (mV)


# ---------------------------------------------------------------------------
# 2. 게이팅 변수의 rate 함수 (alpha/beta)
#    수치 안정성을 위해 분모가 0이 되는 지점은 극한값으로 처리
# ---------------------------------------------------------------------------
def alpha_m(V):
    return np.where(np.isclose(V, -40.0),
                    1.0,
                    0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def alpha_n(V):
    return np.where(np.isclose(V, -55.0),
                    0.1,
                    0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


# ---------------------------------------------------------------------------
# 3. 미분방정식: dy/dt = f(t, y),  y = [V, m, h, n]
# ---------------------------------------------------------------------------
def derivatives(t, y, I_inj, p: HHParams):
    V, m, h, n = y

    I_Na = p.gNa * (m ** 3) * h * (V - p.ENa)
    I_K = p.gK * (n ** 4) * (V - p.EK)
    I_L = p.gL * (V - p.EL)

    dVdt = (I_inj(t) - I_Na - I_K - I_L) / p.Cm
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n

    return np.array([dVdt, dmdt, dhdt, dndt])


# ---------------------------------------------------------------------------
# 4. RK4 적분 루프
# ---------------------------------------------------------------------------
def simulate(t_stop=50.0, dt=0.01, I_inj=None, p=HHParams()):
    """HH 모델을 시간 적분한다. 반환: (t, V, m, h, n)"""
    if I_inj is None:
        # 기본 주입 전류: 5~45 ms 동안 10 uA/cm^2 계단 자극
        def I_inj(t):
            return 10.0 if 5.0 <= t <= 45.0 else 0.0

    n_steps = int(t_stop / dt) + 1
    t = np.linspace(0.0, t_stop, n_steps)

    # 정지 상태(rest)에서의 초기 게이팅 변수
    V0 = -65.0
    y = np.array([
        V0,
        alpha_m(V0) / (alpha_m(V0) + beta_m(V0)),
        alpha_h(V0) / (alpha_h(V0) + beta_h(V0)),
        alpha_n(V0) / (alpha_n(V0) + beta_n(V0)),
    ])

    out = np.zeros((n_steps, 4))
    out[0] = y

    for i in range(1, n_steps):
        ti = t[i - 1]
        k1 = derivatives(ti, y, I_inj, p)
        k2 = derivatives(ti + dt / 2, y + dt / 2 * k1, I_inj, p)
        k3 = derivatives(ti + dt / 2, y + dt / 2 * k2, I_inj, p)
        k4 = derivatives(ti + dt, y + dt * k3, I_inj, p)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = y

    V, m, h, n = out.T
    return t, V, m, h, n, np.array([I_inj(tt) for tt in t])


# ---------------------------------------------------------------------------
# 5. 결과 플롯
# ---------------------------------------------------------------------------
def plot_results(t, V, m, h, n, I, out_path="voltage_plot.png"):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(t, V, color="navy")
    axes[0].set_ylabel("Vm (mV)")
    axes[0].set_title("Hodgkin-Huxley single-neuron simulation")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, m, label="m (Na activation)")
    axes[1].plot(t, h, label="h (Na inactivation)")
    axes[1].plot(t, n, label="n (K activation)")
    axes[1].set_ylabel("gating variables")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, I, color="darkred")
    axes[2].set_ylabel("I_inj (uA/cm^2)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"플롯 저장됨: {Path(out_path).resolve()}")


# ---------------------------------------------------------------------------
# 6. 메인
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Hodgkin-Huxley 시뮬레이션 시작...")
    t, V, m, h, n, I = simulate(t_stop=50.0, dt=0.01)

    # 스파이크 개수 세기 (0 mV 상향 교차)
    spikes = np.sum((V[:-1] < 0.0) & (V[1:] >= 0.0))
    print(f"시뮬레이션 완료: {len(t)} 스텝, 검출된 스파이크 수 = {spikes}")
    print(f"Vm 범위: {V.min():.1f} ~ {V.max():.1f} mV")

    plot_results(t, V, m, h, n, I, out_path="voltage_plot.png")
