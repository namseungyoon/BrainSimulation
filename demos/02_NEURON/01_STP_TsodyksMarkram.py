"""
01_STP_TsodyksMarkram.py
------------------------------------------------------------
단기 가소성 (Short-Term Plasticity, STP) - Tsodyks-Markram 모델

시냅스 전 뉴런이 연속으로 스파이크를 보낼 때, 신경전달물질의
가용 자원(resource)과 방출 확률(utilization)이 동적으로 변하여
시냅스 후 반응(PSC)이 점점 커지거나(facilitation) 작아지는(depression)
현상을 재현한다.

상태 변수
  u : 방출 확률 (utilization).  facilitation 에 의해 증가
  x : 가용 자원 비율 (0~1). 방출 후 소모되고 tau_rec 로 회복
스파이크 발생 시:
  u  <- u + U*(1-u)         (촉진)
  dI <- A * u * x           (PSC 진폭)
  x  <- x - u*x             (자원 소모)
스파이크 사이에는 u 와 x 가 각각 tau_facil, tau_rec 로 이완.
------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt


def simulate_tm(spike_times, U, tau_rec, tau_facil, A=1.0, t_max=500.0, dt=0.1):
    """Tsodyks-Markram STP 시뮬레이션. PSC 진폭 시계열을 반환."""
    t = np.arange(0, t_max, dt)
    psc = np.zeros_like(t)          # 시냅스 후 전류 파형
    amps = []                       # 각 스파이크의 PSC 진폭

    u = 0.0
    x = 1.0
    last_spk = None
    tau_decay = 3.0                 # PSC 자체의 감쇠 시정수 (ms)

    for spk in spike_times:
        if last_spk is None:
            u = U
        else:
            dt_spk = spk - last_spk
            # 스파이크 사이 이완
            u = u * np.exp(-dt_spk / tau_facil)
            x = 1.0 - (1.0 - x) * np.exp(-dt_spk / tau_rec)
            u = u + U * (1.0 - u)   # 촉진
        amp = A * u * x             # 이번 스파이크 진폭
        x = x - u * x               # 자원 소모
        amps.append(amp)
        # PSC 파형에 지수 감쇠 이벤트 추가
        idx = t >= spk
        psc[idx] += amp * np.exp(-(t[idx] - spk) / tau_decay)
        last_spk = spk

    return t, psc, np.array(amps)


def main():
    # 20 Hz 자극열 (50 ms 간격, 8발)
    spike_times = np.arange(50, 50 + 8 * 50, 50, dtype=float)

    # 세 가지 시냅스 유형: depression / facilitation / 혼합
    configs = {
        "Depressing\n(U=0.6, tau_rec=800)":   dict(U=0.6, tau_rec=800, tau_facil=10),
        "Facilitating\n(U=0.1, tau_facil=500)": dict(U=0.1, tau_rec=100, tau_facil=500),
        "Mixed\n(U=0.3, balanced)":            dict(U=0.3, tau_rec=300, tau_facil=300),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Short-Term Plasticity (Tsodyks-Markram model) @ 20 Hz",
                 fontsize=14, fontweight="bold")

    for col, (name, p) in enumerate(configs.items()):
        t, psc, amps = simulate_tm(spike_times, **p)

        # 윗줄: PSC 파형
        ax = axes[0, col]
        ax.plot(t, psc, color="tab:blue")
        for s in spike_times:
            ax.axvline(s, color="gray", ls=":", lw=0.6)
        ax.set_title(name, fontsize=10)
        if col == 0:
            ax.set_ylabel("PSC amplitude")

        # 아랫줄: 스파이크별 진폭 (정규화)
        ax2 = axes[1, col]
        ax2.stem(range(1, len(amps) + 1), amps / amps[0],
                 basefmt=" ", use_line_collection=True)
        ax2.set_xticks(range(1, len(amps) + 1))
        ax2.axhline(1.0, color="gray", ls="--", lw=0.6)
        ax2.set_xlabel("spike #")
        if col == 0:
            ax2.set_ylabel("normalized amp")
        ax2.set_ylim(0, max(2.0, (amps / amps[0]).max() * 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "01_STP_TsodyksMarkram.png"
    plt.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
