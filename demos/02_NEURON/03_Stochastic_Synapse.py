"""
03_Stochastic_Synapse.py
------------------------------------------------------------
확률적(Stochastic / Probabilistic) 시냅스 모델

실제 시냅스는 스파이크가 도착해도 항상 신경전달물질을 방출하지
않는다. 여러 방출 부위(release site, N)가 각각 확률 Pr 로
독립적으로 방출되는 이항(binomial) 과정으로 모델링한다.

  방출 소포 수 k ~ Binomial(N, Pr)
  PSC 진폭     = k * q     (q: 소포 1개당 양자 진폭, quantal size)

여기에 단기 우울(short-term depression)을 결합하여, 방출 후
가용 부위가 줄고 tau_rec 로 회복되는 확률적 시냅스를 보인다.
이로 인해 같은 자극에도 시행(trial)마다 반응이 달라지는
'시냅스 잡음(synaptic variability)'이 나타난다.
------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt


def simulate_stochastic(spike_times, N=5, Pr0=0.5, q=1.0,
                        tau_rec=300.0, rng=None):
    """
    확률적 시냅스 1회 시행.
    각 방출 부위는 '사용 가능' 상태일 때만 확률 Pr0 로 방출하고,
    방출 후 tau_rec 로 회복된다.
    반환: 각 스파이크의 PSC 진폭 배열, 방출 소포 수 배열
    """
    if rng is None:
        rng = np.random.default_rng()

    # 각 부위의 회복 확률 추적 (1 = 완전 가용)
    avail = np.ones(N)
    last_spk = None
    amps, releases = [], []

    for spk in spike_times:
        if last_spk is not None:
            dt_spk = spk - last_spk
            # 비어 있던 부위가 회복될 확률
            recover_p = 1.0 - np.exp(-dt_spk / tau_rec)
            recovered = rng.random(N) < recover_p
            avail = np.clip(avail + recovered, 0, 1)

        # 가용 부위 중 Pr0 확률로 방출
        usable = avail > 0
        release = usable & (rng.random(N) < Pr0)
        k = int(release.sum())
        amps.append(k * q)
        releases.append(k)

        # 방출한 부위는 비워짐
        avail[release] = 0
        last_spk = spk

    return np.array(amps), np.array(releases)


def main():
    spike_times = np.arange(50, 50 + 10 * 40, 40, dtype=float)  # 25 Hz, 10발
    N, Pr0, q, tau_rec = 5, 0.5, 1.0, 300.0
    n_trials = 50
    rng = np.random.default_rng(42)

    all_amps = np.array([
        simulate_stochastic(spike_times, N, Pr0, q, tau_rec, rng)[0]
        for _ in range(n_trials)
    ])  # shape: (n_trials, n_spikes)

    mean_amp = all_amps.mean(axis=0)
    std_amp = all_amps.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"Stochastic Synapse  (binomial release: N={N}, Pr={Pr0}, q={q})",
        fontsize=14, fontweight="bold")

    # (1) 개별 시행들 (variability)
    ax = axes[0]
    x = np.arange(1, len(spike_times) + 1)
    for tr in all_amps[:15]:
        ax.plot(x, tr, color="gray", alpha=0.3, lw=0.8)
    ax.plot(x, mean_amp, color="tab:red", lw=2.5, label="mean")
    ax.set_xlabel("spike #")
    ax.set_ylabel("PSC amplitude (k*q)")
    ax.set_title("Trial-to-trial variability")
    ax.legend()

    # (2) 평균 +/- 표준편차 (depression 경향)
    ax2 = axes[1]
    ax2.errorbar(x, mean_amp, yerr=std_amp, fmt="o-", color="tab:blue",
                 capsize=3)
    ax2.set_xlabel("spike #")
    ax2.set_ylabel("mean PSC +/- SD")
    ax2.set_title("Mean response (depression by depletion)")

    # (3) 첫 스파이크 방출 소포 수 분포 vs 이항분포 이론값
    ax3 = axes[2]
    first_release = np.array([
        simulate_stochastic(spike_times, N, Pr0, q, tau_rec,
                            np.random.default_rng(s))[1][0]
        for s in range(2000)
    ])
    ks = np.arange(0, N + 1)
    obs = np.array([(first_release == k).mean() for k in ks])
    # 이론 이항분포
    from math import comb
    theo = np.array([comb(N, k) * Pr0**k * (1 - Pr0)**(N - k) for k in ks])
    width = 0.35
    ax3.bar(ks - width/2, obs, width, label="simulated", color="tab:green")
    ax3.bar(ks + width/2, theo, width, label="Binomial(N,Pr)",
            color="tab:orange", alpha=0.8)
    ax3.set_xlabel("vesicles released (k)")
    ax3.set_ylabel("probability")
    ax3.set_title("Release distribution (1st spike)")
    ax3.set_xticks(ks)
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = "03_Stochastic_Synapse.png"
    plt.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
