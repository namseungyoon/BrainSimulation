"""
02_STDP.py
------------------------------------------------------------
스파이크 타이밍 의존 가소성 (Spike-Timing-Dependent Plasticity, STDP)

시냅스 전(pre)/후(post) 스파이크의 상대적 타이밍 차이
  dt = t_post - t_pre
에 따라 시냅스 가중치가 강화(LTP) 또는 약화(LTD)된다.

  dt > 0 (pre -> post, 인과적):  강화(LTP),  dw = +A_plus  * exp(-dt/tau_plus)
  dt < 0 (post -> pre):          약화(LTD),  dw = -A_minus * exp( dt/tau_minus)

또한 한 쌍씩이 아니라 연속 자극열에서 가중치가 어떻게
누적 변화하는지(trace 기반 online 규칙)도 함께 보여준다.
------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------- (1) 고전적 STDP 창 (window) ----------
def stdp_window(dt, A_plus=0.10, A_minus=0.12,
                tau_plus=17.0, tau_minus=34.0):
    """dt = t_post - t_pre (ms) 에 대한 가중치 변화량 dw."""
    dw = np.where(
        dt >= 0,
        A_plus * np.exp(-dt / tau_plus),       # LTP
        -A_minus * np.exp(dt / tau_minus),     # LTD
    )
    return dw


# ---------- (2) trace 기반 online STDP ----------
def stdp_online(pre_spikes, post_spikes, t_max=1000.0, dt=0.5,
                A_plus=0.01, A_minus=0.012,
                tau_plus=17.0, tau_minus=34.0):
    """
    pre/post trace 를 적분하며 가중치 w 의 시간 변화를 계산.
    pre 스파이크 시 : w += A_plus  * (post trace)
    post 스파이크 시: w -= A_minus * (pre  trace)
    """
    t = np.arange(0, t_max, dt)
    pre_tr = np.zeros_like(t)
    post_tr = np.zeros_like(t)
    w = np.zeros_like(t)
    w0 = 0.5

    pre_set = set(np.round(pre_spikes / dt).astype(int))
    post_set = set(np.round(post_spikes / dt).astype(int))

    cur_pre = cur_post = 0.0
    cur_w = w0
    for i in range(len(t)):
        # trace 감쇠
        cur_pre *= np.exp(-dt / tau_plus)
        cur_post *= np.exp(-dt / tau_minus)

        if i in pre_set:
            cur_pre += 1.0
            cur_w -= A_minus * cur_post     # pre 도착: post trace 만큼 LTD
        if i in post_set:
            cur_post += 1.0
            cur_w += A_plus * cur_pre       # post 도착: pre trace 만큼 LTP

        cur_w = np.clip(cur_w, 0.0, 1.0)
        pre_tr[i], post_tr[i], w[i] = cur_pre, cur_post, cur_w

    return t, pre_tr, post_tr, w


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Spike-Timing-Dependent Plasticity (STDP)",
                 fontsize=14, fontweight="bold")

    # --- 왼쪽: STDP 학습 창 ---
    dt = np.linspace(-80, 80, 400)
    dw = stdp_window(dt)
    ax = axes[0]
    ax.plot(dt, dw, color="tab:purple", lw=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.fill_between(dt, dw, 0, where=dt >= 0, color="tab:red", alpha=0.2,
                    label="LTP (pre->post)")
    ax.fill_between(dt, dw, 0, where=dt < 0, color="tab:blue", alpha=0.2,
                    label="LTD (post->pre)")
    ax.set_xlabel(r"$\Delta t = t_{post} - t_{pre}$  (ms)")
    ax.set_ylabel(r"$\Delta w$")
    ax.set_title("STDP learning window")
    ax.legend()

    # --- 오른쪽: online trace 기반 가중치 변화 ---
    # pre 가 post 보다 약간 먼저 도달 -> 전체적으로 LTP 경향
    pre_spikes = np.arange(50, 950, 50, dtype=float)
    post_spikes = pre_spikes + 8.0      # pre 후 8 ms 뒤 post
    t, pre_tr, post_tr, w = stdp_online(pre_spikes, post_spikes)

    ax2 = axes[1]
    ax2.plot(t, w, color="tab:green", lw=2, label="weight w")
    for s in pre_spikes:
        ax2.axvline(s, color="tab:blue", ls=":", lw=0.5, alpha=0.5)
    for s in post_spikes:
        ax2.axvline(s, color="tab:red", ls=":", lw=0.5, alpha=0.5)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("synaptic weight")
    ax2.set_title("Online STDP: pre leads post by 8 ms -> potentiation")
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "02_STDP.png"
    plt.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
