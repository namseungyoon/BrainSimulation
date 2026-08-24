# -*- coding: utf-8 -*-
"""lib/refs/stdp.py — 고전 짝기반 STDP 순수 numpy 참조 (번호 없음)

Bi & Poo (1998) 형태의 지수 창. 칼슘도 상태도 없고 **스파이크 짝의 시간차만** 본다.
5-6 에서 우리가 새로 쓸 PairSTDPSyn.mod 와 절대차 < 1e-9 로 대조하는 기준.

  dw = +A_p * exp(-dt/tau_p)   (dt = t_post - t_pre > 0, pre->post = LTP)
  dw = -A_d * exp( dt/tau_d)   (dt < 0, post->pre = LTD)
  dt = 0 은 정의상 0 (실험적으로도 분리 불가)

★ 이 참조의 존재 이유는 "GB 와 무엇이 다른가" 를 보이는 것이다 —
  GB 는 칼슘 누적 때문에 **같은 dt 라도 주파수·이력에 따라 결과가 달라진다**.
  고전 STDP 는 dt 만 보므로 달라지지 않는다. 6-3·6-4 의 대비가 여기서 나온다.

⚠️ tau_p·tau_d 는 Bi & Poo 1998 로 널리 인용되는 값(16.8 / 33.7 ms)이지만
  **원문 미확보**다. 5-6 착수 전 원문으로 확정해야 한다(docs/DECISIONS.md 미결).
"""
import numpy as np

BI_POO_1998 = dict(A_p=1.0, A_d=1.0, tau_p=16.8, tau_d=33.7)   # A 는 상대값(정규화)


def window(dt, p=None):
    """단일 짝의 dw. dt = t_post - t_pre (ms). 배열 입력 허용."""
    p = dict(BI_POO_1998 if p is None else p)
    dt = np.asarray(dt, dtype=float)
    out = np.zeros_like(dt)
    pos, neg = dt > 0, dt < 0
    out[pos] = p["A_p"] * np.exp(-dt[pos] / p["tau_p"])
    out[neg] = -p["A_d"] * np.exp(dt[neg] / p["tau_d"])
    return out


def pairs(pre_times, post_times, p=None, all_to_all=True):
    """스파이크열 -> 누적 dw. all_to_all=True 는 모든 pre-post 짝을 더한다(고전 규약).

    all_to_all=False 는 nearest-neighbour(각 post 가 직전 pre 하나만) — 규약 차이가
    고빈도에서 큰 차이를 만들기 때문에 5-6 에서 어느 규약인지 명시해야 한다.
    """
    p = dict(BI_POO_1998 if p is None else p)
    pre = np.atleast_1d(np.asarray(pre_times, dtype=float))
    post = np.atleast_1d(np.asarray(post_times, dtype=float))
    if all_to_all:
        return float(window((post[:, None] - pre[None, :]).ravel(), p).sum())
    tot = 0.0
    for tp in post:                     # 직전 pre 하나
        prev = pre[pre < tp]
        if prev.size:
            tot += float(window(np.array([tp - prev[-1]]), p)[0])
    for ts in pre:                      # 직후 post 하나 (LTD 쪽)
        nxt = post[post < ts]
        if nxt.size:
            tot += float(window(np.array([nxt[-1] - ts]), p)[0])
    return tot


def protocol(dt_ms, n_pairs, freq_hz, p=None, all_to_all=True):
    """고전 STDP 프로토콜: n_pairs 개 짝을 freq_hz 로 반복. 누적 dw 반환.

    고전 STDP 는 dt 가 같으면 주파수가 달라도 짝당 dw 가 같다(all-to-all 의 꼬리 항만 다름).
    GB 는 그렇지 않다 — 이 대비가 5-1 의 요점이다.
    """
    isi = 1000.0 / freq_hz
    pre = np.arange(n_pairs) * isi
    post = pre + dt_ms
    return pairs(pre, post, p=p, all_to_all=all_to_all)
