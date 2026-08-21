# -*- coding: utf-8 -*-
"""lib/measure.py — EPSP·PPR·CV 등 시냅스 응답 계량 (번호 없음 = import 전용 모듈)

전 단계·실험이 공유하는 측정 함수. 소마 막전위 트레이스에서 EPSP 특성을 뽑는다.
"""
import numpy as np


def epsp_features(t, v, t_event, base_win=5.0, search=40.0):
    """단발 EPSP 특성. t_event(ms) 이후 최대 탈분극을 EPSP 로 본다.

    반환 dict:
      amp_mV      정점 진폭(기저선 대비)
      latency_ms  개시지연 = t_event 부터 5% 도달까지
      rise_ms     상승시간 20->80%
      halfwidth_ms 반치폭
      decay_ms    정점 후 37%(1/e) 로 떨어지는 시간
      t_peak_ms   정점 시각
    """
    t = np.asarray(t); v = np.asarray(v)
    base = v[(t >= t_event - base_win) & (t < t_event)].mean()
    m = (t >= t_event) & (t <= t_event + search)
    tv, vv = t[m], v[m]
    if len(vv) < 3:
        return None
    dv = vv - base
    ipk = int(np.argmax(dv))
    amp = float(dv[ipk]); t_peak = float(tv[ipk])
    if amp <= 0:
        return dict(amp_mV=float(amp), latency_ms=float("nan"), rise_ms=float("nan"),
                    halfwidth_ms=float("nan"), decay_ms=float("nan"), t_peak_ms=t_peak)

    def cross(frac, seg, rising=True):
        thr = amp * frac
        if rising:
            idx = np.where(seg[:ipk+1] >= thr)[0]
            return tv[idx[0]] if len(idx) else float("nan")
        else:
            idx = np.where(seg[ipk:] <= thr)[0]
            return tv[ipk + idx[0]] if len(idx) else float("nan")

    t5 = cross(0.05, dv, True)
    t20 = cross(0.20, dv, True); t80 = cross(0.80, dv, True)
    thalf_up = cross(0.50, dv, True); thalf_dn = cross(0.50, dv, False)
    tdecay = cross(1/np.e, dv, False)
    return dict(
        amp_mV=amp, t_peak_ms=t_peak,
        latency_ms=float(t5 - t_event),
        rise_ms=float(t80 - t20),
        halfwidth_ms=float(thalf_dn - thalf_up),
        decay_ms=float(tdecay - t_peak),
    )


def peak_amp(t, v, t_event, base_win=5.0, search=40.0):
    """EPSP 정점 진폭(mV)만 빠르게."""
    f = epsp_features(t, v, t_event, base_win, search)
    return f["amp_mV"] if f else 0.0


def cv(amps):
    """변동계수 CV = std/mean."""
    a = np.asarray(amps, dtype=float)
    mu = a.mean()
    return float(a.std(ddof=1) / mu) if mu != 0 and len(a) > 1 else float("nan")


def failure_rate(amps, thr_mV=0.02):
    """실패율 = 진폭이 thr 미만인 시행 비율."""
    a = np.asarray(amps, dtype=float)
    return float((a < thr_mV).mean()) if len(a) else float("nan")
