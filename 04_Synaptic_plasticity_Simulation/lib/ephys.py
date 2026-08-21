# -*- coding: utf-8 -*-
"""lib/ephys.py — 단일세포 전기생리 측정 (번호 없음 = import 전용 모듈)

계단전류(f-I)·입력저항·sag·활동전위 특성을 잰다. 소마에 IClamp 를 걸고 소마 Vm 을 기록.
고정 dt(cvode off)는 lib.nrnenv.finit 이 강제한다.
"""
import numpy as np

from lib.nrnenv import h
import lib.nrnenv as nrnenv


def _spike_times(t, v, thresh=-10.0):
    up = (v[:-1] < thresh) & (v[1:] >= thresh)
    return t[1:][up]


def step_response(cell, amp_nA, delay=100.0, dur=500.0, tail=150.0, rec_dt=0.1):
    """소마에 계단전류 1회. (t, v, spike_times) 반환."""
    ic = h.IClamp(cell.soma[0](0.5))
    ic.delay, ic.dur, ic.amp = delay, dur, amp_nA
    t = h.Vector().record(h._ref_t)
    v = h.Vector().record(cell.soma[0](0.5)._ref_v)
    nrnenv.finit(v_init=-70.0)
    h.continuerun(delay + dur + tail)
    t = np.array(t); v = np.array(v)
    return t, v, _spike_times(t, v)


def fi_curve(cell, amps):
    """계단전류 진폭 목록 → (amps, 발화율 Hz, 파형 dict).
    traces[a] = (t, v, sp_all) — 전체 스파이크 시각(재사용용)."""
    rates, traces = [], {}
    for a in amps:
        t, v, sp = step_response(cell, a)
        during = sp[(sp >= 100.0) & (sp <= 600.0)]
        rates.append(len(during) / 0.5)
        traces[a] = (t, v, sp)
    return np.array(amps), np.array(rates), traces


def input_resistance(cell, amp_nA=-0.05):
    """작은 음전류 계단 → 입력저항(MOhm)·sag ratio·Vrest."""
    t, v, _ = step_response(cell, amp_nA, delay=100.0, dur=400.0, tail=150.0)
    vrest = v[t < 100.0].mean()
    dur_m = (t >= 100.0) & (t <= 500.0)
    v_min = v[dur_m].min()                       # sag 최저(가장 과분극)
    v_ss = v[(t > 450.0) & (t <= 500.0)].mean()  # 정상상태
    dv_ss = v_ss - vrest
    Rin = abs(dv_ss / amp_nA)                    # mV/nA = MOhm
    # sag ratio = (최저 - 정상상태) / (최저 - 안정) ... Ih 있으면 정상상태가 최저보다 덜 과분극
    sag = abs((v_min - v_ss) / (v_min - vrest)) if (v_min - vrest) != 0 else 0.0
    return dict(vrest_mV=float(vrest), Rin_MOhm=float(Rin), sag_ratio=float(sag),
                v_min_mV=float(v_min), v_ss_mV=float(v_ss), trace=(t, v))


def ap_features_from_trace(t, v, sp):
    """이미 얻은 (t,v,sp) 에서 첫 활동전위의 진폭·반치폭·역치. sp 없으면 None."""
    if sp is None or len(sp) == 0:
        return None
    vrest = v[t < 100.0].mean()
    t0 = sp[0]
    w = (t >= t0 - 3.0) & (t <= t0 + 5.0)
    tv, vv = t[w], v[w]
    if len(vv) < 3:
        return None
    vpeak = vv.max()
    amp = vpeak - vrest
    half = (vrest + vpeak) / 2.0
    above = vv >= half
    if above.any():
        idx = np.where(above)[0]
        width = float(tv[idx[-1]] - tv[idx[0]])
    else:
        width = float("nan")
    dvdt = np.gradient(vv, tv)
    thi = np.where(dvdt > 20.0)[0]
    vthr = float(vv[thi[0]]) if len(thi) else float("nan")
    return dict(ap_amplitude_mV=float(amp), ap_halfwidth_ms=width,
                ap_threshold_mV=vthr, vpeak_mV=float(vpeak))


def adaptation_from_spikes(sp):
    """스파이크 시각열에서 발화 적응 지수. 계단 구간 3발 미만이면 None."""
    during = sp[(sp >= 100.0) & (sp <= 600.0)] if sp is not None else np.array([])
    if len(during) < 3:
        return None
    isi = np.diff(during)
    return float(isi[-1] / isi[0] - 1.0)


def ap_features(cell, amp_nA):
    """계단전류에서 첫 활동전위의 진폭·반치폭·역치."""
    t, v, sp = step_response(cell, amp_nA)
    if len(sp) == 0:
        return None
    vrest = v[t < 100.0].mean()
    # 첫 스파이크 부근 창
    t0 = sp[0]
    w = (t >= t0 - 3.0) & (t <= t0 + 5.0)
    tv, vv = t[w], v[w]
    vpeak = vv.max()
    amp = vpeak - vrest
    # 반치폭: (vrest+vpeak)/2 교차 두 지점
    half = (vrest + vpeak) / 2.0
    above = vv >= half
    if above.any():
        idx = np.where(above)[0]
        width = float(tv[idx[-1]] - tv[idx[0]])
    else:
        width = float("nan")
    # 역치: dV/dt 가 20 mV/ms 를 처음 넘는 지점의 V
    dvdt = np.gradient(vv, tv)
    thi = np.where(dvdt > 20.0)[0]
    vthr = float(vv[thi[0]]) if len(thi) else float("nan")
    return dict(ap_amplitude_mV=float(amp), ap_halfwidth_ms=width,
                ap_threshold_mV=vthr, vpeak_mV=float(vpeak))


def adaptation_index(cell, amp_nA):
    """계단전류에서 발화 간격 적응 지수(마지막 ISI/첫 ISI − 1, 양수면 적응)."""
    t, v, sp = step_response(cell, amp_nA)
    during = sp[(sp >= 100.0) & (sp <= 600.0)]
    if len(during) < 3:
        return None
    isi = np.diff(during)
    return float(isi[-1] / isi[0] - 1.0)
