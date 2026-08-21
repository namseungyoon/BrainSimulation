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


# 단일세포 실험 기본 dt. EMS 시냅스가 없어 dt=0.1 로 충분(검증: dt0.025와 파형 동일, 4배 빠름).
CELL_DT = 0.1


def step_response(cell, amp_nA, delay=100.0, dur=500.0, tail=150.0, rec_dt=0.1, dt=CELL_DT):
    """소마에 계단전류 1회. (t, v, spike_times) 반환."""
    ic = h.IClamp(cell.soma[0](0.5))
    ic.delay, ic.dur, ic.amp = delay, dur, amp_nA
    t = h.Vector().record(h._ref_t)
    v = h.Vector().record(cell.soma[0](0.5)._ref_v)
    nrnenv.finit(v_init=-70.0, dt=dt)
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


def zap_response(cell, f0=0.5, f1=20.0, amp_nA=0.05, dur_ms=20000.0,
                 hold_nA=0.0, rec_dt=0.5, block_ih=False, v_init=-70.0,
                 seg=None, dt=CELL_DT):
    """ZAP(chirp) 전류 — 주파수가 f0→f1 로 선형 상승하는 정현파를 주입.

    공명 측정 표준 프로토콜. 반환: (t, v, i_inj) — 소마(또는 seg) 막전위 응답.
    block_ih=True 면 모든 구획의 Ih(ghdbar_hd)를 0 으로 (기전 귀속 대조).
    dur_ms 길게(20s) 주어 저주파(theta) 분해능 확보.
    """
    import numpy as _np
    site = seg if seg is not None else cell.soma[0](0.5)

    if block_ih:
        for s in cell.all:
            if h.ismembrane("hd", sec=s):
                for sg in s:
                    sg.hd.ghdbar = 0.0

    # ZAP 파형을 Vector.play 로 IClamp.amp 에 흘려넣는다 (연속 전류).
    # 파형은 시뮬 dt 격자로 만들고, play 간격도 같은 dt. (dt=0.1: 파형 동일·4배 빠름)
    n = int(dur_ms / dt) + 1
    t = _np.arange(n) * dt
    # 순간위상: 주파수가 선형 상승 → phase = 2π (f0 t + (f1-f0)/(2T) t^2)
    T = dur_ms / 1000.0
    tt = t / 1000.0                          # s
    inst = f0 * tt + (f1 - f0) / (2 * T) * tt ** 2
    wave = amp_nA * _np.sin(2 * _np.pi * inst) + hold_nA

    ic = h.IClamp(site)
    ic.delay = 0.0; ic.dur = dur_ms
    iv = h.Vector(wave)
    iv.play(ic._ref_amp, dt)

    vrec = h.Vector().record(site._ref_v, rec_dt)
    trec = h.Vector().record(h._ref_t, rec_dt)
    nrnenv.finit(v_init=v_init, dt=dt)
    h.continuerun(dur_ms)
    # play 벡터를 확실히 떼어 다음 런과 간섭 방지
    iv.play_remove()
    return _np.array(trec), _np.array(vrec), (t, wave, f0, f1, T)


def impedance_profile(t_ms, v, iwave, nbin=60):
    """ZAP 응답에서 주파수별 임피던스 |Z(f)| = |V(f)/I(f)| 프로파일.

    입력전류 파형(iwave)과 막전위 응답을 같은 시간격자로 맞춰 FFT. 순시주파수가 시간에
    선형이므로 시간축을 곧 주파수축으로 읽어, 주파수 구간별 (전압진폭/전류진폭) 을 낸다.
    반환: (freqs, Zmag, f_res, Q)
    """
    import numpy as _np
    tt, wave, f0, f1, T = iwave
    # 막전위를 전류 격자에 보간
    v_on_i = _np.interp(tt, t_ms, v)
    v_ac = v_on_i - v_on_i.mean()
    i_ac = wave - wave.mean()
    inst_f = f0 + (f1 - f0) * (tt / 1000.0) / T   # 시간→순시주파수
    edges = _np.linspace(f0, f1, nbin + 1)
    fc = 0.5 * (edges[:-1] + edges[1:])
    Z = _np.full(nbin, _np.nan)
    for k in range(nbin):
        m = (inst_f >= edges[k]) & (inst_f < edges[k + 1])
        if m.sum() < 5:
            continue
        vamp = _np.sqrt(2) * _np.std(v_ac[m])    # RMS→진폭
        iamp = _np.sqrt(2) * _np.std(i_ac[m])
        if iamp > 0:
            Z[k] = vamp / iamp                   # mV/nA = MOhm
    good = ~_np.isnan(Z)
    fc_g, Z_g = fc[good], Z[good]
    if len(Z_g) == 0:
        return fc, Z, _np.nan, _np.nan
    ipk = int(_np.argmax(Z_g))
    f_res = float(fc_g[ipk])
    # Q = 공명 강도 = 피크 / 저주파(첫 구간) 값
    Q = float(Z_g[ipk] / Z_g[0]) if Z_g[0] > 0 else _np.nan
    return fc, Z, f_res, Q


def adaptation_index(cell, amp_nA):
    """계단전류에서 발화 간격 적응 지수(마지막 ISI/첫 ISI − 1, 양수면 적응)."""
    t, v, sp = step_response(cell, amp_nA)
    during = sp[(sp >= 100.0) & (sp <= 600.0)]
    if len(during) < 3:
        return None
    isi = np.diff(during)
    return float(isi[-1] / isi[0] - 1.0)
