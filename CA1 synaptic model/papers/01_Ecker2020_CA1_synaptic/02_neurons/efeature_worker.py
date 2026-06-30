"""
efeature_worker.py — eFEL 단일세포 e-특징 추출 워커 (라이브러리, 번호 X)
============================================================================
세포 1개를 **격리 프로세스**로 로드 → 전류계단 스윕 → eFEL 로 모든 e-특징 추출.
(NEURON 은 한 프로세스에서 템플릿을 1개만 정의 가능 → 세포마다 별도 subprocess)

`6_efeature_distributions.py` 등이 호출:
    python efeature_worker.py --cell <model_dir>      → 표준출력에 "EFEAT_JSON {...}"

추출 특징: Rin(입력저항)·Vrest·sag비율·AP진폭/반치폭/역치·fAHP깊이·
           적응지수(adaptation_index)·rheobase·f-I 곡선·최대발화율.
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)

# 자극 프로토콜
HYPER = -0.10                                   # Rin·sag 용 과분극 계단(nA)
# f-I 용 양수 계단(nA): PC 는 rheobase~0.4·트레인 0.8, 인터뉴런은 ~0.05↑ → 넓게
SPIKE_AMPS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
DELAY, DUR, TSTOP = 300.0, 500.0, 800.0         # 자극(적응 관찰 충분), ms


def _mean(a):
    import numpy as np
    return float(np.mean(a)) if a is not None and len(a) else None


def extract(model_dir):
    import numpy as np
    import efel
    from common.nrn_env import h
    from common.cell_loader import load_cell
    try:
        efel.set_setting("Threshold", -20.0)
        efel.set_setting("interp_step", 0.025)   # 반치폭 등 미세 보간
    except Exception:
        pass

    h.load_file("stdrun.hoc")
    cell, tname = load_cell(model_dir)
    soma = cell.soma[0]
    ic = h.IClamp(soma(0.5)); ic.delay = DELAY; ic.dur = DUR
    vv = h.Vector().record(soma(0.5)._ref_v)
    tv = h.Vector().record(h._ref_t)
    h.celsius = 34.0
    h.cvode_active(1)

    def sim(amp):
        ic.amp = amp
        h.finitialize(-70.0)
        h.continuerun(TSTOP)
        return np.array(tv), np.array(vv)

    def feats(t, v, names):
        tr = {"T": t, "V": v, "stim_start": [DELAY], "stim_end": [DELAY + DUR]}
        try:
            return efel.get_feature_values([tr], names, raise_warnings=False)[0] or {}
        except Exception:
            return {}

    # ── 과분극 계단: Rin · Vrest · sag ──────────────────────────────
    t, v = sim(HYPER)
    fh = feats(t, v, ["voltage_base", "steady_state_voltage_stimend", "minimum_voltage"])
    vbase = _mean(fh.get("voltage_base"))
    vsteady = _mean(fh.get("steady_state_voltage_stimend"))
    vmin = _mean(fh.get("minimum_voltage"))
    # Rin: 정상상태 ΔV / |I| (MΩ = mV/nA)
    rin = None
    if vbase is not None and vsteady is not None:
        rin = abs(vbase - vsteady) / abs(HYPER)
    # sag 비율(표준 정의): 최저점 대비 정상상태로 되돌아온 정도 = (Vsteady-Vmin)/(Vbase-Vmin)
    sag_ratio = None
    if None not in (vbase, vsteady, vmin) and abs(vbase - vmin) > 1e-6:
        sag_ratio = (vsteady - vmin) / (vbase - vmin)
    vrest = vbase

    # ── 양수 계단 스윕: f-I ─────────────────────────────────────────
    fI = []
    traces = {}
    for amp in SPIKE_AMPS:
        t, v = sim(amp)
        sc = feats(t, v, ["Spikecount"]).get("Spikecount")
        n = int(sc[0]) if sc is not None and len(sc) else 0
        fI.append((amp, n))
        traces[amp] = (t, v, n)

    rheobase = next((a for a, n in fI if n >= 1), None)

    # ── AP 파형: rheobase 근처(스파이크 있는 첫 계단) ───────────────
    ap = {}
    ap_amp = next((a for a, n in fI if n >= 1), None)
    if ap_amp is not None:
        t, v, _ = traces[ap_amp]
        f = feats(t, v, ["AP_amplitude", "AP_duration_half_width", "AP_width",
                         "AP_begin_voltage", "AHP_depth_abs"])
        ap_amplitude = _mean(f.get("AP_amplitude"))
        hw = _mean(f.get("AP_duration_half_width"))         # 반치폭(반진폭에서의 폭) 우선
        ap_width = hw if hw is not None else _mean(f.get("AP_width"))
        ap_thr = _mean(f.get("AP_begin_voltage"))
        ahp_abs = _mean(f.get("AHP_depth_abs"))
        fahp = (ap_thr - ahp_abs) if (ap_thr is not None and ahp_abs is not None) else None
        ap = dict(AP_amplitude_mV=ap_amplitude, AP_width_ms=ap_width,
                  AP_threshold_mV=ap_thr, fAHP_depth_mV=fahp)

    # ── 적응지수: 중간 강도 트레인(첫 n>=6, 없으면 최다 >=4) ────────
    adapt = None
    adapt_amp = next((a for a, n in fI if n >= 6), None)
    if adapt_amp is None:
        best = max(((a, n) for a, n in fI if n >= 4), key=lambda x: x[1], default=None)
        adapt_amp = best[0] if best else None
    if adapt_amp is not None:
        t, v, _ = traces[adapt_amp]
        adapt = _mean(feats(t, v, ["adaptation_index2"]).get("adaptation_index2"))

    # ── 최대발화율: 최대 전류 계단 ─────────────────────────────────
    amax = SPIKE_AMPS[-1]
    t, v, n_at_max = traces[amax]
    fmax = _mean(feats(t, v, ["mean_frequency"]).get("mean_frequency"))
    if fmax is None:
        fmax = n_at_max / (DUR / 1000.0) if n_at_max else 0.0

    out = dict(template=tname, Rin_MOhm=rin, Vrest_mV=vrest, sag_ratio=sag_ratio,
               adaptation_index=adapt, rheobase_nA=rheobase, f_at_max_Hz=fmax,
               fI=fI, **ap)
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--cell":
        data = extract(sys.argv[2])
        print("EFEAT_JSON " + json.dumps(data))
    else:
        print("사용법: python efeature_worker.py --cell <model_dir>")


if __name__ == "__main__":
    main()
