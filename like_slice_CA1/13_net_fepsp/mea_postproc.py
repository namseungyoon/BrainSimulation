# -*- coding: utf-8 -*-
"""mea_postproc.py — MEA fEPSP 결과 **사후 분석** (시뮬레이션과 완전 분리)

왜 분리하는가
-------------
전규모 LTP 런 1회가 약 99시간이다. 분석 코드가 시뮬레이터 안에 있으면 "기울기 재는 창을
30ms에서 25ms로 바꿔볼까" 같은 사소한 재분석에도 99시간을 다시 써야 한다.
시뮬레이터는 **원자료만 저장**하고(0-2), 분석은 저장된 .npz만 읽어 몇 번이든 다시 한다.

무엇을 내놓는가 (CSV 5종 · 엑셀에서 바로 열림)
---------------------------------------------
  slopes.csv       전극 24개 × 자극 펄스 전부 — 기울기·진폭·피크시각·자극전극 거리
  ltp_index.csv    전극별 직후 LTP% · 기저선 흔들림 · 신호대잡음 · **실험군−대조군 차분**
  rho_dist.csv     시냅스 효능 ρ 히스토그램 + 0.5 넘은 비율
  persistence.csv  1/10/60분 뒤 예측 ρ (+ 재측정 파일을 주면 실제 60분 fEPSP%)
  spikes.csv       세포 유형별 · 구간별(기저선/TBS/사후) 스파이크 수

사용법
------
  python mea_postproc.py --npz figures/_mea_ltp_plastic.npz --verify
  python mea_postproc.py --npz A.npz --ctrl B.npz --out figures/csv_A
  python mea_postproc.py --npz A.npz --rho_export figures/_rho_60min.npz --relax_min 60
        → 나온 파일을 mea_experiment.py `--rho_init` 로 넣으면 60분 뒤 상태로 재측정(4단계)

⚠ 이 파일은 NEURON을 import하지 않는다(순수 분석). mea_experiment.py가 여기서
   measure_fepsp를 가져다 쓰므로, 시뮬과 분석의 기울기 계산은 **항상 같은 코드**다.
"""
import os
import sys
import csv
import argparse
import numpy as np

# ── Graupner & Brunel (2012) 효능 동역학 상수 — shared/mechanisms/GBPlasticitySyn.mod 와 동일 ──
#    자극이 끝나면 칼슘 입력이 0이 되고 ρ는 아래 자율 방정식만 따른다:
#        τ dρ/dt = −ρ(1−ρ)(ρ* − ρ)
#    ρ > ρ*(=0.5) → 1로 올라가 굳고 · ρ < 0.5 → 0으로 내려가 사라진다.
GB_TAU_MS = 688355.0     # ≈ 11.5분
GB_RHO_STAR = 0.5
GB_B = 5.28145           # w1/w0 — 강화된 시냅스가 몇 배가 되는가
GB_W0 = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# fEPSP 파형 계량
# ══════════════════════════════════════════════════════════════════════════════
# ★2026-08-06 기울기 정의 교체 — 왜 바꾸는가
# ------------------------------------------------------------------------------
# 옛 방식("legacy")은 "피크의 20~80% 구간에 **들어온 표본**으로 선형회귀"였다.
# 우리 기록 간격은 0.4 ms인데 fEPSP 상승은 ~2 ms라, 그 구간에 표본이 1~2개밖에
# 안 들어온다. 스모크(500세포·5레벨)에서 실제로 이렇게 무너졌다:
#     세기  5% 15% 30% → 표본 2개씩 (우연히 동작)
#     세기 50%          → 표본 **1개** → len(idx)>=2 가 거짓 → 전혀 다른 식(전체 평균)으로
#                         **조용히** 떨어져 |기울기| 0.821 → 0.204 로 급락
#     세기 80%          → 집단스파이크의 되튐까지 회귀에 끌려들어와 0.030 으로 붕괴
# 진폭은 단조 증가(-0.12 → -1.35 µV)인데 기울기만 뒤집혔다 = 물리가 아니라 계량의 고장.
#
# 새 방식("cross")은 문헌 표준인 **교차 시각 기반 초기 기울기**다.
#     기울기 = 0.6 × 진폭 / (t80 − t20),  t20·t80 = 20%·80% 선을 **처음 지나는** 시각(선형보간)
# 표본 개수에 의존하지 않고, 피크 **이전**만 보므로 집단스파이크의 되튐이 섞이지 않는다.
# 같은 0.4 ms 자료로 다시 재면 단조가 회복된다: 0.105 → 0.238 → 0.664 → 1.033 → 1.456
#
# legacy 는 지워두지 않고 남긴다 — 옛 결과 파일의 저장값을 비트 단위로 재현해
# "계산을 망가뜨리지 않았다"를 증명해야 하기 때문이다(0단계 통과기준 #1).
SLOPE_METHOD = "cross"       # 기본. "legacy" 로 바꾸면 옛 표본회귀 방식

# ★흐름 꼬리 판별 문턱 — 피크가 창의 뒤쪽 (1-EDGE_FRAC) 안에 들어오면 실격 표시.
# 근거: 전규모 레벨2(섬유 30) 24전극 실측에서 전극이 두 무리로 완전히 갈렸다.
#   진짜 fEPSP 17개 → 피크 2.4~6.0 ms   /   흐름 꼬리 6개(#0,1,2,3,8,9) → 18.8~29.6 ms
# 30 ms 창에서 0.6*30 = 18 ms. 두 무리 사이(6.0 ~ 18.8 ms)를 가르므로 여유가 크다.
EDGE_FRAC = 0.6

# ★집단스파이크 오염 판별 문턱 — 하강 중 **되돌림**(위로 되올라간 양)의 최대 비율.
# 왜 필요한가: 세기가 올라가면 fEPSP 위에 집단스파이크가 겹친다. 그러면 파형이
#   '내려가다 잠깐 되올랐다가 훨씬 깊게 꽂히는' 두 성분 모양이 되고, 20~80% 교차가
#   전부 집단스파이크의 상승면 안에 몰려 기울기가 폭증한다. 실측 예 — 전극#17 레벨3:
#   되돌림 38.7% · 20~80% 띠 표본 **1개** · 기울기 -8,274 µV/ms(같은 파형 진폭은
#   #18보다 작은데 기울기는 2.6배). 이건 시냅스 세기의 대리 지표가 아니다.
# 문턱 0.20 의 근거: 전규모 레벨2·3 의 24전극 x 2레벨 = 48개 실측에서
#   깨끗한 파형 최대 10.0% / 오염된 파형 최소 27.7% — 사이가 비어 있다.
POP_REV_FRAC = 0.20


def _first_cross(tt, vv, level, ipk):
    """0에서 출발해 피크(ipk)까지 내려오는 동안 `level`(음수)을 **처음 지나는 시각**.

    표본 사이는 선형보간한다 — 이것이 표본 개수 의존을 없애는 핵심이다.
    """
    seg = vv[:ipk + 1]
    idx = np.where(seg <= level)[0]
    if idx.size == 0:
        return float(tt[ipk])
    k = int(idx[0])
    if k == 0:
        return float(tt[0])
    v0, v1 = float(seg[k - 1]), float(seg[k])
    if v1 == v0:
        return float(tt[k])
    f = (level - v0) / (v1 - v0)
    return float(tt[k - 1] + f * (tt[k] - tt[k - 1]))


def measure_fepsp(t, v, t0, dur=30.0, pre=5.0, method=None, base=None):
    """자극 t0 후 dur창: 음성 fEPSP 진폭 + 초기 기울기(µV/ms).

    기준선 = 자극 전 pre(ms) 구간 평균(단일 샘플보다 안정).
    base 를 주면 그 값을 기준선으로 강제한다 — 실측 자료처럼 **이미 기준선 보정이
    끝났고**(base=0.0) 자극 아티팩트 때문에 창 앞쪽을 기준선으로 쓸 수 없을 때 필요하다.
    method: "cross"(기본·교차시각) | "legacy"(옛 표본회귀). None이면 SLOPE_METHOD.

    돌려주는 것 — slope 는 선택된 방식의 값이고, 나머지는 **진단용**으로 항상 함께 낸다:
        amp    피크 진폭(µV, 음수)          tpk   피크 시각(ms)
        slope  기울기(µV/ms, 음수)          t20/t80  20%·80% 교차 시각
        slope_cross / slope_legacy / slope_maxd   세 방식의 값(비교용)
        n_band 20~80% 띠 안에 실제로 들어온 표본 수 (2 미만이면 legacy가 위험)
        fb_cross / fb_legacy  그 방식이 성립하지 않아 **대체식으로 넘어갔는지**.
            True 면 그 값은 정의대로 잰 기울기가 아니다 — 호출부에서 걸러내야 한다.
        base   실제로 쓴 기준선(µV) — 창 앞 표본이 없어 대체값으로 넘어갔는지 추적용
        pre_n  기준선 평균에 쓴 자극 전 표본 수. 0이면 창 첫 표본을 기준선으로 썼다는 뜻
        edge_peak  ★피크가 창 **끝자락**(뒤 20%)에 붙었는가.
            True 면 이것은 fEPSP가 아니라 **느린 흐름의 꼬리**일 가능성이 크다.
            진짜 SR층 fEPSP는 자극 후 2~6 ms에 피크가 온다. 창을 더 길게 잡으면
            그만큼 더 내려가므로 "진폭"이 창 길이에 따라 커지는 가짜 값이 된다.
            EDGE_FRAC 로 기준을 조절한다.
        rev_frac / pop_spike  ★하강 중 되돌림 비율과 그 실격 판정.
            fEPSP 위에 **집단스파이크**가 겹치면 파형이 두 성분으로 꺾인다.
            pop_spike=True 면 slope 가 시냅스 세기가 아니라 집단발화의 상승면을
            잰 값일 수 있다 — 기울기를 LTP 지표로 쓰면 안 된다. POP_REV_FRAC 참조.
    """
    meth = method or SLOPE_METHOD
    m = (t >= t0) & (t < t0 + dur)
    pm = (t >= t0 - pre) & (t < t0)
    pre_n = int(pm.sum())
    if base is None:
        base = float(v[pm].mean()) if pre_n else (float(v[m][0]) if m.sum() else 0.0)
    base = float(base)
    tt = t[m]; vv = v[m] - base
    nil = dict(amp=0.0, slope=0.0, tpk=float(t0), t20=float(t0), t80=float(t0),
               slope_cross=0.0, slope_legacy=0.0, slope_maxd=0.0, n_band=0, dt_legacy=0.0,
               fb_cross=True, fb_legacy=True, base=base, pre_n=pre_n, edge_peak=False,
               rev_frac=0.0, pop_spike=False)
    if len(tt) < 5:
        return nil
    ipk = int(np.argmin(vv)); amp = float(vv[ipk]); tpk = float(tt[ipk])
    edge_peak = bool((tpk - t0) >= EDGE_FRAC * dur)
    if ipk < 2 or amp >= 0:
        out = dict(nil); out.update(amp=amp, tpk=tpk, t20=tpk, t80=tpk, edge_peak=edge_peak)
        return out

    # ★집단스파이크 오염도 — 피크까지 내려가는 동안 되올라간 최대량.
    #   비교 기준은 **그 시점까지의 최저값**이다("이만큼 내려왔는데 이만큼 되올랐다").
    #   아직 거의 안 내려온 초반은 |최저| < 0.1·|진폭| 로 걸러 잡음 확대를 막는다.
    _seg = vv[:ipk + 1]
    _run = np.minimum.accumulate(_seg)
    _ok = np.abs(_run) >= 0.1 * abs(amp)
    _rev = (_seg - _run)[_ok]
    rev_frac = float((_rev / np.abs(_run[_ok])).max()) if _rev.size else 0.0
    pop_spike = bool(rev_frac > POP_REV_FRAC)

    lo, hi = 0.2 * amp, 0.8 * amp          # amp<0 이므로 lo가 0에 가깝고 hi가 더 깊다

    # ① 교차 시각 방식(기본)
    t20 = _first_cross(tt, vv, lo, ipk)
    t80 = _first_cross(tt, vv, hi, ipk)
    dtb = t80 - t20
    # ★t20==t80 이면(=두 교차가 같은 표본에 몰림) 교차 방식이 성립하지 않는다.
    #   조용히 다른 식으로 갈아타면 "왜 이 값이 나왔는지" 추적할 수 없으므로
    #   대체값을 쓰되 **flag를 남긴다**. 호출부는 fb_cross 로 걸러낼 수 있다.
    fb_cross = not (dtb > 1e-9)
    s_cross = (amp / (tpk - float(tt[0]) + 1e-9)) if fb_cross else (0.6 * amp / dtb)

    # ② 옛 표본회귀 방식(재현 확인 전용)
    band = (vv[:ipk + 1] <= lo) & (vv[:ipk + 1] >= hi)
    idx = np.where(band)[0]
    if len(idx) >= 2:
        a, b = int(idx[0]), int(idx[-1])
        s_leg = float(np.polyfit(tt[a:b + 1], vv[a:b + 1], 1)[0])
        dt_leg = float(tt[b] - tt[a])          # legacy가 실제로 쓴 시간 폭(허용오차 유도용)
    else:
        s_leg = float((vv[ipk] - vv[0]) / (tpk - tt[0] + 1e-9))
        dt_leg = float(tpk - tt[0])

    # ③ 인접 표본 최대 하강 속도(진단용 — 기록 간격이 거친지 보는 잣대)
    dv = np.diff(vv[:ipk + 1]); dt_ = np.diff(tt[:ipk + 1])
    s_max = float(np.min(dv / np.where(dt_ == 0, 1e-9, dt_))) if dv.size else 0.0

    slope = {"cross": s_cross, "legacy": s_leg, "maxd": s_max}[meth]
    return dict(amp=amp, slope=float(slope), tpk=tpk, t20=float(t20), t80=float(t80),
                slope_cross=float(s_cross), slope_legacy=float(s_leg),
                slope_maxd=float(s_max), n_band=int(len(idx)), dt_legacy=float(dt_leg),
                fb_cross=bool(fb_cross), fb_legacy=bool(len(idx) < 2),
                base=base, pre_n=pre_n, edge_peak=edge_peak,
                rev_frac=rev_frac, pop_spike=pop_spike)


# ══════════════════════════════════════════════════════════════════════════════
# 효능 ρ — 자율 이완(자극 없는 동안)
# ══════════════════════════════════════════════════════════════════════════════
def relax_rho(rho, minutes, dt_ms=1000.0):
    """자극이 끝난 뒤 `minutes`분이 지났을 때의 ρ. RK4 고정스텝.

    ★이 적분은 **모델 안에서는 근사가 아니라 정확**하다(칼슘 입력 0을 가정한 해석).
      다만 실제 실험은 60분 동안 30초마다 테스트 펄스를 넣어 칼슘을 조금씩 주입한다.
      단일 펄스의 칼슘 피크가 약화 문턱 θ_d=1.0 부근이므로, **현실은 예측보다 조금 더
      내려갈 수 있다** — 한계로 명기한다.
    """
    def f(r):
        return -r * (1.0 - r) * (GB_RHO_STAR - r) / GB_TAU_MS
    r = np.asarray(rho, float).copy()
    n = max(int(round(minutes * 60000.0 / dt_ms)), 0)
    for _ in range(n):
        k1 = f(r); k2 = f(r + 0.5 * dt_ms * k1); k3 = f(r + 0.5 * dt_ms * k2); k4 = f(r + dt_ms * k3)
        r = r + (dt_ms / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.clip(r, 0.0, 1.0)


def rho_to_weight(rho):
    """전달 세기 w = w0 + ρ(b·w0 − w0). ρ=0 → w0 · ρ=1 → b·w0."""
    return GB_W0 + np.asarray(rho, float) * (GB_B * GB_W0 - GB_W0)


# ══════════════════════════════════════════════════════════════════════════════
# 보조
# ══════════════════════════════════════════════════════════════════════════════
def g(d, key, default=None):
    """npz에서 값 하나 꺼내기(없으면 default). 0차원 배열은 파이썬 값으로 푼다."""
    if key not in d.files:
        return default
    v = d[key]
    return v.item() if getattr(v, "ndim", 1) == 0 else v


def flag(d, key):
    """참/거짓 설정값을 **3상태**로 읽는다 — 예 / 아니오 / 기록없음.

    ★ bool(g(d,key,False)) 로 찍으면 '기록이 없는 옛 파일'과 '실제로 꺼져 있던 런'이
      똑같이 '아니오'로 보인다. 0-2 이전 파일에는 freeze_rho 가 아예 없으므로 구분해야 한다.
    """
    if key not in d.files:
        return "기록없음"
    return "예" if bool(g(d, key, False)) else "아니오"


def write_csv(path, header, rows):
    # utf-8-sig = BOM 포함. 엑셀이 한글 헤더를 깨뜨리지 않고 바로 연다.
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  saved: {path}  ({len(rows)}행)")


def cfg_str(d):
    """결과 파일 하나를 '어떤 조건이었나' 한 줄로. 규모·방출모드는 항상 두 줄로 따로 찍는다."""
    return dict(
        file=os.path.basename(str(g(d, "_path", ""))),
        model=str(g(d, "syn_model", "gb(추정)")),
        tag=str(g(d, "tag", "")),
        N=int(g(d, "N", 0)),
        n_sc=int(g(d, "n_sc", 0)),
        n_syn=int(g(d, "n_syn", 0)),
        tbs=int(g(d, "tbs_bursts", -1)),
        frozen=flag(d, "freeze_rho"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 전극 × 펄스 전수 분석
# ══════════════════════════════════════════════════════════════════════════════
def analyse_all_electrodes(d, dur=30.0, pre=5.0):
    """저장된 전극 **전부**(24개) × 자극 펄스 **전부**를 계량한다.

    지금까지는 24개를 계산·저장해놓고 1개만 봤다. 추가 시뮬 비용 0으로 23개가 더 나온다.
    """
    t = np.asarray(d["t"], float); Ve = np.asarray(d["Ve"], float)
    t_base = np.atleast_1d(np.asarray(g(d, "t_base", []), float))
    t_post = np.atleast_1d(np.asarray(g(d, "t_post", []), float))
    E = np.asarray(g(d, "E", np.zeros((Ve.shape[0], 2))), float)
    el_layer = np.asarray(g(d, "el_layer", np.array([""] * Ve.shape[0])), str)
    over = np.asarray(g(d, "over", np.ones(Ve.shape[0], bool)))
    stim_elec = int(g(d, "stim_elec", 0))

    rows = []
    per_elec = {}
    for j in range(Ve.shape[0]):
        dist = float(np.linalg.norm(E[j] - E[stim_elec])) if E.size else np.nan
        bs, ps = [], []
        for phase, times, acc in (("base", t_base, bs), ("post", t_post, ps)):
            for i, tt in enumerate(times):
                fe = measure_fepsp(t, Ve[j], float(tt), dur, pre)
                acc.append(fe["slope"])
                rows.append([j, str(el_layer[j]) if j < len(el_layer) else "",
                             int(bool(over[j])) if j < len(over) else 1,
                             round(dist, 1), int(j == stim_elec),
                             phase, i + 1, float(tt),
                             fe["slope"], fe["amp"], fe["tpk"],
                             fe["slope_cross"], fe["slope_legacy"], fe["n_band"]])
        b = np.abs(np.array(bs)); p = np.abs(np.array(ps))
        bm = float(b.mean()) if b.size else 0.0
        pm = float(p.mean()) if p.size else 0.0
        per_elec[j] = dict(
            layer=str(el_layer[j]) if j < len(el_layer) else "",
            over=int(bool(over[j])) if j < len(over) else 1,
            dist=dist, is_stim=int(j == stim_elec),
            base_mean=bm, base_sd=float(b.std(ddof=1)) if b.size > 1 else 0.0,
            base_cv=float(b.std(ddof=1) / bm * 100.0) if (b.size > 1 and bm > 1e-12) else 0.0,
            post_mean=pm,
            ltp_pct=(100.0 * (pm / bm - 1.0)) if bm > 1e-12 else float("nan"),
            snr=(bm / (b.std(ddof=1) + 1e-12)) if b.size > 1 else float("inf"),
        )
    return rows, per_elec


def phase_of(times, t_base, t_tbs, t_post, dur=30.0):
    """스파이크 시각 배열을 기저선/TBS/사후/기타 로 분류."""
    lab = np.full(len(times), "other", dtype=object)
    for name, ts in (("base", t_base), ("tbs", t_tbs), ("post", t_post)):
        if len(ts) == 0:
            continue
        for t0 in np.atleast_1d(ts):
            m = (times >= t0) & (times < t0 + dur)
            lab[m] = name
    return lab


# ══════════════════════════════════════════════════════════════════════════════
def main():
    global SLOPE_METHOD
    ap = argparse.ArgumentParser(description="MEA fEPSP 결과 사후 분석 (CSV 5종)")
    ap.add_argument("--npz", required=True, help="실험군 결과 파일")
    ap.add_argument("--ctrl", default="", help="대조군 결과 파일(--freeze_rho 런) — 차분 계산용")
    ap.add_argument("--remeas", default="", help="60분 재측정 결과 파일 — persistence.csv에 실측 % 채움")
    ap.add_argument("--out", default="", help="CSV 출력 폴더(기본: npz 옆 <tag>_csv)")
    ap.add_argument("--dur", type=float, default=30.0, help="측정창 길이(ms)")
    ap.add_argument("--pre", type=float, default=5.0, help="기준선 구간(ms)")
    ap.add_argument("--slope_method", default=SLOPE_METHOD, choices=["cross", "legacy", "maxd"],
                    help="기울기 정의. cross=교차시각(기본·문헌표준) · legacy=옛 표본회귀 · maxd=최대하강")
    ap.add_argument("--verify", action="store_true", help="저장된 요약값을 재계산해 일치 확인")
    ap.add_argument("--rho_export", default="", help="이완시킨 ρ를 --rho_init 형식으로 저장할 경로")
    ap.add_argument("--relax_min", type=float, default=60.0, help="--rho_export 시 이완 시간(분)")
    a = ap.parse_args()
    SLOPE_METHOD = a.slope_method

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    d = np.load(a.npz, allow_pickle=True)
    kind = str(g(d, "kind", "?"))
    if kind != "ltp":
        print(f"[경고] kind='{kind}' — 이 스크립트는 ltp 결과를 대상으로 합니다")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.npz)),
                                f"{os.path.splitext(os.path.basename(a.npz))[0]}_csv")
    os.makedirs(out, exist_ok=True)

    # ── 조건 요약: 규모·방출모드는 **두 줄**로 (내부 커넥톰 / SC 경로) ──
    N = int(g(d, "N", 0)); n_sc = int(g(d, "n_sc", 0)); n_syn = int(g(d, "n_syn", 0))
    det = bool(g(d, "det", True)); model = str(g(d, "syn_model", "gb(파일에 없음·추정)"))
    print("=" * 78)
    print(f"[대상] {os.path.basename(a.npz)} · 태그 {g(d,'tag','?')}")
    print(f"[규모] 세포 {N:,} · 내부 연결 {n_syn:,} · SC 시냅스 {n_sc:,} · SC받은세포 {int(g(d,'n_sccell',0)):,}")
    print(f"[방출·내부연결] {'결정론' if det else '확률(BBP EMS)'} · mod Det/Prob AMPANMDA·GABAAB")
    print(f"[방출·SC경로] 결정론(선택지 없음) · 모델 '{model}' · ρ얼림 {flag(d,'freeze_rho')}"
          f"{' (γ_p=γ_d=0 · 엄격 대조군)' if flag(d,'freeze_rho') == '예' else ''}")
    print(f"[유도] TBS {g(d,'tbs_bursts','?')}버스트 · 기저선 {len(np.atleast_1d(g(d,'t_base',[])))}회 "
          f"· 사후 {len(np.atleast_1d(g(d,'t_post',[])))}회")
    _mdesc = {"cross": "교차시각 20~80% (문헌 표준·기본)",
              "legacy": "옛 표본회귀 20~80% (재현 확인 전용)",
              "maxd": "인접표본 최대하강 (진단용)"}[SLOPE_METHOD]
    print(f"[기울기 정의] {SLOPE_METHOD} — {_mdesc} · 측정창 {a.dur:.0f}ms · 기준선 {a.pre:.0f}ms")
    print("=" * 78)

    # ── ① slopes.csv ─────────────────────────────────────────────────────────
    rows, per_elec = analyse_all_electrodes(d, a.dur, a.pre)
    write_csv(os.path.join(out, "slopes.csv"),
              ["전극", "층", "조직위", "자극전극거리um", "자극전극여부",
               "구간", "펄스번호", "자극시각ms", "기울기uV_per_ms", "진폭uV", "피크시각ms",
               "기울기_교차시각", "기울기_옛표본회귀", "띠표본수"],
              rows)

    # ── ② ltp_index.csv (+ 실험군 − 대조군) ──────────────────────────────────
    ctrl_elec = None
    if a.ctrl:
        dc = np.load(a.ctrl, allow_pickle=True)
        _, ctrl_elec = analyse_all_electrodes(dc, a.dur, a.pre)
        print(f"[대조군] {os.path.basename(a.ctrl)} · ρ얼림 {flag(dc,'freeze_rho')}"
              f" · 모델 {g(dc,'syn_model','gb(파일에 없음·추정)')}")
    li = []
    for j, e in sorted(per_elec.items()):
        c = (ctrl_elec or {}).get(j)
        li.append([j, e["layer"], e["over"], round(e["dist"], 1), e["is_stim"],
                   round(e["base_mean"], 5), round(e["base_sd"], 5), round(e["base_cv"], 3),
                   round(e["post_mean"], 5), round(e["ltp_pct"], 3), round(e["snr"], 2),
                   round(c["ltp_pct"], 3) if c else "",
                   round(e["ltp_pct"] - c["ltp_pct"], 3) if c else ""])
    write_csv(os.path.join(out, "ltp_index.csv"),
              ["전극", "층", "조직위", "자극전극거리um", "자극전극여부",
               "기저선평균uV_per_ms", "기저선표준편차", "기저선흔들림pct", "사후평균uV_per_ms",
               "LTP_pct", "신호대잡음", "대조군LTP_pct", "차분_실험군빼기대조군_pct"], li)

    # ── ③ rho_dist.csv ───────────────────────────────────────────────────────
    rho_all = np.asarray(g(d, "rho_all", np.zeros(0)), float)
    rho0_all = np.asarray(g(d, "rho0_all", np.zeros(0)), float)
    if rho_all.size:
        edges = np.linspace(0.0, 1.0, 21)
        hist, _ = np.histogram(rho_all, bins=edges)
        rd = [[round(edges[i], 2), round(edges[i + 1], 2), int(hist[i]),
               round(100.0 * hist[i] / rho_all.size, 4)] for i in range(len(hist))]
        rd.append(["요약", "", int(rho_all.size), ""])
        rd.append(["평균", round(float(rho_all.mean()), 6), "", ""])
        rd.append(["ρ0평균", round(float(rho0_all.mean()), 6) if rho0_all.size else 0.0, "", ""])
        rd.append(["ρ>0.5개수", int((rho_all > 0.5).sum()), "",
                   round(100.0 * float((rho_all > 0.5).mean()), 4)])
        rd.append(["ρ>0.5=굳는LTP", "0개면 몇 분 안에 전부 0으로 돌아감", "", ""])
        write_csv(os.path.join(out, "rho_dist.csv"),
                  ["구간시작", "구간끝", "시냅스수", "비율pct"], rd)
    else:
        print("  [건너뜀] rho_dist.csv — 이 파일에는 시냅스별 ρ(rho_all)가 없습니다"
              " (0-2 이전에 만들어진 파일). 평균만 존재: "
              f"rho_mean={g(d,'rho_mean','?')}, rho_up={g(d,'rho_up','?')}/{g(d,'rho_n','?')}")

    # ── ④ persistence.csv ────────────────────────────────────────────────────
    pr = []
    if rho_all.size:
        w_now = float(rho_to_weight(rho_all).mean())
        for mnt in (0.0, 1.0, 10.0, 60.0):
            r = relax_rho(rho_all, mnt)
            w = rho_to_weight(r)
            pr.append([mnt, round(float(r.mean()), 6), int((r > 0.5).sum()),
                       round(100.0 * float((r > 0.5).mean()), 4),
                       round(float(w.mean()), 6), round(100.0 * (float(w.mean()) / w_now - 1.0), 3), ""])
        if a.remeas:
            dm = np.load(a.remeas, allow_pickle=True)
            _, me = analyse_all_electrodes(dm, a.dur, a.pre)
            rj = int(g(d, "rec_j", 0))
            pr[-1][-1] = round(me[rj]["ltp_pct"], 3) if rj in me else ""
            print(f"[재측정] {os.path.basename(a.remeas)} · 기록전극#{rj} 실측 {pr[-1][-1]}%")
        write_csv(os.path.join(out, "persistence.csv"),
                  ["경과분", "예측ρ평균", "ρ>0.5개수", "ρ>0.5비율pct",
                   "예측전달세기평균", "직후대비변화pct", "재측정_실측LTP_pct"], pr)
    else:
        print("  [건너뜀] persistence.csv — 시냅스별 ρ가 필요합니다")

    # ── ⑤ spikes.csv ─────────────────────────────────────────────────────────
    st = np.asarray(g(d, "spike_t", np.zeros(0)), float)
    sg = np.asarray(g(d, "spike_gid", np.zeros(0)), int)
    gtype = np.asarray(g(d, "gtype", np.zeros(0)), str)
    if st.size and gtype.size:
        lab = phase_of(st, np.atleast_1d(g(d, "t_base", [])), np.atleast_1d(g(d, "t_tbs", [])),
                       np.atleast_1d(g(d, "t_post", [])), a.dur)
        tp = np.array([gtype[i] if 0 <= i < len(gtype) else "?" for i in sg])
        sp = []
        for ct in sorted(set(tp.tolist())):
            for ph in ("base", "tbs", "post", "other"):
                m = (tp == ct) & (lab == ph)
                n = int(m.sum())
                sp.append([ct, ph, n, int(len(set(sg[m].tolist()))) if n else 0])
        sp.append(["전체", "합", int(st.size), int(len(set(sg.tolist())))])
        write_csv(os.path.join(out, "spikes.csv"),
                  ["세포유형", "구간", "스파이크수", "발화세포수"], sp)
    else:
        print(f"  [건너뜀] spikes.csv — 스파이크 시각이 없습니다(0-2 이전 파일). 총 개수만: "
              f"nspk={g(d,'nspk','?')}")

    # ── ρ 내보내기 (4단계 60분 재측정용) ─────────────────────────────────────
    if a.rho_export:
        if not rho_all.size:
            print("[실패] --rho_export 하려면 시냅스별 ρ(rho_all/rho_gid/rho_k)가 필요합니다")
        else:
            r = relax_rho(rho_all, a.relax_min)
            np.savez(a.rho_export, rho_all=r.astype(np.float32),
                     rho_gid=np.asarray(g(d, "rho_gid", np.zeros(0)), np.int32),
                     rho_k=np.asarray(g(d, "rho_k", np.zeros(0)), np.int32),
                     nhost=int(g(d, "nhost", 1)), relax_min=a.relax_min,
                     src=os.path.basename(a.npz))
            print(f"[ρ 내보내기] {a.rho_export} · {a.relax_min:.0f}분 이완 · "
                  f"평균 {float(r.mean()):.4f} · ρ>0.5 {int((r>0.5).sum()):,}개")

    # ── 재현 확인 ────────────────────────────────────────────────────────────
    if a.verify:
        print("-" * 78)
        print("[재현 확인] 저장된 요약값 vs 이 코드로 다시 계산한 값")
        rj = int(g(d, "rec_j", 0))
        Ve_raw = d["Ve"]
        t = np.asarray(d["t"], float); Ve = np.asarray(Ve_raw, float)
        # ★허용오차의 근거 (2026-08-06 재유도) — 파형 Ve가 float32로 저장돼 있으면 원래
        #   float64로 계산된 기울기를 **비트 단위로는 재현할 수 없다**. 문제는 "얼마나
        #   어긋나도 괜찮은가"인데, 예전 기준(상대오차 = float32 eps)은 틀렸다.
        #   기울기는 **두 표본의 차 ÷ 시간간격**이다. 표본 하나가 최대 q = eps·max|v|/2
        #   만큼 어긋나므로 기울기 오차는 최대 2q/Δt = eps·max|v|/Δt 다. 파형이 크고
        #   회귀 구간 Δt가 짧을수록 오차가 **증폭**되므로, 상대오차 eps로 재면 멀쩡한
        #   재현도 실패로 찍힌다(실제로 대조군에서 그렇게 찍혔다).
        #   → 그래서 **절대 오차 한계를 자료에서 직접 계산**해 쓴다.
        #   ※ 이 검사는 "저장 정밀도" 문제만 본다. "코드를 고치며 계산이 바뀌지 않았는가"의
        #      진짜 증명은 _slope_regress_check.py 의 옛 코드 대 새 코드 **비트 단위 전수 대조**다.
        f32 = (Ve_raw.dtype == np.float32)
        eps32 = float(np.finfo(np.float32).eps)
        vmax = float(np.abs(Ve[rj]).max())

        def _tol_abs(fe):
            """이 펄스의 기울기가 float32 저장 때문에 어긋날 수 있는 **절대** 한계(µV/ms)."""
            if not f32:
                return 1e-9 * max(abs(fe["slope_legacy"]), 1.0)
            return eps32 * vmax / max(fe["dt_legacy"], 1e-9)
        print(f"   저장 정밀도 {Ve_raw.dtype} · 파형 최대 {vmax:.3f}µV → "
              f"허용 오차 = eps·max|v|/Δt (펄스마다 계산)"
              + ("  (2026-08-06 이후 파일은 float64)" if f32 else ""))
        # ★저장된 slope_base/slope_post 가 **어느 정의로 계산됐는지**는 파일마다 다르다.
        #   - 기울기 정의를 바꾸기 전(2026-08-06 이전) 파일: legacy. `slope_method` 키가 없다.
        #   - 그 뒤 파일: 파일 안의 `slope_method` 가 알려준다(기본 cross).
        #   재현 확인은 **저장할 때 쓴 그 정의로** 해야 뜻이 있다 — 그래야 "계산을
        #   망가뜨리지 않았다"의 증명이 된다. 다른 정의의 값도 나란히 찍어, 정의를 바꿔서
        #   숫자가 얼마나 달라지는지 숨기지 않고 그대로 보고한다.
        saved_meth = str(g(d, "slope_method", "legacy"))
        skey = "slope_" + ("cross" if saved_meth == "cross" else
                           "maxd" if saved_meth == "maxd" else "legacy")
        okey = "slope_cross" if skey != "slope_cross" else "slope_legacy"
        print(f"   저장 당시 기울기 정의 = {saved_meth}"
              + ("  (파일에 표기 없음 → legacy 로 간주)" if "slope_method" not in d.files else ""))
        tb = np.atleast_1d(g(d, "t_base", [])); tps = np.atleast_1d(g(d, "t_post", []))
        sb_old = np.atleast_1d(np.asarray(g(d, "slope_base", []), float))
        sp_old = np.atleast_1d(np.asarray(g(d, "slope_post", []), float))
        fb = [measure_fepsp(t, Ve[rj], float(x), a.dur, a.pre) for x in tb]
        fp = [measure_fepsp(t, Ve[rj], float(x), a.dur, a.pre) for x in tps]
        sb_new = np.array([f[skey] for f in fb])
        sp_new = np.array([f[skey] for f in fp])
        sb_cr = np.array([f[okey] for f in fb])
        sp_cr = np.array([f[okey] for f in fp])
        ok = True
        tol_b, tol_p = [], []
        for nm, o, n, c, ff, acc in (("기저선", sb_old, sb_new, sb_cr, fb, tol_b),
                                     ("사후", sp_old, sp_new, sp_cr, fp, tol_p)):
            for i, (x, y) in enumerate(zip(o, n)):
                ta = _tol_abs(ff[i]); acc.append(ta)
                dev = abs(x - y)
                same = dev <= ta
                ok &= same
                print(f"   {nm}{i+1}: 저장 {x:+.8f}  {saved_meth}재계산 {y:+.8f}  "
                      f"차 {dev:.2e} (한계 {ta:.2e})  {'일치' if same else '★불일치'}"
                      f"   | {okey[6:]} {c[i]:+.8f} (띠표본 {ff[i]['n_band']}개"
                      + ("·★대체값" if ff[i]["fb_cross"] or ff[i]["fb_legacy"] else "") + ")")
        bm = float(np.abs(sb_new).mean()); pm = float(np.abs(sp_new).mean())
        pct_new = 100.0 * (pm / bm - 1.0) if bm > 1e-12 else float("nan")
        pct_old = float(g(d, "ltp_pct", float("nan")))
        # LTP%는 두 평균의 **비**다. 비가 1에 가까우면(대조군!) 상대오차가 폭발하므로
        #   상대가 아니라 **퍼센트포인트 절대 한계**로 잰다. 오차 전파:
        #   Δ(LTP%) ≈ 100·(사후/기저) · (Δ사후/사후 + Δ기저/기저)
        eb = float(np.mean(tol_b)) if tol_b else 0.0
        ep = float(np.mean(tol_p)) if tol_p else 0.0
        tol_pct = 100.0 * (pm / max(bm, 1e-30)) * (ep / max(pm, 1e-30) + eb / max(bm, 1e-30))
        dev = abs(pct_new - pct_old)
        same = dev <= tol_pct
        ok &= same
        print(f"   LTP%: 저장 {pct_old:+.10f}  {saved_meth}재계산 {pct_new:+.10f}  "
              f"차 {dev:.2e}%p (한계 {tol_pct:.2e}%p)  {'일치' if same else '★불일치'}")
        bmc = float(np.abs(sb_cr).mean()); pmc = float(np.abs(sp_cr).mean())
        pct_cr = 100.0 * (pmc / bmc - 1.0) if bmc > 1e-12 else float("nan")
        print(f"   → 보고용 소수 1자리: 저장 {pct_old:+.1f}%  {saved_meth} {pct_new:+.1f}%  "
              f"★{okey[6:]} {pct_cr:+.1f}%")
        print(f"[재현 확인] {'통과 — 계산을 망가뜨리지 않았습니다' if ok else '★실패 — 원인을 찾을 때까지 다음 단계로 가지 않습니다'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
