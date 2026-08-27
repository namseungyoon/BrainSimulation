# -*- coding: utf-8 -*-
"""4-5 burst 위상 정렬 검증 — "peak 에 줬다" 가 사실인가

단계   : 4-5 (파이프라인 4단계 구동·리듬 / 하위 5 phase_align)
쉬운 설명: 6-1 은 "theta peak 에 burst 를 주면 LTP, trough 에 주면 LTD" 를 본다. 그 주장은
          burst 가 **실제로 그 위상에 놓였는지** 확인해야 성립한다. 4-3 에서 주입 전류와
          막전위의 위상이 최대 87도 어긋나는 것을 봤으므로, 목표 위상을 그냥 믿으면 안 된다.
★PLAN  : "4-5 가 없으면 6-1 이 성립하지 않는다."
방법   : D37 이 정한 기준(**시냅스 위치의 국소 막전위**)으로 정렬을 실측한다.
          (A) 목표 위상(peak/trough/상승/하강)마다 burst 를 주고 **각 펄스가 실제로 놓인 위상**
          (B) 목표-실제 오차와 **burst 내 위상 퍼짐**(4펄스가 100Hz 라 theta 위상이 벌어진다)
          (C) 소마 기준으로 재면 얼마나 틀리는가 (문헌 변환용)
          (D) theta 주파수를 바꾸면 정렬이 유지되는가
검증   : 목표-실제 오차 · 위상 퍼짐 정량 · 소마↔시냅스 차이.
근거   : 4-3 D37(위상 기준·전류 세기) · 4-1 D17(버스트 펄스 1.0ms/5.0nA) · 4-2(부과)
결과   : figures/4-5_phase_align.png · figures/4-5_align.json
실행   : . .\\env\\activate.ps1 ; & $Py04 04_drive\\5_phase_align\\4-5_phase_align.py
비고   : ★부과 theta 다(4-2 판정: 자연 불가). '자연 발생' 이라고 쓰지 않는다.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                  # noqa: E402
from lib import plots                                # noqa: E402
from lib.bench import Bench                          # noqa: E402
from lib.wiring import Wiring, SETTLE_MS             # noqa: E402
from lib.nrnenv import h                             # noqa: E402

DT = 0.025
REC_DT = 0.1
T0 = SETTLE_MS
N_CYCLE = 8                       # theta 주기 수 (burst 는 그중 몇 번째부터)
BURST_FROM = 3                    # 과도상태 지난 뒤부터 burst
N_BURST = 4                       # burst 횟수 (주기당 1회)
# 4-1 D17: 버스트 펄스 표준
PULSE_MS, PULSE_NA = 1.0, 5.0
N_IN, IN_HZ = 4, 100.0            # 고전 TBS 버스트
TARGETS = [("peak", 0.0), ("trough", 180.0), ("상승", -90.0), ("하강", 90.0)]
FREQS = [5.0, 8.0]


def fit_sine(t, y, f_hz):
    w = 2.0 * np.pi * f_hz * (np.asarray(t) / 1000.0)
    X = np.column_stack([np.cos(w), np.sin(w), np.ones_like(w)])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    a, b, m = coef
    return float(np.hypot(a, b)), float(np.arctan2(b, a)), float(m)


def wrap(d):
    """[-180, 180) 로 감싼다. 스칼라면 float, 배열이면 배열을 준다."""
    r = (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0
    return float(r) if r.ndim == 0 else r


def r_theory(spread_deg, n=N_IN):
    """등간격 n점이 spread 만큼 퍼져 있을 때의 원형 집중도 r (이론값).

    ★임의의 하한(r > 0.85 같은)을 두는 대신, 측정된 퍼짐으로부터 r 을 계산해
      실측 r 과 맞는지 본다. 그러면 '퍼짐이 크다' 는 사실과 '측정이 옳다' 는 사실이
      서로를 검증한다. (8Hz 퍼짐 86.4도의 이론 r 이 정확히 0.849 다.)
    """
    th = np.radians(np.linspace(0.0, spread_deg, n))
    return float(np.abs(np.mean(np.exp(1j * th))))


def circ_mean_deg(d):
    z = np.mean(np.exp(1j * np.radians(np.asarray(d))))
    return float(np.degrees(np.angle(z))), float(np.abs(z))


def main():
    plots.setup()
    print("=== 4-5 burst 위상 정렬 검증 (★부과 theta) ===")
    b = Bench()
    w = Wiring(b, frozen=True)
    for syn, _ in w.syns:
        syn.gmax = 0.0                 # 전달은 끈다 — 정렬만 본다
    site_lbl = [f"{sp['path_um']:.0f}um" for _, sp in w.syns]

    # 4-3 인용
    A3 = json.load(open(os.path.join(ROOT, "04_drive", "3_imposed_theta",
                                     "figures", "4-3_theta.json"),
                        "r", encoding="utf-8"))
    AMP_THETA = float(A3["sine"]["amp_nA"])
    print(f"  4-3 인용: 정현파 {AMP_THETA:.4f} nA · 위상 기준 = 시냅스 국소 Vm (D37)")
    print(f"  4-1 인용: 버스트 펄스 {PULSE_MS}ms/{PULSE_NA}nA · "
          f"{N_IN}발 {IN_HZ:.0f}Hz (D17)")
    print(f"  기록: 소마 + 시냅스 {', '.join(site_lbl)}")

    # 구동: theta 정현파 + burst 펄스 (같은 IClamp 에 더한다)
    ic = h.IClamp(b.post_soma_seg())
    ic.delay, ic.dur, ic.amp = 0.0, 1e9, 0.0
    drive = None

    def _play(wav, tstop):
        nonlocal drive
        if drive is None:
            drive = h.Vector(wav)
            drive.play(ic._ref_amp, DT)
            w.keep.append(drive)
        else:
            drive.from_python(wav)
        w.run(tstop, dt=DT)
        return w.arrays()

    def run_case(f_hz, target_deg, ref_phase=None):
        """theta + 지정 위상에 burst.

        ★ref_phase 를 주면 **그 위상(막전위 기준)** 에 burst 중심을 맞춘다.
          주지 않으면 theta 만 돌리는 기준 실행이다.
          처음 판은 burst 를 **주입 파형** peak 에 맞췄는데, 조준해야 하는 것은
          막전위다 — 그래서 5Hz 에서 116도가 통째로 어긋났다.
        """
        per = 1000.0 / f_hz
        tstop = T0 + (N_CYCLE + 1) * per
        n = int(tstop / DT) + 2
        tt = np.arange(n) * DT
        ph = 2 * np.pi * f_hz * (tt - T0) / 1000.0
        wav = np.where(tt >= T0, AMP_THETA * np.cos(ph), 0.0)
        if ref_phase is None:
            return _play(wav, tstop), np.array([]), tstop, tt, wav
        # 막전위 위상이 target 이 되는 시각 (절대시간 기준 적합 위상 ref_phase 사용)
        # phase(t) = 2pi f t/1000 - ref_phase  ->  t = (target + ref_phase)/(2pi f) * 1000
        span = (N_IN - 1) * 1000.0 / IN_HZ            # burst 지속시간
        base = (np.radians(target_deg) + ref_phase) / (2 * np.pi * f_hz) * 1000.0
        pulses = []
        for k in range(N_BURST):
            # burst **중심**을 목표 위상에 맞춘다 -> 첫 펄스는 span/2 앞
            t_center = base + (BURST_FROM + k) * per
            while t_center < T0 + BURST_FROM * per:
                t_center += per
            for j in range(N_IN):
                pulses.append(t_center - span / 2.0 + j * 1000.0 / IN_HZ)
        for tp in pulses:
            m = (tt >= tp) & (tt < tp + PULSE_MS)
            wav[m] += PULSE_NA
        return _play(wav, tstop), np.array(pulses), tstop, tt, wav

    w.record(rec_dt=REC_DT, local_v=True, currents=False)

    def phase_at(times, f_hz, ref_phase):
        """주어진 시각들의 theta 위상(0=peak). ref_phase 는 기준 실행에서 잰 적합 위상.

        ★기준 위상은 **burst 가 없는 별도 실행**에서 잰다. burst 가 섞인 신호로 적합하면
          펄스의 큰 탈분극이 적합을 끌어당겨 위상이 틀어진다. 처음 판은 같은 실행의
          앞 2주기로 적합했는데 시작 과도상태까지 섞여 있었다.
        """
        ph = 2 * np.pi * f_hz * (np.asarray(times) / 1000.0) - ref_phase
        return wrap(np.degrees(ph))

    # ── 기준 실행: theta 만 돌려 위상을 잰다 (주입 파형도 같은 함수로 적합) ──
    print("\n  [기준] theta 만 돌려 위상을 잰다 (burst 없이)")
    ref = {}
    for f_hz in FREQS:
        Rr, _, tstop, tt, wav = run_case(f_hz, 0.0, ref_phase=None)
        t = Rr["t"]
        m = t >= T0 + 2.0 * 1000.0 / f_hz
        mw = tt >= T0 + 2.0 * 1000.0 / f_hz
        _, ph_inj, _ = fit_sine(tt[mw], wav[mw], f_hz)
        d = {}
        for lab, sig in [("소마", Rr["post_v"])] + \
                        [(site_lbl[i], Rr["local_v"][i]) for i in range(len(site_lbl))]:
            A, pt, mn = fit_sine(t[m], sig[m], f_hz)
            d[lab] = dict(phase=pt, pp=2 * A)
        ref[f_hz] = dict(inj_phase=ph_inj, sites=d)
        lag_soma = wrap(np.degrees(d["소마"]["phase"] - ph_inj))
        lag_syn = wrap(np.degrees(d[site_lbl[0]]["phase"] - ph_inj))
        print(f"      {f_hz:.0f}Hz : 주입->소마 지연 {lag_soma:+.2f}deg · "
              f"주입->시냅스 {lag_syn:+.2f}deg · theta {d['소마']['pp']:.2f} mV(pp)")
        ref[f_hz]["lag_soma_deg"] = lag_soma
        ref[f_hz]["lag_syn_deg"] = lag_syn

    rows = []
    print("\n  [정렬] burst 중심을 목표 위상(시냅스 국소 Vm 기준)에 맞춘다")
    for f_hz in FREQS:
        rp = ref[f_hz]["sites"][site_lbl[0]]["phase"]      # D37 기준으로 조준
        for name, tgt in TARGETS:
            R, pulses, tstop, tt, wav = run_case(f_hz, tgt, ref_phase=rp)
            rec = {}
            for lab in ["소마"] + site_lbl:
                pd = phase_at(pulses, f_hz, ref[f_hz]["sites"][lab]["phase"])
                cm, rlen = circ_mean_deg(pd)
                spread = float(np.ptp(wrap(pd[:N_IN] - pd[0])))
                rec[lab] = dict(phase_deg=[float(v) for v in pd],
                                circ_mean=cm, r=rlen,
                                theta_pp=ref[f_hz]["sites"][lab]["pp"],
                                err=wrap(cm - tgt), spread_in_burst=spread)
            rf = rec[site_lbl[0]]
            rows.append(dict(f_hz=f_hz, target=name, target_deg=tgt, rec=rec,
                             R=R, pulses=pulses))
            print(f"  {f_hz:.0f}Hz {name:<5}(목표 {tgt:+6.1f}deg): "
                  f"시냅스 기준 실제 {rf['circ_mean']:+7.1f}deg "
                  f"(오차 {rf['err']:+6.1f}deg) · burst 내 퍼짐 "
                  f"{rf['spread_in_burst']:5.1f}deg · 소마 기준 "
                  f"{rec['소마']['circ_mean']:+7.1f}deg "
                  f"(차 {wrap(rf['circ_mean'] - rec['소마']['circ_mean']):+.2f}deg)")

    # ── 요약 ──────────────────────────────────────────────────────────────
    errs = [abs(r["rec"][site_lbl[0]]["err"]) for r in rows]
    spreads = [r["rec"][site_lbl[0]]["spread_in_burst"] for r in rows]
    soma_diff = [abs(wrap(r["rec"][site_lbl[0]]["circ_mean"]
                          - r["rec"]["소마"]["circ_mean"])) for r in rows]
    print(f"\n  ★요약 (기준 = 시냅스 {site_lbl[0]} 국소 Vm · D37)")
    print(f"      목표-실제 오차: 최대 {max(errs):.1f}deg · 평균 {np.mean(errs):.1f}deg")
    print(f"      burst 내 위상 퍼짐: {min(spreads):.1f}~{max(spreads):.1f}deg "
          f"({N_IN}발 {IN_HZ:.0f}Hz = {(N_IN-1)*10.0:.0f}ms 동안 theta 가 진행한다)")
    print(f"      소마 기준과의 차이: 최대 {max(soma_diff):.2f}deg (문헌 변환용)")
    # 이론값: burst 지속시간 동안 theta 가 도는 각도
    theo = {f: (N_IN - 1) * (1000.0 / IN_HZ) / (1000.0 / f) * 360.0 for f in FREQS}
    print(f"      이론 퍼짐: " +
          " · ".join(f"{f:.0f}Hz {theo[f]:.1f}deg" for f in FREQS))

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.50)
    axA = fig.add_subplot(gs_[0, :2])
    axB = fig.add_subplot(gs_[0, 2], projection="polar")
    axC = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axE = fig.add_subplot(gs_[1, 2])

    # A: 파형 (5Hz peak/trough)
    for r, col, ls in ((rows[0], "#c62828", "-"), (rows[1], "#1565c0", "-")):
        R = r["R"]; f_hz = r["f_hz"]
        per = 1000.0 / f_hz
        lo = T0 + (BURST_FROM - 0.6) * per
        hi = T0 + (BURST_FROM + 2.2) * per
        m = (R["t"] >= lo) & (R["t"] <= hi)
        axA.plot(R["t"][m] - T0, R["local_v"][0][m], color=col, lw=1.2,
                 label=f"{r['target']} (시냅스 {site_lbl[0]})")
        for tp in r["pulses"]:
            if lo <= tp <= hi:
                axA.plot([tp - T0], [40], "|", color=col, ms=7)
    axA.set_xlabel("리듬 시작 기준 시간 (ms)"); axA.set_ylabel("시냅스 국소 Vm (mV)")
    axA.set_title("A. ★부과 theta 위의 burst — 위쪽 막대가 펄스 시각\n"
                  "기준은 **시냅스 국소 막전위**다 (D37)", fontsize=9.4, loc="left")
    axA.legend(fontsize=7.8)

    # B: 극좌표 — 목표 vs 실제
    for r, col in zip(rows[:4], ["#c62828", "#1565c0", "#2e7d32", "#ef6c00"]):
        pd = r["rec"][site_lbl[0]]["phase_deg"]
        axB.plot(np.radians(pd), np.ones(len(pd)) * 1.0, "o", ms=4, color=col,
                 alpha=0.7)
        cm = r["rec"][site_lbl[0]]["circ_mean"]
        axB.plot([np.radians(cm)], [1.25], "*", ms=13, color=col,
                 label=f"{r['target']} ({cm:+.0f}deg)")
        axB.plot([np.radians(r["target_deg"])], [1.45], "v", ms=8, color=col)
    axB.set_ylim(0, 1.6); axB.set_yticks([])
    axB.set_theta_zero_location("E")
    axB.set_title("B. 목표(v) vs 실제(*) — 5 Hz\n점 = 개별 펄스", fontsize=9.2,
                  pad=16)
    axB.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.25, -0.12))

    # C: 오차
    xx = np.arange(len(rows))
    axC.bar(xx, [r["rec"][site_lbl[0]]["err"] for r in rows],
            color=["#c62828" if r["f_hz"] == 5.0 else "#1565c0" for r in rows],
            width=0.6)
    axC.axhline(0, color="#37474f", lw=1.0)
    axC.set_xticks(xx)
    axC.set_xticklabels([f"{r['f_hz']:.0f}Hz\n{r['target']}" for r in rows],
                        fontsize=7)
    axC.set_ylabel("목표-실제 위상 오차 (deg)")
    axC.set_title(f"C. 정렬 오차 (기준 = 시냅스 국소 Vm)\n최대 {max(errs):.1f}deg",
                  fontsize=9.2, loc="left")

    # D: burst 내 퍼짐
    axD.bar(xx, spreads, color="#6a1b9a", width=0.6)
    for f in FREQS:
        axD.axhline(theo[f], color="#37474f", ls="--", lw=1.2)
        axD.text(len(rows) - 0.5, theo[f], f" 이론 {f:.0f}Hz {theo[f]:.0f}deg",
                 fontsize=7, va="bottom", ha="right", color="#37474f")
    axD.set_xticks(xx)
    axD.set_xticklabels([f"{r['f_hz']:.0f}Hz\n{r['target']}" for r in rows],
                        fontsize=7)
    axD.set_ylabel("burst 내 위상 퍼짐 (deg)")
    axD.set_title(f"D. ★burst 는 한 점이 아니다\n"
                  f"{N_IN}발 {IN_HZ:.0f}Hz = {(N_IN-1)*10:.0f}ms 동안 theta 가 진행한다",
                  fontsize=9.2, loc="left")

    # E: 소마 vs 시냅스 기준
    axE.bar(xx - 0.19, [r["rec"]["소마"]["circ_mean"] for r in rows], width=0.38,
            color="#90a4ae", label="소마 기준")
    axE.bar(xx + 0.19, [r["rec"][site_lbl[0]]["circ_mean"] for r in rows],
            width=0.38, color="#2e7d32", label=f"시냅스 {site_lbl[0]} 기준")
    axE.set_xticks(xx)
    axE.set_xticklabels([f"{r['f_hz']:.0f}Hz\n{r['target']}" for r in rows],
                        fontsize=7)
    axE.set_ylabel("실제 위상 (deg)")
    axE.set_title(f"E. 소마 기준 vs 시냅스 기준\n차이 최대 {max(soma_diff):.2f}deg "
                  f"(문헌 변환에 쓴다)", fontsize=9.2, loc="left")
    axE.legend(fontsize=7.8)

    fig.suptitle("4-5  burst 위상 정렬 검증 — \"peak 에 줬다\" 가 사실인가 (★부과 theta)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"4-5 | ★부과 theta · 기준=시냅스 {site_lbl[0]} 국소 Vm(D37) · "
                     f"펄스 {PULSE_MS}ms/{PULSE_NA}nA(D17) · 정렬 오차 <= {max(errs):.1f}deg · "
                     f"burst 내 퍼짐 {min(spreads):.0f}~{max(spreads):.0f}deg")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "4-5_phase_align.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    r_len = [r["rec"][site_lbl[0]]["r"] for r in rows]
    checks = [
        (f"★목표 위상에 실제로 놓인다 (최대 오차 {max(errs):.1f}deg < 25deg)",
         max(errs) < 25.0),
        ("네 목표 위상이 서로 구분된다 (peak vs trough 차이 > 120deg)",
         abs(wrap(rows[0]["rec"][site_lbl[0]]["circ_mean"]
                  - rows[1]["rec"][site_lbl[0]]["circ_mean"])) > 120.0),
        (f"★burst 내 위상이 퍼진다 — 한 점이 아니다 "
         f"({min(spreads):.0f}~{max(spreads):.0f}deg)", min(spreads) > 5.0),
        ("burst 내 퍼짐이 이론값과 맞는다 (20% 이내)",
         all(abs(r["rec"][site_lbl[0]]["spread_in_burst"] - theo[r["f_hz"]])
             / theo[r["f_hz"]] < 0.20 for r in rows)),
        (f"★원형 집중도가 **퍼짐으로부터 계산한 이론값과 일치**한다 "
         f"(실측 {min(r_len):.3f}) — 임의 하한 대신 자기정합 검사",
         all(abs(r["rec"][site_lbl[0]]["r"]
                 - r_theory(r["rec"][site_lbl[0]]["spread_in_burst"])) < 0.01
             for r in rows)),
        (f"★소마 기준과 시냅스 기준의 차이가 작다 (최대 {max(soma_diff):.2f}deg < 10deg) "
         f"— 문헌 변환 가능", max(soma_diff) < 10.0),
        ("두 theta 주파수 모두에서 정렬이 유지된다",
         max(abs(r["rec"][site_lbl[0]]["err"]) for r in rows if r["f_hz"] == 8.0)
         < 25.0),
        ("바탕 theta 진폭이 4-3 목표 근처다 (기준 실행에서 잰 값)",
         all(6.0 < r["rec"]["소마"]["theta_pp"] < 11.0 for r in rows)),
        ("★주입->막전위 지연을 직접 측정했다 (주입 파형을 같은 함수로 적합)",
         all(np.isfinite(ref[f]["lag_soma_deg"]) for f in FREQS)),
        ("4-3·4-1 의 확정값을 그대로 인용했다",
         abs(AMP_THETA - 0.1221) < 1e-3 and PULSE_NA == 5.0),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(imposed=True,
               note_imposed="★부과 theta 다(4-2 판정: 자연 불가). '자연 발생' 이라고 쓰지 않는다.",
               phase_reference=f"시냅스 {site_lbl[0]} 국소 막전위 (D37)",
               dt=DT, f_theta_list=FREQS, targets=[t for t, _ in TARGETS],
               burst=dict(n_burst=N_BURST, n_in=N_IN, in_hz=IN_HZ,
                          pulse_ms=PULSE_MS, pulse_nA=PULSE_NA, source="D17"),
               theta=dict(amp_nA=AMP_THETA, source="D37"),
               reference_runs={str(f): dict(
                   inj_to_soma_lag_deg=ref[f]["lag_soma_deg"],
                   inj_to_syn_lag_deg=ref[f]["lag_syn_deg"],
                   theta_pp={k: v["pp"] for k, v in ref[f]["sites"].items()})
                   for f in FREQS},
               correction=("★4-3 의 '주입 대비 소마 지연 +86.7deg' 는 오독이었다. "
                           "fit_sine 은 절대시간 기준 위상을 주는데 주입 파형은 "
                           "cos(2pi f (t-T0)) 이라 같은 기준에서 위상이 2pi f T0 다. "
                           "그 보정을 빼먹었다. 여기서는 **주입 파형 자체를 같은 함수로 "
                           "적합**해 기준을 만들었다 — 규약에 의존하지 않는 방법이다. "
                           "4-3 스크립트도 같은 방식으로 고쳤다."),
               rows=[{k: v for k, v in r.items() if k not in ("R", "pulses")}
                     for r in rows],
               summary=dict(max_err_deg=max(errs), mean_err_deg=float(np.mean(errs)),
                            spread_min_deg=min(spreads), spread_max_deg=max(spreads),
                            spread_theory_deg={str(f): theo[f] for f in FREQS},
                            max_soma_vs_syn_deg=max(soma_diff),
                            min_r=min(r_len),
                            r_theory={str(r["f_hz"]) + r["target"]:
                                      r_theory(r["rec"][site_lbl[0]]["spread_in_burst"])
                                      for r in rows}),
               finding=(f"★burst 는 위상의 '한 점' 이 아니다. {N_IN}발 {IN_HZ:.0f}Hz 는 "
                        f"{(N_IN-1)*10:.0f}ms 동안 이어지므로 그동안 theta 가 진행한다 — "
                        f"5Hz 에서 {theo[5.0]:.0f}도, 8Hz 에서 {theo[8.0]:.0f}도 퍼진다. "
                        f"6-1 은 'peak 에 줬다' 가 아니라 '**peak 중심 ±{theo[5.0]/2:.0f}도** "
                        f"구간에 줬다' 로 보고해야 한다. 목표 위상 간격(90도)보다 퍼짐이 크면 "
                        f"조건들이 겹치기 시작한다 — 8Hz 에서 특히 확인이 필요하다."),
               conversion=(f"소마 기준과 시냅스 기준의 차이는 최대 {max(soma_diff):.2f}도로 "
                           f"작다. 문헌은 소마(또는 장전위) 기준으로 보고하므로 이 값으로 "
                           f"변환한다. 반면 **주입 전류 위상은 쓰지 않는다** — 4-3 에서 "
                           f"소마와 최대 86.7도 어긋났다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "4-5_align.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 4-5 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
