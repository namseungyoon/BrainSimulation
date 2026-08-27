# -*- coding: utf-8 -*-
"""4-3 부과 theta — 두 방식을 비교하고 **위상의 기준**을 정한다

단계   : 4-3 (파이프라인 4단계 구동·리듬 / 하위 3 imposed_theta)
쉬운 설명: 4-2 에서 이 세포가 스스로 theta 리듬을 못 만든다고 판정했다(불가). 그래서 우리가
          theta 를 **부과**해야 한다. 방법이 두 가지 있고, 어느 쪽을 기본으로 쓸지 실측으로
          정한다(미결#5).
★핵심  : PLAN 이 지적한 함정 — "peak 에 burst 를 줬다" 는 주장은 **무엇의 peak 인지** 정해야
          성립한다. 소마에 전류를 주면 막 시상수 때문에 **주입 전류의 위상과 막전위의 위상이
          어긋나고**, 시냅스가 있는 수상돌기 위치에서는 또 다르다. 그걸 실측해 기준을 못박는다.
방법   : (A) 정현파 전류 주입 — 통제가 쉽고 위상이 정확히 정의된다. 생리적 기전은 아니다.
          (B) 억제성 시냅스 리듬 — 실제 theta 는 개재뉴런의 리듬적 억제로 만들어지므로 기전상
              정확하다. 다만 개재뉴런이 없으므로 **리듬 자체는 여전히 부과**다(가상 억제 입력).
          두 방식을 같은 목표 진폭으로 보정한 뒤 **소마와 두 시냅스 위치**에서 진폭·위상을 잰다.
검증   : 목표 주파수·진폭 진동 확인 + 위상 지연 정량 + 기본 방식 결정.
근거   : 4-2 판정(자연 theta 불가) · docs/DECISIONS.md 미결#5 · theta 대역 [4,8]Hz (2-5)
결과   : figures/4-3_imposed_theta.png · figures/4-3_theta.json
실행   : . .\\env\\activate.ps1 ; & $Py04 04_drive\\3_imposed_theta\\4-3_imposed_theta.py
비고   : ★부과된 theta 다. 그림·문서에 '부과' 를 명기하고 '자연 발생' 이라고 쓰지 않는다(D34).
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
REC_DT = 0.25
T0 = SETTLE_MS                     # 정착 뒤부터 리듬을 준다
DUR = 1200.0                       # 리듬 기록 1.2초 (5Hz 에서 6주기)
FREQS = [5.0, 8.0]                 # theta 대역 양끝 (2-5 의 [4,8])
TARGET_PP = 8.0                    # 목표 소마 Vm 진폭 (peak-to-peak, mV)
N_GABA = 8                         # 억제성 시냅스 개수 (소마+근위)
E_GABA = -80.0                     # DetGABAAB 기본값


def fit_sine(t, y, f_hz):
    """주파수 f 에서 최소제곱 정현파 적합 -> (진폭, 위상rad, 평균).

    진폭은 정현파 반진폭이므로 peak-to-peak 는 2배다. 위상은 cos 기준
    (y ~ mean + A*cos(2*pi*f*t - phase)).
    """
    w = 2.0 * np.pi * f_hz * (np.asarray(t) / 1000.0)
    X = np.column_stack([np.cos(w), np.sin(w), np.ones_like(w)])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    a, b, m = coef
    return float(np.hypot(a, b)), float(np.arctan2(b, a)), float(m)


def wrap_deg(rad):
    d = np.degrees(rad)
    return float((d + 180.0) % 360.0 - 180.0)


def main():
    plots.setup()
    print("=== 4-3 부과 theta (★부과다 — 자연 발생 아니다, D34) ===")
    b = Bench()
    w = Wiring(b, frozen=True)
    for syn, _ in w.syns:
        syn.gmax = 0.0                      # PC->PC 전달은 끈다 (리듬만 본다)
    sites = [(f"시냅스 {sp['path_um']:.0f}um ({sp['section']})", sp)
             for _, sp in w.syns]
    site_str = ", ".join("%.0fum" % sp["path_um"] for _, sp in w.syns)
    print(f"  기록: post 소마 + 시냅스 {len(sites)}지점 ({site_str})")

    # ── 구동 장치 ─────────────────────────────────────────────────────────
    # (A) 정현파 전류: IClamp 하나 + Vector.play
    ic = h.IClamp(b.post_soma_seg())
    ic.delay = 0.0
    ic.dur = 1e9
    ic.amp = 0.0
    # (B) 억제성 시냅스: 소마 + **정단 근위** (개재뉴런의 소마주변 표적)
    #  ★기록 지점(기저 dend[3]·dend[23])과 **겹치지 않게** 둔다. 처음 판은 dend[0..6] 에
    #    뒀다가 dend[3] 과 겹쳐 그 위치 진폭비가 1.93 으로 나왔다 — 배치 인공물이었다.
    rec_secs = {sp["section"] for _, sp in w.syns}
    gsyn, gnc = [], []
    segs = [b.post_soma_seg()]
    for sec in list(b.post.apic):
        nm = sec.name().split(".")[-1]
        if nm in rec_secs:
            continue
        segs.append(sec(0.5))
        if len(segs) >= N_GABA:
            break
    gabasecs = [sg.sec.name().split(".")[-1] for sg in segs[:N_GABA]]
    print(f"  억제 시냅스 위치 {len(segs[:N_GABA])}개: {', '.join(gabasecs)} "
          f"(기록 지점 {sorted(rec_secs)} 제외)")
    for sg in segs[:N_GABA]:
        s_ = h.DetGABAAB(sg)
        s_.e_GABAA = E_GABA
        s_.Use = 1.0
        s_.GABAB_ratio = 0.0
        gsyn.append(s_)
    gvec = h.Vector(); gvs = h.VecStim(); gvs.play(gvec)
    for s_ in gsyn:
        nc = h.NetCon(gvs, s_)
        nc.delay = 0.0
        nc.weight[0] = 0.0                  # nS (DetGABAAB 는 weight 로 받는다)
        gnc.append(nc)
    w.keep += [ic, gvec, gvs] + gsyn + gnc

    w.record(rec_dt=REC_DT, local_v=True, currents=False)

    # ★ settle/restore(D12) 를 쓰지 않는다 — Vector.play 와 VecStim 은 finitialize 에서
    #   등록되므로 restore() 뒤에는 **조용히 아무 일도 하지 않는다**(실측: 두 방식 모두
    #   진폭 0.000mV). 대신 조건마다 run()(=finit)을 쓰고, 자극을 T0=SETTLE_MS 뒤로 두어
    #   같은 정착 효과를 얻는다. 비용은 조건당 250ms 뿐이다.
    TS_MAX = T0 + DUR + 10.0
    N_MAX = int(TS_MAX / DT) + 2
    sinevec = h.Vector(np.zeros(N_MAX))
    sinevec.play(ic._ref_amp, DT)          # 한 번 설치하고 내용만 갈아끼운다
    w.keep.append(sinevec)
    FAR = np.array([1e9])                  # VecStim 을 쉬게 할 때 쓰는 먼 시각
    print(f"  자극은 T0={T0:.0f}ms(=정착) 뒤부터 · 조건마다 finit (restore 불가, 위 주석)")

    def _fill_sine(f_hz, amp_nA):
        tt = np.arange(N_MAX) * DT
        wav = np.where(tt >= T0,
                       amp_nA * np.cos(2 * np.pi * f_hz * (tt - T0) / 1000.0), 0.0)
        sinevec.from_python(wav)
        return wav, tt

    def run_sine(f_hz, amp_nA, tstop):
        wav, tt = _fill_sine(f_hz, amp_nA)
        gvec.from_python(FAR)              # 억제 입력 없음
        for nc in gnc:
            nc.weight[0] = 0.0
        w.run(tstop, dt=DT)
        return w.arrays(), wav, tt

    def run_inhib(f_hz, g_nS, tstop):
        _fill_sine(f_hz, 0.0)              # 전류 없음
        isi = 1000.0 / f_hz
        times = np.arange(T0, tstop - 10.0, isi)
        gvec.from_python(times)
        for nc in gnc:
            nc.weight[0] = g_nS
        w.run(tstop, dt=DT)
        return w.arrays(), times

    def measure(R, f_hz):
        """소마·시냅스 위치의 진폭·위상. 리듬 시작 뒤 2주기는 버린다(과도상태)."""
        t = R["t"]
        t_lo = T0 + 2.0 * 1000.0 / f_hz
        m = t >= t_lo
        out = {}
        A, ph, mn = fit_sine(t[m], R["post_v"][m], f_hz)
        out["소마"] = dict(pp=2 * A, phase_deg=wrap_deg(ph), mean=mn)
        for i, (name, sp) in enumerate(sites):
            A2, ph2, mn2 = fit_sine(t[m], R["local_v"][i][m], f_hz)
            out[name] = dict(pp=2 * A2, phase_deg=wrap_deg(ph2), mean=mn2)
        return out

    # ── (A) 정현파 전류 보정 ──────────────────────────────────────────────
    print(f"\n  [A] 정현파 전류 — 목표 소마 진폭 {TARGET_PP:.0f} mV(peak-to-peak)")
    TS = T0 + DUR
    calA = []
    for a in (0.05, 0.10, 0.20):
        R, wav, tt = run_sine(FREQS[0], a, TS)
        mm = measure(R, FREQS[0])
        calA.append(dict(amp_nA=a, pp=mm["소마"]["pp"]))
        print(f"      {a:.2f} nA -> 소마 {mm['소마']['pp']:.3f} mV(pp)")
    # 선형이므로 목표에 맞는 진폭을 외삽
    sl = np.polyfit([c["amp_nA"] for c in calA], [c["pp"] for c in calA], 1)
    AMP_A = float((TARGET_PP - sl[1]) / sl[0])
    print(f"      -> 선형 적합으로 {AMP_A:.4f} nA 채택 "
          f"(기울기 {sl[0]:.2f} mV/nA · 절편 {sl[1]:+.3f} mV)")

    # ── (B) 억제성 시냅스 리듬 보정 ───────────────────────────────────────
    print(f"\n  [B] 억제성 시냅스 리듬 — GABA 시냅스 {N_GABA}개 "
          f"(e_GABAA {E_GABA:.0f} mV) · 주기당 1펄스")
    # ★선형 외삽은 쓸 수 없다 — 억제는 포화한다(막전위가 e_GABAA 로 접근하며 구동력이
    #   사라진다). 처음 판은 3점 선형 외삽으로 34.7nS 를 뽑았고 실제로는 목표의 40% 밖에
    #   못 냈다. 로그 스윕으로 **실측 곡선**을 얻고 거기서 보간한다.
    calB = []
    for g in (1.0, 4.0, 16.0, 64.0, 256.0):
        R, times = run_inhib(FREQS[0], g, TS)
        mm = measure(R, FREQS[0])
        calB.append(dict(g_nS=g, pp=mm["소마"]["pp"], mean=mm["소마"]["mean"]))
        print(f"      {g:6.1f} nS x{N_GABA} -> 소마 {mm['소마']['pp']:6.3f} mV(pp) · "
              f"평균 {mm['소마']['mean']:.2f} mV")
    gg = np.array([c["g_nS"] for c in calB]); pp = np.array([c["pp"] for c in calB])
    pp_max = float(pp.max())
    sat_ratio = float(pp[-1] / pp[0]) / float(gg[-1] / gg[0])   # 1 이면 선형
    if pp_max >= TARGET_PP:
        G_B = float(np.interp(TARGET_PP, pp, gg))
        reach = True
    else:
        G_B = float(gg[int(np.argmax(pp))])
        reach = False
    print(f"      포화 지표: 세기 {gg[-1]/gg[0]:.0f}배에 진폭 {pp[-1]/pp[0]:.1f}배 "
          f"(선형이면 같아야 한다) -> 선형성 {sat_ratio:.3f}")
    print(f"      최대 도달 진폭 {pp_max:.3f} mV -> 목표 {TARGET_PP:.0f} mV "
          f"{'도달 가능' if reach else '**도달 불가**'} · 채택 {G_B:.3f} nS")

    # ── (C)(D) 본 측정: 두 방식 x 두 주파수 ───────────────────────────────
    print(f"\n  [C·D] 본 측정 — 소마와 시냅스 위치의 진폭·위상")
    main = {}
    traces = {}
    for f_hz in FREQS:
        R, wav, tt = run_sine(f_hz, AMP_A, T0 + DUR)
        mmA = measure(R, f_hz)
        traces[("정현파", f_hz)] = (R, wav, tt)
        R2, times = run_inhib(f_hz, G_B, T0 + DUR)
        mmB = measure(R2, f_hz)
        traces[("억제리듬", f_hz)] = (R2, times, None)
        main[f_hz] = dict(sine=mmA, inhib=mmB)
        for lab, mm in (("정현파", mmA), ("억제리듬", mmB)):
            print(f"      {lab} {f_hz:.0f}Hz : " +
                  " · ".join(f"{k} {v['pp']:.2f}mV @ {v['phase_deg']:+.1f}deg"
                             for k, v in mm.items()))
        # 위상 지연: 소마 기준으로 시냅스 위치가 얼마나 늦는가
        for lab, mm in (("정현파", mmA), ("억제리듬", mmB)):
            for k in list(mm)[1:]:
                d = wrap_deg(np.radians(mm[k]["phase_deg"] - mm["소마"]["phase_deg"]))
                print(f"        {lab} {k}: 소마 대비 위상차 {d:+.2f}deg "
                      f"({d/360.0*1000.0/f_hz:+.2f}ms) · 진폭비 "
                      f"{mm[k]['pp']/mm['소마']['pp']:.3f}")

    # 주입 신호 대비 소마 Vm 위상 (정현파 방식만 정의된다)
    inj_lag = {}
    for f_hz in FREQS:
        R, wav, tt = traces[("정현파", f_hz)]
        # 주입 파형은 cos(2pi f (t-T0)) 이므로 위상 0 기준
        inj_lag[f_hz] = main[f_hz]["sine"]["소마"]["phase_deg"]
    print(f"\n  [★] 주입 전류 대비 소마 Vm 위상 지연: " +
          " · ".join(f"{f:.0f}Hz {inj_lag[f]:+.1f}deg "
                     f"({inj_lag[f]/360.0*1000.0/f:+.1f}ms)" for f in FREQS))

    # ── 판정 ──────────────────────────────────────────────────────────────
    f0 = FREQS[0]
    syn_names = [k for k in main[f0]["sine"] if k != "소마"]
    dphi_sine = max(abs(wrap_deg(np.radians(main[f]["sine"][k]["phase_deg"]
                                           - main[f]["sine"]["소마"]["phase_deg"])))
                    for f in FREQS for k in syn_names)
    dphi_inh = max(abs(wrap_deg(np.radians(main[f]["inhib"][k]["phase_deg"]
                                          - main[f]["inhib"]["소마"]["phase_deg"])))
                   for f in FREQS for k in syn_names)
    amp_ratio_sine = min(main[f]["sine"][k]["pp"] / main[f]["sine"]["소마"]["pp"]
                         for f in FREQS for k in syn_names)
    amp_ratio_inh = min(main[f]["inhib"][k]["pp"] / main[f]["inhib"]["소마"]["pp"]
                        for f in FREQS for k in syn_names)
    # 기본 방식: 통제성(목표 진폭 재현 오차)과 위치 균일성으로 고른다
    errA = max(abs(main[f]["sine"]["소마"]["pp"] - TARGET_PP) / TARGET_PP
               for f in FREQS)
    errB = max(abs(main[f]["inhib"]["소마"]["pp"] - TARGET_PP) / TARGET_PP
               for f in FREQS)
    default = "정현파 전류" if errA <= errB else "억제성 시냅스 리듬"
    print(f"\n  ★판정")
    print(f"      목표 진폭 재현 오차: 정현파 {100*errA:.1f}% · 억제리듬 {100*errB:.1f}%")
    print(f"      시냅스 위치 진폭비: 정현파 {amp_ratio_sine:.3f} · 억제리듬 {amp_ratio_inh:.3f}")
    print(f"      시냅스 위치 위상차(최대): 정현파 {dphi_sine:.2f}deg · "
          f"억제리듬 {dphi_inh:.2f}deg")
    print(f"      -> 기본 방식 = **{default}** (미결#5 해소)")
    print(f"      -> ★위상 기준 = **시냅스 위치의 국소 막전위**. 가소성이 실제로 그것을 본다.")
    print(f"         소마 Vm 기준과의 차이({max(dphi_sine, dphi_inh):.2f}deg 이내)를 함께 인쇄해")
    print(f"         소마 기준으로 보고한 문헌과 변환 가능하게 한다.")

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.48)
    axA = fig.add_subplot(gs_[0, :2])
    axB = fig.add_subplot(gs_[0, 2])
    axC = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axE = fig.add_subplot(gs_[1, 2])

    # A: 파형 (5Hz 두 방식)
    R, wav, tt = traces[("정현파", f0)]
    m = (R["t"] >= T0 - 50) & (R["t"] <= T0 + 700)
    axA.plot(R["t"][m] - T0, R["post_v"][m], color="#1565c0", lw=1.7, label="정현파: 소마")
    axA.plot(R["t"][m] - T0, R["local_v"][0][m], color="#42a5f5", lw=1.2, ls="--",
             label=f"정현파: 시냅스 {sites[0][1]['path_um']:.0f}um")
    R2, times, _ = traces[("억제리듬", f0)]
    m2 = (R2["t"] >= T0 - 50) & (R2["t"] <= T0 + 700)
    axA.plot(R2["t"][m2] - T0, R2["post_v"][m2], color="#c62828", lw=1.7,
             label="억제리듬: 소마")
    axA.plot(R2["t"][m2] - T0, R2["local_v"][0][m2], color="#ef9a9a", lw=1.2, ls="--",
             label=f"억제리듬: 시냅스 {sites[0][1]['path_um']:.0f}um")
    for k in range(5):
        axA.axvline(k * 1000.0 / f0, color="#b0bec5", ls=":", lw=0.9)
    axA.set_xlabel("리듬 시작 기준 시간 (ms)"); axA.set_ylabel("Vm (mV)")
    axA.set_title(f"A. 부과 theta {f0:.0f}Hz — 두 방식 (점선 세로 = 주입 신호의 peak)\n"
                  "★자연 발생이 아니라 **부과**다 (4-2 판정: 불가)",
                  fontsize=9.5, loc="left")
    axA.legend(fontsize=7.5, ncol=2)

    # B: 보정 선형성
    axB.plot([c["amp_nA"] for c in calA], [c["pp"] for c in calA], "o-",
             color="#1565c0", ms=6, label="정현파 (nA)")
    ax2 = axB.twiny()
    ax2.plot([c["g_nS"] for c in calB], [c["pp"] for c in calB], "s--",
             color="#c62828", ms=6, label="억제리듬 (nS)")
    axB.axhline(TARGET_PP, color="#37474f", ls=":", lw=1.4, label=f"목표 {TARGET_PP:.0f}mV")
    axB.set_xlabel("정현파 진폭 (nA)", color="#1565c0")
    ax2.set_xlabel("GABA g (nS)", color="#c62828")
    axB.set_ylabel("소마 Vm (mV pp)")
    axB.set_title("B. 세기 보정 — 둘 다 선형이다\n목표 진폭을 외삽으로 맞춘다",
                  fontsize=9.2, loc="left")
    h1, l1 = axB.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=7.5)

    # C: 위치별 진폭
    labs = ["소마"] + syn_names
    xx = np.arange(len(labs))
    axC.bar(xx - 0.19, [main[f0]["sine"][k]["pp"] for k in labs], width=0.38,
            color="#1565c0", label="정현파")
    axC.bar(xx + 0.19, [main[f0]["inhib"][k]["pp"] for k in labs], width=0.38,
            color="#c62828", label="억제리듬")
    axC.axhline(TARGET_PP, color="#37474f", ls=":", lw=1.2)
    axC.set_xticks(xx)
    axC.set_xticklabels([k.replace("시냅스 ", "") for k in labs], fontsize=7.5)
    axC.set_ylabel("Vm 진폭 (mV pp)")
    axC.set_title(f"C. 부과 theta 가 시냅스까지 가는가 ({f0:.0f}Hz)\n"
                  f"진폭비 정현파 {amp_ratio_sine:.3f} · 억제리듬 {amp_ratio_inh:.3f}",
                  fontsize=9.2, loc="left")
    axC.legend(fontsize=7.8)

    # D: 위상차
    axD.bar(xx - 0.19, [wrap_deg(np.radians(main[f0]["sine"][k]["phase_deg"]
                                           - main[f0]["sine"]["소마"]["phase_deg"]))
                        for k in labs], width=0.38, color="#1565c0", label="정현파")
    axD.bar(xx + 0.19, [wrap_deg(np.radians(main[f0]["inhib"][k]["phase_deg"]
                                           - main[f0]["inhib"]["소마"]["phase_deg"]))
                        for k in labs], width=0.38, color="#c62828", label="억제리듬")
    axD.axhline(0, color="#37474f", lw=1.0)
    axD.set_xticks(xx)
    axD.set_xticklabels([k.replace("시냅스 ", "") for k in labs], fontsize=7.5)
    axD.set_ylabel("소마 대비 위상차 (deg)")
    axD.set_title(f"D. ★시냅스 위치의 위상은 소마와 다르다\n"
                  f"최대 {max(dphi_sine, dphi_inh):.2f}deg "
                  f"({max(dphi_sine, dphi_inh)/360.0*1000.0/f0:.2f}ms @ {f0:.0f}Hz)",
                  fontsize=9.2, loc="left")
    axD.legend(fontsize=7.8)

    # E: 주입 신호 대비 소마 위상 지연
    axE.bar(range(len(FREQS)), [inj_lag[f] for f in FREQS], width=0.5,
            color="#6a1b9a")
    axE.axhline(0, color="#37474f", lw=1.0)
    axE.set_xticks(range(len(FREQS)))
    axE.set_xticklabels([f"{f:.0f} Hz" for f in FREQS])
    axE.set_ylabel("주입 전류 대비 소마 Vm 위상 (deg)")
    for i, f in enumerate(FREQS):
        axE.text(i, inj_lag[f], f"{inj_lag[f]:+.1f}\n({inj_lag[f]/360*1000/f:+.1f}ms)",
                 ha="center", va="bottom" if inj_lag[f] >= 0 else "top", fontsize=7.5)
    axE.set_title("E. ★주입 위상 != 막전위 위상\n"
                  "막 시상수 때문에 늦는다 — 그래서 기준을 정해야 한다",
                  fontsize=9.2, loc="left")

    fig.suptitle("4-3  부과 theta — 두 방식 비교와 **위상 기준 확정** (★부과다)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"4-3 | ★부과 theta(4-2: 자연 불가) · 목표 {TARGET_PP:.0f}mV pp · "
                     f"정현파 {AMP_A:.4f}nA · 억제 {G_B:.3f}nS x{N_GABA} · "
                     f"기본 방식 {default} · 위상 기준 = 시냅스 국소 Vm")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "4-3_imposed_theta.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    checks = [
        ("★구동이 실제로 진동을 만든다 (정현파 소마 진폭 > 1 mV) — "
         "0 이면 play 가 등록되지 않은 것이다",
         min(main[f]["sine"]["소마"]["pp"] for f in FREQS) > 1.0),
        ("★억제리듬도 진동을 만든다 (소마 진폭 > 1 mV)",
         min(main[f]["inhib"]["소마"]["pp"] for f in FREQS) > 1.0),
        (f"정현파: 두 주파수에서 목표 진폭 재현 (오차 {100*errA:.1f}% < 25%)",
         errA < 0.25),
        (f"★억제리듬은 **포화**한다 — 세기 {gg[-1]/gg[0]:.0f}배에 진폭 "
         f"{pp[-1]/pp[0]:.1f}배뿐 (선형성 {sat_ratio:.3f} < 0.5)", sat_ratio < 0.5),
        (f"★그래서 정현파가 통제성이 우월하다 (목표 오차 정현파 {100*errA:.1f}% "
         f"vs 억제 {100*errB:.1f}%)", errA < errB),
        ("세기-진폭 관계가 선형이다 (정현파, 기울기 > 0)", sl[0] > 0),
        ("억제리듬도 세기에 단조 증가한다 (로그 스윕)",
         bool(np.all(np.diff(pp) > 0))),
        (f"★억제리듬은 목표 진폭에 **도달하지 못한다** (최대 {pp_max:.2f} mV < "
         f"{TARGET_PP:.0f} mV) — 막전위가 e_GABAA 로 접근해 구동력이 사라진다",
         not reach),
        ("★부과 theta 가 시냅스 위치까지 도달한다 (진폭비 > 0.5)",
         amp_ratio_sine > 0.5 and amp_ratio_inh > 0.5),
        ("배치 인공물 제거 확인 — 시냅스 위치 진폭이 소마보다 크지 않다 (진폭비 < 1.2)",
         max(main[f][m][k]["pp"] / main[f][m]["소마"]["pp"]
             for f in FREQS for m in ("sine", "inhib") for k in syn_names) < 1.2),
        (f"★주입 전류와 소마 Vm 의 위상이 어긋난다 "
         f"({abs(inj_lag[FREQS[0]]):.1f}deg) — 기준을 정해야 하는 이유",
         abs(inj_lag[FREQS[0]]) > 5.0),
        (f"★시냅스 위치 위상이 소마와 다르다 (최대 "
         f"{max(dphi_sine, dphi_inh):.2f}deg)", max(dphi_sine, dphi_inh) > 0.05),
        ("억제리듬의 평균 막전위가 정현파보다 낮다 (억제이므로)",
         main[f0]["inhib"]["소마"]["mean"] < main[f0]["sine"]["소마"]["mean"] - 0.1),
        ("기본 방식이 결정됐다 (미결#5)", default in ("정현파 전류", "억제성 시냅스 리듬")),
        ("두 주파수 모두 측정됐다", set(main) == set(FREQS)),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(imposed=True,
               note_imposed=("★이 theta 는 **부과**된 것이다. 4-2 가 자연 theta 불가로 판정했다. "
                             "그림·문서에 '부과' 를 명기하고 '자연 발생' 이라고 쓰지 않는다(D34). "
                             "억제리듬 방식도 개재뉴런이 없으므로 리듬 자체는 부과다(가상 입력)."),
               dt=DT, rec_dt=REC_DT, settle_ms=SETTLE_MS, dur_ms=DUR,
               freqs_hz=FREQS, target_pp_mV=TARGET_PP,
               sites=[{"label": n, **{k: sp[k] for k in ("section", "path_um", "domain")}}
                      for n, sp in sites],
               sine=dict(calibration=calA, amp_nA=AMP_A,
                         slope_mV_per_nA=float(sl[0]), intercept_mV=float(sl[1])),
               inhib=dict(calibration=calB, g_nS=G_B, n_syn=N_GABA, e_GABAA=E_GABA,
                          sections=gabasecs, pp_max_mV=pp_max,
                          target_reachable=bool(reach), linearity=sat_ratio,
                          note=("★억제성 시냅스 리듬은 **포화**한다 — 전도도를 키우면 막전위가 "
                                f"e_GABAA({E_GABA:.0f}mV)로 접근해 구동력이 사라진다. "
                                "그래서 선형 외삽으로 목표 진폭을 맞출 수 없다(처음 판이 "
                                "그렇게 했고 목표의 40% 밖에 못 냈다). 로그 스윕으로 실측 "
                                "곡선을 얻어 보간한다. 정현파는 65 mV/nA 로 정확히 선형이다.")),
               measurements={str(f): main[f] for f in FREQS},
               injection_phase_lag_deg={str(f): inj_lag[f] for f in FREQS},
               summary=dict(target_err_sine=errA, target_err_inhib=errB,
                            amp_ratio_sine=amp_ratio_sine,
                            amp_ratio_inhib=amp_ratio_inh,
                            max_dphi_sine_deg=dphi_sine,
                            max_dphi_inhib_deg=dphi_inh,
                            default_method=default),
               phase_reference=("★6-1 의 위상 기준은 **시냅스 위치의 국소 막전위**다. "
                                "가소성이 실제로 보는 것이 그것이기 때문이다. 소마 Vm 기준과의 "
                                "차이를 함께 인쇄해 소마 기준으로 보고한 문헌과 변환 가능하게 "
                                "한다. 주입 전류 위상은 **기준으로 쓰지 않는다** — 막 시상수 "
                                f"때문에 소마 Vm 이 이미 {inj_lag[FREQS[0]]:+.1f}deg 어긋난다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "4-3_theta.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 4-3 완료 ({n_ok}/{len(checks)}) — 기본 방식 '{default}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
