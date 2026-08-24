# -*- coding: utf-8 -*-
"""3-7 전도도 확정 — g 스윕으로 민감도 측정 + 기준 uEPSP 확정

단계   : 3-7 (파이프라인 3단계 시냅스 / 하위 7 calibrate)
쉬운 설명: 시냅스 세기(전도도 g)를 바꾸면 post EPSP 가 얼마나 커지는지 재고, 논문 실측 범위가
          g 축의 어디에 해당하는지 표시한다. 그래서 우리가 쓰는 g 하나를 근거와 함께 확정한다.
방법   : g 를 스윕하며 pre 1발 -> post 소마 EPSP 진폭 측정(lib.measure 통일).
          정착(SETTLE_MS)은 1회만 하고 스냅샷 복원으로 각 g 를 돌린다 — gmax 는 파라미터라
          정착 상태와 무관하므로 이렇게 해도 동일하고 훨씬 빠르다.
★입장  : 이 단계는 '재보정'이 아니라 '확정'이다.
          g = 0.6 nS 는 Ecker2020 Table3 PC->PC(E2) 의 논문값이고, 그 값에서 나온 uEPSP 가
          이미 Deuchars&Thomson 1996 실측 범위(0.17~1.5mV) 안이다. 평균(0.7mV)에 정확히
          맞추려고 g 를 올리면 논문값이 튜닝값으로 바뀌므로 하지 않는다(D9 원칙: 튜닝값 0개).
          대신 '평균에 맞추려면 g 가 얼마여야 하는가'를 스윕에서 읽어 함께 기록한다.
          ⚠️ 이전 계획의 '진폭이 논문 평균의 10.2배라 g 를 낮춘다'는 전제는 폐기됐다 —
             그 비교는 Sayer1990(CA3→CA1 = Schaffer collateral)과 한 것이었다(D9·D10).
근거   : Ecker2020 Table3 (g_hat 0.6nS) · Deuchars&Thomson 1996 PMID 8895869 (uEPSP 실측)
결과   : figures/3-7_g_sweep.png · figures/3-7_calibrate.json
실행   : . .\\env\\activate.ps1 ; & $Py04 03_synapse\\7_calibrate\\3-7_calibrate_g.py
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

import numpy as np                          # noqa: E402
from lib import plots                        # noqa: E402
from lib import measure                       # noqa: E402
from lib import refdata                       # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.wiring import Wiring, SETTLE_MS      # noqa: E402

T_SPIKE = SETTLE_MS + 10.0
TSTOP = T_SPIKE + 70.0
REC_DT = 0.025
G_SWEEP = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.7, 2.0]  # nS (0.6 = 논문값)
# 국소 스파이크 판정은 lib.measure.is_dendritic_spike (국소 최고 전압 문턱) 하나로 통일한다
D = refdata.DEUCHARS1996


def main():
    plots.setup()
    print("=== 3-7 전도도 확정 (g 스윕) ===")
    b = Bench()
    w = Wiring(b, frozen=True)
    g_cfg = float(w.p["g_nS"])
    print(f"  클래스 {w.class_name} · config g = {g_cfg} nS · 시냅스 {b.n_syn()}개")
    print(f"  기준 문헌 {D['src']}")
    print(f"    진폭 {D['amp_mV']['mean']}±{D['amp_mV']['sd']} mV "
          f"(범위 {D['amp_mV']['min']}~{D['amp_mV']['max']})")

    w.drive_pre_iclamp([T_SPIKE], amp_nA=1.2, dur_ms=3.0)
    w.record(rec_dt=REC_DT, local_v=True, currents=False)   # 국소 수상돌기 전압으로 스파이크 판별
    t_event = T_SPIKE + b.syn_specs[0]["delay_ms"]

    # 정착 1회 -> 각 g 마다 복원해서 실행.
    # ⚠️ gmax 설정은 반드시 restore() 뒤에. SaveState 는 파라미터도 되돌린다.
    w.settle()
    amps, traces, vlocs, dsp = [], [], [], []
    for g in G_SWEEP:
        w.restore()
        for syn, _ in w.syns:
            syn.gmax = g / 1000.0          # nS -> uS
        w.run_settled(TSTOP)
        R = w.arrays()
        f = measure.epsp_features(R["t"], R["post_v"], t_event)
        vloc_pk = max(float(lv.max()) for lv in R["local_v"])      # 국소 최고 전압(절대, mV)
        dspike = measure.is_dendritic_spike(vloc_pk)                # 국소 최고 전압 문턱(공용)
        amps.append(f["amp_mV"]); vlocs.append(vloc_pk); dsp.append(dspike)
        traces.append((g, R["t"].copy(), R["post_v"].copy(), f, dspike))
        print(f"  g={g:5.2f} nS -> soma EPSP {f['amp_mV']:7.4f} mV "
              f"(상승 {f['rise_ms']:.2f} · 반치폭 {f['halfwidth_ms']:5.2f} ms) "
              f"· 국소 최고 {vloc_pk:7.2f} mV{'  <-- 국소 스파이크' if dspike else ''}")
    amps = np.array(amps); gs = np.array(G_SWEEP)
    vlocs = np.array(vlocs); dsp = np.array(dsp, dtype=bool)
    # 수동 EPSP 구간(국소 스파이크 없는 g)만 보정·선형성 판단에 쓴다
    pas = ~dsp
    g_dsp_thr = float(gs[dsp].min()) if dsp.any() else None
    print(f"\n  수동 EPSP 구간: g <= {gs[pas].max():.2f} nS ({int(pas.sum())}/{len(gs)}점)"
          + (f" · 국소 스파이크 개시 g >= {g_dsp_thr:.2f} nS" if g_dsp_thr
             else " · 국소 스파이크 없음"))

    # config g 에서의 진폭 (스윕에 포함돼 있어야 한다)
    i_cfg = int(np.argmin(np.abs(gs - g_cfg)))
    amp_cfg = float(amps[i_cfg])

    # 논문 평균/최소/최대에 해당하는 g — ★수동 구간에서만 보간한다.
    #   국소 스파이크 구간을 섞으면 진폭이 단조가 아니어서 보간이 무의미해진다.
    gp, ap_ = gs[pas], amps[pas]

    def g_for(target):
        if target < ap_.min() or target > ap_.max():
            return None
        return float(np.interp(target, ap_, gp))
    g_mean = g_for(D["amp_mV"]["mean"])
    g_min = g_for(D["amp_mV"]["min"])
    g_max = g_for(D["amp_mV"]["max"])

    # 선형성: 원점 통과 선형 적합 대비 편차 (수동 구간만; NMDA 전압의존·구동력 감소로 포화)
    slope = float(np.sum(gp * ap_) / np.sum(gp * gp))        # 최소제곱, 절편 0
    lin_p = slope * gp
    nonlin = float(np.max(np.abs(ap_ - lin_p) / np.maximum(lin_p, 1e-9)) * 100)

    # ★ 수동 EPSP 천장: 국소 스파이크가 나기 전에 낼 수 있는 최대 soma EPSP
    amp_ceiling = float(ap_.max())
    ceiling_vs_paper_max = amp_ceiling / D["amp_mV"]["max"]

    print(f"  config g={g_cfg}nS 진폭 {amp_cfg:.4f} mV = 논문 평균의 {amp_cfg/D['amp_mV']['mean']:.2f}배")
    print(f"  논문 하한 {D['amp_mV']['min']}mV <- g={g_min:.2f}nS" if g_min else
          "  논문 하한 보간 불가")
    print(f"  논문 평균 {D['amp_mV']['mean']}mV <- g={g_mean:.2f}nS (미채택)" if g_mean else
          "  논문 평균 보간 불가")
    if g_max is None:
        print(f"  ★ 논문 상한 {D['amp_mV']['max']}mV 는 수동 EPSP 로 도달 불가 — "
              f"천장 {amp_ceiling:.3f}mV (논문 상한의 {ceiling_vs_paper_max:.2f}배)에서 "
              f"국소 스파이크가 먼저 난다")
    else:
        print(f"  논문 상한 {D['amp_mV']['max']}mV <- g={g_max:.2f}nS")
    print(f"  수동 구간 원점통과 기울기 {slope:.4f} mV/nS · 최대 비선형 편차 {nonlin:.1f}%")
    print(f"  채택 g 대비 스파이크 여유: {g_dsp_thr/g_cfg:.2f}배 "
          f"(효능이 이 배수를 넘으면 정성적으로 다른 영역)" if g_dsp_thr else "")

    import matplotlib.pyplot as plt
    fig, (aA, aB) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                 gridspec_kw={"width_ratios": [1.05, 1.0]})
    cmap = plt.get_cmap("viridis")

    # ---- A: g 별 EPSP 파형 ----
    for k, (g, tt, vv, f, is_ds) in enumerate(traces):
        col = "#c62828" if is_ds else cmap(0.85 * k / max(len(traces) - 1, 1))
        me = abs(g - g_cfg) < 1e-9
        aA.plot(tt - T_SPIKE, vv, color=col, lw=(2.8 if me else 1.3),
                ls=("--" if is_ds else "-"), zorder=(5 if me else 2),
                label=f"{g:g} nS" + (" <- 논문값" if me else ("  (국소 스파이크)" if is_ds else "")))
    aA.set_xlim(-5, 60)
    aA.set_xlabel("pre 발화 후 시간 (ms)"); aA.set_ylabel("post 소마 Vm (mV)")
    aA.set_title("A. g 별 uEPSP 파형 — 굵은 선이 논문값 0.6nS\n"
                 "빨간 점선은 수동 EPSP 가 아니라 국소(수상돌기) 스파이크가 난 경우",
                 fontsize=10, loc="left")
    aA.legend(fontsize=7, ncol=2, loc="upper right")

    # ---- B: 진폭 vs g + 논문 범위 ----
    aB.axhspan(D["amp_mV"]["min"], D["amp_mV"]["max"], color="#ffb300", alpha=0.16, zorder=0,
               label=f"Deuchars1996 범위 {D['amp_mV']['min']}~{D['amp_mV']['max']} mV")
    aB.axhline(D["amp_mV"]["mean"], color="#ef6c00", ls="-", lw=1.4, zorder=1,
               label=f"논문 평균 {D['amp_mV']['mean']} mV")
    if g_dsp_thr:
        aB.axvspan(g_dsp_thr, gs.max() * 1.03, color="#c62828", alpha=0.08, zorder=0)
        aB.text(g_dsp_thr * 1.02, ap_.max() * 1.06,
                f" 국소 스파이크 구간 (g >= {g_dsp_thr:.1f} nS)\n"
                f" 진폭 {amps[dsp].min():.1f}~{amps[dsp].max():.1f} mV — y축 밖\n"
                f" 국소 전압 {vlocs[dsp].min():.0f}~{vlocs[dsp].max():.0f} mV 로 발화",
                fontsize=7.5, color="#c62828", va="bottom")
    aB.plot(gp, lin_p, ls=":", color="#9e9e9e", lw=1.2, label=f"원점통과 선형 ({slope:.3f} mV/nS)")
    aB.plot(gp, ap_, "-o", color="#6a1b9a", ms=5, lw=1.8, label="실측 uEPSP (수동 구간)")
    if dsp.any():
        aB.plot(gs[dsp], np.full(dsp.sum(), ap_.max() * 1.02), "x",
                color="#c62828", ms=9, mew=2,
                label="국소 스파이크 (EPSP 아님 · 진폭은 y축 밖)")
    aB.plot([g_cfg], [amp_cfg], "*", color="#c62828", ms=22, zorder=6,
            label=f"채택 g={g_cfg}nS -> {amp_cfg:.3f} mV")
    if g_mean:
        aB.plot([g_mean], [D["amp_mV"]["mean"]], "D", color="#00695c", ms=8, zorder=6,
                label=f"평균 정합 g~{g_mean:.2f}nS (미채택)")
        aB.vlines(g_mean, 0, D["amp_mV"]["mean"], color="#00695c", ls="--", lw=1.0, alpha=0.6)
    aB.set_xlabel("시냅스 전도도 g (nS, 시냅스당)")
    aB.set_ylabel("post 소마 uEPSP 진폭 (mV)")
    aB.set_title(f"B. 전도도 민감도 — 논문값 g={g_cfg}nS 는 실측 범위 안(평균의 "
                 f"{amp_cfg/D['amp_mV']['mean']:.2f}배)\n수동 구간 비선형 편차 최대 {nonlin:.1f}%",
                 fontsize=10, loc="left")
    aB.legend(fontsize=7.5, loc="upper left")
    aB.set_ylim(0, max(ap_.max(), D["amp_mV"]["max"]) * 1.22)

    fig.suptitle(f"3-7  전도도 확정 — {w.class_name} · 시냅스 {b.n_syn()}개 · "
                 f"g 는 Ecker2020 Table3 논문값을 유지(튜닝하지 않음)", fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.83, wspace=0.24)
    _gmax_s = f"{g_max:.2f}" if g_max else "도달불가"
    plots.stamp(fig, f"3-7 | 정착 {SETTLE_MS:.0f}ms · 채택 g={g_cfg}nS(논문값·미튜닝) -> {amp_cfg:.3f}mV · "
                     f"논문하한 g={g_min:.2f} · 평균정합 g~{g_mean:.2f}(기록만) · 상한 g={_gmax_s} · "
                     f"국소 스파이크 개시 g>={g_dsp_thr}nS")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-7_g_sweep.png")

    checks = [
        ("수동 구간에서 g 증가에 EPSP 단조 증가", bool(np.all(np.diff(ap_) > 0))),
        ("config g 가 스윕에 포함", abs(gs[i_cfg] - g_cfg) < 1e-9),
        ("채택 g 가 수동 구간 안 (국소 스파이크 아님)", bool(pas[i_cfg])),
        ("논문값 g 의 진폭이 실측 범위 안",
         D["amp_mV"]["min"] <= amp_cfg <= D["amp_mV"]["max"]),
        ("논문 평균에 대응하는 g 가 수동 구간 안에서 결정됨", g_mean is not None),
        ("포화(비선형) 확인 — 선형 대비 편차 > 1%", nonlin > 1.0),
        ("국소 스파이크 개시 g 가 채택 g 보다 큼 (기준선이 수동 영역)",
         (g_dsp_thr is None) or (g_dsp_thr > g_cfg)),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, ok in checks if ok)

    out = dict(cls=w.class_name, n_syn=b.n_syn(), settle_ms=SETTLE_MS,
               g_sweep_nS=[float(x) for x in gs],
               amp_mV=[round(float(x), 4) for x in amps],
               local_v_peak_mV=[round(float(x), 2) for x in vlocs],
               is_dendritic_spike=[bool(x) for x in dsp],
               dspike_vloc_threshold_mV=measure.DSPIKE_VLOC_MV,
               g_dendritic_spike_onset_nS=g_dsp_thr,
               g_adopted_nS=g_cfg, amp_at_adopted_mV=round(amp_cfg, 4),
               amp_ratio_to_paper_mean=round(amp_cfg / D["amp_mV"]["mean"], 3),
               g_for_paper_mean_nS=(round(g_mean, 3) if g_mean else None),
               g_for_paper_min_nS=(round(g_min, 3) if g_min else None),
               g_for_paper_max_nS=(round(g_max, 3) if g_max else None),
               passive_ceiling_mV=round(amp_ceiling, 4),
               passive_ceiling_vs_paper_max=round(ceiling_vs_paper_max, 3),
               spike_headroom_x=(round(g_dsp_thr / g_cfg, 2) if g_dsp_thr else None),
               linear_slope_mV_per_nS=round(slope, 4),
               max_nonlinearity_pct=round(nonlin, 2),
               decision=("g = Ecker2020 Table3 논문값 0.6nS 유지(튜닝 안 함). "
                         "진폭이 Deuchars1996 실측 범위 안이므로 조정 근거 없음. "
                         "평균 정합 g 는 기록만 하고 채택하지 않는다. "
                         "★ g>=1.1nS 에서 기저수상돌기 국소 스파이크가 나므로 수동 EPSP 천장은 "
                         "약 0.89mV 다 — 논문 상한 1.5mV 는 이 기하에서 수동으로 도달 불가."),
               reference=D["src"],
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "3-7_calibrate.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 3-7 완료 ({n_ok}/{len(checks)}) — g={g_cfg}nS 확정(논문값 유지)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
