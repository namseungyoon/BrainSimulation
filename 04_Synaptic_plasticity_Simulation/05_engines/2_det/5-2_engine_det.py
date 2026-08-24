# -*- coding: utf-8 -*-
"""5-2 엔진 det — 전달만 하는 기준선 (가소성이 정말 꺼져 있는가)

단계   : 5-2 (5단계 가소성 엔진 / 하위 2 det)
쉬운 설명: 가소성 엔진을 비교하려면 **아무것도 변하지 않는 기준선**이 먼저 있어야 한다.
          "이 엔진이 효능을 20% 올렸다" 는 말은 "안 올리는 것" 과 비교해서만 뜻이 있다.
          여기서 그 기준선 둘을 만들고 **정말 안 변하는지** 수치로 확인한다.
방법   : 기준선 후보 두 가지를 같은 자극으로 돌린다.
          (a) DetAMPANMDA        — 장기가소성이 **아예 없는** mod (단기가소성만)
          (b) GBPlasticitySyn 동결 — gamma_p=gamma_d=0 으로 장기항을 끈 것 (3단계가 쓴 것)
          자극: 단발 · 짝 · 버스트 · 장시간 트레인(효능 표류를 보려고)
검증   : (1) 두 기준선 모두 프로토콜 전체에서 효능이 불변인가
          (2) 동결이 **자율항까지** 껐는가 — GB 의 자율항 -rho(1-rho)(rho*-rho) 는
              gamma 를 0 으로 해도 살아 있어 rho0 가 0/1 이 아니면 표류한다
          (3) 단기가소성 유무가 두 기준선의 차이를 설명하는가
근거   : Graupner & Brunel 2012 (자율항) · docs/DECISIONS.md 미결#3(동결의 의미)
결과   : figures/5-2_engine_det.png · figures/5-2_det.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\2_det\\5-2_engine_det.py
비고   : lib/synprobe.py (단일 구획 프로브) 사용 — 두 세포 벤치 불필요, 초 단위.
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
from lib.synprobe import SynProbe, CAPS              # noqa: E402
from lib.refs import gb                              # noqa: E402
from lib.wiring import load_synapse_cfg              # noqa: E402

REC_DT = 0.025
V_HOLD = -70.0

# ★ 자극은 t>0 에 둔다. GB 계열 mod 는 전달 가중치 w 를 INITIAL 에서 초기화하지 않으므로
#   t=0 스파이크는 w=0 으로 전달되어 전도도가 조용히 0 이 된다(5-2 에서 실측·아래 [함정]).
T0 = 20.0
PROTOCOLS = [
    ("단발", [T0], [], 400.0),
    ("짝 dt=+10ms", [T0], [T0 + 10.0], 400.0),
    ("버스트 4발 100Hz + post 1발",
     [T0, T0 + 10.0, T0 + 20.0, T0 + 30.0], [T0 + 5.0], 600.0),
    ("트레인 60발 5Hz (12초)", list(T0 + np.arange(60) * 200.0), [], 13000.0),
]


def main():
    plots.setup()
    print("=== 5-2 엔진 det (전달만 = 기준선) ===")
    cls, P = load_synapse_cfg()
    print(f"  시냅스 클래스 {cls} · g {P['g_nS']}nS · e_rev {P['e_rev_mV']}mV")

    def make(mech, rho0=0.0, frozen=True):
        p = SynProbe(mech, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        p.set(e=P["e_rev_mV"],
              tau_r_AMPA=P["tau_r_AMPA"], tau_d_AMPA=P["tau_d_AMPA"],
              tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
              NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"])
        if CAPS[mech]["ltp"]:
            p.set(rho0=rho0)
            if frozen:
                p.set(gamma_p=0.0, gamma_d=0.0)
        if CAPS[mech]["stp"]:
            p.set(Use=P["Use"], Dep=P["Dep_ms"], Fac=P["Fac_ms"])
        if CAPS[mech]["prob"]:
            p.seed(1, 2, 3)
        return p

    ENGINES = [("DetAMPANMDA", "장기가소성 없음 (단기만)"),
               ("GBPlasticitySyn", "GB 동결 (gamma=0) — 3단계가 쓴 것")]

    results = {}
    print()
    for mech, desc in ENGINES:
        print(f"  [{mech}] {desc}")
        rows = []
        for name, pre, post, tstop in PROTOCOLS:
            p = make(mech)
            p.drive_pre(pre)
            if post and CAPS[mech]["post_nc"]:
                p.drive_post(post)
            R = p.run(tstop)
            gpk = float(R["g"].max()) * 1e3                    # uS -> nS
            # 펄스별 g 봉우리 (단기가소성 프로파일)
            # ★ 절대 봉우리와 '증분' 을 구분해 잰다.
            #   tau_d_NMDA = 148.5ms 이므로 100Hz 에서는 전도도가 누적된다. 단기가소성이
            #   전혀 없어도 절대 봉우리는 커진다 -> 단기가소성 판정은 **증분**으로 해야 한다.
            peaks, jumps = [], []
            for k, ts in enumerate(pre):
                te = pre[k + 1] if k + 1 < len(pre) else tstop
                m = (R["t"] >= ts) & (R["t"] < te)
                if m.any():
                    peaks.append(float(R["g"][m].max()) * 1e3)
                    mj = (R["t"] >= ts) & (R["t"] < min(ts + 3.0, te))
                    base = float(R["g"][R["t"] <= ts][-1]) if (R["t"] <= ts).any() else 0.0
                    jumps.append((float(R["g"][mj].max()) - base) * 1e3)
            rho = R.get("rho")
            drho = (float(rho[-1] - rho[0]) if rho is not None else 0.0)
            rows.append(dict(name=name, g_peak_nS=gpk, peaks_nS=peaks,
                             jumps_nS=jumps,
                             rho0=(float(rho[0]) if rho is not None else None),
                             rho_end=(float(rho[-1]) if rho is not None else None),
                             drho=drho,
                             c_max=(float(R["c"].max()) if "c" in R else None),
                             R=R))
            extra = (f" · rho {rho[0]:.6f} -> {rho[-1]:.6f} (변화 {drho:+.2e})"
                     if rho is not None else " · rho 없음 (구조적으로 불변)")
            ppr = (peaks[1] / peaks[0]) if len(peaks) > 1 else float("nan")
            print(f"      {name:<28} g 최고 {gpk:7.4f} nS" +
                  (f" · PPR {ppr:.3f}" if len(peaks) > 1 else "          ") + extra)
        results[mech] = rows

    # ── 동결이 자율항까지 껐는가 (미결#3) ────────────────────────────────
    print(f"\n  [동결의 의미] GB 자율항은 gamma=0 이어도 살아 있다 — rho0 에 따라 표류한다")
    drift = []
    for r0 in (0.0, 0.1, 0.5, 0.9, 1.0):
        p = make("GBPlasticitySyn", rho0=r0)
        p.drive_pre([T0])
        R = p.run(13000.0)                       # 13 초
        d = float(R["rho"][-1] - R["rho"][0])
        drift.append(dict(rho0=r0, drho_13s=d))
        print(f"      rho0 {r0:.1f} -> 13초 뒤 변화 {d:+.3e} "
              f"({'불변' if abs(d) < 1e-9 else '표류'})")
    # 참조로 같은 것을 예측 (자율항만, 자극 없음)
    tR = np.arange(0.0, 13000.0, 1.0)
    pr = dict(gb.WITTENBERG2006); pr["gamma_p"] = 0.0; pr["gamma_d"] = 0.0
    ref_drift = []
    for r0 in (0.0, 0.1, 0.5, 0.9, 1.0):
        rr = gb.integrate_rho(tR, np.zeros_like(tR), rho0=r0, p=pr)
        ref_drift.append(float(rr[-1] - rr[0]))
    max_ref_err = max(abs(a["drho_13s"] - b) for a, b in zip(drift, ref_drift))
    print(f"      참조(numpy)와 최대 절대차 {max_ref_err:.2e}")

    # ── [함정] t=0 스파이크는 GB 계열에서 조용히 사라진다 ──────────────────
    print("\n  [함정] GB 계열은 INITIAL 에서 전달 가중치 w 를 초기화하지 않는다")
    trap = {}
    for mech in ("DetAMPANMDA", "GBPlasticitySyn"):
        p = make(mech); p.drive_pre([0.0], allow_t0=True)
        g0 = float(p.run(400.0)["g"].max()) * 1e3
        p = make(mech); p.drive_pre([T0])
        gT = float(p.run(400.0)["g"].max()) * 1e3
        trap[mech] = dict(g_at_t0_nS=g0, g_at_t20_nS=gT)
        print(f"      {mech:<20} t=0 스파이크 g {g0:.4f} nS · t={T0:.0f}ms g {gT:.4f} nS"
              + ("   <- t=0 에서 전달 소실" if g0 < 1e-9 < gT else ""))

    # ── 두 기준선의 차이는 단기가소성인가 ────────────────────────────────
    det_b = results["DetAMPANMDA"][2]["jumps_nS"]
    gb_b = results["GBPlasticitySyn"][2]["jumps_nS"]
    det_pk = results["DetAMPANMDA"][2]["peaks_nS"]
    gb_pk = results["GBPlasticitySyn"][2]["peaks_nS"]
    print("\n  [차이의 원인] 버스트 4발 — 펄스별 **증분** (nS)")
    print("      DetAMPANMDA (단기 O): " + " ".join(f"{v:.4f}" for v in det_b))
    print("      GB 동결      (단기 X): " + " ".join(f"{v:.4f}" for v in gb_b))
    print("      (참고) 절대 봉우리 — GB 도 커진다: " +
          " ".join(f"{v:.3f}" for v in gb_pk) + "  <- NMDA 꼬리 누적")
    det_prof = np.array(det_b) / det_b[0]
    gb_prof = np.array(gb_b) / gb_b[0]
    print(f"      정규화 증분 4번째: Det {det_prof[-1]:.3f} (억압) vs GB {gb_prof[-1]:.3f}")
    gb_flat = float(np.max(np.abs(gb_prof - 1.0)))
    first_ratio = det_b[0] / gb_b[0]

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.0, 8.4))
    gs_ = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.44)
    axA = fig.add_subplot(gs_[0, :2])
    axB = fig.add_subplot(gs_[0, 2])
    axC = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axE = fig.add_subplot(gs_[1, 2])

    # A: 버스트 전도도 파형 비교
    for (mech, _), col in zip(ENGINES, ["#1565c0", "#c62828"]):
        R = results[mech][2]["R"]
        axA.plot(R["t"], R["g"] * 1e3, lw=1.7, color=col,
                 label=f"{mech} (최고 {results[mech][2]['g_peak_nS']:.3f} nS)")
    for ts in PROTOCOLS[2][1]:
        axA.axvline(ts, color="#b0bec5", ls=":", lw=0.9)
    axA.set_xlim(-5, 120); axA.set_xlabel("시간 (ms)"); axA.set_ylabel("시냅스 g (nS)")
    axA.set_title("A. 버스트 4발 100Hz — 두 기준선의 전달 파형\n"
                  "점선 = 전시냅스 스파이크. 둘 다 장기 효능은 변하지 않는다",
                  fontsize=9.5, loc="left")
    axA.legend(fontsize=8.5)

    # B: 펄스별 정규화 (단기가소성 유무)
    x = np.arange(1, len(det_b) + 1)
    axB.plot(x, det_prof, "o-", color="#1565c0", ms=7, lw=2,
             label=f"DetAMPANMDA (단기 O)")
    axB.plot(x, gb_prof, "s-", color="#c62828", ms=7, lw=2,
             label=f"GB 동결 (단기 X)")
    axB.axhline(1.0, color="#90a4ae", ls="--", lw=1.0)
    axB.set_xticks(x); axB.set_xlabel("펄스 번호")
    axB.set_ylabel("정규화 **증분** (첫 펄스=1)")
    axB.set_title("B. 두 기준선의 차이는 **단기가소성**뿐\n"
                  f"GB 동결의 증분은 평평 (편차 {gb_flat:.1e}) — 절대 봉우리는 꼬리로 누적",
                  fontsize=9.5, loc="left")
    axB.legend(fontsize=8)

    # C: 효능 rho 시간추이 (동결 확인)
    R = results["GBPlasticitySyn"][3]["R"]
    axC.plot(R["t"] / 1000.0, R["rho"], color="#c62828", lw=2)
    axC.axhline(0.0, color="#90a4ae", ls="--", lw=1.0)
    axC.set_ylim(-0.02, 0.06)
    axC.set_xlabel("시간 (s)"); axC.set_ylabel("효능 rho")
    d3 = results["GBPlasticitySyn"][3]["drho"]
    axC.set_title(f"C. 동결 확인 — 트레인 60발 5Hz (12초)\n"
                  f"rho0=0 에서 변화 {d3:+.2e} (완전 불변)", fontsize=9.5, loc="left")

    # D: 동결의 진짜 의미 (rho0 별 표류)
    r0s = [d["rho0"] for d in drift]
    dd = [d["drho_13s"] for d in drift]
    axD.bar(range(len(r0s)), dd, color=["#2e7d32" if abs(v) < 1e-9 else "#ef6c00"
                                       for v in dd], width=0.55)
    axD.set_xticks(range(len(r0s))); axD.set_xticklabels([f"{v:.1f}" for v in r0s])
    axD.axhline(0, color="#455a64", lw=1.0)
    axD.set_xlabel("초기 효능 rho0"); axD.set_ylabel("13초 뒤 rho 변화")
    for i, v in enumerate(dd):
        axD.text(i, v, f"{v:+.1e}", ha="center", fontsize=7.5,
                 va="bottom" if v >= 0 else "top")
    axD.set_title("D. ★동결은 '변하지 않음' 이 아니다\n"
                  "gamma=0 이어도 자율항이 살아 rho0 가 0/1 이 아니면 표류한다",
                  fontsize=9.5, loc="left")

    # E: 참조 대조
    axE.plot(r0s, dd, "o", ms=9, color="#c62828", label="mod")
    axE.plot(r0s, ref_drift, "x", ms=11, mew=2, color="#00838f", label="numpy 참조")
    axE.axhline(0, color="#90a4ae", lw=1.0)
    axE.set_xlabel("초기 효능 rho0"); axE.set_ylabel("13초 뒤 rho 변화")
    axE.set_title(f"E. mod ↔ 참조 대조 (동결 자율항)\n"
                  f"최대 절대차 {max_ref_err:.2e}", fontsize=9.5, loc="left")
    axE.legend(fontsize=8)

    fig.suptitle("5-2  엔진 det — 전달만 하는 기준선 (가소성이 정말 꺼져 있는가)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.88)
    plots.stamp(fig, f"5-2 | 프로브=단일구획+VecStim(전압클램프 {V_HOLD:.0f}mV) · "
                     f"클래스 {cls} · 동결 rho0=0 은 완전 불변 · "
                     f"rho0 중간값은 자율항으로 표류 · 참조 대조 {max_ref_err:.1e}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-2_engine_det.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    gbr = results["GBPlasticitySyn"]
    checks = [
        ("DetAMPANMDA 는 장기 효능 상태가 아예 없다 (구조적 기준선)",
         not CAPS["DetAMPANMDA"]["ltp"]),
        ("GB 동결 rho0=0: 모든 프로토콜에서 rho 변화 = 0",
         all(abs(r["drho"]) == 0.0 for r in gbr)),
        ("GB 동결은 단기가소성이 없다 (펄스별 **증분**이 평평, 편차 < 1%)",
         gb_flat < 0.01),
        ("★단기가소성 판정은 증분으로 해야 한다 — 절대 봉우리는 NMDA 꼬리로 누적된다",
         gb_pk[-1] > gb_pk[0] * 1.05 and gb_flat < 0.01),
        ("DetAMPANMDA 는 단기가소성이 있다 (버스트에서 억압)",
         det_prof[-1] < 0.9),
        ("★동결 = 자율항까지 끈 것이 아니다 (rho0=0.1 에서 표류)",
         abs(drift[1]["drho_13s"]) > 1e-6),
        ("★rho*=0.5 는 불안정 고정점이라 **정확히** 그 값에서는 안 움직인다 (칼날 균형)",
         abs(drift[2]["drho_13s"]) == 0.0),
        ("★rho0=0 과 1 은 자율항의 안정 고정점이라 표류하지 않는다",
         abs(drift[0]["drho_13s"]) == 0.0 and abs(drift[-1]["drho_13s"]) == 0.0),
        ("동결 자율항 표류가 numpy 참조와 일치 (절대차 < 1e-4)",
         max_ref_err < 1e-4),
        ("★두 기준선의 첫 펄스 차이는 정확히 방출확률 Use 다 (비 = Use ± 5%)",
         abs(first_ratio - P["Use"]) / P["Use"] < 0.05),
        ("★함정 재현: GB 계열은 t=0 스파이크 전달이 0 (w 미초기화)",
         trap["GBPlasticitySyn"]["g_at_t0_nS"] < 1e-9),
        ("★같은 t=0 에서 DetAMPANMDA 는 정상 전달 (GB 계열만의 문제)",
         trap["DetAMPANMDA"]["g_at_t0_nS"] > 1e-3),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(v_hold=V_HOLD, rec_dt=REC_DT, syn_class=cls, syn_params=P,
               engines={m: [{k: v for k, v in r.items() if k != "R"} for r in rows]
                        for m, rows in results.items()},
               freeze_drift=dict(mod=drift, ref=[round(v, 12) for v in ref_drift],
                                 max_abs_err=max_ref_err, t_ms=13000.0),
               t0_trap=trap,
               burst_profile=dict(det_jumps=[round(v, 6) for v in det_b],
                                  gb_frozen_jumps=[round(v, 6) for v in gb_b],
                                  det_peaks=[round(v, 6) for v in det_pk],
                                  gb_frozen_peaks=[round(v, 6) for v in gb_pk],
                                  first_pulse_ratio=round(first_ratio, 5),
                                  Use=P["Use"],
                                  det_norm=[round(float(v), 6) for v in det_prof],
                                  gb_norm=[round(float(v), 6) for v in gb_prof],
                                  gb_flatness=gb_flat),
               finding_units=("두 mod 계열의 전도도 규약이 다르다. GB 계열은 syn.gmax 에 uS 를 "
                              "받고 NetCon weight 는 전달 플래그(1.0)다. Det/Prob 계열은 gmax 가 "
                              "RANGE 가 아니고(mod 내부 상수 0.001) NetCon weight 에 nS 를 그대로 "
                              "받는다. 섞으면 1000배 어긋나고 오류는 나지 않는다 — "
                              "lib/synprobe.CAPS 의 gmax_via 가 이 규약을 선언한다."),
               finding_jump=("단기가소성 유무는 **펄스별 증분**으로 판정해야 한다. "
                             "tau_d_NMDA=148.5ms 이므로 100Hz 버스트에서는 전도도가 누적되어 "
                             "단기가소성이 전혀 없는 GB 동결도 절대 봉우리가 커진다"
                             "(첫 펄스 대비 4번째). 증분으로 보면 평평하다."),
               finding_t0=("GB 계열 mod(GBPlasticitySyn/StpSyn/StpProbSyn)는 전달 가중치 w 를 "
                           "BREAKPOINT 에서만 계산하고 INITIAL 에서 초기화하지 않는다. 따라서 "
                           "t=0 에 도착한 전시냅스 스파이크는 w=0 으로 전달되어 전도도가 0 이 된다 "
                           "— 오류 없이 조용히 사라진다. 3단계는 정착(250ms) 뒤에 자극했으므로 "
                           "우연히 안전했다. lib/synprobe.drive_pre 가 t<=0 을 막는다. "
                           "근본 수정은 mod INITIAL 에 w=w0+rho0*(b*w0-w0) 추가지만 그 파일은 "
                           "shared/mechanisms 라 01·05 트랙과 공유이므로 별도 결정 필요."),
               finding=("동결(gamma_p=gamma_d=0)은 '효능이 변하지 않는다' 가 아니다. GB 의 자율항 "
                        "-rho(1-rho)(rho*-rho) 는 gamma 와 무관하게 살아 있어 rho0 가 안정 고정점"
                        "(0 또는 1)이 아니면 이중우물 바닥으로 표류한다. 우리 3단계는 rho0=0 을 "
                        "썼으므로 실제로 완전 불변이었다 — 우연히 안전했던 것이고, 5-11 의 동결 "
                        "계약은 'rho0 를 안정 고정점에 둘 것' 을 명시해야 한다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-2_det.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-2 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
