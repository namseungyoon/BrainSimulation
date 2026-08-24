# -*- coding: utf-8 -*-
"""5-10 엔진 간 첫 펄스 정합 — 크기 차이를 없애고 **동역학만** 비교하게 만든다

단계   : 5-10 (5단계 가소성 엔진 / 하위 10 calibrate)
쉬운 설명: 엔진 비교에서 "이 엔진이 LTP 를 더 만든다" 고 말하려면 **출발선이 같아야** 한다.
          지금은 같지 않다 — det 는 방출확률 Use 를 곱하고(절반), 고전 STDP 는 적분기가 달라
          1.15% 크다(D27). 그대로 비교하면 동역학 차이가 아니라 **크기 차이**를 보게 된다.
방법   : 기준 엔진 A 의 첫 펄스 전도도를 목표로 두고 엔진별 `gmax` 배율을 산출한다.
          전도도는 gmax 에 정확히 선형이므로 한 번 재면 배율이 확정된다.
          확률 엔진은 **시행평균으로만** 정합된다 — 개별 시행 분산을 함께 공시한다.
검증   : 교정 후 첫 펄스 편차 < 1% · 교정이 동적범위를 바꾸지 않는다 · 선형성 확인.
근거   : D27(적분기 1.15%) · D22(단위 규약) · D24(관례) · docs/ENGINE_SPEC.md 계약
결과   : figures/5-10_first_pulse_match.png · figures/5-10_calib.json ·
          **config/engines_calib.yaml** (6단계가 읽는 단일 출처)
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\10_calibrate\\5-10_calibrate.py
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
from lib import engines                              # noqa: E402
from lib.synprobe import SynProbe                    # noqa: E402
from lib.wiring import load_synapse_cfg              # noqa: E402

REF_KEY = "A"                # 기준 엔진 (순수 GB)
V_HOLD = -70.0
T0 = 20.0
DT_SIM = 0.025
TOL = 0.01                   # 1%
N_TRIAL_C = 400
RHO0_REF = 0.0               # 교정은 DOWN 상태에서 (안정 고정점, D21)


def main():
    plots.setup()
    print("=== 5-10 엔진 간 첫 펄스 정합 ===")
    cls, P = load_synapse_cfg()
    print(f"  기준 엔진 {REF_KEY} ({engines.mech(REF_KEY)}) · rho0 {RHO0_REF} · "
          f"시냅스 {cls} g {P['g_nS']}nS")

    def first_peak(key, g_nS, rho0=RHO0_REF, seed=None):
        e = engines.get(key)
        p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=DT_SIM)
        p.set_gmax(g_nS)
        engines.apply_params(p.syn, key, P, rho0=rho0, frozen=True)
        if e["prob"]:
            p.seed(*(seed or (1, 2, 3)))
        p.drive_pre([T0])
        return float(p.run(T0 + 60.0, dt=DT_SIM)["g"].max()) * 1e3

    g0 = P["g_nS"]

    # ── (A) 교정 전 ───────────────────────────────────────────────────────
    print(f"\n  [A] 교정 전 첫 펄스 전도도 (모든 엔진 g={g0}nS)")
    raw = {}
    for key in engines.ORDER:
        e = engines.get(key)
        if e["prob"]:
            vals = [first_peak(key, g0, seed=(4000 + k, 3 * k + 1, 7 * k + 5))
                    for k in range(N_TRIAL_C)]
            v = float(np.mean(vals))
            raw[key] = dict(mean=v, sd=float(np.std(vals)),
                            se=float(np.std(vals) / np.sqrt(N_TRIAL_C)),
                            n_zero=int(np.sum(np.array(vals) < 1e-9)))
        else:
            raw[key] = dict(mean=first_peak(key, g0), sd=0.0, se=0.0, n_zero=0)
        r = raw[key]
        print(f"      {key:<5}{e['label']:<18} {r['mean']:.5f} nS" +
              (f"  (시행 표준편차 {r['sd']:.5f} · 실패 {r['n_zero']}/{N_TRIAL_C})"
               if e["prob"] else ""))
    target = raw[REF_KEY]["mean"]
    dev_before = {k: (v["mean"] - target) / target for k, v in raw.items()}
    print(f"      -> 목표 {target:.5f} nS · 교정 전 편차 " +
          " · ".join(f"{k} {100*dev_before[k]:+.2f}%" for k in engines.ORDER))

    # ── (B) 배율 산출 + 선형성 확인 ───────────────────────────────────────
    print(f"\n  [B] gmax 배율 산출 (전도도는 gmax 에 선형)")
    scale = {k: target / raw[k]["mean"] for k in engines.ORDER}
    for k in engines.ORDER:
        print(f"      {k:<5} 배율 {scale[k]:.6f}  -> g {g0*scale[k]:.5f} nS")
    # 선형성: 2배 gmax 가 정확히 2배 전도도인가
    lin = []
    for k in engines.ORDER:
        if engines.get(k)["prob"]:
            continue
        a = first_peak(k, g0); b = first_peak(k, 2 * g0)
        lin.append(dict(key=k, ratio=b / a, err=abs(b / a - 2.0)))
    print(f"      선형성 (gmax 2배 -> 전도도 2배): 최대 오차 "
          f"{max(l['err'] for l in lin):.2e}")

    # ── (C) 교정 후 검증 ──────────────────────────────────────────────────
    print(f"\n  [C] 교정 후 첫 펄스 (배율 적용)")
    after = {}
    for key in engines.ORDER:
        e = engines.get(key)
        gg = g0 * scale[key]
        if e["prob"]:
            vals = [first_peak(key, gg, seed=(4000 + k, 3 * k + 1, 7 * k + 5))
                    for k in range(N_TRIAL_C)]
            after[key] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)),
                              se=float(np.std(vals) / np.sqrt(N_TRIAL_C)))
        else:
            after[key] = dict(mean=first_peak(key, gg), sd=0.0, se=0.0)
        d = (after[key]["mean"] - target) / target
        print(f"      {key:<5}{after[key]['mean']:.5f} nS · 편차 {100*d:+.4f}%" +
              (f"  (개별 시행 표준편차 {100*after[key]['sd']/target:.1f}%)"
               if e["prob"] else ""))
    dev_after = {k: (v["mean"] - target) / target for k, v in after.items()}
    max_dev = max(abs(v) for v in dev_after.values())

    # ── (D) 교정 후 동적범위 ──────────────────────────────────────────────
    print(f"\n  [D] 교정 후 동적범위 — 교정이 배율을 바꾸지 않아야 한다")
    dyn = []
    for key in engines.ORDER:
        gg = g0 * scale[key]
        lo = first_peak(key, gg, rho0=0.0)
        hi = first_peak(key, gg, rho0=1.0)
        dyn.append(dict(key=key, lo=lo, hi=hi,
                        ratio=(hi / lo if lo > 1e-12 else float("nan"))))
        print(f"      {key:<5} rho0=0 {lo:.5f} -> rho0=1 {hi:.5f} nS · "
              f"배율 {hi/lo if lo > 1e-12 else float('nan'):.4f}")

    # ── (E) config 기록 ───────────────────────────────────────────────────
    cpath = os.path.join(ROOT, "config", "engines_calib.yaml")
    lines = [
        "# 5-10 산출 — 엔진 간 첫 펄스 정합 배율. 6단계가 읽는 단일 출처.",
        "#",
        "# 왜 필요한가: 엔진마다 첫 펄스 크기가 다르다. 그대로 비교하면 동역학 차이가 아니라",
        "#   크기 차이를 보게 된다. gmax 에 이 배율을 곱하면 첫 펄스가 기준 엔진과 같아진다.",
        "# 전도도는 gmax 에 정확히 선형이므로 배율은 한 번 재면 확정이다(선형성 검증 포함).",
        f"# 기준: {REF_KEY} ({engines.mech(REF_KEY)}) · rho0 {RHO0_REF} · "
        f"클래스 {cls} · g {g0} nS · dt {DT_SIM}",
        f"# 목표 첫 펄스 전도도: {target:.6f} nS",
        "",
        f"reference: {REF_KEY}",
        f"target_g_nS: {target:.6f}",
        f"base_g_nS: {g0}",
        f"dt_ms: {DT_SIM}",
        f"rho0: {RHO0_REF}",
        "",
        "gmax_scale:",
    ]
    why = {
        "det": "방출확률 Use 를 곱하므로 절반에서 출발한다 (BBP 관례)",
        "A": "기준 엔진",
        "B": "norm_Pr=1 이 첫 펄스를 A 에 맞춘다 (D24)",
        "C": "시행평균으로만 정합된다 — 개별 시행은 흩어진다 (D26)",
        "stdp": "cnexp 라 GB 의 derivimplicit 보다 1.15% 크다 (D27)",
    }
    for key in engines.ORDER:
        lines.append(f"  {key}: {{v: {scale[key]:.6f}, "
                     f"before_pct: {100*dev_before[key]:+.3f}, "
                     f"ref: \"{why.get(key, '')}\"}}")
    lines += ["", "# 확률 엔진은 평균만 맞는다. 개별 시행 표준편차(목표 대비 %):",
              f"#   C: {100*after['C']['sd']/target:.1f}%  "
              f"(방출 실패 {raw['C']['n_zero']}/{N_TRIAL_C}회)",
              "",
              "# 교정 후 동적범위 (rho0=0 -> 1 배율). 6-9 는 이 값을 함께 인쇄한다:",
              ]
    for d in dyn:
        lines.append(f"#   {d['key']}: x{d['ratio']:.4f}")
    with open(cpath, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  [E] config 기록: {cpath}")

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.4))
    gs_ = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.48)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1])
    axC = fig.add_subplot(gs_[0, 2])
    axD = fig.add_subplot(gs_[1, 0])
    axE = fig.add_subplot(gs_[1, 1])
    axF = fig.add_subplot(gs_[1, 2])
    K = engines.ORDER
    xx = np.arange(len(K))

    # A: 교정 전
    axA.bar(xx, [raw[k]["mean"] for k in K], color="#c62828", width=0.6,
            yerr=[raw[k]["sd"] for k in K], capsize=3)
    axA.axhline(target, color="#37474f", ls="--", lw=1.8, label=f"목표 {target:.4f} nS")
    axA.set_xticks(xx); axA.set_xticklabels(K)
    axA.set_ylabel("첫 펄스 전도도 (nS)")
    axA.set_title("A. 교정 전 — 출발선이 다르다\n"
                  f"det 는 Use({P['Use']})배 · stdp 는 +1.15%(적분기)",
                  fontsize=9.2, loc="left")
    axA.legend(fontsize=8)
    for i, k in enumerate(K):
        axA.text(i, raw[k]["mean"], f"{100*dev_before[k]:+.1f}%", ha="center",
                 va="bottom", fontsize=7.5)

    # B: 배율
    axB.bar(xx, [scale[k] for k in K], color="#1565c0", width=0.6)
    axB.axhline(1.0, color="#90a4ae", ls=":", lw=1.2)
    axB.set_xticks(xx); axB.set_xticklabels(K); axB.set_ylabel("gmax 배율")
    for i, k in enumerate(K):
        axB.text(i, scale[k], f"{scale[k]:.4f}", ha="center", va="bottom", fontsize=7.5)
    axB.set_title("B. 산출된 배율 -> config/engines_calib.yaml\n"
                  "전도도가 gmax 에 선형이라 한 번 재면 확정", fontsize=9.2, loc="left")

    # C: 교정 후
    axC.bar(xx, [100 * dev_after[k] for k in K], color="#2e7d32", width=0.6)
    axC.axhline(0, color="#37474f", lw=1.0)
    for sgn in (1, -1):
        axC.axhline(sgn * TOL * 100, color="#c62828", ls="--", lw=1.2)
    axC.set_xticks(xx); axC.set_xticklabels(K); axC.set_ylabel("목표 대비 편차 (%)")
    axC.set_ylim(-1.5, 1.5)
    for i, k in enumerate(K):
        axC.text(i, 100 * dev_after[k], f"{100*dev_after[k]:+.3f}", ha="center",
                 va="bottom", fontsize=7.5)
    axC.set_title(f"C. 교정 후 — 전부 ±{TOL*100:.0f}% 안\n"
                  f"최대 편차 {100*max_dev:.4f}%", fontsize=9.2, loc="left")

    # D: 확률 엔진의 한계
    vals = [first_peak("C", g0 * scale["C"], seed=(4000 + k, 3 * k + 1, 7 * k + 5))
            for k in range(N_TRIAL_C)]
    axD.hist(vals, bins=16, color="#6a1b9a", alpha=0.85, edgecolor="white")
    axD.axvline(target, color="#37474f", ls="--", lw=2.0, label=f"목표 {target:.3f}")
    axD.axvline(float(np.mean(vals)), color="#c62828", lw=2.0,
                label=f"평균 {np.mean(vals):.3f}")
    axD.set_xlabel("첫 펄스 전도도 (nS)"); axD.set_ylabel("시행 수")
    axD.set_title(f"D. ★확률 엔진은 **평균만** 정합된다\n"
                  f"개별 시행 표준편차 {100*np.std(vals)/target:.0f}% · "
                  f"실패 {int(np.sum(np.array(vals) < 1e-9))}/{N_TRIAL_C}",
                  fontsize=9.2, loc="left")
    axD.legend(fontsize=7.8)

    # E: 동적범위 (교정 후)
    axE.bar(xx - 0.19, [d["lo"] for d in dyn], width=0.38, color="#90a4ae",
            label="rho0=0")
    axE.bar(xx + 0.19, [d["hi"] for d in dyn], width=0.38, color="#2e7d32",
            label="rho0=1")
    axE.set_xticks(xx); axE.set_xticklabels(K)
    axE.set_ylabel("첫 펄스 전도도 (nS)")
    for i, d in enumerate(dyn):
        if np.isfinite(d["ratio"]):
            axE.text(i, d["hi"], f"x{d['ratio']:.3f}", ha="center", va="bottom",
                     fontsize=7.5)
    axE.set_title("E. 교정 후 동적범위 — 배율은 그대로다\n"
                  "출발선만 맞췄고 천장 비는 건드리지 않았다", fontsize=9.2, loc="left")
    axE.legend(fontsize=7.8)

    # F: 선형성
    lk = [l["key"] for l in lin]
    axF.bar(np.arange(len(lin)), [l["ratio"] for l in lin], color="#00838f", width=0.55)
    axF.axhline(2.0, color="#37474f", ls="--", lw=1.6, label="이상값 2.000")
    axF.set_xticks(np.arange(len(lin))); axF.set_xticklabels(lk)
    axF.set_ylim(1.98, 2.02); axF.set_ylabel("gmax 2배 시 전도도 비")
    for i, l in enumerate(lin):
        axF.text(i, l["ratio"], f"{l['ratio']:.6f}", ha="center", va="bottom",
                 fontsize=7)
    axF.set_title(f"F. 선형성 확인 (배율이 한 번의 측정으로 확정되는 근거)\n"
                  f"최대 오차 {max(l['err'] for l in lin):.1e}", fontsize=9.2, loc="left")
    axF.legend(fontsize=7.8)

    fig.suptitle("5-10  엔진 간 첫 펄스 정합 — 크기 차이를 없애고 동역학만 비교한다",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"5-10 | 기준 {REF_KEY} · 목표 {target:.4f} nS · 교정 후 최대 편차 "
                     f"{100*max_dev:.4f}% · 확률 엔진 개별 시행 표준편차 "
                     f"{100*after['C']['sd']/target:.0f}% · config/engines_calib.yaml 기록")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-10_first_pulse_match.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    det_scale_expect = 1.0 / P["Use"]
    checks = [
        (f"★교정 후 첫 펄스 편차가 전 엔진 < {TOL*100:.0f}% (최대 {100*max_dev:.4f}%)",
         max_dev < TOL),
        (f"교정 전에는 편차가 컸다 (det {100*dev_before['det']:+.1f}%)",
         abs(dev_before["det"]) > 0.1),
        ("전도도가 gmax 에 선형이다 (2배 -> 2배, 오차 < 1e-6)",
         max(l["err"] for l in lin) < 1e-6),
        (f"det 배율이 1/Use = {det_scale_expect:.3f} 근처다 (2% 이내, 실측 "
         f"{scale['det']:.4f})",
         abs(scale["det"] - det_scale_expect) / det_scale_expect < 0.02),
        ("★det 과 stdp 가 **같은 적분기 인자**를 공유한다 (둘 다 cnexp) — "
         "det배율 x Use = stdp배율",
         abs(scale["det"] * P["Use"] - scale["stdp"]) < 1e-3),
        ("B 배율이 1 이다 (norm_Pr 이 이미 A 에 맞춘다, 0.1% 이내)",
         abs(scale["B"] - 1.0) < 0.001),
        ("★stdp 배율이 D27 의 1.15% 를 되돌린다 (0.988~0.990)",
         0.988 < scale["stdp"] < 0.990),
        ("★확률 엔진은 평균만 정합된다 (개별 시행 표준편차 > 목표의 10%)",
         after["C"]["sd"] / target > 0.10),
        ("교정이 동적범위를 바꾸지 않는다 (LTP 엔진 배율이 모두 5.28 근처)",
         all(abs(d["ratio"] - 5.28145) < 0.01 for d in dyn
             if engines.get(d["key"])["ltp"])),
        ("장기가소성 없는 엔진은 동적범위 1 이다",
         all(abs(d["ratio"] - 1.0) < 1e-9 for d in dyn
             if not engines.get(d["key"])["ltp"])),
        ("config/engines_calib.yaml 이 생성됐다", os.path.exists(cpath)),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(reference=REF_KEY, target_g_nS=target, base_g_nS=g0, dt=DT_SIM,
               rho0=RHO0_REF, syn_class=cls, tol=TOL, n_trial_prob=N_TRIAL_C,
               before={k: dict(**raw[k], dev=dev_before[k]) for k in engines.ORDER},
               gmax_scale=scale,
               after={k: dict(**after[k], dev=dev_after[k]) for k in engines.ORDER},
               max_dev_after=max_dev,
               linearity=lin, dynamic_range=dyn,
               config_path="config/engines_calib.yaml",
               finding_integrator=(f"det 배율이 1/Use={1/P['Use']:.3f} 가 아니라 "
                                   f"{scale['det']:.4f} 인 이유: DetAMPANMDA 도 cnexp 를 쓰므로 "
                                   f"D27 의 적분기 인자가 함께 곱해진다. 실제로 "
                                   f"det배율 x Use = {scale['det']*P['Use']:.6f} 이고 "
                                   f"stdp 배율 {scale['stdp']:.6f} 과 같다. 즉 5종 중 "
                                   f"cnexp 를 쓰는 것은 det·stdp 이고 GB 계열 3종만 "
                                   f"derivimplicit 이다."),
               finding=("엔진 비교에서 크기 차이를 먼저 없애야 한다. 교정 전 편차는 "
                        f"det {100*dev_before['det']:+.1f}% (방출확률 Use 를 곱하므로), "
                        f"stdp {100*dev_before['stdp']:+.2f}% (D27 적분기), B/C 는 거의 0 이었다. "
                        "전도도가 gmax 에 정확히 선형(2배 오차 < 1e-6)이므로 배율은 한 번의 "
                        f"측정으로 확정되고, 교정 후 최대 편차가 {100*max_dev:.4f}% 로 떨어졌다. "
                        "교정은 출발선만 맞추고 동적범위(천장 비)는 건드리지 않는다."),
               prob_limit=(f"★확률 엔진은 **평균만** 정합된다. 교정 후에도 개별 시행 표준편차가 "
                           f"목표의 {100*after['C']['sd']/target:.0f}% 이고 "
                           f"{raw['C']['n_zero']}/{N_TRIAL_C} 시행은 방출 실패로 0 이다. "
                           "6단계에서 확률 엔진의 단일 시행을 다른 엔진의 단일 실행과 직접 "
                           "비교할 수 없다 — 시드 스윕 분포로만 비교한다(D26)."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-10_calib.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-10 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
