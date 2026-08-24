# -*- coding: utf-8 -*-
"""5-11 동결 계약 검증 — 대조군이 정말 대조군인가

단계   : 5-11 (5단계 가소성 엔진 / 하위 11 freeze)
쉬운 설명: 6단계 실험은 "가소성을 켠 것" 과 "끈 것" 을 비교한다. 그 비교가 성립하려면
          **끈 쪽이 전달은 똑같고 효능만 안 변해야** 한다. 전달까지 달라지면 무엇을 비교한
          것인지 알 수 없다. 여기서 그것을 계약으로 못박고 **행동으로** 검사한다.
방법   : 장기가소성 엔진마다 동결런과 가소런을 **완전히 같은 자극**으로 돌려
          (1) 첫 펄스 전달이 **정확히 같은가**(차이 = 0)
          (2) 동결런의 효능이 **시간불변인가**
          (3) 가소런은 실제로 변하는가 (안 변하면 비교가 공허하다)
          (4) D21 의 rho0 조건이 계약에 포함돼 있는가
검증   : 첫 펄스 차이 = 0.0 (부동소수 정확히) · 동결 효능 변화 = 0 · 가소런은 변한다.
근거   : D21(동결의 정의) · D29(계약을 행동으로) · docs/ENGINE_SPEC.md
결과   : figures/5-11_freeze_identity.png · figures/5-11_freeze.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\11_freeze\\5-11_freeze_contract.py
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

V_HOLD = -70.0
T0 = 20.0
DT_SIM = 0.025
REC_DT = 0.1

# 효능을 크게 바꾸는 자극이어야 비교가 뜻을 갖는다 (theta 버스트 = 5-4 와 동일)
N_BURST, BURST_HZ, N_IN, IN_HZ, DT_PAIR = 10, 5.0, 4, 100.0, 5.0
TSTOP = 3000.0
RHO0S = [0.0, 0.3, 0.5, 1.0]


def burst_times():
    bi, ii = 1000.0 / BURST_HZ, 1000.0 / IN_HZ
    pre = [T0 + b * bi + k * ii for b in range(N_BURST) for k in range(N_IN)]
    return pre, [t + DT_PAIR for t in pre]


def main():
    plots.setup()
    print("=== 5-11 동결 계약 검증 ===")
    cls, P = load_synapse_cfg()
    pre, post = burst_times()
    ltp_keys = engines.with_cap("ltp")
    print(f"  대상(장기가소성 보유): {ltp_keys}")
    print(f"  자극: theta 버스트 {N_BURST}회 @{BURST_HZ:.0f}Hz x {N_IN}발 "
          f"{IN_HZ:.0f}Hz ({len(pre)}펄스)")

    def run(key, rho0, frozen):
        e = engines.get(key)
        p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        engines.apply_params(p.syn, key, P, rho0=rho0, frozen=frozen)
        if e["prob"]:
            p.seed(1, 2, 3)                     # 동결·가소 같은 시드
        p.drive_pre(pre)
        if e["post_nc"]:
            p.drive_post(post)
        p.rec_dt = DT_SIM if False else REC_DT
        R = p.run(TSTOP, dt=DT_SIM)
        return R

    def first_pulse(key, rho0, frozen):
        """첫 펄스 전도도만 정밀히 (rec_dt 를 dt 와 같게)."""
        e = engines.get(key)
        p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=DT_SIM)
        p.set_gmax(P["g_nS"])
        engines.apply_params(p.syn, key, P, rho0=rho0, frozen=frozen)
        if e["prob"]:
            p.seed(1, 2, 3)
        p.drive_pre(pre[:1])
        if e["post_nc"]:
            p.drive_post([pre[0] + DT_PAIR])
        R = p.run(pre[0] + 60.0, dt=DT_SIM)
        return float(R["g"].max())

    # ── (1) 첫 펄스 동일성 ────────────────────────────────────────────────
    print(f"\n  [1] 첫 펄스 전달이 동결·가소에서 **정확히 같은가**")
    ident = []
    for key in ltp_keys:
        for r0 in RHO0S:
            a = first_pulse(key, r0, True)
            b = first_pulse(key, r0, False)
            ident.append(dict(key=key, rho0=r0, frozen=a, plastic=b, diff=abs(a - b)))
            print(f"      {key:<5}rho0 {r0:.1f}: 동결 {a*1e3:.8f} nS · "
                  f"가소 {b*1e3:.8f} nS · 차이 {abs(a-b)*1e3:.2e} nS "
                  f"{'(정확히 0)' if a == b else ''}")
    exact = all(i["diff"] == 0.0 for i in ident)

    # ── (2)(3) 효능 시간불변 vs 실제 변화 ─────────────────────────────────
    print(f"\n  [2·3] 동결런은 불변인가 · 가소런은 실제로 변하는가")
    traj = {}
    rows = []
    for key in ltp_keys:
        for r0 in RHO0S:
            Rf = run(key, r0, True)
            Rp = run(key, r0, False)
            df = float(Rf["rho"][-1] - Rf["rho"][0])
            dp = float(Rp["rho"][-1] - Rp["rho"][0])
            ok_decl = engines.freeze_ok(key, r0)
            ok_robust = engines.freeze_robust(key, r0)
            traj[(key, r0)] = (Rf, Rp)
            rows.append(dict(key=key, rho0=r0, drho_frozen=df, drho_plastic=dp,
                             declared_ok=ok_decl, declared_robust=ok_robust,
                             flat=abs(df) < 1e-12, changed=abs(dp) > 1e-6))
            print(f"      {key:<5}rho0 {r0:.1f}: 동결 변화 {df:+.3e} · "
                  f"가소 변화 {dp:+.3e} · 고정점={ok_decl} 견고={ok_robust}"
                  f"{'' if ok_decl == (abs(df) < 1e-12) else '  <- 선언 불일치!'}")
    safe_rows = [r for r in rows if r["declared_ok"]]
    unsafe_rows = [r for r in rows if not r["declared_ok"]]
    knife_rows = [r for r in rows if r["declared_ok"] and not r["declared_robust"]]
    consistent = all(r["declared_ok"] == r["flat"] for r in rows)

    # ── (4) 계약 문구 확인 — 레지스트리가 rho0 조건을 담고 있는가 ─────────
    print(f"\n  [4] 레지스트리가 D21 의 rho0 조건을 선언에 담고 있는가")
    decl = []
    for key in engines.ORDER:
        e = engines.get(key)
        decl.append(dict(key=key, ltp=e["ltp"], freeze=e["freeze"],
                         freeze_rho0=e["freeze_rho0"],
                         freeze_rho0_robust=e["freeze_rho0_robust"]))
        print(f"      {key:<5}동결 파라미터 {e['freeze'] or '{}'} · "
              f"고정점 {e['freeze_rho0'] or '제약 없음'} · "
              f"견고 {e['freeze_rho0_robust'] or '제약 없음'}")
    # 자율항이 없는 엔진(stdp)은 제약이 없어야 한다
    stdp_free = engines.get("stdp")["freeze_rho0"] is None
    gb_constrained = all(engines.get(k)["freeze_rho0_robust"] == (0.0, 1.0)
                         for k in ("A", "B", "C"))

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.4))
    gs_ = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.48)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1:])
    axC = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axE = fig.add_subplot(gs_[1, 2])

    # A: 첫 펄스 차이
    lbl = [f"{i['key']}\n{i['rho0']:.1f}" for i in ident]
    axA.bar(range(len(ident)), [i["diff"] * 1e3 + 1e-18 for i in ident],
            color="#2e7d32", width=0.6)
    axA.set_yscale("log"); plots.ascii_log(axA)
    axA.set_xticks(range(len(ident)))
    axA.set_xticklabels(lbl, fontsize=6.5)
    axA.set_ylabel("|동결 - 가소| 첫 펄스 (nS)")
    axA.set_title("A. 첫 펄스 전달이 정확히 같다\n"
                  f"{'전 조건 차이 = 0 (부동소수 정확히)' if exact else '차이 발생!'}",
                  fontsize=9.2, loc="left")

    # B: 효능 궤적 (엔진 A · rho0 0.3 과 1.0)
    for (key, r0), col, ls in ((("A", 0.3), "#c62828", "-"),
                               (("A", 1.0), "#1565c0", "-")):
        Rf, Rp = traj[(key, r0)]
        axB.plot(Rp["t"] / 1000.0, Rp["rho"], color=col, lw=2.0, ls=ls,
                 label=f"가소 rho0={r0}")
        axB.plot(Rf["t"] / 1000.0, Rf["rho"], color=col, lw=1.2, ls="--",
                 label=f"동결 rho0={r0}")
    axB.set_xlabel("시간 (s)"); axB.set_ylabel("효능 rho")
    axB.set_title("B. 엔진 A — 가소런은 변하고 동결런은 (거의) 평평하다\n"
                  "rho0=0.3 동결선의 미세 기울기 = 자율항 표류 (D21)",
                  fontsize=9.2, loc="left")
    axB.legend(fontsize=8, ncol=2)

    # C: 동결 변화 크기 히트맵
    kk = ltp_keys
    Mf = np.zeros((len(kk), len(RHO0S)))
    for r in rows:
        Mf[kk.index(r["key"]), RHO0S.index(r["rho0"])] = abs(r["drho_frozen"])
    axC.imshow(np.log10(Mf + 1e-16), cmap="RdYlGn_r", aspect="auto",
               vmin=-16, vmax=-3)
    axC.set_xticks(range(len(RHO0S)))
    axC.set_xticklabels([f"{v:.1f}" for v in RHO0S], fontsize=8)
    axC.set_yticks(range(len(kk))); axC.set_yticklabels(kk, fontsize=9)
    axC.set_xlabel("rho0")
    for i in range(len(kk)):
        for j in range(len(RHO0S)):
            v = Mf[i, j]
            axC.text(j, i, "0" if v < 1e-12 else f"{v:.0e}", ha="center",
                     va="center", fontsize=7,
                     color="#1b5e20" if v < 1e-12 else "#b71c1c")
    axC.set_title("C. 동결런 |rho 변화|\nGB 계열은 rho0 가 0/1 일 때만 완전 불변",
                  fontsize=9.2, loc="left")

    # D: 가소런 변화 크기 (대조가 공허하지 않다는 증거)
    Mp = np.zeros((len(kk), len(RHO0S)))
    for r in rows:
        Mp[kk.index(r["key"]), RHO0S.index(r["rho0"])] = r["drho_plastic"]
    im = axD.imshow(Mp, cmap="RdYlGn", aspect="auto",
                    vmin=-abs(Mp).max(), vmax=abs(Mp).max())
    axD.set_xticks(range(len(RHO0S)))
    axD.set_xticklabels([f"{v:.1f}" for v in RHO0S], fontsize=8)
    axD.set_yticks(range(len(kk))); axD.set_yticklabels(kk, fontsize=9)
    axD.set_xlabel("rho0")
    for i in range(len(kk)):
        for j in range(len(RHO0S)):
            axD.text(j, i, f"{Mp[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    axD.set_title("D. 가소런 rho 변화 — 대조가 공허하지 않다\n"
                  "rho0=1 에서는 위로 갈 곳이 없어 0 이다", fontsize=9.2, loc="left")
    plt.colorbar(im, ax=axD, fraction=0.046, pad=0.04)

    # E: 계약 요약
    items = [("첫 펄스 동일 (차이=0)", exact),
             ("동결 선언 = 실제 거동", consistent),
             ("stdp 는 rho0 제약 없음\n(자율항 없다)", stdp_free),
             ("GB 계열 견고 rho0\n따로 선언", gb_constrained),
             ("가소런이 실제로 변한다\n(안전 조건에서)",
              any(r["changed"] for r in safe_rows))]
    axE.barh(range(len(items)), [1.0 if v else 0.0 for _, v in items],
             color=["#2e7d32" if v else "#c62828" for _, v in items])
    axE.set_yticks(range(len(items)))
    axE.set_yticklabels([n for n, _ in items], fontsize=7.5)
    axE.invert_yaxis(); axE.set_xlim(0, 1.2); axE.set_xticks([0, 1])
    axE.set_xticklabels(["X", "O"])
    axE.set_title("E. 계약 요약 — 전부 행동으로 검사\n"
                  "선언만 두면 조용히 틀린다", fontsize=9.2, loc="left")

    fig.suptitle("5-11  동결 계약 검증 — 대조군이 정말 대조군인가",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"5-11 | 대상 {ltp_keys} x rho0 {RHO0S} · 첫 펄스 차이 "
                     f"{'전부 0' if exact else '발생'} · 동결 선언 일치 {consistent} · "
                     f"자극 theta 버스트 {N_BURST}회")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-11_freeze_identity.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    checks = [
        ("★동결런과 가소런의 첫 펄스 전달이 **정확히 같다** (차이 = 0.0)", exact),
        ("★동결 선언(freeze + freeze_rho0)이 실제 거동과 일치한다 (D21)", consistent),
        ("안전 rho0 에서 동결런의 효능 변화가 정확히 0",
         all(r["drho_frozen"] == 0.0 for r in safe_rows)),
        ("★고정점이 아닌 rho0 에서는 동결런도 표류한다 (그래서 계약에 rho0 조건이 있다)",
         len(unsafe_rows) > 0 and all(abs(r["drho_frozen"]) > 1e-9
                                      for r in unsafe_rows)),
        (f"★rho*=0.5 는 고정점이지만 **불안정**이라 대조군 불가로 선언한다 "
         f"({len(knife_rows)}조건)",
         len(knife_rows) > 0 and all(r["drho_frozen"] == 0.0 for r in knife_rows)),
        ("가소런은 실제로 효능을 바꾼다 (대조가 공허하지 않다)",
         any(r["changed"] for r in safe_rows)),
        ("★rho0=1 에서 LTP 프로토콜이 오히려 **LTD 를 만든다** (비대칭 포화)",
         all(r["drho_plastic"] < -1e-3 for r in rows if r["rho0"] == 1.0)),
        ("rho0=0 에서는 같은 자극이 LTP 를 만든다 (부호가 rho0 로 뒤집힌다)",
         all(r["drho_plastic"] > 1e-3 for r in rows if r["rho0"] == 0.0)),
        ("자율항이 없는 엔진(stdp)은 rho0 제약이 없다고 선언한다", stdp_free),
        ("GB 계열 3종은 **견고한** rho0 를 따로 선언한다", gb_constrained),
        ("동결 파라미터가 엔진마다 선언돼 있다",
         all(engines.get(k)["freeze"] for k in ltp_keys)),
        ("장기가소성 없는 엔진은 동결 파라미터가 비어 있다",
         engines.get("det")["freeze"] == {}),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(ltp_engines=ltp_keys, rho0_tested=RHO0S, syn_class=cls,
               protocol=dict(n_burst=N_BURST, burst_hz=BURST_HZ, n_in=N_IN,
                             in_hz=IN_HZ, n_pulses=len(pre), tstop=TSTOP, dt=DT_SIM),
               first_pulse_identity=ident, exact=exact,
               efficacy=rows, declaration_consistent=consistent,
               declarations=decl,
               finding=("동결 계약은 두 조건이다: (1) 전달이 가소런과 **정확히** 같아야 하고 "
                        "(2) 효능이 시간불변이어야 한다. (1)은 전 조건에서 부동소수 차이 0 으로 "
                        "성립한다 — 동결은 갱신 항만 끄고 초기 가중치를 건드리지 않기 때문이다. "
                        "(2)는 **rho0 에 달렸다**(D21): GB 계열은 자율항이 살아 있어 rho0 가 "
                        "0 또는 1 이 아니면 표류한다. 레지스트리가 그 제약을 선언으로 담고, "
                        "이 스크립트가 선언과 실제 거동이 일치하는지 검사한다."),
               finding_ceiling=("★rho0=1(UP)에서 같은 theta 버스트가 **LTD 를 만든다** "
                                "(A/B/C 전부 -0.166 · stdp -0.0145). GB 의 강화 항은 "
                                "gamma_p*(1-rho) 라서 rho=1 에서 정확히 0 이 되는데 약화 항은 "
                                "-gamma_d*rho 로 최대다 — **비대칭 포화**다. 고전 STDP 도 하드 "
                                "상한 때문에 LTP 기여가 잘리고 LTD 만 남는다. rho0=0 에서는 같은 "
                                "자극이 LTP(+0.740)다. => 6단계에서 rho0 가 **결과의 부호를 "
                                "정한다**(D23 강화). '이 프로토콜은 LTP 프로토콜이다' 라는 말은 "
                                "rho0 를 밝히지 않으면 성립하지 않는다."),
               vacuity_note=("대조군 검증은 '가소런이 실제로 변한다' 를 함께 확인해야 의미가 있다. "
                             "안 변하는 자극으로는 '동결과 같다' 가 당연해서 아무것도 증명하지 "
                             "못한다. 그래서 효능을 크게 바꾸는 theta 버스트를 썼다. "
                             "다만 rho0=1 에서는 이중우물 상한이라 가소런도 변하지 않는다 — "
                             "그 조건은 대조 검증에 쓸 수 없다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-11_freeze.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-11 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
