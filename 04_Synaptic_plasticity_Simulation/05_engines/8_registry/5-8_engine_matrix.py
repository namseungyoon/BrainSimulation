# -*- coding: utf-8 -*-
"""5-8 레지스트리 + 어댑터 계약 — 엔진을 하나의 API 로 돌린다

단계   : 5-8 (5단계 가소성 엔진 / 하위 8 registry)
쉬운 설명: 지금까지 엔진마다 코드를 조금씩 달리 썼다. 그러면 엔진을 추가할 때마다 검증
          코드를 고쳐야 한다. 여기서 **엔진이 자기 능력을 선언**하게 하고, 실행·검증 코드는
          **이름이 아니라 선언을 읽고** 분기하도록 만든다.
방법   : `lib/engines.py` 레지스트리로 5종을 **완전히 동일한 루프**에서 돌린다.
          그리고 계약 3가지를 **행동으로** 검사한다.
검증   : (1) 5종이 같은 API 로 구동된다
          (2) efficacy() 가 모두 [0,1] 같은 척도를 준다
          (3) post_nc 선언을 어기면 **오류가 난다** (조용히 틀리지 않는다)
          (4) prob 엔진은 시딩 없이 쓰면 검출된다
          (5) 동결 선언 + rho0 조건(D21)이 실제로 효능을 불변으로 만든다
          (6) 엔진마다 **동적범위**를 공시한다 (5-10·6-9 가 이걸 써야 오도가 없다)
근거   : docs/ENGINE_SPEC.md · D21(동결) · D22(단위·t=0) · D24(관례) · D26(확률)
결과   : figures/5-8_engine_matrix.png · figures/5-8_registry.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\8_registry\\5-8_engine_matrix.py
비고   : 5-7 GluSynapseCa 는 미결#4(라이선스) 결정 후 등록한다. 레지스트리에 자리와
          계약(post_nc=False)을 주석으로 미리 못박아 두었다.
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

REC_DT = 0.5
V_HOLD = -70.0
T0 = 20.0
DT_SIM = 0.025

# 공통 프로토콜: theta 버스트 10회 @5Hz x 4발 100Hz + post (5-4·5-5 와 같은 것)
N_BURST, BURST_HZ, N_IN, IN_HZ, DT_PAIR = 10, 5.0, 4, 100.0, 5.0
TSTOP = 3000.0


def burst_times():
    bi, ii = 1000.0 / BURST_HZ, 1000.0 / IN_HZ
    pre = [T0 + b * bi + k * ii for b in range(N_BURST) for k in range(N_IN)]
    return pre, [t + DT_PAIR for t in pre]


def main():
    plots.setup()
    print("=== 5-8 레지스트리 + 어댑터 계약 ===")
    cls, P = load_synapse_cfg()
    pre, post = burst_times()
    print(f"  등록 엔진 {len(engines.ORDER)}종: {engines.ORDER}")
    print(f"  공통 프로토콜: theta 버스트 {N_BURST}회 @{BURST_HZ:.0f}Hz x {N_IN}발 "
          f"{IN_HZ:.0f}Hz ({len(pre)}펄스) · 시냅스 {cls}")

    # ── 능력표 ────────────────────────────────────────────────────────────
    print(f"\n  [능력표]")
    print(f"      {'키':<6}{'mech':<24}{'단기':<5}{'장기':<5}{'확률':<5}"
          f"{'postNC':<7}{'전도도':<8}{'자체':<5}참조")
    tab = engines.table()
    for r in tab:
        print(f"      {r['key']:<6}{r['mech']:<24}"
              f"{'O' if r['stp'] else '-':<5}{'O' if r['ltp'] else '-':<5}"
              f"{'O' if r['prob'] else '-':<5}{'O' if r['post_nc'] else '-':<7}"
              f"{r['gmax_via']:<8}{'O' if r['own'] else '-':<5}{r['ref']}")

    # ── (1)(2) 동일 루프 구동 ─────────────────────────────────────────────
    print(f"\n  [1·2] 5종을 **완전히 동일한 루프**로 돌린다 (rho0=0)")
    rows = []
    for key in engines.ORDER:
        e = engines.get(key)
        p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        engines.apply_params(p.syn, key, P, rho0=0.0)
        if e["prob"]:
            p.seed(1, 2, 3)                     # 선언을 읽고 시딩
        p.drive_pre(pre)
        if e["post_nc"]:                        # 선언을 읽고 sentinel 연결
            p.drive_post(post)
        R = p.run(TSTOP, dt=DT_SIM)
        eff = engines.efficacy(key, p.syn)
        rows.append(dict(key=key, label=e["label"], mech=e["mech"],
                         eff=eff, g_max_nS=float(R["g"].max()) * 1e3,
                         has_rho=("rho" in R)))
        print(f"      {key:<6}{e['label']:<18} 효능 {eff:.5f} · "
              f"g 최고 {float(R['g'].max())*1e3:.4f} nS")
    effs = [r["eff"] for r in rows]

    # ── (3) post_nc 계약을 어기면 오류가 나는가 ───────────────────────────
    print(f"\n  [3] post_nc=False 엔진에 sentinel 을 붙이면 어떻게 되는가")
    viol = {}
    for key in engines.ORDER:
        e = engines.get(key)
        p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        try:
            p.drive_post([T0])
            viol[key] = "허용됨"
        except RuntimeError:
            viol[key] = "차단됨(RuntimeError)"
        print(f"      {key:<6}(post_nc={e['post_nc']}) -> {viol[key]}")
    blocked_ok = all((viol[k] == "차단됨(RuntimeError)") != engines.get(k)["post_nc"]
                     for k in engines.ORDER)

    # ── (4) 확률 엔진 시딩 검출 ───────────────────────────────────────────
    print(f"\n  [4] 확률 엔진을 시딩하지 않으면 검출되는가")
    seed_chk = {}
    for key in engines.with_cap("prob"):
        e = engines.get(key)
        out = {}
        for seeded in (False, True):
            p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
            p.set_gmax(P["g_nS"]); engines.apply_params(p.syn, key, P, rho0=0.0)
            if seeded:
                p.seed(1, 2, 3)
            p.drive_pre(pre); p.drive_post(post)
            p.run(TSTOP, dt=DT_SIM)
            out["시딩" if seeded else "미시딩"] = int(p.syn.n_rel)
        seed_chk[key] = out
        print(f"      {key}: 미시딩 방출 {out['미시딩']}/{len(pre)} · "
              f"시딩 {out['시딩']}/{len(pre)} -> "
              f"{'검출 가능' if out['미시딩'] <= 1 else '검출 불가'}")
    # 확률 아닌 엔진에 seed() 를 부르면 막혀야 한다
    nonprob_blocked = 0
    for key in engines.with_cap("prob", False):
        p = SynProbe(engines.mech(key))
        try:
            p.seed(1, 2, 3)
        except RuntimeError:
            nonprob_blocked += 1
    print(f"      확률 아닌 엔진에 seed() -> {nonprob_blocked}/"
          f"{len(engines.with_cap('prob', False))}개 차단됨")

    # ── (5) 동결 계약 (D21) ───────────────────────────────────────────────
    print(f"\n  [5] 동결 선언 + rho0 조건(D21)이 정말 불변을 만드는가")
    frz = []
    for key in engines.with_cap("ltp"):
        e = engines.get(key)
        for r0 in (0.0, 0.3, 1.0):
            p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
            p.set_gmax(P["g_nS"])
            engines.apply_params(p.syn, key, P, rho0=r0, frozen=True)
            if e["prob"]:
                p.seed(1, 2, 3)
            p.drive_pre(pre)
            if e["post_nc"]:
                p.drive_post(post)
            R = p.run(TSTOP, dt=DT_SIM)
            d = float(R["rho"][-1] - R["rho"][0])
            ok_decl = engines.freeze_ok(key, r0)
            frz.append(dict(key=key, rho0=r0, drho=d, declared_ok=ok_decl,
                            actually_flat=abs(d) < 1e-12))
            print(f"      {key:<6}rho0 {r0:.1f}: 변화 {d:+.3e} · "
                  f"선언상 안전={ok_decl} · 실제 불변={abs(d) < 1e-12} "
                  f"{'' if ok_decl == (abs(d) < 1e-12) else '  <- 선언과 불일치!'}")
    frz_consistent = all(f["declared_ok"] == f["actually_flat"] for f in frz)

    # ── (6) 동적범위 공시 ─────────────────────────────────────────────────
    print(f"\n  [6] 엔진별 동적범위 — 이걸 안 밝히면 6-9 가 오도된다")
    dyn = []
    for key in engines.ORDER:
        e = engines.get(key)
        vals = {}
        for r0 in (0.0, 1.0):
            p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=0.025)
            p.set_gmax(P["g_nS"])
            engines.apply_params(p.syn, key, P, rho0=r0, frozen=True)
            if e["prob"]:
                p.seed(1, 2, 3)
            p.drive_pre([T0])
            vals[r0] = float(p.run(T0 + 60.0, dt=DT_SIM)["g"].max()) * 1e3
        lo, hi = vals[0.0], vals[1.0]
        dyn.append(dict(key=key, label=e["label"], g_lo_nS=lo, g_hi_nS=hi,
                        ratio=(hi / lo if lo > 1e-12 else float("nan"))))
        print(f"      {key:<6}{e['label']:<18} rho0=0 {lo:.4f} nS -> rho0=1 {hi:.4f} nS "
              f"· 배율 {hi/lo if lo > 1e-12 else float('nan'):.3f}")

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.50,
                           height_ratios=[1.15, 1.0])
    axT = fig.add_subplot(gs_[0, :2])
    axE = fig.add_subplot(gs_[0, 2])
    axF = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axC = fig.add_subplot(gs_[1, 2])

    # T: 능력 행렬
    caps_cols = [("단기", "stp"), ("장기", "ltp"), ("확률", "prob"),
                 ("postNC", "post_nc"), ("04자체", "own")]
    M = np.array([[1.0 if r[c] else 0.0 for _, c in caps_cols] for r in tab])
    axT.imshow(M, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
    axT.set_xticks(range(len(caps_cols)))
    axT.set_xticklabels([n for n, _ in caps_cols], fontsize=9)
    axT.set_yticks(range(len(tab)))
    axT.set_yticklabels([f"{r['key']}  {r['label']}" for r in tab], fontsize=9)
    for i, r in enumerate(tab):
        for j, (_, c) in enumerate(caps_cols):
            axT.text(j, i, "O" if r[c] else "-", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if r[c] else "#607d8b")
        axT.text(len(caps_cols) - 0.35, i, f"  전도도={r['gmax_via']} · 참조={r['ref']}",
                 fontsize=7.5, va="center", color="#37474f")
    axT.set_xlim(-0.5, len(caps_cols) + 2.6)
    axT.set_title("A. 능력 선언 행렬 — 배선·검증 코드는 이름이 아니라 이 표를 읽는다\n"
                  "5-9(단기가소성 검증) 대상이 '단기=O' 로 자동 결정된다",
                  fontsize=9.5, loc="left")

    # E: 동일 루프 효능
    xx = np.arange(len(rows))
    axE.bar(xx, effs, color=["#90a4ae" if not engines.get(r["key"])["ltp"]
                             else "#2e7d32" for r in rows], width=0.6)
    axE.axhline(0.5, color="#90a4ae", ls=":", lw=1.2)
    axE.set_xticks(xx); axE.set_xticklabels([r["key"] for r in rows])
    axE.set_ylim(0, 1.05); axE.set_ylabel("efficacy() [0,1]")
    for i, v in enumerate(effs):
        axE.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axE.set_title("B. 같은 API·같은 자극 -> 같은 척도\n"
                  "회색 = 장기가소성 없음(중립 0.5)", fontsize=9.5, loc="left")

    # F: 동결 계약
    keys = sorted({f["key"] for f in frz}, key=lambda k: engines.ORDER.index(k))
    r0s = sorted({f["rho0"] for f in frz})
    Mf = np.zeros((len(keys), len(r0s)))
    for f in frz:
        Mf[keys.index(f["key"]), r0s.index(f["rho0"])] = abs(f["drho"])
    axF.imshow(np.log10(Mf + 1e-16), cmap="RdYlGn_r", aspect="auto",
               vmin=-16, vmax=-3)
    axF.set_xticks(range(len(r0s)))
    axF.set_xticklabels([f"rho0={v:.1f}" for v in r0s], fontsize=8)
    axF.set_yticks(range(len(keys))); axF.set_yticklabels(keys, fontsize=9)
    for i in range(len(keys)):
        for j in range(len(r0s)):
            v = Mf[i, j]
            axF.text(j, i, "불변" if v < 1e-12 else f"{v:.1e}", ha="center",
                     va="center", fontsize=7.5,
                     color="#1b5e20" if v < 1e-12 else "#b71c1c")
    axF.set_title("C. 동결 계약 (D21) — |rho 변화|\n"
                  "gamma=0 만으로는 부족하고 rho0 가 안정 고정점이어야 한다",
                  fontsize=9.2, loc="left")

    # D: 동적범위
    xx = np.arange(len(dyn))
    axD.bar(xx - 0.19, [d["g_lo_nS"] for d in dyn], width=0.38, color="#90a4ae",
            label="rho0=0 (DOWN)")
    axD.bar(xx + 0.19, [d["g_hi_nS"] for d in dyn], width=0.38, color="#2e7d32",
            label="rho0=1 (UP)")
    axD.set_xticks(xx); axD.set_xticklabels([d["key"] for d in dyn])
    axD.set_ylabel("첫 펄스 전도도 (nS)")
    for i, d in enumerate(dyn):
        if np.isfinite(d["ratio"]):
            axD.text(i, d["g_hi_nS"], f"x{d['ratio']:.2f}", ha="center",
                     va="bottom", fontsize=7.5)
    axD.set_title("D. ★동적범위 공시 — 엔진마다 천장이 다르다\n"
                  "밝히지 않고 '이 엔진이 LTP 를 더 만든다' 고 쓰면 오도다",
                  fontsize=9.2, loc="left")
    axD.legend(fontsize=7.8)

    # C: 계약 위반 차단
    lab = ["post_nc 계약", "확률 시딩 검출", "비확률 seed() 차단", "동결 선언 일치"]
    val = [1.0 if blocked_ok else 0.0,
           1.0 if all(v["미시딩"] <= 1 for v in seed_chk.values()) else 0.0,
           nonprob_blocked / max(len(engines.with_cap("prob", False)), 1),
           1.0 if frz_consistent else 0.0]
    axC.barh(range(len(lab)), val,
             color=["#2e7d32" if v >= 0.999 else "#c62828" for v in val])
    axC.set_yticks(range(len(lab))); axC.set_yticklabels(lab, fontsize=8.5)
    axC.invert_yaxis(); axC.set_xlim(0, 1.15); axC.set_xlabel("통과 비율")
    for i, v in enumerate(val):
        axC.text(v, i, f" {v*100:.0f}%", va="center", fontsize=8)
    axC.set_title("E. 계약을 **행동으로** 검사한다\n"
                  "선언만 두면 조용히 틀린다 (D22·D26 의 교훈)",
                  fontsize=9.2, loc="left")

    fig.suptitle("5-8  레지스트리 + 어댑터 계약 — 엔진을 하나의 API 로 (lib/engines.py)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.90)
    plots.stamp(fig, f"5-8 | 등록 {len(engines.ORDER)}종 (5-7 GluSynapse 는 미결#4 대기) · "
                     f"동일 루프 구동 OK · 계약 4건 행동 검사 · "
                     f"동적범위 배율 {min(d['ratio'] for d in dyn if np.isfinite(d['ratio'])):.2f}"
                     f"~{max(d['ratio'] for d in dyn if np.isfinite(d['ratio'])):.2f}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-8_engine_matrix.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    ltp_keys = engines.with_cap("ltp")
    checks = [
        (f"등록 엔진 {len(engines.ORDER)}종이 **같은 루프**로 구동된다", len(rows) == len(engines.ORDER)),
        ("★efficacy() 가 모두 [0,1] 안이다 (계약 1)",
         all(0.0 <= v <= 1.0 for v in effs)),
        ("장기가소성 없는 엔진은 중립 0.5 를 준다",
         all(abs(r["eff"] - 0.5) < 1e-12 for r in rows
             if not engines.get(r["key"])["ltp"])),
        ("장기가소성 엔진은 rho 상태를 실제로 기록한다",
         all(r["has_rho"] for r in rows if engines.get(r["key"])["ltp"])),
        ("★post_nc 계약 위반이 **오류로** 차단된다 (계약 2)", blocked_ok),
        ("★확률 엔진 미시딩이 검출된다 (방출 <= 1회)",
         all(v["미시딩"] <= 1 for v in seed_chk.values())),
        ("확률 엔진은 시딩하면 정상 작동한다",
         all(v["시딩"] > 1 for v in seed_chk.values())),
        ("비확률 엔진에 seed() 를 부르면 차단된다",
         nonprob_blocked == len(engines.with_cap("prob", False))),
        ("★동결 선언(freeze + freeze_rho0)이 실제 거동과 일치한다 (계약 3 · D21)",
         frz_consistent),
        ("동적범위가 엔진마다 공시된다 (전 엔진 배율 산출)",
         all(np.isfinite(d["ratio"]) for d in dyn)),
        ("★GB 계열과 고전 STDP 의 동적범위 배율이 같다 (같은 효능 축에 맞춘 결과)",
         abs(next(d["ratio"] for d in dyn if d["key"] == "A")
             - next(d["ratio"] for d in dyn if d["key"] == "stdp")) < 0.02),
        ("5-9 대상(단기가소성 보유)이 선언으로 자동 결정된다",
         engines.with_cap("stp") == ["det", "B", "C"]),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(engines=engines.ORDER, n_engines=len(engines.ORDER),
               table=tab, syn_class=cls,
               protocol=dict(n_burst=N_BURST, burst_hz=BURST_HZ, n_in=N_IN,
                             in_hz=IN_HZ, n_pulses=len(pre), tstop=TSTOP,
                             dt=DT_SIM, rho0=0.0),
               uniform_loop=rows,
               contract_post_nc=viol, contract_post_nc_ok=blocked_ok,
               contract_seeding=seed_chk, nonprob_seed_blocked=nonprob_blocked,
               freeze=frz, freeze_consistent=frz_consistent,
               dynamic_range=dyn,
               stp_targets=engines.with_cap("stp"),
               pending=("5-7 GluSynapseCa 미등록 — 미결#4(외부 소스 라이선스) 결정 대기. "
                        "레지스트리에 자리와 계약(post_nc=False · ref=glusyn)을 주석으로 "
                        "미리 못박아 두었다. 등록되면 이 스크립트를 고치지 않아도 "
                        "5-8·5-9·5-11 검증에 자동으로 포함된다 — 그것이 이 설계의 목표다."),
               finding=("능력 선언을 단일 출처(lib/engines.py)로 모으고 lib/synprobe 가 투영만 "
                        "받게 했다. 같은 표를 두 곳에 두면 반드시 어긋난다. 계약은 선언만으로 "
                        "부족하고 **행동으로** 검사해야 한다 — post_nc 위반은 RuntimeError 로 "
                        "차단되고, 확률 미시딩은 방출 횟수로 검출되며, 동결은 rho 변화로 "
                        "확인한다(D21 의 rho0 조건까지)."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-8_registry.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-8 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
