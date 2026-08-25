# -*- coding: utf-8 -*-
"""5-7 엔진 GluSynapse — 칼슘을 **국소 전압에서** 만든다 (GAPS G3·G5 반증 도구)

단계   : 5-7 (5단계 가소성 엔진 / 하위 7 glusyn)
쉬운 설명: GB 계열은 "후시냅스 스파이크가 났다" 는 **사건**만 알고 그때마다 **같은 양**의
          칼슘을 넣는다. 실제 스파인은 그렇지 않다 — **그 자리의 전압**이 얼마나 올랐는지,
          그리고 **글루타메이트가 실제로 나왔는지**에 따라 칼슘이 달라진다.
          이 엔진은 그 차이를 만든다. 그래서 우리가 등재한 결핍 두 개를 직접 시험할 수 있다.
★중요  : 이것은 **Chindemi 2022 의 재현이 아니다.** 외부 소스를 쓰지 않고 개념만 가져와
          04 가 직접 작성했다(미결#4 해소 방식 — 사용자 결정 2026-08-25).
          모든 정량 주장은 "우리 구현" 으로 보고한다. 상세는 mechanisms/GluSynapseCa.md.
방법   : 전압 클램프에 **bAP 파형을 재생**해 국소 전압을 정확히 제어한다
          (`lib/synprobe.play_voltage`). 3-9 가 실측한 **위치별 bAP 진폭**을 그대로 입력으로
          쓰므로 위치 의존성 검사가 직접적이다.
          (A) 칼슘 두 갈래(NMDA·VDCC)가 조건에 따라 갈리는가
          (B) 교정 — 표준 짝이 GB 와 비슷한 c_max 를 내도록 (칼슘은 구동항에 선형 -> 정확히 풀림)
          (C) ★G3 반증 — 16지점 bAP 진폭에서 칼슘·효능이 달라지는가 (GB 는 상수)
          (D) ★G5 반증 — 방출 실패 시 칼슘이 사라지는가 (GB 는 그대로 넣는다)
          (E) 계약 — post sentinel 차단 · 전달이 GB 와 정합(5-10)
검증   : mod↔numpy 참조 대조 + G3·G5 판정 + 계약.
결과   : figures/5-7_spine_calcium.png · figures/5-7_glusyn.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\7_glusyn\\5-7_spine_calcium.py
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
from lib.refs import gb, glusyn                      # noqa: E402
from lib.wiring import load_synapse_cfg              # noqa: E402

KEY = "glu"
DT = 0.025
T0 = 20.0
DT_PAIR = 10.0                # pre -> bAP 봉우리까지 (양수 = pre 먼저)
V_REST = -69.55               # 2-4 실측 정지전위
TOL_CA_REL = 2e-3             # D23 기준

# 교정 목표 (GB 와 같은 문턱 아래에서 비교되도록)
TARGET_PAIR_CMAX = 1.23       # 5-1 실측: GB 짝 dt=+10ms 의 c_max
TARGET_POST_CMAX = 0.2759     # GB 의 C_post (post 단독 기여)

# 유도 프로토콜 (짧게 — 16지점을 돌아야 한다)
N_BURST, BURST_HZ, N_IN, IN_HZ = 5, 5.0, 4, 100.0


def bap_train(t, post_times, amp_mV):
    """여러 bAP 를 합성해 국소 전압 파형을 만든다 (정지전위 기준 합)."""
    v = np.full_like(t, V_REST)
    for tp in post_times:
        v = v + (glusyn.bap_waveform(t, tp, amp_mV, v_rest=V_REST) - V_REST)
    return v


def main():
    plots.setup()
    print("=== 5-7 엔진 GluSynapse (국소 전압 칼슘) ===")
    cls, P = load_synapse_cfg()
    G = gb.WITTENBERG2006
    e = engines.get(KEY)
    print(f"  mod: mechanisms/GluSynapseCa.mod (04 자체 작성) · 능력 {engines.caps(KEY)}")
    print(f"  ★Chindemi 2022 재현이 아니라 개념의 우리 구현 (mechanisms/GluSynapseCa.md)")

    # 3-9 위치별 bAP 실측
    bpath = os.path.join(ROOT, "03_synapse", "9_bap", "figures", "3-9_bap.json")
    B = json.load(open(bpath, "r", encoding="utf-8"))
    sites = sorted(B["sites"], key=lambda s: s["path_um"])
    ref_site = min(sites, key=lambda s: abs(s["path_um"] - 144.3))
    AMP_REF = ref_site["bap_mV"]
    print(f"  기준 위치: {ref_site['domain']} {ref_site['section']} "
          f"{ref_site['path_um']:.1f}um · bAP {AMP_REF:.2f} mV (3-9 실측)")
    print(f"  16지점 bAP 범위 {min(s['bap_mV'] for s in sites):.2f} ~ "
          f"{max(s['bap_mV'] for s in sites):.2f} mV")

    def probe(k_nmda, k_vdcc, rho0=0.0, frozen=False):
        p = SynProbe(e["mech"], clamp=True, v_hold=V_REST, rec_dt=DT)
        p.set_gmax(P["g_nS"])
        engines.apply_params(p.syn, KEY, P, rho0=rho0, frozen=frozen)
        p.set(k_nmda=k_nmda, k_vdcc=k_vdcc)
        return p

    def run(pre_times, post_times, amp_mV, tstop, k_nmda=1.0, k_vdcc=0.0,
            rho0=0.0, frozen=False):
        t = np.arange(0.0, tstop + 0.5 * DT, DT)
        v = bap_train(t, post_times, amp_mV)
        p = probe(k_nmda, k_vdcc, rho0=rho0, frozen=frozen)
        if pre_times:
            p.drive_pre(pre_times)
        p.play_voltage(t, v)
        p.record()
        R = p.run(tstop, dt=DT)
        # 기록 벡터와 재생 격자의 표본 수가 1 어긋날 수 있다 -> 짧은 쪽으로 맞춘다
        n = min(len(R["t"]), t.size)
        R = {k: (vv[:n] if hasattr(vv, "__len__") and not isinstance(vv, list) else vv)
             for k, vv in R.items()}
        return R, t[:n], v[:n]

    # ── (B) 교정 — 칼슘이 구동항에 선형이라 정확히 풀린다 ──────────────────
    print(f"\n  [B] 교정 — 표준 짝이 GB 의 c_max {TARGET_PAIR_CMAX} 를 내도록")
    TS = 400.0
    pre1, post1 = [T0], [T0 + DT_PAIR]
    Rn, tt, vv = run(pre1, post1, AMP_REF, TS, k_nmda=1.0, k_vdcc=0.0)
    Rv, _, _ = run(pre1, post1, AMP_REF, TS, k_nmda=0.0, k_vdcc=1.0)
    Rpost_v, _, _ = run([], post1, AMP_REF, TS, k_nmda=0.0, k_vdcc=1.0)
    c_n, c_v = Rn["c"], Rv["c"]
    K_VDCC = TARGET_POST_CMAX / float(Rpost_v["c"].max())
    # 짝의 c_max = max(k_n*c_n + K_VDCC*c_v) = 목표 -> k_n 을 스칼라 탐색 (단조)
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float((mid * c_n + K_VDCC * c_v).max()) < TARGET_PAIR_CMAX:
            lo = mid
        else:
            hi = mid
    K_NMDA = 0.5 * (lo + hi)
    print(f"      k_vdcc = {K_VDCC:.5f}  (post 단독 c_max -> {TARGET_POST_CMAX})")
    print(f"      k_nmda = {K_NMDA:.5f}  (짝 c_max -> {TARGET_PAIR_CMAX})")
    Rchk, _, _ = run(pre1, post1, AMP_REF, TS, k_nmda=K_NMDA, k_vdcc=K_VDCC)
    print(f"      확인: 짝 c_max {float(Rchk['c'].max()):.4f} · "
          f"선형 예측 {float((K_NMDA*c_n + K_VDCC*c_v).max()):.4f} "
          f"(차 {abs(float(Rchk['c'].max()) - float((K_NMDA*c_n+K_VDCC*c_v).max())):.2e})")
    lin_err = abs(float(Rchk["c"].max())
                  - float((K_NMDA * c_n + K_VDCC * c_v).max()))

    # ── 참조 대조 ─────────────────────────────────────────────────────────
    prm = dict(k_nmda=K_NMDA, k_vdcc=K_VDCC, e_ca=40.0,
               vh_vdcc=-30.0, slope_vdcc=7.0, norm_mV=100.0,
               tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
               NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"], tau_ca=G["tau_ca"])
    c_ref = glusyn.calcium(tt, vv, pre1, w=1.0, p=prm)
    err_c = float(np.max(np.abs(Rchk["c"] - c_ref)))
    err_rel = err_c / max(float(Rchk["c"].max()), 1e-12)
    print(f"      mod↔참조 칼슘 절대차 {err_c:.3e} (상대 {err_rel:.3e})")

    # ── (A) 네 조건에서 갈래가 갈리는가 ───────────────────────────────────
    print(f"\n  [A] 조건별 칼슘 (기준 위치 bAP {AMP_REF:.1f} mV)")
    CASES = [("pre 만 (방출 O · bAP X)", pre1, []),
             ("post 만 (bAP · 방출 X = **방출 실패**)", [], post1),
             ("짝 dt=+10ms (방출 O · bAP)", pre1, post1),
             ("역순 dt=-10ms (bAP 먼저)", [T0 + DT_PAIR], [T0])]
    condA = []
    for name, pr, po in CASES:
        R, t_, v_ = run(pr, po, AMP_REF, TS, k_nmda=K_NMDA, k_vdcc=K_VDCC)
        Rn_, _, _ = run(pr, po, AMP_REF, TS, k_nmda=K_NMDA, k_vdcc=0.0)
        Rv_, _, _ = run(pr, po, AMP_REF, TS, k_nmda=0.0, k_vdcc=K_VDCC)
        cm = float(R["c"].max())
        condA.append(dict(name=name, c_max=cm,
                          c_nmda=float(Rn_["c"].max()), c_vdcc=float(Rv_["c"].max()),
                          over_d=float(DT * np.sum(R["c"] > G["theta_d"])),
                          over_p=float(DT * np.sum(R["c"] > G["theta_p"])),
                          t=t_, v=v_, c=R["c"]))
        a = condA[-1]
        sign = "강화" if a["over_p"] > 0 else ("약화" if a["over_d"] > 0 else "변화없음")
        print(f"      {name:<38} c_max {cm:.4f} "
              f"(NMDA {a['c_nmda']:.4f} · VDCC {a['c_vdcc']:.4f}) -> {sign}")

    # ── (A2) 동시성 선택성을 측정한다 (가정하지 않는다) ────────────────────
    _pre_only = condA[0]; _pair = condA[2]
    coin_gain = _pair["c_nmda"] / max(_pre_only["c_nmda"], 1e-12)
    mg_rest = 1.0 / (1.0 + np.exp(0.062 * -V_REST) * (P["mg_mM"] / 2.62))
    v_pk = V_REST + AMP_REF
    mg_pk = 1.0 / (1.0 + np.exp(0.062 * -v_pk) * (P["mg_mM"] / 2.62))
    print(f"\n  [A2] 동시성 선택성 = 짝 NMDA / pre 단독 NMDA = {coin_gain:.3f}")
    print(f"      Mg 게이트: 안정막전위 {V_REST:.1f}mV 에서 {100*mg_rest:.1f}% 개방 · "
          f"bAP 봉우리 {v_pk:.1f}mV 에서 {100*mg_pk:.1f}% ({mg_pk/mg_rest:.0f}배)")
    print(f"      그런데 tau_d_NMDA = {P['tau_d_NMDA']:.1f}ms 라 안정막전위 누출이 "
          f"길게 누적되고 bAP 는 수 ms 뿐이다")
    print(f"      -> 게이트 비는 {mg_pk/mg_rest:.0f}배인데 실제 칼슘 이득은 "
          f"{coin_gain:.2f}배에 그친다. **7단계 설계의 실마리다.**")

    # ── (D) ★G5 — 방출 실패 시 칼슘 ───────────────────────────────────────
    print(f"\n  [D] ★G5 반증 — 방출 성공 vs 실패")
    succ = next(a for a in condA if "짝 dt=+10ms" in a["name"])
    fail = next(a for a in condA if "방출 실패" in a["name"])
    ours_drop = (succ["c_max"] - fail["c_max"]) / succ["c_max"]
    # GB(ca_stp=0) 는 방출 여부와 무관하게 C_pre 를 넣는다 -> 두 조건이 같다
    # ★GB(ca_stp=0)는 방출 여부를 모른다 — pre 스파이크가 오면 무조건 C_pre 를 넣는다.
    #   그래서 '성공' 과 '실패' 의 입력이 **완전히 같다**. 두 번 계산하지 않고 그 사실을 쓴다.
    gb_succ = float(gb.calcium(tt, [T0], [T0 + DT_PAIR], G).max())
    gb_fail = gb_succ
    gb_drop = 0.0
    print(f"      우리 엔진: 성공 {succ['c_max']:.4f} -> 실패 {fail['c_max']:.4f} "
          f"({100*ours_drop:.1f}% 감소)")
    print(f"      GB(ca_stp=0): 성공 {gb_succ:.4f} -> 실패 {gb_fail:.4f} "
          f"({100*gb_drop:.1f}% 감소) — 방출 실패를 구분하지 못한다")

    # ── (B2) ★트레인 정합 교정 — 단발 짝 교정은 트레인에서 안 맞는다 ───────
    bi, ii = 1000.0 / BURST_HZ, 1000.0 / IN_HZ
    pre_b = [T0 + b * bi + k * ii for b in range(N_BURST) for k in range(N_IN)]
    post_b = [t + DT_PAIR for t in pre_b]
    TSB = pre_b[-1] + 800.0
    tb = np.arange(0.0, TSB + 0.5 * 0.1, 0.1)
    gb_burst_cmax = float(gb.calcium(tb, pre_b, post_b, G).max())
    Rbn, _, _ = run(pre_b, post_b, AMP_REF, TSB, k_nmda=1.0, k_vdcc=0.0)
    Rbv, _, _ = run(pre_b, post_b, AMP_REF, TSB, k_nmda=0.0, k_vdcc=1.0)
    n_b = min(Rbn["c"].size, Rbv["c"].size)
    pair_cal_burst = float((K_NMDA * Rbn["c"][:n_b] + K_VDCC * Rbv["c"][:n_b]).max())
    S_TR = gb_burst_cmax / pair_cal_burst          # 공통 배율 (갈래 비율 보존)
    KN_TR, KV_TR = K_NMDA * S_TR, K_VDCC * S_TR
    print(f"\n  [B2] ★트레인 정합 교정 — 단발 짝 교정은 트레인에서 안 맞는다")
    print(f"      단발 짝 교정으로 버스트를 돌리면 c_max {pair_cal_burst:.2f} "
          f"vs GB {gb_burst_cmax:.3f}  ->  {pair_cal_burst/gb_burst_cmax:.1f}배 과다")
    print(f"      원인: 연속 유입은 이산 점프와 누적 방식이 다르다 "
          f"(tau_d_NMDA {P['tau_d_NMDA']:.1f}ms 꼬리 x {len(pre_b)}펄스)")
    print(f"      -> 갈래 비율을 보존한 공통 배율 {S_TR:.5f} 적용 "
          f"(k_nmda {KN_TR:.5f} · k_vdcc {KV_TR:.5f})")
    print(f"      ★단발용과 트레인용 교정이 다르다는 것 자체가 두 정식화의 실질적 차이다.")

    # ── (B3) ★결과 정합 교정 — 스칼라 하나로는 두 정식화를 비교할 수 없다 ──
    gb_rho_ref = float(gb.integrate_rho(tb, gb.calcium(tb, pre_b, post_b, G),
                                        rho0=0.0, p=G)[-1])
    print(f"\n  [B3] ★결과 정합 교정 — 봉우리를 맞추면 결과가 안 맞는다")
    print(f"      트레인 정합(c_max 일치)으로 기준 위치를 돌리면:")
    Rtr, _, _ = run(pre_b, post_b, AMP_REF, TSB, k_nmda=KN_TR, k_vdcc=KV_TR)
    print(f"        c_max {float(Rtr['c'].max()):.3f} (GB {gb_burst_cmax:.3f} 와 일치) "
          f"이지만 rho {float(Rtr['rho'][-1]):.4f} vs GB {gb_rho_ref:.4f}")
    print(f"      원인: GB 는 짧고 높은 점프, 우리는 넓은 고원 — 봉우리를 맞추면 "
          f"문턱 초과 **시간**이 어긋난다.")
    print(f"      -> **최종 효능**을 기준으로 다시 교정한다 (6-8 비교의 기준)")
    lo_s, hi_s = 1e-3, 5.0
    for _ in range(28):
        mid = 0.5 * (lo_s + hi_s)
        Rm, _, _ = run(pre_b, post_b, AMP_REF, TSB,
                       k_nmda=K_NMDA * mid, k_vdcc=K_VDCC * mid)
        if float(Rm["rho"][-1]) < gb_rho_ref:
            lo_s = mid
        else:
            hi_s = mid
    S_OUT = 0.5 * (lo_s + hi_s)
    KN_OUT, KV_OUT = K_NMDA * S_OUT, K_VDCC * S_OUT
    Rout, _, _ = run(pre_b, post_b, AMP_REF, TSB, k_nmda=KN_OUT, k_vdcc=KV_OUT)
    print(f"      공통 배율 {S_OUT:.5f} (k_nmda {KN_OUT:.5f} · k_vdcc {KV_OUT:.5f})")
    print(f"      확인: 기준 위치 rho {float(Rout['rho'][-1]):.4f} vs GB {gb_rho_ref:.4f} "
          f"(차 {abs(float(Rout['rho'][-1])-gb_rho_ref):.2e}) · "
          f"c_max {float(Rout['c'].max()):.3f}")
    out_err = abs(float(Rout["rho"][-1]) - gb_rho_ref)

    # ── (C) ★G3 — 위치 의존성 (결과 정합 교정으로) ────────────────────────
    print(f"\n  [C] ★G3 반증 — 16지점 x theta 버스트 {N_BURST}회 "
          f"({len(pre_b)}펄스, {TSB/1000:.1f}초) · **결과 정합 교정**")
    g3 = []
    for s in sites:
        R, _, _ = run(pre_b, post_b, s["bap_mV"], TSB,
                      k_nmda=KN_OUT, k_vdcc=KV_OUT, rho0=0.0)
        g3.append(dict(domain=s["domain"], section=s["section"],
                       path_um=s["path_um"], bap_mV=s["bap_mV"],
                       c_max=float(R["c"].max()),
                       over_p=float(DT * np.sum(R["c"] > G["theta_p"])),
                       rho_end=float(R["rho"][-1])))
    for r in g3[::3] + [g3[-1]]:
        print(f"      {r['domain']:<7}{r['path_um']:>7.1f}um bAP {r['bap_mV']:>6.2f}mV "
              f"-> c_max {r['c_max']:.3f} · theta_p 초과 {r['over_p']:7.1f}ms · "
              f"rho {r['rho_end']:.4f}")
    # GB 대조 (위치와 무관하게 같다)
    Rgb_c = gb.calcium(tb, pre_b, post_b, G)
    gb_rho = float(gb.integrate_rho(tb, Rgb_c, rho0=0.0, p=G)[-1])
    print(f"      GB(상수 칼슘): 전 지점 동일 c_max {float(Rgb_c.max()):.3f} · "
          f"rho {gb_rho:.4f}")
    rho_span = max(r["rho_end"] for r in g3) - min(r["rho_end"] for r in g3)
    cmax_span = max(r["c_max"] for r in g3) - min(r["c_max"] for r in g3)
    dist_far = [r for r in g3 if r["bap_mV"] < 20.0]
    print(f"      -> 우리 엔진 rho 범위 {min(r['rho_end'] for r in g3):.4f} ~ "
          f"{max(r['rho_end'] for r in g3):.4f} (폭 {rho_span:.4f}) vs GB 는 폭 0")

    # ── (E) 계약 ──────────────────────────────────────────────────────────
    print(f"\n  [E] 계약")
    p = SynProbe(e["mech"], clamp=True, v_hold=V_REST, rec_dt=DT)
    try:
        p.drive_post([T0]); blocked = False
    except RuntimeError:
        blocked = True
    print(f"      post sentinel 차단: {'O' if blocked else 'X'} (post_nc=False)")
    trans = []
    for r0 in (0.0, 0.5, 1.0):
        vals = {}
        for kk, mech in ((KEY, e["mech"]), ("A", engines.mech("A"))):
            q = SynProbe(mech, clamp=True, v_hold=V_REST, rec_dt=DT)
            q.set_gmax(P["g_nS"])
            engines.apply_params(q.syn, kk, P, rho0=r0, frozen=True)
            if kk == KEY:
                q.set(k_nmda=0.0, k_vdcc=0.0)      # 칼슘 끄고 전달만
            q.drive_pre([T0])
            vals[kk] = float(q.run(T0 + 60.0, dt=DT)["g"].max()) * 1e3
        rel = abs(vals[KEY] - vals["A"]) / vals["A"]
        trans.append(dict(rho0=r0, glu_nS=vals[KEY], gb_nS=vals["A"], rel=rel))
        print(f"      rho0 {r0:.1f}: glu {vals[KEY]:.5f} nS · GB {vals['A']:.5f} nS · "
              f"상대차 {rel:.2e}")
    max_trans = max(t_["rel"] for t_ in trans)

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.8))
    gs_ = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.50)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1])
    axC = fig.add_subplot(gs_[0, 2])
    axD = fig.add_subplot(gs_[1, 0])
    axE = fig.add_subplot(gs_[1, 1])
    axF = fig.add_subplot(gs_[1, 2])

    # A: 입력 전압 파형 + 칼슘
    a = condA[2]
    ax2 = axA.twinx()
    axA.plot(a["t"], a["v"], color="#90a4ae", lw=1.3, label="국소 전압(재생)")
    ax2.plot(a["t"], a["c"], color="#c62828", lw=1.8, label="칼슘 c")
    ax2.axhline(G["theta_p"], color="#2e7d32", ls=":", lw=1.2)
    ax2.axhline(G["theta_d"], color="#ef6c00", ls=":", lw=1.2)
    axA.set_xlim(10, 120); axA.set_xlabel("시간 (ms)")
    axA.set_ylabel("국소 Vm (mV)", color="#607d8b")
    ax2.set_ylabel("칼슘 c", color="#c62828")
    axA.set_title(f"A. 입력 — bAP 파형을 클램프에 재생\n"
                  f"진폭 {AMP_REF:.1f} mV = 3-9 실측 (기저 {ref_site['path_um']:.0f}um)",
                  fontsize=9.2, loc="left")

    # B: 조건별 두 갈래
    nm = ["pre 만", "post 만\n(=방출 실패)", "짝\n+10ms", "역순\n-10ms"]
    xx = np.arange(len(condA))
    axB.bar(xx, [a["c_nmda"] for a in condA], width=0.55, color="#1565c0",
            label="NMDA 갈래")
    axB.bar(xx, [a["c_vdcc"] for a in condA], width=0.55,
            bottom=[a["c_nmda"] for a in condA], color="#f9a825", label="VDCC 갈래")
    axB.plot(xx, [a["c_max"] for a in condA], "ko", ms=6, label="실제 c_max")
    axB.axhline(G["theta_p"], color="#2e7d32", ls="--", lw=1.2)
    axB.axhline(G["theta_d"], color="#ef6c00", ls=":", lw=1.2)
    axB.set_xticks(xx); axB.set_xticklabels(nm, fontsize=7.5)
    axB.set_ylabel("칼슘 c_max")
    axB.set_title("B. 두 갈래가 조건에 따라 갈린다\n"
                  "짝에서만 NMDA 가 크게 열린다 (동시성 검출)", fontsize=9.2, loc="left")
    axB.legend(fontsize=7.3)

    # C: G5
    axC.bar([0, 1], [succ["c_max"], fail["c_max"]], width=0.5, color="#2e7d32",
            label="우리 엔진")
    axC.bar([2.2, 3.2], [gb_succ, gb_fail], width=0.5, color="#c62828",
            label="GB (ca_stp=0)")
    axC.set_xticks([0, 1, 2.2, 3.2])
    axC.set_xticklabels(["성공", "실패", "성공", "실패"], fontsize=8)
    axC.set_ylabel("칼슘 c_max")
    for x, v_ in ((0, succ["c_max"]), (1, fail["c_max"]), (2.2, gb_succ), (3.2, gb_fail)):
        axC.text(x, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=7.5)
    axC.set_title(f"C. ★GAPS G5 반증 — 방출 실패\n"
                  f"우리 {100*ours_drop:.0f}% 감소 · GB 는 0% (구분 못 함)",
                  fontsize=9.2, loc="left")
    axC.legend(fontsize=7.5)

    # D: G3 — bAP 진폭 vs 칼슘
    for dom, col, mk in (("basal", "#c62828", "o"), ("apical", "#1565c0", "s")):
        rr = [r for r in g3 if r["domain"] == dom]
        axD.plot([r["bap_mV"] for r in rr], [r["c_max"] for r in rr], mk,
                 color=col, ms=6, label=dom)
    axD.axhline(float(Rgb_c.max()), color="#37474f", ls="--", lw=1.6,
                label=f"GB (상수) {float(Rgb_c.max()):.2f}")
    axD.axhline(G["theta_p"], color="#2e7d32", ls=":", lw=1.2)
    axD.set_xlabel("국소 bAP 진폭 (mV · 3-9 실측)"); axD.set_ylabel("칼슘 c_max")
    axD.set_title("D. ★GAPS G3 반증 — 칼슘이 국소 bAP 를 따라간다\n"
                  "GB 는 위치와 무관하게 한 값(점선)", fontsize=9.2, loc="left")
    axD.legend(fontsize=7.5)

    # E: G3 — 거리 vs 효능
    for dom, col, mk in (("basal", "#c62828", "o"), ("apical", "#1565c0", "s")):
        rr = [r for r in g3 if r["domain"] == dom]
        axE.plot([r["path_um"] for r in rr], [r["rho_end"] for r in rr], mk + "-",
                 color=col, ms=6, lw=1.4, label=dom)
    axE.axhline(gb_rho, color="#37474f", ls="--", lw=1.6, label=f"GB {gb_rho:.3f}")
    axE.set_xlabel("소마로부터 경로거리 (µm)"); axE.set_ylabel("유도 후 효능 rho")
    axE.set_title(f"E. 같은 프로토콜인데 위치에 따라 결과가 다르다\n"
                  f"우리 폭 {rho_span:.3f} vs GB 폭 0", fontsize=9.2, loc="left")
    axE.legend(fontsize=7.5)

    # F: 계약 + 참조 대조
    items = [("post sentinel\n차단", 1.0 if blocked else 0.0),
             ("전달 정합 (GB)\n< 1e-9", 1.0 if max_trans < 1e-9 else 0.0),
             ("mod↔참조 칼슘\n상대차 < 2e-3", 1.0 if err_rel < TOL_CA_REL else 0.0),
             ("교정 선형성\n확인", 1.0 if lin_err < 1e-6 else 0.0)]
    axF.barh(range(len(items)), [v for _, v in items],
             color=["#2e7d32" if v >= 0.999 else "#c62828" for _, v in items])
    axF.set_yticks(range(len(items)))
    axF.set_yticklabels([n for n, _ in items], fontsize=7.5)
    axF.invert_yaxis(); axF.set_xlim(0, 1.2); axF.set_xticks([0, 1])
    axF.set_xticklabels(["X", "O"])
    axF.set_title(f"F. 계약·정확도\n참조 상대차 {err_rel:.1e} · 전달 {max_trans:.1e}",
                  fontsize=9.2, loc="left")

    fig.suptitle("5-7  엔진 GluSynapse — 칼슘을 국소 전압에서 만든다 "
                 "(04 자체 작성 · 개념의 우리 구현)", fontsize=12.2, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"5-7 | k_nmda {K_NMDA:.4f} · k_vdcc {K_VDCC:.4f} (교정) · "
                     f"bAP 진폭 3-9 실측 · G5 감소 {100*ours_drop:.0f}% · "
                     f"G3 rho 폭 {rho_span:.3f} (GB 0)")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-7_spine_calcium.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    pre_only, post_only, pair, rev = condA
    checks = [
        (f"★mod↔numpy 참조 칼슘 상대차 < {TOL_CA_REL:.0e} (실측 {err_rel:.2e})",
         err_rel < TOL_CA_REL),
        (f"칼슘이 구동항에 선형이다 — 교정이 정확히 풀린다 (오차 {lin_err:.1e})",
         lin_err < 1e-6),
        ("★방출 실패(post 만)는 NMDA 칼슘이 정확히 0 이다",
         post_only["c_nmda"] == 0.0),
        ("★그래도 VDCC 칼슘은 들어간다 (bAP 단독) — 지나치게 극단적이지 않다",
         post_only["c_vdcc"] > 0.0),
        (f"짝이 pre 단독보다 칼슘이 크다 (동시성 이득 {coin_gain:.2f}배 > 1)",
         coin_gain > 1.0),
        (f"★그러나 동시성 선택성이 약하다 — Mg 게이트 비 {mg_pk/mg_rest:.0f}배에 비해 "
         f"이득이 {coin_gain:.2f}배뿐 (긴 tau_d_NMDA {P['tau_d_NMDA']:.0f}ms 탓) "
         f"-> 7단계 설계 입력", coin_gain < mg_pk / mg_rest / 3.0),
        (f"★GAPS G5 반증: 방출 실패 시 칼슘이 {100*ours_drop:.0f}% 줄어든다 "
         f"(GB 는 0%)", ours_drop > 0.3),
        (f"★GAPS G3 반증: 16지점에서 **효능** 폭 {rho_span:.4f} > 0 (GB 는 정확히 0)",
         rho_span > 0.05),
        (f"칼슘도 위치에 따라 갈린다 (폭 {cmax_span:.3f})", cmax_span > 0.1),
        (f"★결과 정합 교정이 기준 위치에서 GB 와 같은 효능을 낸다 (차 {out_err:.1e})",
         out_err < 1e-3),
        ("★봉우리를 맞추는 교정으로는 결과가 맞지 않는다 — 스칼라 하나로 두 정식화를 "
         "비교할 수 없다",
         abs(float(Rtr["rho"][-1]) - gb_rho_ref) > 0.05),
        (f"★단발 짝 교정과 트레인 정합 교정이 다르다 ({pair_cal_burst/gb_burst_cmax:.1f}배) "
         f"— 두 정식화의 실질적 차이", pair_cal_burst / gb_burst_cmax > 2.0),
        ("★칼슘이 국소 bAP 진폭과 함께 커진다 (단조 경향)",
         float(np.corrcoef([r["bap_mV"] for r in g3],
                           [r["c_max"] for r in g3])[0, 1]) > 0.8),
        ("★post_nc=False 계약이 오류로 차단된다", blocked),
        (f"전달이 GB 와 같다 (최대 상대차 {max_trans:.2e} < 1e-9)", max_trans < 1e-9),
        ("레지스트리에 등록돼 능력으로 조회된다",
         KEY in engines.ORDER and not engines.caps(KEY)["post_nc"]),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(mech=e["mech"], own=True,
               provenance=("Chindemi et al. 2022 (Nat Commun 13:3038) 의 **개념만** 가져와 "
                           "04 가 직접 작성했다. 외부 소스 미사용(미결#4 해소 방식 — 사용자 "
                           "결정 2026-08-25). 그 논문은 신피질 연구이고 CA1 적용은 어차피 "
                           "외삽이다. 모든 정량 주장은 '우리 구현' 으로 보고한다."),
               doc="mechanisms/GluSynapseCa.md",
               dt=DT, v_rest=V_REST, dt_pair=DT_PAIR, syn_class=cls,
               reference_site=ref_site,
               calibration_outcome=dict(k_nmda=KN_OUT, k_vdcc=KV_OUT, scale=S_OUT,
                                        gb_rho=gb_rho_ref,
                                        our_rho=float(Rout["rho"][-1]),
                                        err=out_err,
                                        c_max=float(Rout["c"].max()),
                                        train_matched_rho=float(Rtr["rho"][-1]),
                                        note=("★스칼라 하나(c_max)를 맞추는 것으로는 두 "
                                              "정식화를 비교할 수 없다. 봉우리를 맞추면 "
                                              "문턱 초과 시간이 어긋나 결과가 달라진다 "
                                              "(GB 는 짧고 높은 점프, 우리는 넓은 고원). "
                                              "6-8 의 위치 의존 비교는 **기준 위치에서 최종 "
                                              "효능을 맞춘 뒤** 위치를 바꿔야 뜻이 있다. "
                                              "이것이 이 단계의 가장 실무적인 결론이다.")),
               calibration_train=dict(k_nmda=KN_TR, k_vdcc=KV_TR, scale=S_TR,
                                      gb_burst_cmax=gb_burst_cmax,
                                      pair_cal_burst_cmax=pair_cal_burst,
                                      overshoot_factor=pair_cal_burst / gb_burst_cmax,
                                      note=("단발 짝으로 교정한 이득을 트레인에 쓰면 c_max 가 "
                                            "GB 의 여러 배가 되어 문턱을 완전히 포화시킨다. "
                                            "연속 유입(전압·전도도 구동)과 이산 점프의 누적 "
                                            "방식이 다르기 때문이다. G3 검사는 갈래 비율을 "
                                            "보존한 공통 배율로 트레인을 정합시킨 뒤 한다. "
                                            "**두 교정이 다르다는 사실 자체가 결과다.**")),
               coincidence=dict(gain=coin_gain, mg_rest=float(mg_rest),
                                mg_peak=float(mg_pk),
                                gate_ratio=float(mg_pk / mg_rest),
                                note=("동시성 선택성이 Mg 게이트 비보다 훨씬 작다. "
                                      f"tau_d_NMDA={P['tau_d_NMDA']:.1f}ms 가 길어서 "
                                      "안정막전위의 잔여 개방이 길게 누적되고 bAP 의 강한 "
                                      "개방은 수 ms 뿐이다. 즉 **곱셈 항을 넣는 것만으로는 "
                                      "동시성 검출이 강해지지 않는다** — 빠른 칼슘 성분이나 "
                                      "초선형 항이 필요하다. 7단계 보완 설계의 직접 입력이다.")),
               calibration=dict(k_nmda=K_NMDA, k_vdcc=K_VDCC,
                                target_pair_cmax=TARGET_PAIR_CMAX,
                                target_post_cmax=TARGET_POST_CMAX,
                                linearity_err=lin_err,
                                note=("칼슘 방정식이 구동항에 선형이라 두 갈래를 따로 풀고 "
                                      "선형결합으로 정확히 교정된다. 목표값은 GB 의 값 "
                                      "(짝 c_max 1.23 · C_post 0.2759) — 같은 문턱 아래에서 "
                                      "비교되게 하려는 **우리 관례**다.")),
               ref_check=dict(abs_err=err_c, rel_err=err_rel, tol_rel=TOL_CA_REL),
               A_conditions=[{k: v for k, v in a.items() if k not in ("t", "v", "c")}
                             for a in condA],
               D_g5=dict(ours_success=succ["c_max"], ours_failure=fail["c_max"],
                         ours_drop_frac=ours_drop,
                         gb_success=gb_succ, gb_failure=gb_fail, gb_drop_frac=gb_drop),
               C_g3=dict(sites=g3, gb_c_max=float(Rgb_c.max()), gb_rho=gb_rho,
                         rho_span=rho_span, cmax_span=cmax_span,
                         protocol=dict(n_burst=N_BURST, burst_hz=BURST_HZ,
                                       n_in=N_IN, in_hz=IN_HZ, n_pulses=len(pre_b))),
               E_contract=dict(post_nc_blocked=blocked, transmission=trans,
                               max_rel=max_trans),
               finding=(f"★GAPS G3·G5 를 **둘 다 반증할 수 있는 도구가 생겼다.** "
                        f"같은 theta 버스트 프로토콜에서 GB 는 위치와 무관하게 한 값(rho "
                        f"{gb_rho:.4f})을 주지만 우리 엔진은 국소 bAP 진폭에 따라 "
                        f"{min(r['rho_end'] for r in g3):.4f}~{max(r['rho_end'] for r in g3):.4f} "
                        f"로 갈린다(폭 {rho_span:.4f}). 방출이 실패하면 칼슘이 "
                        f"{100*ours_drop:.0f}% 줄지만 GB(ca_stp=0)는 전혀 줄지 않는다. "
                        f"단 이것은 '우리 구현이 옳다' 는 뜻이 아니라 '두 가정이 갈리는 "
                        f"조건을 만들었다' 는 뜻이다 — 어느 쪽이 실험과 맞는지는 6-8·6-9 가 "
                        f"판정한다."),
               limits=("(1) bAP 파형은 3-9 의 **봉우리 진폭만** 실측이고 파형 모양은 합성이다 "
                       "(이중지수 tau_r 0.4 / tau_d 3.0 ms — 우리 선택). 절대 칼슘값은 파형에 "
                       "의존하므로 위치 **의존성**(상대 비교)이 결론이고 절대값은 아니다. "
                       "(2) k_nmda·k_vdcc 는 GB 에 맞춘 교정값이지 측정값이 아니다. "
                       "(3) VDCC 반활성 -30mV·기울기 7mV 를 이 세포의 cal/can mod 와 "
                       "정합시키지 않았다 — 확인요. "
                       "(4) 칼슘 완충·확산을 tau_ca 하나로 뭉뚱그렸다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-7_glusyn.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-7 완료 ({n_ok}/{len(checks)}) — 5단계 전부 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
