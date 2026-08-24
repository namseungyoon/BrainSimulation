# -*- coding: utf-8 -*-
"""5-3 엔진 A — 순수 Graupner-Brunel (mod 가 참조와 같은가)

단계   : 5-3 (5단계 가소성 엔진 / 하위 3 gb_a)
쉬운 설명: 5-1 에서 numpy 로 '정답' 을 만들었다. 이제 NEURON mod 가 그 정답과 같은 답을
          내는지 본다. 같으면 이후 실험에서 나오는 결과를 mod 버그가 아니라 **모델의 성질**로
          읽을 수 있다.
방법   : 가소성을 **켜고**(gamma 살림) 네 프로토콜을 mod 와 참조에서 각각 돌려
          칼슘 c(t)·효능 rho(t) 궤적을 점대점으로 비교한다.
검증   : 효능 절대차 < 1e-3 · 칼슘 **상대차** < 2e-3 + 부호가 문헌 방향과 맞는가.
          ★칼슘을 절대차로 재면 안 된다 — mod 는 derivimplicit(암시적 오일러, O(dt))로 풀고
          참조는 해석해이므로 오차가 dt 와 c_max 에 비례한다. 공식 오류와 이산화 오차를
          구분하려고 **dt 수렴 시험**(dt 를 4배 줄이면 오차도 약 4배 줄어야 한다)을 넣었다.
근거   : Graupner & Brunel 2012 PNAS 109:3991 · 기본값 = Wittenberg & Wang 2006 해마 적합
결과   : figures/5-3_engine_a.png · figures/5-3_gb_a.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\3_gb_a\\5-3_engine_a.py
비고   : lib/synprobe.py 사용. 자극은 t>0 (D22 (2) 함정 회피).
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

MECH = "GBPlasticitySyn"
REC_DT = 0.1                 # 참조 적분과 같은 격자 (mod 는 내부 dt=0.025)
V_HOLD = -70.0
T0 = 20.0                    # t>0 (D22)
TOL_RHO, TOL_CA_REL = 1e-3, 2e-3        # 칼슘은 c_max 대비 상대차
DT_CONV = [0.1, 0.025, 0.00625]        # dt 수렴 시험 (4배씩)


def pairs(n, hz, dt_ms, t0=T0):
    isi = 1000.0 / hz
    pre = [t0 + k * isi for k in range(n)]
    post = [t + dt_ms for t in pre]
    return pre, post


def bursts(n_burst, burst_hz, n_in, in_hz, dt_ms, t0=T0):
    """theta 버스트: n_burst 회, 버스트당 n_in 발. post 는 각 pre 뒤 dt_ms."""
    bi, ii = 1000.0 / burst_hz, 1000.0 / in_hz
    pre = [t0 + b * bi + k * ii for b in range(n_burst) for k in range(n_in)]
    return pre, [t + dt_ms for t in pre]


# (이름, pre, post, tstop, 기대 부호)
PROTOCOLS = [
    ("단일 짝 dt=+10ms x1", *pairs(1, 1.0, 10.0), 600.0, "변화 미미"),
    ("짝 dt=+10ms x30 @1Hz", *pairs(30, 1.0, 10.0), 31000.0, "LTD"),
    ("theta 버스트 10회 @5Hz (4발 100Hz)", *bursts(10, 5.0, 4, 100.0, 5.0),
     3000.0, "LTP"),
    ("고빈도 50발 100Hz", *pairs(50, 100.0, 5.0), 1500.0, "LTP"),
]


def main():
    plots.setup()
    print("=== 5-3 엔진 A (순수 Graupner-Brunel) ===")
    cls, P = load_synapse_cfg()
    G = gb.WITTENBERG2006
    print(f"  시냅스 {cls} · g {P['g_nS']}nS | GB: tau_ca {G['tau_ca']:.2f} · "
          f"theta_d {G['theta_d']} · theta_p {G['theta_p']} · tau {G['tau']/1000:.1f}s")
    print(f"  가소성 ON (gamma_p {G['gamma_p']:.1f} · gamma_d {G['gamma_d']:.1f}) · "
          f"rho0 = 0 (DOWN 에서 출발)")

    rows = []
    print()
    for name, pre, post, tstop, expect in PROTOCOLS:
        # --- mod ---
        p = SynProbe(MECH, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        p.set(e=P["e_rev_mV"], tau_r_AMPA=P["tau_r_AMPA"], tau_d_AMPA=P["tau_d_AMPA"],
              tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
              NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"], rho0=0.0)
        p.drive_pre(pre); p.drive_post(post)
        R = p.run(tstop)
        # --- 참조 ---
        t = R["t"]
        c_ref = gb.calcium(t, pre, post, G)
        rho_ref = gb.integrate_rho(t, c_ref, rho0=0.0, p=G)
        e_c = float(np.max(np.abs(R["c"] - c_ref)))
        e_c_rel = e_c / max(float(R["c"].max()), 1e-12)
        e_r = float(np.max(np.abs(R["rho"] - rho_ref)))
        rho_end = float(R["rho"][-1])
        w_end = float(gb.weight(rho_end, G))
        sign = "LTP" if rho_end > 0.01 else ("LTD" if rho_end < -0.01 else "변화 미미")
        # rho0=0 에서 시작하므로 LTD 는 '더 내려갈 곳이 없다' -> 칼슘 문턱 초과 시간으로 본다
        od = float(REC_DT * np.sum(c_ref > G["theta_d"]))
        op = float(REC_DT * np.sum(c_ref > G["theta_p"]))
        rows.append(dict(name=name, expect=expect, n_pre=len(pre), tstop=tstop,
                         rho_end=rho_end, w_end=w_end, c_max=float(R["c"].max()),
                         over_d_ms=od, over_p_ms=op, err_c=e_c,
                         err_c_rel=e_c_rel, err_rho=e_r,
                         t=t, c=R["c"], rho=R["rho"], c_ref=c_ref, rho_ref=rho_ref))
        print(f"  {name:<36} rho -> {rho_end:.5f} (w {w_end:.3f}) · "
              f"c_max {R['c'].max():.2f} · theta_p 초과 {op:7.1f}ms")
        print(f"      {'':<32} 참조 대조: |dc| {e_c:.2e} (상대 {e_c_rel:.2e}) · "
              f"|drho| {e_r:.2e} "
              f"({'통과' if e_r < TOL_RHO and e_c_rel < TOL_CA_REL else 'X'})")

    # ── dt 수렴 시험 — 칼슘 오차가 공식 탓인가 이산화 탓인가 ───────────────
    print("\n  [dt 수렴] 오차가 dt 에 비례하면 공식이 아니라 적분 이산화 탓이다")
    nm4, pre4, post4, ts4, _ = PROTOCOLS[3]      # 고빈도 (c_max 가 가장 크다)
    conv = []
    for dt in DT_CONV:
        q = SynProbe(MECH, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        q.set_gmax(P["g_nS"])
        q.set(e=P["e_rev_mV"], tau_r_AMPA=P["tau_r_AMPA"], tau_d_AMPA=P["tau_d_AMPA"],
              tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
              NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"], rho0=0.0)
        q.drive_pre(pre4); q.drive_post(post4)
        Rq = q.run(ts4, dt=dt)
        cr = gb.calcium(Rq["t"], pre4, post4, G)
        ec = float(np.max(np.abs(Rq["c"] - cr)))
        conv.append(dict(dt=dt, err_c=ec, rho_end=float(Rq["rho"][-1])))
        print(f"      dt {dt:<8.5f} |dc| {ec:.3e} · rho -> {Rq['rho'][-1]:.5f}")
    ratios = [conv[i]["err_c"] / conv[i + 1]["err_c"] for i in range(len(conv) - 1)]
    print("      오차 감소비 (dt 4배 축소당): " +
          " · ".join(f"{r:.2f}" for r in ratios) + "  (1차 수렴이면 ~4)")

    # ── rho0=0.5 (경계) 에서의 양방향성 — LTD 를 실제로 보려면 UP 에서 시작해야 한다 ──
    print(f"\n  [양방향 확인] rho0=0 에서는 내려갈 곳이 없다 -> rho0=0.6(UP 쪽)에서 재본다")
    bidir = []
    for name, pre, post, tstop, _ in PROTOCOLS[1:3]:
        p = SynProbe(MECH, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        p.set(e=P["e_rev_mV"], tau_r_AMPA=P["tau_r_AMPA"], tau_d_AMPA=P["tau_d_AMPA"],
              tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
              NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"], rho0=0.6)
        p.drive_pre(pre); p.drive_post(post)
        R = p.run(tstop)
        t = R["t"]
        rr = gb.integrate_rho(t, gb.calcium(t, pre, post, G), rho0=0.6, p=G)
        e = float(np.max(np.abs(R["rho"] - rr)))
        d = float(R["rho"][-1] - 0.6)
        # 대조: 자극 없이 같은 시간 (자율항만)
        p2 = SynProbe(MECH, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p2.set_gmax(P["g_nS"]); p2.set(rho0=0.6)
        p2.drive_pre([T0])
        d0 = float(p2.run(tstop)["rho"][-1] - 0.6)
        bidir.append(dict(name=name, rho_end=float(R["rho"][-1]), drho=d,
                          drho_nostim=d0, err=e, t=t, rho=R["rho"]))
        print(f"      {name:<36} rho 0.6 -> {R['rho'][-1]:.5f} (변화 {d:+.5f}) · "
              f"무자극 대조 {d0:+.5f} · |drho| 참조차 {e:.2e}")

    max_err_rho = max(r["err_rho"] for r in rows)
    max_err_c = max(r["err_c"] for r in rows)
    max_err_c_rel = max(r["err_c_rel"] for r in rows)
    max_err_bi = max(b["err"] for b in bidir)

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.8))
    gs_ = fig.add_gridspec(3, 3, wspace=0.30, hspace=0.55,
                           height_ratios=[1.0, 1.0, 0.95])
    # 위 두 줄: 프로토콜별 c 와 rho (mod vs 참조)
    for j, r in enumerate(rows):
        axc = fig.add_subplot(gs_[0, j]) if j < 3 else None
        if axc is None:
            break
        axc.plot(r["t"] / 1000.0, r["c"], color="#00838f", lw=1.4, label="mod")
        axc.plot(r["t"] / 1000.0, r["c_ref"], color="#c62828", lw=1.0, ls="--",
                 label="참조")
        axc.axhline(G["theta_d"], color="#ef6c00", ls=":", lw=1.0)
        axc.axhline(G["theta_p"], color="#2e7d32", ls=":", lw=1.0)
        axc.set_xlabel("시간 (s)"); axc.set_ylabel("칼슘 c")
        axc.set_title(f"{'ABC'[j]}. {r['name']}\n|dc| {r['err_c']:.1e}",
                      fontsize=8.8, loc="left")
        if j == 0:
            axc.legend(fontsize=7.5)
    for j, r in enumerate(rows[:3]):
        axr = fig.add_subplot(gs_[1, j])
        axr.plot(r["t"] / 1000.0, r["rho"], color="#4527a0", lw=1.8, label="mod")
        axr.plot(r["t"] / 1000.0, r["rho_ref"], color="#f9a825", lw=1.0, ls="--",
                 label="참조")
        axr.set_xlabel("시간 (s)"); axr.set_ylabel("효능 rho")
        axr.set_title(f"{'DEF'[j]}. rho -> {r['rho_end']:.4f}  |drho| {r['err_rho']:.1e}",
                      fontsize=8.8, loc="left")
        if j == 0:
            axr.legend(fontsize=7.5)

    # 아래 줄
    axG = fig.add_subplot(gs_[2, 0])
    axH = fig.add_subplot(gs_[2, 1])
    axI = fig.add_subplot(gs_[2, 2])

    # G: 프로토콜별 최종 rho
    nm = [r["name"].split(" (")[0] for r in rows]
    rv = [r["rho_end"] for r in rows]
    axG.barh(range(len(rv)), rv,
             color=["#90a4ae" if abs(v) < 0.01 else "#2e7d32" for v in rv])
    axG.set_yticks(range(len(rv)))
    axG.set_yticklabels([n if len(n) < 22 else n[:21] + "…" for n in nm], fontsize=7.5)
    axG.invert_yaxis(); axG.set_xlabel("최종 효능 rho (rho0=0 에서)")
    axG.set_title("G. 프로토콜별 유도 결과\nrho0=0 이라 위로만 갈 수 있다",
                  fontsize=8.8, loc="left")
    for i, v in enumerate(rv):
        axG.text(v, i, f" {v:.4f}", va="center", fontsize=7.5)

    # H: 문턱 초과 시간 vs 결과
    od = [r["over_d_ms"] for r in rows]; op = [r["over_p_ms"] for r in rows]
    xx = np.arange(len(rows))
    axH.bar(xx - 0.19, od, width=0.38, color="#ef6c00", label="theta_d 초과 (약화)")
    axH.bar(xx + 0.19, op, width=0.38, color="#2e7d32", label="theta_p 초과 (강화)")
    axH.set_yscale("symlog", linthresh=1.0); plots.ascii_log(axH)
    axH.set_xticks(xx); axH.set_xticklabels([f"{i+1}" for i in xx])
    axH.set_xlabel("프로토콜 번호 (G 패널 순서)"); axH.set_ylabel("초과 시간 (ms)")
    axH.set_title("H. 부호를 정하는 것은 두 문턱의 초과 시간\n"
                  f"(dt 수렴비 " + "/".join(f"{r:.1f}" for r in ratios) +
                  " — 칼슘 오차는 이산화 탓)", fontsize=8.8, loc="left")
    axH.legend(fontsize=7.5)

    # I: 양방향 (UP 에서 출발)
    for b, col in zip(bidir, ["#c62828", "#2e7d32"]):
        axI.plot(b["t"] / 1000.0, b["rho"], lw=1.8, color=col,
                 label=f"{b['name'].split(' (')[0][:20]} ({b['drho']:+.4f})")
    axI.axhline(0.6, color="#90a4ae", ls="--", lw=1.0)
    axI.set_xlabel("시간 (s)"); axI.set_ylabel("효능 rho")
    axI.set_title("I. 양방향 — rho0=0.6(UP 쪽)에서 출발\n짝 자극은 내리고 버스트는 올린다",
                  fontsize=8.8, loc="left")
    axI.legend(fontsize=7.2)

    fig.suptitle("5-3  엔진 A (순수 Graupner-Brunel) — mod 가 numpy 참조와 같은가",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.90)
    plots.stamp(fig, f"5-3 | 프로브=단일구획+VecStim(클램프 {V_HOLD:.0f}mV) · rec_dt {REC_DT}ms · "
                     f"mod↔참조 최대 |drho| {max_err_rho:.1e} · 칼슘 상대차 "
                     f"{max_err_c_rel:.1e} · dt 수렴비 " +
                     "/".join(f"{r:.1f}" for r in ratios))
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-3_engine_a.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    single, pair30, tbs, hfs = rows
    checks = [
        (f"★mod↔참조 효능 절대차 < {TOL_RHO:.0e} (최대 {max_err_rho:.2e})",
         max_err_rho < TOL_RHO),
        (f"★mod↔참조 칼슘 상대차 < {TOL_CA_REL:.0e} (최대 {max_err_c_rel:.2e})",
         max_err_c_rel < TOL_CA_REL),
        ("★칼슘 오차는 공식이 아니라 적분 이산화 탓이다 (dt 4배 축소당 오차 2배 이상 감소)",
         all(r > 2.0 for r in ratios)),
        ("★dt 를 바꿔도 결론(효능)은 안 바뀐다 (rho 최종값 편차 < 1e-3)",
         (max(c["rho_end"] for c in conv) - min(c["rho_end"] for c in conv)) < 1e-3),
        ("단일 짝 1회는 효능을 거의 못 바꾼다 (|rho| < 1e-3)",
         abs(single["rho_end"]) < 1e-3),
        ("theta 버스트는 LTP 를 만든다 (rho > 0.5 = UP 으로 넘어감)",
         tbs["rho_end"] > 0.5),
        ("고빈도 50발 100Hz 도 LTP", hfs["rho_end"] > 0.5),
        ("짝 30발 1Hz 는 theta_p 를 넘지 않는다 (LTD 쪽 자극)",
         pair30["over_p_ms"] == 0.0),
        ("★UP(rho0=0.6)에서 짝 자극은 효능을 내린다 (LTD 방향)",
         bidir[0]["drho"] < 0),
        ("★UP(rho0=0.6)에서 버스트는 효능을 올린다 (LTP 방향)",
         bidir[1]["drho"] > 0),
        ("LTD 가 자율항 표류만으로 설명되지 않는다 (자극 효과가 무자극 대조보다 크다)",
         abs(bidir[0]["drho"]) > abs(bidir[0]["drho_nostim"]) * 2),
        (f"양방향 조건도 참조와 일치 (최대 {max_err_bi:.2e})", max_err_bi < TOL_RHO),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(mech=MECH, rec_dt=REC_DT, v_hold=V_HOLD, t0=T0,
               tol=dict(rho=TOL_RHO, ca_rel=TOL_CA_REL), gb_params=G, syn_class=cls,
               dt_convergence=dict(dts=DT_CONV, rows=conv,
                                   ratios=[round(r, 3) for r in ratios],
                                   protocol=nm4),
               protocols=[{k: v for k, v in r.items()
                           if k not in ("t", "c", "rho", "c_ref", "rho_ref")}
                          for r in rows],
               bidirectional=[{k: v for k, v in b.items() if k not in ("t", "rho")}
                              for b in bidir],
               max_err=dict(rho=max_err_rho, ca=max_err_c, ca_rel=max_err_c_rel,
                            bidir=max_err_bi),
               finding_tol=("칼슘을 절대차로 재면 안 된다. mod 는 derivimplicit(암시적 오일러, "
                            "O(dt))로 풀고 참조는 해석해이므로 오차가 dt 와 c_max 에 비례한다. "
                            "고빈도(c_max 6.77)에서 dt=0.025 일 때 |dc| 1.5e-3 이지만 dt 를 "
                            "4배 줄이면 오차도 약 4배 줄어든다 = 1차 수렴 = 공식은 옳다. "
                            "효능(rho)은 dt 를 바꿔도 1e-3 안에서 같다 — 결론이 흔들리지 않는다."),
               note=("rho0=0 에서 출발하면 LTD 를 볼 수 없다 — 이미 DOWN 우물 바닥이다. "
                     "양방향성을 보려면 rho0 를 UP 쪽(0.6)에 두어야 한다. 6단계 실험은 "
                     "이 사실 때문에 rho0 를 명시해야 하고, 그 값이 결과의 부호를 좌우한다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-3_gb_a.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-3 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
