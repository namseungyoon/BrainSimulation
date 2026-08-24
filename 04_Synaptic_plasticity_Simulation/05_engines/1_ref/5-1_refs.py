# -*- coding: utf-8 -*-
"""5-1 순수 numpy 참조 구현 — 엔진을 검증할 '정답'을 먼저 만든다

단계   : 5-1 (파이프라인 5단계 가소성 엔진 / 하위 1 ref)
쉬운 설명: NEURON mod 가 맞는지 확인하려면 **비교할 정답**이 있어야 한다. 그 정답을
          NEURON 없이 numpy 로 따로 구현한다. 같은 수식을 두 번 독립으로 쓰면 한쪽 오타가
          드러난다. 이후 5-3~5-6 은 "mod 와 이 참조의 절대차" 로 통과를 판정한다.
방법   : 참조 3종을 만들고 각각의 정성적 성질을 확인한다.
          (A) Tsodyks-Markram 단기가소성 — 억압형(PC->PC) vs 촉진형(E1) 대비
          (B) Graupner-Brunel 칼슘 궤적 — 두 문턱(theta_d/theta_p) 통과 여부가 부호를 정한다
          (C) GB 효능 이중우물 — 자극이 없으면 rho 는 0 또는 1 로 수렴한다
          (D) 고전 STDP 창 — dt 만 보는 모델(GB 와의 대비)
검증   : 참조가 정성적으로 옳은가 + **mod 파라미터와 값이 일치하는가**(드리프트 방지).
근거   : Graupner & Brunel 2012 PNAS 109:3991 (기본값 = Wittenberg & Wang 2006 해마
          슬라이스 적합, G&B Table S2) · Tsodyks & Markram 1997 / Fuhrmann 2002 ·
          Bi & Poo 1998 (원문 미확보 — 5-6 전 확정 필요)
결과   : figures/5-1_refs.png · figures/5-1_refs.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\1_ref\\5-1_refs.py
비고   : ★ NEURON 을 쓰지 않는다 — 초 단위로 끝난다. 두 세포 벤치도 필요 없다.
"""
import os
import re
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
from lib import refdata                              # noqa: E402
from lib.refs import tm, gb, stdp                    # noqa: E402

MOD_PATH = os.path.join(ROOT, "mechanisms", "_build", "GBPlasticitySyn.mod")
DT = 0.025


def mod_params(path):
    """mod PARAMETER 블록에서 이름=값을 뽑는다(단위·주석 제거)."""
    src = open(path, "r", encoding="utf-8", errors="replace").read()
    m = re.search(r"PARAMETER\s*\{(.*?)\n\}", src, re.S)
    out = {}
    for line in m.group(1).splitlines():
        line = line.split(":")[0].strip()            # NMODL 주석은 ':'
        mm = re.match(r"([A-Za-z_]\w*)\s*=\s*([-\d.eE+]+)", line)
        if mm:
            out[mm.group(1)] = float(mm.group(2))
    return out


def main():
    plots.setup()
    print("=== 5-1 순수 numpy 참조 (NEURON 미사용) ===")

    # ── (A) TM 단기가소성 — 억압 vs 촉진 ──────────────────────────────────
    E = refdata.ECKER2020["PC_PC"]
    U_pp, D_pp, F_pp = E["Use"][0], float(E["Dep_ms"][0]), float(E["Fac_ms"][0])
    C = refdata.ECKER_E1_CONTRAST
    U_e1, D_e1, F_e1 = C["Use"], C["Dep_ms"], C["Fac_ms"]
    N_TR, HZ = 8, 50.0
    _, a_pp = tm.train(N_TR, HZ, U_pp, D_pp, F_pp)
    _, a_e1 = tm.train(N_TR, HZ, U_e1, D_e1, F_e1)
    ppr_pp, ppr_e1 = float(a_pp[1]), float(a_e1[1])
    print(f"\n  [A] TM {N_TR}펄스 {HZ:.0f}Hz")
    print(f"      PC->PC (Use {U_pp} · Dep {D_pp:.0f} · Fac {F_pp:.0f}) PPR {ppr_pp:.3f} "
          f"· 8번째 {a_pp[-1]:.3f}  -> {'억압' if ppr_pp < 1 else '촉진'}")
    print(f"      E1     (Use {U_e1} · Dep {D_e1:.0f} · Fac {F_e1:.0f}) PPR {ppr_e1:.3f} "
          f"· 8번째 {a_e1[-1]:.3f}  -> {'억압' if ppr_e1 < 1 else '촉진'}")

    # ── (B) GB 칼슘 궤적 — 문턱 통과가 부호를 정한다 ──────────────────────
    P = gb.WITTENBERG2006
    print(f"\n  [B] GB 칼슘 (tau_ca {P['tau_ca']:.2f}ms · C_pre {P['C_pre']} · "
          f"C_post {P['C_post']:.4f} · 지연 D {P['D']:.4f}ms)")
    print(f"      문턱: theta_d {P['theta_d']} (넘으면 약화) · theta_p {P['theta_p']} (넘으면 강화)")
    CASES = [("pre 단독", [0.0], []),
             ("post 단독", [], [0.0]),
             ("짝 dt=+10ms (pre->post)", [0.0], [10.0]),
             ("짝 dt=-10ms (post->pre)", [0.0], [-10.0]),
             ("버스트 4발 100Hz (pre) + post 1발", [0.0, 10.0, 20.0, 30.0], [5.0])]
    tB = np.arange(-40.0, 200.0 + 0.5 * DT, DT)
    traces = []
    for name, pr, po in CASES:
        c = gb.calcium(tB, pr, po, P)
        cmax = float(c.max())
        over_d = float(DT * np.sum(c > P["theta_d"]))
        over_p = float(DT * np.sum(c > P["theta_p"]))
        traces.append((name, c, cmax, over_d, over_p))
        sign = "강화" if over_p > 0 else ("약화" if over_d > 0 else "변화없음")
        print(f"      {name:<30} c_max {cmax:.3f} · theta_d 초과 {over_d:6.2f}ms · "
              f"theta_p 초과 {over_p:6.2f}ms -> {sign}")

    # ── (C) 이중우물 — 자극 없으면 0 또는 1 ───────────────────────────────
    fps = gb.fixed_points(P)
    print(f"\n  [C] 효능 이중우물 — 고정점 " +
          " · ".join(f"rho={r:.1f}({s})" for r, s in fps))
    T_REL = 3.0e6                       # 3000 s 완화 (tau 688 s)
    tC = np.arange(0.0, T_REL, 200.0)   # 자극 없음 -> 큰 dt 로 충분
    zc = np.zeros_like(tC)
    relax = {}
    for r0 in (0.40, 0.49, 0.51, 0.60):
        rr = gb.integrate_rho(tC, zc, rho0=r0, p=P)
        relax[r0] = rr
        print(f"      rho0 {r0:.2f} -> {rr[-1]:.4f} "
              f"({'DOWN' if rr[-1] < 0.5 else 'UP'} 로 수렴)")

    # ── (D) 고전 STDP 창 ─────────────────────────────────────────────────
    Q = stdp.BI_POO_1998
    dts = np.arange(-100.0, 100.01, 0.5)
    wD = stdp.window(dts, Q)
    print(f"\n  [D] 고전 STDP (tau_p {Q['tau_p']}ms · tau_d {Q['tau_d']}ms · 원문 미확보)")
    print(f"      dt=+1ms {stdp.window(np.array([1.0]), Q)[0]:+.4f} · "
          f"dt=-1ms {stdp.window(np.array([-1.0]), Q)[0]:+.4f} · "
          f"dt=0 {stdp.window(np.array([0.0]), Q)[0]:+.4f}")
    # 주파수 무관성: 같은 dt, 다른 주파수 -> 짝당 dw 가 (거의) 같다
    fr = [1.0, 5.0, 20.0, 50.0]
    per_pair = [stdp.protocol(10.0, 10, f, Q, all_to_all=False) / 10.0 for f in fr]
    print("      같은 dt(+10ms) 주파수 스윕 짝당 dw (nearest): " +
          " · ".join(f"{f:.0f}Hz {v:+.4f}" for f, v in zip(fr, per_pair)))
    # ★ 고전 STDP 도 ISI 가 창(tau 17~34ms)과 겹치면 주파수 의존적이다 — 50Hz 에서 부호까지
    #   뒤집힌다. '고전 STDP 는 주파수 무관' 은 **고립된 단일 짝**에서만 참이다.
    iso = float(stdp.pairs([0.0], [10.0], Q, all_to_all=False))
    iso_exact = float(stdp.window(np.array([10.0]), Q)[0])
    iso_err = abs(iso - iso_exact)
    lowf_spread = abs(per_pair[0] - per_pair[1]) / abs(per_pair[0])
    print(f"      고립 단일 짝 dw {iso:+.6f} (해석해 {iso_exact:+.6f} · 차 {iso_err:.2e})")
    print(f"      저주파(1 vs 5Hz) 짝당 dw 상대차 {lowf_spread*100:.2f}% · "
          f"50Hz 는 {per_pair[-1]:+.4f} 로 **부호 반전**")

    # GB 는 같은 dt 라도 주파수에 따라 달라진다 (대비 실측)
    gb_by_freq = []
    for f in fr:
        isi = 1000.0 / f
        pre = np.arange(10) * isi
        post = pre + 10.0
        tstop = float(pre[-1] + 600.0)
        tt = np.arange(0.0, tstop + 0.5 * DT, DT)
        cc = gb.calcium(tt, pre, post, P)
        gb_by_freq.append((f, float(cc.max()),
                           float(DT * np.sum(cc > P["theta_p"]))))
    print("      GB 대비 — 같은 dt(+10ms) 인데 주파수마다 다르다: " +
          " · ".join(f"{f:.0f}Hz c_max {cm:.2f}/theta_p초과 {op:.1f}ms"
                     for f, cm, op in gb_by_freq))
    gb_dep = float(gb_by_freq[-1][2] - gb_by_freq[0][2])

    # ── mod 파라미터 대조 ────────────────────────────────────────────────
    mp = mod_params(MOD_PATH)
    diffs = {k: (v, mp.get(k)) for k, v in P.items()
             if k in mp and abs(mp[k] - v) > 1e-9}
    missing = [k for k in P if k not in mp]
    print(f"\n  [대조] mod PARAMETER {len(mp)}개 파싱 · 참조와 불일치 {len(diffs)}개 · "
          f"mod 에 없음 {len(missing)}개")
    for k, (a, b) in diffs.items():
        print(f"      X {k}: 참조 {a} vs mod {b}")

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.0, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.42)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1:])
    axC = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axE = fig.add_subplot(gs_[1, 2])

    # A: TM
    x = np.arange(1, N_TR + 1)
    axA.plot(x, a_pp, "o-", color="#c62828", ms=6, lw=1.8,
             label=f"PC->PC 억압 (Fac {F_pp:.0f} < Dep {D_pp:.0f})")
    axA.plot(x, a_e1, "s-", color="#1565c0", ms=6, lw=1.8,
             label=f"E1 촉진 (Fac {F_e1:.0f} > Dep {D_e1:.0f})")
    axA.axhline(1.0, color="#90a4ae", ls="--", lw=1.0)
    axA.set_xlabel("펄스 번호"); axA.set_ylabel("정규화 방출량 (첫 펄스=1)")
    axA.set_title(f"A. TM 단기가소성 — {HZ:.0f}Hz {N_TR}펄스\n"
                  "Fac 와 Dep 의 크기 관계가 부호를 정한다", fontsize=9.5, loc="left")
    axA.legend(fontsize=8)

    # B: 칼슘
    cols = ["#8d6e63", "#26a69a", "#c62828", "#1565c0", "#6a1b9a"]
    for (name, c, cmax, od, op), col in zip(traces, cols):
        axB.plot(tB, c, lw=1.7, color=col, label=f"{name} (c_max {cmax:.2f})")
    axB.axhline(P["theta_d"], color="#ef6c00", ls="--", lw=1.4)
    axB.axhline(P["theta_p"], color="#2e7d32", ls="--", lw=1.4)
    axB.text(196, P["theta_d"], f" theta_d={P['theta_d']:g} (약화)", fontsize=8,
             color="#ef6c00", va="bottom", ha="right")
    axB.text(196, P["theta_p"], f" theta_p={P['theta_p']:g} (강화)", fontsize=8,
             color="#2e7d32", va="bottom", ha="right")
    axB.set_xlim(-40, 200); axB.set_xlabel("시간 (ms)"); axB.set_ylabel("칼슘 c (무차원)")
    axB.set_title("B. GB 칼슘 궤적 — 어느 문턱을 넘느냐가 강화/약화를 정한다\n"
                  f"pre 는 지연 D={P['D']:.1f}ms 뒤에 C_pre={P['C_pre']:g}, "
                  f"post 는 즉시 C_post={P['C_post']:.3f}", fontsize=9.5, loc="left")
    axB.legend(fontsize=7.8, loc="upper right")

    # C: 이중우물
    rr = np.linspace(0, 1, 400)
    U = gb.potential(rr, P)
    axC.plot(rr, U, color="#37474f", lw=2)
    for r0, tr in relax.items():
        axC.plot([r0], [gb.potential(r0, P)], "o", ms=7,
                 color="#c62828" if tr[-1] < 0.5 else "#2e7d32")
        axC.annotate("", xy=(tr[-1], gb.potential(tr[-1], P)),
                     xytext=(r0, gb.potential(r0, P)),
                     arrowprops=dict(arrowstyle="->", lw=1.2,
                                     color="#c62828" if tr[-1] < 0.5 else "#2e7d32"))
    axC.axvline(P["rho_star"], color="#90a4ae", ls=":", lw=1.2)
    axC.set_xlabel("효능 rho"); axC.set_ylabel("포텐셜 U(rho)")
    axC.set_title(f"C. 이중우물 — 자극 없으면 0 또는 1 로 간다\n"
                  f"경계 rho*={P['rho_star']:g} 는 불안정 (0.49→DOWN · 0.51→UP)",
                  fontsize=9.5, loc="left")

    # D: STDP 창
    axD.plot(dts, wD, color="#4527a0", lw=2)
    axD.axhline(0, color="#90a4ae", lw=1.0)
    axD.axvline(0, color="#90a4ae", ls=":", lw=1.0)
    axD.fill_between(dts, wD, 0, where=(wD > 0), color="#2e7d32", alpha=0.18)
    axD.fill_between(dts, wD, 0, where=(wD < 0), color="#c62828", alpha=0.18)
    axD.text(45, 0.55, "pre→post\nLTP", fontsize=9, color="#2e7d32", ha="center")
    axD.text(-52, -0.55, "post→pre\nLTD", fontsize=9, color="#c62828", ha="center")
    axD.set_xlabel("dt = t_post - t_pre (ms)"); axD.set_ylabel("dw (상대)")
    axD.set_title(f"D. 고전 STDP 창 (tau_p {Q['tau_p']} · tau_d {Q['tau_d']} ms)\n"
                  "※ Bi&Poo 1998 원문 미확보 — 5-6 전 확정", fontsize=9.5, loc="left")

    # E: 두 모델의 근본 차이 (주파수 의존성)
    axE.plot(fr, per_pair, "o-", color="#4527a0", ms=7, lw=2,
             label="고전 STDP 짝당 dw (좌)")
    axE.axhline(0, color="#90a4ae", lw=1.0)
    axE.set_xscale("log"); axE.set_xlabel("짝 반복 주파수 (Hz)")
    axE.set_ylabel("고전 STDP 짝당 dw", color="#4527a0")
    axE.tick_params(axis="y", labelcolor="#4527a0")
    axE2 = axE.twinx()
    gp = [g[2] for g in gb_by_freq]
    axE2.plot(fr, gp, "s--", color="#00838f", ms=7, lw=2,
              label="GB theta_p 초과 시간 (우)")
    axE2.set_ylabel("GB theta_p 초과 (ms)", color="#00838f")
    axE2.tick_params(axis="y", labelcolor="#00838f")
    axE.set_title("E. 같은 dt(+10ms)인데 둘 다 주파수 의존적 — 그러나 기전이 다르다\n"
                  "STDP=창 겹침으로 부호 반전 · GB=칼슘 누적으로 LTD -> LTP 전환",
                  fontsize=9.5, loc="left")
    h1, l1 = axE.get_legend_handles_labels(); h2, l2 = axE2.get_legend_handles_labels()
    axE.legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper left")

    fig.suptitle("5-1  순수 numpy 참조 — 엔진 검증의 '정답' (NEURON 미사용)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.88)
    plots.stamp(fig, f"5-1 | 참조 3종 (TM · GB · 고전 STDP) · GB 기본값 = Wittenberg2006 "
                     f"해마 슬라이스 적합 · mod 대조 불일치 {len(diffs)}개")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-1_refs.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    pre_only = traces[0]; post_only = traces[1]
    pair_pos = traces[2]; pair_neg = traces[3]; burst = traces[4]
    checks = [
        ("TM: PC->PC 는 억압 (PPR < 1)", ppr_pp < 1.0),
        ("TM: E1 은 촉진 (PPR > 1)", ppr_e1 > 1.0),
        ("GB: pre 단독은 theta_p 를 못 넘는다 (강화 안 됨)", pre_only[4] == 0.0),
        ("GB: post 단독은 theta_d 도 못 넘는다 (변화 없음)", post_only[3] == 0.0),
        ("GB: pre->post 짝이 pre 단독보다 칼슘이 크다",
         pair_pos[2] > pre_only[2]),
        ("GB: 버스트가 단일 짝보다 칼슘이 크다 (누적)", burst[2] > pair_pos[2]),
        ("GB: 이중우물 고정점이 0(안정)·rho*(불안정)·1(안정)",
         [s for _, s in fps] == ["stable", "unstable", "stable"]),
        ("GB: rho0=0.49 -> DOWN · 0.51 -> UP", relax[0.49][-1] < 0.5 and relax[0.51][-1] > 0.5),
        ("STDP: dt>0 은 양수 · dt<0 은 음수 · dt=0 은 0",
         bool(stdp.window(np.array([1.0]), Q)[0] > 0
              and stdp.window(np.array([-1.0]), Q)[0] < 0
              and stdp.window(np.array([0.0]), Q)[0] == 0.0)),
        ("STDP: 고립 단일 짝은 dt 만으로 정해진다 (해석해와 일치, 차 < 1e-12)",
         iso_err < 1e-12),
        ("STDP: 저주파(1~5Hz)에서는 짝당 dw 가 거의 같다 (상대차 < 1%)",
         lowf_spread < 0.01),
        ("STDP: 고빈도(50Hz)에서는 창 겹침으로 **부호가 뒤집힌다** (LTP -> LTD)",
         per_pair[-1] < 0.0),
        ("★GB: 같은 dt 에서 저빈도는 LTD(theta_p 미달) · 고빈도는 LTP(theta_p 초과) 로 **전환**",
         gb_by_freq[0][2] == 0.0 and gb_by_freq[-1][2] > 0.0),
        ("★단일 짝은 dt 부호와 무관하게 LTD 쪽 (Wittenberg&Wang 2006 해마 관측과 일치)",
         pair_pos[4] == 0.0 and pair_pos[3] > 0.0
         and pair_neg[4] == 0.0 and pair_neg[3] > 0.0),
        ("★버스트는 theta_p 를 넘어 LTP 쪽 (단일 짝과 부호가 갈린다)", burst[4] > 0.0),
        (f"참조 파라미터가 mod 와 일치 (불일치 0)", len(diffs) == 0),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(
        dt_ms=DT,
        A_tm=dict(n=N_TR, hz=HZ,
                  PC_PC=dict(Use=U_pp, Dep_ms=D_pp, Fac_ms=F_pp,
                             ppr=round(ppr_pp, 4),
                             profile=[round(float(v), 4) for v in a_pp]),
                  E1=dict(Use=U_e1, Dep_ms=D_e1, Fac_ms=F_e1,
                          ppr=round(ppr_e1, 4),
                          profile=[round(float(v), 4) for v in a_e1])),
        B_calcium=[dict(case=n, c_max=round(cm, 4),
                        over_theta_d_ms=round(od, 3), over_theta_p_ms=round(op, 3))
                   for n, _, cm, od, op in traces],
        C_bistable=dict(fixed_points=[[r, s] for r, s in fps],
                        relax={str(k): round(float(v[-1]), 5) for k, v in relax.items()},
                        t_relax_ms=T_REL),
        D_stdp=dict(params=Q, freq_hz=fr,
                    per_pair_dw=[round(v, 6) for v in per_pair],
                    isolated_pair_dw=round(iso, 8),
                    isolated_pair_exact=round(iso_exact, 8),
                    isolated_pair_err=iso_err,
                    lowfreq_rel_spread=round(lowf_spread, 6),
                    sign_flip_at_50hz=bool(per_pair[-1] < 0),
                    gb_by_freq=[dict(hz=f, c_max=round(cm, 4),
                                     over_theta_p_ms=round(op, 3))
                                for f, cm, op in gb_by_freq]),
        gb_params=P, mod_param_count=len(mp),
        mod_mismatch={k: [a, b] for k, (a, b) in diffs.items()},
        mod_missing=missing,
        resolved=dict(
            issue="칼슘 지연 D = 13.7 vs 18.8008 ms 모순 (docs/DECISIONS 미결#1)",
            finding=("04 트랙 안에는 13.7 이 존재하지 않는다 — 다른 트랙 문서에서 온 값이다. "
                     "04 의 GBPlasticitySyn.mod 와 lib/refs/gb.py 는 모두 D=18.8008ms 하나만 "
                     "쓰며, 이 값은 mod 헤더가 밝힌 대로 Wittenberg&Wang 2006 해마 슬라이스 "
                     "적합(G&B 2012 Table S2)이다. G&B 논문은 데이터셋마다 다른 적합값을 주므로 "
                     "13.7 은 다른 데이터셋(신피질 계열) 적합값으로 보인다 — 원문 미확보(확인요)."),
            decision="04 는 해마 슬라이스 적합값 18.8008ms 단일 사용. 다른 값은 쓰지 않는다."),
        notes=["Bi&Poo 1998 tau_p/tau_d 는 널리 인용되는 값이나 원문 미확보 — 5-6 전 확정",
               "GluSynapse(스파인 칼슘) 참조는 5-7 로 미룬다 — 외부 소스 라이선스 미결(#4)"],
        checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-1_refs.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-1 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
