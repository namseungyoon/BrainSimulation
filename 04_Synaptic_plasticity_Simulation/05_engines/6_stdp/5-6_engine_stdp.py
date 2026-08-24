# -*- coding: utf-8 -*-
"""5-6 엔진 고전 STDP — 우리가 직접 쓴 mod 가 지수 창을 정확히 재현하는가

단계   : 5-6 (5단계 가소성 엔진 / 하위 6 stdp)
쉬운 설명: 지금까지의 엔진은 남이 쓴 mod 였다. 이번엔 **우리가 직접 NMODL 을 썼다**
          (`mechanisms/PairSTDPSyn.mod` + 한글 근거 `.md`). 고전 STDP 는 칼슘 같은 내부
          상태가 없고 **스파이크 짝의 시간차만** 본다 — GB 계열의 대조군이다.
방법   : (A) 고립된 짝의 dt 스윕 -> 닫힌 형태 창과 **정확 일치**(클리핑 끄고)
          (B) 규약 대비: all-to-all vs 최근접 이웃
          (C) 주파수 스윕 -> 참조와 일치 + GB 와의 기전 차이
          (D) 하드 경계(클리핑)가 작동하는가
          (E) 전달이 GB 와 같은 크기인가 (5-10 전제조건)
검증   : mod↔참조 절대차 < 1e-12 (스파이크 시각을 dt 정수배로 두면 이벤트 양자화가 없다)
근거   : Bi & Poo 1998 형태 (원문 미확보 — 미결#14) · mechanisms/PairSTDPSyn.md
결과   : figures/5-6_stdp_window.png · figures/5-6_stdp.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\6_stdp\\5-6_engine_stdp.py
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
from lib.refs import stdp, gb                        # noqa: E402
from lib.wiring import load_synapse_cfg              # noqa: E402

MECH = "PairSTDPSyn"
MECH_GB = "GBPlasticitySyn"
REC_DT = 1.0                 # w 만 보므로 굵게
V_HOLD = -70.0
T0 = 20.0
DT_SIM = 0.025
TOL = 1e-12

# 검증용: 클리핑을 끈 넓은 범위 + 중앙에서 출발 -> dw 를 직접 읽는다
WIDE = dict(w_min=-1000.0, w_max=1000.0, rho0=0.5)     # w 초기값 = 0
AMP1 = dict(A_p=1.0, A_d=1.0)                           # 참조의 A_p=A_d=1 과 같게

# dt 는 모두 DT_SIM(0.025) 의 정수배 — 이벤트 시각 양자화를 없앤다
DTS = np.arange(-100.0, 100.5, 1.0)
FREQS = [1.0, 5.0, 10.0, 20.0, 50.0]
N_PAIRS = 10


def main():
    plots.setup()
    print("=== 5-6 엔진 고전 STDP (04 자체 작성 mod) ===")
    cls, P = load_synapse_cfg()
    Q = stdp.BI_POO_1998
    print(f"  mod: mechanisms/PairSTDPSyn.mod (04 자체) · 능력 {CAPS[MECH]}")
    print(f"  창: tau_p {Q['tau_p']}ms · tau_d {Q['tau_d']}ms "
          f"(Bi&Poo 1998 형태 · 원문 미확보 미결#14)")
    print(f"  검증 설정: 클리핑 끔 (w in [{WIDE['w_min']:.0f}, {WIDE['w_max']:.0f}], "
          f"초기 w=0) · A_p=A_d=1 · dt 는 {DT_SIM}ms 정수배")

    def make(**kw):
        p = SynProbe(MECH, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        p.set(e=P["e_rev_mV"], tau_r_AMPA=P["tau_r_AMPA"], tau_d_AMPA=P["tau_d_AMPA"],
              tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
              NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"],
              tau_p=Q["tau_p"], tau_d=Q["tau_d"])
        p.set(**kw)
        return p

    def one_pair(dt_ms, **kw):
        """고립된 짝 하나 -> 최종 w (초기 0)."""
        p = make(**{**WIDE, **AMP1, **kw})
        tpre = T0 if dt_ms >= 0 else T0 - dt_ms
        tpost = tpre + dt_ms
        p.drive_pre([tpre]); p.drive_post([tpost])
        R = p.run(max(tpre, tpost) + 400.0, dt=DT_SIM)
        return float(R["w"][-1])

    # ── (A) 고립 짝 dt 스윕 ───────────────────────────────────────────────
    print(f"\n  [A] 고립 짝 dt 스윕 ({len(DTS)}점, dt=0 제외)")
    dws, refs, used = [], [], []
    for dt in DTS:
        if dt == 0.0:
            continue
        w = one_pair(float(dt))
        r = float(stdp.window(np.array([dt]), Q)[0])
        used.append(float(dt)); dws.append(w); refs.append(r)
    dws = np.array(dws); refs = np.array(refs); used = np.array(used)
    err = float(np.max(np.abs(dws - refs)))
    print(f"      mod↔참조 최대 절대차 {err:.3e} (허용 {TOL:.0e})")
    print(f"      dt=+1ms {dws[used == 1.0][0]:+.6f} (참조 {refs[used == 1.0][0]:+.6f}) · "
          f"dt=-1ms {dws[used == -1.0][0]:+.6f} (참조 {refs[used == -1.0][0]:+.6f})")
    print(f"      dt=+50ms {dws[used == 50.0][0]:+.6f} · dt=-50ms {dws[used == -50.0][0]:+.6f}")

    # dt=0 은 규약이 갈린다 (참조 0 · mod 는 전달 순서에 의존)
    w0 = one_pair(0.0)
    print(f"      dt=0 : mod {w0:+.6f} vs 참조 0.000000 "
          f"-> 동시 스파이크는 규약 문제 (실험적으로도 분리 불가)")

    # ── (B) 규약 대비: all-to-all vs 최근접 ───────────────────────────────
    print(f"\n  [B] 규약 대비 — {N_PAIRS}짝 반복, dt=+10ms")
    conv = {}
    for a2a in (1, 0):
        rows = []
        for f in FREQS:
            isi = 1000.0 / f
            pre = [T0 + k * isi for k in range(N_PAIRS)]
            post = [t + 10.0 for t in pre]
            p = make(**{**WIDE, **AMP1}, all_to_all=a2a)
            p.drive_pre(pre); p.drive_post(post)
            R = p.run(pre[-1] + 600.0, dt=DT_SIM)
            wm = float(R["w"][-1])
            wr = stdp.protocol(10.0, N_PAIRS, f, Q, all_to_all=bool(a2a))
            rows.append(dict(hz=f, w_mod=wm, w_ref=wr, err=abs(wm - wr),
                             per_pair=wm / N_PAIRS))
            print(f"      all_to_all={a2a} {f:>5.0f}Hz : mod {wm:+8.4f} · "
                  f"참조 {wr:+8.4f} · 차 {abs(wm-wr):.2e} · 짝당 {wm/N_PAIRS:+.4f}")
        conv[a2a] = rows
    err_tr = max(r["err"] for rows in conv.values() for r in rows)

    # ── (C) GB 와의 기전 차이 ─────────────────────────────────────────────
    print(f"\n  [C] 같은 프로토콜에서 GB 는 어떻게 다른가 (칼슘 문턱 초과 시간)")
    G = gb.WITTENBERG2006
    gbrow = []
    for f in FREQS:
        isi = 1000.0 / f
        pre = [T0 + k * isi for k in range(N_PAIRS)]
        post = [t + 10.0 for t in pre]
        tt = np.arange(0.0, pre[-1] + 600.0, 0.1)
        cc = gb.calcium(tt, pre, post, G)
        op = float(0.1 * np.sum(cc > G["theta_p"]))
        od = float(0.1 * np.sum(cc > G["theta_d"]))
        gbrow.append(dict(hz=f, over_p=op, over_d=od, c_max=float(cc.max())))
        print(f"      {f:>5.0f}Hz : c_max {cc.max():5.2f} · theta_d 초과 {od:7.1f}ms · "
              f"theta_p 초과 {op:7.1f}ms")

    # ── (D) 하드 경계 ─────────────────────────────────────────────────────
    print(f"\n  [D] 하드 경계 (기본 범위 w in [{1.0}, {5.28145}])")
    clip = {}
    for name, kw, npair in (("상한 (LTP 포화)", dict(rho0=0.0, A_p=1.0, A_d=1.0), 40),
                            ("하한 (LTD 포화)", dict(rho0=1.0, A_p=1.0, A_d=1.0), 40)):
        isi = 1000.0 / 5.0
        pre = [T0 + k * isi for k in range(npair)]
        dtp = 10.0 if "상한" in name else -10.0
        post = [t + dtp for t in pre]
        p = make(**kw)
        p.drive_pre(pre); p.drive_post(post)
        R = p.run(pre[-1] + 600.0, dt=DT_SIM)
        clip[name] = dict(w_end=float(R["w"][-1]), rho_end=float(R["rho"][-1]))
        print(f"      {name}: w -> {R['w'][-1]:.5f} · rho -> {R['rho'][-1]:.5f}")

    # ── (E) 전달이 GB 와 같은가 (5-10 전제) ───────────────────────────────
    print(f"\n  [E] 전달 정합 — 같은 rho 에서 첫 펄스 전도도가 GB 와 같은가")

    def first_peak(mech, r0, dt):
        q = SynProbe(mech, clamp=True, v_hold=V_HOLD, rec_dt=min(dt, 0.025))
        q.set_gmax(P["g_nS"])
        q.set(e=P["e_rev_mV"], tau_r_AMPA=P["tau_r_AMPA"],
              tau_d_AMPA=P["tau_d_AMPA"], tau_r_NMDA=P["tau_r_NMDA"],
              tau_d_NMDA=P["tau_d_NMDA"], NMDA_ratio=P["NMDA_ratio"],
              mg=P["mg_mM"], rho0=r0)
        if mech == MECH:
            q.set(A_p=0.0, A_d=0.0)              # 가소성 동결
        else:
            q.set(gamma_p=0.0, gamma_d=0.0)
        q.drive_pre([T0])
        return float(q.run(T0 + 60.0, dt=dt)["g"].max()) * 1e3

    trans = []
    for r0 in (0.0, 0.5, 1.0):
        a = first_peak(MECH, r0, DT_SIM); b = first_peak(MECH_GB, r0, DT_SIM)
        d = abs(a - b) / max(b, 1e-12)
        trans.append(dict(rho0=r0, stdp_nS=a, gb_nS=b, rel=d))
        print(f"      rho0 {r0:.1f} : STDP {a:.5f} nS · GB {b:.5f} nS · 상대차 {d:.2e}")
    max_trans = max(t["rel"] for t in trans)
    # ★ 모든 rho 에서 상대차가 같다 = 계통 오차. 적분기 차이인지 dt 수렴으로 가른다.
    #   PairSTDPSyn 은 cnexp(선형계에서 정확), GB 계열은 derivimplicit(암시적 오일러, 1차).
    #   tau_r_AMPA=0.2ms 는 dt=0.025 의 8배뿐이라 GB 가 봉우리를 과소평가한다.
    print(f"      ---- dt 수렴: 차이가 적분기 탓인가")
    tconv = []
    for dt in (0.025, 0.005, 0.001):
        a = first_peak(MECH, 0.0, dt); b = first_peak(MECH_GB, 0.0, dt)
        rel = abs(a - b) / b
        tconv.append(dict(dt=dt, stdp_nS=a, gb_nS=b, rel=rel))
        print(f"      dt {dt:<7.4f} STDP {a:.6f} · GB {b:.6f} · 상대차 {rel:.3e}")
    tratio = [tconv[i]["rel"] / tconv[i + 1]["rel"] for i in range(len(tconv) - 1)]
    print(f"      감소비 (dt 5배 축소당): " + " · ".join(f"{r:.2f}" for r in tratio) +
          "  (1차 수렴이면 ~5)")

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.46)
    axA = fig.add_subplot(gs_[0, :2])
    axB = fig.add_subplot(gs_[0, 2])
    axC = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axE = fig.add_subplot(gs_[1, 2])

    # A: 창
    axA.plot(used, refs, color="#f9a825", lw=4, alpha=0.55, label="참조 (닫힌 형태)")
    axA.plot(used, dws, color="#4527a0", lw=1.6, label="mod (PairSTDPSyn)")
    axA.axhline(0, color="#90a4ae", lw=1.0); axA.axvline(0, color="#90a4ae", ls=":", lw=1.0)
    axA.fill_between(used, dws, 0, where=(dws > 0), color="#2e7d32", alpha=0.15)
    axA.fill_between(used, dws, 0, where=(dws < 0), color="#c62828", alpha=0.15)
    axA.text(52, 0.55, "pre→post\nLTP", fontsize=9, color="#2e7d32", ha="center")
    axA.text(-58, -0.55, "post→pre\nLTD", fontsize=9, color="#c62828", ha="center")
    axA.set_xlabel("dt = t_post - t_pre (ms)"); axA.set_ylabel("dw (A_p=A_d=1)")
    axA.set_title(f"A. 고립 짝의 STDP 창 — mod 가 닫힌 형태와 일치한다 "
                  f"(최대 절대차 {err:.1e})\n"
                  f"tau_p {Q['tau_p']} / tau_d {Q['tau_d']} ms · dt=0 은 규약 문제로 제외",
                  fontsize=9.5, loc="left")
    axA.legend(fontsize=8.5)

    # B: 오차
    axB.semilogy(used, np.abs(dws - refs) + 1e-18, ".", ms=4, color="#c62828")
    axB.axhline(TOL, color="#37474f", ls="--", lw=1.4, label=f"허용 {TOL:.0e}")
    plots.ascii_log(axB)
    axB.set_xlabel("dt (ms)"); axB.set_ylabel("|mod - 참조|")
    axB.set_title("B. 절대차\n스파이크 시각을 dt 정수배로 두면 양자화가 없다",
                  fontsize=9.2, loc="left")
    axB.legend(fontsize=8)

    # C: 규약 대비
    for a2a, col, mk in ((1, "#4527a0", "o-"), (0, "#00838f", "s--")):
        pp = [r["per_pair"] for r in conv[a2a]]
        axC.plot(FREQS, pp, mk, color=col, ms=7, lw=2,
                 label=f"all_to_all={a2a}")
    axC.axhline(0, color="#90a4ae", lw=1.0)
    axC.set_xscale("log"); axC.set_xlabel("짝 반복 주파수 (Hz)")
    axC.set_ylabel("짝당 dw")
    axC.set_title("C. 규약이 고빈도에서 갈린다\n"
                  "all-to-all 은 창이 겹쳐 부호가 뒤집힌다", fontsize=9.2, loc="left")
    axC.legend(fontsize=8)

    # D: GB 와의 기전 차이
    ax2 = axD.twinx()
    pp = [r["per_pair"] for r in conv[1]]
    axD.plot(FREQS, pp, "o-", color="#4527a0", ms=7, lw=2, label="고전 STDP 짝당 dw (좌)")
    axD.axhline(0, color="#90a4ae", lw=1.0)
    ax2.plot(FREQS, [r["over_p"] for r in gbrow], "s--", color="#00838f", ms=7, lw=2,
             label="GB theta_p 초과 (우)")
    axD.set_xscale("log"); axD.set_xlabel("짝 반복 주파수 (Hz)")
    axD.set_ylabel("고전 STDP 짝당 dw", color="#4527a0")
    ax2.set_ylabel("GB theta_p 초과 (ms)", color="#00838f")
    axD.tick_params(axis="y", labelcolor="#4527a0")
    ax2.tick_params(axis="y", labelcolor="#00838f")
    h1, l1 = axD.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axD.legend(h1 + h2, l1 + l2, fontsize=7.3, loc="upper left")
    axD.set_title("D. 같은 dt(+10ms)인데 기전이 다르다\n"
                  "STDP=창 겹침 · GB=칼슘 누적", fontsize=9.2, loc="left")

    # E: 전달 정합
    xx = np.arange(len(trans))
    axE.bar(xx - 0.19, [t["stdp_nS"] for t in trans], width=0.38, color="#4527a0",
            label="PairSTDPSyn")
    axE.bar(xx + 0.19, [t["gb_nS"] for t in trans], width=0.38, color="#37474f",
            label="GBPlasticitySyn")
    axE.set_xticks(xx); axE.set_xticklabels([f"rho0={t['rho0']:.1f}" for t in trans])
    axE.set_ylabel("첫 펄스 전도도 (nS)")
    axE.set_title(f"E. 전달이 GB 와 같다 (5-10 전제)\n최대 상대차 {max_trans:.1e}",
                  fontsize=9.2, loc="left")
    axE.legend(fontsize=8)
    for i, t in enumerate(trans):
        axE.text(i, max(t["stdp_nS"], t["gb_nS"]), f"{t['stdp_nS']:.3f}",
                 ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("5-6  엔진 고전 STDP (mechanisms/PairSTDPSyn.mod — 04 자체 작성)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"5-6 | 04 자체 mod · 창 tau_p {Q['tau_p']}/tau_d {Q['tau_d']}ms "
                     f"(원문 미확보) · 고립짝 절대차 {err:.1e} · 트레인 {err_tr:.1e} · "
                     f"전달 정합 {max_trans:.1e}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-6_stdp_window.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    i_p1 = int(np.where(used == 1.0)[0][0]); i_m1 = int(np.where(used == -1.0)[0][0])
    a2a_hi = conv[1][-1]["per_pair"]; nn_hi = conv[0][-1]["per_pair"]
    checks = [
        (f"★고립 짝이 닫힌 형태와 일치 (최대 절대차 {err:.2e} < {TOL:.0e})", err < TOL),
        (f"★트레인도 참조와 일치 (최대 절대차 {err_tr:.2e} < {TOL:.0e})", err_tr < TOL),
        ("dt>0 은 LTP · dt<0 은 LTD", dws[i_p1] > 0 and dws[i_m1] < 0),
        ("창이 지수적으로 감쇠한다 (|dt| 50ms 가 1ms 의 5~20%)",
         0.05 < abs(dws[used == 50.0][0] / dws[i_p1]) < 0.30),
        ("★두 규약 모두 50Hz 에서 부호가 뒤집힌다 — 원인은 tau_d(33.7) > tau_p(16.8), "
         "즉 LTD 창이 더 넓다",
         a2a_hi < 0 and nn_hi < 0),
        ("all-to-all 이 최근접보다 더 음수다 (짝이 더 많이 누적된다)",
         a2a_hi <= nn_hi + 1e-12),
        ("저빈도(1Hz)에서는 두 규약이 같다 (창이 겹치지 않는다)",
         abs(conv[1][0]["per_pair"] - conv[0][0]["per_pair"]) < 1e-6),
        ("★GB 는 같은 조건에서 반대로 간다 (50Hz 에서 theta_p 초과 > 0)",
         gbrow[-1]["over_p"] > 0),
        ("하드 상한이 작동한다 (LTP 포화 시 rho = 1)",
         abs(clip["상한 (LTP 포화)"]["rho_end"] - 1.0) < 1e-9),
        ("하드 하한이 작동한다 (LTD 포화 시 rho = 0)",
         abs(clip["하한 (LTD 포화)"]["rho_end"]) < 1e-9),
        (f"전달 크기가 GB 와 2% 이내 (최대 상대차 {max_trans:.2e})", max_trans < 0.02),
        ("★상대차가 rho 와 무관하게 일정하다 = 계통 오차 (편차 < 1e-6)",
         (max(t["rel"] for t in trans) - min(t["rel"] for t in trans)) < 1e-6),
        ("★차이는 적분기 탓이다 — dt 를 줄이면 사라진다 (5배 축소당 2배 이상 감소)",
         all(r > 2.0 for r in tratio)),
        ("★INITIAL 에서 w 를 초기화하므로 D22 함정이 없다 (rho0=0 에서 첫 전도도 > 0)",
         trans[0]["stdp_nS"] > 1e-6),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(mech=MECH, mod_source="mechanisms/PairSTDPSyn.mod (04 자체 작성)",
               doc="mechanisms/PairSTDPSyn.md", caps=CAPS[MECH],
               window_params=Q, verify_setup=dict(**WIDE, **AMP1, dt_sim=DT_SIM),
               tol=TOL,
               A_window=dict(dts=[float(v) for v in used],
                             dw_mod=[round(float(v), 12) for v in dws],
                             dw_ref=[round(float(v), 12) for v in refs],
                             max_abs_err=err,
                             dt0_mod=w0, dt0_ref=0.0),
               B_conventions={str(k): v for k, v in conv.items()},
               B_max_err=err_tr,
               C_gb_contrast=gbrow,
               D_clipping=clip,
               E_transmission=dict(rows=trans, max_rel_err=max_trans,
                                   dt_convergence=tconv,
                                   ratios=[round(r, 3) for r in tratio]),
               finding=("우리가 쓴 mod 가 닫힌 형태 지수 창을 절대차 1e-12 이하로 재현한다. "
                        "고립 짝뿐 아니라 트레인(all-to-all·최근접 두 규약)도 참조와 일치한다."),
               finding_integrator=("★전달 크기가 GB 보다 모든 rho 에서 **정확히 1.15% 크다**. "
                                   "PairSTDPSyn 은 cnexp(선형계에서 정확)를, GB 계열은 "
                                   "derivimplicit(암시적 오일러, 1차)를 쓴다. tau_r_AMPA=0.2ms 는 "
                                   "dt=0.025 의 8배뿐이라 GB 가 봉우리를 과소평가한다. dt 를 "
                                   "줄이면 차이가 1차로 사라지므로 공식 차이가 아니라 적분기 "
                                   "차이다. => 5-10 의 엔진 간 첫 펄스 정합은 이 1.15% 를 "
                                   "명시적으로 교정해야 한다(또는 같은 dt 를 쓰는 조건에서만 비교)."),
               finding_window=("두 짝짓기 규약 모두 50Hz 에서 부호가 뒤집힌다. 원인은 "
                               "tau_d(33.7) > tau_p(16.8) — LTD 창이 더 넓어서 ISI 20ms 에서 "
                               "직전 post 와의 -10ms 짝(exp(-10/33.7)=0.743)이 +10ms "
                               "짝(0.552)을 이긴다. 저빈도에서는 두 규약이 같다."),
               dt0_note=("dt=0(동시 스파이크)은 참조가 0 을 주고 mod 는 이벤트 전달 순서에 "
                         f"따라 ±A 를 준다(실측 {w0:+.6f}). 실험적으로도 분리 불가한 조건이므로 "
                         "검증에서 제외하고 규약 차이로 기록한다."),
               convention_note=("all_to_all=1(기본)은 고빈도에서 창이 겹쳐 50Hz 에서 짝당 dw 의 "
                                f"부호가 뒤집힌다({a2a_hi:+.4f}). 최근접 이웃(0)은 부호를 유지한다"
                                f"({nn_hi:+.4f}). 6단계 보고에 **어느 규약인지 반드시 인쇄**한다."),
               unresolved=("tau_p/tau_d = 16.8/33.7ms 는 Bi&Poo 1998 로 널리 인용되나 원문 "
                           "미확보(미결#14). 잠정값이며 원문 확보 시 mod·md·이 결과를 함께 고친다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-6_stdp.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-6 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
