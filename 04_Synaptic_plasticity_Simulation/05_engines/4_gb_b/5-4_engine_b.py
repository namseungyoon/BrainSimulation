# -*- coding: utf-8 -*-
"""5-4 엔진 B — GB + Tsodyks-Markram (단기가소성이 장기 결과를 바꾸는가)

단계   : 5-4 (5단계 가소성 엔진 / 하위 4 gb_b)
쉬운 설명: 엔진 A(GB)는 버스트의 네 펄스를 **똑같이** 취급한다. 실제 시냅스는 그렇지 않다 —
          우리 PC->PC 는 뒤 펄스가 **작아진다**(억압형). 엔진 B 는 그 사실을 넣은 것이고,
          그러면 버스트가 만드는 칼슘이 달라지므로 **장기 결과의 크기가 바뀐다.**
방법   : 엔진 B 의 두 관례를 정면으로 검증한다 (PLAN 문헌 불일치 #5).
          norm_Pr : 첫 펄스를 A 와 같게 맞추는 정규화 (mod 기본 ON) — 우리 관례, 논문 처방 아님
          ca_stp  : 칼슘이 방출량에 비례하는가 (mod 기본 ON) — 우리 가설, 논문은 상수
          네 조합을 모두 돌리고 numpy 참조(TM x GB 합성)와 대조한다.
검증   : mod↔참조 효능 절대차 < 1e-3 · 칼슘 상대차 < 2e-3 (D23) + 관례별 거동 확인.
근거   : Graupner & Brunel 2012 (장기) · Tsodyks & Markram 1997 / Fuhrmann 2002 (단기)
          Ecker 2020 Table 3 PC->PC (Use 0.50 · Dep 671 · Fac 17 = 억압형)
결과   : figures/5-4_engine_b.png · figures/5-4_gb_b.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\4_gb_b\\5-4_engine_b.py
비고   : ★엔진 B 의 mod 주석은 촉진형 SC->PC(Fac 250)를 전제로 쓰였다. 우리 클래스는
          억압형이므로 **가설의 부호가 뒤집힌다** — 그것을 실측으로 확인하는 것이 이 단계다.
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
from lib import refdata                              # noqa: E402
from lib.synprobe import SynProbe                    # noqa: E402
from lib.refs import gb, tm                          # noqa: E402
from lib.wiring import load_synapse_cfg              # noqa: E402

MECH_A, MECH_B = "GBPlasticitySyn", "GBPlasticityStpSyn"
REC_DT = 0.1
V_HOLD = -70.0
T0 = 20.0
TOL_RHO, TOL_CA_REL = 1e-3, 2e-3
RHO0 = 0.0

# theta 버스트 10회 @5Hz, 버스트당 4발 100Hz (6-5 가 쓸 프로토콜)
N_BURST, BURST_HZ, N_IN, IN_HZ, DT_PAIR = 10, 5.0, 4, 100.0, 5.0
TSTOP = 3000.0


def burst_times():
    bi, ii = 1000.0 / BURST_HZ, 1000.0 / IN_HZ
    pre = [T0 + b * bi + k * ii for b in range(N_BURST) for k in range(N_IN)]
    return pre, [t + DT_PAIR for t in pre]


def ref_engine_b(t, pre, post, U, Dep, Fac, norm_Pr, ca_stp, rho0=RHO0, G=None):
    """엔진 B 참조 = TM(방출) x GB(칼슘·효능) 합성. mod 와 같은 순서로 계산한다."""
    G = dict(gb.WITTENBERG2006 if G is None else G)
    _, _, pr = tm.simulate(pre, U, Dep, Fac)          # Pr = u*R (소모 전 R)
    pr_ref = U if norm_Pr else 1.0
    prn = pr / pr_ref
    ca = G["C_pre"] * (1.0 + (1.0 if ca_stp else 0.0) * (prn - 1.0))
    ca = np.clip(ca, 0.0, None)
    c = gb.calcium_amp(t, pre, ca, post, p=G)
    return c, gb.integrate_rho(t, c, rho0=rho0, p=G), prn, ca


def main():
    plots.setup()
    print("=== 5-4 엔진 B (GB + Tsodyks-Markram) ===")
    cls, P = load_synapse_cfg()
    G = gb.WITTENBERG2006
    U, Dp, Fc = P["Use"], P["Dep_ms"], P["Fac_ms"]
    print(f"  시냅스 {cls} · Use {U} · Dep {Dp}ms · Fac {Fc}ms "
          f"-> {'억압형' if Fc < Dp else '촉진형'}")
    print(f"  프로토콜: theta 버스트 {N_BURST}회 @{BURST_HZ:.0f}Hz "
          f"(버스트당 {N_IN}발 {IN_HZ:.0f}Hz) · rho0 {RHO0}")

    pre, post = burst_times()

    def make(mech, **kw):
        p = SynProbe(mech, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        p.set(e=P["e_rev_mV"], tau_r_AMPA=P["tau_r_AMPA"], tau_d_AMPA=P["tau_d_AMPA"],
              tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
              NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"], rho0=RHO0)
        p.set(**kw)
        return p

    # ── 엔진 A 기준 ───────────────────────────────────────────────────────
    pa = make(MECH_A)
    pa.drive_pre(pre); pa.drive_post(post)
    RA = pa.run(TSTOP)
    tA = RA["t"]
    cA_ref = gb.calcium(tA, pre, post, G)
    rhoA_ref = gb.integrate_rho(tA, cA_ref, rho0=RHO0, p=G)
    print(f"\n  [엔진 A] rho -> {RA['rho'][-1]:.5f} · c_max {RA['c'].max():.3f}")

    def jumps(R, times, win=3.0):
        out = []
        for k, ts in enumerate(times):
            te = times[k + 1] if k + 1 < len(times) else ts + win + 1.0
            m = (R["t"] >= ts) & (R["t"] < min(ts + win, te))
            base = float(R["g"][R["t"] <= ts][-1]) if (R["t"] <= ts).any() else 0.0
            if m.any():
                out.append((float(R["g"][m].max()) - base) * 1e3)
        return out

    jA = jumps(RA, pre)

    # ── 엔진 B 네 조합 ────────────────────────────────────────────────────
    COMBOS = [(1, 1, "mod 기본 (정규화 ON · 칼슘 비례 ON)"),
              (1, 0, "정규화 ON · 칼슘 상수 (= 논문 원본 칼슘)"),
              (0, 1, "정규화 OFF · 칼슘 비례 ON"),
              (0, 0, "정규화 OFF · 칼슘 상수")]
    rows = []
    print(f"\n  [엔진 B] 관례 네 조합 (norm_Pr, ca_stp)")
    for npr, cst, desc in COMBOS:
        p = make(MECH_B, Use=U, Dep=Dp, Fac=Fc, norm_Pr=npr, ca_stp=cst)
        p.drive_pre(pre); p.drive_post(post)
        p.record(extra=("pr_last", "ca_last"))
        R = p.run(TSTOP)
        t = R["t"]
        c_ref, rho_ref, prn, ca_amt = ref_engine_b(t, pre, post, U, Dp, Fc, npr, cst)
        e_c = float(np.max(np.abs(R["c"] - c_ref)))
        e_cr = e_c / max(float(R["c"].max()), 1e-12)
        e_r = float(np.max(np.abs(R["rho"] - rho_ref)))
        jB = jumps(R, pre)
        rows.append(dict(norm_Pr=npr, ca_stp=cst, desc=desc,
                         rho_end=float(R["rho"][-1]), c_max=float(R["c"].max()),
                         over_p_ms=float(REC_DT * np.sum(R["c"] > G["theta_p"])),
                         first_jump=jB[0], jumps=jB, prn=prn.tolist(),
                         ca_amt=ca_amt.tolist(), err_c=e_c, err_c_rel=e_cr,
                         err_rho=e_r, t=t, c=R["c"], rho=R["rho"],
                         c_ref=c_ref, rho_ref=rho_ref))
        print(f"      norm_Pr={npr} ca_stp={cst}  rho -> {R['rho'][-1]:.5f} · "
              f"c_max {R['c'].max():.3f} · 첫 증분 {jB[0]:.4f} nS · "
              f"|drho| {e_r:.2e} · 칼슘 상대차 {e_cr:.2e}")
        print(f"          {desc}")

    # ── 버스트 내 방출 프로파일 (억압형이라 줄어든다) ──────────────────────
    prn1 = np.array(rows[0]["prn"][:N_IN])
    print(f"\n  [버스트 내 방출] 첫 버스트의 정규화 방출 prn: " +
          " ".join(f"{v:.4f}" for v in prn1) +
          f"  -> {'억압' if prn1[-1] < prn1[0] else '촉진'}")
    print(f"  [칼슘 기여] 같은 버스트의 C_pre_eff (ca_stp=1): " +
          " ".join(f"{v:.4f}" for v in np.array(rows[0]['ca_amt'][:N_IN])))

    # ── 촉진형 대조 (E1) — mod 주석이 전제한 상황 ─────────────────────────
    C1 = refdata.ECKER_E1_CONTRAST
    p = make(MECH_B, Use=C1["Use"], Dep=C1["Dep_ms"], Fac=C1["Fac_ms"],
             norm_Pr=1, ca_stp=1)
    p.drive_pre(pre); p.drive_post(post)
    RE = p.run(TSTOP)
    _, _, prnE = tm.simulate(pre, C1["Use"], C1["Dep_ms"], C1["Fac_ms"])
    prnE = prnE / C1["Use"]
    print(f"\n  [촉진형 대조 E1] Use {C1['Use']} · Dep {C1['Dep_ms']} · Fac {C1['Fac_ms']}")
    print(f"      첫 버스트 prn: " + " ".join(f"{v:.4f}" for v in prnE[:N_IN]) +
          f" -> 촉진 · rho -> {RE['rho'][-1]:.5f} (A 는 {RA['rho'][-1]:.5f})")

    base = float(RA["rho"][-1])
    r_bb = rows[0]["rho_end"]
    r_const = rows[1]["rho_end"]

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.46)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1])
    axC = fig.add_subplot(gs_[0, 2])
    axD = fig.add_subplot(gs_[1, 0])
    axE = fig.add_subplot(gs_[1, 1])
    axF = fig.add_subplot(gs_[1, 2])

    # A: 버스트 내 방출·칼슘 프로파일
    x = np.arange(1, N_IN + 1)
    axA.plot(x, prn1, "o-", color="#c62828", ms=7, lw=2, label="PC->PC (억압형)")
    axA.plot(x, prnE[:N_IN], "s-", color="#1565c0", ms=7, lw=2, label="E1 (촉진형)")
    axA.axhline(1.0, color="#90a4ae", ls="--", lw=1.0, label="엔진 A (항상 1)")
    axA.set_xticks(x); axA.set_xlabel("버스트 내 펄스"); axA.set_ylabel("정규화 방출 prn")
    axA.set_title("A. 엔진 A 는 네 펄스를 같게 본다\n"
                  "우리 클래스는 **억압형** — mod 주석의 전제와 반대", fontsize=9.2, loc="left")
    axA.legend(fontsize=7.8)

    # B: 전달 증분 (첫 펄스 정합)
    xa = np.arange(1, min(len(jA), N_IN) + 1)
    axB.plot(xa, jA[:N_IN], "^-", color="#37474f", ms=7, lw=2, label="엔진 A")
    for r, col, ls in zip(rows[:2], ["#c62828", "#ef6c00"], ["-", "--"]):
        axB.plot(xa, r["jumps"][:N_IN], "o", ls=ls, color=col, ms=6, lw=1.6,
                 label=f"B norm_Pr={r['norm_Pr']} ca_stp={r['ca_stp']}")
    axB.plot(xa, rows[3]["jumps"][:N_IN], "v:", color="#6a1b9a", ms=6, lw=1.6,
             label="B 정규화 OFF")
    axB.set_xticks(xa); axB.set_xlabel("버스트 내 펄스"); axB.set_ylabel("전달 증분 (nS)")
    axB.set_title(f"B. norm_Pr 의 역할 — 첫 펄스를 A 와 맞춘다\n"
                  f"OFF 면 Use({U}) 배 작아진다", fontsize=9.2, loc="left")
    axB.legend(fontsize=7.3)

    # C: 칼슘 궤적 (A vs B 두 관례)
    axC.plot(tA / 1000.0, RA["c"], color="#37474f", lw=1.6, label="엔진 A")
    axC.plot(rows[0]["t"] / 1000.0, rows[0]["c"], color="#c62828", lw=1.4,
             label="B 칼슘 비례 ON")
    axC.plot(rows[1]["t"] / 1000.0, rows[1]["c"], color="#ef6c00", lw=1.0, ls="--",
             label="B 칼슘 상수")
    axC.axhline(G["theta_p"], color="#2e7d32", ls=":", lw=1.2)
    axC.axhline(G["theta_d"], color="#ef6c00", ls=":", lw=1.0)
    axC.set_xlim(0, 0.6)
    axC.set_xlabel("시간 (s)"); axC.set_ylabel("칼슘 c")
    axC.set_title("C. 억압형에서는 칼슘 비례가 칼슘을 **깎는다**\n"
                  "(처음 3 버스트 확대)", fontsize=9.2, loc="left")
    axC.legend(fontsize=7.5)

    # D: 효능 궤적
    axD.plot(tA / 1000.0, RA["rho"], color="#37474f", lw=2, label=f"A ({base:.4f})")
    for r, col in zip(rows, ["#c62828", "#ef6c00", "#6a1b9a", "#00838f"]):
        axD.plot(r["t"] / 1000.0, r["rho"], lw=1.4, color=col,
                 label=f"B {r['norm_Pr']}{r['ca_stp']} ({r['rho_end']:.4f})")
    axD.set_xlabel("시간 (s)"); axD.set_ylabel("효능 rho")
    axD.set_title("D. 같은 자극·같은 장기 파라미터인데 결과가 갈린다\n"
                  "차이의 원인은 전부 **관례 선택**이다", fontsize=9.2, loc="left")
    axD.legend(fontsize=7.2, ncol=2)

    # E: 관례별 최종 효능
    lab = [f"norm{r['norm_Pr']}\nca{r['ca_stp']}" for r in rows]
    val = [r["rho_end"] for r in rows]
    axE.bar(range(len(val)), val, color=["#c62828", "#ef6c00", "#6a1b9a", "#00838f"],
            width=0.6)
    axE.axhline(base, color="#37474f", ls="--", lw=1.6, label=f"엔진 A = {base:.4f}")
    axE.set_xticks(range(len(val))); axE.set_xticklabels(lab, fontsize=8)
    axE.set_ylabel("최종 효능 rho")
    for i, v in enumerate(val):
        axE.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    axE.set_title("E. 관례 네 조합의 결과\n논문 원본 칼슘(ca_stp=0)이 A 와 같아야 한다",
                  fontsize=9.2, loc="left")
    axE.legend(fontsize=8)

    # F: mod↔참조 오차
    er = [r["err_rho"] for r in rows]; ec = [r["err_c_rel"] for r in rows]
    xx = np.arange(len(rows))
    axF.bar(xx - 0.19, er, width=0.38, color="#4527a0", label="|drho| 절대")
    axF.bar(xx + 0.19, ec, width=0.38, color="#00838f", label="칼슘 상대차")
    axF.axhline(TOL_RHO, color="#4527a0", ls="--", lw=1.0)
    axF.axhline(TOL_CA_REL, color="#00838f", ls=":", lw=1.0)
    axF.set_yscale("log"); plots.ascii_log(axF)
    axF.set_xticks(xx); axF.set_xticklabels(lab, fontsize=8)
    axF.set_ylabel("mod ↔ 참조 오차")
    axF.set_title(f"F. 네 조합 모두 참조와 일치\n허용 {TOL_RHO:.0e} / {TOL_CA_REL:.0e}",
                  fontsize=9.2, loc="left")
    axF.legend(fontsize=7.8)

    fig.suptitle("5-4  엔진 B (GB + Tsodyks-Markram) — 단기가소성이 장기 결과를 바꾼다",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"5-4 | {cls} Use {U}/Dep {Dp}/Fac {Fc}(억압형) · theta버스트 {N_BURST}회 · "
                     f"rho0 {RHO0} · A {base:.4f} vs B(기본) {r_bb:.4f} vs B(논문칼슘) {r_const:.4f}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-4_engine_b.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    max_er = max(r["err_rho"] for r in rows)
    max_ecr = max(r["err_c_rel"] for r in rows)
    j_on = rows[0]["first_jump"]; j_off = rows[2]["first_jump"]
    checks = [
        (f"★mod↔참조 효능 절대차 < {TOL_RHO:.0e} (최대 {max_er:.2e})", max_er < TOL_RHO),
        (f"★mod↔참조 칼슘 상대차 < {TOL_CA_REL:.0e} (최대 {max_ecr:.2e})",
         max_ecr < TOL_CA_REL),
        ("norm_Pr=1 이면 첫 펄스 전달이 엔진 A 와 같다 (1% 이내)",
         abs(j_on - jA[0]) / jA[0] < 0.01),
        (f"norm_Pr=0 이면 첫 펄스가 Use({U}) 배로 작아진다 (5% 이내)",
         abs(j_off / jA[0] - U) / U < 0.05),
        ("★ca_stp=0 (논문 원본 칼슘) 은 엔진 A 와 칼슘이 같다",
         abs(rows[1]["c_max"] - float(RA["c"].max())) / float(RA["c"].max()) < 0.01),
        ("★ca_stp=0 은 최종 효능도 엔진 A 와 같다 (1e-3 이내)",
         abs(rows[1]["rho_end"] - base) < 1e-3),
        ("★우리 PC->PC 는 버스트 내에서 방출이 줄어든다 (억압형)",
         prn1[-1] < prn1[0] * 0.9),
        ("★그래서 ca_stp=1 은 칼슘을 **깎고** LTP 를 줄인다 (mod 주석 전제와 반대 방향)",
         rows[0]["c_max"] < float(RA["c"].max()) and rows[0]["rho_end"] < base),
        ("촉진형(E1)에서는 방향이 뒤집힌다 — 버스트 내 방출이 늘어난다",
         prnE[N_IN - 1] > prnE[0]),
        ("촉진형에서는 ca_stp=1 이 LTP 를 늘린다 (엔진 A 보다 큼)",
         float(RE["rho"][-1]) > base),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(mech=dict(A=MECH_A, B=MECH_B), rec_dt=REC_DT, v_hold=V_HOLD, t0=T0,
               rho0=RHO0, tol=dict(rho=TOL_RHO, ca_rel=TOL_CA_REL),
               syn_class=cls, stp=dict(Use=U, Dep_ms=Dp, Fac_ms=Fc,
                                       kind="억압형" if Fc < Dp else "촉진형"),
               protocol=dict(n_burst=N_BURST, burst_hz=BURST_HZ, n_in=N_IN,
                             in_hz=IN_HZ, dt_pair=DT_PAIR, tstop=TSTOP),
               engine_A=dict(rho_end=base, c_max=float(RA["c"].max()),
                             first_jump_nS=jA[0], jumps_nS=jA[:N_IN]),
               engine_B=[{k: v for k, v in r.items()
                          if k not in ("t", "c", "rho", "c_ref", "rho_ref")}
                         for r in rows],
               facilitating_contrast=dict(params=dict(Use=C1["Use"], Dep_ms=C1["Dep_ms"],
                                                     Fac_ms=C1["Fac_ms"]),
                                          prn_first_burst=[round(float(v), 5)
                                                           for v in prnE[:N_IN]],
                                          rho_end=float(RE["rho"][-1])),
               max_err=dict(rho=max_er, ca_rel=max_ecr),
               finding=("엔진 B 의 mod 주석은 **촉진형 SC->PC(Fac 250ms)** 를 전제로 쓰였다 — "
                        "'버스트 뒤 펄스가 더 방출하니 엔진 A 는 칼슘을 과소평가한다' 는 가설이다. "
                        "그런데 우리 클래스(Ecker PC->PC)는 **억압형**(Fac 17 < Dep 671)이므로 "
                        "가설의 부호가 뒤집힌다: ca_stp=1 은 칼슘을 깎아 **LTP 를 줄인다.** "
                        "촉진형(E1)으로 바꾸면 원래 방향(늘림)이 나온다 — 즉 이 관례의 효과는 "
                        "시냅스 클래스에 종속이고, 6단계 해석에 클래스를 함께 적어야 한다."),
               conventions=("norm_Pr 과 ca_stp 는 둘 다 논문 처방이 아니라 우리 관례다. "
                            "ca_stp=0 은 논문 원본을 정확히 복원하며 실측으로 확인했다 — "
                            "엔진 A 와 칼슘·최종 효능이 같다. 따라서 '엔진 A 와 B 의 차이' 를 "
                            "논문 수준의 주장으로 쓰려면 ca_stp=0 을 기준으로 삼고, ca_stp=1 은 "
                            "명시적 가설로 따로 보고해야 한다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-4_gb_b.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-4 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
