# -*- coding: utf-8 -*-
"""5-5 엔진 C — B + 확률 다소포 방출 (전달은 흩어지고, 칼슘은 관례가 정한다)

단계   : 5-5 (5단계 가소성 엔진 / 하위 5 gb_c)
쉬운 설명: 실제 시냅스는 스파이크가 와도 **방출에 실패**할 수 있다. 엔진 C 는 소포 하나하나를
          시뮬레이션한다(Nrrp 개 방출 부위, 각각 채워졌거나 비었다). 그래서 같은 자극이라도
          시행마다 전달이 다르다.
          ★그런데 **가소성까지 흩어지는지는 관례가 정한다.** 논문 원본(ca_stp=0)은 방출이
          실패해도 칼슘을 그대로 주입하므로 **효능은 전혀 흩어지지 않는다.**
방법   : (A) 첫 펄스 방출률이 이론 1-(1-Use)^Nrrp 와 맞는가
          (B) 시행평균 정규화 방출 prn 이 1 로 수렴하는가 (= B 와 같은 크기)
          (C) ca_stp=0 과 1 에서 효능 분산이 어떻게 다른가
          (D) mod 주석이 경고한 **Nrrp=1 + ca_stp=1 인공물**을 재현한다
          (E) **RNG 미시딩 함정** — 첫 펄스만 방출하고 영구 침묵
검증   : 이론 방출률 일치 · 평균 수렴 · 관례별 분산 · 인공물 재현 · 미시딩 검출.
근거   : Ecker 2020 Table 3 PC->PC (Use 0.50 · Nrrp 2) · Fuhrmann 2002 ·
          GBPlasticityStpProbSyn.mod 주석(OUR CHOICE 1·2 · RNG)
결과   : figures/5-5_engine_c.png · figures/5-5_gb_c.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\5_gb_c\\5-5_engine_c.py
비고   : mod 의 RANGE 계수기 n_pre·n_rel·ves_last 를 직접 읽는다 — 전도도에서 추론하지 않는다.
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
from lib.synprobe import SynProbe                    # noqa: E402
from lib.refs import gb                              # noqa: E402
from lib.wiring import load_synapse_cfg              # noqa: E402

MECH_B, MECH_C = "GBPlasticityStpSyn", "GBPlasticityStpProbSyn"
REC_DT = 0.5                  # 효능·계수기만 보므로 굵게 (속도)
V_HOLD = -70.0
T0 = 20.0
RHO0 = 0.0
N_TRIAL = 60

N_BURST, BURST_HZ, N_IN, IN_HZ, DT_PAIR = 10, 5.0, 4, 100.0, 5.0
TSTOP = 3000.0


def burst_times():
    bi, ii = 1000.0 / BURST_HZ, 1000.0 / IN_HZ
    pre = [T0 + b * bi + k * ii for b in range(N_BURST) for k in range(N_IN)]
    return pre, [t + DT_PAIR for t in pre]


def main():
    plots.setup()
    print("=== 5-5 엔진 C (B + 확률 다소포 방출) ===")
    cls, P = load_synapse_cfg()
    G = gb.WITTENBERG2006
    U, Dp, Fc, Nrrp = P["Use"], P["Dep_ms"], P["Fac_ms"], int(P["Nrrp"])
    p_rel1 = 1.0 - (1.0 - U) ** Nrrp
    print(f"  시냅스 {cls} · Use {U} · **Nrrp {Nrrp}** · Dep {Dp}ms · Fac {Fc}ms")
    print(f"  이론(첫 펄스, 쉰 시냅스): 방출확률 1-(1-{U:g})^{Nrrp} = {p_rel1:.4f} · "
          f"실패율 {1-p_rel1:.4f}")
    pre, post = burst_times()
    n_pulse = len(pre)
    print(f"  프로토콜: theta 버스트 {N_BURST}회 @{BURST_HZ:.0f}Hz x {N_IN}발 "
          f"{IN_HZ:.0f}Hz = {n_pulse}펄스 · rho0 {RHO0}")

    def make(mech, ca_stp, nrrp=None, **kw):
        p = SynProbe(mech, clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        p.set(e=P["e_rev_mV"], tau_r_AMPA=P["tau_r_AMPA"], tau_d_AMPA=P["tau_d_AMPA"],
              tau_r_NMDA=P["tau_r_NMDA"], tau_d_NMDA=P["tau_d_NMDA"],
              NMDA_ratio=P["NMDA_ratio"], mg=P["mg_mM"], rho0=RHO0,
              Use=U, Dep=Dp, Fac=Fc, norm_Pr=1, ca_stp=ca_stp)
        if mech == MECH_C:
            p.set(Nrrp=(Nrrp if nrrp is None else nrrp))
        p.set(**kw)
        return p

    def run_trial(mech, ca_stp, seed=None, nrrp=None, tstop=TSTOP):
        p = make(mech, ca_stp, nrrp=nrrp)
        if seed is not None:
            p.seed(*seed)
        p.drive_pre(pre); p.drive_post(post)
        R = p.run(tstop)
        info = dict(rho_end=float(R["rho"][-1]), c_max=float(R["c"].max()))
        if mech == MECH_C:
            info.update(n_pre=int(p.syn.n_pre), n_rel=int(p.syn.n_rel))
        return info, R, p

    # ── 엔진 B 기준 (결정론) ──────────────────────────────────────────────
    print()
    refs = {}
    for cst in (0, 1):
        iB, RB, _ = run_trial(MECH_B, cst)
        refs[cst] = iB
        print(f"  [엔진 B ca_stp={cst}] rho -> {iB['rho_end']:.5f} · "
              f"c_max {iB['c_max']:.3f}")

    # ── (E) RNG 미시딩 함정 ───────────────────────────────────────────────
    print(f"\n  [E] RNG 미시딩 함정 (mod 주석: urand()=0 이면 방출 1회 후 영구 침묵)")
    iU, RU, pU = run_trial(MECH_C, 0, seed=None)
    iS, RS, pS = run_trial(MECH_C, 0, seed=(1, 2, 3))
    print(f"      미시딩: pre {iU['n_pre']}회 중 방출 {iU['n_rel']}회 "
          f"({'첫 펄스만 = 죽은 시냅스' if iU['n_rel'] == 1 else '이상'})")
    print(f"      시딩:   pre {iS['n_pre']}회 중 방출 {iS['n_rel']}회")

    # ── (A)(B) 첫 펄스 통계 — 쉰 시냅스라 이론과 직접 비교된다 ─────────────
    print(f"\n  [A] 첫 펄스만 주고 {N_TRIAL*5}시행 — 이론 방출률과 대조")
    NT1 = N_TRIAL * 5
    rel1 = 0
    ves1 = []
    for k in range(NT1):
        p = make(MECH_C, 0)
        p.seed(50000 + k, 3 * k + 1, 5 * k + 7)
        p.drive_pre([T0]); p.drive_post([T0 + DT_PAIR])
        p.run(80.0)
        v = float(p.syn.ves_last)
        ves1.append(v)
        rel1 += 1 if v > 0 else 0
    ves1 = np.array(ves1)
    obs_p1 = rel1 / NT1
    prn_all = ves1 / Nrrp / U
    mean_prn1 = float(prn_all.mean())
    se_prn1 = float(prn_all.std(ddof=1) / np.sqrt(NT1))   # 표준오차 (통계 검정용)
    se_p1 = float(np.sqrt(p_rel1 * (1 - p_rel1) / NT1))
    print(f"      방출률 {obs_p1:.4f} (이론 {p_rel1:.4f} · 차 {abs(obs_p1-p_rel1):.4f})")
    print(f"      소포 수 분포: " +
          " · ".join(f"{v}개 {int(np.sum(ves1 == v))}회" for v in range(Nrrp + 1)))
    print(f"      정규화 방출 prn 평균 {mean_prn1:.4f} +- {se_prn1:.4f}(SE) "
          f"(관례상 1.0 · 편차 {abs(mean_prn1-1)/se_prn1:.2f} SE)")
    print(f"      방출률 SE {se_p1:.4f} -> 실측 편차 {abs(obs_p1-p_rel1)/se_p1:.2f} SE")

    # ── (C) 관례별 시행 분산 ──────────────────────────────────────────────
    print(f"\n  [C] {N_TRIAL}시행 — ca_stp 가 '가소성이 흩어지는가' 를 정한다")
    trials = {}
    for cst in (0, 1):
        rr, nr = [], []
        for k in range(N_TRIAL):
            info, _, _ = run_trial(MECH_C, cst, seed=(1000 + k, 7 * k + 1, 13 * k + 3))
            rr.append(info["rho_end"]); nr.append(info["n_rel"])
        rr = np.array(rr); nr = np.array(nr)
        trials[cst] = dict(rho=rr, n_rel=nr)
        n_up = int(np.sum(rr > 0.5))
        print(f"      ca_stp={cst}: rho 평균 {rr.mean():.5f} · 표준편차 {rr.std():.5f} · "
              f"범위 {rr.min():.5f}~{rr.max():.5f} · UP {n_up}/{N_TRIAL}")
        print(f"                  방출 성공 {nr.mean():.1f}/{n_pulse}펄스 "
              f"(범위 {nr.min()}~{nr.max()}) · B 는 rho {refs[cst]['rho_end']:.5f}")

    sd0, sd1 = trials[0]["rho"].std(), trials[1]["rho"].std()

    # ── (D) 문서화된 인공물 재현: Nrrp=1 + ca_stp=1 ────────────────────────
    print(f"\n  [D] mod 주석이 경고한 인공물 — Nrrp=1 + ca_stp=1")
    print(f"      Nrrp=1 이면 성공 시 prn = 1/Use = {1/U:.2f} -> 칼슘 {1/U:.2f} 가 "
          f"theta_p({G['theta_p']}) 를 {1/U/G['theta_p']:.1f}배 넘는다")
    art = {}
    for nr_ in (1, Nrrp):
        rr = []
        for k in range(N_TRIAL):
            info, _, _ = run_trial(MECH_C, 1, seed=(2000 + k, 11 * k + 1, 17 * k + 5),
                                   nrrp=nr_)
            rr.append(info["rho_end"])
        rr = np.array(rr)
        art[nr_] = rr
        print(f"      Nrrp={nr_} ca_stp=1: rho 평균 {rr.mean():.5f} · "
              f"UP {int(np.sum(rr > 0.5))}/{N_TRIAL} · 표준편차 {rr.std():.5f}")
    # ★ mod 주석의 6.67배는 Use=0.15(SC->PC E1s) 예시다. 우리 Use=0.50 이면 1/Use=2.0 뿐이다.
    #   인공물이 실제로 존재하는지 확인하려면 그 조건을 직접 재현해야 한다.
    U_ART = 0.15
    print(f"      ---- 주석이 전제한 조건 재현: Use={U_ART} · Nrrp=1 "
          f"(성공 시 prn=1/Use={1/U_ART:.2f} -> theta_p 의 {1/U_ART/G['theta_p']:.1f}배)")
    rr_art = []
    for k in range(N_TRIAL):
        pa = make(MECH_C, 1, nrrp=1)
        pa.set(Use=U_ART)
        pa.seed(3000 + k, 13 * k + 1, 19 * k + 7)
        pa.drive_pre(pre); pa.drive_post(post)
        rr_art.append(float(pa.run(TSTOP)["rho"][-1]))
    rr_art = np.array(rr_art)
    art["use015_nrrp1"] = rr_art
    print(f"      Use={U_ART} Nrrp=1 ca_stp=1: rho 평균 {rr_art.mean():.5f} · "
          f"UP {int(np.sum(rr_art > 0.5))}/{N_TRIAL} · 표준편차 {rr_art.std():.5f}")
    up_art = int(np.sum(rr_art > 0.5))
    up_ours = int(np.sum(art[Nrrp] > 0.5))

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.48)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1])
    axC = fig.add_subplot(gs_[0, 2])
    axD = fig.add_subplot(gs_[1, 0])
    axE = fig.add_subplot(gs_[1, 1])
    axF = fig.add_subplot(gs_[1, 2])

    # A: 소포 수 분포 vs 이론 (이항)
    from math import comb
    obs = [int(np.sum(ves1 == v)) / NT1 for v in range(Nrrp + 1)]
    th = [comb(Nrrp, v) * U ** v * (1 - U) ** (Nrrp - v) for v in range(Nrrp + 1)]
    xx = np.arange(Nrrp + 1)
    axA.bar(xx - 0.19, obs, width=0.38, color="#00838f", label=f"실측 ({NT1}시행)")
    axA.bar(xx + 0.19, th, width=0.38, color="#f9a825", label="이항 이론")
    axA.set_xticks(xx); axA.set_xlabel("첫 펄스에서 방출된 소포 수")
    axA.set_ylabel("확률")
    axA.set_title(f"A. 첫 펄스 방출은 이항분포 B({Nrrp}, {U})\n"
                  f"방출률 실측 {obs_p1:.3f} vs 이론 {p_rel1:.3f}",
                  fontsize=9.2, loc="left")
    axA.legend(fontsize=7.8)
    for i, (o, t_) in enumerate(zip(obs, th)):
        axA.text(i - 0.19, o, f"{o:.3f}", ha="center", va="bottom", fontsize=7)
        axA.text(i + 0.19, t_, f"{t_:.3f}", ha="center", va="bottom", fontsize=7)

    # B: 개별 시행 전도도
    for k, col in zip(range(4), ["#c62828", "#1565c0", "#2e7d32", "#6a1b9a"]):
        p = make(MECH_C, 0)
        p.seed(1000 + k, 7 * k + 1, 13 * k + 3)
        p.rec_dt = 0.05
        p.drive_pre(pre); p.drive_post(post)
        Rk = p.run(400.0)
        axB.plot(Rk["t"], Rk["g"] * 1e3, lw=1.1, color=col, alpha=0.85,
                 label=f"시행 {k}")
    pb = make(MECH_B, 0); pb.rec_dt = 0.05
    pb.drive_pre(pre); pb.drive_post(post)
    Rb = pb.run(400.0)
    axB.plot(Rb["t"], Rb["g"] * 1e3, lw=2.0, color="#37474f", ls="--", label="B 결정론")
    axB.set_xlim(0, 250); axB.set_xlabel("시간 (ms)"); axB.set_ylabel("시냅스 g (nS)")
    axB.set_title("B. 전달은 시행마다 다르다\n빠진 봉우리 = 방출 실패",
                  fontsize=9.2, loc="left")
    axB.legend(fontsize=7.2)

    # C: 미시딩 함정
    axC.bar([0, 1], [iU["n_rel"], iS["n_rel"]], color=["#c62828", "#2e7d32"],
            width=0.5)
    axC.axhline(trials[0]["n_rel"].mean(), color="#37474f", ls="--", lw=1.4,
                label=f"시딩 평균 {trials[0]['n_rel'].mean():.1f}")
    axC.set_xticks([0, 1]); axC.set_xticklabels(["미시딩", "시딩 (1,2,3)"])
    axC.set_ylabel(f"방출 성공 펄스 수 (총 {n_pulse})")
    for x, v in zip([0, 1], [iU["n_rel"], iS["n_rel"]]):
        axC.text(x, v, str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
    axC.set_title("C. ★RNG 미시딩 = 죽은 시냅스\n"
                  "urand()=0 -> 첫 펄스만 방출하고 회복 못 한다",
                  fontsize=9.2, loc="left")
    axC.legend(fontsize=7.8)

    # D: 관례별 효능 분산
    for cst, col in zip((0, 1), ["#37474f", "#c62828"]):
        rr = trials[cst]["rho"]
        axD.scatter(np.arange(N_TRIAL), rr, s=14, color=col, alpha=0.8,
                    label=f"ca_stp={cst} (sd {rr.std():.4f})")
        axD.axhline(refs[cst]["rho_end"], color=col, ls="--", lw=1.2)
    axD.axhline(0.5, color="#2e7d32", ls=":", lw=1.6, label="rho*=0.5")
    axD.set_xlabel("시행 번호"); axD.set_ylabel("최종 효능 rho")
    axD.set_title("D. ★가소성이 흩어지는지는 **관례**가 정한다\n"
                  "ca_stp=0: 방출 실패해도 칼슘 주입 -> 분산 0", fontsize=9.2, loc="left")
    axD.legend(fontsize=7.3)

    # E: 방출 성공 수 분포
    axE.hist(trials[0]["n_rel"],
             bins=np.arange(trials[0]["n_rel"].min() - 0.5,
                            trials[0]["n_rel"].max() + 1.5, 1),
             color="#00838f", alpha=0.85, edgecolor="white")
    axE.axvline(trials[0]["n_rel"].mean(), color="#c62828", lw=2.0,
                label=f"평균 {trials[0]['n_rel'].mean():.1f}/{n_pulse}")
    axE.set_xlabel(f"시행당 방출 성공 펄스 수 (총 {n_pulse})"); axE.set_ylabel("시행 수")
    axE.set_title("E. 트레인 전체 방출률은 첫 펄스보다 훨씬 낮다\n"
                  f"억압형이라 자원이 고갈된다 (Dep {Dp:.0f}ms)", fontsize=9.2, loc="left")
    axE.legend(fontsize=7.8)

    # F: 인공물 (Nrrp=1 vs 2)
    for key, col, lb in ((1, "#ef6c00", f"Nrrp=1 Use={U:g}"),
                         (Nrrp, "#2e7d32", f"Nrrp={Nrrp} Use={U:g} (우리 설정)"),
                         ("use015_nrrp1", "#c62828", f"Nrrp=1 Use={U_ART} (주석 조건)")):
        v = art[key]
        axF.hist(v, bins=14, alpha=0.55, color=col,
                 label=f"{lb} — UP {int(np.sum(v > 0.5))}/{N_TRIAL}")
    axF.axvline(0.5, color="#37474f", ls=":", lw=1.8)
    axF.set_xlabel("최종 효능 rho (ca_stp=1)"); axF.set_ylabel("시행 수")
    axF.set_title("F. 문서화된 인공물은 **Use 에 달렸다**\n"
                  f"성공 시 prn=1/Use: {1/U_ART:.1f}(Use {U_ART}) vs {1/U:.1f}(우리 {U:g})",
                  fontsize=9.2, loc="left")
    axF.legend(fontsize=6.9)

    fig.suptitle("5-5  엔진 C (B + 확률 다소포 방출) — 전달은 흩어지고 칼슘은 관례가 정한다",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"5-5 | {cls} Use {U}/Nrrp {Nrrp} · {N_TRIAL}시행 · "
                     f"첫 펄스 방출률 {obs_p1:.3f}(이론 {p_rel1:.3f}) · "
                     f"rho 표준편차 ca_stp0 {sd0:.4f} / ca_stp1 {sd1:.4f} · "
                     f"미시딩 방출 {iU['n_rel']}/{n_pulse}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "5-5_engine_c.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    max_bin_err = max(abs(o - t_) for o, t_ in zip(obs, th))
    checks = [
        (f"★첫 펄스 방출률이 이론 1-(1-Use)^Nrrp={p_rel1:.4f} 와 통계적으로 일치 "
         f"(3 SE 이내, 실측 {obs_p1:.4f} · {abs(obs_p1-p_rel1)/se_p1:.2f} SE)",
         abs(obs_p1 - p_rel1) < 3 * se_p1),
        (f"소포 수 분포가 이항분포 B({Nrrp},{U}) 와 일치 (최대 편차 {max_bin_err:.4f} < 0.05)",
         max_bin_err < 0.05),
        (f"정규화 방출 prn 시행평균이 1 과 통계적으로 같다 (3 SE 이내, "
         f"실측 {mean_prn1:.4f} · {abs(mean_prn1-1)/se_prn1:.2f} SE)",
         abs(mean_prn1 - 1.0) < 3 * se_prn1),
        (f"★RNG 미시딩 검출: 방출이 첫 펄스 1회뿐 (실측 {iU['n_rel']}/{n_pulse})",
         iU["n_rel"] == 1),
        (f"시딩하면 방출이 여러 번 일어난다 (실측 {iS['n_rel']}회)", iS["n_rel"] > 1),
        ("★ca_stp=0 은 효능이 전혀 흩어지지 않는다 (표준편차 = 0) — "
         "방출 실패에도 칼슘을 주입하기 때문",
         sd0 < 1e-12),
        ("★ca_stp=0 의 효능이 결정론 B 와 정확히 같다",
         abs(trials[0]["rho"].mean() - refs[0]["rho_end"]) < 1e-9),
        ("★ca_stp=1 은 효능이 시행마다 흩어진다 (표준편차 > 0)", sd1 > 0.0),
        (f"★문서화된 인공물 재현 (Use={U_ART}·Nrrp=1): UP {up_art}/{N_TRIAL} 로 "
         f"우리 설정({up_ours}/{N_TRIAL})보다 훨씬 많다", up_art > up_ours),
        (f"★우리 Use={U:g} 는 이 인공물을 크게 완화한다 (성공 시 prn "
         f"{1/U_ART:.2f} -> {1/U:.2f})", (1 / U) < (1 / U_ART) / 2),
        ("Nrrp=1 은 Nrrp=2 보다 시행간 분산이 크다 (전부-또는-전무 방출)",
         art[1].std() > art[Nrrp].std()),
        ("★확률성이 평균 효능을 **올린다** (문턱 비선형성) — ca_stp=1 에서 "
         "확률 평균 > 결정론 B",
         trials[1]["rho"].mean() > refs[1]["rho_end"]),
        (f"트레인 전체 방출률이 첫 펄스보다 낮다 (억압형 자원 고갈)",
         trials[0]["n_rel"].mean() / n_pulse < p_rel1),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(mech=dict(B=MECH_B, C=MECH_C), rec_dt=REC_DT, v_hold=V_HOLD, t0=T0,
               rho0=RHO0, syn_class=cls,
               stp=dict(Use=U, Dep_ms=Dp, Fac_ms=Fc, Nrrp=Nrrp),
               theory=dict(p_release_first=p_rel1, p_fail_first=1 - p_rel1,
                           binomial_obs=[round(v, 5) for v in obs],
                           binomial_theory=[round(v, 5) for v in th],
                           max_bin_err=max_bin_err),
               protocol=dict(n_burst=N_BURST, burst_hz=BURST_HZ, n_in=N_IN,
                             in_hz=IN_HZ, n_pulses=n_pulse, tstop=TSTOP),
               engine_B=refs,
               first_pulse=dict(n_trial=NT1, release_rate=obs_p1, se_p=se_p1,
                                mean_prn=mean_prn1, se_prn=se_prn1),
               trials={str(k): dict(n=N_TRIAL, rho_mean=float(v["rho"].mean()),
                                    rho_sd=float(v["rho"].std()),
                                    rho_min=float(v["rho"].min()),
                                    rho_max=float(v["rho"].max()),
                                    n_up=int(np.sum(v["rho"] > 0.5)),
                                    n_rel_mean=float(v["n_rel"].mean()),
                                    n_rel_min=int(v["n_rel"].min()),
                                    n_rel_max=int(v["n_rel"].max()))
                       for k, v in trials.items()},
               artifact={str(k): dict(rho_mean=float(v.mean()), rho_sd=float(v.std()),
                                      n_up=int(np.sum(v > 0.5)))
                         for k, v in art.items()},
               artifact_use=U_ART,
               unseeded_trap=dict(n_pre=iU["n_pre"], n_rel=iU["n_rel"],
                                  seeded_n_rel=iS["n_rel"],
                                  rho_end=iU["rho_end"]),
               finding=("★엔진 C 의 확률성이 **가소성까지 흩어지게 하는지는 관례가 정한다.** "
                        "논문 원본(ca_stp=0)은 방출이 실패해도 C_pre 를 그대로 주입하므로 효능 "
                        "표준편차가 정확히 0 이고 결정론 B 와 완전히 같다 — 전달만 확률적이다. "
                        "생리적으로는 방출 실패 = 글루타메이트 없음 = NMDA 칼슘 없음 이어야 하므로 "
                        "이것은 논문 모델의 내적 불일치다(GAPS 후보). ca_stp=1 로 켜면 효능이 "
                        "흩어지지만, 그 경로는 mod 주석이 인공물이라고 경고한 정규화에 의존한다."),
               artifact_note=(f"mod 주석이 경고한 'prn=6.67' 인공물은 **Use=0.15(SC->PC E1s) "
                              f"조건의 값**이다. 우리 Use={U:g} 에서는 1/Use={1/U:.2f} 뿐이라 "
                              f"theta_p={G['theta_p']} 를 {1/U/G['theta_p']:.1f}배만 넘는다. "
                              f"직접 재현해 확인했다: Use={U_ART}·Nrrp=1 은 UP {up_art}/{N_TRIAL}, "
                              f"우리 설정은 {up_ours}/{N_TRIAL}. 즉 이 인공물은 Nrrp 뿐 아니라 "
                              f"**Use 에 더 강하게 달렸고, 우리 파라미터가 그것을 피한다.**"),
               jensen=(f"★확률성이 평균 효능을 올린다. ca_stp=1 에서 확률 엔진 시행평균 "
                       f"{float(trials[1]['rho'].mean()):.5f} > 결정론 B "
                       f"{refs[1]['rho_end']:.5f}. 칼슘 문턱이 비선형이라 '가끔 크게' 가 "
                       f"'항상 조금' 보다 강화를 많이 만든다(Jensen 효과). 즉 확률 방출은 "
                       f"단순한 잡음이 아니라 **평균값 자체를 바꾼다** — 6단계에서 결정론 "
                       f"결과로 확률 엔진을 대신할 수 없다."),
               reporting_rule=("6단계에서 확률 엔진 결과는 (가) 시드 고정 비교와 (나) 시드 스윕 "
                               "분포 두 줄로 보고한다. ca_stp=0 이면 분산이 0 이므로 시드 스윕이 "
                               "불필요하다는 것도 함께 적는다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-5_gb_c.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-5 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
