# -*- coding: utf-8 -*-
"""5-9 단기가소성 검증 (횡단) — 단기 능력을 가진 엔진 전부

단계   : 5-9 (5단계 가소성 엔진 / 하위 9 stp_verify)
쉬운 설명: 단기가소성은 별도 단계가 아니라 **엔진의 능력**이다. 그래서 이 검증은 특정 엔진이
          아니라 "단기 능력을 선언한 엔진 전부" 를 횡단한다. 대상은 레지스트리가 정한다.
방법   : 짝펄스(PPR) · 트레인 · 회복 세 가지를 재고 Tsodyks-Markram 참조와 대조한다.
★필수 : **장기가소성을 반드시 동결하고 잰다.** 칼슘 감쇠 시상수가 48.8ms 이므로 짧은 ISI
          짝펄스는 잔류 칼슘을 남겨 약화 문턱(theta_d)을 넘길 수 있다. 즉 단기가소성을
          재려는 자극이 스스로 LTD 를 유발해 측정을 오염시킨다. 그래서
          (1) 전부 동결 상태로 재고
          (2) **동결하지 않으면 어느 ISI 부터 오염되는지 칼슘으로 실측**한다.
검증   : 방출량이 TM 참조와 절대차 < 1e-6 (직접 노출하는 엔진) · 오염 ISI 하한 산출.
근거   : Tsodyks & Markram 1997 / Fuhrmann 2002 · Ecker 2020 PC->PC (억압형)
결과   : figures/5-9_ppr.png · figures/5-9_train.png · figures/5-9_recovery.png ·
          figures/5-9_stp.json
실행   : . .\\env\\activate.ps1 ; & $Py04 05_engines\\9_stp_verify\\5-9_stp_verify.py
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
from lib.refs import tm, gb                          # noqa: E402
from lib.wiring import load_synapse_cfg              # noqa: E402

REC_DT = 0.1                 # pr_last 표본·전도도 증분에 충분 (속도)
V_HOLD = -70.0
T0 = 20.0
DT_SIM = 0.025
TOL_PR = 1e-6                # 방출량을 직접 노출하는 엔진
TOL_G = 1e-5                 # 간접 측정(차분법)도 정확하다 — 아래 release_sub 참조
# ★확률 엔진은 A·B·C 본 스윕에서 제외하고 **별도 수렴 검정**을 한다.
#   prn 이 {0, 1, 2} 이산값이라 분산이 크다 — 40시행이면 SE 0.13 인데 참값이 0.65 다.
#   게다가 같은 시드 집합을 전 조건에 재사용하면 오차가 상관되어 계통 편향처럼 보인다.
#   -> 대표 조건 몇 개에 많은 시행을 주고 3 SE 로 검정한다(D26 (5)).
N_TRIAL_C = 400
C_ISIS = [10.0, 50.0, 400.0]        # 수렴 검정용 대표 ISI
C_TRAIN_HZ = 50.0

ISIS = [10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
TRAIN_HZ = [5.0, 10.0, 20.0, 50.0, 100.0]
N_TRAIN = 8
REC_DELAYS = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]


def main():
    plots.setup()
    print("=== 5-9 단기가소성 검증 (횡단) ===")
    cls, P = load_synapse_cfg()
    U, Dp, Fc = P["Use"], P["Dep_ms"], P["Fac_ms"]
    G = gb.WITTENBERG2006
    targets_all = engines.with_cap("stp")
    targets = [k for k in targets_all if not engines.get(k)["prob"]]
    prob_targets = [k for k in targets_all if engines.get(k)["prob"]]
    print(f"  대상(레지스트리가 결정): {targets_all}")
    print(f"    결정론 {targets} 은 전 스윕 · 확률 {prob_targets} 은 별도 수렴 검정")
    print(f"  시냅스 {cls} · Use {U} · Dep {Dp}ms · Fac {Fc}ms "
          f"-> {'억압형' if Fc < Dp else '촉진형'}")
    print(f"  ★전 측정 동결 상태 (rho0=0 = 안정 고정점, D21)")

    def probe(key, seed=None):
        e = engines.get(key)
        p = SynProbe(e["mech"], clamp=True, v_hold=V_HOLD, rec_dt=REC_DT)
        p.set_gmax(P["g_nS"])
        engines.apply_params(p.syn, key, P, rho0=0.0, frozen=True)
        if e["prob"] and seed is not None:
            p.seed(*seed)
        return p, e

    def jumps(R, times, win=3.0):
        out = []
        for k, ts in enumerate(times):
            te = times[k + 1] if k + 1 < len(times) else ts + win + 1.0
            m = (R["t"] >= ts) & (R["t"] < min(ts + win, te))
            base = float(R["g"][R["t"] <= ts][-1]) if (R["t"] <= ts).any() else 0.0
            out.append((float(R["g"][m].max()) - base) * 1e3 if m.any() else 0.0)
        return np.array(out)

    def raw_release(key, times, tstop, seed=None):
        """**정규화하지 않은** 방출량 배열. 확률 엔진은 여기서 나눠선 안 된다 —
        첫 펄스가 실패하면 0 으로 나누게 된다(실측: 전부 NaN). 시행평균을 먼저 낸다."""
        p, e = probe(key, seed)
        p.drive_pre(times)
        p.record(extra=("pr_last",) if "pr_last" in e["states"] else ())
        Rr = p.run(tstop, dt=DT_SIM)
        if "pr_last" in Rr:
            vals = []
            for ts in times:
                idx = np.searchsorted(Rr["t"], ts + 0.15)
                vals.append(float(Rr["pr_last"][min(idx, len(Rr["pr_last"]) - 1)]))
            return np.array(vals), "직접(pr_last)"
        return None, "간접"

    def release_sub(key, times, tstop):
        """★차분법 — n번째 펄스의 기여를 **정확히** 분리한다.

        n 펄스 실행과 n-1 펄스 실행의 전도도 차이는 n번째 펄스의 커널 그 자체다
        (앞선 이력이 동일하므로 방출량도 동일하다). 그 봉우리 = 방출량 x 단위봉우리.
        '봉우리 - 직전값' 방식은 tau_d_AMPA=3ms 꼬리가 겹치는 짧은 ISI 에서 1.5% 까지
        과소평가한다(실측 ISI 10ms: 0.6387 vs 참조 0.6483). 차분법은 그 편향이 없다.
        결정론 엔진에만 쓴다(확률 엔진은 시행마다 이력이 달라 성립하지 않는다).
        """
        gs = []
        for n in range(1, len(times) + 1):
            p, e = probe(key)
            p.drive_pre(times[:n])
            p.record()
            Rr = p.run(tstop, dt=DT_SIM)
            gs.append(Rr["g"].copy())
            tt = Rr["t"]
        amps = [float(gs[0].max())]
        for n in range(1, len(times)):
            d = gs[n] - gs[n - 1]
            amps.append(float(d.max()))
        v = np.array(amps)
        return v / v[0], "간접(차분법)"

    def release(key, times, tstop, seed=None):
        """방출 프로파일(첫 펄스=1). 확률 엔진은 raw 를 반환해 호출부가 평균한다."""
        e = engines.get(key)
        if e["prob"]:
            raw, mode = raw_release(key, times, tstop, seed=seed)
            return raw, mode + "·원시", None
        raw, mode = raw_release(key, times, tstop)
        if raw is not None:
            return raw / raw[0], mode, None
        v, mode = release_sub(key, times, tstop)
        return v, mode, None

    # ── (A) 짝펄스 ────────────────────────────────────────────────────────
    print(f"\n  [A] 짝펄스 (PPR) — ISI 스윕")
    ppr = {}
    for key in targets:
        e = engines.get(key)
        rows = []
        for isi in ISIS:
            times = [T0, T0 + isi]
            v, mode, _ = release(key, times, T0 + isi + 300.0)
            r = float(v[1]); sd = 0.0
            _, _, amp = tm.simulate(times, U, Dp, Fc)
            ref = float(amp[1] / amp[0])
            rows.append(dict(isi=isi, ppr=r, ref=ref, err=abs(r - ref), se=sd,
                             mode=mode))
        ppr[key] = rows
        errs = [r["err"] for r in rows]
        print(f"      {key:<5}({rows[0]['mode']}) PPR " +
              " ".join(f"{r['ppr']:.3f}" for r in rows))
        print(f"      {'':<5} 참조     " + " ".join(f"{r['ref']:.3f}" for r in rows) +
              f"   최대 절대차 {max(errs):.2e}")

    # ── (B) 트레인 ────────────────────────────────────────────────────────
    print(f"\n  [B] 트레인 {N_TRAIN}펄스 — 주파수 스윕")
    train = {}
    for key in targets:
        e = engines.get(key)
        rows = []
        for hz in TRAIN_HZ:
            isi = 1000.0 / hz
            times = [T0 + k * isi for k in range(N_TRAIN)]
            tstop = times[-1] + 400.0
            prof, mode, _ = release(key, times, tstop)
            _, ref_prof = tm.train(N_TRAIN, hz, U, Dp, Fc)
            err = float(np.max(np.abs(prof - ref_prof)))
            rows.append(dict(hz=hz, prof=[float(v) for v in prof],
                             ref=[float(v) for v in ref_prof], err=err, mode=mode))
            print(f"      {key:<5}{hz:>5.0f}Hz : " +
                  " ".join(f"{v:.3f}" for v in prof) + f"   최대차 {err:.2e}")
        train[key] = rows

    # ── (C) 회복 ──────────────────────────────────────────────────────────
    print(f"\n  [C] 회복 — 트레인({N_TRAIN}발 50Hz) 뒤 시험펄스 지연 스윕")
    recov = {}
    isi_tr = 1000.0 / 50.0
    for key in targets:
        e = engines.get(key)
        rows = []
        for d in REC_DELAYS:
            times = [T0 + k * isi_tr for k in range(N_TRAIN)]
            times.append(times[-1] + d)
            tstop = times[-1] + 400.0
            v, mode, _ = release(key, times, tstop)
            r = float(v[-1])
            _, _, amp = tm.simulate(times, U, Dp, Fc)
            ref = float(amp[-1] / amp[0])
            rows.append(dict(delay=d, rec=r, ref=ref, err=abs(r - ref), mode=mode))
        recov[key] = rows
        print(f"      {key:<5} 회복 " + " ".join(f"{r['rec']:.3f}" for r in rows))
        print(f"      {'':<5} 참조 " + " ".join(f"{r['ref']:.3f}" for r in rows) +
              f"   최대 절대차 {max(r['err'] for r in rows):.2e}")

    # ── (C2) 확률 엔진 수렴 검정 (별도, 많은 시행 + 3 SE) ──────────────────
    print(f"\n  [C2] 확률 엔진 수렴 검정 — {N_TRIAL_C}시행, 3 SE 로 판정")
    cconv = []
    for key in prob_targets:
        for isi in C_ISIS:
            times = [T0, T0 + isi]
            acc = []
            for k in range(N_TRIAL_C):
                v, _, _ = release(key, times, T0 + isi + 300.0,
                                  seed=(9000 + k, 5 * k + 1, 7 * k + 3))
                acc.append(v)
            A = np.array(acc)
            M = A.mean(axis=0)
            r = float(M[1] / M[0])
            # 비의 표준오차 (델타법, 두 항 상관 무시 = 보수적)
            se = float(np.sqrt((A[:, 1].var(ddof=1) / M[0] ** 2
                                + (M[1] ** 2 / M[0] ** 4) * A[:, 0].var(ddof=1))
                               / N_TRIAL_C))
            _, _, amp = tm.simulate(times, U, Dp, Fc)
            ref = float(amp[1] / amp[0])
            z = abs(r - ref) / max(se, 1e-12)
            cconv.append(dict(key=key, isi=isi, ppr=r, ref=ref, se=se, z=z,
                              mean_prn_first=float(M[0])))
            print(f"      {key} ISI {isi:>5.0f}ms : PPR {r:.4f} +- {se:.4f}(SE) · "
                  f"참조 {ref:.4f} · {z:.2f} SE · 첫 펄스 prn 평균 {M[0]:.4f}")
    max_z = max(c["z"] for c in cconv) if cconv else 0.0
    prn1_err = max(abs(c["mean_prn_first"] - 1.0) for c in cconv) if cconv else 0.0
    print(f"      -> 최대 {max_z:.2f} SE · 첫 펄스 prn 평균이 1 에서 최대 {prn1_err:.4f} 벗어남")

    # ── (D) ★동결하지 않으면 어느 ISI 부터 오염되는가 ──────────────────────
    print(f"\n  [D] ★동결하지 않으면 오염된다 — 짝펄스만으로 칼슘이 theta_d 를 넘는 ISI")
    contam = []
    for isi in ISIS:
        times = [T0, T0 + isi]
        tt = np.arange(0.0, T0 + isi + 500.0, 0.1)
        c = gb.calcium(tt, times, [], G)      # pre 만 (post 없음)
        cmax = float(c.max())
        od = float(0.1 * np.sum(c > G["theta_d"]))
        op = float(0.1 * np.sum(c > G["theta_p"]))
        contam.append(dict(isi=isi, c_max=cmax, over_d_ms=od, over_p_ms=op))
        print(f"      ISI {isi:>6.0f}ms : c_max {cmax:.3f} · theta_d 초과 {od:6.1f}ms · "
              f"theta_p 초과 {op:5.1f}ms  {'<- 오염' if od > 0 else ''}")
    safe = [r["isi"] for r in contam if r["over_d_ms"] == 0.0]
    isi_safe = min(safe) if safe else None
    print(f"      -> 동결 없이 안전한 ISI 하한: "
          f"{f'{isi_safe:.0f} ms 이상' if isi_safe else '없음 (전 구간 오염)'}")
    # 실제로 동결을 풀면 효능이 변하는가 (확인 사살)
    unfrozen = []
    for isi in (10.0, 400.0):
        p = SynProbe("GBPlasticityStpSyn", clamp=True, v_hold=V_HOLD, rec_dt=1.0)
        p.set_gmax(P["g_nS"])
        engines.apply_params(p.syn, "B", P, rho0=0.3, frozen=False)
        p.drive_pre([T0, T0 + isi])
        R = p.run(T0 + isi + 2000.0, dt=DT_SIM)
        d = float(R["rho"][-1] - R["rho"][0])
        unfrozen.append(dict(isi=isi, drho=d))
        print(f"      동결 해제(rho0=0.3) ISI {isi:.0f}ms -> rho 변화 {d:+.3e}")

    # ── 그림 3장 ──────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    outdir = plots.figdir(__file__)
    COL = {"det": "#1565c0", "B": "#c62828", "C": "#2e7d32"}

    # 1) PPR
    fig1, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for key in targets:
        rows = ppr[key]
        a1.semilogx([r["isi"] for r in rows], [r["ppr"] for r in rows], "o-",
                    color=COL.get(key, "#607d8b"), ms=6, lw=1.8, label=f"{key} (mod)")
    a1.semilogx([r["isi"] for r in ppr[targets[0]]],
                [r["ref"] for r in ppr[targets[0]]], "k--", lw=1.4, label="TM 참조")
    a1.axhline(1.0, color="#90a4ae", ls=":", lw=1.0)
    a1.set_xlabel("ISI (ms)"); a1.set_ylabel("PPR (2번째/1번째)")
    a1.set_title(f"A. 짝펄스 — {cls} 는 억압형 (Fac {Fc:.0f} < Dep {Dp:.0f})\n"
                 "전 엔진이 TM 참조와 겹친다", fontsize=9.5, loc="left")
    a1.legend(fontsize=8)
    for key in targets:
        rows = ppr[key]
        a2.semilogx([r["isi"] for r in rows],
                    [max(r["err"], 1e-18) for r in rows], "o-",
                    color=COL.get(key, "#607d8b"), ms=5, lw=1.4, label=key)
    a2.set_yscale("log"); plots.ascii_log(a2)
    a2.axhline(TOL_PR, color="#c62828", ls="--", lw=1.2, label=f"직접 허용 {TOL_PR:.0e}")
    a2.axhline(TOL_G, color="#1565c0", ls=":", lw=1.2, label=f"간접 허용 {TOL_G:.0e}")
    a2.set_xlabel("ISI (ms)"); a2.set_ylabel("|mod - 참조|")
    a2.set_title("B. 참조와의 절대차\n확률 엔진(C)은 시행평균이라 표집오차가 남는다",
                 fontsize=9.5, loc="left")
    a2.legend(fontsize=7.5)
    fig1.suptitle("5-9  단기가소성 검증 — 짝펄스 (전 측정 장기 동결)", fontsize=12, y=0.99)
    fig1.subplots_adjust(top=0.80)
    plots.stamp(fig1, f"5-9 | 대상 {targets} (레지스트리 결정) · Use {U}/Dep {Dp}/Fac {Fc} · "
                      f"동결 rho0=0 · 확률 엔진 {N_TRIAL_C}시행 평균")
    plots.save(fig1, outdir, "5-9_ppr.png")

    # 2) 트레인
    fig2, axs = plt.subplots(1, len(TRAIN_HZ), figsize=(15.2, 3.9), sharey=True)
    x = np.arange(1, N_TRAIN + 1)
    for j, hz in enumerate(TRAIN_HZ):
        ax = axs[j]
        for key in targets:
            r = train[key][j]
            ax.plot(x, r["prof"], "o-", color=COL.get(key, "#607d8b"), ms=4, lw=1.5,
                    label=key)
        ax.plot(x, train[targets[0]][j]["ref"], "k--", lw=1.2, label="TM 참조")
        ax.axhline(1.0, color="#90a4ae", ls=":", lw=0.9)
        ax.set_xticks(x); ax.set_xlabel("펄스")
        ax.set_title(f"{hz:.0f} Hz", fontsize=10)
        if j == 0:
            ax.set_ylabel("정규화 방출 (첫 펄스=1)"); ax.legend(fontsize=7.5)
    fig2.suptitle(f"5-9  단기가소성 검증 — 트레인 {N_TRAIN}펄스 (장기 동결)",
                  fontsize=12, y=0.99)
    fig2.subplots_adjust(top=0.76)
    plots.stamp(fig2, f"5-9 | 최대 절대차 " +
                " · ".join(f"{k} {max(r['err'] for r in train[k]):.1e}" for k in targets))
    plots.save(fig2, outdir, "5-9_train.png")

    # 3) 회복 + 오염
    fig3, (b1, b2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for key in targets:
        rows = recov[key]
        b1.semilogx([r["delay"] for r in rows], [r["rec"] for r in rows], "o-",
                    color=COL.get(key, "#607d8b"), ms=6, lw=1.8, label=key)
    b1.semilogx([r["delay"] for r in recov[targets[0]]],
                [r["ref"] for r in recov[targets[0]]], "k--", lw=1.4, label="TM 참조")
    b1.axhline(1.0, color="#90a4ae", ls=":", lw=1.0)
    b1.set_xlabel("트레인 종료 후 시험펄스 지연 (ms)")
    b1.set_ylabel("시험펄스 방출 (첫 펄스=1)")
    b1.set_title(f"C. 회복 — {N_TRAIN}발 50Hz 뒤\nDep {Dp:.0f}ms 로 느리게 돌아온다",
                 fontsize=9.5, loc="left")
    b1.legend(fontsize=8)

    od = [r["over_d_ms"] for r in contam]
    op = [r["over_p_ms"] for r in contam]
    xx = np.arange(len(contam))
    b2.bar(xx - 0.19, od, width=0.38, color="#ef6c00", label="theta_d 초과 (약화)")
    b2.bar(xx + 0.19, op, width=0.38, color="#2e7d32", label="theta_p 초과 (강화)")
    b2.set_xticks(xx); b2.set_xticklabels([f"{r['isi']:.0f}" for r in contam],
                                          fontsize=8)
    b2.set_xlabel("짝펄스 ISI (ms)"); b2.set_ylabel("칼슘 문턱 초과 시간 (ms)")
    if isi_safe:
        b2.axvline(ISIS.index(isi_safe) - 0.5, color="#c62828", ls="--", lw=1.6,
                   label=f"안전 하한 {isi_safe:.0f}ms")
    b2.set_title("D. ★동결하지 않으면 짝펄스가 스스로 LTD 를 유발한다\n"
                 "pre 만 줘도 칼슘이 theta_d 를 넘는다", fontsize=9.5, loc="left")
    b2.legend(fontsize=7.8)
    fig3.suptitle("5-9  단기가소성 검증 — 회복 + 측정 오염 조건", fontsize=12, y=0.99)
    fig3.subplots_adjust(top=0.80)
    plots.stamp(fig3, f"5-9 | 동결 없이 안전한 ISI "
                      f"{f'{isi_safe:.0f}ms 이상' if isi_safe else '없음'} · "
                      f"동결 해제 시 rho 변화 ISI 10ms {unfrozen[0]['drho']:+.2e} / "
                      f"400ms {unfrozen[1]['drho']:+.2e}")
    plots.save(fig3, outdir, "5-9_recovery.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    def maxerr(d, key):
        return max(r["err"] for r in d[key])

    direct = [k for k in targets if "pr_last" in engines.get(k)["states"]]
    indirect = [k for k in targets if k not in direct]
    checks = [
        (f"대상이 레지스트리 선언으로 결정된다 ({targets_all})",
         targets_all == engines.with_cap("stp")),
        (f"★직접 노출 엔진 {direct} 의 PPR 이 TM 참조와 절대차 < {TOL_PR:.0e}",
         all(maxerr(ppr, k) < TOL_PR for k in direct)),
        (f"★직접 노출 엔진 트레인도 < {TOL_PR:.0e}",
         all(maxerr(train, k) < TOL_PR for k in direct)),
        (f"★직접 노출 엔진 회복도 < {TOL_PR:.0e}",
         all(maxerr(recov, k) < TOL_PR for k in direct)),
        (f"★차분법 간접 측정 엔진 {indirect} 도 < {TOL_G:.0e}",
         all(maxerr(ppr, k) < TOL_G and maxerr(train, k) < TOL_G
             and maxerr(recov, k) < TOL_G for k in indirect)),
        (f"★확률 엔진 {prob_targets} 의 {N_TRIAL_C}시행 평균이 결정론 참조와 "
         f"통계적으로 같다 (최대 {max_z:.2f} SE < 3)", max_z < 3.0),
        (f"확률 엔진의 첫 펄스 정규화 방출 평균이 1 이다 (편차 {prn1_err:.4f} < 0.1)",
         prn1_err < 0.1),
        (f"{cls} 는 전 ISI 에서 억압형 (PPR < 1)",
         all(r["ppr"] < 1.0 for k in targets for r in ppr[k])),
        ("ISI 가 길어지면 PPR 이 1 로 회복한다",
         all(ppr[k][-1]["ppr"] > ppr[k][0]["ppr"] for k in targets)),
        ("트레인은 주파수가 높을수록 더 억압된다",
         all(train[k][-1]["prof"][-1] < train[k][0]["prof"][-1] for k in targets)),
        ("회복은 지연이 길수록 커진다 (단조)",
         all(all(recov[k][i]["rec"] <= recov[k][i + 1]["rec"] + 1e-9
                 for i in range(len(REC_DELAYS) - 1)) for k in targets)),
        ("★동결 없이는 짧은 ISI 짝펄스가 오염된다 (ISI 10ms 에서 theta_d 초과 > 0)",
         contam[0]["over_d_ms"] > 0),
        (f"★안전 ISI 하한이 산출된다 "
         f"({f'{isi_safe:.0f}ms' if isi_safe else '없음'})", isi_safe is not None),
        ("★동결을 풀면 실제로 효능이 변한다 (짧은 ISI 에서)",
         abs(unfrozen[0]["drho"]) > 1e-9),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(targets=targets_all, deterministic=targets,
               probabilistic=prob_targets, syn_class=cls,
               stp=dict(Use=U, Dep_ms=Dp, Fac_ms=Fc,
                        kind="억압형" if Fc < Dp else "촉진형"),
               frozen=True, rho0=0.0, n_trial_prob=N_TRIAL_C,
               tol=dict(direct=TOL_PR, indirect=TOL_G),
               A_ppr={k: v for k, v in ppr.items()},
               B_train={k: v for k, v in train.items()},
               C_recovery={k: v for k, v in recov.items()},
               C2_prob_convergence=dict(rows=cconv, n_trial=N_TRIAL_C,
                                        max_z=max_z, prn_first_err=prn1_err),
               D_contamination=dict(rows=contam, safe_isi_ms=isi_safe,
                                    unfrozen_check=unfrozen),
               finding=("단기가소성 검증은 반드시 장기가소성을 동결하고 해야 한다. "
                        f"pre 짝펄스만 줘도 ISI {ISIS[0]:.0f}ms 에서 칼슘이 theta_d 를 "
                        f"{contam[0]['over_d_ms']:.1f}ms 동안 넘는다 — 즉 측정 자극이 스스로 "
                        f"LTD 를 유발한다. 동결을 풀고 확인하니 실제로 효능이 변했다"
                        f"({unfrozen[0]['drho']:+.2e}). 동결 없이 안전한 ISI 하한은 "
                        f"{f'{isi_safe:.0f}ms' if isi_safe else '없음'} 이다."),
               measurement_note=("★측정 방법이 결과를 바꾼다. '봉우리 - 직전값' 으로 펄스별 "
                                 "증분을 재면 tau_d_AMPA=3ms 꼬리가 겹치는 짧은 ISI 에서 "
                                 "최대 1.5% 과소평가한다(ISI 10ms: 0.6387 vs 참조 0.6483). "
                                 "n펄스 실행과 n-1펄스 실행의 **차분**으로 n번째 커널을 분리하면 "
                                 "그 편향이 사라진다. 확률 엔진에는 쓸 수 없다(시행마다 이력이 "
                                 "다르다) — 대신 원시 방출량을 시행평균한 뒤 정규화한다. "
                                 "시행별로 먼저 정규화하면 첫 펄스 실패 시 0 으로 나눠 NaN 이 된다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "5-9_stp.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 5-9 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
