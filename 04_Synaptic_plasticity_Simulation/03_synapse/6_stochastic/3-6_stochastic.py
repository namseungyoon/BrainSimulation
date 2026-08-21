# -*- coding: utf-8 -*-
"""3-6 확률 방출 — 같은 자극에도 시행마다 다른 방출 (분포·CV·실패율)

단계   : 3-6 (파이프라인 3단계 시냅스 / 하위 6 stochastic)
쉬운 설명: 실제 시냅스는 자극이 와도 신경전달물질을 '확률적으로' 방출한다 — 어떤 때는 방출,
          어떤 때는 실패(0). 결정론 시냅스(3-5)와 달리 시행마다 EPSP 크기가 흔들린다.
방법   : 확률 방출 시냅스(모델 C)로 pre 1발을 N회 반복(시행마다 RNG 재시딩).
          정착(SETTLE_MS)은 1회만 하고 스냅샷에서 복원해 매 시행 기저선을 동일하게 만든다.
          시냅스별 방출 소포 수와 soma EPSP 진폭 분포를 모아 방출확률·CV·실패율을 잰다.
검증   : 다소포 방출(EMS)에서 시냅스당 방출확률 = 1-(1-Use)^Nrrp (Nrrp 자리가 각각 Use 로 방출).
         연결 실패율 = (1-Use)^(Nrrp x 시냅스수) (전 시냅스·전 자리가 모두 실패).
         ★ RNG 미시딩 함정(방출확률이 1.0 에 붙으면 setRNG 안 된 것) 동시 점검.
근거   : 확률 다소포 방출(BBP EMS, Random123). PC->PC(Ecker Table3): Use=0.50 · Nrrp=2
         -> 시냅스당 방출확률 1-0.5^2 = 0.75 · 시냅스 2개 연결 실패율 0.5^4 = 0.0625.
결과   : figures/3-6_amp_hist.png · figures/3-6_stochastic.json
실행   : . .\\env\\activate.ps1 ; & $Py04 03_synapse\\6_stochastic\\3-6_stochastic.py
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

import numpy as np                          # noqa: E402
from lib import plots                        # noqa: E402
from lib import measure                       # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.wiring import Wiring                 # noqa: E402
from lib.nrnenv import h                     # noqa: E402
import lib.nrnenv as nrnenv                  # noqa: E402

from lib.wiring import SETTLE_MS   # noqa: E402
T_SPIKE = SETTLE_MS + 10.0     # 정착 후 자극 (기저선 표류 제거)
TSTOP = T_SPIKE + 40.0
N_TRIAL = 200


def main():
    plots.setup()
    print("=== 3-6 확률 방출 ===")
    b = Bench()
    w = Wiring(b, frozen=True, prob=True)      # 모델 C(확률방출), 장기가소성 동결
    Use = w.p["Use"]; Nrrp = int(w.p["Nrrp"])
    print(f"  확률 시냅스 {len(w.syns)}개 · Use={Use} · Nrrp={Nrrp}")
    w.drive_pre_iclamp([T_SPIKE], amp_nA=1.2, dur_ms=3.0)
    w.record(rec_dt=0.1, local_v=False, currents=False)
    t_event = T_SPIKE + b.syn_specs[0]["delay_ms"]

    amps = []               # 시행별 soma EPSP 진폭
    ves = []                # 시행별 시냅스별 방출 소포 수
    # 정착은 1회만 하고 스냅샷을 저장 -> 매 시행은 거기서 복원해 이어간다(빠르고 기저선 동일)
    w.settle()
    for tr in range(N_TRIAL):
        w.restore()
        w.seed_prob(tr)
        w.run_settled(TSTOP)
        R = w.arrays()
        amps.append(measure.peak_amp(R["t"], R["post_v"], t_event))
        ves.append([float(s.ves_last) for s, _ in w.syns])
    amps = np.array(amps); ves = np.array(ves)   # ves: (N_TRIAL, 시냅스수)

    # 통계
    rel_prob = (ves > 0).mean(axis=0)            # 시냅스별 방출확률
    conn_fail = float((ves.sum(axis=1) == 0).mean())   # 연결(전 시냅스 실패) 비율
    cv = measure.cv(amps)
    mean_amp = float(amps.mean())
    # 다소포 방출: Nrrp 개 자리가 각각 Use 로 방출 -> 시냅스가 '무언가 방출'할 확률
    theory_syn = 1.0 - (1.0 - Use) ** Nrrp
    theory_conn_fail = (1 - Use) ** (Nrrp * len(w.syns))   # 전 시냅스·전 자리 실패
    print(f"  시냅스별 방출확률 {[round(x,3) for x in rel_prob]} "
          f"(이론 1-(1-{Use})^{Nrrp} = {theory_syn:.3f})")
    print(f"  평균 방출 소포 수 {ves.mean():.3f} (이론 Use*Nrrp = {Use*Nrrp:.3f})")
    print(f"  연결 실패율 {conn_fail:.3f} (이론 (1-Use)^{Nrrp*len(w.syns)}={theory_conn_fail:.3f})")
    print(f"  soma EPSP 평균 {mean_amp:.3f} mV · CV {cv:.3f}")

    # RNG 함정 점검: 방출확률이 1.0 근처면 setRNG 안 된 것
    rng_ok = bool(np.all(rel_prob < 0.9))

    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                 gridspec_kw={"width_ratios": [1.3, 1]})

    # A: soma EPSP 진폭 히스토그램 (시행간 변동)
    a1.hist(amps, bins=24, color="#7b1fa2", alpha=0.85, edgecolor="white")
    a1.axvline(mean_amp, color="#d81b60", ls="--", lw=1.5, label=f"평균 {mean_amp:.3f} mV")
    a1.set_xlabel("soma EPSP 진폭 (mV)"); a1.set_ylabel(f"시행 수 (총 {N_TRIAL})")
    a1.set_title(f"A. 시행간 EPSP 진폭 분포 — CV {cv:.3f}\n"
                 f"확률 방출이라 시행마다 흔들림(결정론 3-5 는 고정)", fontsize=10, loc="left")
    a1.legend(fontsize=9)

    # B: 시냅스별 방출확률 vs 이론 Use
    x = np.arange(len(w.syns))
    a2.bar(x, rel_prob, color="#7b1fa2", width=0.6, label="실측")
    a2.axhline(theory_syn, color="#2e7d32", ls="--", lw=1.5,
               label=f"이론 1-(1-Use)^Nrrp = {theory_syn:.2f}")
    a2.set_xticks(x); a2.set_xticklabels([f"syn{i+1}\n{round(sp['path_um'])}um"
                                          for i, (_, sp) in enumerate(w.syns)], fontsize=8)
    a2.set_ylabel("방출확률 (1개 이상 방출한 시행 / 전체)")
    a2.set_ylim(0, max(1.0, rel_prob.max()*1.25))
    a2.set_title(f"B. 시냅스별 방출확률 ~ 1-(1-Use)^Nrrp = {theory_syn:.2f}\n"
                 f"연결 실패율 {conn_fail:.3f}(이론 {theory_conn_fail:.3f})",
                 fontsize=10, loc="left")
    a2.legend(fontsize=9)
    for i, rp in enumerate(rel_prob):
        a2.text(i, rp, f"{rp:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"3-6  확률 방출 — pre 1발 × {N_TRIAL}시행 (모델 C, Nrrp={Nrrp}, Use={Use})",
                 fontsize=12.5, y=0.99)
    fig.subplots_adjust(top=0.86, wspace=0.24)
    tag = "정상" if rng_ok else "★RNG 미시딩 의심(방출확률~1)"
    plots.stamp(fig, f"3-6 | {w.class_name} · Use={Use}·Nrrp={Nrrp} · 방출확률 {rel_prob.mean():.3f}"
                     f"(이론 {theory_syn:.3f}) · 실패율 {conn_fail:.3f}(이론 {theory_conn_fail:.3f}) · CV {cv:.3f} · {tag}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-6_amp_hist.png")

    checks = [
        ("RNG 시딩 정상(방출확률<0.9)", rng_ok),
        (f"시냅스 방출확률 ~ 1-(1-Use)^Nrrp={theory_syn:.2f} (±0.08)",
         abs(rel_prob.mean() - theory_syn) < 0.08),
        (f"평균 소포 수 ~ Use*Nrrp={Use*Nrrp:.2f} (±0.15)",
         abs(float(ves.mean()) - Use * Nrrp) < 0.15),
        ("시행간 변동 존재(CV>0.1)", cv > 0.1),
        ("연결 실패율 이론과 근접(±0.06)", abs(conn_fail - theory_conn_fail) < 0.06),
    ]
    n_ok = sum(1 for _, ok in checks if ok)
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")

    out = dict(cls=w.class_name, mech="GBPlasticityStpProbSyn", Use=Use, Nrrp=Nrrp,
               n_trial=N_TRIAL, settle_ms=SETTLE_MS,
               theory_rel_prob=round(theory_syn, 3),
               mean_vesicles=round(float(ves.mean()), 3),
               rel_prob=[round(float(x), 3) for x in rel_prob],
               rel_prob_mean=round(float(rel_prob.mean()), 3),
               conn_fail=round(conn_fail, 3), conn_fail_theory=round(theory_conn_fail, 3),
               mean_amp_mV=round(mean_amp, 4), cv=round(cv, 3), rng_ok=rng_ok,
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "3-6_stochastic.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 3-6 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
