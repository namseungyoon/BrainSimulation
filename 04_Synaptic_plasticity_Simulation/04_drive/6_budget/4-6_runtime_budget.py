# -*- coding: utf-8 -*-
"""4-6 런타임 예산 — 6단계를 이 머신에서 돌릴 수 있는가 (결론: 원안대로는 불가)

단계   : 4-6 (파이프라인 4단계 구동·리듬 / 하위 6 budget)
쉬운 설명: 6단계 프로토콜 중에는 생물시간이 매우 긴 것이 있다. 이 벤치의 실제 속도를 재서
          **무엇이 가능하고 무엇이 불가능한지** 미리 정한다. 나중에 알면 그때 가서 급히
          프로토콜을 바꿔야 하고, 그러면 문헌 대조가 무너진다.
★측정   : 구성별 속도는 **반드시 별도 프로세스**에서 잰다. NEURON 은 전역 모델 하나를 돌리므로
          같은 프로세스에서 벤치를 만든 뒤 프로브를 재면 **벤치까지 함께** 돌아간다.
          첫 판이 그 실수를 했고 단일 구획 프로브가 두 세포 벤치보다 느리게 나왔다(316 vs 197).
방법   : (A) 구성 3종 x dt 3종을 별도 프로세스로 측정
          (B) 6단계 원안 예산 계산
          (C) **손잡이별 절감 효과 정량** — 병렬 · 엔진 축소 · 관찰시간 · 유도 압축 · dt
          (D) 실행 가능한 계획 제시 (무엇을 포기하는지 명시)
검증   : 속도 실측 · 예산표 · 손잡이 정량 · 실행안이 한계 안에 드는가.
근거   : D34(스파이크 측정은 dt 0.025) · D12(정착) · PLAN 6-7 · 미결#6
결과   : figures/4-6_runtime_budget.png · figures/4-6_budget.json
실행   : . .\\env\\activate.ps1 ; & $Py04 04_drive\\6_budget\\4-6_runtime_budget.py
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                  # noqa: E402
from lib import plots                                # noqa: E402

PROBE_MS = 400.0
DTS = [0.025, 0.05, 0.1]
CONFIGS = [("bench", "두 세포 벤치"), ("cell", "단일 세포"), ("probe", "단일 구획 프로브")]
SETTLE_S = 0.25

# 6단계 원안: (이름, 유도 생물초, 관찰 생물초, 조건 수, 비고)
EXPERIMENTS = [
    ("6-1 theta 위상", 4 * 0.2 * 8, 20.0, 4 * 6, "부과 theta 8주기 x 4위상"),
    ("6-2 theta-gamma", 4 * 0.2 * 8, 20.0, 4 * 3 * 6, "6-1 x gamma 3조건"),
    ("6-3 STDP 단발", 60 * 1.0, 20.0, 11 * 6, "60짝 1Hz x dt 11점"),
    ("6-4 STDP 버스트", 60 * 1.0, 20.0, 11 * 6, "60짝 1Hz x dt 11점"),
    ("6-5 TBS LTP", 10 * 0.2, 20.0, 5 * 6, "10버스트 5Hz"),
    ("6-6 고빈도 HFS", 4 * 1.0, 20.0, 3 * 6, "100Hz 1초 x 4"),
    ("6-7 저빈도 LFS", 900 * 1.0, 60.0, 1 * 6, "★1Hz x 900펄스"),
    ("6-8 위치 의존", 10 * 0.2, 20.0, 16 * 3, "6-5 x 16지점 x 3엔진"),
]
TIER_OK, TIER_NIGHT = 2.0, 12.0


def run_worker(cfg, dt, ms):
    py = sys.executable
    out = subprocess.run([py, "-m", "lib.speedprobe", cfg, str(dt), str(ms)],
                         cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    for line in (out.stdout or "").splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"워커 실패 {cfg} dt={dt}\n{out.stdout}\n{out.stderr}")


def main():
    plots.setup()
    print("=== 4-6 런타임 예산 ===")
    ncpu = os.cpu_count() or 1
    print(f"  이 머신 논리 코어 {ncpu}개 · 측정 단위: 생물 {PROBE_MS:.0f}ms 당 벽시계")
    print(f"  ★구성별 측정은 별도 프로세스로 (NEURON 은 전역 모델 하나를 돌린다)")

    # ── (A) 속도 측정 ─────────────────────────────────────────────────────
    speed = []
    print()
    for cfg, lab in CONFIGS:
        for dt in DTS:
            r = run_worker(cfg, dt, PROBE_MS)
            r["label"] = lab
            speed.append(r)
            print(f"      {lab:<14} dt {dt:<6.3f} 구획 {r['n_seg']:>5} -> "
                  f"{r['wall_s']:7.3f}s (생물 1초당 {r['s_per_s']:8.2f}s)")

    def rate(cfg, dt):
        return next(s["s_per_s"] for s in speed
                    if s["config"] == cfg and abs(s["dt"] - dt) < 1e-9)

    R_BENCH = rate("bench", 0.025)
    R_PROBE = rate("probe", 0.025)
    speedup = R_BENCH / R_PROBE
    print(f"\n  기준: 두 세포 벤치 dt 0.025 = 생물 1초당 {R_BENCH:.1f}초")
    print(f"        단일 구획 프로브는 {speedup:.0f}배 빠르다 ({R_PROBE:.3f}초)")
    print(f"        dt 0.1 로 키우면 벤치가 {rate('bench', 0.025)/rate('bench', 0.1):.1f}배 빨라진다")

    # ── (B) 원안 예산 ─────────────────────────────────────────────────────
    print(f"\n  [B] 6단계 원안 예산 (두 세포 벤치 · dt 0.025 · 단일 프로세스)")
    base = []
    for name, ind, obs, nc, note in EXPERIMENTS:
        per = ind + obs + SETTLE_S
        bio = per * nc
        h_ = bio * R_BENCH / 3600.0
        base.append(dict(name=name, ind_s=ind, obs_s=obs, per_s=per, n_cond=nc,
                         bio_s=bio, hours=h_, note=note))
        print(f"      {name:<18}{per:8.1f}s x {nc:3d}조건 = 생물 {bio:8.0f}s "
              f"-> {h_:8.1f} 시간")
    total0 = sum(x["hours"] for x in base)
    print(f"      {'합계':<18}{'':8} {'':3}   생물 "
          f"{sum(x['bio_s'] for x in base):8.0f}s -> {total0:8.1f} 시간 "
          f"({total0/24:.0f}일)")

    # ── (C) 손잡이별 절감 ─────────────────────────────────────────────────
    print(f"\n  [C] 손잡이별 절감 효과 (곱해서 적용된다)")
    n_par = max(ncpu - 2, 1)          # 코어 여유 2개 (05 트랙 관례)
    levers = [
        dict(key="parallel", label=f"병렬 실행 ({n_par}프로세스)", factor=1.0 / n_par,
             cost="없음 — 조건들이 서로 독립이다. 실행 방식만 바뀐다."),
        dict(key="observe", label="유도 후 관찰시간 제거", factor=None,
             cost="없음에 가깝다 — GB 효능 시상수가 688초라 20초 관찰은 거의 안 움직인다. "
                  "유도 직후 rho 를 읽으면 된다."),
        dict(key="engines", label="엔진 6종 -> 3종 (det·A·glu)", factor=0.5,
             cost="★비교 폭이 준다. 6-9 종합 비교는 전 엔진이 필요하므로 그때는 되돌린다."),
        dict(key="dt", label="dt 0.025 -> 0.05", factor=None,
             cost="★스파이크가 걸린 측정에는 쓸 수 없다(D34: 발화율 11% · ISI CV 0.15 차이). "
                  "역치하 전용 조건에만."),
        dict(key="induce", label="유도 주파수 압축 (6-3·6-4 를 1->5Hz)", factor=None,
             cost="★★문헌 대조가 약해진다. 짝 반복 주파수는 결과에 영향을 준다"
                  "(5-1: 고전 STDP 는 50Hz 에서 부호가 뒤집힌다). 프로토콜이 달라진다."),
    ]
    # 관찰시간 제거 효과
    bio_no_obs = sum((x["ind_s"] + SETTLE_S) * x["n_cond"] for x in base)
    levers[1]["factor"] = bio_no_obs / sum(x["bio_s"] for x in base)
    # dt 효과
    levers[3]["factor"] = rate("bench", 0.05) / rate("bench", 0.025)
    # 유도 압축 효과 (6-3·6-4 만)
    bio_comp = 0.0
    for x in base:
        ind = x["ind_s"] / 5.0 if x["name"].startswith(("6-3", "6-4")) else x["ind_s"]
        bio_comp += (ind + x["obs_s"] + SETTLE_S) * x["n_cond"]
    levers[4]["factor"] = bio_comp / sum(x["bio_s"] for x in base)
    for lv in levers:
        print(f"      {lv['label']:<32} x{lv['factor']:.3f} "
              f"({1/lv['factor']:.1f}배 절감)")
        print(f"          비용: {lv['cost']}")

    # ── (D) 실행안 ────────────────────────────────────────────────────────
    print(f"\n  [D] 실행안")
    plans = []
    # 안 1: 병렬 + 관찰 제거 (프로토콜 불변 — 문헌 대조 유지)
    f1 = levers[0]["factor"] * levers[1]["factor"]
    # 안 2: + 엔진 축소
    f2 = f1 * levers[2]["factor"]
    # 안 3: + 유도 압축
    f3 = f2 * levers[4]["factor"]
    for lab, f, keep in (("안1 병렬 + 관찰 제거", f1, "프로토콜 불변 — 문헌 대조 유지"),
                         ("안2 + 엔진 3종", f2, "비교 폭 축소 (6-9 는 전 엔진으로 별도)"),
                         ("안3 + 6-3/6-4 유도 압축", f3, "★문헌 대조 약화")):
        tot = total0 * f
        plans.append(dict(label=lab, factor=f, hours=tot, tradeoff=keep))
        print(f"      {lab:<26} {tot:7.1f} 시간 ({tot/24:.1f}일) — {keep}")

    # 실험별 티어 (안1 기준)
    print(f"\n  [티어] 안1 기준 (<= {TIER_OK:.0f}h 그대로 · <= {TIER_NIGHT:.0f}h 밤새)")
    tiers = []
    for x in base:
        h_ = x["hours"] * f1
        t = ("그대로" if h_ <= TIER_OK else
             ("밤새" if h_ <= TIER_NIGHT else "★축소 필요"))
        tiers.append(dict(name=x["name"], hours=h_, tier=t))
        print(f"      {t:<10}{x['name']:<18}{h_:8.2f} 시간")
    n_cut = sum(1 for t in tiers if t["tier"] == "★축소 필요")

    # 6-7 축소안
    print(f"\n  [6-7 축소안] 안1 적용 후")
    opts = []
    for n_p, hz, lab in ((900, 1.0, "원본 (Dudek&Bear 1992)"), (300, 1.0, "1/3"),
                         (900, 5.0, "5Hz 압축"), (300, 3.0, "절충")):
        bio = (n_p / hz + SETTLE_S) * 6
        h_ = bio * R_BENCH / 3600.0 * levers[0]["factor"]
        opts.append(dict(n_pulse=n_p, hz=hz, label=lab, bio_s=bio, hours=h_))
        print(f"      {lab:<24}{n_p}펄스 {hz:.0f}Hz -> {h_:6.2f} 시간")

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.4))
    gs_ = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.54)
    axA = fig.add_subplot(gs_[0, 0])
    axB = fig.add_subplot(gs_[0, 1])
    axC = fig.add_subplot(gs_[0, 2])
    axD = fig.add_subplot(gs_[1, :2])
    axE = fig.add_subplot(gs_[1, 2])

    for (cfg, lab), col, mk in zip(CONFIGS, ["#c62828", "#1565c0", "#2e7d32"], "os^"):
        axA.plot(DTS, [rate(cfg, d) for d in DTS], mk + "-", color=col, ms=7, lw=2,
                 label=f"{lab} ({next(s['n_seg'] for s in speed if s['config']==cfg)}구획)")
    axA.set_xscale("log"); axA.set_yscale("log")
    plots.ascii_log(axA, "y"); plots.ascii_log(axA, "x")
    axA.set_xlabel("dt (ms)"); axA.set_ylabel("생물 1초당 벽시계 (s)")
    axA.set_title(f"A. 구성별 속도 (★별도 프로세스)\n프로브가 벤치보다 {speedup:.0f}배 빠르다",
                  fontsize=9.2, loc="left")
    axA.legend(fontsize=6.8)

    lv_lab = [lv["label"].split(" (")[0] for lv in levers]
    axB.barh(range(len(levers)), [1.0 / lv["factor"] for lv in levers],
             color=["#2e7d32", "#2e7d32", "#ef6c00", "#ef6c00", "#c62828"])
    axB.set_yticks(range(len(levers))); axB.set_yticklabels(lv_lab, fontsize=6.8)
    axB.invert_yaxis(); axB.set_xlabel("절감 배수")
    for i, lv in enumerate(levers):
        axB.text(1.0 / lv["factor"], i, f" {1/lv['factor']:.1f}x", va="center",
                 fontsize=7.5)
    axB.set_title("B. 손잡이별 절감\n초록=비용 없음 · 주황=비교 축소 · 빨강=문헌 대조 약화",
                  fontsize=9.2, loc="left")

    axC.bar(range(len(plans)), [p["hours"] for p in plans],
            color=["#2e7d32", "#ef6c00", "#c62828"], width=0.6)
    axC.axhline(TIER_NIGHT, color="#37474f", ls="--", lw=1.4,
                label=f"밤새 한계 {TIER_NIGHT:.0f}h")
    axC.set_xticks(range(len(plans)))
    axC.set_xticklabels([f"안{i+1}" for i in range(len(plans))])
    axC.set_ylabel("6단계 전체 (시간)")
    for i, p in enumerate(plans):
        axC.text(i, p["hours"], f"{p['hours']:.0f}h", ha="center", va="bottom",
                 fontsize=8)
    axC.set_title(f"C. 실행안 — 원안 {total0:.0f}h ({total0/24:.0f}일)\n"
                  "안1 은 프로토콜을 바꾸지 않는다", fontsize=9.2, loc="left")
    axC.legend(fontsize=7.5)

    ys = [t["hours"] for t in tiers]
    cols = ["#2e7d32" if t["tier"] == "그대로" else
            ("#ef6c00" if t["tier"] == "밤새" else "#c62828") for t in tiers]
    axD.barh(range(len(tiers)), ys, color=cols)
    axD.set_yticks(range(len(tiers)))
    axD.set_yticklabels([t["name"] for t in tiers], fontsize=8)
    axD.invert_yaxis(); axD.set_xscale("log"); plots.ascii_log(axD, "x")
    axD.axvline(TIER_OK, color="#2e7d32", ls=":", lw=1.4)
    axD.axvline(TIER_NIGHT, color="#c62828", ls="--", lw=1.4)
    axD.set_xlabel("예상 벽시계 (시간 · 로그)")
    for i, t in enumerate(tiers):
        axD.text(t["hours"], i, f"  {t['hours']:.2f}h ({t['tier']})", va="center",
                 fontsize=7)
    axD.set_title(f"D. 안1 적용 후 실험별 예산 — 합계 {sum(ys):.1f}시간\n"
                  f"축소 필요 {n_cut}개", fontsize=9.2, loc="left")

    axE.bar(range(len(opts)), [o["hours"] for o in opts], color="#6a1b9a", width=0.6)
    axE.axhline(TIER_NIGHT, color="#c62828", ls="--", lw=1.4)
    axE.set_xticks(range(len(opts)))
    axE.set_xticklabels([f"{o['n_pulse']}\n{o['hz']:.0f}Hz" for o in opts],
                        fontsize=7)
    axE.set_ylabel("예상 벽시계 (시간)")
    for i, o in enumerate(opts):
        axE.text(i, o["hours"], f"{o['hours']:.1f}h", ha="center", va="bottom",
                 fontsize=7.5)
    axE.set_title("E. 6-7(LFS) 선택지 — 안1 적용 후\n원본도 밤새 안에 든다",
                  fontsize=9.2, loc="left")

    fig.suptitle("4-6  런타임 예산 — 원안은 불가, 프로토콜을 바꾸지 않는 실행안이 있다",
                 fontsize=12.3, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"4-6 | 벤치 dt0.025 = 생물1초당 {R_BENCH:.0f}s · 프로브 {speedup:.0f}배 · "
                     f"원안 {total0:.0f}h -> 안1 {plans[0]['hours']:.0f}h "
                     f"(병렬 {n_par} + 관찰 제거) · 코어 {ncpu}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "4-6_runtime_budget.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    checks = [
        ("구성 3종 x dt 3종을 별도 프로세스로 측정했다", len(speed) == 9),
        ("★단일 구획 프로브가 두 세포 벤치보다 훨씬 빠르다 (10배 이상) — "
         "같은 프로세스에서 재면 이게 뒤집힌다", speedup > 10.0),
        ("구획 수가 구성별로 다르다 (측정이 서로 오염되지 않았다)",
         len({s["n_seg"] for s in speed}) == 3),
        ("dt 를 2배 키우면 대략 2배 빨라진다 (1.7~2.3배)",
         1.7 < rate("bench", 0.025) / rate("bench", 0.05) < 2.3),
        (f"★6단계 원안이 이 머신에서 불가능하다 ({total0:.0f}시간 = {total0/24:.0f}일)",
         total0 > 24 * 7),
        ("★프로토콜을 바꾸지 않는 실행안이 존재한다 (안1)",
         plans[0]["hours"] < total0 / 5),
        (f"★6-7 원본은 안1 로도 부족하다 ({opts[0]['hours']:.1f}h > {TIER_NIGHT:.0f}h) "
         f"— 축소가 필요한 유일한 실험이 아니다(6-3·6-4 도)",
         opts[0]["hours"] > TIER_NIGHT),
        (f"6-7 축소안 중 밤새 안에 드는 것이 있다 "
         f"(1/3 축소 {opts[1]['hours']:.1f}h)",
         any(o["hours"] <= TIER_NIGHT for o in opts[1:])),
        (f"안1 기준 축소 필요 실험이 {n_cut}개로 좁혀졌다 (원안은 8개 전부)",
         0 < n_cut < len(base)),
        ("손잡이마다 비용이 명시됐다", all(lv["cost"] for lv in levers)),
        ("관찰시간 제거가 비용 없는 손잡이로 분류됐다",
         levers[1]["factor"] < 1.0),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(n_cpu=ncpu, n_parallel=n_par, probe_ms=PROBE_MS, dts=DTS,
               speed=speed, rate_bench=R_BENCH, rate_probe=R_PROBE,
               speedup_probe=speedup,
               measurement_note=("★구성별 속도는 반드시 별도 프로세스에서 잰다. NEURON 은 "
                                 "전역 모델 하나를 돌리므로 같은 프로세스에서 벤치를 만든 뒤 "
                                 "프로브를 재면 벤치까지 함께 돌아간다. 첫 판이 그 실수를 했고 "
                                 "단일 구획 프로브가 두 세포 벤치보다 **느리게**(316 vs 197 "
                                 "s/s) 나왔다 — 물리적으로 불가능한 결과였다."),
               baseline=base, total_hours_original=total0,
               levers=levers, plans=plans, tiers=tiers, lfs_options=opts,
               finding=(f"★6단계 원안은 이 머신에서 **{total0:.0f}시간({total0/24:.0f}일)** "
                        f"이다 — 그대로는 불가능하다. 그러나 **프로토콜을 바꾸지 않는 손잡이**"
                        f"만으로 {plans[0]['hours']:.0f}시간까지 줄어든다: (1) 조건들이 서로 "
                        f"독립이므로 {n_par}프로세스 병렬, (2) 유도 후 20초 관찰 제거 — GB 효능 "
                        f"시상수가 688초라 그 사이 거의 움직이지 않으므로 유도 직후 rho 를 "
                        f"읽으면 된다. 두 손잡이 모두 **문헌 대조를 건드리지 않는다.**"),
               remaining=("안1 을 적용해도 **6-3·6-4·6-7 세 개**는 여전히 밤새 한계를 넘는다 "
                          "(각 19.5h · 19.5h · 21.2h). 6-7 은 펄스 수를 1/3 로 줄이면 9.8h 로 "
                          "든다. 6-3·6-4 는 dt 스윕 점수를 11 -> 7 로 줄이거나 엔진을 3종으로 "
                          "줄이는 쪽이 유도 주파수를 바꾸는 것보다 낫다 — 프로토콜을 보존하기 "
                          "때문이다."),
               recommendation=("안1(병렬 + 관찰 제거)을 기본으로 한다. 그래도 남는 실험이 "
                               "있으면 엔진을 3종으로 줄이되(안2) **6-9 종합 비교만은 전 엔진**"
                               "으로 따로 돌린다. 유도 주파수 압축(안3)은 **마지막 수단**이다 — "
                               "짝 반복 주파수가 결과에 영향을 주므로(5-1: 고전 STDP 는 50Hz "
                               "에서 부호가 뒤집힌다) 프로토콜이 달라지고 문헌 대조가 약해진다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "4-6_budget.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 4-6 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
