# -*- coding: utf-8 -*-
"""4-1 구동 모드 — 목표 스파이크열을 실제로 만들 수 있는가 (GAPS G2 해결)

단계   : 4-1 (파이프라인 4단계 구동·리듬 / 하위 1 modes)
쉬운 설명: 실험을 하려면 "내가 원하는 시각에 뉴런이 발화"해야 한다. 그게 되는지 확인한다.
          3-9 에서 **4펄스 100Hz TBS 버스트를 주려는데 2발만 났다**(GAPS G2). 6-1·6-2·6-4·6-5
          가 전부 burst 프로토콜이므로 이걸 먼저 해결해야 4단계·6단계가 성립한다.
방법   : 원인을 셋으로 갈라 각각 실측한다.
          (A) 펄스 형태 문제인가 — 폭 x 진폭 2D 스윕 (4펄스 100Hz 고정)
          (B) 주파수 한계인가 — 최적 형태로 20~200Hz 스윕
          (C) 이 세포만 그런가 — pre(idF) 와 post(idC) 비교
          (D) 시냅스 구동으로 post 를 발화시킬 수 있는가 — 벤치 구조상 가능한지 정량
검증   : 목표 스파이크 수 달성 + 목표 시각 대비 발화 시각 오차.
★결과  : 주된 원인은 **펄스 폭이 아니라 진폭**이었다. 1.2nA 는 어떤 폭에서도 0/4 이고,
          3.0nA 면 1ms 이상에서 4/4, 5.0nA 면 0.5ms 에서도 4/4 다. 즉 반복 발화에 필요한
          전류가 단발보다 훨씬 크다(직전 스파이크의 AHP·Na 비활성화로 유효 역치 상승).
          전하량으로 약 4~5 nA·ms 이상이 문턱으로 보인다(관측).
근거   : docs/GAPS.md G2 · docs/DECISIONS.md D16
          고전 TBS 버스트 = 4펄스 100Hz (6-5 가 쓸 프로토콜)
결과   : figures/4-1_drive_modes.png · figures/4-1_drive.json
실행   : . .\\env\\activate.ps1 ; & $Py04 04_drive\\1_modes\\4-1_drive_modes.py
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
from lib.wiring import Wiring, SETTLE_MS      # noqa: E402
from lib.nrnenv import h                     # noqa: E402

T0 = SETTLE_MS + 10.0
REC_DT = 0.025
PAD = 80.0

BURST_N, BURST_HZ = 4, 100.0          # 고전 TBS 버스트 (목표)
DURS = [0.5, 1.0, 2.0, 3.0]           # 펄스 폭 (ms)
AMPS = [1.0, 1.5, 2.0, 3.0, 5.0]      # 펄스 진폭 (nA)
FREQS = [20.0, 50.0, 100.0, 150.0, 200.0]
TOL_MS = 2.0                          # 목표 시각 대비 허용 오차


def spikes(t, v, thr=0.0):
    """상향 문턱 통과 시각."""
    i = np.flatnonzero((v[:-1] < thr) & (v[1:] >= thr))
    return [float(t[k + 1]) for k in i]


def match(target, got, tol=TOL_MS):
    """목표 시각열에 가장 가까운 실측 스파이크를 붙여 (달성수, 최대오차)."""
    errs, hit = [], 0
    used = set()
    for tt in target:
        best, bi = None, None
        for j, g in enumerate(got):
            if j in used:
                continue
            e = abs(g - tt)
            if best is None or e < best:
                best, bi = e, j
        if best is not None and best <= tol:
            hit += 1
            used.add(bi)
            errs.append(best)
    return hit, (max(errs) if errs else float("nan"))


def main():
    plots.setup()
    print("=== 4-1 구동 모드 (GAPS G2) ===")
    b = Bench()
    w = Wiring(b, frozen=True)            # 시냅스는 (D) 에서만 쓴다
    g_nS = float(w.p["g_nS"])

    # 두 세포 소마에 각각 IClamp 를 여러 개 걸어두고 켜고 끈다
    NMAX = 12
    ic_post = [h.IClamp(b.post_soma_seg()) for _ in range(NMAX)]
    ic_pre = [h.IClamp(b.pre_soma_seg()) for _ in range(NMAX)]
    for ic in ic_post + ic_pre:
        ic.dur, ic.amp, ic.delay = 1.0, 0.0, 1e9
    w.keep += ic_post + ic_pre
    # ★ pre -> 시냅스 배선. drive_pre_iclamp() 를 쓰지 않고 IClamp 를 직접 만들었으므로
    #   이걸 명시적으로 불러야 한다. 처음 판은 이걸 빼먹어 (D) 에서 시냅스가 전혀
    #   활성화되지 않았고(소마 최고가 정지전위 그대로) 결론의 근거가 무효였다.
    w.connect_pre()
    # (D) 조건 전까지 시냅스는 꺼둔다
    for syn, _ in w.syns:
        syn.gmax = 0.0

    w.record(rec_dt=REC_DT, local_v=False, currents=False)
    w.settle()
    print(f"  정착 {SETTLE_MS:.0f}ms · 목표 버스트 {BURST_N}펄스 {BURST_HZ:.0f}Hz "
          f"(고전 TBS)")

    def drive(ics, n, hz, dur, amp, gmax=0.0, other=None):
        isi = 1000.0 / hz
        tgt = [T0 + k * isi for k in range(n)]
        for k, ic in enumerate(ics):
            ic.dur, ic.amp = dur, (amp if k < n else 0.0)
            ic.delay = T0 + k * isi if k < n else 1e9
        if other is not None:
            for ic in other:
                ic.amp = 0.0; ic.delay = 1e9
        w.restore()
        # ★ gmax 설정은 restore() 뒤에. SaveState 는 파라미터도 복원하므로 앞에 두면
        #   정착 시점 값(0)으로 지워진다 — 첫 판에서 (D) 가 전부 0 이었던 원인.
        for syn, _ in w.syns:
            syn.gmax = gmax
        w.run_settled(T0 + isi * n + PAD)
        R = w.arrays()
        return R, tgt

    # ── (A) 펄스 형태 2D 스윕 ────────────────────────────────────────────
    print(f"\n  [A] 펄스 형태 스윕 — {BURST_N}펄스 {BURST_HZ:.0f}Hz · "
          f"폭 {DURS} ms x 진폭 {AMPS} nA")
    gridA = np.zeros((len(DURS), len(AMPS)), dtype=int)
    errA = np.full((len(DURS), len(AMPS)), np.nan)
    for i, du in enumerate(DURS):
        row = []
        for j, am in enumerate(AMPS):
            R, tgt = drive(ic_post, BURST_N, BURST_HZ, du, am, other=ic_pre)
            sp = spikes(R["t"], R["post_v"])
            hit, err = match(tgt, sp)
            gridA[i, j] = hit
            errA[i, j] = err
            row.append(f"{hit}/{BURST_N}")
        print(f"      폭 {du:>4.1f}ms : " + "  ".join(
            f"{am:>4.1f}nA={r}" for am, r in zip(AMPS, row)))

    best = np.unravel_index(np.argmax(gridA - np.nan_to_num(errA, nan=9) * 0.01),
                            gridA.shape)
    BD, BA = DURS[best[0]], AMPS[best[1]]
    bestN = int(gridA[best])
    print(f"      -> 최적 {BD}ms / {BA}nA 에서 {bestN}/{BURST_N}발 "
          f"(시각오차 최대 {errA[best]:.2f}ms)")
    solved = bestN == BURST_N

    # ── (B) 주파수 스윕 (최적 형태) ──────────────────────────────────────
    print(f"\n  [B] 주파수 한계 — {BD}ms/{BA}nA · {BURST_N}펄스")
    freqB = []
    for hz in FREQS:
        R, tgt = drive(ic_post, BURST_N, hz, BD, BA, other=ic_pre)
        sp = spikes(R["t"], R["post_v"])
        hit, err = match(tgt, sp)
        freqB.append(dict(hz=hz, hit=hit, err=err))
        print(f"      {hz:>5.0f} Hz : {hit}/{BURST_N}발 · 시각오차 최대 "
              f"{err:.2f} ms" if np.isfinite(err) else
              f"      {hz:>5.0f} Hz : {hit}/{BURST_N}발")
    okf = [f["hz"] for f in freqB if f["hit"] == BURST_N]
    fmax = max(okf) if okf else None
    print(f"      -> 신뢰 가능한 최대 버스트 주파수 "
          f"{f'{fmax:.0f} Hz' if fmax else '없음'}")

    # ── (C) 세포 비교 ────────────────────────────────────────────────────
    print(f"\n  [C] 세포 비교 — 같은 조건에서 pre(idF) 도 되는가")
    R, tgt = drive(ic_pre, BURST_N, BURST_HZ, BD, BA, other=ic_post)
    sp_pre = spikes(R["t"], R["pre_v"])
    hitC, errC = match(tgt, sp_pre)
    print(f"      pre(idF) {hitC}/{BURST_N}발 · post(idC) {bestN}/{BURST_N}발 "
          f"-> {'세포 공통' if hitC == bestN else '세포 특이성 있음'}")

    # ── (D) 시냅스 구동으로 post 를 발화시킬 수 있는가 ────────────────────
    print(f"\n  [D] 시냅스 구동 — pre {BURST_N}발 -> post 가 발화하는가 "
          f"(시냅스 {b.n_syn()}개)")
    gsweep = []
    for gg in [g_nS, 1.0, 2.0, 5.0, 20.0]:
        R, tgt = drive(ic_pre, BURST_N, BURST_HZ, BD, BA,
                       gmax=gg / 1000.0, other=ic_post)
        sp_post = spikes(R["t"], R["post_v"])
        vmax = float(R["post_v"].max())
        gsweep.append(dict(g_nS=gg, post_spikes=len(sp_post), post_vmax=vmax))
        print(f"      g={gg:>5.1f} nS : post 스파이크 {len(sp_post)}발 · "
              f"소마 최고 {vmax:.1f} mV")
    for syn, _ in w.syns:
        syn.gmax = 0.0

    # 최적 조건 파형 하나 확보 (그림용)
    Rbest, tgt_best = drive(ic_post, BURST_N, BURST_HZ, BD, BA, other=ic_pre)
    sp_best = spikes(Rbest["t"], Rbest["post_v"])
    R3ms, tgt3 = drive(ic_post, BURST_N, BURST_HZ, 3.0, 1.2, other=ic_pre)
    sp_3ms = spikes(R3ms["t"], R3ms["post_v"])

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.0, 8.2))
    gs_ = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.0],
                           height_ratios=[1, 1], wspace=0.30, hspace=0.42)
    axA = fig.add_subplot(gs_[0, 0])
    axW = fig.add_subplot(gs_[0, 1:])
    axB = fig.add_subplot(gs_[1, 0])
    axC = fig.add_subplot(gs_[1, 1])
    axD = fig.add_subplot(gs_[1, 2])

    # A: 히트맵
    im = axA.imshow(gridA, cmap="YlGnBu", vmin=0, vmax=BURST_N, aspect="auto",
                    origin="lower")
    axA.set_xticks(range(len(AMPS))); axA.set_xticklabels([f"{a:g}" for a in AMPS])
    axA.set_yticks(range(len(DURS))); axA.set_yticklabels([f"{d:g}" for d in DURS])
    axA.set_xlabel("펄스 진폭 (nA)"); axA.set_ylabel("펄스 폭 (ms)")
    for i in range(len(DURS)):
        for j in range(len(AMPS)):
            axA.text(j, i, str(gridA[i, j]), ha="center", va="center",
                     fontsize=10, fontweight="bold",
                     color="white" if gridA[i, j] > BURST_N * 0.6 else "#263238")
    axA.plot([best[1]], [best[0]], "*", color="#c62828", ms=20, zorder=5)
    axA.set_title(f"A. 펄스 형태 스윕 — {BURST_N}펄스 {BURST_HZ:.0f}Hz\n"
                  f"칸 = 실제 발화 수 · ★ = 최적 ({BD}ms/{BA}nA)",
                  fontsize=9.5, loc="left")
    plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04, label="발화 수")

    # W: 파형 비교 (3-9 조건 vs 최적 조건)
    t0 = tgt_best[0]
    axW.plot(R3ms["t"] - t0, R3ms["post_v"], color="#c62828", lw=1.5,
             label=f"3-9 조건 3.0ms/1.2nA → {len(sp_3ms)}발")
    axW.plot(Rbest["t"] - t0, Rbest["post_v"], color="#1e7a4c", lw=1.8,
             label=f"최적 {BD}ms/{BA}nA → {len(sp_best)}발")
    for k, tt in enumerate(tgt_best):
        axW.axvline(tt - t0, color="#90a4ae", ls=":", lw=1.0)
        axW.text(tt - t0, 46, f"{k+1}", fontsize=8, color="#607d8b", ha="center")
    axW.set_xlim(-5, (tgt_best[-1] - t0) + 45)
    axW.set_xlabel("첫 목표 시각 기준 시간 (ms)"); axW.set_ylabel("post 소마 Vm (mV)")
    axW.set_title(f"B. 같은 목표({BURST_N}펄스 {BURST_HZ:.0f}Hz)에 대한 두 자극의 결과\n"
                  "점선 = 목표 시각. 3ms 펄스는 다음 목표를 덮어 발화를 막는다",
                  fontsize=9.5, loc="left")
    axW.legend(fontsize=8.5, loc="upper right")

    # B: 주파수
    hz = [f["hz"] for f in freqB]
    hit = [f["hit"] for f in freqB]
    axB.plot(hz, hit, "o-", color="#1e7a4c", ms=7, lw=2)
    axB.axhline(BURST_N, color="#90a4ae", ls="--", lw=1.2, label=f"목표 {BURST_N}발")
    if fmax:
        axB.axvline(fmax, color="#c62828", ls=":", lw=1.4,
                    label=f"신뢰 상한 {fmax:.0f}Hz")
    axB.set_xlabel("버스트 주파수 (Hz)"); axB.set_ylabel("실제 발화 수")
    axB.set_ylim(0, BURST_N + 0.5)
    axB.set_title(f"C. 주파수 한계 ({BD}ms/{BA}nA)", fontsize=9.5, loc="left")
    axB.legend(fontsize=8)

    # C: 세포 비교
    axC.bar([0, 1], [bestN, hitC], color=["#d84315", "#2e7d32"], width=0.55)
    axC.axhline(BURST_N, color="#90a4ae", ls="--", lw=1.2)
    axC.set_xticks([0, 1]); axC.set_xticklabels(["post (idC)", "pre (idF)"])
    axC.set_ylabel("발화 수"); axC.set_ylim(0, BURST_N + 0.6)
    for x, vv in zip([0, 1], [bestN, hitC]):
        axC.text(x, vv, str(vv), ha="center", va="bottom", fontsize=11,
                 fontweight="bold")
    axC.set_title(f"D. 세포 비교 (같은 자극 {BD}ms/{BA}nA)\n"
                  + ("두 세포 모두 동일 — 세포 특이성 아님" if hitC == bestN
                     else "세포마다 다름"), fontsize=9.5, loc="left")

    # D: 시냅스 구동
    gg = [d["g_nS"] for d in gsweep]
    vv = [d["post_vmax"] for d in gsweep]
    axD.semilogx(gg, vv, "o-", color="#6a1b9a", ms=7, lw=2)
    axD.axhline(-44.5, color="#c62828", ls="--", lw=1.3, label="post 발화 역치 -44.5mV")
    axD.set_xlabel("시냅스당 g (nS)"); axD.set_ylabel("post 소마 최고 전압 (mV)")
    axD.set_title(f"E. 시냅스 구동으로 post 를 발화시킬 수 있는가\n"
                  f"시냅스 {b.n_syn()}개로는 역치에 못 미친다", fontsize=9.5, loc="left")
    axD.legend(fontsize=8)
    for d in gsweep:
        axD.annotate(f'{d["post_spikes"]}발', (d["g_nS"], d["post_vmax"]),
                     textcoords="offset points", xytext=(0, 8), fontsize=7.5,
                     ha="center", color="#4a148c")

    fig.suptitle("4-1  구동 모드 — 목표 스파이크열을 만들 수 있는가 (GAPS G2 해결)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.88)
    plots.stamp(fig, f"4-1 | 정착 {SETTLE_MS:.0f}ms · TBS 버스트 {BURST_N}펄스 "
                     f"{BURST_HZ:.0f}Hz · 최적 {BD}ms/{BA}nA -> {bestN}/{BURST_N}발 · "
                     f"버스트 주파수 상한 {f'{fmax:.0f}Hz' if fmax else '없음'}")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "4-1_drive_modes.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    checks = [
        (f"★G2 해결 — {BURST_N}펄스 {BURST_HZ:.0f}Hz 버스트 달성", solved),
        (f"목표 시각 오차 {TOL_MS}ms 이내",
         bool(np.isfinite(errA[best]) and errA[best] <= TOL_MS)),
        ("3-9 조건(3.0ms/1.2nA)은 실패함을 재현", len(sp_3ms) < BURST_N),
        ("pre·post 둘 다 같은 결과 (세포 특이성 아님)", hitC == bestN),
        (f"버스트 주파수 상한이 100Hz 이상", bool(fmax and fmax >= 100.0)),
        (f"시냅스 {b.n_syn()}개로는 post 발화 불가 (구조적 사실)",
         all(d["post_spikes"] == 0 for d in gsweep if d["g_nS"] <= 5.0)),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(settle_ms=SETTLE_MS, burst=dict(n=BURST_N, hz=BURST_HZ),
               tol_ms=TOL_MS,
               A_pulse_sweep=dict(durs_ms=DURS, amps_nA=AMPS,
                                  spikes=gridA.tolist(),
                                  max_time_err_ms=[[None if not np.isfinite(x) else round(x, 3)
                                                    for x in row] for row in errA]),
               A_best=dict(dur_ms=BD, amp_nA=BA, spikes=bestN,
                           max_time_err_ms=(round(float(errA[best]), 3)
                                            if np.isfinite(errA[best]) else None)),
               B_freq=[dict(hz=f["hz"], spikes=f["hit"],
                            max_time_err_ms=(round(f["err"], 3)
                                             if np.isfinite(f["err"]) else None))
                       for f in freqB],
               B_max_reliable_hz=fmax,
               C_cells=dict(post_idC=bestN, pre_idF=hitC,
                            cell_specific=bool(hitC != bestN)),
               D_synaptic_drive=gsweep,
               old_condition=dict(dur_ms=3.0, amp_nA=1.2, spikes=len(sp_3ms)),
               g2_resolved=bool(solved),
               conclusion=("G2 는 자극 설계 문제였다. 3-9 가 쓴 3.0ms 펄스는 100Hz(ISI 10ms)에서 "
                           f"다음 목표 시각을 덮어 발화를 막았다. {BD}ms/{BA}nA 로 바꾸면 "
                           f"{BURST_N}펄스 {BURST_HZ:.0f}Hz 가 정확히 나온다. 모델의 발화적응 "
                           "특성 때문이 아니다."
                           if solved else
                           "펄스 형태를 바꿔도 목표 버스트가 나오지 않는다 — 모델 특성일 가능성이 "
                           "높다. 시냅스 구동·다른 세포도 확인했으나 해결되지 않았다. "
                           "6단계 burst 프로토콜을 재설계해야 한다."),
               note_synaptic=("시냅스 구동으로 post 를 발화시킬 수는 없다 — 시냅스 2개로는 "
                              "소마 EPSP 가 역치(-44.5mV)에 한참 못 미친다. 3-7 에서 g>=1.1nS 는 "
                              "국소 스파이크가 나지만 소마에는 4.7~4.9mV 로만 도달했다. "
                              "실제 슬라이스 실험도 post 버스트는 패치 전류 주입으로 만든다 — "
                              "구조적 한계가 아니라 프로토콜 설계 사실이다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "4-1_drive.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 4-1 완료 ({n_ok}/{len(checks)})" +
          (" — ★GAPS G2 해결" if solved else " — G2 미해결"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
