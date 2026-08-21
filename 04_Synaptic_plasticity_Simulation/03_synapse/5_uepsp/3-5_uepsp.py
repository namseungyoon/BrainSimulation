# -*- coding: utf-8 -*-
"""3-5 단발 uEPSP — pre 1발에 대한 연결 단위 EPSP 특성 (진폭·지연·상승·감쇠)

단계   : 3-5 (파이프라인 3단계 시냅스 / 하위 5 uepsp)
쉬운 설명: pre 가 한 번 발화하면 post 소마에 작은 흥분성 전위(EPSP)가 생긴다. 그 크기·모양을
          잰다. 두 세포의 '연결 1개'는 시냅스 접촉의 합이므로, 여기 EPSP 는 연결 단위(uEPSP).
방법   : lib.wiring(동결 전달) 로 pre 1발 발화 → post 소마 EPSP 파형에서 진폭·개시지연·상승시간
          (20→80%)·반치폭·감쇠(1/e)를 측정(lib.measure). 문헌 범위와 대조.
근거   : ★Deuchars & Thomson 1996 Neuroscience 74:1009 (PMID 8895869) — CA1 PC->PC 쌍 실측.
         989쌍 중 연결 9개. 진폭 0.7±0.5mV(0.17~1.5) · 10-90% 상승 2.7±0.9ms · 반치폭 16.8±4.1ms.
         ⚠️ 이전 판은 Sayer1990(CA3→CA1 = Schaffer collateral)과 비교해 '평균의 10.2배'라
            판정했으나, 그것은 이 벤치의 연결이 아니다(D9). 기준 문헌을 교체했다.
결과   : figures/3-5_uepsp_trace.png · figures/3-5_uepsp_stats.png · figures/3-5_uepsp.json
실행   : . .\\env\\activate.ps1 ; & $Py04 03_synapse\\5_uepsp\\3-5_uepsp.py
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
from lib import refdata                       # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.wiring import Wiring                 # noqa: E402

from lib.wiring import SETTLE_MS   # noqa: E402  (정착 후 자극)
T_SPIKE = SETTLE_MS + 10.0
TSTOP = T_SPIKE + 70.0
S = refdata.DEUCHARS1996       # 이 벤치(CA1 PC->PC)의 기준 실측 (PubMed 확인)


def main():
    plots.setup()
    print("=== 3-5 단발 uEPSP ===")
    b = Bench()
    w = Wiring(b, frozen=True)
    w.drive_pre_iclamp([T_SPIKE], amp_nA=1.2, dur_ms=3.0)
    w.record(rec_dt=0.025, local_v=False, currents=False)
    w.run(TSTOP)
    R = w.arrays()
    t, v = R["t"], R["post_v"]

    f = measure.epsp_features(t, v, T_SPIKE + b.syn_specs[0]["delay_ms"])
    print(f"  진폭 {f['amp_mV']:.3f} mV · 개시지연 {f['latency_ms']:.2f} ms · "
          f"상승 {f['rise_ms']:.2f} ms · 반치폭 {f['halfwidth_ms']:.2f} ms · 감쇠 {f['decay_ms']:.2f} ms")

    # 논문(Deuchars1996) 대조: 진폭·상승·반치폭 (평균±SD, 범위)
    amp_ratio = f["amp_mV"] / S["amp_mV"]["mean"]
    amp_vs_max = f["amp_mV"] / S["amp_mV"]["max"]
    print(f"  [Deuchars1996] 진폭 우리 {f['amp_mV']:.3f} vs 평균 {S['amp_mV']['mean']:.3f} "
          f"(최대 {S['amp_mV']['max']:.3f}) mV -> 평균의 {amp_ratio:.1f}배·최대의 {amp_vs_max:.1f}배")
    print(f"              상승 우리 {f['rise_ms']:.2f} vs {S['rise_ms']['mean']}±{S['rise_ms']['sd']} ms")
    print(f"              반치폭 우리 {f['halfwidth_ms']:.2f} vs {S['halfwidth_ms']['mean']}±{S['halfwidth_ms']['sd']} ms")

    import matplotlib.pyplot as plt
    base = v[(t >= T_SPIKE - 5) & (t < T_SPIKE)].mean()

    # ---- 그림 1: EPSP 파형 (trace) ----
    figT, axT = plt.subplots(figsize=(8.2, 5.0))
    axT.plot(t, v, color="#d84315", lw=1.9)
    axT.axhline(base, ls=":", color="#999", lw=0.8)
    vpk = base + f["amp_mV"]
    axT.annotate(f"진폭 {f['amp_mV']:.3f} mV", xy=(f["t_peak_ms"], vpk),
                 xytext=(f["t_peak_ms"]+8, vpk), fontsize=10, color="#b71c1c",
                 arrowprops=dict(arrowstyle="->", color="#b71c1c"))
    axT.axvline(T_SPIKE, ls="--", color="#2e7d32", lw=1, alpha=0.7)
    axT.text(T_SPIKE, axT.get_ylim()[0], " pre 발화", fontsize=8, color="#2e7d32", va="bottom")
    axT.set_xlim(T_SPIKE-5, T_SPIKE+50)
    axT.set_xlabel("시간 (ms)"); axT.set_ylabel("post 소마 Vm (mV)")
    axT.set_title(f"3-5  단발 uEPSP 파형 (연결=시냅스 {b.n_syn()}개 합)\n"
                  f"상승 {f['rise_ms']:.2f}ms · 반치폭 {f['halfwidth_ms']:.2f}ms · 감쇠 {f['decay_ms']:.2f}ms",
                  fontsize=11, loc="left")
    plots.stamp(figT, f"3-5 | {w.class_name} · 동결 전달 · 진폭 {f['amp_mV']:.3f}mV")
    outdir = plots.figdir(__file__)
    plots.save(figT, outdir, "3-5_uepsp_trace.png")

    # ---- 그림 2: 논문(Deuchars1996) 대조 (stats) ----
    figS, (b1, b2, b3) = plt.subplots(1, 3, figsize=(12.5, 4.6))
    # 진폭: 우리 vs Deuchars 범위(min~max)+평균
    b1.bar([0], [f["amp_mV"]], color="#d84315", width=0.5, label="우리")
    b1.bar([1], [S["amp_mV"]["mean"]], color="#546e7a", width=0.5, label="Deuchars 평균")
    b1.errorbar([1], [S["amp_mV"]["mean"]],
                yerr=[[S["amp_mV"]["mean"]-S["amp_mV"]["min"]], [S["amp_mV"]["max"]-S["amp_mV"]["mean"]]],
                fmt="none", ecolor="#263238", capsize=6, lw=1.5)
    b1.axhspan(S["amp_mV"]["min"], S["amp_mV"]["max"], xmin=0.55, xmax=0.95,
               color="#ffb300", alpha=0.18)
    b1.set_xticks([0, 1]); b1.set_xticklabels(["우리", "Deuchars1996\n(0.17~1.5 mV)"], fontsize=8.5)
    b1.set_ylabel("진폭 (mV)"); b1.set_title(f"진폭 — 평균의 {amp_ratio:.1f}배", fontsize=10, loc="left")
    b1.text(0, f["amp_mV"], f"{f['amp_mV']:.3f}", ha="center", va="bottom", fontsize=9)
    b1.text(1, S["amp_mV"]["mean"], f"{S['amp_mV']['mean']:.3f}", ha="center", va="bottom", fontsize=9)

    # 상승시간
    b2.bar([0], [f["rise_ms"]], color="#d84315", width=0.5)
    b2.bar([1], [S["rise_ms"]["mean"]], color="#546e7a", width=0.5)
    b2.errorbar([1], [S["rise_ms"]["mean"]], yerr=[S["rise_ms"]["sd"]], fmt="none",
                ecolor="#263238", capsize=6, lw=1.5)
    b2.set_xticks([0, 1]); b2.set_xticklabels(["우리\n(20-80%)", "Deuchars\n2.7±0.9 (10-90%)"], fontsize=8.5)
    b2.set_ylabel("상승시간 (ms)")
    b2.set_title("상승시간 — 우리 20-80% vs 논문 10-90%\n(기준이 달라 우리 값이 체계적으로 작다)",
                 fontsize=9.5, loc="left")
    b2.text(0, f["rise_ms"], f"{f['rise_ms']:.2f}", ha="center", va="bottom", fontsize=9)

    # 반치폭
    b3.bar([0], [f["halfwidth_ms"]], color="#d84315", width=0.5)
    b3.bar([1], [S["halfwidth_ms"]["mean"]], color="#546e7a", width=0.5)
    b3.errorbar([1], [S["halfwidth_ms"]["mean"]], yerr=[S["halfwidth_ms"]["sd"]], fmt="none",
                ecolor="#263238", capsize=6, lw=1.5)
    b3.set_xticks([0, 1]); b3.set_xticklabels(["우리", "Deuchars\n16.8±4.1"], fontsize=8.5)
    b3.set_ylabel("반치폭 (ms)"); b3.set_title("반치폭", fontsize=10, loc="left")
    b3.text(0, f["halfwidth_ms"], f"{f['halfwidth_ms']:.2f}", ha="center", va="bottom", fontsize=9)

    figS.suptitle("3-5  단발 uEPSP vs 논문 (Deuchars & Thomson 1996, Neuroscience 74:1009 · CA1 PC→PC)",
                  fontsize=12, y=0.99)
    figS.subplots_adjust(top=0.84, wspace=0.30)
    _ok = S["amp_mV"]["min"] <= f["amp_mV"] <= S["amp_mV"]["max"]
    plots.stamp(figS, f"3-5 | PC→PC · 시냅스 {b.n_syn()}개 · 진폭 {f['amp_mV']:.3f}mV = 실측범위 "
                      f"{S['amp_mV']['min']}~{S['amp_mV']['max']}mV {'안' if _ok else '밖'}"
                      f" (평균의 {amp_ratio:.1f}배)")
    plots.save(figS, outdir, "3-5_uepsp_stats.png")

    out = dict(cls=w.class_name, n_syn=b.n_syn(),
               amp_mV=round(f["amp_mV"], 4), latency_ms=round(f["latency_ms"], 3),
               rise_ms=round(f["rise_ms"], 3), halfwidth_ms=round(f["halfwidth_ms"], 3),
               decay_ms=round(f["decay_ms"], 3),
               deuchars1996=dict(amp_mean_mV=S["amp_mV"]["mean"], amp_max_mV=S["amp_mV"]["max"],
                              rise_ms=f"{S['rise_ms']['mean']}±{S['rise_ms']['sd']}",
                              halfwidth_ms=f"{S['halfwidth_ms']['mean']}±{S['halfwidth_ms']['sd']}"),
               amp_ratio_to_mean=round(amp_ratio, 2), amp_ratio_to_max=round(amp_vs_max, 2))
    jpath = os.path.join(outdir, "3-5_uepsp.json")
    with open(jpath, "w", encoding="utf-8") as f2:
        json.dump(out, f2, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    in_range = S["amp_mV"]["min"] <= f["amp_mV"] <= S["amp_mV"]["max"]
    print(f"\n[통과] 3-5 완료 — 진폭 {f['amp_mV']:.3f}mV, Deuchars1996 범위"
          f"({S['amp_mV']['min']}~{S['amp_mV']['max']}mV) "
          f"{'안' if in_range else '밖'} · 평균의 {amp_ratio:.1f}배")
    return 0


if __name__ == "__main__":
    sys.exit(main())
