# -*- coding: utf-8 -*-
"""3-5 단발 uEPSP — pre 1발에 대한 연결 단위 EPSP 특성 (진폭·지연·상승·감쇠)

단계   : 3-5 (파이프라인 3단계 시냅스 / 하위 5 uepsp)
쉬운 설명: pre 가 한 번 발화하면 post 소마에 작은 흥분성 전위(EPSP)가 생긴다. 그 크기·모양을
          잰다. 두 세포의 '연결 1개'는 시냅스 접촉 5개의 합이므로, 여기 EPSP 는 연결 단위(uEPSP).
방법   : lib.wiring(동결 전달) 로 pre 1발 발화 → post 소마 EPSP 파형에서 진폭·개시지연·상승시간
          (20→80%)·반치폭·감쇠(1/e)를 측정(lib.measure). 문헌 범위와 대조.
근거   : Sayer, Friedlander & Redman 1990 — CA3→CA1 단일 연결 uEPSP (전형 ~0.1~1.0 mV).
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
from lib.bench import Bench                   # noqa: E402
from lib.wiring import Wiring                 # noqa: E402

T_SPIKE = 20.0
TSTOP = 90.0
# 문헌 전형 범위 (Sayer 1990: 단일 연결 uEPSP). 평균은 작으나 분포 상단은 ~1mV.
LIT = {"amp_mV": (0.1, 1.0)}


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
    in_lit = LIT["amp_mV"][0] <= f["amp_mV"] <= LIT["amp_mV"][1]
    print(f"  문헌 범위(Sayer1990 {LIT['amp_mV'][0]}~{LIT['amp_mV'][1]}mV): "
          f"{'안' if in_lit else '상단 초과' if f['amp_mV']>LIT['amp_mV'][1] else '하단 미만'}")

    import matplotlib.pyplot as plt
    fig, (axT, axS) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # 왼쪽: EPSP 파형 + 특성 표시
    base = v[(t >= T_SPIKE - 5) & (t < T_SPIKE)].mean()
    axT.plot(t, v, color="#d84315", lw=1.8)
    axT.axhline(base, ls=":", color="#999", lw=0.8)
    tpk = T_SPIKE + b.syn_specs[0]["delay_ms"] + f["t_peak_ms"] - (T_SPIKE + b.syn_specs[0]["delay_ms"])
    # 정점 표시
    vpk = base + f["amp_mV"]
    axT.annotate(f"진폭 {f['amp_mV']:.3f} mV", xy=(f["t_peak_ms"], vpk),
                 xytext=(f["t_peak_ms"]+8, vpk), fontsize=10, color="#b71c1c",
                 arrowprops=dict(arrowstyle="->", color="#b71c1c"))
    axT.axvline(T_SPIKE, ls="--", color="#2e7d32", lw=1, alpha=0.7)
    axT.text(T_SPIKE, axT.get_ylim()[0], " pre 발화", fontsize=8, color="#2e7d32", va="bottom")
    axT.set_xlim(T_SPIKE-5, T_SPIKE+50)
    axT.set_xlabel("시간 (ms)"); axT.set_ylabel("post 소마 Vm (mV)")
    axT.set_title(f"A. 단발 uEPSP 파형 (연결 = 시냅스 {b.n_syn()}개 합)", fontsize=10.5, loc="left")

    # 오른쪽: 특성 막대 + 진폭 문헌범위
    names = ["진폭\n(mV)", "개시지연\n(ms)", "상승20-80\n(ms)", "반치폭\n(ms)", "감쇠1/e\n(ms)"]
    vals = [f["amp_mV"], f["latency_ms"], f["rise_ms"], f["halfwidth_ms"], f["decay_ms"]]
    cols = ["#d84315", "#546e7a", "#546e7a", "#546e7a", "#546e7a"]
    axS.bar(range(len(vals)), vals, color=cols)
    axS.set_xticks(range(len(vals))); axS.set_xticklabels(names, fontsize=8.5)
    for i, val in enumerate(vals):
        axS.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
    # 진폭 문헌 범위 띠
    axS.axhspan(LIT["amp_mV"][0], LIT["amp_mV"][1], xmin=0.0, xmax=0.2,
                color="#ffb300", alpha=0.25)
    axS.text(0, LIT["amp_mV"][1]+0.05, f"문헌 {LIT['amp_mV'][0]}~{LIT['amp_mV'][1]}",
             fontsize=7.5, color="#b26a00", ha="center")
    axS.set_title("B. EPSP 특성 (진폭=주황, 문헌범위 띠)", fontsize=10.5, loc="left")
    axS.set_ylabel("값")

    fig.suptitle("3-5  단발 uEPSP — pre 1발 → post 소마 EPSP 특성 (동결 전달 시냅스)",
                 fontsize=12.5, y=0.99)
    fig.subplots_adjust(top=0.88, wspace=0.22)
    tag = "안" if in_lit else ("상단 초과" if f["amp_mV"] > LIT["amp_mV"][1] else "하단")
    plots.stamp(fig, f"3-5 | {w.class_name} · 진폭 {f['amp_mV']:.3f}mV (문헌 {tag}) · 3-7 에서 g 재보정")
    outdir = plots.figdir(__file__)
    # 두 파일명(같은 번호, slug 다름) 규약대로: trace / stats
    fig.savefig(os.path.join(outdir, "3-5_uepsp_trace.png"))
    fig.savefig(os.path.join(outdir, "3-5_uepsp_stats.png"))
    import matplotlib.pyplot as _plt; _plt.close(fig)
    print(f"saved: 3-5_uepsp_trace.png · 3-5_uepsp_stats.png")

    out = dict(cls=w.class_name, n_syn=b.n_syn(),
               amp_mV=round(f["amp_mV"], 4), latency_ms=round(f["latency_ms"], 3),
               rise_ms=round(f["rise_ms"], 3), halfwidth_ms=round(f["halfwidth_ms"], 3),
               decay_ms=round(f["decay_ms"], 3),
               lit_amp_range=LIT["amp_mV"], amp_in_lit=bool(in_lit))
    jpath = os.path.join(outdir, "3-5_uepsp.json")
    with open(jpath, "w", encoding="utf-8") as f2:
        json.dump(out, f2, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    print("\n[통과] 3-5 완료" + ("" if in_lit else " (진폭 문헌 상단 초과 — 3-7 재보정 예정)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
