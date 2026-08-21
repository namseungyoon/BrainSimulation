# -*- coding: utf-8 -*-
"""3-4 기록 장치 — 벤치의 모든 기록 채널이 작동하는지 확인 (재사용 lib.wiring 검증)

단계   : 3-4 (파이프라인 3단계 시냅스 / 하위 4 record)
방법   : lib.wiring.Wiring 으로 고정 벤치를 배선하고, pre 1발 발화에 대해 기록 가능한 모든
         신호 — pre 소마 전압 · post 소마 전압 · 시냅스별 국소 수상돌기 전압 · 전도도 g · 전류 i —
         를 한 번에 기록·표시한다. 국소 수상돌기 전압은 나중에 GluSynapse(5-7)가 쓸 신호라 미리 확인.
검증   : 모든 채널이 비어있지 않고 npz 왕복 성공 · 국소 전압이 소마보다 큰 EPSP(원위일수록).
결과   : figures/3-4_record_check.png · figures/3-4_record.json
실행   : . .\\env\\activate.ps1 ; & $Py04 03_synapse\\4_record\\3-4_record.py
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
from lib.bench import Bench                   # noqa: E402
from lib.wiring import Wiring                 # noqa: E402

T_SPIKE = 20.0
TSTOP = 80.0


def main():
    plots.setup()
    print("=== 3-4 기록 장치 ===")
    b = Bench()
    w = Wiring(b, frozen=True)
    print(f"  클래스 {w.class_name} · 시냅스 {len(w.syns)}개 · e_rev={w.p['e_rev_mV']}mV")
    w.drive_pre_iclamp([T_SPIKE], amp_nA=1.2, dur_ms=3.0)
    w.record(rec_dt=0.05, local_v=True, currents=True)
    w.run(TSTOP)
    R = w.arrays()

    t = R["t"]
    base = t < T_SPIKE
    pre_spikes = int(((R["pre_v"][:-1] < -10) & (R["pre_v"][1:] >= -10)).sum())
    post_epsp = float(R["post_v"].max() - R["post_v"][base].mean())
    local_epsp = [float(lv.max() - lv[base].mean()) for lv in R["local_v"]]
    g_peak = [float(g.max()) for g in R["g"]]
    dists = [spec["path_um"] for _, spec in w.syns]

    print(f"  pre 스파이크 {pre_spikes}발")
    print(f"  post 소마 EPSP {post_epsp:.3f} mV")
    print(f"  시냅스 국소 EPSP {[round(x,2) for x in local_epsp]} mV (경로거리 {[round(d) for d in dists]}um)")

    # ---- 그림: 4패널 ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    (a1, a2), (a3, a4) = axes
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(w.syns)))

    a1.plot(t, R["pre_v"], color="#2e7d32", lw=1.3)
    a1.set_title("A. pre 소마 전압 (자극 세포)", fontsize=10, loc="left")
    a1.set_ylabel("Vm (mV)"); a1.axvline(T_SPIKE, ls=":", color="#999", lw=0.8)

    for i, lv in enumerate(R["local_v"]):
        a2.plot(t, lv, color=colors[i], lw=1.1, label=f"{round(dists[i])}um")
    a2.plot(t, R["post_v"], color="#d84315", lw=1.8, label="소마")
    a2.set_title("B. post 국소 수상돌기 전압 vs 소마 (원위일수록 큰 국소 EPSP)", fontsize=10, loc="left")
    a2.set_ylabel("Vm (mV)"); a2.legend(fontsize=7.5, ncol=2, title="시냅스 위치")

    for i, g in enumerate(R["g"]):
        a3.plot(t, np.array(g)*1e3, color=colors[i], lw=1.1)
    a3.set_title("C. 시냅스 전도도 g (nS)", fontsize=10, loc="left")
    a3.set_ylabel("g (nS)"); a3.set_xlabel("시간 (ms)")

    for i, cur in enumerate(R["i"]):
        a4.plot(t, np.array(cur), color=colors[i], lw=1.1)
    a4.set_title("D. 시냅스 전류 i (nA)", fontsize=10, loc="left")
    a4.set_ylabel("i (nA)"); a4.set_xlabel("시간 (ms)")

    for ax in (a1, a2, a3, a4):
        ax.set_xlim(15, 55)

    fig.suptitle("3-4  벤치 기록 채널 검증 — pre 1발 → 모든 신호 동시 기록 (lib.wiring)",
                 fontsize=12.5, y=0.98)
    fig.subplots_adjust(top=0.90, hspace=0.32, wspace=0.20)
    plots.stamp(fig, f"3-4 | {w.class_name} · 동결 전달 · rec_dt 0.05ms · 국소전압=GluSynapse(5-7) 대비")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-4_record_check.png")

    # npz 왕복 검증 (밑줄 = gitignore)
    npz = os.path.join(outdir, "_3-4_traces.npz")
    np.savez(npz, t=t, pre_v=R["pre_v"], post_v=R["post_v"],
             g=np.array(R["g"]), i=np.array(R["i"]), local_v=np.array(R["local_v"]))
    reloaded = np.load(npz)
    roundtrip = bool(np.allclose(reloaded["post_v"], R["post_v"]))

    checks = [
        ("pre 1발 발화", pre_spikes == 1),
        ("post 소마 EPSP>0", post_epsp > 0),
        ("국소 전압 5채널 기록", len(R["local_v"]) == 5 and all(x > 0 for x in local_epsp)),
        ("전도도 5채널", len(R["g"]) == 5 and all(x > 0 for x in g_peak)),
        ("전류 5채널", len(R["i"]) == 5),
        ("국소 EPSP > 소마 EPSP", min(local_epsp) > post_epsp),
        ("npz 왕복", roundtrip),
    ]
    n_ok = sum(1 for _, ok in checks if ok)
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")

    out = dict(cls=w.class_name, pre_spikes=pre_spikes,
               post_epsp_mv=round(post_epsp, 4),
               local_epsp_mv=[round(x, 3) for x in local_epsp],
               g_peak_nS=[round(x*1e3, 3) for x in g_peak],
               dists_um=[round(d, 1) for d in dists],
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "3-4_record.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 3-4 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
