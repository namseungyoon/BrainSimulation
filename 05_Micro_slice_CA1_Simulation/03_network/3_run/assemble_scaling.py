# -*- coding: utf-8 -*-
"""
03_network/3_run/assemble_scaling.py  —  물리적 구조 완성: 전체 5,610 조립 실현가능성 측정

세포 인스턴스화를 소수→전체(50·200·1000·전체)로 늘려가며 메모리(RSS)·시간을
실측하고, 전체(5,610) 조립을 시도한다. 축소 아님 — 완전형태 세포를 실제로 NEURON에
올릴 수 있는지 확인(메모리/시간 예산).

실행(반드시 mechbuild cwd 아님 · 컴파일된 mechanism):
  python 03_network/3_run/assemble_scaling.py
"""
import os
import sys
import time
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
import net_build as nb

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    print("=== 물리적 구조: 전체 5,610 조립 실현가능성 측정 ===", flush=True)
    t0 = time.time()
    B = nb.NetBuilder()
    N = len(B.mt)
    print(f"[창 세포] 총 {N:,}개 · mechanism 로드 완료 ({time.time()-t0:.1f}s, RSS {nb.rss_mb():.0f}MB)", flush=True)

    order = np.arange(N)              # gid 순서대로 누적
    marks = [50, 200, 1000, 3000, N]
    rows = []
    built = 0
    for target in marks:
        t = time.time()
        B.build_cells(order[built:target])
        built = target
        nsec, nseg = B.counts()
        dt = time.time() - t
        rss = nb.rss_mb()
        rows.append((built, nsec, nseg, rss, time.time() - t0))
        print(f"[누적 {built:>5,}세포] +{dt:6.1f}s · section {nsec:,} · segment {nseg:,} · "
              f"RSS {rss:,.0f}MB · 경과 {time.time()-t0:.0f}s", flush=True)

    tot = time.time() - t0
    print(f"\n[완료] 전체 {built:,}세포 조립 · 총 {tot:.0f}s · 최종 RSS {nb.rss_mb():,.0f}MB", flush=True)
    print(f"       세포당 평균 {tot/built*1000:.1f}ms · segment 총 {rows[-1][2]:,}개", flush=True)
    json.dump({"rows": rows, "total_s": tot, "n": built},
              open(os.path.join(ROOT, "scratch", "assemble_scaling.json"), "w"))


if __name__ == "__main__":
    main()
