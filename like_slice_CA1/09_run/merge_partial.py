# -*- coding: utf-8 -*-
"""
09_run/merge_partial.py  —  아직 구동 중인 런의 '완료된' 세그먼트만 병합(미리보기용)

랭크별 _rank{r}_seg{s}.csv 중 현재까지 완성된 세그먼트를 모아
_PREVIEW_spikes.csv(gid,type,t_ms) 로 병합. 진행 중 런과 충돌 없이(완료분만 읽음) 미리보기 GIF 재료 생성.

실행: python 09_run/merge_partial.py
출력: spikes/_PREVIEW_spikes.csv, 최대 시간(ms) 콘솔 출력
"""
import os
import csv
import glob
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSVDIR = os.path.join(HERE, "spikes")


def main():
    pos = np.load(os.path.join(CSVDIR, "FULL_positions.npz"), allow_pickle=True)
    gtype = pos["type"].astype(str)  # gid 순서(0..N-1)

    # 완료된 세그먼트 index: 모든 10랭크 파일이 존재하는 seg만 사용(부분기록 배제)
    seg_files = glob.glob(os.path.join(CSVDIR, "_rank*_seg*.csv"))
    segs = {}
    for f in seg_files:
        m = re.search(r"_rank(\d+)_seg(\d+)\.csv$", os.path.basename(f))
        if m:
            segs.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
    n_rank = max((max(v) for v in segs.values()), default=-1) + 1
    complete = sorted(s for s, ranks in segs.items() if len(ranks) == n_rank)
    if not complete:
        print("[미리보기] 완료 세그먼트 없음"); return
    # 연속된 앞부분만(중간 빠짐 방지)
    contiguous = []
    for i, s in enumerate(complete):
        if s == i:
            contiguous.append(s)
        else:
            break

    rows = []
    for s in contiguous:
        for r in range(n_rank):
            fn = os.path.join(CSVDIR, f"_rank{r}_seg{s:02d}.csv")
            with open(fn, encoding="utf-8") as f:
                rd = csv.reader(f); next(rd, None)
                for gid_s, t_s in rd:
                    g = int(gid_s); rows.append((g, gtype[g], float(t_s)))
    rows.sort(key=lambda x: (x[2], x[0]))
    out = os.path.join(CSVDIR, "_PREVIEW_spikes.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["gid", "type", "t_ms"]); w.writerows(rows)
    tmax = (contiguous[-1] + 1) * 50  # seg_ms=50 가정
    fired = len({x[0] for x in rows})
    print(f"[미리보기] 세그 {contiguous[0]}~{contiguous[-1]} ({tmax}ms) · "
          f"스파이크 {len(rows):,} · 발화세포 {fired}/{len(gtype)} → {os.path.basename(out)}")
    print(f"TMAX={tmax}")


if __name__ == "__main__":
    main()
