# -*- coding: utf-8 -*-
"""Ex3 예상 3D UI 빌더 — 섬유 모집률 → 발화 뉴런 구름(예상치).

실제 커넥텀 기반 물리 모델:
 - 자극 세기(모집률 r) = locus에서 가까운 순으로 활성화되는 섬유의 비율.
   각 섬유의 locus 거리 = 그 섬유 시냅스들의 평균 dist_e3. 가까운 섬유부터 문턱을 넘음.
 - 모집률 r에서 활성 시냅스 = 자기 섬유의 거리-백분위 q_f <= r 인 시냅스.
 - 각 PC는 SC 시냅스 200개. 그중 need_i(=필요 비율, 세포별 흥분성 잡음) 만큼이
   활성화되면 발화. need_i>1 이면 (SC 단독으론) 발화 안 함.
 - need_i 분포는 r=1(전 섬유)에서 ~39%(Ex1 volley) 발화하도록 보정.
 => locus 근처 시냅스를 많이 받는 PC가 낮은 r에서 먼저 발화 → 구름이 밖으로 성장.
결과: ex3_recruit3d.html(자립형). 실행: python build_ex3_recruit3d.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
ANCHOR = 0.39            # Ex1 volley 발화율 (r=1 보정 목표)
STEPS = [10, 25, 50, 75, 100]
SEED = 42


def main():
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    syn = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    fib = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    xyz = wc["xyz"].astype(float)
    mt = wc["mtype"].astype(str)
    is_pc = (mt == "SP_PC")

    post = syn["post_gid"].astype(int)     # window_cells 행 인덱스
    dist = syn["dist_e3"].astype(float)    # 시냅스별 locus(E3) 거리
    fid = fib["fiber_id"].astype(int)      # 시냅스별 섬유 id (0..9999)
    e3 = syn["e3_xyz"].astype(float)
    nfib = int(fib["n_fiber"]) if "n_fiber" in fib.files else int(fid.max() + 1)

    # 섬유별 locus 거리 = 그 섬유 시냅스들의 평균 dist_e3 → 가까운 순 백분위 q_f (0=최근접)
    fsum = np.bincount(fid, weights=dist, minlength=nfib)
    fcnt = np.bincount(fid, minlength=nfib)
    fdist = fsum / np.maximum(fcnt, 1)
    order = np.argsort(fdist)
    qf = np.empty(nfib)
    qf[order] = np.arange(nfib) / (nfib - 1)   # 모집률 r에서 활성: qf <= r  (=r 비율의 섬유)
    sq = qf[fid]                               # 시냅스별 활성화 임계 모집률

    # PC별로 자기 시냅스들의 sq 정렬 수집
    pc_rows = np.where(is_pc & (np.bincount(post, minlength=len(xyz)) > 0))[0]
    sidx = np.argsort(post, kind="stable")
    post_s = post[sidx]; sq_s = sq[sidx]
    uniq, start = np.unique(post_s, return_index=True)
    start = np.append(start, len(post_s))
    row2slot = {int(u): k for k, u in enumerate(uniq)}

    rng = np.random.default_rng(SEED)
    fireR = np.full(len(pc_rows), 99.0)
    nsyn = np.zeros(len(pc_rows), int)
    for j, row in enumerate(pc_rows):
        k = row2slot[int(row)]
        qs = np.sort(sq_s[start[k]:start[k + 1]])   # 오름차순 (가까운 섬유부터)
        nsyn[j] = len(qs)
        need = 0.5 + 1.3 * rng.random()             # 세포별 필요비율(흥분성 잡음)
        nc = int(np.ceil(len(qs) * need))
        if nc <= len(qs):
            fireR[j] = float(qs[nc - 1])            # nc번째 시냅스 활성 시점 = 발화 모집률

    n_full = int(np.sum(fireR <= 1.0))
    print(f"[ex3-3d] PC {len(pc_rows)}개 · SC시냅스/PC {int(np.median(nsyn))} · 섬유 {nfib}", flush=True)
    print(f"[ex3-3d] r=1 발화 {n_full}/{len(pc_rows)} ({100*n_full/len(pc_rows):.0f}%) (목표 {int(ANCHOR*100)}%)", flush=True)
    for s in STEPS:
        nf = int(np.sum(fireR <= s / 100.0))
        print(f"   volley {s:3d}% -> 발화 {nf:4d} ({100*nf/len(pc_rows):2.0f}%)", flush=True)

    P = xyz[pc_rows].copy()
    c = P.mean(axis=0)
    P = P - c                         # 중심 정렬
    locus = (e3 - c)
    # 공간 구배 진단: fireR vs locus 측면거리 상관
    latd = np.hypot(P[:, 0] - locus[0], P[:, 2] - locus[2])
    fin = fireR <= 1.0
    if fin.sum() > 5:
        cc = np.corrcoef(latd[fin], fireR[fin])[0, 1]
        print(f"[ex3-3d] 공간구배 corr(측면거리, 발화모집률)={cc:+.2f} (양수=밖으로 성장 OK)", flush=True)

    # I-O 예상(예보 대시보드와 동일)
    io = {"x": STEPS, "normal": [8, 24, 48, 68, 80], "block": [12, 36, 68, 90, 100]}

    out = {
        "n": int(len(pc_rows)),
        "pos": [[round(float(P[i, 0]), 1), round(float(P[i, 1]), 1), round(float(P[i, 2]), 1)] for i in range(len(P))],
        "fireR": [round(float(x), 4) for x in fireR],
        "locus": [round(float(v), 1) for v in locus],
        "steps": STEPS, "anchor": ANCHOR, "io": io,
        "bx": [round(float(P[:, 0].min()), 1), round(float(P[:, 0].max()), 1)],
        "by": [round(float(P[:, 1].min()), 1), round(float(P[:, 1].max()), 1)],
        "bz": [round(float(P[:, 2].min()), 1), round(float(P[:, 2].max()), 1)],
    }
    data = json.dumps(out, separators=(",", ":"))
    tpl_p = os.path.join(HERE, "ex3_recruit3d_tpl.html")
    if os.path.exists(tpl_p):
        html = open(tpl_p, encoding="utf-8").read().replace("__INJECT__", data)
        _uidir = os.path.join(ROOT, "04_experiments", "Ex3_io_inhibition", "ui"); os.makedirs(_uidir, exist_ok=True)
        outp = os.path.join(_uidir, "ex3_recruit3d.html")
        open(outp, "w", encoding="utf-8").write(html)
        print(f"[ex3-3d] UI -> {outp} ({len(html)//1024}KB)", flush=True)
    else:
        print("[ex3-3d] 템플릿 없음:", tpl_p, flush=True)


if __name__ == "__main__":
    main()
