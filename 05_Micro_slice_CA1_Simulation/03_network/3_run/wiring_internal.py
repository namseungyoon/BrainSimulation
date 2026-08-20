# -*- coding: utf-8 -*-
"""
03_network/3_run/wiring_internal.py  —  시냅스 배선 ②: 내부 시냅스 부착 (실현가능성 실측)

조립된 세포(subset)에 내부 시냅스(E=ProbAMPANMDA_EMS, I=ProbGABAAB_EMS, STP)를 부착.
- 위치: 연결쌍만 저장돼 있어 부착 위치는 **pre mtype로 타깃 구획 유도** 후 post
  수상돌기에서 샘플(touch+prune으로 정한 연결·개수는 보존, 정확 apposition xyz는 근사).
    perisomatic(PVBC/CCKBC/AA)·SLM(OLM/BS_SO/BP/PPA)·SR(SCA)·전수상(Ivy/BS/Tri/PC)
- STP·gsyn: 22 internal rules(id별 U/D/F/NRRP·gsyn·tdecay·nmda_ampa) 배정
- 구동: NetCon(pre 소마 스파이크 → syn). 내부 pre는 실제 세포라 정체성 자동.
소수 세포로 개당 시간·메모리 실측 → 전체 483만 예산 예측.

실행: python 03_network/3_run/wiring_internal.py [--ncell 400]
"""
import os
import sys
import time
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
import net_build as nb

DERIVED = os.path.join(ROOT, "data", "derived")
CFG = os.path.join(ROOT, "config", "synapse_rules.json")

NCELL = int(sys.argv[sys.argv.index("--ncell") + 1]) if "--ncell" in sys.argv else 400

PERI = {"SP_PVBC", "SP_CCKBC", "SP_AA"}
SLM = {"SO_OLM", "SO_BS", "SO_BP", "SLM_PPA"}
SR = {"SR_SCA"}


def target_comp(pre_mt):
    if pre_mt in PERI:
        return "peri"
    if pre_mt in SLM:
        return "slm"
    if pre_mt in SR:
        return "sr"
    return "dend"


def compartments(h, cell, XYZg, rot, seed, radial):
    """post 세포의 부착점을 구획별 인덱스로 분류(벡터화). 반환 (secs, xs, comp)."""
    secs, xs, P, soma = [], [], [], []
    for sec in cell.all:
        nm = sec.name(); n = int(sec.n3d())
        if n < 2 or ("axon" in nm) or ("node" in nm) or ("myelin" in nm):
            continue
        Lt = sec.arc3d(n - 1) or 1.0
        so = "soma" in nm
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            secs.append(sec); xs.append(min(max(sec.arc3d(i) / Lt, 0.0), 1.0)); soma.append(so)
    P = np.asarray(P, float)
    r = (XYZg + rot.apply(P) - seed) @ radial      # 세포 전체 1회 변환
    soma = np.asarray(soma); idx = np.arange(len(P))
    dend = idx[~soma]
    rd = r[dend]
    comp = {"soma": idx[soma], "dend": dend,
            "peri": np.concatenate([idx[soma], dend[np.abs(rd) < 60]]),
            "sr": dend[(rd >= 25) & (rd <= 450)], "slm": dend[rd > 450]}
    return secs, xs, comp


def main():
    t0 = time.time()
    B = nb.NetBuilder(); h = B.h
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]
    cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
    fr = cfg["frame_um"]; seed = np.array(fr["seed"])
    Mrows = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    rules = {r["id"]: r for r in json.load(open(CFG, encoding="utf-8"))["internal_rules"]}

    d = np.load(os.path.join(DERIVED, "synapses_internal.npz"), allow_pickle=True)
    pre = d["pre_gid"]; post = d["post_gid"]; nsyn = d["n_syn"]
    p = np.load(os.path.join(DERIVED, "synapse_params.npz"), allow_pickle=True)
    irule = p["internal_rule"]; igsyn = p["internal_gsyn"]; imech = p["internal_mech"].astype(str)
    mt = B.mt
    print(f"[rule id 범위] {irule.min()}~{irule.max()} · 규칙키 {sorted(rules)[:5]}...", flush=True)

    gids = np.arange(min(NCELL, len(mt)))
    sel = np.isin(pre, gids) & np.isin(post, gids)
    cidx = np.where(sel)[0]
    tot_syn = int(nsyn[cidx].sum())
    print(f"=== 내부 배선 실측 (세포 {len(gids):,} · 연결 {len(cidx):,} · 시냅스 {tot_syn:,}) ===", flush=True)

    tb = time.time(); B.build_cells(gids)
    print(f"[조립] {len(gids):,}세포 {time.time()-tb:.1f}s · RSS {nb.rss_mb():.0f}MB", flush=True)

    ta = time.time(); keep = []; n_made = 0; rng = np.random.default_rng(0)
    compcache = {}
    for ci in cidx:
        pg, qg, ns = int(pre[ci]), int(post[ci]), int(nsyn[ci])
        if qg not in compcache:
            rot = Rot.from_quat(Q[qg][[1, 2, 3, 0]])
            compcache[qg] = compartments(h, B.cells[qg], XYZ[qg], rot, seed, Mrows[:, 1])
        secs, xs, comp = compcache[qg]
        tc = target_comp(mt[pg])
        pool = comp[tc]
        if len(pool) == 0:
            pool = comp["dend"] if len(comp["dend"]) else comp["soma"]
        if len(pool) == 0:
            continue
        pick = pool[rng.integers(0, len(pool), ns)]
        rl = rules.get(int(irule[ci])); gs = float(igsyn[ci]); mech = imech[ci]
        pre_soma = B.cells[pg].soma[0]
        for k in pick:
            sec, x = secs[k], xs[k]
            if mech == "E":
                syn = h.ProbAMPANMDA_EMS(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
                    try: syn.NMDA_ratio = rl["nmda_ampa"]
                    except Exception: pass
            else:
                syn = h.ProbGABAAB_EMS(sec(x))
                if rl:
                    syn.Use = rl["U"]; syn.Dep = rl["D"]; syn.Fac = rl["F"]; syn.Nrrp = rl["NRRP"]
            nc = h.NetCon(pre_soma(0.5)._ref_v, syn, sec=pre_soma)
            nc.weight[0] = gs / 1000.0; nc.delay = 1.0; nc.threshold = -10
            keep.append((syn, nc)); n_made += 1

    dt = time.time() - ta; rss = nb.rss_mb()
    print(f"[부착] 내부 {n_made:,}개 · {dt:.1f}s (개당 {dt/max(n_made,1)*1000:.2f}ms) · RSS {rss:,.0f}MB", flush=True)
    NTOT = int(nsyn.sum())
    print(f"\n[전체 예측] 내부 {NTOT:,}개 부착 ≈ {dt/max(n_made,1)*NTOT/60:.1f}분(부착만)", flush=True)
    print(f"[요약] 총 {time.time()-t0:.0f}s", flush=True)
    json.dump({"ncell": int(len(gids)), "nsyn": n_made, "attach_s": dt,
               "per_syn_ms": dt / max(n_made, 1) * 1000, "rss_mb": rss, "ntot": NTOT},
              open(os.path.join(ROOT, "scratch", "wiring_internal_test.json"), "w"))


if __name__ == "__main__":
    main()
