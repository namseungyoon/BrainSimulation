# -*- coding: utf-8 -*-
"""
03_network/1_connectome/wiring_full.py  —  3-1(b) 직관: 완전체 wiring

대상 추체 + 연결된 전시냅스 세포들의 **전체 형태**를 그린다. 색 규칙:
  - 뉴런 종류 = 같은 색
  - 수상돌기 = 그 색의 진한 톤 / 축삭 = 연한 톤
  - 시냅스 = '*' 마커 (실제 접촉 위치)
가독성 위해 유형별 대표 세포(--all 로 전체). 결과: figures/3-1b_wiring_full.png

실행: python 03_network/1_connectome/wiring_full.py [--all]
"""
import os
import sys
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
ALLC = "--all" in sys.argv
CMAP = {"SP_PC": "#C44E52", "SP_Ivy": "#8172B3", "SP_PVBC": "#4C72B0", "SP_CCKBC": "#55A868",
        "SO_OLM": "#CCB974", "SP_BS": "#DD8452", "SO_Tri": "#937860", "SR_SCA": "#DA8BC3",
        "SO_BS": "#8C8C8C", "SLM_PPA": "#64B5CD", "SP_AA": "#E377C2", "SO_BP": "#7F7F7F"}


def dark(c, f=0.55):
    r, g, b = mc.to_rgb(c); return (r*f, g*f, b*f)


def light(c, f=0.55):
    r, g, b = mc.to_rgb(c); return (r+(1-r)*f, g+(1-g)*f, b+(1-b)*f)


def load_swc(path):
    r = np.loadtxt(path, comments="#")
    return r[:, 1].astype(int), r[:, 2:5].astype(np.float64), r[:, 0].astype(int), r[:, 6].astype(int)


def L(pts, q, xyz0, seed, M):
    return (xyz0 + Rot.from_quat(q[[1, 2, 3, 0]]).apply(pts) - seed) @ M


def cell_segs(typ, pts, idx, par, q, xyz0, seed, M, which):
    id2 = {i: k for k, i in enumerate(idx)}
    loc = L(pts, q, xyz0, seed, M)
    return [[(loc[id2[par[k]], 0], loc[id2[par[k]], 1]), (loc[k, 0], loc[k, 1])]
            for k in range(len(idx)) if par[k] in id2 and typ[k] in which], loc


def main():
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]; mt = wc["mtype"].astype(str); morph = wc["morphology"].astype(str)
    syn = np.load(os.path.join(DERIVED, "synapses_internal.npz"))
    pre = syn["pre_gid"]; post = syn["post_gid"]; ns = syn["n_syn"]
    cfg = json.load(open(CFG, encoding="utf-8")); fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); M = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    e3 = np.array(next(e for e in cfg["electrodes"]["list"] if e["role"] == "stim")["xyz_um"])
    rng = np.random.default_rng(1)
    pc = np.where(mt == "SP_PC")[0]
    tgt = pc[np.argmin(np.linalg.norm(XYZ[pc] - e3, axis=1))]

    inmask = post == tgt; ins = pre[inmask]; insyn = ns[inmask]
    # 대표 세포 선택 (유형별 시냅스 최다) 또는 전체
    chosen = []
    for m in sorted(set(mt[ins])):
        cand = ins[mt[ins] == m]; sc = insyn[mt[ins] == m]
        if ALLC:
            chosen += list(cand)
        else:
            chosen.append(cand[np.argmax(sc)])
    chosen = list(dict.fromkeys(chosen))

    fig, ax = plt.subplots(figsize=(12, 13))
    # 대상 추체(형태) — 수상돌기 진하게, 축삭 연하게
    typ, pts, idx, par = load_swc(os.path.join(LIB, morph[tgt] + ".swc"))
    dseg, dloc = cell_segs(typ, pts, idx, par, Q[tgt], XYZ[tgt], seed, M, (1, 3, 4))
    aseg, _ = cell_segs(typ, pts, idx, par, Q[tgt], XYZ[tgt], seed, M, (2,))
    tc = CMAP["SP_PC"]
    ax.add_collection(LineCollection(aseg, colors=[light(tc, 0.7)], linewidths=0.3, alpha=0.5, zorder=1))
    ax.add_collection(LineCollection(dseg, colors=[dark(tc)], linewidths=1.0, alpha=0.9, zorder=3))
    # 대상 수상돌기 KDTree (시냅스 위치용)
    dmask = (typ == 1) | (typ == 3) | (typ == 4)
    tree = cKDTree(L(pts[dmask], Q[tgt], XYZ[tgt], seed, M))
    tgt_dloc = L(pts[dmask], Q[tgt], XYZ[tgt], seed, M)

    used = set()
    for p in chosen:
        c = CMAP.get(mt[p], "gray")
        t2, p2, i2, pr2 = load_swc(os.path.join(LIB, morph[p] + ".swc"))
        ds, _ = cell_segs(t2, p2, i2, pr2, Q[p], XYZ[p], seed, M, (1, 3, 4))
        as_, _ = cell_segs(t2, p2, i2, pr2, Q[p], XYZ[p], seed, M, (2,))
        ax.add_collection(LineCollection(as_, colors=[light(c)], linewidths=0.25, alpha=0.35, zorder=1))
        ax.add_collection(LineCollection(ds, colors=[dark(c)], linewidths=0.7, alpha=0.8, zorder=2))
        # 시냅스 위치: 이 세포 축삭이 대상 수상돌기에 닿는 점
        axpts = L(p2[t2 == 2][::10], Q[p], XYZ[p], seed, M)
        if len(axpts):
            d, ii = tree.query(axpts, distance_upper_bound=4.0)
            hit = np.unique(ii[np.isfinite(d)])
            k = int(insyn[np.where(ins == p)[0][0]])
            if len(hit):
                sel = rng.choice(hit, min(k, len(hit)), replace=len(hit) < k)
                ax.scatter(tgt_dloc[sel, 0], tgt_dloc[sel, 1], s=60, marker="*", c=[c],
                           edgecolors="black", linewidths=0.4, zorder=5)
        used.add(mt[p])

    ax.plot(dloc[0, 0], dloc[0, 1], "o", color=dark(tc), ms=10, zorder=6)
    ax.set_aspect("equal"); ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm, SP=0 · +위=SR/SLM)")
    ttl = "전체" if ALLC else "유형별 대표"
    ax.set_title(f"3-1(b) 완전체 wiring ({ttl}) — 대상 추체 + 연결 세포 전체 형태\n"
                 f"종류=색 · 수상돌기=진한톤 · 축삭=연한톤 · 시냅스=★")
    handles = [Patch(color=CMAP[m], label=m) for m in sorted(used)]
    handles = [Patch(color=dark(CMAP["SP_PC"]), label="대상 추체(수상돌기)")] + handles
    handles += [plt.Line2D([], [], marker="*", ls="", mfc="gray", mec="black", ms=13, label="시냅스"),
                plt.Line2D([], [], color="0.5", lw=3, label="수상돌기=진함 / 축삭=연함")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, title="완전체 색 규칙")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-1b_wiring_full.png"), dpi=135)
    plt.close(fig)
    print(f"[3-1b] 완전체 그림 ({ttl}, 세포 {len(chosen)+1}개) -> {FIG}/3-1b_wiring_full.png")


if __name__ == "__main__":
    main()
