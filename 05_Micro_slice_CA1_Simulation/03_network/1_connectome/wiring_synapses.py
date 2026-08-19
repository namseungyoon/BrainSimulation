# -*- coding: utf-8 -*-
"""
03_network/1_connectome/wiring_synapses.py  —  3-1(b) 직관: 시냅스 위치 버전

세포 하나(E3 근처 추체)에 연결된 내부 시냅스를 **실제 수상돌기 위치**에 점으로
찍는다. 대상 세포의 수상돌기 KDTree에 각 전시냅스 세포의 축삭을 질의해 접촉점을
찾고, 연결당 시냅스 수만큼 배치. 전시냅스 종류별 색.
결과: figures/3-3_wiring_synapses.png

재료: synapses_internal.npz · window_cells.npz · morphology_library · config
실행: python 03_network/1_connectome/wiring_synapses.py
"""
import os
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
CMAP = {"SP_PC": "#C44E52", "SP_Ivy": "#8172B3", "SP_PVBC": "#4C72B0", "SP_CCKBC": "#55A868",
        "SO_OLM": "#CCB974", "SP_BS": "#DD8452", "SO_Tri": "#937860", "SR_SCA": "#DA8BC3",
        "SO_BS": "#8C8C8C", "SLM_PPA": "#64B5CD", "SP_AA": "#E377C2", "SO_BP": "#7F7F7F"}


def load_swc(path):
    r = np.loadtxt(path, comments="#")
    return r[:, 1].astype(int), r[:, 2:5].astype(np.float64), r[:, 0].astype(int), r[:, 6].astype(int)


def to_local(pts, q, xyz0, seed, M):
    return (xyz0 + Rot.from_quat(q[[1, 2, 3, 0]]).apply(pts) - seed) @ M


def main():
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"]; Q = wc["orientation_wxyz"]; uvw = wc["uvw"]; mt = wc["mtype"].astype(str)
    morph = wc["morphology"].astype(str)
    syn = np.load(os.path.join(DERIVED, "synapses_internal.npz"))
    pre = syn["pre_gid"]; post = syn["post_gid"]; ns = syn["n_syn"]
    cfg = json.load(open(CFG, encoding="utf-8")); fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); M = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    e3 = np.array(next(e for e in cfg["electrodes"]["list"] if e["role"] == "stim")["xyz_um"])
    rng = np.random.default_rng(1)

    pc = np.where(mt == "SP_PC")[0]
    tgt = pc[np.argmin(np.linalg.norm(XYZ[pc] - e3, axis=1))]

    # 대상 수상돌기(local) + KDTree
    typ, pts, idx, par = load_swc(os.path.join(LIB, morph[tgt] + ".swc"))
    id2 = {i: k for k, i in enumerate(idx)}
    dmask = (typ == 1) | (typ == 3) | (typ == 4)
    dloc = to_local(pts[dmask], Q[tgt], XYZ[tgt], seed, M)
    tree = cKDTree(dloc)
    segs = [[(to_local(pts[[id2[par[k]]]], Q[tgt], XYZ[tgt], seed, M)[0, :2]),
             (to_local(pts[[k]], Q[tgt], XYZ[tgt], seed, M)[0, :2])]
            for k in range(len(idx)) if par[k] in id2 and typ[k] in (1, 3, 4)]

    inmask = post == tgt
    ins = pre[inmask]; insyn = ns[inmask]
    syn_u, syn_r, syn_c = [], [], []
    acache = {}
    from collections import Counter
    cnt = Counter()
    for p, k in zip(ins, insyn):
        if morph[p] not in acache:
            t2, pt2, i2, pr2 = load_swc(os.path.join(LIB, morph[p] + ".swc"))
            acache[morph[p]] = pt2[t2 == 2][::10]   # 축삭 서브샘플
        ax = acache[morph[p]]
        if len(ax) == 0:
            continue
        axl = to_local(ax, Q[p], XYZ[p], seed, M)
        d, ii = tree.query(axl, distance_upper_bound=4.0)
        hit = ii[np.isfinite(d)]
        if len(hit) == 0:
            _, ii2 = tree.query(axl); hit = ii2[:1]
        chosen = rng.choice(np.unique(hit), min(int(k), len(np.unique(hit))), replace=len(np.unique(hit)) < int(k))
        for c in chosen:
            syn_u.append(dloc[c, 0]); syn_r.append(dloc[c, 1]); syn_c.append(CMAP.get(mt[p], "gray"))
        cnt[mt[p]] += 1

    fig, ax = plt.subplots(figsize=(11, 12))
    ax.add_collection(LineCollection(segs, colors="0.6", linewidths=0.5, alpha=0.7, zorder=1))
    ax.scatter(syn_u, syn_r, s=14, c=syn_c, alpha=0.75, edgecolors="white", linewidths=0.2, zorder=3)
    ax.plot(uvw[tgt, 0], uvw[tgt, 1], "*", color="black", ms=24, zorder=6)
    ax.set_aspect("equal"); ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm, SP=0 · +위=SR/SLM)")
    ax.set_title(f"3-1(b) 시냅스 위치 — 추체 1개 수상돌기 위 내부 시냅스 {len(syn_u)}개\n"
                 f"(회색=수상돌기, 점=시냅스 실제 위치, 종류별 색)")
    handles = [Patch(color=CMAP[m], label=f"{m} ({cnt[m]}쌍)") for m in sorted(cnt, key=lambda x: -cnt[x])]
    handles += [plt.Line2D([], [], marker="*", ls="", mfc="black", mec="black", ms=13, label="대상 추체 소마")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, title=f"전시냅스 유형 · 시냅스 {len(syn_u)}개")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-3_wiring_synapses.png"), dpi=135)
    plt.close(fig)
    print(f"[3-3] 추체 gid={tgt} · 내부 시냅스 위치 {len(syn_u)}개 배치")
    print(f"[3-3] 그림 -> {FIG}/3-3_wiring_synapses.png")


if __name__ == "__main__":
    main()
