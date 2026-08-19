# -*- coding: utf-8 -*-
"""
03_network/1_connectome/wiring_figure.py  —  3-1(b) 직관: 세포 하나 wiring

E3 근처 추체세포 1개를 골라, 실제 형태(수상돌기)와 그 세포에 연결된 전시냅스
세포들(내부 커넥텀 결과)을 소마+연결선으로 그린다. 억제뉴런 종류별 색·연결선.
결과: figures/3-3_wiring.png

재료: synapses_internal.npz · window_cells.npz · morphology_library · config
실행: python 03_network/1_connectome/wiring_figure.py
"""
import os
import json
import numpy as np
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


def main():
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    xyz = wc["xyz"]; Q = wc["orientation_wxyz"]; uvw = wc["uvw"]; mt = wc["mtype"].astype(str)
    morph = wc["morphology"].astype(str)
    syn = np.load(os.path.join(DERIVED, "synapses_internal.npz"))
    pre = syn["pre_gid"]; post = syn["post_gid"]; ns = syn["n_syn"]
    cfg = json.load(open(CFG, encoding="utf-8")); fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); M = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    e3 = np.array(next(e for e in cfg["electrodes"]["list"] if e["role"] == "stim")["xyz_um"])

    # E3 근처 추체 1개
    pc = np.where(mt == "SP_PC")[0]
    tgt = pc[np.argmin(np.linalg.norm(xyz[pc] - e3, axis=1))]

    # 대상 세포 수상돌기(local u-r)
    rows = np.loadtxt(os.path.join(LIB, morph[tgt] + ".swc"), comments="#")
    typ = rows[:, 1].astype(int); pts = rows[:, 2:5]; idx = rows[:, 0].astype(int); par = rows[:, 6].astype(int)
    id2 = {i: k for k, i in enumerate(idx)}
    loc = (xyz[tgt] + Rot.from_quat(Q[tgt][[1, 2, 3, 0]]).apply(pts) - seed) @ M
    segs = [[(loc[id2[par[k]], 0], loc[id2[par[k]], 1]), (loc[k, 0], loc[k, 1])]
            for k in range(len(idx)) if par[k] in id2 and typ[k] in (1, 3, 4)]
    tu, tr = uvw[tgt, 0], uvw[tgt, 1]

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.add_collection(LineCollection(segs, colors="0.55", linewidths=0.5, alpha=0.7, zorder=1))

    # 입력 세포들
    inmask = post == tgt
    ins = pre[inmask]; insyn = ns[inmask]
    order = np.argsort(mt[ins])
    from collections import Counter
    cnt = Counter(mt[ins])
    for j in order:
        p = ins[j]; c = CMAP.get(mt[p], "gray")
        ax.plot([uvw[p, 0], tu], [uvw[p, 1], tr], color=c, lw=0.3 + 0.12 * insyn[j], alpha=0.5, zorder=2)
        ax.plot(uvw[p, 0], uvw[p, 1], "o", color=c, ms=4, zorder=3)
    ax.plot(tu, tr, "*", color="black", ms=26, zorder=6, label="대상 추체")

    ax.set_aspect("equal"); ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm, SP=0)")
    ax.set_title(f"3-1(b) 직관 wiring — E3 근처 추체 1개에 연결된 내부 입력 {inmask.sum()}쌍\n"
                 f"(수상돌기=회색, 전시냅스 소마·연결선=종류별 색, 선굵기∝시냅스수)")
    handles = [Patch(color=CMAP[m], label=f"{m} ({cnt.get(m,0)})") for m in
               sorted(cnt, key=lambda x: -cnt[x])]
    handles += [plt.Line2D([], [], marker="*", ls="", mfc="black", mec="black", ms=14, label="대상 추체")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, title=f"전시냅스 유형(쌍수) · 총 {inmask.sum()}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-3_wiring.png"), dpi=135)
    plt.close(fig)
    print(f"[3-3] 대상 추체 gid={tgt} (E3 근처) · 내부 입력쌍 {inmask.sum()} · 시냅스 {insyn.sum()}")
    print("[입력 유형]", dict(cnt))
    print(f"[3-3] 그림 -> {FIG}/3-3_wiring.png")


if __name__ == "__main__":
    main()
