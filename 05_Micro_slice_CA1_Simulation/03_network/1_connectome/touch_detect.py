# -*- coding: utf-8 -*-
"""
03_network/1_connectome/touch_detect.py  —  3-1(b) 내부 커넥텀: touch detection

BBP/Romani 표준: 각 세포의 축삭(pre)과 다른 세포의 수상돌기(post)가 근접(~R µm)
하는 지점 = apposition(접촉후보)을 검출한다. (다음 단계 pruning에서 솎음)
전세포 완전형태(축삭·수상돌기)가 필요한 이유가 바로 이것.
  - pre  = axon(type 2) 점, post = dendrite(soma/basal/apical) 점
  - 서브샘플링(밀도보정)으로 로컬 계산 가능하게. 자기연결(pre==post) 제외.
결과: data/derived/appositions.npz (pre_gid,post_gid,n_app) + mtype×mtype 히트맵

재료: window_cells.npz · morphology_library
실행: python 03_network/1_connectome/touch_detect.py [--axon N] [--dend N] [--r µm]
"""
import os
import sys
import json
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
DERIVED = os.path.join(ROOT, "data", "derived")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

AXON_STRIDE = arg("--axon", 30)   # 축삭 점 서브샘플 간격
DEND_STRIDE = arg("--dend", 15)   # 수상돌기 점 서브샘플 간격
RADIUS = arg("--r", 3.0)          # apposition 근접 임계(µm)
MTYPES = ["SP_PC", "SP_Ivy", "SP_PVBC", "SP_CCKBC", "SO_OLM", "SP_BS",
          "SO_Tri", "SR_SCA", "SO_BS", "SLM_PPA", "SP_AA", "SO_BP"]


def load_pts(path):
    rows = np.loadtxt(path, comments="#")
    typ = rows[:, 1].astype(int); xyz = rows[:, 2:5].astype(np.float32)
    ax = xyz[typ == 2][::AXON_STRIDE]
    de = xyz[(typ == 1) | (typ == 3) | (typ == 4)][::DEND_STRIDE]
    return ax, de


def main():
    t0 = time.time()
    d = np.load(NPZ, allow_pickle=True)
    XYZ = d["xyz"]; Q = d["orientation_wxyz"]; morph = d["morphology"].astype(str); mt = d["mtype"].astype(str)
    n = len(XYZ)
    cache = {}
    AX, AXG, DE, DEG = [], [], [], []
    for i in range(n):
        if morph[i] not in cache:
            cache[morph[i]] = load_pts(os.path.join(LIB, morph[i] + ".swc"))
        ax, de = cache[morph[i]]
        Rq = Rot.from_quat(Q[i][[1, 2, 3, 0]])
        if len(ax):
            AX.append((XYZ[i] + Rq.apply(ax)).astype(np.float32)); AXG.append(np.full(len(ax), i, np.int32))
        if len(de):
            DE.append((XYZ[i] + Rq.apply(de)).astype(np.float32)); DEG.append(np.full(len(de), i, np.int32))
    AX = np.concatenate(AX); AXG = np.concatenate(AXG)
    DE = np.concatenate(DE); DEG = np.concatenate(DEG)
    print(f"[재료] 축삭점 {len(AX):,}(stride {AXON_STRIDE}) · 수상돌기점 {len(DE):,}(stride {DEND_STRIDE}) · R={RADIUS}µm · {time.time()-t0:.0f}s")

    tree = cKDTree(DE)
    print(f"[KDTree] 구축 완료 {time.time()-t0:.0f}s")
    pairs = {}   # (pre,post) -> apposition count
    CH = 200000
    for s in range(0, len(AX), CH):
        idx = tree.query_ball_point(AX[s:s+CH], RADIUS, workers=-1)
        for k, nb in enumerate(idx):
            if not nb:
                continue
            pre = AXG[s + k]
            for j in nb:
                post = DEG[j]
                if post == pre:
                    continue
                key = (pre, post)
                pairs[key] = pairs.get(key, 0) + 1
        if (s // CH) % 5 == 0:
            print(f"  축삭 {s+CH:,}/{len(AX):,} 처리 · 현재 쌍 {len(pairs):,} · {time.time()-t0:.0f}s")

    if not pairs:
        print("[경고] apposition 0 — 임계/서브샘플 조정 필요"); return
    keys = np.array(list(pairs.keys())); cnt = np.array(list(pairs.values()), np.int32)
    pre = keys[:, 0]; post = keys[:, 1]
    np.savez_compressed(os.path.join(DERIVED, "appositions.npz"),
                        pre_gid=pre.astype(np.int32), post_gid=post.astype(np.int32), n_app=cnt,
                        axon_stride=AXON_STRIDE, dend_stride=DEND_STRIDE, radius=RADIUS)
    print(f"\n[결과] apposition {cnt.sum():,}개 · 연결쌍(후보) {len(pre):,}개 · {time.time()-t0:.0f}s")
    print(f"[쌍당 apposition] 중앙 {np.median(cnt):.0f} · 평균 {cnt.mean():.1f} · 최대 {cnt.max()}")
    # mtype x mtype 쌍 수
    mat = np.zeros((12, 12))
    mi = {m: k for k, m in enumerate(MTYPES)}
    for a, b in zip(mt[pre], mt[post]):
        mat[mi[a], mi[b]] += 1
    print("[상위 pathway(쌍수)]")
    flat = [(MTYPES[i], MTYPES[j], int(mat[i, j])) for i in range(12) for j in range(12) if mat[i, j] > 0]
    for a, b, v in sorted(flat, key=lambda x: -x[2])[:10]:
        print(f"   {a:>9} -> {b:<9} {v:,}")
    fig_matrix(mat)
    print(f"[3-1b] 저장 -> data/derived/appositions.npz · 그림 -> {FIG}/3-1b_touch_matrix.png")


def fig_matrix(mat):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    lm = np.log10(mat + 1)
    im = ax.imshow(lm, cmap="magma", aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(MTYPES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(12)); ax.set_yticklabels(MTYPES, fontsize=8)
    ax.set_xlabel("post (수상돌기)"); ax.set_ylabel("pre (축삭)")
    for i in range(12):
        for j in range(12):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=6,
                        color="white" if lm[i, j] < lm.max()*0.6 else "black")
    fig.colorbar(im, label="log10(연결쌍 수 +1)")
    ax.set_title(f"3-1(b) touch detection — mtype×mtype 연결쌍(후보, pruning 전)\n"
                 f"축삭 stride {AXON_STRIDE}·수상돌기 stride {DEND_STRIDE}·R {RADIUS}µm")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-1b_touch_matrix.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
