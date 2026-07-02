# -*- coding: utf-8 -*-
"""
07_connectivity/touch_connectome.py  —  형태 접촉 기반 커넥텀 (Romani 방식, 축소판)

원리 (Romani 2024 touch-detection 근사):
  배치된 세포들의 축삭(pre)이 다른 세포의 수상돌기/소마(post)에 접촉반경 이내로
  근접하는 지점(apposition)을 시냅스 후보로 본다. 접촉 = 잠재 시냅스.

⚠️ 축소·근사:
  - 전체 17,647세포 전량 접촉검색은 로컬 PC 불가 → 하위 부피(박스) 세포만.
  - 형태 점을 세포당 축삭/수상 각 N_SUB개로 서브샘플 (표면 아닌 골격점).
  - 접촉반경 TOUCH_R µm (서브샘플 골격점 기준 근사; 실제 appositions는 ~1-2µm).
  - bouton 밀도 기반 pruning 미적용 → 'apposition≈잠재시냅스' 로 보고.

산출: touch_connectivity.npz(+summary.json), figures/V3b_*.png

실행: python 07_connectivity/touch_connectome.py
"""
import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.path.join(ROOT, "..", "papers",
                                "01_Ecker2020_CA1_synaptic", "05_paired_recording"))
import morph_transform as mt                    # noqa: E402
from pathway_map import pathway_class           # noqa: E402

MORPH_DIR = os.path.join(ROOT, "data", "morphology_library")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
ASSIGN = os.path.join(ROOT, "05b_memap", "model_assignment.npz")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

HALF_T1, HALF_T2 = 250.0, 50.0   # 박스 반폭: 가로500 × 두께100 µm × 전층
N_SUB = 500                       # 세포당 축삭/수상 서브샘플 점 수
TOUCH_R = 5.0                     # 접촉 반경 (µm)
rng = np.random.default_rng(0)


def local_frame(quat):
    R = mt.quat_to_R(quat)
    return R.apply([0., 1., 0.]), R.apply([1., 0., 0.]), R.apply([0., 0., 1.])


def main():
    c = np.load(CELLS, allow_pickle=True)
    a = np.load(ASSIGN, allow_pickle=True)
    xyz = c["xyz"].astype(float); quat = c["quat_wxyz"].astype(float)
    mt_us = c["mtype"].astype(str); morph = a["morphology"].astype(str)
    N = len(xyz)
    mtypes_hyphen = np.array([m.replace("_", "-") for m in mt_us])

    # 중심 추체 기준 박스 선택
    pc = np.where(mt_us == "SP_PC")[0]
    ctr = xyz[pc].mean(0)
    c0 = pc[np.argmin(np.linalg.norm(xyz[pc] - ctr, axis=1))]
    radial, t1, t2 = local_frame(quat[c0])
    rel = xyz - xyz[c0]
    inbox = (np.abs(rel @ t1) < HALF_T1) & (np.abs(rel @ t2) < HALF_T2)
    sub = np.where(inbox)[0]
    print(f"[box] 하위부피 세포 {len(sub)} / {N:,} "
          f"(가로{2*HALF_T1:.0f}×두께{2*HALF_T2:.0f}µm×전층)")

    # 각 세포 형태 변환 + 축삭(pre)/수상+소마(post) 점 서브샘플
    axon_pts, axon_cell = [], []
    dend_pts, dend_cell = [], []
    for local_i, k in enumerate(sub):
        try:
            s = mt.load_swc(os.path.join(MORPH_DIR, morph[k] + ".swc"))
        except FileNotFoundError:
            continue
        w, _ = mt.transform(s["xyz"], mt.soma_center(s), quat[k], xyz[k])
        ax = w[s["type"] == 2]
        de = w[(s["type"] == 1) | (s["type"] == 3) | (s["type"] == 4)]
        if len(ax) > N_SUB:
            ax = ax[rng.choice(len(ax), N_SUB, replace=False)]
        if len(de) > N_SUB:
            de = de[rng.choice(len(de), N_SUB, replace=False)]
        axon_pts.append(ax); axon_cell.append(np.full(len(ax), k))
        dend_pts.append(de); dend_cell.append(np.full(len(de), k))
        if (local_i + 1) % 200 == 0:
            print(f"  로드 {local_i+1}/{len(sub)}")
    axon_pts = np.vstack(axon_pts); axon_cell = np.concatenate(axon_cell)
    dend_pts = np.vstack(dend_pts); dend_cell = np.concatenate(dend_cell)
    print(f"[points] 축삭 {len(axon_pts):,} · 수상/소마 {len(dend_pts):,}")

    # 접촉 검색: 축삭점 근처(≤TOUCH_R) 수상점 → (pre,post) appositions
    tree = cKDTree(dend_pts)
    appo = Counter()                              # (pre,post) -> apposition 수
    nbr = tree.query_ball_point(axon_pts, TOUCH_R)
    for ai, hits in enumerate(nbr):
        if not hits:
            continue
        pre = int(axon_cell[ai])
        posts = dend_cell[hits]
        for po in posts:
            po = int(po)
            if po != pre:
                appo[(pre, po)] += 1
    print(f"[touch] 접촉쌍(pre,post) {len(appo):,} · 총 apposition {sum(appo.values()):,}")

    # 연결 = apposition ≥1. 시냅스 수 ≈ apposition 수.
    pairs = np.array(list(appo.keys()))
    syn = np.array(list(appo.values()))
    pre_c, post_c = pairs[:, 0], pairs[:, 1]
    # pathway_class 분류
    cls = np.array([pathway_class(mtypes_hyphen[p], mtypes_hyphen[q])
                    for p, q in zip(pre_c, post_c)], dtype=object)
    valid = cls != None                            # noqa: E711
    n_conn = len(pairs); n_conn_valid = int(valid.sum())
    n_syn = int(syn.sum())
    dist = np.linalg.norm(xyz[pre_c] - xyz[post_c], axis=1)
    print(f"[conn] 연결(pair) {n_conn:,} (pathway 유효 {n_conn_valid:,})  "
          f"시냅스≈{n_syn:,}  평균 {n_syn/n_conn:.1f} syn/연결")

    np.savez_compressed(os.path.join(HERE, "touch_connectivity.npz"),
                        pre=pre_c.astype(np.int32), post=post_c.astype(np.int32),
                        n_syn=syn.astype(np.int32), dist=dist.astype(np.float32))

    # 검증 집계
    indeg = Counter(post_c.tolist()); outdeg = Counter(pre_c.tolist())
    cls_cnt = Counter([c for c in cls if c is not None])
    summary = {
        "step": "7b touch-connectome (Romani 방식 축소판)",
        "subvolume_cells": int(len(sub)),
        "box_um": [2 * HALF_T1, 2 * HALF_T2, "full-depth"],
        "n_subsample_pts": N_SUB, "touch_radius_um": TOUCH_R,
        "axon_points": int(len(axon_pts)), "dend_points": int(len(dend_pts)),
        "apposition_pairs": n_conn, "total_appositions": n_syn,
        "connections_pathway_valid": n_conn_valid,
        "mean_syn_per_conn": float(n_syn / n_conn) if n_conn else 0.0,
        "mean_in_degree": float(np.mean(list(indeg.values()))) if indeg else 0.0,
        "mean_out_degree": float(np.mean(list(outdeg.values()))) if outdeg else 0.0,
        "by_class": {str(k): int(v) for k, v in cls_cnt.most_common()},
        "note": "축소부피·서브샘플·pruning 미적용. apposition≈잠재시냅스(과대추정 가능).",
    }
    json.dump(summary, open(os.path.join(HERE, "touch_connectivity_summary.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    _fig_syn_per_conn(syn)
    _fig_dist(dist)
    _fig_class(cls_cnt)
    print(f"[OK] touch_connectivity.npz + figures -> {FIG}")


def _fig_syn_per_conn(syn):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(syn, bins=np.arange(0.5, min(syn.max(), 40) + 1.5), color="#55A868")
    ax.set_xlabel("연결당 접촉(~시냅스) 수"); ax.set_ylabel("연결 수")
    ax.set_title(f"V3b-1  연결당 시냅스 수 분포 (평균 {syn.mean():.1f})\n"
                 "형태 접촉 기반 — 연결마다 여러 접촉점")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3b_1_syn_per_conn.png"), dpi=130)
    plt.close(fig)


def _fig_dist(dist):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dist, bins=40, color="#4C72B0")
    ax.axvline(np.median(dist), color="r", lw=2, label=f"중앙값 {np.median(dist):.0f}µm")
    ax.set_xlabel("연결된 소마 간 거리 (µm)"); ax.set_ylabel("연결 수")
    ax.set_title("V3b-2  형태접촉 연결의 소마간 거리 분포\n"
                 "(소마거리 규칙과 달리 정점수상 도달로 원거리 연결도 포함)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3b_2_distance.png"), dpi=130)
    plt.close(fig)


def _fig_class(cls_cnt):
    items = cls_cnt.most_common()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#DD8452" if "PC->" in k else "#4C72B0" for k, _ in items]
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
            color=colors[::-1])
    ax.set_xscale("log"); ax.set_xlabel("연결 수 (로그)")
    ax.set_title("V3b-3  형태접촉 커넥텀 — pathway 클래스별 연결 수")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3b_3_class_counts.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
