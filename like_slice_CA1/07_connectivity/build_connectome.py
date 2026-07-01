# -*- coding: utf-8 -*-
"""
07_connectivity/build_connectome.py  —  단계 7: 커넥텀 (V3)

목적:
  slice400 세포 간 연결(edges)을 생성.
    - 방향쌍(pre→post) 을 Ecker pathway_class 로 9클래스 중 하나로 분류 (None=연결없음)
    - 거리의존 확률: P(d) = P0(클래스타입) · exp(-d² / 2σ²), d=소마간 거리
  edges{pre, post, cls, dist} 저장.

검증 (V3, 철저):
  - 모든 edge 가 9클래스 안에 속함 (None 없음)
  - 수렴(in-degree)·발산(out-degree) 분포가 타당 (세포유형별)
  - 연결확률이 거리에 따라 감소 (측정 vs 이론 exp)
  - m-type×m-type 연결수 행렬 · pathway_class 배정 행렬 · E/I 비율

파라미터: σ=100µm, R=250µm, P0(E)=0.03, P0(I)=0.06 (거리0 최대확률).
난수 시드 고정.

산출: slice_connectivity.npz(+summary.json), figures/V3_*.png
"""
import os
import sys
import json
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "papers",
                                "01_Ecker2020_CA1_synaptic", "05_paired_recording"))
from pathway_map import pathway_class            # noqa: E402

CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SIGMA = 100.0        # 거리 감쇠 (µm)
RADIUS = 250.0       # 최대 연결거리
P0 = {"E": 0.03, "I": 0.06}
SEED = 0


def main():
    d = np.load(CELLS, allow_pickle=True)
    xyz = d["xyz"].astype(float)
    mt_us = d["mtype"].astype(str)                    # SP_PC ...
    sclass = d["sclass"].astype(str)
    N = len(xyz)
    mt = np.array([m.replace("_", "-") for m in mt_us])  # SP-PC ...
    mtypes = sorted(set(mt))
    code = {m: i for i, m in enumerate(mtypes)}
    ccode = np.array([code[m] for m in mt])
    print(f"[load] {N:,} cells, m-type {len(mtypes)}종")

    # 12×12 클래스 배정 행렬 + 9클래스 인덱스
    classes = []
    for p in mtypes:
        for q in mtypes:
            c = pathway_class(p, q)
            if c and c not in classes:
                classes.append(c)
    classes = sorted(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    CLASS_ID = -np.ones((len(mtypes), len(mtypes)), int)
    P0MAT = np.zeros((len(mtypes), len(mtypes)))
    for pi, p in enumerate(mtypes):
        for qi, q in enumerate(mtypes):
            c = pathway_class(p, q)
            if c:
                CLASS_ID[pi, qi] = cls_idx[c]
                P0MAT[pi, qi] = P0["E"] if p == "SP-PC" else P0["I"]
    print(f"[classes] {n_cls}종: {classes}")

    # 후보쌍 (거리 R 이내, 무방향) → 양방향 처리
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(RADIUS, output_type="ndarray")   # (M,2), a<b
    a, b = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(xyz[a] - xyz[b], axis=1)
    w = np.exp(-dist**2 / (2 * SIGMA**2))
    rng = np.random.default_rng(SEED)
    print(f"[pairs] 후보 무방향쌍 {len(pairs):,} (R={RADIUS}µm)")

    edges_pre, edges_post, edges_cls, edges_d = [], [], [], []
    for (src, dst) in ((a, b), (b, a)):               # 두 방향
        cid = CLASS_ID[ccode[src], ccode[dst]]
        p0 = P0MAT[ccode[src], ccode[dst]]
        P = p0 * w
        hit = (cid >= 0) & (rng.random(len(src)) < P)
        edges_pre.append(src[hit]); edges_post.append(dst[hit])
        edges_cls.append(cid[hit]); edges_d.append(dist[hit])
    pre = np.concatenate(edges_pre); post = np.concatenate(edges_post)
    ecls = np.concatenate(edges_cls); ed = np.concatenate(edges_d)
    E = len(pre)
    print(f"[edges] 생성 {E:,}개  (평균 out-degree {E/N:.1f})")

    # 저장
    np.savez_compressed(os.path.join(HERE, "slice_connectivity.npz"),
                        pre=pre.astype(np.int32), post=post.astype(np.int32),
                        cls=ecls.astype(np.int8), dist=ed.astype(np.float32),
                        classes=np.array(classes, dtype="U20"))

    # ---- 검증 집계 ----
    indeg = np.bincount(post, minlength=N)
    outdeg = np.bincount(pre, minlength=N)
    is_pc = mt == "SP-PC"
    cls_cnt = Counter(ecls.tolist())
    none_edges = int((ecls < 0).sum())
    print(f"[V3] None(미분류) edge = {none_edges} (0 이어야 함)")
    print(f"[V3] in-degree 평균 {indeg.mean():.1f} (PC {indeg[is_pc].mean():.1f} / "
          f"INT {indeg[~is_pc].mean():.1f})")
    print(f"[V3] out-degree 평균 {outdeg.mean():.1f} (PC {outdeg[is_pc].mean():.1f} / "
          f"INT {outdeg[~is_pc].mean():.1f})")

    # E/I edge
    pre_is_e = mt[pre] == "SP-PC"
    n_e = int(pre_is_e.sum()); n_i = E - n_e
    summary = {
        "step": "7 connectivity (V3)", "n_cells": N, "n_edges": E,
        "sigma_um": SIGMA, "radius_um": RADIUS, "P0": P0,
        "n_classes": n_cls, "classes": classes,
        "none_edges": none_edges,
        "mean_in_degree": float(indeg.mean()),
        "mean_out_degree": float(outdeg.mean()),
        "in_degree_PC": float(indeg[is_pc].mean()), "in_degree_INT": float(indeg[~is_pc].mean()),
        "edges_by_class": {classes[i]: int(cls_cnt.get(i, 0)) for i in range(n_cls)},
        "excitatory_edges": n_e, "inhibitory_edges": n_i,
        "EI_edge_ratio": f"{100*n_e/E:.1f}:{100*n_i/E:.1f}",
    }
    json.dump(summary, open(os.path.join(HERE, "connectivity_summary.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    _fig_class_counts(classes, cls_cnt)
    _fig_degree(indeg, outdeg, is_pc)
    _fig_dist_prob(dist, a, b, ccode, CLASS_ID, w, pre, post, ed)
    _fig_conn_matrix(mtypes, ccode, pre, post, N)
    _fig_pathway_matrix(mtypes, CLASS_ID, classes)
    print(f"[OK] slice_connectivity.npz + figures -> {FIG}")


def _fig_class_counts(classes, cls_cnt):
    vals = [cls_cnt.get(i, 0) for i in range(len(classes))]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#DD8452" if "PC->" in classes[i] else "#4C72B0" for i in order]
    ax.barh([classes[i] for i in order], [vals[i] for i in order], color=colors)
    ax.set_xscale("log"); ax.set_xlabel("edge 수 (로그)")
    for k, i in enumerate(order):
        ax.text(vals[i], k, f" {vals[i]:,}", va="center", fontsize=8)
    ax.set_title("V3-1  9 pathway 클래스별 연결(edge) 수\n주황=흥분(PC 출력) · 파랑=억제(인터뉴런 출력)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3_1_class_counts.png"), dpi=130)
    plt.close(fig)


def _fig_degree(indeg, outdeg, is_pc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, deg, name in zip(axes, [indeg, outdeg], ["수렴 in-degree", "발산 out-degree"]):
        ax.hist(deg[is_pc], bins=40, alpha=0.6, color="#DD8452", label="추체(PC)")
        ax.hist(deg[~is_pc], bins=40, alpha=0.6, color="#4C72B0", label="인터뉴런")
        ax.set_xlabel(f"{name} (연결 수)"); ax.set_ylabel("세포 수")
        ax.axvline(deg.mean(), color="k", ls="--", label=f"전체평균 {deg.mean():.0f}")
        ax.legend(fontsize=8)
    fig.suptitle("V3-2  수렴(in)·발산(out) 차수 분포 (세포유형별)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3_2_degree.png"), dpi=130)
    plt.close(fig)


def _fig_dist_prob(dist, a, b, ccode, CLASS_ID, w, pre, post, ed):
    """거리 bin별 연결확률 (연결/후보) vs 이론 exp."""
    # 후보쌍(양방향 중 클래스 유효한 것)의 거리 분포와 연결된 거리 분포로 확률 추정
    bins = np.linspace(0, RADIUS, 26)
    # 후보(유효클래스) 거리: 두 방향 각각
    valid_d = []
    for (src, dst) in ((a, b), (b, a)):
        cid = CLASS_ID[ccode[src], ccode[dst]]
        valid_d.append(dist[cid >= 0])
    valid_d = np.concatenate(valid_d)
    h_cand, _ = np.histogram(valid_d, bins=bins)
    h_conn, _ = np.histogram(ed, bins=bins)
    ctr = (bins[:-1] + bins[1:]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = np.where(h_cand > 0, h_conn / h_cand, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ctr, prob, "o-", color="#55A868", label="측정 연결확률")
    # 이론: 평균 P0 * exp(-d²/2σ²); 대략 확인용 (E/I 혼합이라 스케일만)
    ref = np.nanmax(prob) * np.exp(-ctr**2 / (2 * SIGMA**2)) / np.exp(0)
    ax.plot(ctr, ref, "--", color="gray", label=f"exp(-d²/2σ²), σ={SIGMA:.0f}")
    ax.set_xlabel("소마 간 거리 (µm)"); ax.set_ylabel("연결확률")
    ax.set_title("V3-3  거리의존 연결확률 (측정 vs 이론 감쇠)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3_3_conn_prob_distance.png"), dpi=130)
    plt.close(fig)


def _fig_conn_matrix(mtypes, ccode, pre, post, N):
    M = np.zeros((len(mtypes), len(mtypes)))
    np.add.at(M, (ccode[pre], ccode[post]), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = np.log10(M + 1)
    im = ax.imshow(disp, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(mtypes))); ax.set_xticklabels(mtypes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mtypes))); ax.set_yticklabels(mtypes, fontsize=7)
    ax.set_xlabel("post m-type"); ax.set_ylabel("pre m-type")
    fig.colorbar(im, ax=ax, label="log10(edge+1)")
    ax.set_title("V3-4  m-type × m-type 연결수 행렬 (pre→post)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3_4_conn_matrix.png"), dpi=130)
    plt.close(fig)


def _fig_pathway_matrix(mtypes, CLASS_ID, classes):
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab10", max(len(classes), 1))
    disp = np.where(CLASS_ID >= 0, CLASS_ID, np.nan)
    im = ax.imshow(disp, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, aspect="auto")
    ax.set_xticks(range(len(mtypes))); ax.set_xticklabels(mtypes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mtypes))); ax.set_yticklabels(mtypes, fontsize=7)
    ax.set_xlabel("post m-type"); ax.set_ylabel("pre m-type")
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(classes)))
    cbar.ax.set_yticklabels(classes, fontsize=7)
    ax.set_title("V3-5  pathway_class 배정 (pre→post → 9클래스; 흰칸=연결없음)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V3_5_pathway_matrix.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
