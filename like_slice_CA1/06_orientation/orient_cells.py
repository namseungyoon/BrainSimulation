# -*- coding: utf-8 -*-
"""
06_orientation/orient_cells.py  —  단계 6: 방향성 주입 (V2d)

목적:
  각 세포의 변이 형태(.swc)를 (1) 소마를 세포 좌표로 평행이동, (2) orientation
  quaternion 으로 회전(정점축 +Y → 방사방향)해 slice400 안에 배치.

검증 (V2d):
  - 길이 불변: 강체 변환이라 모든 구간 길이 보존 (총 수상 길이 비 = 1.0)
  - 소마 위치 == 목표 좌표
  - 정점축 ≈ orientation(방사방향): cos(정점방향_after, R·[0,1,0]) ≈ 1

산출 그림 (한글; 3D=회전 GIF):
  figures/V2d_1_single_cell.gif    : 배치된 추체 1개 (구획별 색) 회전
  figures/V2d_2_length_invariance.png : 표본 세포 길이 보존 비 히스토그램
  figures/V2d_3_placed_cells.gif   : 여러 세포 배치 미리보기 (층별 색) 회전
  figures/V2d_4_apical_align.png   : 정점축-방사방향 정렬 cos 분포
출력: (검증 요약 stdout/summary.json)
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lib"))
from gif_util import save_rotate_gif           # noqa: E402
import morph_transform as mt                   # noqa: E402

MORPH_DIR = os.path.join(ROOT, "data", "morphology_library")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
ASSIGN = os.path.join(ROOT, "05b_memap", "model_assignment.npz")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}
rng = np.random.default_rng(0)


def load_cells():
    c = np.load(CELLS, allow_pickle=True)
    a = np.load(ASSIGN, allow_pickle=True)
    return (c["xyz"].astype(float), c["quat_wxyz"].astype(float),
            c["mtype"].astype(str), c["layer"].astype(str),
            a["morphology"].astype(str))


def swc_path(name):
    return os.path.join(MORPH_DIR, name + ".swc")


def place(swc, quat, pos):
    soma_c = mt.soma_center(swc)
    world, R = mt.transform(swc["xyz"], soma_c, quat, pos)
    return world, R, soma_c


def main():
    xyz, quat, mtype, layer, morph = load_cells()
    N = len(xyz)
    print(f"[load] {N:,} cells")

    # ---- 그림 1: 배치된 추체 1개 (구획별 색) 회전 GIF ----
    i = int(np.where(mtype == "SP_PC")[0][0])
    swc = mt.load_swc(swc_path(morph[i]))
    world, R, soma_c = place(swc, quat[i], xyz[i])
    _fig_single(swc, world, xyz[i], R)

    # ---- 그림 2 & 4: 길이 불변 + 정점 정렬 (표본) ----
    idx = rng.choice(N, 60, replace=False)
    ratios, apical_cos = [], []
    for k in idx:
        try:
            s = mt.load_swc(swc_path(morph[k]))
        except FileNotFoundError:
            continue
        soma_c = mt.soma_center(s)
        w, R = mt.transform(s["xyz"], soma_c, quat[k], xyz[k])
        L0 = mt.segment_lengths(s["xyz"], s["parent"], s["id"]).sum()
        L1 = mt.segment_lengths(w, s["parent"], s["id"]).sum()
        ratios.append(L1 / L0 if L0 else 1.0)
        ap = mt.apical_direction(s, xyz=w)          # 변환 후 정점방향
        if ap is not None:
            radial = R.apply(np.array([0.0, 1.0, 0.0]))
            apical_cos.append(float(np.dot(ap, radial)))
    ratios = np.array(ratios); apical_cos = np.array(apical_cos)
    print(f"[V2d] 길이보존 비: 평균 {ratios.mean():.6f} (min {ratios.min():.6f}, "
          f"max {ratios.max():.6f})  — 1.0 이어야 강체")
    print(f"[V2d] 정점축-방사 cos: 평균 {apical_cos.mean():.4f} "
          f"(n={len(apical_cos)})")
    _fig_length(ratios)
    _fig_apical(apical_cos)

    # ---- 그림 3: 여러 세포 배치 미리보기 (층별) 회전 GIF ----
    _fig_placed(xyz, quat, mtype, layer, morph)

    json.dump({"step": "6 orientation (V2d)",
               "length_ratio_mean": float(ratios.mean()),
               "length_ratio_min": float(ratios.min()),
               "length_ratio_max": float(ratios.max()),
               "apical_radial_cos_mean": float(apical_cos.mean()),
               "n_sampled": len(idx)},
              open(os.path.join(HERE, "orient_summary.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] figures -> {FIG}")


def _sub(pts, n=2500):
    if len(pts) > n:
        return pts[rng.choice(len(pts), n, replace=False)]
    return pts


def _fig_single(swc, world, pos, R):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for t in [2, 3, 4, 1]:
        m = swc["type"] == t
        if m.any():
            p = world[m]
            ps = _sub(p, 1500 if t == 2 else 2500)
            ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=(6 if t == 1 else 1),
                       c=mt.TYPE_COLOR[t], alpha=0.5,
                       label=mt.TYPE_NAME[t])
    # 방사(정점) 방향 화살표
    radial = R.apply(np.array([0.0, 1.0, 0.0]))
    ax.quiver(pos[0], pos[1], pos[2], radial[0], radial[1], radial[2],
              length=180, color="red", linewidth=2, label="방사방향(정점축)")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    ax.legend(fontsize=8, markerscale=3)
    save_rotate_gif(fig, ax, os.path.join(FIG, "V2d_1_single_cell.gif"),
                    title="V2d-1  배치된 추체세포 1개 (구획별 색, 빨강=정점축) 회전")
    plt.close(fig)


def _fig_length(ratios):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ratios, bins=np.linspace(0.99, 1.01, 41), color="#55A868")
    ax.axvline(1.0, color="r", lw=2, label="이상값 1.0")
    ax.set_xlabel("변환 후/전 총 수상 길이 비"); ax.set_ylabel("세포 수")
    ax.set_xlim(0.99, 1.01)
    ax.set_title(f"V2d-2  길이 불변 검증 (강체 변환)\n"
                 f"평균 {ratios.mean():.6f} — 회전·평행이동은 길이 보존")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2d_2_length_invariance.png"), dpi=130)
    plt.close(fig)


def _fig_apical(cos):
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(cos):
        ax.hist(cos, bins=20, color="#4C72B0")
        ax.axvline(cos.mean(), color="r", lw=2, label=f"평균 {cos.mean():.3f}")
    ax.set_xlabel("cos(정점축_변환후, 방사방향 R·[0,1,0])")
    ax.set_ylabel("세포 수"); ax.set_xlim(0, 1.02)
    ax.set_title("V2d-4  정점축이 방사방향에 정렬되는가 (추체 표본)\n"
                 "1에 가까울수록 정점수상이 SO→SLM 깊이축과 일치")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2d_4_apical_align.png"), dpi=130)
    plt.close(fig)


def _fig_placed(xyz, quat, mtype, layer, morph, n_per_layer=3):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for L in LAYER_ORDER:
        cand = np.where(layer == L)[0]
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, min(n_per_layer, len(cand)), replace=False)
        for k in pick:
            try:
                s = mt.load_swc(swc_path(morph[k]))
            except FileNotFoundError:
                continue
            w, _ = mt.transform(s["xyz"], mt.soma_center(s), quat[k], xyz[k])
            # 수상+소마만 (축삭 제외해 가독)
            m = s["type"] != 2
            ps = _sub(w[m], 1200)
            ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=1, c=LAYER_COLOR[L],
                       alpha=0.5)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    ax.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER],
              fontsize=8)
    save_rotate_gif(fig, ax, os.path.join(FIG, "V2d_3_placed_cells.gif"),
                    title="V2d-3  slice400 세포 배치 미리보기 (층별 색, 축삭 제외) 회전")
    plt.close(fig)


if __name__ == "__main__":
    main()
