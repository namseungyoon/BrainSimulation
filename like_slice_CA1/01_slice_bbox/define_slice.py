# -*- coding: utf-8 -*-
"""
01_slice_bbox/define_slice.py  —  단계 1: 아틀라스 슬라이스 3D bbox 정의 (V1a)

목적:
  Romani atlas 를 불러와 like-slice 의 3D bounding box 를 정의한다.
  슬라이스 위치 = Romani 제공 표준 기준 cylinder300 (CA1 관통 300um 컬럼).
  그 bbox 안에서 SO/SP/SR/SLM 4개 층이 모두 관통되는지, 층두께가 얼마인지 검증.

검증 기준 (V1a, atlas 실측 확정):
  층두께 SO230 / SP80 / SR380 / SLM199 um (합 ~887), 누적경계 0->230->310->692->887.

입력:
  ../data/atlas/{brain_regions.nrrd, [PH]<layer>.nrrd, meshes/cylinder300_intersection.obj}
  ../data/circuit/.../nodes.h5  (슬라이스 위치 시각화용 세포 좌표)
출력:
  ./slice_bbox.json           (다음 단계가 읽는 bbox 정의)
  ./figures/V1a_*.png

실행:
  C:\\Users\\SYNAM-OFFICE\\.conda\\envs\\ca1sim\\python.exe 01_slice_bbox/define_slice.py
"""
import os
import json

import numpy as np
import nrrd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
NODES_H5 = os.path.join(ROOT, "data", "circuit", "networks", "nodes",
                        "hippocampus_neurons", "nodes.h5")
POP = "nodes/hippocampus_neurons/0"
FIG_DIR = os.path.join(HERE, "figures")
OUT_JSON = os.path.join(HERE, "slice_bbox.json")
os.makedirs(FIG_DIR, exist_ok=True)

LAYER_ID = {"SO": 1, "SP": 2, "SR": 3, "SLM": 4}     # brain_regions 라벨
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]               # 심부->표면
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}


def obj_vertices(path):
    """.obj 파일에서 정점 좌표(v)만 읽어 (N,3) 배열로."""
    vs = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    return np.asarray(vs)


def main():
    # --- 1) 슬라이스 위치: cylinder300 메쉬의 월드 bbox ---
    cyl = obj_vertices(os.path.join(ATLAS, "meshes",
                                    "cylinder300_intersection.obj"))
    lo = cyl.min(0)
    hi = cyl.max(0)
    center = (lo + hi) / 2.0
    print("=== 슬라이스 bbox (cylinder300, 월드 um) ===")
    for i, ax in enumerate("xyz"):
        print(f"  {ax}: [{lo[i]:8.1f}, {hi[i]:8.1f}]  span={hi[i]-lo[i]:7.1f}")
    print(f"  center = ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")

    # --- 2) atlas 격자 정보 ---
    br, h = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])         # 16.0 등방
    print(f"\n=== atlas 격자: shape={br.shape}, voxel={vsize}um, "
          f"origin={origin} ===")

    # bbox 에 해당하는 복셀 인덱스 마스크
    def axis_mask(n, oi, l, hh):
        coord = np.arange(n) * vsize + oi
        return (coord >= l) & (coord <= hh)
    mx = axis_mask(br.shape[0], origin[0], lo[0], hi[0])
    my = axis_mask(br.shape[1], origin[1], lo[1], hi[1])
    mz = axis_mask(br.shape[2], origin[2], lo[2], hi[2])
    sel = np.ix_(mx, my, mz)
    br_slice = br[sel]
    print(f"  bbox 내 복셀: {mx.sum()} x {my.sum()} x {mz.sum()} "
          f"= {br_slice.size:,}  (CA1 복셀 {int((br_slice>0).sum()):,})")

    # --- 3) V1a: bbox 내 층두께 (atlas 실측) ---
    thick, bounds = {}, {}
    cum = 0.0
    print("\n=== V1a 층두께 (atlas 실측, bbox 내 평균) ===")
    for L in LAYER_ORDER:
        d, _ = nrrd.read(os.path.join(ATLAS, f"[PH]{L}.nrrd"))   # (2,X,Y,Z)
        th = (d[1] - d[0])[sel]
        th = th[np.isfinite(th)]
        t = float(np.mean(th))
        thick[L] = t
        bounds[L] = (cum, cum + t)
        cum += t
        nvox = int((br_slice == LAYER_ID[L]).sum())
        print(f"  {L:4s}: {t:6.1f} um   경계 {bounds[L][0]:6.1f}->"
              f"{bounds[L][1]:6.1f}   (복셀 {nvox:,})")
    print(f"  전층 합계 = {cum:.1f} um")

    # --- 4) slice_bbox.json 저장 ---
    out = {
        "source": "cylinder300_intersection (Romani 표준 300um 기준 컬럼)",
        "bbox_world_um": {"min": lo.tolist(), "max": hi.tolist(),
                          "center": center.tolist(),
                          "span": (hi - lo).tolist()},
        "atlas": {"voxel_um": vsize, "origin": origin.tolist(),
                  "grid_shape": list(br.shape)},
        "layers": {L: {"thickness_um": thick[L],
                       "bounds_um": list(bounds[L]),
                       "label": LAYER_ID[L]} for L in LAYER_ORDER},
        "total_thickness_um": cum,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] slice 정의 저장 -> {OUT_JSON}")

    # --- 5) 그림 ---
    _fig_thickness(thick, bounds, cum)
    _fig_cross_section(br_slice, vsize)
    _fig_location(lo, hi, center)
    print(f"[OK] 그림 저장 -> {FIG_DIR}/V1a_*.png")


def _fig_thickness(thick, bounds, total):
    """누적 막대 (심부 SO 아래 -> 표면 SLM 위)."""
    fig, ax = plt.subplots(figsize=(4.5, 7))
    for L in LAYER_ORDER:
        lo_b, hi_b = bounds[L]
        ax.bar(0, hi_b - lo_b, bottom=lo_b, width=0.6,
               color=LAYER_COLOR[L], edgecolor="w")
        ax.text(0, (lo_b + hi_b) / 2,
                f"{L}\n{thick[L]:.0f} um", ha="center", va="center",
                fontsize=11, color="w", fontweight="bold")
    ax.set_ylim(0, total * 1.02)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("SO 바닥부터의 깊이 (µm)")
    ax.set_title(f"V1a  층 두께 (atlas)\n합계 = {total:.0f} µm")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V1a_layer_thickness.png"), dpi=130)
    plt.close(fig)


def _fig_cross_section(br_slice, vsize):
    """bbox 중앙을 가르는 2개 단면(브레인리전 라벨)으로 층 적층 확인."""
    nx, ny, nz = br_slice.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = matplotlib.colors.ListedColormap(
        ["white"] + [LAYER_COLOR[L] for L in LAYER_ORDER])
    # x 중앙 단면 (y-z 평면)
    axes[0].imshow(br_slice[nx // 2].T, origin="lower", cmap=cmap,
                   vmin=0, vmax=4, aspect="auto")
    axes[0].set_title("단면 @ x-중앙  (y vs z)")
    axes[0].set_xlabel("y 복셀"); axes[0].set_ylabel("z 복셀")
    # z 중앙 단면 (x-y 평면)
    axes[1].imshow(br_slice[:, :, nz // 2].T, origin="lower", cmap=cmap,
                   vmin=0, vmax=4, aspect="auto")
    axes[1].set_title("단면 @ z-중앙  (x vs y)")
    axes[1].set_xlabel("x 복셀"); axes[1].set_ylabel("y 복셀")
    handles = [Rectangle((0, 0), 1, 1, color=LAYER_COLOR[L]) for L in LAYER_ORDER]
    axes[1].legend(handles, LAYER_ORDER, loc="upper right", fontsize=8)
    fig.suptitle("V1a  슬라이스 단면 (brain_regions 층 라벨)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V1a_cross_section.png"), dpi=130)
    plt.close(fig)


def _fig_location(lo, hi, center, n_sample=40000):
    """전체 CA1 세포 top-view(x-z)에 슬라이스 bbox 위치 표시."""
    with h5py.File(NODES_H5, "r") as f:
        g = f[POP]
        N = g["x"].shape[0]
        idx = np.random.default_rng(0).choice(N, min(n_sample, N), False)
        idx.sort()
        x = g["x"][:][idx]; z = g["z"][:][idx]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, z, s=1, alpha=0.25, c="#bbbbbb")
    ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2],
                           fill=False, edgecolor="red", lw=2.5))
    ax.plot(center[0], center[2], "r*", ms=16)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
    ax.set_title("V1a  CA1 내 슬라이스 위치 (위에서 본 x-z)\n"
                 "빨간 박스 = cylinder300 슬라이스 bbox")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V1a_slice_location.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
