# -*- coding: utf-8 -*-
"""
03_vectorize/vectorize.py  —  단계 3: 좌표/방향 벡터화 (V1b)

목적:
  Romani atlas 의 방향장(orientation.nrrd, quaternion)과 좌표장(coordinates.nrrd, l/t/r)을
  불러와(load) → 방사(radial) 벡터장으로 전처리(preprocess)한다.
  방사벡터 = 각 복셀 quaternion 으로 local Y축 [0,1,0] 을 회전한 월드방향.
  (BBP orientation 은 scalar-first (w,x,y,z); scipy 는 scalar-last (x,y,z,w) → 순서변환 필수)

검증 (V1b): 방사벡터가 층(SO/SP/SR/SLM)에 수직이고 SO→SLM(깊이증가) 방향.
  - 정량: cos(radial, ∇[PH]y) 분포
  - 정성: 층 단면에 방사벡터 quiver 가 층경계에 수직

산출 그림 (하나하나 확인용):
  figures/V1b_1_load_coordinates.png   : 좌표장 l/t/r 3채널 불러오기 확인
  figures/V1b_2_load_orientation.png   : 방향장 quaternion 성분 불러오기 확인
  figures/V1b_3_radial_quiver.png      : (전처리) 층 단면 + 방사벡터 수직성 (V1b 핵심)
  figures/V1b_4_radial_3d.png          : 방사벡터 3D (층별 색)
  figures/V1b_5_alignment_hist.png     : cos(radial, ∇depth) 정렬 히스토그램
출력:
  radial_field_summary.json

실행:
  python 03_vectorize/vectorize.py
"""
import os
import json
import numpy as np
import nrrd
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_ID = {"SO": 1, "SP": 2, "SR": 3, "SLM": 4}
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}


def load():
    br, h = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    ori, _ = nrrd.read(os.path.join(ATLAS, "orientation.nrrd"))   # (4,X,Y,Z) w,x,y,z
    co, _ = nrrd.read(os.path.join(ATLAS, "coordinates.nrrd"))    # (3,X,Y,Z)
    phy, _ = nrrd.read(os.path.join(ATLAS, "[PH]y.nrrd"))         # depth SO->SLM
    return br, origin, vsize, ori, co, phy


def radial_at(ori, idx):
    """idx=(N,3) 복셀인덱스 -> 방사벡터(N,3). w,x,y,z -> scipy x,y,z,w, R·[0,1,0]."""
    q = ori[:, idx[:, 0], idx[:, 1], idx[:, 2]].T       # (N,4) w,x,y,z
    q_scipy = q[:, [1, 2, 3, 0]]
    return Rot.from_quat(q_scipy).apply(np.array([0.0, 1.0, 0.0]))


# ---------------------------------------------------------------- 그림 1
def fig_load_coordinates(br, co):
    """좌표장 l/t/r 3채널을 한 단면(z-mid)으로 불러오기 확인."""
    nx, ny, nz = br.shape
    k = nz // 2
    mask = br[:, :, k] > 0
    titles = ["coord ch0 (tangential / longitudinal?)",
              "coord ch1 (tangential / transverse?)",
              "coord ch2 (RADIAL / depth, corr0.91 vs [PH]y)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c in range(3):
        img = np.where(mask, co[c, :, :, k], np.nan)
        im = axes[c].imshow(img.T, origin="lower", cmap="viridis",
                            vmin=0, vmax=1, aspect="auto")
        axes[c].set_title(titles[c], fontsize=10)
        axes[c].set_xlabel("x voxel"); axes[c].set_ylabel("y voxel")
        fig.colorbar(im, ax=axes[c], fraction=0.046)
    fig.suptitle("V1b-1  LOAD coordinates.nrrd (l/t/r, normalized 0~1)  @ z-mid")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1b_1_load_coordinates.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 그림 2
def fig_load_orientation(br, ori):
    """방향장 quaternion 4성분(w,x,y,z) 단면 불러오기 확인."""
    nx, ny, nz = br.shape
    k = nz // 2
    mask = br[:, :, k] > 0
    names = ["w", "x", "y (=0)", "z"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for c in range(4):
        img = np.where(mask, ori[c, :, :, k], np.nan)
        im = axes[c].imshow(img.T, origin="lower", cmap="coolwarm",
                            vmin=-1, vmax=1, aspect="auto")
        axes[c].set_title(f"quaternion {names[c]}", fontsize=10)
        axes[c].set_xlabel("x voxel"); axes[c].set_ylabel("y voxel")
        fig.colorbar(im, ax=axes[c], fraction=0.046)
    fig.suptitle("V1b-2  LOAD orientation.nrrd (quaternion w,x,y,z; BBP scalar-first)  @ z-mid")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1b_2_load_orientation.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 그림 3
def fig_radial_quiver(br, origin, vsize, ori):
    """한 x-단면(y-z 평면)에서 층 배경 + 방사벡터(quiver) 수직성 확인 (V1b 핵심)."""
    nx, ny, nz = br.shape
    # 층이 잘 보이는 단면: cylinder300 중심 부근 x-index
    cx = int((2642.7 - origin[0]) / vsize)
    plane = br[cx]                      # (ny,nz) layer labels
    ys, zs = np.where(plane > 0)
    # 방사벡터: 이 평면 복셀들
    idx = np.stack([np.full_like(ys, cx), ys, zs], 1)
    rad = radial_at(ori, idx)           # world (N,3)
    # y-z 평면으로 투영 (월드 y=축1, z=축2)
    u = rad[:, 1]; w = rad[:, 2]

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = matplotlib.colors.ListedColormap(
        ["white"] + [LAYER_COLOR[L] for L in LAYER_ORDER])
    ax.imshow(plane.T, origin="lower", cmap=cmap, vmin=0, vmax=4, aspect="equal")
    step = max(1, len(ys) // 700)       # quiver 서브샘플
    ax.quiver(ys[::step], zs[::step], u[::step], w[::step],
              color="k", scale=30, width=0.003, alpha=0.8)
    handles = [Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel("y voxel"); ax.set_ylabel("z voxel")
    ax.set_title(f"V1b-3  PREPROCESS radial vectors @ x-index {cx}\n"
                 "(arrows = R·[0,1,0], should be perpendicular to layer bands, SO->SLM)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1b_3_radial_quiver.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 그림 4
def fig_radial_3d(br, origin, vsize, ori, n=3000):
    ca1 = np.argwhere(br > 0)
    sel = ca1[np.random.default_rng(0).choice(len(ca1), n, False)]
    rad = radial_at(ori, sel)
    world = sel * vsize + origin
    lab = br[sel[:, 0], sel[:, 1], sel[:, 2]]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for L in LAYER_ORDER:
        m = lab == LAYER_ID[L]
        ax.quiver(world[m, 0], world[m, 1], world[m, 2],
                  rad[m, 0], rad[m, 1], rad[m, 2], length=60,
                  color=LAYER_COLOR[L], alpha=0.5, linewidth=0.6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("V1b-4  radial vector field 3D (color = layer)")
    handles = [Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1b_4_radial_3d.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 그림 5 + 정량
def fig_alignment(br, ori, phy, n=8000):
    ca1 = np.argwhere(br > 0)
    sel = ca1[np.random.default_rng(1).choice(len(ca1), n, False)]
    rad = radial_at(ori, sel)
    gx, gy, gz = np.gradient(phy)
    g = np.stack([gx[sel[:, 0], sel[:, 1], sel[:, 2]],
                  gy[sel[:, 0], sel[:, 1], sel[:, 2]],
                  gz[sel[:, 0], sel[:, 1], sel[:, 2]]], 1)
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    cos = np.einsum("ij,ij->i", rad, gn)
    cos = cos[np.isfinite(cos)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(cos, bins=60, color="#4C72B0", alpha=0.85)
    ax.axvline(np.median(cos), color="r", lw=2,
               label=f"median={np.median(cos):.3f}")
    ax.set_xlabel("cos( radial , ∇[PH]y )"); ax.set_ylabel("voxels")
    ax.set_title("V1b-5  radial perp. to layers & SO->SLM  "
                 f"(mean={np.mean(cos):.3f}, >0: {np.mean(cos>0)*100:.0f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1b_5_alignment_hist.png"), dpi=130)
    plt.close(fig)
    return float(np.mean(cos)), float(np.median(cos)), float(np.mean(cos > 0))


def main():
    print("[load] atlas ...")
    br, origin, vsize, ori, co, phy = load()
    print(f"  grid {br.shape}, CA1 voxels {int((br>0).sum()):,}")

    print("[fig] 1 load coordinates"); fig_load_coordinates(br, co)
    print("[fig] 2 load orientation"); fig_load_orientation(br, ori)
    print("[fig] 3 radial quiver (V1b)"); fig_radial_quiver(br, origin, vsize, ori)
    print("[fig] 4 radial 3D"); fig_radial_3d(br, origin, vsize, ori)
    print("[fig] 5 alignment hist"); mean, med, pos = fig_alignment(br, ori, phy)

    summary = {
        "step": "3 vectorize (V1b)",
        "radial_def": "R(quaternion w,x,y,z -> scipy x,y,z,w) applied to local Y [0,1,0]",
        "coordinates_channels": {"ch0": "tangential", "ch1": "tangential",
                                  "ch2": "radial/depth (corr 0.91 vs [PH]y)"},
        "V1b_alignment_cos": {"mean": mean, "median": med, "fraction_positive": pos},
        "note": "radial vector perpendicular to layers and points SO->SLM",
    }
    with open(os.path.join(HERE, "radial_field_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[V1b] cos mean={mean:.3f} median={med:.3f} positive={pos*100:.1f}%")
    print(f"[OK] figures -> {FIG}")


if __name__ == "__main__":
    main()
