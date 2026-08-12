# -*- coding: utf-8 -*-
"""
03_vectorize/slice400_radial.py  —  단계 3을 채택 슬라이스 slice400 에 국한

slice400 내부에서:
  - 방사벡터 정렬도 히스토그램 (V1b 를 slice400 한정으로 재확인)
  - 방사벡터 3D (회전 GIF)

산출:
  figures/V1b_slice400_align.png
  figures/V1b_slice400_radial_3d.gif   (3D → 회전 GIF)
"""
import os
import sys
import numpy as np
import nrrd
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gif_util import save_rotate_gif  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_ID = {"SO": 1, "SP": 2, "SR": 3, "SLM": 4}
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}


def radial_at(ori, idx):
    q = ori[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    return Rot.from_quat(q[:, [1, 2, 3, 0]]).apply(np.array([0.0, 1.0, 0.0]))


def _fig_coords(vox, co):
    """slice400 국한 l/t/r 좌표장. 펼친 슬라이스(가로 t=ch1, 세로 r=ch2)에 3채널 색."""
    l = co[0, vox[:, 0], vox[:, 1], vox[:, 2]]   # 종축 longitudinal
    t = co[1, vox[:, 0], vox[:, 1], vox[:, 2]]   # 횡축 transverse
    r = co[2, vox[:, 0], vox[:, 1], vox[:, 2]]   # 방사 radial(깊이)
    labels = [("l 종축(longitudinal)", l), ("t 횡축(transverse)", t),
              ("r 방사(radial/깊이)", r)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, val) in zip(axes, labels):
        sc = ax.scatter(t, r, c=val, s=4, cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("t 횡좌표 (ch1)"); ax.set_ylabel("r 깊이좌표 (ch2)")
        ax.set_title(f"색 = {name}")
        fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle("V1b(slice400)  좌표장 l/t/r — 펼친 슬라이스(가로 t · 세로 r)\n"
                 "l(종축)은 절편 내 거의 일정, t·r로 절편 면이 정의됨")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1b_slice400_coords.png"), dpi=130)
    plt.close(fig)


def main():
    mask, h = nrrd.read(os.path.join(ATLAS, "nrrd_volumes", "slices", "slice400.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    br, _ = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    ori, _ = nrrd.read(os.path.join(ATLAS, "orientation.nrrd"))
    phy, _ = nrrd.read(os.path.join(ATLAS, "[PH]y.nrrd"))
    co, _ = nrrd.read(os.path.join(ATLAS, "coordinates.nrrd"))    # l/t/r 좌표장
    vox = np.argwhere(mask > 0)
    radial = radial_at(ori, vox)

    # 좌표장 l/t/r 그림 (slice400 국한, 펼친 슬라이스: 횡 t vs 깊이 r)
    _fig_coords(vox, co)

    # 정렬도
    gx, gy, gz = np.gradient(phy)
    g = np.stack([gx[vox[:, 0], vox[:, 1], vox[:, 2]],
                  gy[vox[:, 0], vox[:, 1], vox[:, 2]],
                  gz[vox[:, 0], vox[:, 1], vox[:, 2]]], 1)
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    cos = np.einsum("ij,ij->i", radial, gn)
    cos = cos[np.isfinite(cos)]
    print(f"[V1b/slice400] cos 평균={np.mean(cos):.3f} 중앙값={np.median(cos):.3f} "
          f">0={np.mean(cos>0)*100:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(cos, bins=50, color="#4C72B0", alpha=0.85)
    ax.axvline(np.median(cos), color="r", lw=2, label=f"중앙값={np.median(cos):.3f}")
    ax.set_xlabel("cos( 방사벡터 , ∇[PH]y )"); ax.set_ylabel("복셀 수")
    ax.set_title(f"V1b(slice400)  방사벡터 수직·SO→SLM 정렬  "
                 f"(평균={np.mean(cos):.3f}, >0:{np.mean(cos>0)*100:.0f}%)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1b_slice400_align.png"), dpi=130)
    plt.close(fig)

    # 방사벡터 3D GIF
    n = 4000
    sel = np.random.default_rng(0).choice(len(vox), min(n, len(vox)), False)
    v = vox[sel]; world = v * vsize + origin; rad = radial[sel]
    lab = br[v[:, 0], v[:, 1], v[:, 2]]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for L in LAYER_ORDER:
        m = lab == LAYER_ID[L]
        if m.any():
            ax.quiver(world[m, 0], world[m, 1], world[m, 2],
                      rad[m, 0], rad[m, 1], rad[m, 2], length=45,
                      color=LAYER_COLOR[L], alpha=0.6, linewidth=0.6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER],
              fontsize=8)
    save_rotate_gif(fig, ax, os.path.join(FIG, "V1b_slice400_radial_3d.gif"),
                    title="V1b(slice400)  방사벡터장 3D (회전)")
    plt.close(fig)
    print(f"[OK] -> {FIG}")


if __name__ == "__main__":
    main()
