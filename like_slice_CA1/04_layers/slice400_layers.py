# -*- coding: utf-8 -*-
"""
04_layers/slice400_layers.py  —  단계 4를 채택 슬라이스 slice400 에 국한

slice400 내부에서:
  - 층별 정규화깊이 분포 (V1c 를 slice400 한정으로 재확인)
  - 정규화깊이 3D (회전 GIF, 연속 깊이색)

산출:
  figures/V1c_slice400_depth.png
  figures/V1c_slice400_depth_3d.gif   (3D → 회전 GIF)
"""
import os
import sys
import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _fig_cross(tcoord, ndv, lab):
    """펼친 슬라이스 층 단면: 가로 t 횡좌표, 세로 nd 깊이, 색=층."""
    ok = np.isfinite(ndv) & np.isfinite(tcoord)
    fig, ax = plt.subplots(figsize=(9, 6))
    for L in LAYER_ORDER:
        m = ok & (lab == LAYER_ID[L])
        if m.any():
            ax.scatter(tcoord[m], ndv[m], s=5, c=LAYER_COLOR[L], label=L)
    ax.set_xlabel("t 횡좌표 (ch1, 절편 폭 방향)")
    ax.set_ylabel("정규화 깊이 nd (0=SO 바닥, 1=SLM 천장)")
    ax.set_ylim(0, 1)
    ax.legend(markerscale=2)
    ax.set_title("V1c(slice400)  펼친 슬라이스 층 단면\n"
                 "깊이(nd)를 따라 SO→SP→SR→SLM 이 가로 띠로 깔끔히 층서")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1c_slice400_cross.png"), dpi=130)
    plt.close(fig)


def main():
    mask, h = nrrd.read(os.path.join(ATLAS, "nrrd_volumes", "slices", "slice400.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    br, _ = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    phy, _ = nrrd.read(os.path.join(ATLAS, "[PH]y.nrrd"))
    so, _ = nrrd.read(os.path.join(ATLAS, "[PH]SO.nrrd"))
    slm, _ = nrrd.read(os.path.join(ATLAS, "[PH]SLM.nrrd"))
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = (phy - so[0]) / (slm[1] - so[0])

    co, _ = nrrd.read(os.path.join(ATLAS, "coordinates.nrrd"))   # l/t/r
    vox = np.argwhere(mask > 0)
    lab = br[vox[:, 0], vox[:, 1], vox[:, 2]]
    ndv = nd[vox[:, 0], vox[:, 1], vox[:, 2]]
    tcoord = co[1, vox[:, 0], vox[:, 1], vox[:, 2]]   # 횡좌표 t

    # 펼친 슬라이스 층 단면 (가로 t, 세로 nd, 색=층)
    _fig_cross(tcoord, ndv, lab)

    # 층별 nd box
    data = [ndv[(lab == LAYER_ID[L]) & np.isfinite(ndv)] for L in LAYER_ORDER]
    print("=== slice400 내 층별 정규화깊이 평균 ===")
    for L, d in zip(LAYER_ORDER, data):
        print(f"  {L:4s}: {np.mean(d):.3f}")
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, vert=False, tick_labels=LAYER_ORDER,
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch, L in zip(bp["boxes"], LAYER_ORDER):
        patch.set_facecolor(LAYER_COLOR[L])
    ax.set_xlabel("정규화 깊이 (0 = SO 바닥, 1 = SLM 천장)"); ax.set_xlim(0, 1)
    ax.set_title("V1c(slice400)  층별 정규화깊이 분포")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1c_slice400_depth.png"), dpi=130)
    plt.close(fig)

    # 정규화깊이 3D GIF (연속 깊이색)
    n = 9000
    sel = np.random.default_rng(0).choice(len(vox), min(n, len(vox)), False)
    v = vox[sel]; world = v * vsize + origin; c = ndv[sel]
    ok = np.isfinite(c)
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(world[ok, 0], world[ok, 1], world[ok, 2], s=3,
                    c=c[ok], cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label="정규화 깊이", shrink=0.6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    save_rotate_gif(fig, ax, os.path.join(FIG, "V1c_slice400_depth_3d.gif"),
                    title="V1c(slice400)  정규화깊이 3D (회전, SO→SLM)")
    plt.close(fig)
    print(f"[OK] -> {FIG}")


if __name__ == "__main__":
    main()
