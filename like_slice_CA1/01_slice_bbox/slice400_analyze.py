# -*- coding: utf-8 -*-
"""
01_slice_bbox/slice400_analyze.py  —  단계 1을 채택 슬라이스 slice400 에 국한

slice400 내부에서:
  - 층두께 측정 (V1a 를 slice400 한정으로 재확인)
  - 층별 3D 형태 (회전 GIF)

산출:
  figures/V1a_slice400_thickness.png
  figures/V1a_slice400_layers_3d.gif   (3D → 회전 GIF)
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


def main():
    mask, h = nrrd.read(os.path.join(ATLAS, "nrrd_volumes", "slices", "slice400.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    br, _ = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    inside = mask > 0

    # 층두께 (slice400 복셀 한정, [PH] 상단-하단 평균)
    thick, cum = {}, 0.0
    bounds = {}
    print("=== slice400 내 층두께 ===")
    for L in LAYER_ORDER:
        d, _ = nrrd.read(os.path.join(ATLAS, f"[PH]{L}.nrrd"))
        th = (d[1] - d[0])[inside]
        th = th[np.isfinite(th)]
        t = float(np.mean(th)); thick[L] = t
        bounds[L] = (cum, cum + t); cum += t
        print(f"  {L:4s}: {t:6.1f} µm")
    print(f"  합계 {cum:.1f} µm")

    _fig_thickness(thick, bounds, cum)
    _fig_layers_3d(br, inside, origin, vsize)
    print(f"[OK] -> {FIG}")


def _fig_thickness(thick, bounds, total):
    fig, ax = plt.subplots(figsize=(4.5, 7))
    for L in LAYER_ORDER:
        lo, hi = bounds[L]
        ax.bar(0, hi - lo, bottom=lo, width=0.6, color=LAYER_COLOR[L], edgecolor="w")
        ax.text(0, (lo + hi) / 2, f"{L}\n{thick[L]:.0f} µm", ha="center",
                va="center", color="w", fontweight="bold")
    ax.set_ylim(0, total * 1.02); ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
    ax.set_ylabel("SO 바닥부터 깊이 (µm)")
    ax.set_title(f"V1a(slice400)  층두께\n합계 = {total:.0f} µm")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V1a_slice400_thickness.png"), dpi=130)
    plt.close(fig)


def _fig_layers_3d(br, inside, origin, vsize, n=9000):
    vox = np.argwhere(inside)
    if len(vox) > n:
        vox = vox[np.random.default_rng(0).choice(len(vox), n, False)]
    world = vox * vsize + origin
    lab = br[vox[:, 0], vox[:, 1], vox[:, 2]]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for L in LAYER_ORDER:
        m = lab == LAYER_ID[L]
        if m.any():
            ax.scatter(world[m, 0], world[m, 1], world[m, 2], s=3,
                       c=LAYER_COLOR[L], alpha=0.5, label=L)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(markerscale=3, fontsize=8)
    save_rotate_gif(fig, ax, os.path.join(FIG, "V1a_slice400_layers_3d.gif"),
                    title="V1a(slice400)  층별 3D 형태 (회전)")
    plt.close(fig)


if __name__ == "__main__":
    main()
