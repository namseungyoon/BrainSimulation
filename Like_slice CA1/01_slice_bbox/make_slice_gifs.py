# -*- coding: utf-8 -*-
"""
01_slice_bbox/make_slice_gifs.py  —  atlas 슬라이스 48개를 360도 회전 GIF로 저장

각 슬라이스의 3D voxel(층별 색)을 azimuth 0->360 회전시키며 GIF 로 저장.
  - slice400  + slice0~46  = 48개
출력: ./figures/slices_gif/sliceXXX.gif

옵션:
  --only NAME   : 특정 슬라이스 하나만 (예: --only slice400)  테스트용

실행:
  python 01_slice_bbox/make_slice_gifs.py
  python 01_slice_bbox/make_slice_gifs.py --only slice400
"""
import os
import sys
import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
SLICE_DIR = os.path.join(ATLAS, "nrrd_volumes", "slices")
SERIAL_DIR = os.path.join(SLICE_DIR, "nrrd_masks_in_user_target")
OUT = os.path.join(HERE, "figures", "slices_gif")
os.makedirs(OUT, exist_ok=True)

LAYER_ID = {"SO": 1, "SP": 2, "SR": 3, "SLM": 4}
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}
N_SERIAL = 47
N_FRAMES = 36          # 10도 간격
N_POINTS = 4000        # 프레임당 산점 점 수 (속도/용량 균형)


def load_grid():
    br, h = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    return br, origin, vsize


def make_gif(name, mask_path, br, origin, vsize):
    d, _ = nrrd.read(mask_path)
    vox = np.argwhere(d > 0)
    if len(vox) == 0:
        print(f"  {name}: empty, skip"); return
    labels = br[vox[:, 0], vox[:, 1], vox[:, 2]]
    world = vox.astype(float) * vsize + origin
    if len(world) > N_POINTS:
        s = np.random.default_rng(0).choice(len(world), N_POINTS, False)
        world, labels = world[s], labels[s]

    # 축 범위 고정 (회전 중 흔들림 방지) + 등축비
    mn, mx = world.min(0), world.max(0)
    ctr = (mn + mx) / 2
    rad = (mx - mn).max() / 2 * 1.05

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for L in LAYER_ID:
        m = labels == LAYER_ID[L]
        if m.any():
            ax.scatter(world[m, 0], world[m, 1], world[m, 2],
                       s=3, c=LAYER_COLOR[L], alpha=0.6, label=L)
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.set_box_aspect([1, 1, 1])     # xyz 1:1:1 (왜곡 없이 방향 일치)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper right", fontsize=8)
    span = mx - mn
    ax.set_title(f"{name}  ({len(vox):,} vox, "
                 f"{span[0]:.0f}x{span[1]:.0f}x{span[2]:.0f} um)", fontsize=10)

    frames = []
    for k in range(N_FRAMES):
        ax.view_init(elev=20, azim=k * (360 / N_FRAMES))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))
    plt.close(fig)

    out = os.path.join(OUT, f"{name}.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True)
    print(f"  {name}: {len(vox):,} vox -> {os.path.basename(out)}")


def main():
    br, origin, vsize = load_grid()
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    targets = [("slice400", os.path.join(SLICE_DIR, "slice400.nrrd"))]
    targets += [(f"slice{i}", os.path.join(SERIAL_DIR, f"slice{i}.nrrd"))
                for i in range(N_SERIAL)]
    if only:
        targets = [t for t in targets if t[0] == only]

    for i, (name, path) in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {name}")
        make_gif(name, path, br, origin, vsize)
    print(f"\n[OK] GIF {len(targets)}개 -> {OUT}")


if __name__ == "__main__":
    main()
