# -*- coding: utf-8 -*-
"""
02_neurons/2_placement/make_allcells_zoom.py  —  2-2 보조: 전세포 확대 GIF (2D 고속)

5,610 전세포 실제 수상돌기를 2D(u-r) 고해상으로 한 번 렌더한 뒤, 이미지를
점진 크롭+리사이즈해 '서서히 확대' GIF를 만든다(프레임마다 재렌더 없음 → 빠름).
결과: figures/2-2_allcells_zoom.gif (+ 고해상 정지본 2-2_allcells_fur.png)

실행: python 02_neurons/2_placement/make_allcells_zoom.py [--seg N]
"""
import os
import sys
import json

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
LIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEGCAP = int(sys.argv[sys.argv.index("--seg") + 1]) if "--seg" in sys.argv else 260


def morph_segs(path, cap=SEGCAP):
    rows = np.loadtxt(path, comments="#")
    idx = rows[:, 0].astype(int); typ = rows[:, 1].astype(int)
    xyz = rows[:, 2:5].astype(np.float32); par = rows[:, 6].astype(int)
    id2 = {i: k for k, i in enumerate(idx)}
    seg = [(id2[par[k]], k) for k in range(len(idx)) if par[k] in id2 and typ[k] in (1, 3, 4)]
    seg = np.array(seg, int)
    if len(seg) > cap:
        seg = seg[np.linspace(0, len(seg) - 1, cap).astype(int)]
    return xyz, seg


def build2d():
    d = np.load(NPZ, allow_pickle=True)
    xyz = d["xyz"]; Q = d["orientation_wxyz"]; morph = d["morphology"].astype(str)
    fr = json.load(open(CFG, encoding="utf-8"))["frame_um"]
    seed = np.array(fr["seed"]); M = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    cache = {}; segs = []
    for i in range(len(xyz)):
        mm = morph[i]
        if mm not in cache:
            cache[mm] = morph_segs(os.path.join(LIB, mm + ".swc"))
        pts, sg = cache[mm]
        if len(sg) == 0:
            continue
        loc = (xyz[i] + Rot.from_quat(Q[i][[1, 2, 3, 0]]).apply(pts) - seed) @ M
        segs.append(loc[sg][:, :, :2])   # u,r 만
    S = np.concatenate(segs).astype(np.float32)
    print(f"[zoom] {len(xyz):,}세포 · 세그먼트 {len(S):,}")
    return S


def main():
    S = build2d()
    # 고해상 정지본 (u-r, 다크, 파랑 글로우 2겹)
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="black")
    ax.set_facecolor("black"); ax.set_axis_off()
    ax.add_collection(LineCollection(S, colors=[(0.40, 0.68, 1.0, 0.09)], linewidths=0.7))
    ax.add_collection(LineCollection(S, colors=[(0.80, 0.93, 1.0, 0.28)], linewidths=0.22))
    m = S.reshape(-1, 2)
    ax.set_xlim(m[:, 0].min(), m[:, 0].max()); ax.set_ylim(m[:, 1].min(), m[:, 1].max())
    ax.set_aspect("equal")
    fig.subplots_adjust(0, 0, 1, 1)
    still = os.path.join(FIG, "2-2_allcells_fur.png")
    fig.savefig(still, dpi=180, facecolor="black"); plt.close(fig)
    print(f"[zoom] 정지본 -> {still}")

    # 이미지 크롭 확대 GIF (재렌더 없음)
    img = Image.open(still).convert("RGB")
    W, H = img.size
    cx, cy = W * 0.52, H * 0.46      # 조직 중심 근처
    frames = []
    N = 60
    for i in range(N):
        t = i / (N - 1)
        e = t * t * (3 - 2 * t)              # smoothstep
        scale = 1.0 - 0.80 * e               # 1.0 → 0.20 (5배 확대)
        hw, hh = W * scale / 2, H * scale / 2
        l = max(0, min(W - 2 * hw, cx - hw)); tp = max(0, min(H - 2 * hh, cy - hh))
        crop = img.crop((int(l), int(tp), int(l + 2 * hw), int(tp + 2 * hh))).resize((W // 2, H // 2), Image.LANCZOS)
        frames.append(crop)
    hold = [frames[-1]] * 8
    out = os.path.join(FIG, "2-2_allcells_zoom.gif")
    (frames + hold)[0].save(out, save_all=True, append_images=(frames + hold)[1:],
                            duration=90, loop=0, optimize=True)
    print(f"[zoom] GIF -> {out}")


if __name__ == "__main__":
    main()
