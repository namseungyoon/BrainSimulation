# -*- coding: utf-8 -*-
"""
01_slice_bbox/inspect_slices.py  —  atlas 제공 슬라이스 48개 형태 시각화

목적:
  Romani atlas 가 제공하는 슬라이스 마스크들을 각각 그림으로 저장해
  "어디에, 어떤 모양으로" 잘렸는지 눈으로 비교한다.
    - slice400                         : 단일 400um 기준 절편 (1개)
    - slice0 ~ slice46                 : CA1 종축 연속 박절편 (47개)
  -> 총 48개 그림 + 전체 비교 overview 1개.

각 그림 = 2패널:
  (좌) top-view(x-z): CA1 전체 footprint(회색) 위에 해당 슬라이스 voxel(층별 색)
  (우) 3D: 해당 슬라이스 voxel 을 층별 색으로 (슬라브 형태 확인)

출력: ./figures/slices/sliceXXX.png ,  ./figures/slices/_overview_all47.png

실행:
  C:\\Users\\SYNAM-OFFICE\\.conda\\envs\\ca1sim\\python.exe 01_slice_bbox/inspect_slices.py
"""
import os
import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
SLICE_DIR = os.path.join(ATLAS, "nrrd_volumes", "slices")
SERIAL_DIR = os.path.join(SLICE_DIR, "nrrd_masks_in_user_target")
OUT = os.path.join(HERE, "figures", "slices")
os.makedirs(OUT, exist_ok=True)

LAYER_ID = {"SO": 1, "SP": 2, "SR": 3, "SLM": 4}
ID_LAYER = {v: k for k, v in LAYER_ID.items()}
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}
N_SERIAL = 47


def load_grid():
    br, h = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    return br, origin, vsize


def vox_to_world(idx, origin, vsize):
    return idx * vsize + origin


def plot_one(name, mask_path, br, origin, vsize, foot_xz):
    d, _ = nrrd.read(mask_path)
    vox = np.argwhere(d > 0)
    if len(vox) == 0:
        print(f"  {name}: empty, skip"); return None
    labels = br[vox[:, 0], vox[:, 1], vox[:, 2]]
    world = vox_to_world(vox.astype(float), origin, vsize)
    # 층 구성
    comp = {L: int((labels == LAYER_ID[L]).sum()) for L in LAYER_ID}

    # 산점 서브샘플
    if len(world) > 9000:
        s = np.random.default_rng(0).choice(len(world), 9000, False)
        world_s, lab_s = world[s], labels[s]
    else:
        world_s, lab_s = world, labels

    fig = plt.figure(figsize=(13, 5.5))
    # (좌) top-view x-z
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.scatter(foot_xz[0], foot_xz[1], s=1, c="#dddddd", alpha=0.5)
    for L in LAYER_ID:
        m = lab_s == LAYER_ID[L]
        if m.any():
            ax0.scatter(world_s[m, 0], world_s[m, 2], s=3,
                        c=LAYER_COLOR[L], label=L)
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (µm)"); ax0.set_ylabel("z (µm)")
    ax0.set_title(f"{name}  CA1 내 위치 (위에서 본 x-z)")

    # (우) 3D
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    for L in LAYER_ID:
        m = lab_s == LAYER_ID[L]
        if m.any():
            ax1.scatter(world_s[m, 0], world_s[m, 1], world_s[m, 2],
                        s=2, c=LAYER_COLOR[L], alpha=0.5, label=L)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title("3D 형태 (색 = 층)")

    span = world.max(0) - world.min(0)
    fig.suptitle(f"{name}   복셀수={len(vox):,}   "
                 f"범위 x{span[0]:.0f} y{span[1]:.0f} z{span[2]:.0f} µm   |   "
                 + "  ".join(f"{L}:{comp[L]:,}" for L in LAYER_ID),
                 fontsize=10)
    handles = [Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ID]
    ax0.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
    plt.close(fig)
    return world.mean(0)


def main():
    br, origin, vsize = load_grid()
    print(f"atlas grid {br.shape}, voxel {vsize}um")

    # CA1 footprint (top view) 서브샘플
    ca1 = np.argwhere(br > 0)
    s = np.random.default_rng(1).choice(len(ca1), 25000, False)
    fw = vox_to_world(ca1[s].astype(float), origin, vsize)
    foot_xz = (fw[:, 0], fw[:, 2])

    # 1) slice400
    print("\n[1/48] slice400")
    plot_one("slice400", os.path.join(SLICE_DIR, "slice400.nrrd"),
             br, origin, vsize, foot_xz)

    # 2) slice0~46
    centers = []
    for i in range(N_SERIAL):
        name = f"slice{i}"
        print(f"[{i+2}/48] {name}")
        c = plot_one(name, os.path.join(SERIAL_DIR, f"{name}.nrrd"),
                     br, origin, vsize, foot_xz)
        if c is not None:
            centers.append((i, c))

    # overview: 47개 연속절편 중심을 top-view 에 번호로
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(foot_xz[0], foot_xz[1], s=1, c="#dddddd", alpha=0.5)
    cm = plt.cm.viridis(np.linspace(0, 1, N_SERIAL))
    for (i, c) in centers:
        ax.scatter(c[0], c[2], s=40, color=cm[i])
        ax.text(c[0], c[2], str(i), fontsize=7, ha="center", va="center")
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
    ax.set_title("개요: 47개 연속 절편(slice0~46) 중심 위치 (위에서 본 그림)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "_overview_all47.png"), dpi=120)
    plt.close(fig)

    print(f"\n[OK] 48개 슬라이스 그림 + overview -> {OUT}")


if __name__ == "__main__":
    main()
