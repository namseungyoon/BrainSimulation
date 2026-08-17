# -*- coding: utf-8 -*-
"""
01_tissue/1_inspect/inspect_atlas.py  —  Stage 1: Romani atlas(NRRD) 검사 (V0-atlas)

목적:
  Romani(2024) CA1 atlas 의 복셀 볼륨(NRRD)을 열어
    - 복셀 그리드·복셀크기·물리 bbox(µm)
    - brain_regions 층 라벨(0~4) ↔ 층(SO/SP/SR/SLM) 매핑 + 층별 복셀수·부피
    - 좌표장(coordinates, l/t/r)·방향장(orientation, quaternion) 성분·값범위
  를 확인하고, CA1 층 단면 그림(figures/V0_atlas_layers.png)을 저장한다.
  → 이 atlas 가 다음 단계(2_bbox: 800×500×400µm 마이크로 창 절취)의 재료.

검증 기준 (V0-atlas): 층 4종(SO/SP/SR/SLM) 라벨 확인 · 복셀 16µm · 방향장 quaternion 4성분.

필요 패키지: numpy, pynrrd, matplotlib
실행 (VS Code/WSL):  python 01_tissue/1_inspect/inspect_atlas.py
"""
import os
import glob
import logging

import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LAYERS = ["SO", "SP", "SR", "SLM"]
LAYER_COLORS = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find_atlas_dir():
    """brain_regions.nrrd 를 품은 atlas 디렉토리 자동 탐지."""
    hits = glob.glob(os.path.join(DATA, "**", "brain_regions.nrrd"), recursive=True)
    if not hits:
        raise SystemExit(f"[에러] brain_regions.nrrd 없음. data/ 아래 atlas 압축 확인: {DATA}")
    return os.path.dirname(sorted(hits, key=len)[0])


def voxel_size_um(header):
    sd = np.array(header.get("space directions"), dtype=float)
    sd = sd[~np.isnan(sd).any(axis=1)]          # 성분축(NaN 행) 제거
    return np.array([np.linalg.norm(r) for r in sd])


def main():
    atlas = find_atlas_dir()
    print("=" * 70)
    print(f"[V0-atlas] atlas 디렉토리 = {atlas}")
    print("=" * 70)

    # --- brain_regions (층 라벨) ---
    br_path = os.path.join(atlas, "brain_regions.nrrd")
    regions, hdr = nrrd.read(br_path)
    vox = voxel_size_um(hdr)
    origin = np.array(hdr.get("space origin", [0, 0, 0]), dtype=float)
    dims = np.array(regions.shape)
    phys = dims * vox
    print(f"\n[그리드] 복셀 {tuple(dims)}  복셀크기 {vox} µm")
    print(f"[물리 bbox] origin={origin} µm  →  범위 {phys} µm")
    print(f"           x[{origin[0]:.0f}, {origin[0]+phys[0]:.0f}] "
          f"y[{origin[1]:.0f}, {origin[1]+phys[1]:.0f}] "
          f"z[{origin[2]:.0f}, {origin[2]+phys[2]:.0f}]")

    # --- 라벨 ↔ 층 매핑 (CA1_<layer> 마스크로 확정) ---
    label_to_layer = {}
    for lyr in LAYERS:
        cand = glob.glob(os.path.join(atlas, "**", f"CA1_{lyr}.nrrd"), recursive=True)
        if not cand:
            continue
        mask, _ = nrrd.read(cand[0])
        vals = np.unique(regions[mask > 0])
        vals = vals[vals != 0]
        if len(vals):
            label_to_layer[int(vals[0])] = lyr
    if not label_to_layer:   # 마스크 없으면 1..4 = SO..SLM 가정
        label_to_layer = {i + 1: lyr for i, lyr in enumerate(LAYERS)}
        print("\n[주의] CA1_<layer> 마스크 미발견 → 라벨 1~4 = SO~SLM 로 가정")

    total = regions.size
    ca1_vox = int((regions > 0).sum())
    print(f"\n=== brain_regions 라벨 ↔ 층 (복셀 {ca1_vox:,} = CA1) ===")
    voxvol = float(np.prod(vox)) / 1e9   # µm³ → (10^-3 mm)³ ... 표기는 mm³
    for lab in sorted(label_to_layer):
        n = int((regions == lab).sum())
        print(f"   라벨 {lab} = {label_to_layer[lab]:<4}  {n:>10,} 복셀  "
              f"({100*n/ca1_vox:4.1f}% of CA1,  {n*np.prod(vox)/1e9:.4f} mm³)")

    # --- 좌표장 / 방향장 ---
    for name, comp in [("coordinates", 3), ("orientation", 4)]:
        p = os.path.join(atlas, f"{name}.nrrd")
        if os.path.exists(p):
            d, _ = nrrd.read(p)
            inside = None
            try:  # 성분축이 앞(comp,X,Y,Z)이라 가정
                arr = np.moveaxis(d, 0, -1) if d.shape[0] == comp else d
                m = regions > 0
                inside = arr[m]
            except Exception:
                pass
            rng = (float(np.nanmin(d)), float(np.nanmax(d)))
            print(f"\n=== {name}.nrrd  shape={d.shape} (성분 {comp})  값범위 {rng[0]:.3f}~{rng[1]:.3f} ===")
            if name == "orientation" and inside is not None:
                nrm = np.linalg.norm(inside, axis=1)
                nrm = nrm[np.isfinite(nrm) & (nrm > 0)]
                if len(nrm):
                    print(f"   CA1 내부 quaternion norm: {nrm.min():.3f}~{nrm.max():.3f} (1.0 이어야 정규화)")

    # --- 그림: 3 직교 단면 (CA1 가장 풍부한 슬라이스) ---
    fig_layers(regions, label_to_layer, vox)
    print(f"\n[V0-atlas] 그림 저장 -> {FIG_DIR}/V0_atlas_layers.png")
    print("[V0-atlas] 완료.")


def fig_layers(regions, label_to_layer, vox):
    maxlab = int(regions.max())
    colors = ["#ffffff"] * (maxlab + 1)
    for lab, lyr in label_to_layer.items():
        if lab <= maxlab:
            colors[lab] = LAYER_COLORS[lyr]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, maxlab + 1.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axis_names = [("y", "z"), ("x", "z"), ("x", "y")]
    for a in range(3):
        # CA1(>0) 복셀이 가장 많은 슬라이스 선택
        counts = (regions > 0).sum(axis=tuple(i for i in range(3) if i != a))
        k = int(np.argmax(counts))
        sl = np.take(regions, k, axis=a).T   # 보기 좋게 전치
        axes[a].imshow(sl, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        axes[a].set_title(f"축{a} = {k}번 단면 (CA1 최다)")
        axes[a].set_xlabel(f"{axis_names[a][0]} 복셀")
        axes[a].set_ylabel(f"{axis_names[a][1]} 복셀")
    handles = [Patch(facecolor=LAYER_COLORS[l], label=l) for l in LAYERS]
    axes[0].legend(handles=handles, loc="upper right", title="층")
    fig.suptitle("V0-atlas  CA1 층 구조 (brain_regions 직교 단면, 복셀 16µm)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V0_atlas_layers.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
