# -*- coding: utf-8 -*-
"""
01_tissue/5_layers/layers.py  —  1-5: 층 경계·두께 확정 (1-5)

확정 창(층관통_v1) 안에서, 국소 크롭 atlas(lib/atlas_query)로 방사축을 따라
층(SO/SP/SR/SLM) 경계와 두께(µm)를 산출한다.
검증(1-5): 4층이 SO→SP→SR→SLM 순서로 나타나고 두께가 생리적으로 타당.

재료: config/window_layout.json · data/derived/atlas_crop.npz (lib/atlas_query)
실행: python 01_tissue/5_layers/layers.py
"""
import os
import sys
import json
import logging

import numpy as np
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
sys.path.insert(0, ROOT)
from lib.atlas_query import AtlasQuery  # noqa: E402

CFG = os.path.join(ROOT, "config", "window_layout.json")
NPZ = os.path.join(ROOT, "data", "derived", "atlas_crop.npz")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}
STEP = 3.0


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; w = cfg["window_um"]; c = w["center_local"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])
    aq = AtlasQuery(NPZ)

    r_s = np.arange(c["r"] - w["radial"] / 2 - 60, c["r"] + w["radial"] / 2 + 60, STEP)

    def line_layers(uu):
        xyz = seed[None, :] + uu * L + r_s[:, None] * R + c["w"] * Tk
        return aq.layer(xyz)

    # 종축 여러 위치 라인 → 두께 중앙값
    us = np.linspace(c["u"] - w["long"] / 2, c["u"] + w["long"] / 2, 9)
    th = {Ln: [] for Ln in LAYER_ORDER}
    for uu in us:
        ll = line_layers(uu)
        for Ln in LAYER_ORDER:
            m = ll == Ln
            if m.sum() > 0:
                th[Ln].append(r_s[m].max() - r_s[m].min() + STEP)
    thick = {Ln: float(np.median(th[Ln])) if th[Ln] else 0.0 for Ln in LAYER_ORDER}

    # 중앙 라인 경계(r 값)
    lc = line_layers(c["u"])
    print(f"[1-5] 창 방사축 층 두께(µm, 종축 9라인 중앙값):")
    for Ln in LAYER_ORDER:
        print(f"   {Ln:<4} {thick[Ln]:6.0f}")
    print(f"   합계 {sum(thick.values()):.0f} µm (창 층관통 {w['radial']}µm 중 실제 조직)")
    # 경계 r
    print("[1-5] 중앙라인(u=center) 층 r-범위(µm):")
    for Ln in LAYER_ORDER:
        m = lc == Ln
        if m.any():
            print(f"   {Ln:<4} r=[{r_s[m].min():.0f}, {r_s[m].max():.0f}]")

    fig_layers(r_s, us, line_layers, lc, thick, c, w)
    print(f"\n[1-5] 그림 저장 -> {FIG}/1-5_layers.png")


def fig_layers(r_s, us, line_layers, lc, thick, c, w):
    lab2i = {Ln: i + 1 for i, Ln in enumerate(LAYER_ORDER)}
    grid = np.zeros((len(r_s), len(us)))
    for j, uu in enumerate(us):
        ll = line_layers(uu)
        grid[:, j] = [lab2i.get(x, 0) for x in ll]
    cmap = ListedColormap(["#ffffff"] + [LC[Ln] for Ln in LAYER_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.pcolormesh(us, r_s, grid, cmap=cmap, norm=norm, shading="auto")
    ax.axhline(c["r"] - w["radial"] / 2, color="k", ls="--", lw=1)
    ax.axhline(c["r"] + w["radial"] / 2, color="k", ls="--", lw=1)
    ax.set_aspect("equal")  # 종축500 < 층관통800 실제 비율 반영(세로가 더 김)
    ax.set_xlabel("종축 u (µm, 가로 500)"); ax.set_ylabel("층관통 r (µm, SP=0, 세로 800)")
    ax.set_title("창 방사축 층 (종축 라인별) · 점선=창 층관통 범위")
    ax.legend(handles=[Patch(facecolor=LC[Ln], label=Ln) for Ln in LAYER_ORDER],
              title="층", loc="upper right", fontsize=8)
    ax = axes[1]
    vals = [thick[Ln] for Ln in LAYER_ORDER]
    ax.barh(LAYER_ORDER[::-1], vals[::-1], color=[LC[Ln] for Ln in LAYER_ORDER[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.0f}µm", va="center", fontsize=10)
    ax.set_xlabel("두께 (µm)"); ax.set_title("층 두께 (종축 9라인 중앙값)")
    fig.suptitle("1-5  층 경계·두께 (atlas 국소질의)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1-5_layers.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
