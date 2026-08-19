# -*- coding: utf-8 -*-
"""
01_tissue/2_bbox/render_hierarchy.py  —  크롭 계층 한눈에: 전체 CA1 → slice400 → 층관통_v1

3단 패널:
  (1) 전체 CA1(물리 x-z) — 회색, 그 안에 slice400(색) + 창 위치(✚)
  (2) slice400(국소 종축×층관통) — 층별 색 + 창(사각형)
  (3) 층관통_v1 창 확대 — 층 + 전극(자극/기록)

실행: python 01_tissue/2_bbox/render_hierarchy.py  →  figures/1-2_confirmed_hierarchy.png
"""
import os
import glob
import json
import logging

import numpy as np
import h5py
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(HERE, "figures")
CFG = os.path.join(ROOT, "config", "window_layout.json")
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find(p):
    return sorted(glob.glob(os.path.join(DATA, "**", glob.escape(p)), recursive=True), key=len)[0]


def decode(g, name):
    lib = [s.decode() if isinstance(s, bytes) else s for s in g["@library"][name][:]]
    return np.array(lib, dtype=object)[g[name][:]]


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; w = cfg["window_um"]; c = w["center_local"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])

    mask, h = nrrd.read(find(os.path.join("slices", "slice400.nrrd")))
    origin = np.asarray(h["space origin"], float); vs = float(h["space directions"][0][0])
    nx, ny, nz = mask.shape
    with h5py.File(find(os.path.join("hippocampus_neurons", "nodes.h5")), "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        layer = decode(g, "layer")
    idx = np.floor((xyz - origin) / vs).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    ins = np.zeros(len(xyz), bool); ii = idx[ok]
    ins[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sl = np.where(ins)[0]
    Cs, Ls = xyz[sl], layer[sl]
    d = Cs - seed; u = d @ L; r = d @ R

    rng = np.random.default_rng(0)
    allsub = rng.choice(len(xyz), size=min(30000, len(xyz)), replace=False)

    # 창 8모서리 물리좌표(패널1 마커)
    corners = []
    for su in (-1, 1):
        for sr in (-1, 1):
            for sw in (-1, 1):
                corners.append(seed + (c["u"] + su * w["long"] / 2) * L +
                               (c["r"] + sr * w["radial"] / 2) * R + (c["w"] + sw * w["thick"] / 2) * Tk)
    corners = np.array(corners)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    # (1) 전체 CA1 물리 x-z
    ax = axes[0]
    ax.scatter(xyz[allsub, 0], xyz[allsub, 2], s=1, c="#cfd6de", alpha=0.5, label="전체 CA1")
    ax.scatter(Cs[:, 0], Cs[:, 2], s=2, c="#0d9488", alpha=0.5, label="slice400 슬랩")
    ax.scatter(corners[:, 0], corners[:, 2], s=1, c="none")
    ax.plot(cfg["window_um"]["center_xyz"][0], cfg["window_um"]["center_xyz"][2],
            marker="P", ms=16, mfc="#e23b3b", mec="white", mew=1.5, zorder=5)
    ax.annotate("층관통_v1", (cfg["window_um"]["center_xyz"][0], cfg["window_um"]["center_xyz"][2]),
                textcoords="offset points", xytext=(12, 6), fontsize=10, fontweight="bold", color="#e23b3b")
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
    ax.set_title("① 전체 CA1 (물리) — slice400 슬랩 + 창 위치")
    ax.legend(markerscale=5, loc="upper right", fontsize=8)
    # (2) slice400 국소 (u,r) + 창
    ax = axes[1]
    for Ln in LAYER_ORDER:
        m = Ls == Ln
        if m.any():
            ax.scatter(u[m], r[m], s=3, c=LC[Ln], alpha=0.5, label=Ln)
    ax.add_patch(Rectangle((c["u"] - w["long"] / 2, c["r"] - w["radial"] / 2),
                           w["long"], w["radial"], fill=False, ec="black", lw=2.4))
    ax.set_aspect("equal"); ax.set_xlabel("종축 (µm)"); ax.set_ylabel("층관통 (µm, SP=0)")
    ax.set_title("② slice400 (국소 프레임) — 창 층관통_v1"); ax.legend(markerscale=3, fontsize=8)
    # (3) 창 확대 + 전극
    ax = axes[2]
    for Ln in LAYER_ORDER:
        m = Ls == Ln
        if m.any():
            ax.scatter(u[m], r[m], s=14, c=LC[Ln], alpha=0.7, label=Ln)
    ax.add_patch(Rectangle((c["u"] - w["long"] / 2, c["r"] - w["radial"] / 2),
                           w["long"], w["radial"], fill=False, ec="black", lw=2.4))
    stim = cfg["electrodes"]["stim_id"]
    for e in cfg["electrodes"]["list"]:
        isS = e["id"] == stim; col = "#e23b3b" if isS else "#111111"
        ax.plot(e["u"], e["r"], marker="P" if isS else "o", ms=17 if isS else 12,
                mfc=col, mec="white", mew=1.6, zorder=5)
        ax.annotate(f'{e["id"]}·{e["layer"]}', (e["u"], e["r"]), textcoords="offset points",
                    xytext=(13, 0), va="center", fontsize=10, fontweight="bold", color=col)
    ax.set_xlim(c["u"] - w["long"] / 2 - 80, c["u"] + w["long"] / 2 + 80)
    ax.set_ylim(c["r"] - w["radial"] / 2 - 80, c["r"] + w["radial"] / 2 + 80)
    ax.set_aspect("equal"); ax.set_xlabel("종축 (µm)"); ax.set_ylabel("층관통 (µm)")
    ax.set_title("③ 층관통_v1 확대 — 전극(자극 E3/SR·기록 E2·E1)")
    fig.suptitle(f'크롭 계층  전체 CA1 → slice400 → 층관통_v1  ({cfg["name"]})', fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIG, "1-2_confirmed_hierarchy.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[계층 그림] {out}")


if __name__ == "__main__":
    main()
