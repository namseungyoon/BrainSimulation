# -*- coding: utf-8 -*-
"""
01_tissue/2_bbox/define_window.py  —  Stage 2: 마이크로 창(bbox) 후보 정의 (V1a)

축 배정(2026-08-18 수정):
  - 종축(long, proximodistal = SC 섬유방향 = 전극 라인) = 800 µm  → 전극 스팬+여유 확보
  - 층관통(radial, SP+SR) = 500 µm                              → PC 소마(SP)+SC 시냅스(SR)
  - 두께(thick) = 400 µm
  전극을 긴 축(종축)에 놓아야 바깥 전극의 기여반경(~200µm)이 창 안에 들어옴.

방법:
  - atlas slice400.nrrd(400µm 슬랩)의 세포 선택(소마 복셀 mask 내부).
  - 국소 프레임: long=PCA 최장축 · radial=중앙 SP 정점방향 평균(국소 층수직) · thick=long×radial.
  - 창: |종축|≤400 · |층관통 − R_CENTER|≤250(SP+SR로 치우침) · |두께|≤200.

실행: python 01_tissue/2_bbox/define_window.py
"""
import os
import glob
import json
import logging
from collections import Counter

import numpy as np
import h5py
import nrrd
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

W_LONG = 800.0          # 종축(SC섬유·전극라인, proximodistal)
W_RADIAL = 500.0        # 층관통(SP+SR)
W_THICK = 400.0         # 두께
R_CENTER = 175.0        # 층관통 창 중심(SP=0 기준 +쪽=SR). SP+SR 커버되게 SR쪽으로.
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find(pattern):
    hits = glob.glob(os.path.join(DATA, "**", pattern), recursive=True)
    if not hits:
        raise SystemExit(f"[에러] {pattern} 없음 (data/ 확인)")
    return sorted(hits, key=len)[0]


def decode(grp, name):
    lib = [s.decode() if isinstance(s, bytes) else s for s in grp["@library"][name][:]]
    return np.array(lib, dtype=object)[grp[name][:]]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def main():
    slice_path = find(os.path.join("slices", "slice400.nrrd"))
    nodes_h5 = find(os.path.join("hippocampus_neurons", "nodes.h5"))
    mask, h = nrrd.read(slice_path)
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    nx, ny, nz = mask.shape

    with h5py.File(nodes_h5, "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        quat = np.stack([g[f"orientation_{c}"][:] for c in "wxyz"], 1)
        mtype = decode(g, "mtype"); layer = decode(g, "layer")
        sclass = decode(g, "synapse_class")
    Nall = len(xyz)

    idx = np.floor((xyz - origin) / vsize).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    inside = np.zeros(Nall, bool); ii = idx[ok]
    inside[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sl = np.where(inside)[0]
    Cs, Ls, Qs = xyz[sl], layer[sl], quat[sl]
    print(f"[슬랩] slice400 세포수 = {len(sl):,} / 전체 {Nall:,}")

    # 종축 = PCA 최장축(proximodistal)
    c0 = Cs.mean(0)
    _, _, Vt = np.linalg.svd(Cs - c0, full_matrices=False)
    spans = (Cs - c0) @ Vt.T
    long_dir = Vt[int(np.argmax(spans.max(0) - spans.min(0)))]
    u = (Cs - c0) @ long_dir; u_c = np.median(u)

    # 층관통 = 중앙 종축밴드 SP 정점방향 평균(국소 층수직)
    band = np.abs(u - u_c) <= W_LONG / 2
    sp_band = band & (Ls == "SP")
    radial_dir = unit(Rot.from_quat(Qs[sp_band][:, [1, 2, 3, 0]]).apply([0.0, 1.0, 0.0]).mean(0))
    radial_dir = unit(radial_dir - (radial_dir @ long_dir) * long_dir)
    thick_dir = unit(np.cross(long_dir, radial_dir))
    seed = Cs[sp_band].mean(0)            # SP 중심(층관통 r=0 기준점)

    d = Cs - seed
    l = d @ long_dir; r = d @ radial_dir; t = d @ thick_dir
    inwin = ((np.abs(l) <= W_LONG / 2) &
             (np.abs(r - R_CENTER) <= W_RADIAL / 2) &
             (np.abs(t) <= W_THICK / 2))
    win = sl[inwin]; Lw = layer[win]; n = len(win)
    exc = int((sclass[win] == "EXC").sum()); inh = n - exc
    comp_layer = {L: int((Lw == L).sum()) for L in LAYER_ORDER}
    comp_m = Counter(mtype[win].tolist())
    print(f"[프레임] long·radial 직교={long_dir@radial_dir:.1e}")
    print(f"\n[창 후보] 종축(전극축) {W_LONG:.0f} × 층관통 {W_RADIAL:.0f} × 두께 {W_THICK:.0f} µm")
    print(f"[창] 세포수 = {n:,}   E:I = {100*exc/max(n,1):.1f}:{100*inh/max(n,1):.1f}")
    print(f"[창 층별] {comp_layer}")
    print(f"[창 m-type] {dict(sorted(comp_m.items(), key=lambda kv:-kv[1]))}")

    json.dump({
        "size_um": {"long_electrode_axis": W_LONG, "radial_layer": W_RADIAL, "thick": W_THICK},
        "radial_center_um": R_CENTER,
        "seed_center_um": seed.tolist(),
        "long_dir": long_dir.tolist(), "radial_dir": radial_dir.tolist(),
        "thick_dir": thick_dir.tolist(),
        "n_cells": n, "EI": f"{100*exc/max(n,1):.1f}:{100*inh/max(n,1):.1f}",
        "by_layer": comp_layer,
    }, open(os.path.join(HERE, "window_def.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    fig_preview(u - u_c, (d @ radial_dir), Ls, l[inwin], r[inwin], Lw, n, exc, inh)
    print(f"\n[V1a] window_def.json + 그림 저장 -> {HERE}")


def fig_preview(u_all, r_all, Ls, lw, rw, Lw, n, exc, inh):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    for L in LAYER_ORDER:
        m = Ls == L
        if m.any():
            ax.scatter(u_all[m], r_all[m], s=3, alpha=0.3, c=LAYER_COLOR[L], label=L)
    ax.add_patch(Rectangle((-W_LONG / 2, R_CENTER - W_RADIAL / 2), W_LONG, W_RADIAL,
                           fill=False, ec="black", lw=2.2))
    ax.set_xlabel("종축 proximodistal = 전극라인 (µm, 중앙=0)")
    ax.set_ylabel("층관통 radial (µm, SP=0, +는 SR)")
    ax.set_title(f"slice400 슬랩 + 창(사각형, {W_LONG:.0f}×{W_RADIAL:.0f}µm)")
    ax.legend(markerscale=3, fontsize=8, title="층", loc="upper right")
    ax.set_ylim(-500, 700)
    ax = axes[1]
    for L in LAYER_ORDER:
        m = Lw == L
        if m.any():
            ax.scatter(lw[m], rw[m], s=12, c=LAYER_COLOR[L], label=L)
    ax.set_aspect("equal"); ax.set_xlabel("종축 (µm, 전극라인)"); ax.set_ylabel("층관통 (µm, SP=0)")
    ax.set_title(f"창 내부 {n:,}개  E:I {100*exc/max(n,1):.0f}:{100*inh/max(n,1):.0f}")
    ax.legend(markerscale=2, fontsize=8, title="층")
    fig.suptitle("V1a  마이크로 창 — 전극축=종축800(여유) · 층관통500(SP+SR)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V1a_window_candidate.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
