# -*- coding: utf-8 -*-
"""
05_placement/place_cells.py  —  단계 5: 뉴런 배치 (V2b)

목적:
  채택 슬라이스 slice400 안에 소마가 있는 세포를 nodes.h5 에서 추출한다.
  각 세포: id · 좌표(x/y/z) · 방향(orientation quaternion w,x,y,z) ·
           mtype · etype · layer · synapse_class · 정규화깊이 nd.
  -> slice_cells.npz (다운스트림용) + slice_cells_summary.json.

검증 (V2b):
  - 추출 세포수 = 17,647 (단계 1 사전집계와 일치)
  - 층/타입 조성·E:I 가 전체 CA1 대비 타당
  - 좌표가 slice400 mask 내부

산출 그림 (한글):
  figures/V2b_1_cells_3d.png        : 추출 세포 3D 배치 (층별 색)
  figures/V2b_2_topview.png         : CA1 위 슬라이스 세포 위치 (top view)
  figures/V2b_3_composition.png     : 슬라이스 조성 (m-type·E:I) vs 전체
  figures/V2b_4_cross_layers.png    : 단면에서 층별 세포 분포
  figures/V2b_5_orientation.png     : 세포 방사(정점) 방향 화살표 (단계6 미리보기)

실행:
  python 05_placement/place_cells.py
"""
import os
import json
from collections import Counter

import numpy as np
import h5py
import nrrd
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
NODES_H5 = os.path.join(ROOT, "data", "circuit", "networks", "nodes",
                        "hippocampus_neurons", "nodes.h5")
POP = "nodes/hippocampus_neurons/0"
SLICE = os.path.join(ATLAS, "nrrd_volumes", "slices", "slice400.nrrd")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}


def decode(grp, name):
    lib = [s.decode() if isinstance(s, bytes) else s
           for s in grp["@library"][name][:]]
    return np.array(lib, dtype=object)[grp[name][:]]


def main():
    # atlas 격자 + slice mask + 정규화깊이 재료
    mask, h = nrrd.read(SLICE)
    origin = np.asarray(h["space origin"], float)
    vsize = float(h["space directions"][0][0])
    phy, _ = nrrd.read(os.path.join(ATLAS, "[PH]y.nrrd"))
    so, _ = nrrd.read(os.path.join(ATLAS, "[PH]SO.nrrd"))
    slm, _ = nrrd.read(os.path.join(ATLAS, "[PH]SLM.nrrd"))
    base, total = so[0], slm[1] - so[0]

    with h5py.File(NODES_H5, "r") as f:
        g = f[POP]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        quat = np.stack([g[f"orientation_{c}"][:] for c in "wxyz"], 1)
        mtype = decode(g, "mtype"); etype = decode(g, "etype")
        layer = decode(g, "layer"); sclass = decode(g, "synapse_class")
    Nall = len(xyz)

    # slice400 소속 판정 (소마 복셀이 mask 내부)
    idx = np.floor((xyz - origin) / vsize).astype(int)
    nx, ny, nz = mask.shape
    ok = ((idx >= 0).all(1) & (idx[:, 0] < nx) &
          (idx[:, 1] < ny) & (idx[:, 2] < nz))
    inside = np.zeros(Nall, bool)
    ii = idx[ok]
    inside[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sel = np.where(inside)[0]
    n = len(sel)
    print(f"[V2b] slice400 추출 세포수 = {n:,} / 전체 {Nall:,}")

    # 정규화깊이(세포별)
    ci = idx[sel]
    nd = (phy[ci[:, 0], ci[:, 1], ci[:, 2]] - base[ci[:, 0], ci[:, 1], ci[:, 2]]) \
        / total[ci[:, 0], ci[:, 1], ci[:, 2]]

    # 저장
    np.savez_compressed(
        os.path.join(HERE, "slice_cells.npz"),
        node_id=sel.astype(np.int32), xyz=xyz[sel].astype(np.float32),
        quat_wxyz=quat[sel].astype(np.float32),
        mtype=mtype[sel].astype("U16"), etype=etype[sel].astype("U16"),
        layer=layer[sel].astype("U8"), sclass=sclass[sel].astype("U4"),
        nd=nd.astype(np.float32))

    exc = int((sclass[sel] == "EXC").sum()); inh = n - exc
    comp_layer = Counter(layer[sel]); comp_m = Counter(mtype[sel])
    summary = {
        "step": "5 placement (V2b)", "slice": "slice400",
        "n_cells": n, "n_all": Nall,
        "EI": {"exc": exc, "inh": inh,
               "ratio": f"{100*exc/n:.1f}:{100*inh/n:.1f}"},
        "by_layer": dict(comp_layer),
        "by_mtype": dict(sorted(comp_m.items(), key=lambda kv: -kv[1])),
    }
    with open(os.path.join(HERE, "slice_cells_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[V2b] E:I = {summary['EI']['ratio']}  층별={dict(comp_layer)}")

    # 그림
    C = xyz[sel]; LY = layer[sel]
    _fig_3d(C, LY, n)
    _fig_topview(C, LY, xyz, origin, vsize, mask)
    _fig_composition(comp_m, exc, inh, n, Nall, sclass)
    _fig_cross(C, LY, origin, vsize)
    _fig_orientation(C, quat[sel], LY)
    print(f"[OK] slice_cells.npz + summary + figures -> {FIG}")


def _fig_3d(C, LY, n):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for L in LAYER_ORDER:
        m = LY == L
        if m.any():
            ax.scatter(C[m, 0], C[m, 1], C[m, 2], s=3, alpha=0.5,
                       c=LAYER_COLOR[L], label=L)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    ax.set_title(f"V2b-1  slice400 추출 세포 3D 배치 ({n:,}개, 층별 색)")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V2b_1_cells_3d.png"), dpi=130)
    plt.close(fig)


def _fig_topview(C, LY, xyz_all, origin, vsize, mask, n_bg=25000):
    fig, ax = plt.subplots(figsize=(8, 7))
    s = np.random.default_rng(0).choice(len(xyz_all), n_bg, False)
    ax.scatter(xyz_all[s, 0], xyz_all[s, 2], s=1, c="#dddddd", alpha=0.4,
               label="전체 CA1")
    for L in LAYER_ORDER:
        m = LY == L
        if m.any():
            ax.scatter(C[m, 0], C[m, 2], s=4, c=LAYER_COLOR[L], label=L)
    ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
    ax.set_title("V2b-2  CA1 내 slice400 세포 위치 (위에서 본 x-z)")
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V2b_2_topview.png"), dpi=130)
    plt.close(fig)


def _fig_composition(comp_m, exc, inh, n, Nall, sclass_all):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    items = sorted(comp_m.items(), key=lambda kv: -kv[1])
    keys = [k for k, _ in items]; vals = [v for _, v in items]
    axes[0].barh(keys[::-1], vals[::-1], color="#55A868")
    axes[0].set_xscale("log"); axes[0].set_xlabel("세포 수 (로그)")
    axes[0].set_title(f"slice400 m-type 조성 ({len(keys)}종)")
    # E:I 비교 (슬라이스 vs 전체)
    exc_all = int((sclass_all == "EXC").sum()); inh_all = Nall - exc_all
    x = np.arange(2)
    axes[1].bar(x - 0.2, [100*exc/n, 100*inh/n], 0.4, label="slice400",
                color="#DD8452")
    axes[1].bar(x + 0.2, [100*exc_all/Nall, 100*inh_all/Nall], 0.4,
                label="전체 CA1", color="#4C72B0")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["흥분 EXC", "억제 INH"])
    axes[1].set_ylabel("비율 (%)"); axes[1].legend()
    axes[1].set_title(f"E:I 비교  slice400 {100*exc/n:.1f}:{100*inh/n:.1f}  vs  전체 89:11")
    fig.suptitle("V2b-3  slice400 세포 조성 (전체 CA1과 비교)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V2b_3_composition.png"), dpi=130)
    plt.close(fig)


def _fig_cross(C, LY, origin, vsize, thick=40):
    """슬라이스 중앙 얇은 x-슬랩의 세포를 y-z 로 (층 구조 확인)."""
    xc = (C[:, 0].min() + C[:, 0].max()) / 2
    m0 = np.abs(C[:, 0] - xc) < thick
    fig, ax = plt.subplots(figsize=(8, 7))
    for L in LAYER_ORDER:
        m = m0 & (LY == L)
        if m.any():
            ax.scatter(C[m, 1], C[m, 2], s=6, c=LAYER_COLOR[L], label=L)
    ax.set_aspect("equal"); ax.set_xlabel("y (µm)"); ax.set_ylabel("z (µm)")
    ax.set_title(f"V2b-4  슬라이스 중앙 단면(x±{thick}µm) 층별 세포 분포")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V2b_4_cross_layers.png"), dpi=130)
    plt.close(fig)


def _fig_orientation(C, quat, LY, n_arrow=1500):
    """세포별 방사(정점) 방향 = R([0,1,0]) 화살표 (단계6 미리보기)."""
    q_scipy = quat[:, [1, 2, 3, 0]]
    radial = Rot.from_quat(q_scipy).apply(np.array([0.0, 1.0, 0.0]))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    s = np.random.default_rng(0).choice(len(C), min(n_arrow, len(C)), False)
    for L in LAYER_ORDER:
        m = LY[s] == L
        if m.any():
            ax.quiver(C[s][m, 0], C[s][m, 1], C[s][m, 2],
                      radial[s][m, 0], radial[s][m, 1], radial[s][m, 2],
                      length=40, color=LAYER_COLOR[L], alpha=0.6, linewidth=0.6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("V2b-5  세포별 정점(방사) 방향 R·[0,1,0] (단계6 회전 미리보기)")
    ax.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER],
              fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V2b_5_orientation.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
