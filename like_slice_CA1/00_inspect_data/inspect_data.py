# -*- coding: utf-8 -*-
"""
00_inspect_data/inspect_data.py  —  단계 0: SONATA nodes.h5 구조·세포수 확인 (V0)

목적:
  Romani 2024 circuit 의 nodes.h5 (CA1 456k 세포 배치)를 열어
  - 세포수
  - 필드(좌표 x/y/z, orientation quaternion, mtype/etype/layer 등)
  - 층/타입 분포, E:I 비율
  - 좌표 범위
  를 확인하고, 검증용 그림(figures/V0_*.png)을 생성한다.

검증 기준 (V0): 세포수 ≈ 456,380, E:I ≈ 89:11, 층 SO/SP/SR/SLM, mtype 12종.

실행 (어느 위치에서 실행해도 동작):
  C:\\Users\\SYNAM-OFFICE\\.conda\\envs\\ca1sim\\python.exe 00_inspect_data/inspect_data.py
"""
import os
from collections import Counter

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")  # 파일 저장 전용
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 한글 폰트 (Windows 맑은 고딕)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 경로: 이 스크립트(00_inspect_data/) 기준으로 프로젝트 루트를 찾는다.
HERE = os.path.dirname(os.path.abspath(__file__))   # .../00_inspect_data
ROOT = os.path.dirname(HERE)                         # .../like_slice_CA1
NODES_H5 = os.path.join(ROOT, "data", "circuit", "networks", "nodes",
                        "hippocampus_neurons", "nodes.h5")
POP = "nodes/hippocampus_neurons/0"      # population group
FIG_DIR = os.path.join(HERE, "figures")  # 단계 0 검증 그림은 이 폴더 안에
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]  # 표면(SO)→심부(SLM) 표시 순서
os.makedirs(FIG_DIR, exist_ok=True)


def _decode(grp, name):
    """@library 인덱스 필드를 실제 문자열 배열로 복원."""
    lib = [s.decode() if isinstance(s, bytes) else s
           for s in grp["@library"][name][:]]
    idx = grp[name][:]
    return np.array(lib, dtype=object)[idx], lib


def main():
    with h5py.File(NODES_H5, "r") as f:
        grp = f[POP]
        N = grp["x"].shape[0]
        print(f"[V0] nodes.h5 = {NODES_H5}")
        print(f"[V0] 총 세포수 N = {N:,}\n")

        # --- 카테고리 필드 분포 ---
        dist = {}
        for field in ["layer", "mtype", "etype",
                      "morph_class", "synapse_class"]:
            dec, lib = _decode(grp, field)
            c = Counter(dec)
            dist[field] = c
            print(f"=== {field}  (library={len(lib)}종) ===")
            for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
                print(f"   {k:<14} {v:>8,}  ({100*v/N:4.1f}%)")
            print()

        # --- E:I ---
        sc = dist["synapse_class"]
        exc, inh = sc.get("EXC", 0), sc.get("INH", 0)
        print(f"=== E:I ===  EXC={exc:,}  INH={inh:,}  "
              f"->  {100*exc/N:.1f} : {100*inh/N:.1f}\n")

        # --- 좌표 ---
        xyz = {ax: grp[ax][:] for ax in ("x", "y", "z")}
        print("=== coordinate range (um) ===")
        for ax in ("x", "y", "z"):
            d = xyz[ax]
            print(f"   {ax}: [{d.min():.1f}, {d.max():.1f}]  "
                  f"span={d.max()-d.min():.1f}")

        # --- orientation quaternion norm 점검 ---
        q = np.stack([grp[f"orientation_{c}"][:] for c in "wxyz"], axis=1)
        norms = np.linalg.norm(q, axis=1)
        print(f"\n=== orientation quaternion norm: "
              f"min={norms.min():.4f} max={norms.max():.4f} "
              f"(1.0 이어야 정규화) ===")

        # 그림용 layer 배열 (서브샘플)
        layer_dec, _ = _decode(grp, "layer")

    # ------------------------------------------------------------------
    # 그림 생성
    # ------------------------------------------------------------------
    _fig_distributions(dist, N)
    _fig_ei_pie(exc, inh)
    _fig_3d_scatter(xyz, layer_dec)
    print(f"\n[V0] 그림 저장 완료 -> {FIG_DIR}/V0_*.png")


def _fig_distributions(dist, N):
    """층·mtype·etype 분포 막대그래프 (한 장)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # layer (고정 순서)
    c = dist["layer"]
    vals = [c.get(k, 0) for k in LAYER_ORDER]
    axes[0].bar(LAYER_ORDER, vals, color="#4C72B0")
    axes[0].set_title("소마 층 분포")
    axes[0].set_ylabel("세포 수")
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:,}\n{100*v/N:.1f}%",
                     ha="center", va="bottom", fontsize=8)

    # mtype
    c = dist["mtype"]
    items = sorted(c.items(), key=lambda kv: -kv[1])
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    axes[1].barh(keys[::-1], vals[::-1], color="#55A868")
    axes[1].set_title("m-type 분포 (12종)")
    axes[1].set_xlabel("세포 수 (로그)")
    axes[1].set_xscale("log")

    # etype
    c = dist["etype"]
    items = sorted(c.items(), key=lambda kv: -kv[1])
    axes[2].bar([k for k, _ in items], [v for _, v in items],
                color="#C44E52")
    axes[2].set_title("e-type 분포")
    axes[2].set_ylabel("세포 수")
    axes[2].set_yscale("log")

    fig.suptitle(f"V0  CA1 nodes.h5 — 총 세포수 N = {N:,}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V0_distributions.png"), dpi=130)
    plt.close(fig)


def _fig_ei_pie(exc, inh):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([exc, inh], labels=[f"흥분 EXC\n{exc:,}", f"억제 INH\n{inh:,}"],
           autopct="%1.1f%%", colors=["#DD8452", "#4C72B0"],
           startangle=90, wedgeprops=dict(edgecolor="w"))
    ax.set_title("V0  흥분 : 억제 비율 (목표 ~89:11)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V0_EI_ratio.png"), dpi=130)
    plt.close(fig)


def _fig_3d_scatter(xyz, layer_dec, n_sample=20000):
    """전체 CA1 세포 배치 3D 산점도 (층별 색, 서브샘플)."""
    N = len(xyz["x"])
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(n_sample, N), replace=False)
    colors = {"SO": "#4C72B0", "SP": "#DD8452",
              "SR": "#55A868", "SLM": "#C44E52"}

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for lyr in LAYER_ORDER:
        m = layer_dec[idx] == lyr
        if m.any():
            ax.scatter(xyz["x"][idx][m], xyz["y"][idx][m], xyz["z"][idx][m],
                       s=2, alpha=0.4, c=colors[lyr], label=lyr)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    ax.set_title(f"V0  CA1 세포 배치 (3D, {len(idx):,}개 샘플)")
    ax.legend(markerscale=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "V0_placement_3d.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
