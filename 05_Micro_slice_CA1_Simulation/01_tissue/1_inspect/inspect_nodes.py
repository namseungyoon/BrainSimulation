# -*- coding: utf-8 -*-
"""
01_tissue/1_inspect/inspect_nodes.py  —  Stage 1: Romani circuit SONATA nodes 검사 (1-1)

목적:
  Romani(2024) CA1 circuit 의 nodes.h5 (약 456k 세포 배치)를 열어
    - 총 세포수
    - 데이터셋(필드) 목록: 좌표 x/y/z, orientation quaternion, 범주형(layer/mtype/etype 등)
    - 층/타입 분포, E:I 비율
    - 좌표 범위(µm), quaternion 정규화 점검
  를 확인하고, 검증 그림(figures/1-1_*.png)을 저장한다.

검증 기준 (1-1): 세포수 ≈ 456,380 · E:I ≈ 89:11 · 층 SO/SP/SR/SLM · m-type 12종.

필요 패키지: numpy, h5py, matplotlib   (NEURON·pynrrd 불필요)
실행 (VS Code 터미널에서, 05 어디서든):
    python 01_tissue/1_inspect/inspect_nodes.py
    # 또는 이 폴더에서:  python inspect_nodes.py
"""
import os
import sys
import glob
from collections import Counter

import logging

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")  # 파일 저장 전용(창 안 띄움)
# 폰트 폴백 목록 탐색 시 나오는 'font not found' 경고 소거(그림엔 영향 없음)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 투영 등록)

# 한글 폰트: WSL/리눅스=NanumGothic · Windows=맑은 고딕. 목록 중 있는 것을 자동 사용.
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 경로: 이 스크립트(05/01_tissue/1_inspect/) 기준으로 05 루트를 찾는다.
HERE = os.path.dirname(os.path.abspath(__file__))            # .../01_tissue/1_inspect
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))       # .../05_Micro_slice_CA1_Simulation
DATA = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LAYER_ORDER = ["SO", "SP", "SR", "SLM"]   # 표면(SO)→심부(SLM)
CAT_FIELDS = ["layer", "mtype", "etype", "morph_class", "synapse_class", "region"]
EXPECT_N = 456_380                        # Romani CA1 세포수(참고값)


def find_nodes_h5():
    """data/ 아래에서 hippocampus_neurons/nodes.h5 를 자동 탐지(압축 이중폴더 대응)."""
    hits = glob.glob(os.path.join(DATA, "**", "hippocampus_neurons", "nodes.h5"),
                     recursive=True)
    if not hits:
        # 백업: 아무 nodes.h5 나(경고와 함께)
        hits = glob.glob(os.path.join(DATA, "**", "nodes.h5"), recursive=True)
    if not hits:
        sys.exit(f"[에러] nodes.h5 를 찾지 못함. data/ 아래에 circuit 압축을 풀었는지 확인.\n"
                 f"        탐색 위치: {DATA}")
    hits.sort(key=len)  # 가장 짧은 경로(주 population) 우선
    return hits[0]


def find_population_group(f):
    """nodes.h5 안의 population group('nodes/<pop>/0')을 찾는다."""
    pops = list(f["nodes"].keys())
    pop = "hippocampus_neurons" if "hippocampus_neurons" in pops else pops[0]
    return pop, f[f"nodes/{pop}/0"]


def decode_library(grp, name):
    """@library 인덱스 필드를 실제 문자열 배열로 복원. 없으면 (None, None)."""
    if name not in grp:
        return None, None
    if "@library" in grp and name in grp["@library"]:
        lib = [s.decode() if isinstance(s, bytes) else s
               for s in grp["@library"][name][:]]
        idx = grp[name][:]
        return np.array(lib, dtype=object)[idx], lib
    # @library 없이 문자열 배열로 직접 저장된 경우
    raw = grp[name][:]
    dec = np.array([s.decode() if isinstance(s, bytes) else s for s in raw], dtype=object)
    return dec, sorted(set(dec.tolist()))


def main():
    nodes_h5 = find_nodes_h5()
    print("=" * 70)
    print(f"[1-1] nodes.h5 = {nodes_h5}")
    print("=" * 70)

    with h5py.File(nodes_h5, "r") as f:
        pop, grp = find_population_group(f)
        # --- 데이터셋(필드) 목록 덤프 ---
        print(f"\n[population] '{pop}'   그룹 = nodes/{pop}/0")
        print("[필드 목록] (@library 는 범주형 사전)")
        for k in grp.keys():
            item = grp[k]
            if isinstance(item, h5py.Dataset):
                print(f"   - {k:<22} shape={item.shape} dtype={item.dtype}")
            else:
                print(f"   - {k:<22} (group: {list(item.keys())})")

        N = grp["x"].shape[0]
        print(f"\n[1-1] 총 세포수 N = {N:,}   (참고 기대값 {EXPECT_N:,})")

        # --- 범주형 분포 ---
        dist = {}
        for field in CAT_FIELDS:
            dec, lib = decode_library(grp, field)
            if dec is None:
                print(f"\n=== {field}: (없음) ===")
                continue
            c = Counter(dec.tolist())
            dist[field] = c
            print(f"\n=== {field}  ({len(lib)}종) ===")
            for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
                print(f"   {str(k):<16} {v:>8,}  ({100*v/N:4.1f}%)")

        # --- E:I ---
        exc = inh = 0
        if "synapse_class" in dist:
            sc = dist["synapse_class"]
            exc, inh = sc.get("EXC", 0), sc.get("INH", 0)
            print(f"\n=== E:I ===  EXC={exc:,}  INH={inh:,}  "
                  f"->  {100*exc/N:.1f} : {100*inh/N:.1f}   (참고 목표 89:11)")

        # --- 좌표 범위 ---
        xyz = {ax: grp[ax][:] for ax in ("x", "y", "z")}
        print("\n=== 좌표 범위 (µm) ===")
        for ax in ("x", "y", "z"):
            d = xyz[ax]
            print(f"   {ax}: [{d.min():10.1f}, {d.max():10.1f}]   span = {d.max()-d.min():8.1f}")

        # --- orientation quaternion norm ---
        if all(f"orientation_{c}" in grp for c in "wxyz"):
            q = np.stack([grp[f"orientation_{c}"][:] for c in "wxyz"], axis=1)
            nrm = np.linalg.norm(q, axis=1)
            print(f"\n=== orientation quaternion norm: min={nrm.min():.4f} "
                  f"max={nrm.max():.4f}  (1.0 이어야 정규화) ===")
        else:
            print("\n=== orientation_wxyz 필드 없음 — 방향 인코딩 방식 재확인 필요 ===")

        layer_dec, _ = decode_library(grp, "layer")

    # --- 그림 ---
    if dist:
        fig_distributions(dist, N)
    if exc or inh:
        fig_ei_pie(exc, inh)
    if layer_dec is not None:
        fig_scatter_2d(xyz, layer_dec)
    print(f"\n[1-1] 그림 저장 완료 -> {FIG_DIR}")
    print("[1-1] 완료. 위 세포수·E:I·층/타입 분포를 확인하세요.")


def fig_distributions(dist, N):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # layer(고정 순서)
    if "layer" in dist:
        c = dist["layer"]; vals = [c.get(k, 0) for k in LAYER_ORDER]
        axes[0].bar(LAYER_ORDER, vals, color="#4C72B0")
        axes[0].set_title("소마 층 분포"); axes[0].set_ylabel("세포 수")
        for i, v in enumerate(vals):
            axes[0].text(i, v, f"{v:,}\n{100*v/N:.1f}%", ha="center", va="bottom", fontsize=8)
    # mtype
    if "mtype" in dist:
        items = sorted(dist["mtype"].items(), key=lambda kv: -kv[1])
        axes[1].barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#55A868")
        axes[1].set_title(f"m-type 분포 ({len(items)}종)"); axes[1].set_xlabel("세포 수 (로그)")
        axes[1].set_xscale("log")
    # etype
    if "etype" in dist:
        items = sorted(dist["etype"].items(), key=lambda kv: -kv[1])
        axes[2].bar([k for k, _ in items], [v for _, v in items], color="#C44E52")
        axes[2].set_title("e-type 분포"); axes[2].set_ylabel("세포 수 (로그)")
        axes[2].set_yscale("log")
        axes[2].tick_params(axis="x", rotation=45)
    fig.suptitle(f"1-1  CA1 nodes.h5 — 총 세포수 N = {N:,}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "1-1_distributions.png"), dpi=130)
    plt.close(fig)


def fig_ei_pie(exc, inh):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([exc, inh], labels=[f"흥분 EXC\n{exc:,}", f"억제 INH\n{inh:,}"],
           autopct="%1.1f%%", colors=["#DD8452", "#4C72B0"],
           startangle=90, wedgeprops=dict(edgecolor="w"))
    ax.set_title("1-1  흥분 : 억제 (참고 목표 ~89:11)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "1-1_EI_ratio.png"), dpi=130)
    plt.close(fig)


def fig_scatter_2d(xyz, layer_dec, n_sample=30000):
    """전체 CA1 배치 2D 투영(x-y, x-z) 산점도 — 층별 색(서브샘플)."""
    N = len(xyz["x"])
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(n_sample, N), replace=False)
    colors = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for (a, b, ax) in [("x", "y", axes[0]), ("x", "z", axes[1])]:
        for lyr in LAYER_ORDER:
            m = layer_dec[idx] == lyr
            if m.any():
                ax.scatter(xyz[a][idx][m], xyz[b][idx][m], s=2, alpha=0.35,
                           c=colors[lyr], label=lyr)
        ax.set_xlabel(f"{a} (µm)"); ax.set_ylabel(f"{b} (µm)")
        ax.set_title(f"CA1 배치 {a}-{b} 투영"); ax.set_aspect("equal", "datalim")
    axes[0].legend(markerscale=4, loc="best")
    fig.suptitle(f"1-1  CA1 세포 배치 (2D 투영, {len(idx):,}개 샘플)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "1-1_placement_2d.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
