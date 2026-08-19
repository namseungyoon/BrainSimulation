# -*- coding: utf-8 -*-
"""
03_network/1_connectome/sc_fibers.py  —  3-1 보강: SC 섬유(가상 CA3 축삭) 정체성 부여

각 SC 시냅스에 fiber_id(어느 가상 CA3 축삭에서 왔는가)를 부여한다.
모델: 가상 축삭 = 종축(u)을 따라 흐르는 선. 섬유공간 = (r, w)(깊이·두께).
  - 축삭 1개 = 특정 (r,w)에 자리, u 전체를 지나며 여러 세포에 시냅스 발산.
  - 자극(E3) = E3의 (r,w) 근처를 지나는 축삭 모집 → 그 축삭의 모든 시냅스 발화.
효과: 정체성 O = 상관된·공간분산 활성(입력특이성 LTP 가능) / X = E3 국소 블롭.

결과: data/derived/sc_fibers.npz(fiber_id·fiber_rw) + 그림 3-1_fiber_compare.png
실행: python 03_network/1_connectome/sc_fibers.py [--nfiber 10000] [--radius 150]
"""
import os
import sys
import json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")

NFIBER = int(sys.argv[sys.argv.index("--nfiber") + 1]) if "--nfiber" in sys.argv else 10000
RADIUS = float(sys.argv[sys.argv.index("--radius") + 1]) if "--radius" in sys.argv else 150.0


def main():
    d = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    uvw = d["uvw"].astype(float)          # (N,3) 국소 u,r,w
    post = d["post_gid"]; dist_e3 = d["dist_e3"].astype(float)
    N = len(uvw)
    U, R, W = uvw[:, 0], uvw[:, 1], uvw[:, 2]

    # 섬유공간 = (r,w). 앵커 = 데이터에서 무작위 추출.
    rng = np.random.default_rng(0)
    seed_idx = rng.choice(N, NFIBER, replace=False)
    anchors = uvw[seed_idx][:, [1, 2]]     # (NFIBER,2) = (r,w)
    tree = cKDTree(anchors)
    _, fiber_id = tree.query(uvw[:, [1, 2]], k=1)   # 각 시냅스 → 최근접 섬유

    # 통계: 섬유당 시냅스 수 · 발산(섬유가 닿는 세포 수)
    cnt = np.bincount(fiber_id, minlength=NFIBER)
    ncell = np.array([len(np.unique(post[fiber_id == f])) for f in rng.choice(NFIBER, 300, replace=False)])

    np.savez_compressed(os.path.join(DERIVED, "sc_fibers.npz"),
                        fiber_id=fiber_id.astype(np.int32), fiber_rw=anchors.astype(np.float32),
                        n_fiber=NFIBER)

    print(f"=== 3-1 SC 섬유 정체성 부여 ===")
    print(f"[섬유] {NFIBER:,}개 · 시냅스 {N:,}개 → 섬유당 평균 {cnt.mean():.1f}개 "
          f"(중앙 {np.median(cnt):.0f} · 최대 {cnt.max()})")
    print(f"[발산] 섬유 1개가 닿는 세포 수: 평균 {ncell.mean():.1f} (표본300)")

    # E3 국소좌표 (r,w)
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; seed = np.array(fr["seed"])
    Mrows = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    e3 = d["e3_xyz"]; e3loc = (e3 - seed) @ Mrows
    e3u, e3r, e3w = e3loc

    # 자극 모집: 정체성 X = E3 3D반경 내 시냅스 / 정체성 O = 그 시냅스가 속한 섬유의 전체 시냅스
    near = dist_e3 < RADIUS
    fibers_hit = np.unique(fiber_id[near])
    ident = np.isin(fiber_id, fibers_hit)
    print(f"[자극 R={RADIUS:.0f}µm] 정체성X: 시냅스 {near.sum():,}개·세포 {len(np.unique(post[near])):,} "
          f"| 정체성O: 섬유 {len(fibers_hit):,}개→시냅스 {ident.sum():,}개·세포 {len(np.unique(post[ident])):,}")

    # 그림: u-r 평면, 정체성 유무 비교
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    s = rng.choice(N, min(60000, N), replace=False)
    for a, mask, ttl, sub in [
        (ax[0], near, "정체성 없음 — E3 국소만 발화", f"활성 시냅스 {near.sum():,} · 세포 {len(np.unique(post[near])):,}"),
        (ax[1], ident, "정체성 있음 — 모집 축삭 전체 발화", f"활성 시냅스 {ident.sum():,} · 세포 {len(np.unique(post[ident])):,} (발산)")]:
        a.scatter(U[s], R[s], s=1.5, c="#cccccc", alpha=0.35, linewidths=0)
        ms = mask[s]
        a.scatter(U[s][ms], R[s][ms], s=4, c="#C44E52", alpha=0.7, linewidths=0)
        a.scatter([e3u], [e3r], s=260, marker="*", c="red", edgecolors="black", zorder=6)
        a.annotate("E3", (e3u, e3r), fontsize=11, ha="center", va="bottom", xytext=(0, 10), textcoords="offset points")
        a.set_xlabel("종축 u (µm)"); a.set_title(ttl + "\n" + sub)
    ax[0].set_ylabel("층관통 r (µm)")
    fig.suptitle(f"3-1  SC 자극 모집 — 섬유 정체성 유무 비교 (R={RADIUS:.0f}µm · 섬유 {NFIBER:,}개)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3-1_fiber_compare.png"), dpi=130)
    plt.close(fig)
    print(f"[그림] -> {FIG}/3-1_fiber_compare.png")


if __name__ == "__main__":
    main()
