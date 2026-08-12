# -*- coding: utf-8 -*-
"""
06_orientation/render_model_cell.py  —  ★시뮬레이션이 실제로 계산하는 세포 형태를 그린다

왜 필요했나 (2026-08-11)
------------------------
기존 형태 그림(V2d-1~8)은 전부 `data/morphology_library/*.swc` **원본 파일**을 그린다.
그런데 NEURON 은 세포를 만들 때 BBP 템플릿의 `replace_axon()` 을 실행해
**원본 축삭을 삭제하고 60µm 토막 2개로 교체**한다. 즉 그림과 시뮬레이션의 형태가 다르다.

이 스크립트는 **NEURON 이 실제로 만든 세포**에서 3D 좌표를 뽑아 그린다.
원본 SWC 를 나란히 놓아 무엇이 버려졌는지 눈으로 확인할 수 있게 한다.

산출: figures/V2d_9_model_vs_original.png

실행: python 06_orientation/render_model_cell.py
"""
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # like_slice_CA1
REPO = os.path.dirname(ROOT)                       # 03_BrainSimulator
sys.path.insert(0, os.path.join(REPO, "shared"))
FIG = os.path.join(HERE, "figures")

# 구획 색 — SWC type 코드와 동일하게 맞춘다 (1 소마 / 2 축삭 / 3 기저 / 4 정점)
COL = {1: "#000000", 2: "#e08214", 3: "#2b6cb0", 4: "#2f8f4e"}
NAME = {1: "소마", 2: "축삭", 3: "기저수상", 4: "정점수상"}


def sec_type(name):
    """NEURON section 이름 -> SWC type 코드."""
    if "soma" in name:
        return 1
    if "axon" in name or "myelin" in name:
        return 2
    if "apic" in name:
        return 4
    return 3


def recenter(segs, cx, cy):
    """소마 중심을 원점으로 옮긴다 — 두 그림을 같은 자리에 겹쳐 비교하기 위해."""
    return [(x0 - cx, y0 - cy, x1 - cx, y1 - cy, t) for x0, y0, x1, y1, t in segs]


def soma_center(segs):
    pts = [(x0, y0) for x0, y0, _, _, t in segs if t == 1]
    if not pts:
        pts = [(x0, y0) for x0, y0, _, _, _ in segs]
    a = np.asarray(pts, float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def neuron_segments(cell, h):
    """NEURON 세포의 모든 section 에서 (x, y, type) 선분 목록을 뽑는다."""
    h.define_shape()                                # 3D 좌표가 없으면 생성
    segs = []
    for sec in cell.all:
        t = sec_type(sec.name())
        n = sec.n3d()
        for i in range(n - 1):
            segs.append((sec.x3d(i), sec.y3d(i), sec.x3d(i + 1), sec.y3d(i + 1), t))
    return segs


def swc_segments(path):
    """원본 SWC 파일에서 (x, y, type) 선분 목록을 뽑는다."""
    raw = np.loadtxt(path, comments="#")
    idx = {int(r[0]): k for k, r in enumerate(raw)}
    segs = []
    for r in raw:
        p = int(r[6])
        if p < 0 or p not in idx:
            continue
        q = raw[idx[p]]
        segs.append((q[2], q[3], r[2], r[3], int(r[1])))
    return segs


def draw(ax, segs, title, lim):
    """축삭은 옅게 뒤에, 수상돌기는 진하게 앞에 그린다."""
    for order, alpha, lw in ((2, 0.5, 0.5), (3, 0.9, 0.6), (4, 0.9, 0.6), (1, 1.0, 3.0)):
        sub = [s for s in segs if s[4] == order]
        if not sub:
            continue
        for x0, y0, x1, y1, _ in sub:
            ax.plot([x0, x1], [y0, y1], color=COL[order], alpha=alpha, lw=lw,
                    solid_capstyle="round")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_xlabel("소마 기준 x (um)", fontsize=10)
    ax.grid(alpha=0.15, lw=0.5)


def main():
    from common.cell_loader import load_cell
    from common.nrn_env import h

    md = glob.glob(os.path.join(REPO, "shared", "models", "pyramidal", "*_model_files"))
    if not md:
        print("추체세포 모델을 찾지 못했습니다."); return 2
    md = os.path.abspath(md[0])
    swc = os.path.join(md, "morphology", "morphology.swc")

    cell, _ = load_cell(md, gid=0)
    nseg_ = neuron_segments(cell, h)
    sseg_ = swc_segments(swc)

    # 통계 (캡션용) — 길이는 NEURON section 에서 직접 합산
    tot = {}
    for sec in cell.all:
        t = sec_type(sec.name())
        tot.setdefault(t, [0, 0.0])
        tot[t][0] += 1
        tot[t][1] += sec.L
    raw = np.loadtxt(swc, comments="#")
    n_ax_raw = int((raw[:, 1] == 2).sum())

    # 소마를 원점으로 정렬 → 두 그림을 같은 범위로 그려야 차이가 보인다
    sseg_ = recenter(sseg_, *soma_center(sseg_))
    nseg_ = recenter(nseg_, *soma_center(nseg_))
    allxy = np.asarray([(s[0], s[1]) for s in sseg_], float)
    pad = 60.0
    lim = (allxy[:, 0].min() - pad, allxy[:, 0].max() + pad,
           allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)

    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    draw(axes[0], sseg_,
         f"① 원본 형태 파일 (morphology.swc)\n축삭 {n_ax_raw:,}점 · 최대 820um 까지 뻗음", lim)
    draw(axes[1], nseg_,
         f"② NEURON 이 실제로 계산하는 세포\n축삭 {tot[2][0]}개 section · {tot[2][1]:.0f}um 로 교체 (99% 삭제)", lim)
    axes[0].set_ylabel("소마 기준 y (um)", fontsize=10)

    handles = [Line2D([], [], color=COL[k], lw=2.5, label=NAME[k]) for k in (1, 2, 3, 4)]
    axes[0].legend(handles=handles, fontsize=10, loc="upper left", framealpha=0.9)

    fig.suptitle("V2d-9  추체세포 — 원본 형태 vs 시뮬레이션 형태 (BBP replace_axon)",
                 fontsize=14)
    fig.text(0.5, 0.028,
             "정점수상 {a}개 section {b:,.0f}um · 기저수상 {c}개 {d:,.0f}um 는 ①②가 동일하다 — 원본 그대로 계산된다.\n"
             "축삭(주황)만 60um 토막 2개로 교체된다. 교체 이유: 축삭 채널 파라미터 7개가 이 60um 를 전제로 "
             "최적화돼 있다 (gbar_nax 0.1509 = 정점수상의 6.3배 -> 활동전위 시작부 역할).\n"
             "★세포 간 전달은 축삭이 아니라 NetCon 이벤트(소마 -20mV 통과 -> 고정 지연 1.0ms)로 이뤄진다.".format(
                 a=tot[4][0], b=tot[4][1], c=tot[3][0], d=tot[3][1]),
             ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.085, 1, 0.955])
    out = os.path.join(FIG, "V2d_9_model_vs_original.png")
    fig.savefig(out, dpi=140)
    print(f"[저장] {out}")
    for t in (1, 2, 3, 4):
        if t in tot:
            print(f"  {NAME[t]:6s} section {tot[t][0]:4d}개 · {tot[t][1]:9,.1f} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
