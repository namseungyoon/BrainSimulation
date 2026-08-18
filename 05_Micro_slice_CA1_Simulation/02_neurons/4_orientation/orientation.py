# -*- coding: utf-8 -*-
"""
02_neurons/4_orientation/orientation.py  —  2-4: 세포 방향 적용·검증 (2-4)

세포 원장의 orientation quaternion을 실제 형태(.swc)에 적용해, 각 세포가 조직
안에서 올바른 방향으로 서는지 검증한다.
  - 정량(전 5,610세포): 정단축 apical = R(quat)·[0,1,0] → 국소 투영,
    mtype별 apical·r̂ 중앙값 / +r(SR쪽) 정렬 비율
  - 시각(표본): 대표 세포의 전체 형태를 배치+회전해 국소 u-r 단면에 렌더
    (추체 정단수상돌기가 SR(+r)로 뻗는지 육안 확인)
결과: 그림 2-4_orientation.png

재료: data/derived/window_cells.npz · config/window_layout.json · data/morphology_library
실행: python 02_neurons/4_orientation/orientation.py
"""
import os
import json
import logging
import collections

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Patch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "window_cells.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
MORPHLIB = os.path.join(ROOT, "data", "morphology_library", "morphology_library")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def load_swc(path, keep=(1, 3, 4)):
    """soma(1)/basal(3)/apical(4)만. 반환: pts(N,3), seg(부모연결 인덱스쌍)."""
    rows = np.loadtxt(path, comments="#")
    idx = rows[:, 0].astype(int); typ = rows[:, 1].astype(int)
    xyz = rows[:, 2:5]; par = rows[:, 6].astype(int)
    id2row = {i: k for k, i in enumerate(idx)}
    segs = []
    for k in range(len(idx)):
        p = par[k]
        if p in id2row and (typ[k] in keep):
            segs.append((id2row[p], k))
    return xyz, np.array(segs, int), typ


def to_local(pts, q_wxyz, soma_xyz, seed, M):
    world = soma_xyz + Rot.from_quat(q_wxyz[[1, 2, 3, 0]]).apply(pts)
    return (world - seed) @ M    # (N,3) → u,r,w


def main():
    d = np.load(NPZ, allow_pickle=True)
    uvw = d["uvw"]; xyz = d["xyz"]; Q = d["orientation_wxyz"]
    layer = d["layer"]; mt = d["mtype"]; morph = d["morphology"]
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; w = cfg["window_um"]; c = w["center_local"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])
    M = np.column_stack([L, R, Tk])

    # 정량: 정단축 apical = R(quat)·[0,1,0]
    apical = Rot.from_quat(Q[:, [1, 2, 3, 0]]).apply([0.0, 1.0, 0.0])
    ap_r = apical @ R; ap_u = apical @ L; ap_w = apical @ Tk
    print("=== 2-4 세포 방향 검증 ===")
    print(f"[전체 {len(Q):,}세포] 정단·층관통 apical·r̂ 중앙={np.median(ap_r):.3f} · "
          f"+r 정렬(>0)={100*np.mean(ap_r>0):.1f}% · |apical·û| 중앙={np.median(np.abs(ap_u)):.3f}")
    print("[mtype별 apical·r̂ 중앙값 / +r 비율]")
    for m in [x for x, _ in collections.Counter(mt).most_common()]:
        s = mt == m
        print(f"   {m:<10} {np.median(ap_r[s]):+.3f}  ({100*np.mean(ap_r[s] > 0):.0f}% +r, n={int(s.sum())})")

    fig_orientation(uvw, xyz, Q, layer, mt, morph, ap_r, c, w, seed, M)
    print(f"\n[2-4] 그림 저장 -> {FIG}/2-4_orientation.png")


def _sample(mt, uvw, layer, want_pc=15, want_int=6, seed=0):
    rng = np.random.default_rng(seed)
    pc = np.where(mt == "SP_PC")[0]
    # u축으로 고루 퍼지게 선택
    order = pc[np.argsort(uvw[pc, 0])]
    pick_pc = order[np.linspace(0, len(order) - 1, want_pc).astype(int)]
    ints = np.where(mt != "SP_PC")[0]
    pick_int = rng.choice(ints, min(want_int, len(ints)), replace=False)
    return pick_pc, pick_int


def fig_orientation(uvw, xyz, Q, layer, mt, morph, ap_r, c, w, seed, M):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.5, 1]})

    # (a) 표본 전체형태 배치+회전 (u-r 단면)
    ax = axes[0]
    pick_pc, pick_int = _sample(mt, uvw, layer)
    rmin, rmax = [0.0], [0.0]   # 그린 형태의 층관통 도달범위 추적
    def draw(ci, color, lw):
        p = os.path.join(MORPHLIB, str(morph[ci]) + ".swc")
        if not os.path.exists(p):
            return
        pts, segs, typ = load_swc(p)
        loc = to_local(pts, Q[ci], xyz[ci], seed, M)
        if len(segs):
            lcset = LineCollection([[(loc[a, 0], loc[a, 1]), (loc[b, 0], loc[b, 1])] for a, b in segs],
                                   colors=color, linewidths=lw, alpha=0.7)
            ax.add_collection(lcset)
            rmin[0] = min(rmin[0], loc[:, 1].min()); rmax[0] = max(rmax[0], loc[:, 1].max())
        ax.plot(uvw[ci, 0], uvw[ci, 1], "o", color=color, ms=3, zorder=5)
    for ci in pick_pc:
        draw(ci, "#C44E52", 0.35)
    for ci in pick_int:
        draw(ci, "#3B75AF", 0.5)

    r_lo = c["r"] - w["radial"] / 2; r_hi = c["r"] + w["radial"] / 2
    box_u0 = c["u"] - w["long"] / 2
    # soma 배치 창(800) — 굵은 점선 + 음영
    ax.add_patch(Rectangle((box_u0, r_lo), w["long"], w["radial"],
                           fill=True, fc="black", alpha=0.05, ec="black", lw=1.8, ls="--", zorder=1))
    ax.annotate(f"soma 배치 창 (층관통 {w['radial']:.0f}µm)", (c["u"], r_hi), ha="center", va="bottom",
                fontsize=9, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    # 수상돌기 도달범위 표시
    for rr, lab, va in [(rmax[0], f"정단수상돌기 도달 ~+{rmax[0]:.0f}µm (창 밖, SR/SLM 조직)", "bottom"),
                        (rmin[0], f"기저수상돌기 ~{rmin[0]:.0f}µm (SO)", "top")]:
        ax.axhline(rr, color="gray", lw=0.8, ls="-.")
        ax.annotate(lab, (box_u0 + 10, rr), ha="left", va=va, fontsize=8, color="dimgray",
                    xytext=(0, 3 if va == "bottom" else -3), textcoords="offset points")
    # 총 span 화살표(오른쪽)
    x_ar = c["u"] + w["long"] / 2 + 60
    ax.annotate("", (x_ar, rmax[0]), (x_ar, rmin[0]),
                arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.5))
    ax.text(x_ar + 12, (rmax[0] + rmin[0]) / 2, f"세포 형태 span\n~{rmax[0]-rmin[0]:.0f}µm",
            fontsize=8, color="darkgreen", va="center")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_aspect("equal"); ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm, SP=0 · +r=SR/SLM)")
    ax.set_title(f"(a) 표본 전체형태 배치+회전 (u-r 단면)\n"
                 f"체세포=창 안 / 수상돌기=조직으로 뻗음 · 추체 {len(pick_pc)}(빨강)·억제 {len(pick_int)}(파랑)")
    ax.legend(handles=[Patch(color="#C44E52", label="추체 SP_PC"), Patch(color="#3B75AF", label="억제뉴런"),
                       Patch(facecolor="black", alpha=0.05, ec="black", ls="--", label="soma 창 800µm")],
              loc="upper right", fontsize=8)

    # (b) mtype별 apical·r̂ 정렬
    ax = axes[1]
    mts = [x for x, _ in collections.Counter(mt).most_common()]
    med = [np.median(ap_r[mt == m]) for m in mts]
    cols = ["#C44E52" if m == "SP_PC" else "#3B75AF" for m in mts]
    y = np.arange(len(mts))
    ax.barh(y, med, color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(mts, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(-1, 1); ax.set_xlabel("apical·r̂ 중앙값  (+1=SR쪽 정렬)")
    ax.set_title("(b) mtype별 정단축 층관통 정렬")

    fig.suptitle("2-4  세포 방향(quaternion) 적용·검증 — 정단수상돌기 SO→SR/SLM 정렬", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2-4_orientation.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
