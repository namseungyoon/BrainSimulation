# -*- coding: utf-8 -*-
"""3-2 시냅스 생성 + 가지치기 — post 정단수상돌기 위 실제 위치에

단계   : 3-2 (파이프라인 3단계 시냅스 / 하위 2 placement)
방법   : post 정단수상돌기 전역에 접촉 후보(touch 유사)를 만들고, SR 대역 밖·과잉 접촉을
         솎아(prune) 현실적 개수만 남긴다. 남은 시냅스와 제거된 후보를 **실제 형태 위 위치**에 찍는다.
결과   : figures/3-2_syn_sites.png · figures/3-2_placement.json

실행:
  . .\\env\\activate.ps1
  & $Py04 03_synapse\\2_placement\\3-2_placement.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                          # noqa: E402
import yaml                                 # noqa: E402
from lib import plots                        # noqa: E402
from lib import cells                        # noqa: E402
from lib import morphology as mo             # noqa: E402
from lib.nrnenv import h                     # noqa: E402

SR_MIN, SR_MAX = 100.0, 300.0
N_CANDIDATE = 12
N_KEEP = 5
SEED = 20260820


def post_points_and_transform(post):
    """post 3D 점구름 + 정렬 변환. 시냅스도 같은 변환으로 옮기려고 변환을 돌려준다."""
    h.define_shape()
    xyz, typ, par = [], [], []
    for s in post.all:
        base = s.name().split(".")[-1].split("[")[0]
        tt = {"soma": mo.SOMA, "apic": mo.APICAL, "dend": mo.BASAL,
              "axon": mo.AXON, "myelin": mo.AXON}.get(base, mo.BASAL)
        first = len(xyz)
        for i in range(s.n3d()):
            xyz.append((s.x3d(i), s.y3d(i), s.z3d(i))); typ.append(tt)
            par.append(first + i - 1 if i > 0 else -1)
    m = dict(xyz=np.array(xyz, float), type=np.array(typ), parent_row=np.array(par),
             radius=np.ones(len(xyz)), index=np.arange(len(xyz)), parent=np.array(par))
    c, R = mo.align_transform(m, mode="apical")
    m["xyz"] = mo.apply_transform(m["xyz"], c, R)
    return m, c, R


def seg_xyz(seg):
    """세그먼트가 속한 구획의 3D 중점 좌표."""
    sec = seg.sec
    i = sec.n3d() // 2
    return np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)])


def candidates(post):
    """정단 전역에서 경로거리 균등하게 N_CANDIDATE 개 후보 seg."""
    h.distance(0, post.soma[0](0.5))
    apics = [s for s in post.all if ".apic" in s.name()]
    segs = [(s(0.5), h.distance(s(0.5))) for s in apics]
    segs.sort(key=lambda z: z[1])
    if len(segs) <= N_CANDIDATE:
        return segs
    idx = np.linspace(0, len(segs) - 1, N_CANDIDATE).astype(int)
    return [segs[i] for i in idx]


def prune(cands):
    in_sr = [(s, d) for s, d in cands if SR_MIN <= d <= SR_MAX]
    rm_band = [(s, d) for s, d in cands if not (SR_MIN <= d <= SR_MAX)]
    if len(in_sr) <= N_KEEP:
        return in_sr, rm_band, []
    idx = set(np.linspace(0, len(in_sr) - 1, N_KEEP).astype(int).tolist())
    kept = [in_sr[i] for i in range(len(in_sr)) if i in idx]
    rm_excess = [in_sr[i] for i in range(len(in_sr)) if i not in idx]
    return kept, rm_band, rm_excess


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "cells.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["pair"]
    pre, _ = cells.load_cell(os.path.join(REPO, "Models", cfg["pre_bundle"]), "pre")
    post, _ = cells.load_cell(os.path.join(REPO, "Models", cfg["post_bundle"]), "post")

    print("=== 3-2 시냅스 생성 + 가지치기 ===")
    m, c, R = post_points_and_transform(post)
    m_pre, _, _ = post_points_and_transform(pre)   # 같은 방식으로 pre 점구름 정렬
    cands = candidates(post)
    kept, rm_band, rm_excess = prune(cands)
    print(f"  touch 후보 {len(cands)} -> 유지 {len(kept)} (SR밖 {len(rm_band)} · 과잉 {len(rm_excess)} 제거)")
    print(f"  유지 시냅스 경로거리: {[round(d) for _, d in kept]} um")

    # 시냅스 3D 좌표를 형태와 같은 변환으로
    def proj(items):
        if not items:
            return np.empty((0, 2)), []
        p = np.array([seg_xyz(s) for s, d in items])
        p2 = mo.apply_transform(p, c, R)[:, :2]
        return p2, [d for _, d in items]
    kp, kd = proj(kept)
    bp, _ = proj(rm_band)
    ep, _ = proj(rm_excess)

    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 6.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # A: pre + post 를 함께 · post 형태 위 시냅스 위치
    xyd = m["xyz"][np.isin(m["type"], (mo.SOMA, mo.BASAL, mo.APICAL))][:, :2]
    hx = np.percentile(np.abs(xyd[:, 0]), 99.8) * 1.2
    # pre 를 왼쪽으로 offset (도해: pre 자극세포 → post 시냅스). D8: 상대 위치는 결과 무관.
    pre_xyd = m_pre["xyz"][np.isin(m_pre["type"], (mo.SOMA, mo.BASAL, mo.APICAL))][:, :2]
    pre_hx = np.percentile(np.abs(pre_xyd[:, 0]), 99.8) * 1.2
    offset = hx + pre_hx + 90
    m_pre_shift = dict(m_pre); m_pre_shift["xyz"] = m_pre["xyz"].copy()
    m_pre_shift["xyz"][:, 0] -= offset

    mo.render(axA, m_pre_shift, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False)
    mo.render(axA, m, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False)

    ylo = min(np.percentile(xyd[:, 1], 0.3), np.percentile(pre_xyd[:, 1], 0.3)) - 30
    yhi = max(np.percentile(xyd[:, 1], 99.8), np.percentile(pre_xyd[:, 1], 99.8)) + 70
    axA.set_xlim(-offset - pre_hx - 30, hx + 30)
    axA.set_ylim(ylo, yhi)
    axA.set_aspect("equal", adjustable="box"); axA.set_xticks([]); axA.set_yticks([]); axA.grid(False)
    for s in axA.spines.values():
        s.set_color("#dddddd")
    # SR 대역은 post 쪽만
    axA.fill_between([-hx, hx], SR_MIN, SR_MAX, color="#ffb300", alpha=0.13, zorder=0)
    axA.text(hx, (SR_MIN+SR_MAX)/2, " SR", fontsize=8.5, color="#b26a00", va="center", ha="left")

    if len(bp): axA.scatter(bp[:, 0], bp[:, 1], s=55, marker="x", color="#9e9e9e", lw=1.7, zorder=5)
    if len(ep): axA.scatter(ep[:, 0], ep[:, 1], s=55, marker="x", color="#9e9e9e", lw=1.7, zorder=5)
    if len(kp): axA.scatter(kp[:, 0], kp[:, 1], s=230, marker="*", color="#7b1fa2",
                            edgecolor="white", lw=1.3, zorder=6)

    # pre 소마 → 유지 시냅스 연결(도해)
    pre_soma = m_pre_shift["xyz"][m_pre_shift["type"] == mo.SOMA].mean(axis=0)
    if len(kp):
        tgt = kp.mean(axis=0)
        axA.annotate("", xy=(tgt[0], tgt[1]), xytext=(pre_soma[0], pre_soma[1]),
                     arrowprops=dict(arrowstyle="-|>", color="#7b1fa2", lw=1.8,
                                     linestyle=(0, (5, 3)), shrinkA=6, shrinkB=12), zorder=4)
        axA.text((pre_soma[0]+tgt[0])/2, (pre_soma[1]+tgt[1])/2 + 35,
                 "NetCon+지연", fontsize=8.5, color="#7b1fa2", ha="center", va="bottom",
                 fontweight="bold", bbox=dict(fc="white", ec="#7b1fa2", alpha=0.9,
                                              boxstyle="round,pad=0.3"))

    # 세포 라벨 (제목과 겹치지 않게 살짝 아래)
    post_soma = m["xyz"][m["type"] == mo.SOMA].mean(axis=0)
    axA.text(pre_soma[0], yhi - 45, f"pre (자극)\n{cfg['pre_tag']}", fontsize=8.5, ha="center",
             va="top", color="#212121", fontweight="bold")
    axA.text(post_soma[0], yhi - 45, f"post (기록)\n{cfg['post_tag']}", fontsize=8.5, ha="center",
             va="top", color="#212121", fontweight="bold")

    axA.set_title("A. pre → post 시냅스 배치 (실제 형태·위치)", fontsize=10.5, loc="left", pad=18)
    axA.scatter([], [], s=200, marker="*", color="#7b1fa2", label=f"유지 시냅스 {len(kept)}")
    axA.scatter([], [], s=55, marker="x", color="#9e9e9e", label=f"가지치기 제거 {len(rm_band)+len(rm_excess)}")
    axA.legend(loc="lower left", fontsize=8, framealpha=0.9)
    mo.scalebar(axA, 200, "200 um", loc=(0.72, 0.03))

    # B: 경로거리 분포 (후보 vs 유지)
    all_d = [d for _, d in cands]
    axB.hist(all_d, bins=np.arange(0, max(all_d)+50, 50), color="#e0e0e0",
             edgecolor="#bbb", label=f"touch 후보 {len(cands)}")
    axB.hist(kd, bins=np.arange(0, max(all_d)+50, 50), color="#7b1fa2",
             alpha=0.85, label=f"가지치기 후 {len(kept)}")
    axB.axvspan(SR_MIN, SR_MAX, color="#ffb300", alpha=0.16, zorder=0)
    axB.set_xlabel("소마 경로거리 (um)"); axB.set_ylabel("시냅스 수")
    axB.set_title("B. 가지치기 — SR 대역 밖·과잉 제거", fontsize=10.5, loc="left")
    axB.legend(fontsize=9)
    axB.text(SR_MIN+5, axB.get_ylim()[1]*0.9, "SR 대역", fontsize=8.5, color="#b26a00", va="top")

    fig.suptitle(f"3-2  두 뉴런 연결의 시냅스 생성 + 가지치기 (touch {len(cands)} -> 유지 {len(kept)})",
                 fontsize=12.5, y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.15)
    plots.stamp(fig, f"3-2 | SR 대역 {SR_MIN:.0f}~{SR_MAX:.0f}um · N_keep={N_KEEP} · 실제 seg 3D 위치")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-2_syn_sites.png")

    out = dict(post=cfg["post_tag"], sr_band=[SR_MIN, SR_MAX],
               candidates=len(cands), kept=len(kept),
               removed_band=len(rm_band), removed_excess=len(rm_excess),
               kept_dist_um=[round(d, 1) for _, d in kept],
               kept_sections=[s.sec.name().split(".")[-1] for s, d in kept])
    jpath = os.path.join(outdir, "3-2_placement.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    print("\n[통과] 3-2 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
