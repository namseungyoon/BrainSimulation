# -*- coding: utf-8 -*-
"""3-2 시냅스 생성 + 가지치기 — 두 세포를 인접 배치하고 접촉점에 시냅스

단계   : 3-2 (파이프라인 3단계 시냅스 / 하위 2 placement)
방법   : pre·post 를 3D 로 인접 배치(수상돌기 필드가 SR 대역에서 겹치게)하고,
         post 정단수상돌기(SR 대역)가 pre 구조와 **접촉(apposition)** 하는 지점을 찾는다(touch).
         과잉 접촉을 솎아(prune) 현실적 개수만 남긴다. 각 시냅스의 전도지연은 pre 소마->시냅스
         3D 거리 / 전도속도 로 계산한다 -- 임의값이 아니라 배치 기하에서 나온다.
★사실  : NEURON 은 전달 자체는 접촉해도 NetCon 으로 계산한다(모든 상세 시뮬레이터 공통).
         이 단계가 물리적으로 정하는 것은 (1) 시냅스 위치 (2) 전도지연 이다.
         축삭은 스텁/원본 모두 ~87um 라 접촉 구조로 쓸 수 없어, 인접은 두 수상돌기 필드로 잡는다.
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
TOUCH_UM = 12.0          # apposition 반경(투영 2D). 이 안에 pre 구조가 있으면 접촉으로 본다
OVERLAP_FRAC = 0.45      # 두 수상돌기 필드를 얼마나 겹칠지 (SR 대역 인접)
N_KEEP = 5               # 가지치기 후 남길 접촉 수
V_COND = 0.5             # 전도속도 um/us = 0.5 m/s (CA3 SC 무수초 근사)
SYN_DELAY = 0.5          # 시냅스 지연 ms (전도지연에 더함)


def points_and_transform(cell):
    """세포 3D 점구름(형태) + 정렬 변환. 세그먼트 좌표도 같은 변환으로 옮기려고 변환 반환."""
    h.define_shape()
    xyz, typ, par = [], [], []
    for s in cell.all:
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


def apical_sr_segs(post, c, R):
    """post 정단수상돌기 중 SR 대역(경로거리 100~300) 세그먼트 + 정렬좌표 중점."""
    h.distance(0, post.soma[0](0.5))
    out = []
    for s in post.all:
        if ".apic" not in s.name():
            continue
        d = h.distance(s(0.5))
        if not (SR_MIN <= d <= SR_MAX):
            continue
        i = s.n3d() // 2
        p = mo.apply_transform(np.array([s.x3d(i), s.y3d(i), s.z3d(i)]), c, R)
        out.append((s(0.5), d, p))           # (seg, 경로거리, 정렬3D좌표)
    return out


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "cells.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["pair"]
    pre, _ = cells.load_cell(os.path.join(REPO, "Models", cfg["pre_bundle"]), "pre")
    post, _ = cells.load_cell(os.path.join(REPO, "Models", cfg["post_bundle"]), "post")

    print("=== 3-2 인접 배치 + 접촉 시냅스 + 가지치기 ===")
    m_post, cP, RP = points_and_transform(post)
    m_pre, _, _ = points_and_transform(pre)

    # 두 세포를 인접 배치: pre 를 왼쪽으로 옮기되 수상돌기 필드가 SR 대역에서 겹치게.
    post_dxy = m_post["xyz"][np.isin(m_post["type"], (mo.SOMA, mo.BASAL, mo.APICAL))]
    pre_dxy = m_pre["xyz"][np.isin(m_pre["type"], (mo.SOMA, mo.BASAL, mo.APICAL))]
    post_xmin = post_dxy[:, 0].min()
    pre_xmax = pre_dxy[:, 0].max()
    # pre 를 왼쪽으로 옮겨 pre 오른쪽 필드가 post 왼쪽 필드를 OVERLAP_FRAC 만큼 겹치게.
    # shift 가 작을수록 더 많이 겹친다. (pre_xmax - post_xmin) 은 '딱 맞닿는' 이동량.
    shift = (pre_xmax - post_xmin) * (1.0 - OVERLAP_FRAC)
    m_pre["xyz"][:, 0] -= shift
    pre_pts = m_pre["xyz"]                    # 접촉 탐지용 pre 전체 점
    pre_soma3d = m_pre["xyz"][m_pre["type"] == mo.SOMA].mean(axis=0)

    # 접촉 탐지: 각 post SR 정단 세그먼트에서 가장 가까운 pre 점까지의 2D 투영 거리.
    # (z 무시한 투영 apposition — 벤치 도해용. 3D 정밀 접촉은 조직 스케일 몫)
    sr_segs = apical_sr_segs(post, cP, RP)
    scored = []
    for seg, d, p in sr_segs:
        mind = float(np.hypot(pre_pts[:, 0] - p[0], pre_pts[:, 1] - p[1]).min())
        scored.append(dict(seg=seg, path=d, pos=p, touch=mind))
    n_touch = sum(1 for s in scored if s["touch"] <= TOUCH_UM)
    print(f"  SR 정단 세그먼트 {len(sr_segs)}개 · 접촉(<{TOUCH_UM:.0f}um) {n_touch}개")

    cands = [s for s in scored if s["touch"] <= TOUCH_UM]
    if len(cands) < N_KEEP:
        # 접촉이 부족하면 가장 가까운(apposition 최소) 지점을 채워 N_KEEP 확보
        scored.sort(key=lambda z: z["touch"])
        cands = scored[:max(N_KEEP * 2, len(cands))]
        print(f"  접촉 부족 -> 가장 가까운 지점으로 후보 {len(cands)}개 채움 "
              f"(최소 접촉거리 {scored[0]['touch']:.1f}um)")

    # 가지치기: 경로거리 균등하게 N_KEEP 개 유지
    cands.sort(key=lambda z: z["path"])
    if len(cands) <= N_KEEP:
        kept = cands; pruned = []
    else:
        idx = set(np.linspace(0, len(cands) - 1, N_KEEP).astype(int).tolist())
        kept = [cands[i] for i in range(len(cands)) if i in idx]
        pruned = [cands[i] for i in range(len(cands)) if i not in idx]

    # 각 시냅스 전도지연 = pre 소마->시냅스 3D 거리 / 전도속도 + 시냅스지연
    for k in kept:
        dist3d = float(np.linalg.norm(k["pos"] - pre_soma3d))
        k["dist3d"] = dist3d
        k["delay"] = dist3d / (V_COND * 1000.0) + SYN_DELAY   # um / (um/ms) ; V_COND um/us*1000=um/ms
    print(f"  유지 {len(kept)}개 · 접촉거리 {[round(k['touch'],1) for k in kept]} um")
    print(f"  전도지연 {[round(k['delay'],2) for k in kept]} ms (pre소마->시냅스 거리 기반)")

    # ---- 그림 ----
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 6.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    mo.render(axA, m_pre, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False)
    mo.render(axA, m_post, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False)
    allx = np.concatenate([m_pre["xyz"][:, 0], m_post["xyz"][:, 0]])
    ally = np.concatenate([m_pre["xyz"][:, 1], m_post["xyz"][:, 1]])
    axA.set_xlim(allx.min() - 40, allx.max() + 40)
    axA.set_ylim(np.percentile(ally, 0.3) - 30, np.percentile(ally, 99.8) + 60)
    axA.set_aspect("equal", adjustable="box"); axA.set_xticks([]); axA.set_yticks([]); axA.grid(False)
    for s in axA.spines.values():
        s.set_color("#dddddd")
    # SR 대역 (post 기준, 전체 폭)
    axA.axhspan(SR_MIN, SR_MAX, color="#ffb300", alpha=0.10, zorder=0)
    axA.text(allx.max()+35, (SR_MIN+SR_MAX)/2, "SR", fontsize=9, color="#b26a00",
             va="center", ha="right")
    if pruned:
        pp = np.array([k["pos"][:2] for k in pruned])
        axA.scatter(pp[:, 0], pp[:, 1], s=55, marker="x", color="#9e9e9e", lw=1.7, zorder=5)
    if kept:
        kp = np.array([k["pos"][:2] for k in kept])
        axA.scatter(kp[:, 0], kp[:, 1], s=250, marker="*", color="#7b1fa2",
                    edgecolor="white", lw=1.3, zorder=6)
    axA.set_title("A. 두 세포 인접 배치 + 접촉점 시냅스 (touch)", fontsize=10.5, loc="left")
    axA.text(pre_soma3d[0], np.percentile(m_pre["xyz"][:, 1], 99.8)+40, f"pre (자극)",
             fontsize=9, ha="center", color="#212121", fontweight="bold")
    post_soma3d = m_post["xyz"][m_post["type"] == mo.SOMA].mean(axis=0)
    axA.text(post_soma3d[0], np.percentile(m_post["xyz"][:, 1], 99.8)+40, f"post (기록)",
             fontsize=9, ha="center", color="#212121", fontweight="bold")
    axA.scatter([], [], s=220, marker="*", color="#7b1fa2", label=f"유지 시냅스 {len(kept)}")
    axA.scatter([], [], s=55, marker="x", color="#9e9e9e", label=f"가지치기 제거 {len(pruned)}")
    axA.legend(loc="lower left", fontsize=8, framealpha=0.9)
    mo.scalebar(axA, 200, "200 um", loc=(0.72, 0.03))

    # B: 시냅스별 접촉거리 & 전도지연 (표 대신 막대)
    ypos = np.arange(len(kept))
    labels = [f"{round(k['path'])}um" for k in kept]
    delays = [k["delay"] for k in kept]
    axB.barh(ypos, delays, color="#7b1fa2", alpha=0.85)
    axB.set_yticks(ypos); axB.set_yticklabels(labels, fontsize=9)
    axB.invert_yaxis()
    axB.set_xlabel("전도지연 (ms) = 거리/속도 + 시냅스지연")
    axB.set_ylabel("시냅스 (post 소마 경로거리)")
    axB.set_title("B. 접촉 기하에서 계산한 전도지연", fontsize=10.5, loc="left")
    for i, k in enumerate(kept):
        axB.text(delays[i] + 0.02, i, f"{delays[i]:.2f} ms  (거리 {k['dist3d']:.0f}um)",
                 va="center", fontsize=8, color="#4a148c")
    axB.set_xlim(0, max(delays) * 1.45)

    fig.suptitle(f"3-2  두 뉴런 인접 배치 · 접촉 시냅스 · 가지치기 (유지 {len(kept)})",
                 fontsize=12.5, y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.18)
    plots.stamp(fig, f"3-2 | 접촉<{TOUCH_UM:.0f}um · 전도속도 {V_COND}um/us · 지연=거리기반 · SR {SR_MIN:.0f}~{SR_MAX:.0f}um")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-2_syn_sites.png")

    out = dict(pre=cfg["pre_tag"], post=cfg["post_tag"], sr_band=[SR_MIN, SR_MAX],
               touch_um=TOUCH_UM, overlap_frac=OVERLAP_FRAC,
               v_cond_um_per_us=V_COND, syn_delay_ms=SYN_DELAY,
               sr_apical_segs=len(sr_segs), touch_candidates=len(cands), kept=len(kept),
               synapses=[dict(path_um=round(k["path"], 1), touch_um=round(k["touch"], 1),
                              dist3d_um=round(k["dist3d"], 1), delay_ms=round(k["delay"], 2),
                              section=k["seg"].sec.name().split(".")[-1]) for k in kept])
    jpath = os.path.join(outdir, "3-2_placement.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    print("\n[통과] 3-2 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
