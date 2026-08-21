# -*- coding: utf-8 -*-
"""3-2 시냅스 생성 + 가지치기 — 방사축 정렬 + 회전각 탐색으로 인접 배치

단계   : 3-2 (파이프라인 3단계 시냅스 / 하위 2 placement)
방법   : (1) 두 세포를 정단(방사)축 +y 로 정렬 = 조직의 실제 방위(CA1 PC 는 방사축을 따라 정렬).
         (2) 소마를 방사축에 수직으로 거리 L 만큼 벌려 인접.
         (3) pre 를 방사축(+y) 둘레로 회전각 θ 스윕 -> 두 수상돌기 필드의 접촉(apposition)이
             최대가 되는 θ* 선택. (방위각은 생물학적으로 자유 파라미터)
         (4) θ* 배치에서 ★PC->PC 표적 구역★ 안에서 pre 와 가장 가까운 지점에 시냅스, 나머지는 가지치기.
         (5) 전도지연 = pre소마->시냅스 3D 거리 / 전도속도 + 시냅스지연.

★표적 구역·개수는 PC->PC 문헌 근거다 (D10):
  - Deuchars & Thomson 1996 Neuroscience 74:1009 (PMID 8895869) — CA1 추체세포 989쌍 중
    단일시냅스 연결 9개(희소). 완전 재구성된 1쌍에서 전시냅스 축삭이 post 의 **3차 기저수상돌기**에
    접촉 **2개**(스파인 1·shaft 1). 짝펄스 억압 확인.
  - Crepel, Khazipov & Ben-Ari 1997 J Neurophysiol 77:2071 (PMID 9114256) — CA1 국소 재귀
    축삭은 stratum oriens 를 지나고(oriens TTX 로 다시냅스 성분 감소), 전류원밀도상 응답은
    **정단 근위 50~150um**(추체층 아래 radiatum)에서 생성.
  - Ecker 2020 Fig.3b — E->E 연결당 시냅스 수 1.3.
  => 표적 = 기저수상돌기 전체 + 정단 근위 50~150um · 유지 개수 = 2.
  ⚠️ 이전 판은 SC(Schaffer collateral) 전제로 정단 SR 100~300um 에 5개를 놓았다. SC 는 이 벤치의
     연결이 아니므로(D9) 폐기했다.
근거   : docs/DECISIONS.md D8(방사축 정렬·회전) · D9(PC->PC) · D10(표적 구역·개수)
★사실  : 04는 아틀라스를 로드하지 않으므로(독립 트랙) 세포 고유 정단축을 방사축 대응물로 쓴다.
         전달 자체는 NetCon(불가피). 이 단계가 정하는 것은 위치·방위·지연이다.
결과   : figures/3-2_syn_sites.png · figures/3-2_rotation.png · figures/3-2_placement.json

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

# PC->PC 표적 구역 (D10, 문헌 근거는 위 docstring)
APIC_PROX = (50.0, 150.0)   # 정단 근위 (Crepel 1997 CSD)
USE_BASAL = True            # 기저수상돌기 전체 (Deuchars & Thomson 1996 EM)
SOMA_LATERAL_L = 120.0    # 소마-소마 측방거리 um (모델링 값, 수상돌기 필드 겹침 범위)
TOUCH_R = 10.0            # 접촉(apposition) 반경 um (3D)
ANGLE_STEP = 10           # 회전 스윕 간격 도
N_KEEP = 2                # Deuchars 1996 재구성 쌍 = 접촉 2개 (Ecker E->E 평균 1.3)
V_COND = 0.5             # 전도속도 um/us = 0.5 m/s
SYN_DELAY = 0.5          # 시냅스 지연 ms


def points_and_transform(cell):
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


def target_segs(post, c, R):
    """PC->PC 표적 구역의 post 세그먼트 (D10): 기저수상돌기 전체 + 정단 근위 50~150um."""
    h.distance(0, post.soma[0](0.5))
    out = []
    for s in post.all:
        nm = s.name()
        if ".dend" in nm:
            dom = "basal" if USE_BASAL else None
        elif ".apic" in nm:
            d0 = h.distance(s(0.5))
            dom = "apical" if APIC_PROX[0] <= d0 <= APIC_PROX[1] else None
        else:
            dom = None
        if dom is None:
            continue
        d = h.distance(s(0.5))
        i = s.n3d() // 2
        p = mo.apply_transform(np.array([s.x3d(i), s.y3d(i), s.z3d(i)]), c, R)
        out.append((s(0.5), d, p, dom))
    return out


def Ry(deg):
    """정단(방사)축 +y 둘레 회전행렬."""
    t = np.deg2rad(deg)
    ct, st = np.cos(t), np.sin(t)
    return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])


def place_pre(pre_pts0, deg, L):
    """pre 점구름을 방사축 둘레 deg 회전 후 측방으로 -L 이동."""
    p = pre_pts0 @ Ry(deg).T
    p = p.copy(); p[:, 0] -= L
    return p


def apposition_count(pre_pts, sr_pos, R):
    """post SR 세그먼트 중 pre 점과 3D 거리 R 안에 드는 개수 + 평균 최소거리."""
    mind = np.array([np.linalg.norm(pre_pts - p, axis=1).min() for p in sr_pos])
    return int((mind <= R).sum()), float(mind.mean()), mind


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "cells.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["pair"]
    pre, _ = cells.load_cell(os.path.join(REPO, "Models", cfg["pre_bundle"]), "pre")
    post, _ = cells.load_cell(os.path.join(REPO, "Models", cfg["post_bundle"]), "post")

    print("=== 3-2 방사축 정렬 + 회전각 탐색 + 접촉 시냅스 ===")
    m_post, cP, RP = points_and_transform(post)
    m_pre0, _, _ = points_and_transform(pre)
    pre_pts0 = m_pre0["xyz"].copy()           # 회전 전(소마 원점, 정단 +y)

    sr_segs = target_segs(post, cP, RP)
    sr_pos = np.array([p for _, _, p, _ in sr_segs])
    n_bas = sum(1 for t in sr_segs if t[3] == "basal")
    print(f"  PC->PC 표적 세그먼트 {len(sr_segs)}개 "
          f"(기저 {n_bas} · 정단근위 {len(sr_segs)-n_bas}) · 소마-소마 측방 L={SOMA_LATERAL_L:.0f}um")

    # 회전각 스윕: 방사축 둘레 θ 마다 접촉 개수
    angles = np.arange(0, 360, ANGLE_STEP)
    counts, meandist = [], []
    for a in angles:
        pp = place_pre(pre_pts0, a, SOMA_LATERAL_L)
        n, md, _ = apposition_count(pp, sr_pos, TOUCH_R)
        counts.append(n); meandist.append(md)
    counts = np.array(counts); meandist = np.array(meandist)
    # 접촉 최대(동률이면 평균거리 최소) 각
    best = int(np.lexsort((meandist, -counts))[0])
    theta = float(angles[best])
    print(f"  회전 스윕 {len(angles)}각 · 최적 θ*={theta:.0f}도 "
          f"(접촉 {counts[best]}개 · 평균최소거리 {meandist[best]:.1f}um)")

    # θ* 배치에서 시냅스
    pre_pts = place_pre(pre_pts0, theta, SOMA_LATERAL_L)
    pre_soma3d = pre_pts[m_pre0["type"] == mo.SOMA].mean(axis=0)
    _, _, mind = apposition_count(pre_pts, sr_pos, TOUCH_R)
    order = np.argsort(mind)                  # 가까운 순
    # 가지치기: 가까운 후보 중 경로거리 균등하게 N_KEEP
    cand_idx = order[:max(N_KEEP * 2, N_KEEP)]
    cand = sorted(cand_idx, key=lambda i: sr_segs[i][1])
    keep_i = [cand[j] for j in np.linspace(0, len(cand) - 1, N_KEEP).astype(int)]
    prune_i = [i for i in cand_idx if i not in keep_i]

    syn = []
    for i in keep_i:
        seg, d, p, dom = sr_segs[i]
        dist3d = float(np.linalg.norm(p - pre_soma3d))
        syn.append(dict(seg=seg, path=d, pos=p, touch=float(mind[i]), dom=dom,
                        dist3d=dist3d, delay=dist3d / (V_COND * 1000.0) + SYN_DELAY))
    print(f"  유지 {len(syn)}개 {[s['dom'] for s in syn]} · "
          f"경로거리 {[round(s['path']) for s in syn]} um · "
          f"접촉거리 {[round(s['touch'],1) for s in syn]} um · "
          f"지연 {[round(s['delay'],2) for s in syn]} ms")

    # pre 형태 dict (그리기용, θ* 배치)
    m_pre = dict(m_pre0); m_pre["xyz"] = pre_pts

    import matplotlib.pyplot as plt

    # ---- 그림 1: 회전각 vs 접촉 (아이디어 자체) ----
    fig1, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax1.plot(angles, counts, "-o", color="#7b1fa2", ms=4, label=f"접촉 개수 (<{TOUCH_R:.0f}um)")
    ax1.axvline(theta, color="#e53935", ls="--", lw=1.5)
    ax1.plot([theta], [counts[best]], "*", color="#e53935", ms=18, zorder=5)
    ax1.annotate(f"θ* = {theta:.0f}도\n접촉 {counts[best]}개",
                 xy=(theta, counts[best]), xytext=(theta + 20, counts[best] + 0.5),
                 fontsize=10, color="#b71c1c", fontweight="bold")
    ax1.set_xlabel("방사축 둘레 회전각 θ (도)"); ax1.set_ylabel(f"접촉 세그먼트 수 (<{TOUCH_R:.0f}um)")
    ax1.set_title("3-2  방사축 둘레 회전 → 인접(접촉) 최대 각 탐색", fontsize=12, loc="left")
    ax1b = ax1.twinx()
    ax1b.plot(angles, meandist, "-", color="#90a4ae", lw=1.2, alpha=0.8)
    ax1b.set_ylabel("평균 최소거리 (um)", color="#607d8b")
    ax1b.tick_params(axis="y", labelcolor="#607d8b")
    ax1.legend(loc="upper right", fontsize=9)
    plots.stamp(fig1, f"3-2 | 방사축=정단축(+y) · L={SOMA_LATERAL_L:.0f}um · CA1 PC 방위각은 자유 파라미터")
    outdir = plots.figdir(__file__)
    plots.save(fig1, outdir, "3-2_rotation.png")

    # ---- 그림 2: θ* 배치 + 접촉 시냅스 ----
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 6.8),
                                    gridspec_kw={"width_ratios": [1.3, 1]})
    PRE_C, POST_C = "#2e7d32", "#d84315"     # pre=초록, post=주황 (세포 구분)
    mo.render(axA, m_pre, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False,
              color=PRE_C, soma_color="#1b5e20")
    mo.render(axA, m_post, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False,
              color=POST_C, soma_color="#bf360c")
    allx = np.concatenate([m_pre["xyz"][:, 0], m_post["xyz"][:, 0]])
    ally = np.concatenate([m_pre["xyz"][:, 1], m_post["xyz"][:, 1]])
    axA.set_xlim(allx.min() - 40, allx.max() + 40)
    axA.set_ylim(np.percentile(ally, 0.3) - 30, np.percentile(ally, 99.8) + 60)
    axA.set_aspect("equal", adjustable="box"); axA.set_xticks([]); axA.set_yticks([]); axA.grid(False)
    for s in axA.spines.values():
        s.set_color("#dddddd")
    axA.axhspan(APIC_PROX[0], APIC_PROX[1], color="#ffb300", alpha=0.12, zorder=0)
    axA.text(allx.max() + 35, sum(APIC_PROX) / 2,
             f"정단 근위\n{APIC_PROX[0]:.0f}~{APIC_PROX[1]:.0f}um", fontsize=8, color="#b26a00",
             va="center", ha="right")
    ybas = m_post["xyz"][m_post["type"] == mo.BASAL, 1]
    if len(ybas):
        axA.axhspan(float(ybas.min()), 0.0, color="#4fc3f7", alpha=0.10, zorder=0)
        axA.text(allx.max() + 35, float(ybas.min()) / 2, "기저수상돌기", fontsize=8,
                 color="#0277bd", va="center", ha="right")
    if prune_i:
        pp = np.array([sr_segs[i][2][:2] for i in prune_i])
        axA.scatter(pp[:, 0], pp[:, 1], s=55, marker="x", color="#9e9e9e", lw=1.7, zorder=5)
    kp = np.array([s["pos"][:2] for s in syn])
    axA.scatter(kp[:, 0], kp[:, 1], s=250, marker="*", color="#7b1fa2",
                edgecolor="white", lw=1.3, zorder=6)
    axA.text(pre_soma3d[0], np.percentile(m_pre["xyz"][:, 1], 99.8) + 40, "pre (자극)",
             fontsize=9.5, ha="center", color=PRE_C, fontweight="bold")
    post_soma3d = m_post["xyz"][m_post["type"] == mo.SOMA].mean(axis=0)
    axA.text(post_soma3d[0], np.percentile(m_post["xyz"][:, 1], 99.8) + 40, "post (기록)",
             fontsize=9.5, ha="center", color=POST_C, fontweight="bold")
    axA.set_title(f"A. θ*={theta:.0f}도 배치 + 접촉 시냅스 (touch)", fontsize=10.5, loc="left")

    # 범례: 세포 구분(pre/post) + 도메인(정단/기저/축삭) + 시냅스
    import matplotlib.lines as mlines
    def hline(c, lw=2.6):
        return mlines.Line2D([], [], color=c, lw=lw)
    handles = [
        hline(PRE_C),                                   # pre 정단
        hline(mo._lighten(PRE_C, 0.45)),                # pre 기저
        hline(POST_C),                                  # post 정단
        hline(mo._lighten(POST_C, 0.45)),               # post 기저
        hline(mo.TYPE_COLOR[mo.AXON]),                  # 축삭(공통)
        mlines.Line2D([], [], color="#7b1fa2", marker="*", lw=0, markersize=13),
        mlines.Line2D([], [], color="#9e9e9e", marker="x", lw=0, markersize=8),
    ]
    labels = ["pre 정단수상돌기", "pre 기저수상돌기", "post 정단수상돌기", "post 기저수상돌기",
              "축삭(스텁)", f"유지 시냅스 {len(syn)}", f"가지치기 제거 {len(prune_i)}"]
    # 범례를 축 아래 바깥으로 빼서 뉴런과 겹치지 않게. 가로로 눕혀 공간 절약.
    axA.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               fontsize=8, framealpha=0.95, ncol=3, handlelength=1.4,
               columnspacing=1.2, borderaxespad=0.3)
    mo.scalebar(axA, 200, "200 um", loc=(0.72, 0.05))

    ypos = np.arange(len(syn))
    delays = [s["delay"] for s in syn]
    axB.barh(ypos, delays, color="#7b1fa2", alpha=0.85)
    axB.set_yticks(ypos)
    axB.set_yticklabels([f"{'기저' if s['dom']=='basal' else '정단'} {round(s['path'])}um"
                         for s in syn], fontsize=9)
    axB.invert_yaxis()
    axB.set_xlabel("전도지연 (ms) = 거리/속도 + 시냅스지연")
    axB.set_ylabel("시냅스 (post 소마 경로거리)")
    axB.set_title("B. 접촉 기하에서 계산한 전도지연", fontsize=10.5, loc="left")
    for i, s in enumerate(syn):
        axB.text(delays[i] + 0.02, i, f"{delays[i]:.2f} ms (거리 {s['dist3d']:.0f}um)",
                 va="center", fontsize=8, color="#4a148c")
    axB.set_xlim(0, max(delays) * 1.5)

    fig2.suptitle(f"3-2  방사축 정렬·회전 배치 (θ*={theta:.0f}도) → 접촉 시냅스 {len(syn)}개",
                  fontsize=12.5, y=0.98)
    fig2.subplots_adjust(top=0.88, bottom=0.20, wspace=0.18)
    plots.stamp(fig2, f"3-2 | PC->PC 표적: 기저 + 정단근위 {APIC_PROX[0]:.0f}~{APIC_PROX[1]:.0f}um · "
                      f"유지 {N_KEEP}개 (Deuchars1996) · θ*={theta:.0f}도 · L={SOMA_LATERAL_L:.0f}um")
    plots.save(fig2, outdir, "3-2_syn_sites.png")

    out = dict(pre=cfg["pre_tag"], post=cfg["post_tag"],
               target_zones=dict(basal=USE_BASAL, apical_proximal_um=list(APIC_PROX)),
               n_keep=N_KEEP,
               basis="Deuchars&Thomson1996 PMID8895869 (기저 3차, 접촉 2) · "
                     "Crepel1997 PMID9114256 (정단근위 50-150um) · Ecker2020 Fig3b (E->E 1.3)",
               soma_lateral_L_um=SOMA_LATERAL_L, touch_r_um=TOUCH_R, angle_step_deg=ANGLE_STEP,
               best_angle_deg=theta, best_touch_count=int(counts[best]),
               best_mean_mindist_um=round(float(meandist[best]), 1),
               v_cond_um_per_us=V_COND, syn_delay_ms=SYN_DELAY,
               kept=len(syn),
               synapses=[dict(path_um=round(s["path"], 1), touch_um=round(s["touch"], 1),
                              domain=s["dom"],
                              dist3d_um=round(s["dist3d"], 1), delay_ms=round(s["delay"], 2),
                              section=s["seg"].sec.name().split(".")[-1]) for s in syn])
    jpath = os.path.join(outdir, "3-2_placement.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    print("\n[통과] 3-2 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
