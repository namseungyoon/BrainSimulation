# -*- coding: utf-8 -*-
"""2-6 세포 쌍 배치 — pre·post 를 한 공간에 놓고 연결 도해를 그린다

단계   : 2-6 (파이프라인 2단계 뉴런 / 하위 6 pair)
방법   : 두 세포를 한 그림에 배치한다. post 는 기록 세포(정단 상방), pre 는 자극원으로 왼쪽에
         offset 배치. post 정단수상돌기 SR 대역에 시냅스가 놓일 지점을 표시하고, pre->시냅스
         연결을 화살표로 그린다.
★주의  : 이 배치는 **도해**다. D8 에 따라 pre->post 전달은 물리적 막 접촉이 아니라
         NetCon + 전도지연이므로, 두 세포의 상대 위치는 시뮬레이션 결과에 영향이 없다.
         pre 를 왼쪽에 둔 것은 도해상의 배치일 뿐이다(두 세포 다 CA1 추체세포).
근거   : docs/DECISIONS.md D8 · config/cells.yaml · 2-3 의 SR 대역
재료   : lib/cells.py · lib/morphology.py · config/cells.yaml
결과   : figures/2-6_two_cells.png · figures/2-6_pair.json

실행:
  . .\\env\\activate.ps1
  & $Py04 02_neurons\\6_pair\\2-6_two_cells.py
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

SR_MIN, SR_MAX = 100.0, 300.0                # 2-3 과 동일한 SR 대역


def cell_points(cell):
    h.define_shape()
    dom_type = {"soma": mo.SOMA, "axon": mo.AXON, "myelin": mo.AXON,
                "dend": mo.BASAL, "apic": mo.APICAL}
    xyz, typ, parent = [], [], []
    for sec in cell.all:
        base = sec.name().split(".")[-1].split("[")[0]
        t = dom_type.get(base, mo.BASAL)
        first = len(xyz)
        for i in range(sec.n3d()):
            xyz.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            typ.append(t)
            parent.append(first + i - 1 if i > 0 else -1)
    xyz = np.array(xyz, float)
    return dict(xyz=xyz, type=np.array(typ, np.int64),
                parent_row=np.array(parent, np.int64),
                radius=np.ones(len(xyz)), index=np.arange(len(xyz)),
                parent=np.array(parent, np.int64))


def pick_sr_point(cell):
    """post 정단수상돌기 SR 대역에서 시냅스 예시 지점(경로거리 중앙 근처)을 하나 고른다.
    정렬된 좌표계에서의 (x, y) 를 반환. 실제 배치는 3-2 에서 시드 기반으로 여러 개.
    """
    soma = cell.soma[0]
    h.distance(0, soma(0.5))
    best = None
    target = (SR_MIN + SR_MAX) / 2
    for sec in cell.all:
        if ".apic" not in sec.name():
            continue
        d = h.distance(sec(0.5))
        if SR_MIN <= d <= SR_MAX:
            if best is None or abs(d - target) < best[0]:
                best = (abs(d - target), sec, d)
    if best is None:
        return None
    _, sec, d = best
    # 3D 좌표(중점)
    i = sec.n3d() // 2
    return np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]), d


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "cells.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["pair"]
    models_root = os.path.join(REPO, "Models")

    print("=== 2-6 세포 쌍 배치 (도해) ===")
    loaded = {}
    for role in ("pre", "post"):
        bdir = os.path.join(models_root, cfg[f"{role}_bundle"])
        cell, tname = cells.load_cell(bdir, role)
        loaded[role] = cell

    # 정렬(소마 원점·정단 상방)
    m_pre = mo.align(cell_points(loaded["pre"]), mode="apical")
    m_post = mo.align(cell_points(loaded["post"]), mode="apical")

    # post 는 원점, pre 는 왼쪽으로 offset (도해상 간격)
    dend_half = np.percentile(
        np.abs(m_post["xyz"][np.isin(m_post["type"], (mo.SOMA, mo.BASAL, mo.APICAL))][:, 0]),
        99.5)
    offset = dend_half * 2.4                    # 두 세포가 안 겹칠 만큼
    m_pre["xyz"][:, 0] -= offset

    # 시냅스 예시 지점 (정렬된 post 좌표계에서 뽑고, 배치 이동 없음: post 는 원점)
    sr = pick_sr_point(loaded["post"])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    mo.render(ax, m_pre, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False)
    mo.render(ax, m_post, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False)

    pre_soma = m_pre["xyz"][m_pre["type"] == mo.SOMA].mean(axis=0)
    post_soma = m_post["xyz"][m_post["type"] == mo.SOMA].mean(axis=0)
    pre_top = np.percentile(m_pre["xyz"][:, 1], 99.8)
    post_top = np.percentile(m_post["xyz"][:, 1], 99.8)

    # 축 범위 — 위쪽에 라벨·화살표 여백 확보
    allx = np.concatenate([m_pre["xyz"][:, 0], m_post["xyz"][:, 0]])
    ally = np.concatenate([m_pre["xyz"][:, 1], m_post["xyz"][:, 1]])
    ax.set_xlim(allx.min() - 70, allx.max() + 70)
    ax.set_ylim(np.percentile(ally, 0.3) - 40, max(pre_top, post_top) + 150)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_color("#dddddd")

    # SR 대역(post) 표시 — post 소마 기준 정단 100~300um. post 는 원점이라 y=100~300.
    post_xmin = m_post["xyz"][:, 0].min() - 20
    post_xmax = m_post["xyz"][:, 0].max() + 20
    ax.fill_between([post_xmin, post_xmax], SR_MIN, SR_MAX,
                    color="#ffb300", alpha=0.14, zorder=0)
    ax.text(post_xmax + 8, (SR_MIN + SR_MAX) / 2, "SR 대역\n정단 100~300um",
            fontsize=9, color="#b26a00", va="center", ha="left")

    # 시냅스 예시 지점 + pre->시냅스 연결 (SR 대역 높이로 수평에 가깝게)
    if sr is not None:
        sp, sd = sr
        # 화살표 시작 = pre 오른쪽 가장자리, SR 높이
        pre_right = m_pre["xyz"][:, 0].max()
        start = (pre_right + 10, sp[1])
        ax.scatter([sp[0]], [sp[1]], s=320, marker="*", color="#7b1fa2",
                   edgecolor="white", lw=1.4, zorder=6)
        ax.annotate("", xy=(sp[0], sp[1]), xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#7b1fa2", lw=2.0,
                                    linestyle=(0, (5, 3)), shrinkA=2, shrinkB=10),
                    zorder=5)
        midx = (start[0] + sp[0]) / 2
        ax.text(midx, max(pre_top, post_top) + 70,
                "연결 = NetCon + 전도지연\n(물리 접촉 아님 · D8)",
                fontsize=9.5, color="#7b1fa2", ha="center", va="center",
                fontweight="bold",
                bbox=dict(fc="white", ec="#7b1fa2", alpha=0.95, boxstyle="round,pad=0.4"))
        ax.annotate("", xy=(midx, (SR_MIN + SR_MAX) / 2 + 30),
                    xytext=(midx, max(pre_top, post_top) + 55),
                    arrowprops=dict(arrowstyle="-", color="#7b1fa2", lw=0.8,
                                    linestyle=(0, (2, 2))), zorder=4)

    # 세포 라벨
    ax.text(pre_soma[0], pre_top + 55, f"pre (자극)\n{cfg['pre_tag']}",
            fontsize=10.5, ha="center", color="#212121", fontweight="bold")
    ax.text(post_soma[0], post_top + 55, f"post (기록)\n{cfg['post_tag']}",
            fontsize=10.5, ha="center", color="#212121", fontweight="bold")

    mo.scalebar(ax, 200, "200 um")
    handles = [plt.Line2D([], [], color=mo.TYPE_COLOR[t], lw=2.4, label=mo.TYPE_KO[t])
               for t in (mo.SOMA, mo.APICAL, mo.BASAL, mo.AXON)]
    handles.append(plt.Line2D([], [], color="#7b1fa2", marker="*", lw=0,
                              markersize=12, label="시냅스 예시 지점(3-2 에서 확정)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_title("2-6  두 세포 배치 도해 — pre(자극) → post(기록) SR 대역 시냅스",
                 fontsize=12.5, loc="left", pad=10)
    plots.stamp(fig, "2-6 | 배치는 도해(NetCon 설계라 상대 위치는 결과 무관) · pre 왼쪽=SC 입력 상징")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "2-6_two_cells.png")

    out = dict(
        pre_tag=cfg["pre_tag"], post_tag=cfg["post_tag"],
        arrangement="schematic",
        note=("D8: pre->post 는 NetCon+지연. 두 세포의 상대 위치는 시뮬레이션 결과에 영향 없음. "
              "pre 를 왼쪽에 둔 것은 도해상의 배치(두 세포 다 CA1 추체세포)."),
        offset_um=round(float(offset), 1),
        sr_band=[SR_MIN, SR_MAX],
        example_synapse_dist_um=round(float(sr[1]), 1) if sr else None,
    )
    jpath = os.path.join(outdir, "2-6_pair.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  시냅스 예시 지점 경로거리: {sr[1]:.0f} um" if sr else "  SR 대역 지점 없음")
    print(f"saved: {jpath}")
    print("\n[통과] 2-6 완료 (배치 도해)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
