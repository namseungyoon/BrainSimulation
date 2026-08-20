# -*- coding: utf-8 -*-
"""2-2 단일 세포 로드 — 선별한 번들을 실제로 NEURON 에 올린다

단계   : 2-2 (파이프라인 2단계 뉴런 / 하위 2 load)
방법   : config/cells.yaml 의 pre·post 번들을 각각 고유 템플릿 이름으로 로드하고,
         구획 수·세그먼트 수·총 길이·도메인 구성을 실측한다. NEURON 이 실제로 만든 구획을
         3D 로 그려, 2-1 의 SWC 그림과 같은 세포인지 눈으로 대조한다.
근거   : docs/DECISIONS.md D2 · lib/cells.py 의 템플릿 이름 치환(2-1 에서 경고한 위험)
재료   : config/cells.yaml · ../Models/<번들>/  · lib/cells.py · lib/morphology.py
결과   : figures/2-2_cell_loaded.png · figures/2-2_load.json

실행:
  . .\\env\\activate.ps1
  & $Py04 02_neurons\\2_load\\2-2_load_cell.py
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


def cell_points(cell):
    """NEURON 이 인스턴스화한 세포의 3D 좌표·타입·반지름을 뽑아 선분으로.

    h.define_shape() 로 3D 좌표를 확정한 뒤 각 구획의 pt3d 를 읽는다.
    반환: morphology.render 가 먹는 형태의 dict (xyz·type·parent_row).
    """
    h.define_shape()
    dom_type = {"soma": mo.SOMA, "axon": mo.AXON, "myelin": mo.AXON,
                "dend": mo.BASAL, "apic": mo.APICAL}
    xyz, typ, seg_of = [], [], []
    seclist = list(cell.all) if hasattr(cell, "all") else list(h.allsec())
    for sec in seclist:
        nm = sec.name().split(".")[-1]
        base = nm.split("[")[0]
        t = dom_type.get(base, mo.BASAL)
        n = sec.n3d()
        first = len(xyz)
        for i in range(n):
            xyz.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            typ.append(t)
            # 같은 구획 안에서는 앞 점이 부모, 구획 첫 점은 부모구획 끝(근사로 -1 처리)
            seg_of.append(first + i - 1 if i > 0 else -1)
    xyz = np.array(xyz, dtype=float)
    typ = np.array(typ, dtype=np.int64)
    parent_row = np.array(seg_of, dtype=np.int64)
    return dict(xyz=xyz, type=typ, parent_row=parent_row,
                radius=np.ones(len(xyz)), index=np.arange(len(xyz)),
                parent=parent_row.copy())


def render_cell(ax, cell, title):
    m = mo.align(cell_points(cell), mode="apical")
    mo.render(ax, m, types=(mo.SOMA, mo.BASAL, mo.APICAL, mo.AXON), autoscale=False)
    xy = m["xyz"][np.isin(m["type"], (mo.SOMA, mo.BASAL, mo.APICAL))][:, :2]
    hx = np.percentile(np.abs(xy[:, 0]), 99.8) * 1.1
    ax.set_xlim(-hx, hx)
    ax.set_ylim(np.percentile(xy[:, 1], 0.2) * 1.15, np.percentile(xy[:, 1], 99.8) * 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_color("#dddddd")
    ax.set_title(title, fontsize=10, pad=6)
    return m


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "cells.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["pair"]
    models_root = os.path.join(REPO, "Models")

    print("=== 2-2 단일 세포 로드 ===")
    recs = []
    for role in ("pre", "post"):
        bdir = os.path.join(models_root, cfg[f"{role}_bundle"])
        cell, tname = cells.load_cell(bdir, role)
        s = cells.summary(cell)
        recs.append(dict(role=role, tag=cfg[f"{role}_tag"], cell=cell,
                         tname=tname, summary=s))
        print(f"  [{role}] {cfg[f'{role}_tag']}  템플릿={tname}")
        print(f"        구획 {s['n_sections']}  세그먼트 {s['n_segments']}  "
              f"총길이 {s['total_length_um']:.0f} um  도메인 {s['domains']}")

    # ★ 두 세포가 정말 독립인가 = 서로 다른 고유 템플릿 이름을 가졌는가
    independent = recs[0]["tname"] != recs[1]["tname"]
    print(f"\n  독립 템플릿 이름 : {recs[0]['tname']} vs {recs[1]['tname']}  "
          f"-> {'독립 OK' if independent else '★충돌'}")

    # --- 그림: 실제 인스턴스화된 두 세포 -------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.2))
    ms = []
    for ax, rec in zip(axes, recs):
        role_ko = "pre (자극)" if rec["role"] == "pre" else "post (기록)"
        m = render_cell(ax, rec["cell"], f"{role_ko}\n{rec['tag']}")
        ms.append(m)
        s = rec["summary"]
        ax.text(0.03, 0.975,
                f"구획 {s['n_sections']}\n세그먼트 {s['n_segments']}\n"
                f"총길이 {s['total_length_um']/1000:.1f} mm",
                transform=ax.transAxes, va="top", fontsize=8.5, color="#455a64")
    mo.scalebar(axes[0], 200, "200 um")
    handles = [plt.Line2D([], [], color=mo.TYPE_COLOR[t], lw=2.4, label=mo.TYPE_KO[t])
               for t in (mo.SOMA, mo.APICAL, mo.BASAL, mo.AXON)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("2-2  NEURON 에 실제 인스턴스화한 두 세포 (고유 템플릿 이름으로 독립 로드)",
                 fontsize=12.5, y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.05)
    plots.stamp(fig, f"2-2 | pre={recs[0]['tname']} · post={recs[1]['tname']} | "
                     f"replace_axon 적용된 실제 구획")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "2-2_cell_loaded.png")

    # --- 검증: 축삭이 replace_axon 으로 짧아졌는가 ---------------------------
    # SWC 원본 축삭은 수천 um 인데, replace_axon 후엔 스텁 2개(~60um)만 남아야 한다.
    axon_reduced = []
    for rec in recs:
        secs = list(rec["cell"].all)
        ax_secs = [s for s in secs if ".axon" in s.name()]
        ax_len = sum(s.L for s in ax_secs)
        axon_reduced.append((len(ax_secs), round(ax_len, 1)))
        print(f"  [{rec['role']}] 축삭 구획 {len(ax_secs)}개 · 총 {ax_len:.1f} um "
              f"(replace_axon: 스텁만 남아야 정상)")

    checks = [
        ("pre 로드", recs[0]["summary"]["n_sections"] > 50),
        ("post 로드", recs[1]["summary"]["n_sections"] > 50),
        ("독립 템플릿", independent),
        ("pre 축삭 스텁화", axon_reduced[0][1] < 200),
        ("post 축삭 스텁화", axon_reduced[1][1] < 200),
    ]
    n_ok = sum(1 for _, ok in checks if ok)

    out = dict(
        pre=dict(tag=cfg["pre_tag"], template=recs[0]["tname"],
                 **recs[0]["summary"], axon_sections=axon_reduced[0][0],
                 axon_len_um=axon_reduced[0][1]),
        post=dict(tag=cfg["post_tag"], template=recs[1]["tname"],
                  **recs[1]["summary"], axon_sections=axon_reduced[1][0],
                  axon_len_um=axon_reduced[1][1]),
        independent_templates=independent,
        checks={k: bool(v) for k, v in checks},
        checks_passed=n_ok, checks_total=len(checks),
    )
    jpath = os.path.join(outdir, "2-2_load.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")

    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 항목 미통과")
        return 1
    print(f"\n[통과] 2-2 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
