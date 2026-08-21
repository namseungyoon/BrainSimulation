# -*- coding: utf-8 -*-
"""2-7 거리 지도 — post 수상돌기를 소마 경로거리로 색칠 + SR 대역·고정 시냅스 표시

단계   : 2-7 (파이프라인 2단계 뉴런 / 하위 7 distance)
쉬운 설명: 시냅스가 소마에서 얼마나 '멀리'(전선 길이 기준) 있는지가 신호 감쇠와 가소성에
          영향을 준다. post 세포의 모든 가지를 소마로부터의 경로거리로 색칠해, 3-2 에서
          고정한 시냅스 5개가 어느 거리대에 있는지 한눈에 보인다.
방법   : h.distance() 로 각 세그먼트의 소마 경로거리를 재고, 형태 위에 컬러맵으로 그린다.
          SR 대역(정단 100~300µm)과 고정 시냅스(★) 를 겹쳐 표시.
결과   : figures/2-7_distance_map.png · figures/2-7_distance.json
실행   : . .\\env\\activate.ps1 ; & $Py04 02_neurons\\7_distance\\2-7_distance_map.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                          # noqa: E402
from lib import plots                        # noqa: E402
from lib import morphology as mo             # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.nrnenv import h                     # noqa: E402


def build_colored(post):
    """post 각 구획을 소마 경로거리로 라벨링. (선분배열, 거리배열, 정렬변환) 반환."""
    h.define_shape()
    h.distance(0, post.soma[0](0.5))
    # 점구름 + 각 점의 경로거리
    xyz, typ, par, dist = [], [], [], []
    for s in post.all:
        base = s.name().split(".")[-1].split("[")[0]
        tt = {"soma": mo.SOMA, "apic": mo.APICAL, "dend": mo.BASAL,
              "axon": mo.AXON, "myelin": mo.AXON}.get(base, mo.BASAL)
        d0 = h.distance(s(0.0)); d1 = h.distance(s(1.0))
        first = len(xyz)
        n = s.n3d()
        for i in range(n):
            xyz.append((s.x3d(i), s.y3d(i), s.z3d(i))); typ.append(tt)
            par.append(first + i - 1 if i > 0 else -1)
            # 구획 내 선형 보간 거리
            frac = i / max(n - 1, 1)
            dist.append(d0 + (d1 - d0) * frac)
    m = dict(xyz=np.array(xyz, float), type=np.array(typ), parent_row=np.array(par),
             radius=np.ones(len(xyz)), index=np.arange(len(xyz)), parent=np.array(par),
             dist=np.array(dist))
    c, R = mo.align_transform(m, mode="apical")
    m["xyz"] = mo.apply_transform(m["xyz"], c, R)
    return m, c, R


def main():
    plots.setup()
    print("=== 2-7 거리 지도 ===")
    b = Bench()
    geo = b.geo
    sr = geo["sr_band_um"]
    m, c, R = build_colored(b.post)

    # 축삭 제외하고 수상돌기만 색칠
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    pr = m["parent_row"]; ok = pr >= 0
    dend = np.isin(m["type"], (mo.SOMA, mo.BASAL, mo.APICAL))
    sel = ok & dend[pr] & dend
    child = np.nonzero(sel)[0]
    par = pr[child]
    segs = np.stack([m["xyz"][par][:, :2], m["xyz"][child][:, :2]], axis=1)
    seg_dist = 0.5 * (m["dist"][par] + m["dist"][child])

    dmax = float(m["dist"][dend].max())
    norm = mcolors.Normalize(vmin=0, vmax=dmax)
    import matplotlib as mpl
    cmap = mpl.colormaps["viridis"]        # matplotlib 3.11: cm.get_cmap 제거됨

    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    lc = LineCollection(segs, colors=cmap(norm(seg_dist)), linewidths=1.3,
                        capstyle="round")
    ax.add_collection(lc)

    xyd = m["xyz"][dend][:, :2]
    hx = np.percentile(np.abs(xyd[:, 0]), 99.8) * 1.15
    ax.set_xlim(-hx, hx)
    ax.set_ylim(np.percentile(xyd[:, 1], 0.2) - 30, np.percentile(xyd[:, 1], 99.8) + 40)
    ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_color("#dddddd")

    # SR 대역 표시
    ax.axhspan(sr[0], sr[1], color="#ff6f00", alpha=0.08, zorder=0)
    ax.text(hx, (sr[0]+sr[1])/2, " SR 대역\n 100~300um", fontsize=8.5, color="#b26a00",
            va="center", ha="right")

    # 고정 시냅스 5개 위치(★) + 경로거리 라벨
    syn_d = []
    for seg, spec in b.post_syn_segs():
        sec = seg.sec; i = sec.n3d() // 2
        p = mo.apply_transform(np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]), c, R)
        ax.scatter([p[0]], [p[1]], s=200, marker="*", color="#d81b60",
                   edgecolor="white", lw=1.2, zorder=6)
        syn_d.append(spec["path_um"])

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("소마 경로거리 (um)")

    ax.set_title(f"2-7  post 수상돌기 경로거리 지도 ({geo['pair']['post_tag']})\n"
                 f"색=소마로부터 전선길이 · ★ 고정 시냅스 5개 · 정단 최대 {dmax:.0f}um",
                 fontsize=11, loc="left")
    ax.scatter([], [], s=180, marker="*", color="#d81b60", label="고정 시냅스")
    ax.legend(loc="lower left", fontsize=8.5)
    mo.scalebar(ax, 200, "200 um", loc=(0.68, 0.02))

    plots.stamp(fig, f"2-7 | h.distance 경로거리 · SR {sr[0]:.0f}~{sr[1]:.0f}um · 시냅스 {[round(d) for d in syn_d]}um")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "2-7_distance_map.png")

    out = dict(post=geo["pair"]["post_tag"], sr_band=sr,
               apical_max_dist_um=round(dmax, 1),
               syn_path_um=[round(d, 1) for d in syn_d],
               syn_in_sr=[bool(sr[0] <= d <= sr[1]) for d in syn_d])
    jpath = os.path.join(outdir, "2-7_distance.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  정단 최대 경로거리 {dmax:.0f}um · 시냅스 거리 {[round(d) for d in syn_d]}um")
    print(f"  SR 대역 안 시냅스: {sum(out['syn_in_sr'])}/5")
    print(f"saved: {jpath}")
    print("\n[통과] 2-7 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
