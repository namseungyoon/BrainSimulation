# -*- coding: utf-8 -*-
"""2-6 벤치 회전 GIF — 두 세포(pre·post)를 방사축 둘레로 돌리며 3D 구조를 보인다

단계   : 2-6 (파이프라인 2단계 뉴런 / 하위 6 pair) 보조 산출물
목적   : 고정 기하(θ*=160°, L=120µm)의 두 세포 배치와 시냅스 5개를, 방사축(수직) 둘레로
         회전시켜 3D 로 직관적으로 보인다. pre=초록·post=주황·시냅스=별.
결과   : figures/2-6_bench_rotation.gif
실행   : . .\\env\\activate.ps1 ; & $Py04 02_neurons\\6_pair\\2-6_bench_rotation.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                          # noqa: E402
import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from lib import plots                        # noqa: E402
from lib import morphology as mo             # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.nrnenv import h                     # noqa: E402

PRE_C, POST_C = "#2e7d32", "#d84315"
N_FRAME = 36                                 # 10도 간격 한 바퀴


def cell_segments(cell, deg_y=0.0, shift_x=0.0):
    """세포의 3D 선분(부모-자식)과 타입. 방사축(y) deg_y 회전 + x 이동 적용."""
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
    p = mo.apply_transform(m["xyz"], c, R)
    if deg_y:
        t = np.deg2rad(deg_y); ct, st = np.cos(t), np.sin(t)
        Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        p = p @ Ry.T
    p = p.copy(); p[:, 0] += shift_x
    segs, _, typ2 = mo.segments(dict(m, xyz=p))
    return segs, typ2, c, R, p


def syn_points_3d(bench, c, R, deg_y, shift_x):
    pts = []
    for seg, spec in bench.post_syn_segs():
        sec = seg.sec; i = sec.n3d() // 2
        q = mo.apply_transform(np.array([sec.x3d(i), sec.y3d(i), sec.z3d(i)]), c, R)
        pts.append(q)
    return np.array(pts)


def rotate_project(segs3d, phi_deg):
    """장면 전체를 방사축(y) 둘레 phi 회전 후 (x,y) 투영. 깊이 z' 도 반환(정렬용)."""
    t = np.deg2rad(phi_deg); ct, st = np.cos(t), np.sin(t)
    x = segs3d[..., 0]; y = segs3d[..., 1]; z = segs3d[..., 2]
    xp = x * ct - z * st
    zp = x * st + z * ct
    return np.stack([xp, y], axis=-1), zp


def main():
    plots.setup()
    print("=== 2-6 벤치 회전 GIF ===")
    b = Bench()
    geo = b.geo
    L = geo["placement"]["soma_lateral_L_um"]
    theta = geo["placement"]["pre_rotation_deg"]

    # pre: θ 회전 + 왼쪽 -L / post: 원점
    pre_seg, pre_typ, cPre, RPre, pre_p = cell_segments(b.pre, deg_y=theta, shift_x=-L)
    post_seg, post_typ, cP, RP, post_p = cell_segments(b.post, deg_y=0.0, shift_x=0.0)
    # 시냅스 3D (post 좌표계, 이동 없음)
    syn3d = syn_points_3d(b, cP, RP, 0.0, 0.0)

    # 색: 도메인별 명암(정단 진하게/기저 옅게), 세포별 색
    def seg_colors(typ, base):
        out = []
        for t in typ:
            if t == mo.APICAL:
                out.append(base)
            elif t == mo.BASAL:
                out.append(mo._lighten(base, 0.45))
            elif t == mo.SOMA:
                out.append("#111111")
            else:
                out.append(mo.TYPE_COLOR[mo.AXON])
        return out
    pre_col = seg_colors(pre_typ, PRE_C)
    post_col = seg_colors(post_typ, POST_C)

    # 축 범위(회전 중 안 잘리게 x,z 최대 반경 기준)
    allp = np.vstack([pre_p, post_p])
    rmax = np.percentile(np.sqrt(allp[:, 0]**2 + allp[:, 2]**2), 99.9) * 1.1
    ylo = np.percentile(allp[:, 1], 0.3) - 40
    yhi = np.percentile(allp[:, 1], 99.8) + 60

    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    fig.patch.set_facecolor("white")

    def draw(frame):
        ax.clear()
        phi = frame * (360.0 / N_FRAME)
        for seg3d, col in [(pre_seg, pre_col), (post_seg, post_col)]:
            p2, _ = rotate_project(seg3d, phi)
            lc = LineCollection(p2, colors=col, linewidths=0.9)
            ax.add_collection(lc)
        # 시냅스
        s2, _ = rotate_project(syn3d[:, None, :].repeat(2, axis=1), phi)
        s2 = s2[:, 0, :]
        ax.scatter(s2[:, 0], s2[:, 1], s=90, marker="*", color="#7b1fa2",
                   edgecolor="white", lw=0.8, zorder=6)
        ax.set_xlim(-rmax - L, rmax); ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#dddddd")
        ax.set_title(f"두 세포 벤치 (방사축 회전 {phi:.0f}°)\n"
                     f"pre(초록)→post(주황) SR 시냅스 5개(★)", fontsize=10, loc="left")
        return []

    print(f"  프레임 {N_FRAME}개 렌더링 ...")
    anim = FuncAnimation(fig, draw, frames=N_FRAME, blit=False)
    outdir = plots.figdir(__file__)
    gif = os.path.join(outdir, "2-6_bench_rotation.gif")
    anim.save(gif, writer=PillowWriter(fps=12))
    plt.close(fig)
    sz = os.path.getsize(gif) / 1024
    print(f"saved: {gif} ({sz:.0f} KB)")
    print("\n[통과] 2-6 회전 GIF 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
