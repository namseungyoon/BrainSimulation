# -*- coding: utf-8 -*-
"""
11_schaffer/sc_epsp_placement.py  —  E2-1 시냅스 실제 배치 3D 시각화 (회전 GIF, 느리게)

sc_epsp_test.py 와 동일한 대표 PC·동일한 3 표적(근위SR/원위SR/SLM)을 형태 위에 마커로 표시.
→ "EPSP를 어디에 놓고 쟀는지" 물리적 위치를 봄. 파형(E2_1_sc_epsp.png)과 짝.
3D 규칙에 따라 회전 GIF(gif_util). 회전은 느리게(프레임 多·duration 김).
실행: python 11_schaffer/sc_epsp_placement.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT); SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
sys.path.insert(0, SHARED); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))
sys.path.insert(0, os.path.join(PAPER, "04_network"))
from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
from gif_util import save_rotate_gif
MODELS = os.path.join(SHARED, "models")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

# sc_epsp_test.py 결과 (동일 표적)
EPSP = {"근위 SR": 0.151, "원위 SR": 0.060, "SLM 부근": 0.014}
MCOL = {"근위 SR": "#DD8452", "원위 SR": "#8172B3", "SLM 부근": "#55A868"}


def seg_xyz(sec, x):
    """섹션 sec 의 정규화 위치 x(0~1) 에서의 3D 좌표(형태 3d점 보간)."""
    n = sec.n3d()
    if n == 0:
        return None
    arcs = np.array([sec.arc3d(i) / sec.L for i in range(n)])
    xs = np.array([sec.x3d(i) for i in range(n)])
    ys = np.array([sec.y3d(i) for i in range(n)])
    zs = np.array([sec.z3d(i) for i in range(n)])
    return (np.interp(x, arcs, xs), np.interp(x, arcs, ys), np.interp(x, arcs, zs))


def main():
    type_dir = net.load_representatives(MODELS)
    cell, tname = load_cell(type_dir["PC"], gid=0)
    soma = cell.soma[0]
    print(f"[세포] {tname}", flush=True)

    fig = plt.figure(figsize=(8.5, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 형태 그리기: 섹션별 3d 폴리라인
    def draw(sec, color, lw):
        n = sec.n3d()
        if n < 2:
            return
        xs = [sec.x3d(i) for i in range(n)]
        ys = [sec.y3d(i) for i in range(n)]
        zs = [sec.z3d(i) for i in range(n)]
        ax.plot(xs, ys, zs, color=color, lw=lw, solid_capstyle="round")

    for sec in cell.all:
        nm = sec.name()
        if ".apic" in nm:
            draw(sec, "#4C72B0", 1.0)       # 정단(SR·SLM) 파랑
        elif ".dend" in nm:
            draw(sec, "#B0B0B0", 0.8)       # 기저(SO) 회색
        elif ".soma" in nm:
            draw(sec, "black", 4.0)
        elif ".axon" in nm:
            draw(sec, "#E0E0E0", 0.5)       # 축삭 stub 연회색

    # 소마 위치
    sx = seg_xyz(soma, 0.5)
    ax.scatter(*[[v] for v in sx], color="black", s=60, marker="o", label="소마", depthshade=False)

    # 3 표적(sc_epsp_test 와 동일 선택): apical 경로거리 25/55/85%
    h.distance(0, soma(0.5))
    apic = [s for s in cell.all if ".apic" in s.name()]
    seg_by_dist = sorted([(h.distance(s(0.5)), s) for s in apic], key=lambda t: t[0])
    dmax = seg_by_dist[-1][0]
    for frac, lab in [(0.25, "근위 SR"), (0.55, "원위 SR"), (0.85, "SLM 부근")]:
        d, s = min(seg_by_dist, key=lambda t: abs(t[0] - frac * dmax))
        p = seg_xyz(s, 0.5)
        ax.scatter(*[[v] for v in p], color=MCOL[lab], s=260, marker="*",
                   edgecolors="k", linewidths=0.6, depthshade=False,
                   label=f"{lab} ({d:.0f}µm) · EPSP {EPSP[lab]:.3f}mV")

    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    try:
        ax.set_box_aspect((1, 1.6, 1))
    except Exception:
        pass
    ax.legend(loc="upper left", fontsize=8.5)
    title = ("E2-a  단일 SC→PC 시냅스 실제 배치 (대표 추체세포 형태 위)\n"
             "정단(SR·SLM) 3위치에 시냅스 → 소마 EPSP 거리감쇠 (파형=E2a_sc_epsp.png)")
    out = os.path.join(FIG, "E2a_sc_epsp_placement.gif")
    # 느린 회전: 프레임 많이·프레임당 길게
    save_rotate_gif(fig, ax, out, n_frames=72, elev=12, duration=130, title=title)
    plt.close(fig)
    print(f"[OK] {out} (72프레임·느린회전)", flush=True)


if __name__ == "__main__":
    main()
