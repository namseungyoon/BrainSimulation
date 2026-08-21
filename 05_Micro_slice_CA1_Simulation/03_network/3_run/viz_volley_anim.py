# -*- coding: utf-8 -*-
"""
03_network/3_run/viz_volley_anim.py  —  단일 volley 발화 애니메이션 (느린 재생)

발화 데이터(spk_t·spk_id·cell_uvw)를 읽어 자극 볼리 순간과 뉴런 발화를 시간축을
엄청 느리게(~2000×) 애니메이션한다. 기본 입력=스모크(미리보기), --base=baseline 지정 시 전체망.
결과: figures/volley_anim.gif
실행: python 03_network/3_run/viz_volley_anim.py [--src scratch/mpi_smoke_viz.npz]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "figures")
SRC = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else os.path.join(ROOT, "scratch", "mpi_smoke_viz.npz")

d = np.load(SRC, allow_pickle=True)
uvw = d["cell_uvw"]; gid = d["cell_gid"]; spk_t = d["spk_t"]; spk_id = d["spk_id"].astype(int)
e3 = d["e3"]; R = float(d["radius"]); stim = float(d["stim_t"])
U, Rr = uvw[:, 0], uvw[:, 1]
g2i = {int(g): i for i, g in enumerate(gid)}
# 각 스파이크의 세포 인덱스
spk_ci = np.array([g2i.get(int(s), -1) for s in spk_id])
valid = spk_ci >= 0
spk_t = spk_t[valid]; spk_ci = spk_ci[valid]

t0, t1, step = stim - 2.0, spk_t.max() + 3.0, 0.25
frames = np.arange(t0, t1, step)
GLOW = 1.2  # ms 잔광

fig, ax = plt.subplots(figsize=(9, 7.5))

def draw(f):
    ax.clear()
    t = frames[f]
    ax.axhspan(25, 460, color="#55A868", alpha=0.06); ax.axhspan(-410, -65, color="#4C72B0", alpha=0.06)
    # 기본 뉴런
    ax.scatter(U, Rr, s=30, c="#d5d5d5", edgecolors="#bbb", linewidths=0.4, zorder=2)
    # 최근 발화 세포 (잔광)
    recent = (spk_t <= t) & (spk_t > t - GLOW)
    if recent.any():
        ci = spk_ci[recent]; age = (t - spk_t[recent]) / GLOW
        ax.scatter(U[ci], Rr[ci], s=140 * (1 - age) + 40, c="#C44E52",
                   alpha=0.9 * (1 - age) + 0.1, edgecolors="black", linewidths=0.5, zorder=5)
    # E3 자극 지점 + 볼리 펄스
    ax.scatter([e3[0]], [e3[1]], s=200, marker="*", c="red", edgecolors="black", zorder=6)
    if stim <= t < stim + 1.0:   # 볼리 순간 펄스
        ax.add_patch(Circle((e3[0], e3[1]), R * (t - stim + 0.2) / 1.2, fill=False, ec="#e34", lw=3, alpha=0.8, zorder=4))
        ax.text(e3[0], e3[1] + 120, "⚡ 자극 볼리!", color="#e34", fontsize=14, ha="center", zorder=7)
    ax.add_patch(Circle((e3[0], e3[1]), R, fill=False, ec="#2222aa", lw=1.2, ls="--", zorder=3))
    nfired = int(((spk_t <= t)).sum() and len(np.unique(spk_ci[spk_t <= t])))
    ax.text(0.02, 0.98, f"t = {t-stim:+.2f} ms (자극기준)", transform=ax.transAxes, fontsize=13,
            va="top", fontweight="bold")
    ax.text(0.02, 0.93, f"누적 발화세포 {len(np.unique(spk_ci[spk_t<=t]))} · 스파이크 {int((spk_t<=t).sum())}",
            transform=ax.transAxes, fontsize=11, va="top", color="#C44E52")
    ax.set_xlim(U.min() - 40, U.max() + 40); ax.set_ylim(Rr.min() - 40, Rr.max() + 40)
    ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm)")
    ax.set_title(f"단일 volley 발화 (느린 재생 ~2000×) — 150세포 스모크 미리보기")
    return []

anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
out = os.path.join(FIG, "volley_anim.gif")
anim.save(out, writer=PillowWriter(fps=10))
plt.close(fig)
print(f"[프레임] {len(frames)}개 · {t0:.1f}~{t1:.1f}ms · step {step}ms")
print(f"[GIF] -> {out}")
