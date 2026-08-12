# -*- coding: utf-8 -*-
"""12_lfp/e4b_anim_gif.py  —  단일 전극 fEPSP 생성 과정 자동재생 GIF

_e4b_anim.json(실제 시뮬 시계열)으로 자동 재생·반복 GIF 생성.
왼쪽=뉴런/전극 장면(발화·sink·전극반응), 오른쪽=3파형 진행. 클릭 불필요.
실행: <ca1sim>/python.exe 12_lfp/e4b_anim_gif.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle, Polygon

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
D = json.load(open(os.path.join(FIG, "_e4b_anim.json"), encoding="utf-8"))
t = np.array(D["t"]); vm = np.array(D["vm_soma"]); im = np.array(D["im_soma"])
iy = np.array(D["im_syn"]); ve = np.array(D["ve"])
N = len(t)
sub = np.arange(0, N, 2)                       # 다운샘플(GIF 크기)
frames = list(sub) + [N - 1] * 12              # 끝에서 잠깐 정지


def lerp(a, b, x):
    a = np.array(matplotlib.colors.to_rgb(a)); b = np.array(matplotlib.colors.to_rgb(b))
    return tuple(a + (b - a) * np.clip(x, 0, 1))


fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.7], hspace=0.55, wspace=0.18)
axS = fig.add_subplot(gs[:, 0]); axS.set_xlim(0, 1); axS.set_ylim(0, 1); axS.axis("off")
axV = [fig.add_subplot(gs[i, 1]) for i in range(3)]

# --- 장면(정적 요소) ---
axS.add_patch(Rectangle((0, 0), 1, 0.12, color="#5f5e5a"))                 # 유리 전극면
axS.text(0.02, 0.05, "유리 MEA 전극면", color="w", fontsize=8)
axS.plot([0.36, 0.36], [0.30, 0.62], color="#888780", lw=6, solid_capstyle="round")
axS.plot([0.36, 0.36], [0.62, 0.86], color="#888780", lw=4, solid_capstyle="round")
glow = Circle((0.36, 0.30), 0.09, color="#E24B4A", alpha=0.0, zorder=2); axS.add_patch(glow)
soma = Circle((0.36, 0.30), 0.045, fc="#B4B2A9", ec="#5f5e5a", lw=1, zorder=3); axS.add_patch(soma)
syn = Polygon([[0.33, 0.62], [0.39, 0.62], [0.36, 0.575]], fc="#85B7EB", zorder=3); axS.add_patch(syn)
axS.text(0.41, 0.60, "SR 시냅스", fontsize=8, color="#5f5e5a")
axS.text(0.18, 0.30, "소마", fontsize=8, color="#5f5e5a")
axS.add_patch(Rectangle((0.60, 0.075), 0.05, 0.05, color="#E24B4A", zorder=4))
ring = Circle((0.625, 0.10), 0.03, fill=False, ec="#185FA5", lw=0.1, zorder=4); axS.add_patch(ring)
axS.text(0.625, 0.16, "전극 j", ha="center", fontsize=8, color="#185fa5")
contrib, = axS.plot([0.36, 0.625], [0.30, 0.10], color="#378ADD", lw=1, ls="--", alpha=0.3, zorder=1)
ttl = axS.text(0.5, 0.97, "", ha="center", fontsize=11, fontweight="bold")

# --- 파형(정적 축 + 동적 라인) ---
specs = [("① 소마 막전위 V_m (mV)", vm, "#E24B4A", -72, 42),
         ("② 소마 막전류 I_m (nA)", im, "#7F77DD", -1.3, 0.95),
         ("⑤ 전극 세포외 fEPSP (µV)", ve, "#185FA5", -2.6, 2.7)]
lines = []; cursors = []; dots = []
for ax, (lab, y, col, lo, hi) in zip(axV, specs):
    ax.plot(t, y, color=col, lw=0.6, alpha=0.18)                # 전체(옅게)
    ln, = ax.plot([], [], color=col, lw=2)                       # 진행
    cu = ax.axvline(0, color="0.5", lw=1)
    dt, = ax.plot([], [], "o", color=col, ms=5)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlim(0, 25); ax.set_ylim(lo, hi)
    ax.set_title(lab, fontsize=10, loc="left"); ax.tick_params(labelsize=8)
    lines.append(ln); cursors.append(cu); dots.append(dt)
axV[2].set_xlabel("시간 (ms)", fontsize=9)


def update(f):
    sp = np.clip((vm[f] + 66) / 104, 0, 1)
    glow.set_alpha(float(sp * 0.85)); soma.set_facecolor(lerp("#B4B2A9", "#E24B4A", sp))
    sk = np.clip(abs(iy[f]) / 0.1, 0, 1); syn.set_facecolor(lerp("#D3D1C7", "#378ADD" if iy[f] < 0 else "#D85A30", sk))
    vn = min(1, abs(ve[f]) / 2.4); cc = "#378ADD" if ve[f] < 0 else "#D85A30"
    contrib.set_alpha(0.2 + vn * 0.8); contrib.set_color(cc); contrib.set_linewidth(1 + vn * 3.5)
    ring.set_radius(0.008 + vn * 0.05); ring.set_edgecolor(cc); ring.set_linewidth(1 + vn * 2)
    ttl.set_text(f"t = {t[f]:.1f} ms    V_m {vm[f]:.0f}mV → I_m {im[f]:.2f}nA → 전극 {ve[f]:.2f}µV")
    for (lab, y, col, lo, hi), ln, cu, dt in zip(specs, lines, cursors, dots):
        ln.set_data(t[:f + 1], y[:f + 1]); cu.set_xdata([t[f], t[f]]); dt.set_data([t[f]], [y[f]])
    return [glow, soma, syn, contrib, ring, ttl] + lines + cursors + dots


fig.suptitle("한 전극에서 fEPSP가 만들어지는 과정 — 발화 → 막전류 → 거리·MEA영상법 → 전극 세포외전위 (실제 시뮬)",
             fontsize=11.5, y=0.99)
ani = FuncAnimation(fig, update, frames=frames, interval=55, blit=False)
out = os.path.join(FIG, "E4b_fepsp_play.gif")
ani.save(out, writer=PillowWriter(fps=18))
print("saved:", out)
