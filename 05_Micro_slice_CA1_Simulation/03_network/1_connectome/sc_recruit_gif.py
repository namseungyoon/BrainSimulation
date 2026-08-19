# -*- coding: utf-8 -*-
"""
03_network/1_connectome/sc_recruit_gif.py  —  3-1: 자극 세기 recruitment GIF

E3 자극 반경 R을 키우며(=자극 세기↑) 섬유(가상 CA3 축삭)가 점점 더 많이
'모집(recruit)'되는 과정을 애니메이션으로 보여준다. 섬유 정체성 기반:
한 축삭이 E3 반경에 걸리면 그 축삭의 모든 시냅스(u 전체 발산)가 함께 발화.
좌: 공간 활성 지도(u-r) / 우: recruitment 곡선(R vs 활성 시냅스·세포).

실행: python 03_network/1_connectome/sc_recruit_gif.py
결과: 03_network/1_connectome/figures/3-1_recruit.gif
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")


def main():
    d = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    f = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    uvw = d["uvw"].astype(float); post = d["post_gid"]; dist_e3 = d["dist_e3"].astype(float)
    fiber_id = f["fiber_id"]
    U, R = uvw[:, 0], uvw[:, 1]
    N = len(uvw)

    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; seed = np.array(fr["seed"])
    Mrows = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    e3loc = (d["e3_xyz"] - seed) @ Mrows

    radii = np.arange(30, 431, 40.0)   # 자극 세기 단계
    # 각 R에서 활성 집합(섬유 모집) 사전계산
    frames = []
    for Rr in radii:
        near = dist_e3 < Rr
        hit = np.unique(fiber_id[near])
        act = np.isin(fiber_id, hit)
        frames.append((Rr, act, int(len(hit)), int(act.sum()), int(len(np.unique(post[act])))))
    curve_syn = [fr3 for _, _, _, fr3, _ in frames]
    curve_cell = [fr4 for _, _, _, _, fr4 in frames]

    rng = np.random.default_rng(1)
    s = rng.choice(N, min(50000, N), replace=False)
    Us, Rs = U[s], R[s]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})

    def draw(i):
        axL.clear(); axR.clear()
        Rr, act, nhit, nsyn, ncell = frames[i]
        axL.scatter(Us, Rs, s=1.5, c="#dddddd", alpha=0.3, linewidths=0)
        am = act[s]
        axL.scatter(Us[am], Rs[am], s=5, c="#C44E52", alpha=0.75, linewidths=0)
        axL.scatter([e3loc[0]], [e3loc[1]], s=280, marker="*", c="red", edgecolors="black", zorder=6)
        circ = plt.Circle((e3loc[0], e3loc[1]), Rr, fill=False, ec="#2222aa", lw=1.6, ls="--")
        axL.add_patch(circ)
        axL.annotate("E3", (e3loc[0], e3loc[1]), fontsize=11, ha="center", va="bottom",
                     xytext=(0, 12), textcoords="offset points")
        axL.set_xlim(U.min() - 20, U.max() + 20); axL.set_ylim(R.min() - 20, R.max() + 20)
        axL.set_xlabel("종축 u (µm)"); axL.set_ylabel("층관통 r (µm)")
        axL.set_title(f"자극 세기 R={Rr:.0f}µm → 모집 축삭 {nhit:,}개\n활성 시냅스 {nsyn:,} · 세포 {ncell:,}")

        axR.plot(radii[:i + 1], np.array(curve_syn[:i + 1]) / 1e3, "o-", color="#C44E52", lw=2, label="활성 시냅스")
        axR.plot(radii[:i + 1], np.array(curve_cell[:i + 1]) / 1e3 * 10, "s-", color="#4C72B0", lw=2, label="활성 세포 ×10")
        axR.set_xlim(0, radii[-1] + 20); axR.set_ylim(0, max(curve_syn) / 1e3 * 1.1)
        axR.set_xlabel("자극 세기 R (µm)"); axR.set_ylabel("활성 시냅스 (×10³)")
        axR.set_title("recruitment 곡선 (모집량 — 기하학 계산, 시뮬 아님)"); axR.legend(loc="upper left"); axR.grid(alpha=0.3)
        fig.suptitle("3-1  SC 자극 recruitment — 세기↑ → 축삭 동원↑ (섬유 정체성 · 배선前 기하학)", fontsize=13)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out = os.path.join(FIG, "3-1_recruit.gif")
    anim.save(out, writer=PillowWriter(fps=1.5))
    plt.close(fig)
    print(f"[recruitment] R {radii[0]:.0f}→{radii[-1]:.0f}µm · {len(frames)}프레임")
    for Rr, _, nhit, nsyn, ncell in frames:
        print(f"  R={Rr:5.0f}µm  축삭 {nhit:5,}  시냅스 {nsyn:7,}  세포 {ncell:5,}")
    print(f"[GIF] -> {out}")


if __name__ == "__main__":
    main()
