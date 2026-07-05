# -*- coding: utf-8 -*-
"""
09_run/make_firing_gif.py  —  전체 슬라이스 발화 애니메이션 GIF

FULL_positions.npz(gid,xyz,type) + FULL_spikes_all.csv(gid,type,t_ms) 를 읽어
시간에 따라 세포가 발화(플래시 후 감쇠)하는 모습을 3D 산점도로 애니메이션.
표준 지침(3D는 회전)도 만족하도록 카메라를 clip 동안 완만히 회전.

  - 비발화 세포: 작고 흐린 점(타입색). 발화 세포: 스파이크 순간 밝게 커졌다 tau 로 감쇠.
  - 프레임당 시간창 frame_ms. 타이틀에 현재 t(ms)·활성 세포수.

실행: python 09_run/make_firing_gif.py [--tstop 1000] [--frame_ms 10] [--tau 25]
      [--spikes spikes/FULL_spikes_all.csv] [--pos spikes/FULL_positions.npz]
      [--out figures/FULL_firing_1s.gif] [--rot 40] [--elev 16]
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
CSVDIR = os.path.join(HERE, "spikes"); FIG = os.path.join(HERE, "figures")
TYPE_COLOR = {"PC": (0.84, 0.19, 0.15), "PV": (0.13, 0.32, 0.66),
              "cAC": (0.18, 0.55, 0.24), "bAC": (0.95, 0.55, 0.15)}


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    pos_fn = argval("--pos", os.path.join(CSVDIR, "FULL_positions.npz"))
    spk_fn = argval("--spikes", os.path.join(CSVDIR, "FULL_spikes_all.csv"))
    out = argval("--out", os.path.join(FIG, "FULL_firing_1s.gif"))
    frame_ms = float(argval("--frame_ms", "10"))
    tau = float(argval("--tau", "25"))
    rot_total = float(argval("--rot", "40"))
    elev = float(argval("--elev", "16"))

    P = np.load(pos_fn, allow_pickle=True)
    xyz = P["xyz"].astype(float); ctype = P["type"].astype(str); N = len(xyz)
    base_col = np.array([TYPE_COLOR.get(t, (0.5, 0.5, 0.5)) for t in ctype])

    gids = []; times = []
    with open(spk_fn, encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            gids.append(int(row[0])); times.append(float(row[2]))
    gids = np.array(gids, dtype=int); times = np.array(times, dtype=float)
    tstop = float(argval("--tstop", str(int(np.ceil(times.max())) if len(times) else 100)))
    n_frames = max(2, int(round(tstop / frame_ms)))
    print(f"[GIF] 세포 {N} · 스파이크 {len(times):,} · tstop {tstop:.0f}ms · "
          f"{n_frames}프레임(frame={frame_ms}ms, tau={tau}ms)", flush=True)

    fig = plt.figure(figsize=(9, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])))
    lo = xyz.min(0); hi = xyz.max(0)
    frames = []
    win = 4 * tau  # 감쇠 유효창
    for fr in range(n_frames):
        tf = (fr + 1) * frame_ms
        sel = (times <= tf) & (times > tf - win)
        inten = np.zeros(N)
        if sel.any():
            contrib = np.exp(-(tf - times[sel]) / tau)
            np.maximum.at(inten, gids[sel], contrib)
        # 색/크기/투명도
        col = base_col * (0.34 + 0.66 * inten[:, None])       # 어두운 기저→밝게
        col = np.clip(col, 0, 1)
        rgba = np.concatenate([col, (0.20 + 0.80 * inten)[:, None]], axis=1)
        size = 2.2 + 30.0 * inten
        n_active = int((inten > 0.15).sum())

        ax.cla()
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=rgba,
                   edgecolors="none", depthshade=False)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=elev, azim=-60 + rot_total * fr / max(1, n_frames - 1))
        ax.set_title(f"슬라이스 전체 발화  t = {tf:6.1f} ms   활성 {n_active}/{N}세포",
                     fontsize=12)
        # 범례(첫 프레임에만 텍스트로)
        for i, (tn, cc) in enumerate(TYPE_COLOR.items()):
            ax.text2D(0.02, 0.96 - i * 0.04, f"● {tn}", transform=ax.transAxes,
                      color=cc, fontsize=9, fontweight="bold")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE))
        if (fr + 1) % 20 == 0:
            print(f"   {fr+1}/{n_frames} 프레임", flush=True)

    os.makedirs(FIG, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=True)
    plt.close(fig)
    print(f"[OK] → {out}  ({len(frames)}프레임)", flush=True)


if __name__ == "__main__":
    main()
