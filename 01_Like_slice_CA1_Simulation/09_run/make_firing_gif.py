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
ROOT = os.path.dirname(HERE)
CSVDIR = os.path.join(HERE, "spikes"); FIG = os.path.join(HERE, "figures")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
TYPE_COLOR = {"PC": (0.84, 0.19, 0.15), "PV": (0.13, 0.32, 0.66),
              "cAC": (0.18, 0.55, 0.24), "bAC": (0.95, 0.55, 0.15)}
LAYER_COLOR = {"SO": (0.42, 0.36, 0.48), "SP": (0.75, 0.22, 0.17),
               "SR": (0.18, 0.53, 0.76), "SLM": (0.15, 0.68, 0.38)}
EI_COLOR = {"EXC": (0.87, 0.52, 0.32), "INH": (0.30, 0.45, 0.70)}


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def hexrgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


def build_colors(colorby, N, fallback_type):
    """관점(colorby)별 (세포별 카테고리, 팔레트dict) 반환. gid=행번호로 slice_cells 조인."""
    if colorby == "etype":
        cats = np.array(fallback_type)
        pal = TYPE_COLOR
    else:
        c = np.load(CELLS, allow_pickle=True)
        cats = c[{"layer": "layer", "mtype": "mtype", "sclass": "sclass"}[colorby]].astype(str)[:N]
        if colorby == "layer":
            pal = LAYER_COLOR
        elif colorby == "sclass":
            pal = EI_COLOR
        else:  # mtype: tab20 팔레트
            uniq = sorted(set(cats))
            import matplotlib.cm as cm
            pal = {u: cm.tab20(i % 20)[:3] for i, u in enumerate(uniq)}
    # 존재하는 카테고리만
    present = [k for k in pal if (cats == k).any()]
    pal = {k: pal[k] for k in present}
    return cats, pal


def main():
    pos_fn = argval("--pos", os.path.join(CSVDIR, "FULL_positions.npz"))
    spk_fn = argval("--spikes", os.path.join(CSVDIR, "FULL_spikes_all.csv"))
    out = argval("--out", os.path.join(FIG, "FULL_firing_1s.gif"))
    frame_ms = float(argval("--frame_ms", "10"))
    tau = float(argval("--tau", "25"))
    rot_total = float(argval("--rot", "40"))
    elev = float(argval("--elev", "16"))

    colorby = argval("--colorby", "etype")   # etype|layer|mtype|sclass
    P = np.load(pos_fn, allow_pickle=True)
    xyz = P["xyz"].astype(float); ctype = P["type"].astype(str); N = len(xyz)
    cats, palette = build_colors(colorby, N, ctype)
    base_col = np.array([palette.get(cats[i], (0.5, 0.5, 0.5)) for i in range(N)])
    cat_total = {k: int((cats == k).sum()) for k in palette}   # 카테고리별 전체 세포수
    cat_mask = {k: (cats == k) for k in palette}

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
        active = inten > 0.15; n_active = int(active.sum())

        ax.cla()
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=rgba,
                   edgecolors="none", depthshade=False)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=elev, azim=-60 + rot_total * fr / max(1, n_frames - 1))
        cb_label = {"etype": "e-type", "layer": "해부학적 층", "mtype": "m-type",
                    "sclass": "흥분/억제"}.get(colorby, colorby)
        ax.set_title(f"슬라이스 전체 발화 (색={cb_label})  t = {tf:6.1f} ms   활성 {n_active}/{N}세포",
                     fontsize=12)
        # 범례(관점별 카테고리): 지금 발화중 / 전체 세포수
        ax.text2D(0.02, 0.985, "범례: ●유형  발화중/전체", transform=ax.transAxes,
                  fontsize=7, color="#555555")
        for i, (tn, cc) in enumerate(palette.items()):
            na = int(active[cat_mask[tn]].sum())
            ax.text2D(0.02, 0.95 - i * 0.032, f"● {tn}  {na:,}/{cat_total[tn]:,}",
                      transform=ax.transAxes, color=cc, fontsize=8, fontweight="bold")
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
