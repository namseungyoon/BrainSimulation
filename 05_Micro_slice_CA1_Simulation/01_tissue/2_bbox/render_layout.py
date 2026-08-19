# -*- coding: utf-8 -*-
"""
01_tissue/2_bbox/render_layout.py  —  확정 창·전극 배치를 정적 PNG로 렌더 (기록용)

config/window_layout.json(도구로 확정)을 읽어 slice400 세포(층별 색) 위에
확정 창 + 전극(자극/기록)을 그려 figures/confirmed_layout.png 로 저장한다.

실행: python 01_tissue/2_bbox/render_layout.py
"""
import os
import glob
import json
import logging

import numpy as np
import h5py
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find(p):
    return sorted(glob.glob(os.path.join(DATA, "**", p), recursive=True), key=len)[0]


def decode(g, name):
    lib = [s.decode() if isinstance(s, bytes) else s for s in g["@library"][name][:]]
    return np.array(lib, dtype=object)[g[name][:]]


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]
    seed = np.array(fr["seed"]); Ld = np.array(fr["long_dir"]); Rd = np.array(fr["radial_dir"])

    mask, h = nrrd.read(find(os.path.join("slices", "slice400.nrrd")))
    origin = np.asarray(h["space origin"], float); vs = float(h["space directions"][0][0])
    nx, ny, nz = mask.shape
    with h5py.File(find(os.path.join("hippocampus_neurons", "nodes.h5")), "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        layer = decode(g, "layer")
    idx = np.floor((xyz - origin) / vs).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    ins = np.zeros(len(xyz), bool); ii = idx[ok]
    ins[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sl = np.where(ins)[0]
    d = xyz[sl] - seed
    u = d @ Ld; r = d @ Rd; Ls = layer[sl]

    win = cfg["window_um"]; cu = win["center_local"]["u"]; cr = win["center_local"]["r"]
    Lu = win["long"]; Lr = win["radial"]
    eps = cfg["electrodes"]["list"]; stim = cfg["electrodes"]["stim_id"]

    def render(zoom):
        fig, ax = plt.subplots(figsize=(9, 8) if zoom else (11, 8))
        for L in LAYER_ORDER:
            m = Ls == L
            if m.any():
                ax.scatter(u[m], r[m], s=(14 if zoom else 5), c=LC[L], label=L, alpha=0.6)
        ax.add_patch(Rectangle((cu - Lu / 2, cr - Lr / 2), Lu, Lr, fill=False, ec="black", lw=2.4))
        for e in eps:
            isS = e["id"] == stim
            col = "#e23b3b" if isS else "#111111"
            mk = "P" if isS else "o"
            ms = (18 if isS else 13) if zoom else (10 if isS else 6)
            ax.plot(e["u"], e["r"], marker=mk, ms=ms, mfc=col, mec="white", mew=1.5, zorder=5)
            if zoom:
                ax.annotate(f'{"자극 " if isS else "기록 "}{e["id"]}·{e["layer"]}',
                            (e["u"], e["r"]), textcoords="offset points", xytext=(13, 0),
                            va="center", fontsize=10, fontweight="bold", color=col)
        ax.set_aspect("equal"); ax.set_xlabel("종축 proximodistal (µm)")
        ax.set_ylabel("층관통 radial (µm, SP=0)")
        if zoom:
            ax.set_xlim(cu - Lu * 0.85, cu + Lu * 0.85); ax.set_ylim(cr - Lr * 0.68, cr + Lr * 0.68)
            ax.set_title(f'확정 배치 「{cfg["name"]}」 상세 — 창 {Lu:.0f}×{Lr:.0f}×{win["thick"]:.0f}µm · '
                         f'전극 자극{stim}(SR)/기록(SO·SP)', fontsize=12)
        else:
            ax.set_title(f'확정 배치 「{cfg["name"]}」 — slice400 전체 속 창 위치', fontsize=12)
        ax.legend(title="층", loc="upper right")
        fig.tight_layout()
        out = os.path.join(FIG, "1-2_confirmed_detail.png" if zoom else "1-2_confirmed_overview.png")
        fig.savefig(out, dpi=140); plt.close(fig); return out

    o1 = render(False); o2 = render(True)
    print(f"[확정 그림] {cfg['name']} → {o1} · {o2}")


if __name__ == "__main__":
    main()
