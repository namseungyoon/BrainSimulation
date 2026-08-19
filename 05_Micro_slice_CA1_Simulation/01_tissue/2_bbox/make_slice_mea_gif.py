# -*- coding: utf-8 -*-
"""
01_tissue/2_bbox/make_slice_mea_gif.py  —  400µm 슬라이스 슬랩 + 표면 MEA 3D 회전 GIF

확정 창(층관통_v1)의 400µm 슬랩(atlas 층, 국소 u×r×w)을 3D로 그리고,
그 **표면(w = mea_face)에 MEA 전극이 얹힌** 모습 + 반투명 MEA 면을 표시 → 회전 GIF.
(전극이 조직 속이 아니라 표면에 있음을 확인 — 실제 MEA 형태.)

재료: config/window_layout.json · data/derived/atlas_crop.npz
실행: python 01_tissue/2_bbox/make_slice_mea_gif.py  →  figures/1-2_confirmed_slice_mea_3d.gif
"""
import os
import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NPZ = os.path.join(ROOT, "data", "derived", "atlas_crop.npz")
CFG = os.path.join(ROOT, "config", "window_layout.json")
FIG = os.path.join(HERE, "figures")
LAYERS = {1: "SO", 2: "SP", 3: "SR", 4: "SLM"}
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}
PER_LAYER = 1800


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; w = cfg["window_um"]; c = w["center_local"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])
    Lu, Lr, Lw = w["long"], w["radial"], w["thick"]
    face = cfg["electrodes"].get("mea_face_w_um", -Lw / 2)

    d = np.load(NPZ, allow_pickle=True)
    regions = d["regions"]; origin = np.asarray(d["origin"], float); vs = float(d["vsize"])

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(0)
    for lab, nm in LAYERS.items():
        idx = np.argwhere(regions == lab)
        if len(idx) == 0:
            continue
        xyz = origin + idx * vs
        loc = xyz - seed
        u = loc @ L; r = loc @ R; ww = loc @ Tk
        m = (np.abs(u - c["u"]) <= Lu / 2) & (np.abs(r - c["r"]) <= Lr / 2) & (np.abs(ww - c["w"]) <= Lw / 2)
        u, r, ww = u[m], r[m], ww[m]
        if len(u) > PER_LAYER:
            s = rng.choice(len(u), PER_LAYER, replace=False); u, r, ww = u[s], r[s], ww[s]
        ax.scatter(u, r, ww, s=4, c=LC[nm], label=nm, alpha=0.35, depthshade=True, edgecolors="none")

    # MEA 면 (반투명 판) at w=face
    uu = [c["u"] - Lu / 2, c["u"] + Lu / 2]; rr = [c["r"] - Lr / 2, c["r"] + Lr / 2]
    verts = [[(uu[0], rr[0], face), (uu[1], rr[0], face), (uu[1], rr[1], face), (uu[0], rr[1], face)]]
    ax.add_collection3d(Poly3DCollection(verts, facecolor="#888888", alpha=0.18, edgecolor="#555555"))
    ax.text(c["u"], c["r"] - Lr / 2, face, "  MEA 면", color="#333333", fontsize=10, fontweight="bold")

    # 전극 (표면 위)
    stim = cfg["electrodes"]["stim_id"]
    for e in cfg["electrodes"]["list"]:
        isS = e["id"] == stim
        ax.scatter([e["u"]], [e["r"]], [face], s=220 if isS else 130,
                   marker="P" if isS else "o", c="#e23b3b" if isS else "#111111",
                   edgecolors="white", linewidths=1.6, depthshade=False, zorder=8)
        # 전극 몸통(표면에서 바깥으로 짧은 막대)
        ax.plot([e["u"], e["u"]], [e["r"], e["r"]], [face, face - 70],
                c="#e23b3b" if isS else "#333333", lw=2.5)
        ax.text(e["u"], e["r"], face - 90, e["id"], fontsize=9, fontweight="bold",
                color="#e23b3b" if isS else "#111111", ha="center")

    ax.set_box_aspect((Lu, Lr, Lw))
    ax.set_xlabel("종축 u (µm)"); ax.set_ylabel("층관통 r (µm)"); ax.set_zlabel("두께 w (µm)")
    ax.legend(loc="upper right", fontsize=9, title="층")
    ax.set_title(f'400µm 슬라이스 + 표면 MEA (전극이 조직 위에)  「{cfg["name"]}」', fontsize=12)

    def update(f):
        ax.view_init(elev=12, azim=f * 10)
        return []

    anim = FuncAnimation(fig, update, frames=36, interval=110, blit=False)
    out = os.path.join(FIG, "1-2_confirmed_slice_mea_3d.gif")
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"[GIF] {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
