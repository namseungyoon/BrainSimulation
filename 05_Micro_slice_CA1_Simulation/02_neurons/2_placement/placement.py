# -*- coding: utf-8 -*-
"""
02_neurons/2_placement/placement.py  —  2-2: 창 세포 3D 배치 추출 (2-2)

확정 창(층관통_v1) 안 세포를 실제 좌표로 추출해 **세포 원장**을 만든다.
  - 세포별: node_id · xyz(원자좌표) · uvw(국소) · layer · mtype · etype
             synapse_class · morphology · model_template · orientation(wxyz)
  - 저장: data/derived/window_cells.npz  (이후 연결·배선·시뮬레이션 재사용)
  - 그림 2-2_placement.png: 국소 3정투영(u-r/u-w/r-w) + 층별 색 + 전극/창

재료: config/window_layout.json · circuit nodes.h5(hippocampus_neurons) · slice400.nrrd
실행: python 02_neurons/2_placement/placement.py
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
from matplotlib.patches import Patch, Rectangle
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "window_layout.json")
DERIVED = os.path.join(DATA, "derived")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DERIVED, exist_ok=True)
LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find(p):
    return sorted(glob.glob(os.path.join(DATA, "**", glob.escape(p)), recursive=True), key=len)[0]


def decode(g, name):
    lib = [s.decode() if isinstance(s, bytes) else s for s in g["@library"][name][:]]
    return np.array(lib, dtype=object)[g[name][:]]


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    fr = cfg["frame_um"]; w = cfg["window_um"]; c = w["center_local"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])

    mask, h = nrrd.read(find(os.path.join("slices", "slice400.nrrd")))
    origin = np.asarray(h["space origin"], float); vs = float(h["space directions"][0][0])
    nx, ny, nz = mask.shape
    with h5py.File(find(os.path.join("hippocampus_neurons", "nodes.h5")), "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        quat = np.stack([g[f"orientation_{a}"][:] for a in "wxyz"], 1)
        layer = decode(g, "layer"); mtype = decode(g, "mtype"); etype = decode(g, "etype")
        sclass = decode(g, "synapse_class")
        morph = decode(g, "morphology"); mtpl = decode(g, "model_template")

    idx = np.floor((xyz - origin) / vs).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz)
    ins = np.zeros(len(xyz), bool); ii = idx[ok]
    ins[ok] = mask[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    sl = np.where(ins)[0]
    d = xyz[sl] - seed
    u = d @ L; r = d @ R; ww = d @ Tk
    inwin = ((np.abs(u - c["u"]) <= w["long"] / 2) & (np.abs(r - c["r"]) <= w["radial"] / 2) &
             (np.abs(ww - c["w"]) <= w["thick"] / 2))
    win = sl[inwin]
    uvw = np.stack([u[inwin], r[inwin], ww[inwin]], 1)
    Lw = layer[win]

    np.savez_compressed(
        os.path.join(DERIVED, "window_cells.npz"),
        node_id=win.astype(np.int64), xyz=xyz[win], uvw=uvw,
        layer=Lw.astype("U8"), mtype=mtype[win].astype("U16"),
        etype=etype[win].astype("U16"), synapse_class=sclass[win].astype("U4"),
        morphology=morph[win].astype("U128"), model_template=mtpl[win].astype("U128"),
        orientation_wxyz=quat[win],
        frame_seed=seed, frame_long=L, frame_radial=R, frame_thick=Tk,
        window_um=np.array([w["long"], w["radial"], w["thick"]], float),
        center_local=np.array([c["u"], c["r"], c["w"]], float),
    )
    print(f"=== 2-2 창 세포 배치 추출 ===")
    print(f"[세포 원장] {len(win):,}개 -> data/derived/window_cells.npz")
    print(f"[국소범위] u[{uvw[:,0].min():.0f},{uvw[:,0].max():.0f}] "
          f"r[{uvw[:,1].min():.0f},{uvw[:,1].max():.0f}] w[{uvw[:,2].min():.0f},{uvw[:,2].max():.0f}] µm")
    for Ln in LAYER_ORDER:
        n = int(np.sum(Lw == Ln))
        if n:
            print(f"   {Ln:<4} {n:>5,}")

    ecfg = cfg.get("electrodes", {})
    elecs = ecfg.get("list", [])
    face_w = ecfg.get("mea_face_w_um", -w["thick"] / 2)
    fig_placement(uvw, Lw, c, w, elecs, face_w, len(win))
    print(f"\n[2-2] 그림 저장 -> {FIG}/2-2_placement.png")


def fig_placement(uvw, Lw, c, w, elecs, face_w, n):
    U, Rr, W = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    hu, hr, hw = w["long"] / 2, w["radial"] / 2, w["thick"] / 2
    panels = [("종축 u (µm)", "층관통 r (µm)", U, Rr, c["u"], c["r"], hu, hr, "u-r (정면)"),
              ("종축 u (µm)", "두께 w (µm)", U, W, c["u"], c["w"], hu, hw, "u-w (평면)"),
              ("층관통 r (µm)", "두께 w (µm)", Rr, W, c["r"], c["w"], hr, hw, "r-w (단면)")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    for ax, (xl, yl, X, Y, cx, cy, hx, hy, ttl) in zip(axes, panels):
        for Ln in LAYER_ORDER:
            m = Lw == Ln
            if m.any():
                ax.scatter(X[m], Y[m], s=5, c=LC[Ln], alpha=0.4, label=Ln, linewidths=0)
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ec="black", lw=1.8, ls="--"))
        for e in elecs:
            eu, er, ew = e["u"], e["r"], face_w
            ex, ey = (eu, er) if ttl.startswith("u-r") else (eu, ew) if ttl.startswith("u-w") else (er, ew)
            mk = "*" if e.get("role") == "stim" else "s"
            ax.scatter(ex, ey, s=180, marker=mk, c="red", edgecolors="black", zorder=5)
            ax.annotate(e["id"], (ex, ey), fontsize=8, ha="center", va="bottom", xytext=(0, 8),
                        textcoords="offset points")
        ax.set_aspect("equal"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
    axes[0].legend(handles=[Patch(facecolor=LC[v], label=v) for v in LAYER_ORDER] +
                   [plt.Line2D([], [], marker="*", ls="", mfc="red", mec="black", ms=12, label="자극전극"),
                    plt.Line2D([], [], marker="s", ls="", mfc="red", mec="black", ms=9, label="기록전극")],
                   loc="upper left", fontsize=8)
    fig.suptitle(f"2-2  창 세포 3D 배치 (층관통_v1) — {n:,}개 · 국소 3정투영 + MEA 전극", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2-2_placement.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
