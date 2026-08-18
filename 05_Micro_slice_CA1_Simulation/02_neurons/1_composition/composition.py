# -*- coding: utf-8 -*-
"""
02_neurons/1_composition/composition.py  —  2-1: 창 세포 조성 집계 (V2a)

확정 창(층관통_v1) 안 세포의 조성을 SONATA nodes.h5에서 층별로 집계한다.
  - 층별 세포수, m-type(12) 분포, e-type(4), 흥분/억제(EXC/INH), 형태강(PYR/INT)
  - 밀도(cells/mm³), E:I 비, 대표 형태 개수(morphology 다양성)
결과: data/derived/composition.json (2-2 배치 재사용) + 그림 V2a_composition.png

재료: config/window_layout.json · circuit nodes.h5(hippocampus_neurons) · slice400.nrrd
실행: python 02_neurons/1_composition/composition.py
"""
import os
import glob
import json
import logging
from collections import Counter

import numpy as np
import h5py
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    hip = os.path.join("hippocampus_neurons", "nodes.h5")
    with h5py.File(find(hip), "r") as f:
        g = f["nodes/hippocampus_neurons/0"]
        xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], 1)
        layer = decode(g, "layer")
        mtype = decode(g, "mtype")
        etype = decode(g, "etype")
        sclass = decode(g, "synapse_class")
        mclass = decode(g, "morph_class")
        morph = decode(g, "morphology")

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
    Lw = layer[win]; Mt = mtype[win]; Et = etype[win]; Sc = sclass[win]; Mc = mclass[win]; Mo = morph[win]
    N = len(win)

    vol_mm3 = (w["long"] * w["radial"] * w["thick"]) * 1e-9  # µm³ → mm³
    n_exc = int(np.sum(Sc == "EXC")); n_inh = int(np.sum(Sc == "INH"))

    print(f"=== 2-1 창 세포 조성 (층관통_v1) ===")
    print(f"[총] {N:,}개 · 창 부피 {vol_mm3*1e3:.1f}e-3 mm³ · 밀도 {N/vol_mm3:,.0f} cells/mm³")
    print(f"[흥분/억제] EXC {n_exc:,} ({100*n_exc/N:.1f}%) · INH {n_inh:,} ({100*n_inh/N:.1f}%) · E:I = {n_exc/max(n_inh,1):.1f}:1")
    print(f"[형태강] " + " · ".join(f"{k} {v:,}" for k, v in Counter(Mc).most_common()))
    print(f"[e-type] " + " · ".join(f"{k} {v:,}" for k, v in Counter(Et).most_common()))
    print(f"[형태 다양성] 고유 morphology {len(set(Mo)):,}종")
    print("\n[층별 세포수]")
    comp = {"total": N, "volume_mm3": vol_mm3, "density_per_mm3": N / vol_mm3,
            "n_exc": n_exc, "n_inh": n_inh, "by_layer": {}, "by_mtype": {}}
    for Ln in LAYER_ORDER:
        m = Lw == Ln
        nl = int(m.sum())
        if nl == 0:
            continue
        ne = int(np.sum(Sc[m] == "EXC")); ni = nl - ne
        mts = Counter(Mt[m])
        comp["by_layer"][Ln] = {"n": nl, "exc": ne, "inh": ni, "mtypes": dict(mts)}
        print(f"  {Ln:<4} {nl:>5,}  (EXC {ne:>4,} / INH {ni:>4,})  mtypes: " +
              ", ".join(f"{k}={v}" for k, v in mts.most_common()))
    comp["by_mtype"] = dict(Counter(Mt))
    print("\n[m-type 전체]")
    for k, v in Counter(Mt).most_common():
        print(f"  {k:<10} {v:>5,} ({100*v/N:.1f}%)")

    json.dump(comp, open(os.path.join(DERIVED, "composition.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[V2a] 조성표 저장 -> data/derived/composition.json")
    fig_composition(comp, N, n_exc, n_inh)
    print(f"[V2a] 그림 저장 -> {FIG}/V2a_composition.png")


def fig_composition(comp, N, n_exc, n_inh):
    layers = [L for L in LAYER_ORDER if L in comp["by_layer"]]
    int_mts = sorted({m for L in comp["by_layer"].values() for m in L["mtypes"] if m != "SP_PC"})
    cmap = plt.get_cmap("tab20")
    mcol = {m: cmap(i % 20) for i, m in enumerate(int_mts)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1.5, 1]})

    # (a) 층별 EXC/INH 총계
    ax = axes[0]
    ne = [comp["by_layer"][L]["exc"] for L in layers]
    ni = [comp["by_layer"][L]["inh"] for L in layers]
    ax.bar(layers, ne, color="#DD8452", label="EXC(SP_PC)")
    ax.bar(layers, ni, bottom=ne, color="#4C72B0", label="INH")
    for i, L in enumerate(layers):
        ax.text(i, comp["by_layer"][L]["n"], f"{comp['by_layer'][L]['n']:,}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("세포수"); ax.set_title("(a) 층별 세포수 (EXC/INH)")
    ax.legend(fontsize=9)

    # (b) 억제뉴런 m-type 층별 (SP_PC 제외 → 다양성 가시화)
    ax = axes[1]
    bottom = np.zeros(len(layers))
    for m in int_mts:
        vals = np.array([comp["by_layer"][L]["mtypes"].get(m, 0) for L in layers], float)
        ax.bar(layers, vals, bottom=bottom, color=mcol[m], label=m, edgecolor="white", lw=0.4)
        bottom += vals
    for i, L in enumerate(layers):
        tot_inh = comp["by_layer"][L]["inh"]
        if tot_inh:
            ax.text(i, tot_inh, f"{tot_inh}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.set_ylabel("억제뉴런 수"); ax.set_title("(b) 층별 억제뉴런 m-type (SP_PC 제외)")
    ax.legend(ncol=2, fontsize=8, loc="upper right", title="m-type")

    # (c) E:I 파이
    ax = axes[2]
    ax.pie([n_exc, n_inh], labels=[f"EXC\n{n_exc:,}", f"INH\n{n_inh:,}"],
           colors=["#DD8452", "#4C72B0"], autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white"))
    ax.set_title(f"(c) 흥분:억제 = {n_exc/max(n_inh,1):.1f}:1\n총 {N:,}개 · {comp['density_per_mm3']:,.0f} cells/mm³")

    fig.suptitle("V2a  창 세포 조성 (층관통_v1 · 국소 atlas)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V2a_composition.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
