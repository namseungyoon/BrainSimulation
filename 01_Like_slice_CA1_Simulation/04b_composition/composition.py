# -*- coding: utf-8 -*-
"""
04b_composition/composition.py  —  단계 4b: 세포 조성 (V2a)

목적:
  nodes.h5 + atlas 부피로 CA1 세포 조성을 정량화한다.
    - 층별(소마 기준) 세포수 · m-type · e-type 분해
    - 층별 부피(brain_regions 복셀) → 밀도(cells/mm^3)
    - 층별/전체 E:I 비율

검증 (V2a): 전체 E:I ≈ 89:11 일치, 층별 밀도(특히 SP 추체층 고밀도) 타당.

산출 그림 (한글):
  figures/V2a_1_layer_counts_EI.png   : 층별 세포수 + E:I 분해 (선형/로그)
  figures/V2a_2_mtype_by_layer.png    : m-type × 층 개수 히트맵
  figures/V2a_3_etype_by_layer.png    : e-type × 층 개수 히트맵
  figures/V2a_4_density.png           : 층별 밀도 (cells/mm^3)
  figures/V2a_5_EI_per_layer.png      : 층별 E:I 100% 누적막대
  figures/V2a_6_mtype_composition.png : m-type 전체 조성 (공칭층 색)
출력: composition.json

실행:
  python 04b_composition/composition.py
"""
import os
import json
from collections import Counter

import numpy as np
import h5py
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas")
NODES_H5 = os.path.join(ROOT, "data", "circuit", "networks", "nodes",
                        "hippocampus_neurons", "nodes.h5")
POP = "nodes/hippocampus_neurons/0"
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_ID = {"SO": 1, "SP": 2, "SR": 3, "SLM": 4}
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452",
               "SR": "#55A868", "SLM": "#C44E52"}
VOX_UM3 = 16.0 ** 3      # 복셀 부피


def decode(grp, name):
    lib = [s.decode() if isinstance(s, bytes) else s
           for s in grp["@library"][name][:]]
    return np.array(lib, dtype=object)[grp[name][:]]


def load_nodes():
    with h5py.File(NODES_H5, "r") as f:
        g = f[POP]
        layer = decode(g, "layer")
        mtype = decode(g, "mtype")
        etype = decode(g, "etype")
        sclass = decode(g, "synapse_class")
    return layer, mtype, etype, sclass


def layer_volumes():
    br, _ = nrrd.read(os.path.join(ATLAS, "brain_regions.nrrd"))
    vol = {}
    for L in LAYER_ORDER:
        nvox = int((br == LAYER_ID[L]).sum())
        vol[L] = nvox * VOX_UM3 / 1e9        # mm^3
    return vol


def main():
    layer, mtype, etype, sclass = load_nodes()
    N = len(layer)
    vol = layer_volumes()
    print(f"[load] N={N:,}, 층부피(mm^3)="
          + ", ".join(f"{L}:{vol[L]:.2f}" for L in LAYER_ORDER))

    # 층별 세포수 / E:I
    per_layer = {}
    for L in LAYER_ORDER:
        m = layer == L
        n = int(m.sum())
        exc = int((sclass[m] == "EXC").sum())
        inh = int((sclass[m] == "INH").sum())
        per_layer[L] = {"n": n, "exc": exc, "inh": inh,
                        "density": n / vol[L] if vol[L] else 0.0}
    # m-type × layer, e-type × layer
    mtypes = sorted(set(mtype), key=lambda t: -int((mtype == t).sum()))
    etypes = sorted(set(etype), key=lambda t: -int((etype == t).sum()))
    M = np.zeros((len(LAYER_ORDER), len(mtypes)), int)
    E = np.zeros((len(LAYER_ORDER), len(etypes)), int)
    for i, L in enumerate(LAYER_ORDER):
        m = layer == L
        cm = Counter(mtype[m]); ce = Counter(etype[m])
        for j, t in enumerate(mtypes):
            M[i, j] = cm.get(t, 0)
        for j, t in enumerate(etypes):
            E[i, j] = ce.get(t, 0)

    exc_tot = int((sclass == "EXC").sum()); inh_tot = N - exc_tot
    print(f"[V2a] 전체 E:I = {100*exc_tot/N:.1f} : {100*inh_tot/N:.1f}")
    for L in LAYER_ORDER:
        d = per_layer[L]
        print(f"  {L:4s}: n={d['n']:>8,}  밀도={d['density']:>10,.0f} cells/mm^3  "
              f"E:I={d['exc']:,}:{d['inh']:,}")

    # ------- 그림 -------
    _fig_counts_EI(per_layer)
    _fig_heatmap(M, mtypes, "V2a-2  m-type × 층 세포수 (로그색)",
                 "V2a_2_mtype_by_layer.png", rot=45)
    _fig_heatmap(E, etypes, "V2a-3  e-type × 층 세포수 (로그색)",
                 "V2a_3_etype_by_layer.png", rot=0)
    _fig_density(per_layer, vol)
    _fig_EI_per_layer(per_layer)
    _fig_mtype_comp(mtype)

    out = {
        "step": "4b composition (V2a)",
        "N": N,
        "overall_EI": {"exc": exc_tot, "inh": inh_tot,
                       "ratio": f"{100*exc_tot/N:.1f}:{100*inh_tot/N:.1f}"},
        "layer_volume_mm3": vol,
        "per_layer": per_layer,
    }
    with open(os.path.join(HERE, "composition.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] composition.json + figures -> {FIG}")


def _fig_counts_EI(per_layer):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(LAYER_ORDER))
    exc = [per_layer[L]["exc"] for L in LAYER_ORDER]
    inh = [per_layer[L]["inh"] for L in LAYER_ORDER]
    for ax, logy in zip(axes, [False, True]):
        ax.bar(x, exc, color="#DD8452", label="흥분 EXC")
        ax.bar(x, inh, bottom=exc, color="#4C72B0", label="억제 INH")
        ax.set_xticks(x); ax.set_xticklabels(LAYER_ORDER)
        ax.set_ylabel("세포 수")
        if logy:
            ax.set_yscale("log"); ax.set_title("로그 스케일")
        else:
            ax.set_title("선형 스케일")
            for i, L in enumerate(LAYER_ORDER):
                ax.text(i, per_layer[L]["n"], f"{per_layer[L]['n']:,}",
                        ha="center", va="bottom", fontsize=8)
        ax.legend()
    fig.suptitle("V2a-1  층별(소마 기준) 세포수 · 흥분/억제 분해")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2a_1_layer_counts_EI.png"), dpi=130)
    plt.close(fig)


def _fig_heatmap(mat, cols, title, fname, rot):
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.9), 4.2))
    disp = np.log10(mat + 1)
    im = ax.imshow(disp, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=rot, ha="right" if rot else "center", fontsize=8)
    ax.set_yticks(range(len(LAYER_ORDER))); ax.set_yticklabels(LAYER_ORDER)
    for i in range(len(LAYER_ORDER)):
        for j in range(len(cols)):
            if mat[i, j] > 0:
                ax.text(j, i, f"{mat[i,j]:,}", ha="center", va="center",
                        fontsize=7, color="w" if disp[i, j] > disp.max() * 0.5 else "k")
    fig.colorbar(im, ax=ax, label="log10(세포수+1)")
    ax.set_title(title); ax.set_ylabel("소마 층")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close(fig)


def _fig_density(per_layer, vol):
    fig, ax = plt.subplots(figsize=(7, 5))
    d = [per_layer[L]["density"] for L in LAYER_ORDER]
    bars = ax.bar(LAYER_ORDER, d, color=[LAYER_COLOR[L] for L in LAYER_ORDER])
    ax.set_yscale("log")
    ax.set_ylabel("밀도 (cells/mm³, 로그)")
    for i, L in enumerate(LAYER_ORDER):
        ax.text(i, d[i], f"{d[i]:,.0f}\n({vol[L]:.2f}mm³)",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("V2a-4  층별 세포 밀도 (소마수 / 층부피)\nSP 추체층이 압도적 고밀도")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2a_4_density.png"), dpi=130)
    plt.close(fig)


def _fig_EI_per_layer(per_layer):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(LAYER_ORDER))
    exc_f = [100 * per_layer[L]["exc"] / per_layer[L]["n"] for L in LAYER_ORDER]
    inh_f = [100 * per_layer[L]["inh"] / per_layer[L]["n"] for L in LAYER_ORDER]
    ax.barh(y, exc_f, color="#DD8452", label="흥분 EXC")
    ax.barh(y, inh_f, left=exc_f, color="#4C72B0", label="억제 INH")
    ax.set_yticks(y); ax.set_yticklabels(LAYER_ORDER)
    ax.set_xlabel("비율 (%)"); ax.set_xlim(0, 100)
    for i, L in enumerate(LAYER_ORDER):
        ax.text(exc_f[i] / 2, i, f"{exc_f[i]:.0f}%", ha="center", va="center",
                color="w", fontsize=8)
    ax.legend(loc="lower right")
    ax.set_title("V2a-5  층별 흥분:억제 비율 (SP는 추체 우세, SO/SR/SLM은 인터뉴런)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2a_5_EI_per_layer.png"), dpi=130)
    plt.close(fig)


def _fig_mtype_comp(mtype):
    c = Counter(mtype)
    items = sorted(c.items(), key=lambda kv: -kv[1])
    keys = [k for k, _ in items]; vals = [v for _, v in items]
    # 공칭층(접두사)로 색
    def pref(t):
        for L in LAYER_ORDER:
            if t.startswith(L + "_"):
                return L
        return "SP"
    colors = [LAYER_COLOR[pref(k)] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(keys[::-1], vals[::-1], color=colors[::-1])
    ax.set_xscale("log"); ax.set_xlabel("세포 수 (로그)")
    ax.set_title("V2a-6  m-type 전체 조성 (12종, 색=공칭 층)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LAYER_COLOR[L], label=L) for L in LAYER_ORDER],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "V2a_6_mtype_composition.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
