"""
cell_types.py — 네트워크에 사용한 뉴런 4종(e-type 대표)의 형태 그림 저장
============================================================================
축소 마이크로서킷은 me-model 20개 중 **e-type 대표 4종**을 사용:
  PC(피라미드, cACpyr) · PV형(cNAC) · cAC · bAC
각 세포의 형태 재구성(.swc)을 2D 투영으로 그려 개별 + 합본 PNG 저장.
색: 소마(검정)·첨단 apical(빨강)·기저 basal(파랑)·축삭 axon(회색).

실행: <ca1sim python> papers/01_Ecker2020_CA1_synaptic/04_network/cell_types.py
"""
import os
import sys
import glob

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
from common.plotstyle import set_korean_font   # noqa: E402

set_korean_font()
MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(THIS, "figures")
TYPE_COLOR = {1: "black", 2: "0.6", 3: "tab:blue", 4: "tab:red"}

# 네트워크 4종 (역할 라벨, e-type, 모델 폴더 선택자)
TYPES = [
    ("PC (피라미드)", "cACpyr", "pyramidal", None),
    ("PV형 (cNAC)",   "cNAC",   "interneurons", "_cNAC_"),
    ("cAC",           "cAC",    "interneurons", "_cAC_"),
    ("bAC",           "bAC",    "interneurons", "_bAC_"),
]


def find_swc(subdir, match):
    d = os.path.join(MODELS, subdir)
    names = sorted(os.listdir(d))
    if match:
        names = [n for n in names if match in n]
    f = glob.glob(os.path.join(d, names[0], "morphology", "*.swc"))
    return f[0]


def parse_swc(path):
    nodes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            nodes[int(p[0])] = (int(p[1]), float(p[2]), float(p[3]), int(p[6]))
    by_type = {1: [], 2: [], 3: [], 4: []}
    for nid, (typ, x, y, par) in nodes.items():
        if par in nodes:
            px, py = nodes[par][1], nodes[par][2]
            by_type.setdefault(typ, by_type.get(typ, []))
            by_type[typ].append([(px, py), (x, y)])
    return by_type


def draw(ax, by_type, title):
    for t in (2, 3, 4, 1):   # 소마 마지막(위로)
        if by_type.get(t):
            ax.add_collection(LineCollection(by_type[t], colors=TYPE_COLOR[t],
                                             linewidths=(1.4 if t == 1 else 0.4)))
    ax.autoscale(); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = []
    for label, et, subdir, match in TYPES:
        sw = find_swc(subdir, match)
        data.append((label, et, parse_swc(sw)))
        # 개별 저장
        fig, ax = plt.subplots(figsize=(3.2, 4.2))
        draw(ax, data[-1][2], f"{label}")
        fig.tight_layout()
        fname = os.path.join(OUT, f"celltype_{et}.png")
        fig.savefig(fname, dpi=130); plt.close(fig)
        print(f"[개별] {fname}")

    # 합본(가로 4종)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.3))
    fig.suptitle("네트워크 뉴런 4종 (e-type 대표) — 빨강=첨단·파랑=기저·회색=축삭·검정=소마",
                 fontsize=12, fontweight="bold")
    for ax, (label, et, bt) in zip(axes, data):
        draw(ax, bt, label)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    comb = os.path.join(OUT, "celltypes_4.png")
    fig.savefig(comb, dpi=130); plt.close(fig)
    print(f"[합본] {comb}")
    print(f"[요약] 네트워크 사용 뉴런 = 4종 (PC, PV/cNAC, cAC, bAC) · me-model 20개 중 e-type 대표")


if __name__ == "__main__":
    main()
