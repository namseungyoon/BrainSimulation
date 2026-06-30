"""
4_morphology.py — 뉴런 형태(morphology) 시각화 (.swc 직접 파싱)
============================================================================
보유 세포의 형태 재구성(.swc)을 읽어 수상돌기 나무를 그린다.
구역 색: 소마(검정) · 첨단수상돌기 apical(빨강) · 기저수상돌기 basal(파랑) · 축삭 axon(회색).
산출:
  - figures/4_morphology.png      : PC + 인터뉴런 e-type 대표 비교(4종)
  - figures/4_morphology_PC.json  : PC 형태 SVG 경로(채팅 인라인 렌더용)

실행: conda activate ca1sim
      python papers/01_Ecker2020_CA1_synaptic/02_neurons/4_morphology.py
"""
import os
import sys
import glob
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
from common.plotstyle import set_korean_font   # noqa: E402
set_korean_font()
MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(THIS, "figures")

TYPE_COLOR = {1: "black", 2: "0.6", 3: "tab:blue", 4: "tab:red"}   # soma/axon/basal/apical
TYPE_NAME = {1: "soma", 2: "axon(축삭)", 3: "basal(기저)", 4: "apical(첨단)"}


def parse_swc(path):
    nodes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            nodes[int(p[0])] = (int(p[1]), float(p[2]), float(p[3]), float(p[5]), int(p[6]))
    segs = []   # (x0,y0,x1,y1,type)
    for nid, (typ, x, y, r, par) in nodes.items():
        if par in nodes:
            segs.append((nodes[par][1], nodes[par][2], x, y, typ))
    return segs


def swc_path(model_dir):
    f = glob.glob(os.path.join(model_dir, "morphology", "*.swc"))
    return f[0] if f else None


def first_int(etype):
    d = os.path.join(MODELS, "interneurons")
    for name in sorted(os.listdir(d)):
        if f"_{etype}_" in name:
            return os.path.join(d, name)
    return None


def draw(ax, segs, title):
    for x0, y0, x1, y1, typ in segs:
        ax.plot([x0, x1], [y0, y1], color=TYPE_COLOR.get(typ, "0.5"), lw=0.5)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("µm", fontsize=8)


def make_svg_json(segs, W=420, H=620, pad=12):
    xs = [c for s in segs for c in (s[0], s[2])]
    ys = [c for s in segs for c in (s[1], s[3])]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    sx = (W - 2 * pad) / max(xmax - xmin, 1); sy = (H - 2 * pad) / max(ymax - ymin, 1)
    s = min(sx, sy)
    def tx(x): return pad + (x - xmin) * s
    def ty(y): return H - pad - (y - ymin) * s    # y 뒤집기(SVG)
    # 너무 많으면 다운샘플
    step = max(1, len(segs) // 1500)
    paths = {1: [], 2: [], 3: [], 4: []}
    for i, (x0, y0, x1, y1, typ) in enumerate(segs):
        if typ != 1 and i % step:    # 소마는 항상 유지
            continue
        paths.setdefault(typ, paths.get(typ, []))
        paths[typ].append(f"M{tx(x0):.1f} {ty(y0):.1f}L{tx(x1):.1f} {ty(y1):.1f}")
    return {"W": W, "H": H, "paths": {k: "".join(v) for k, v in paths.items() if v}}


def main():
    os.makedirs(OUT, exist_ok=True)
    pyr = os.path.join(MODELS, "pyramidal")
    pc_dir = os.path.join(pyr, sorted(os.listdir(pyr))[0])
    cells = [("PC (피라미드)", pc_dir)]
    for et in ("bAC", "cAC", "cNAC"):
        d = first_int(et)
        if d:
            cells.append((f"INT-{et}", d))

    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 7))
    fig.suptitle("뉴런 형태 (morphology) — 빨강=첨단 apical · 파랑=기저 basal · 회색=축삭 · 검정=소마",
                 fontsize=12, fontweight="bold")
    pc_segs = None
    for ax, (name, d) in zip(axes, cells):
        sw = swc_path(d)
        segs = parse_swc(sw)
        draw(ax, segs, f"{name}\n({os.path.basename(d).split('_')[3] if len(os.path.basename(d).split('_'))>3 else ''})")
        if name.startswith("PC"):
            pc_segs = segs
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "4_morphology.png")
    plt.savefig(out, dpi=120)
    print(f"[그림] {out}")

    svg = make_svg_json(pc_segs)
    with open(os.path.join(OUT, "4_morphology_PC.json"), "w") as f:
        json.dump(svg, f)
    print(f"[SVG json] PC 세그먼트 {len(pc_segs)}개 → 인라인 렌더용 저장")


if __name__ == "__main__":
    main()
