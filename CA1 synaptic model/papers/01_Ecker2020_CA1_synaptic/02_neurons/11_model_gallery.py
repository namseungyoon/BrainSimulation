"""
11_model_gallery.py — 23개 me-type 모델 로드·확인 갤러리 (보고용)
============================================================================
보고 자료 #1: shared/models 의 **23개 me-type 모델 전부**를 형태(2D)로 그려
한눈에 확인한다. 모델당 그림 1장(23장) + 전체 몽타주 1장.

NEURON 불필요 — 각 모델의 morphology/morphology.swc 를 직접 파싱(빠름).
색: soma=검정, axon=회색, basal=파랑, apical=빨강 (cell_types.py 와 동일 규약).

레지스트리: shared/models/models_registry.json (role·mtype·etype·morph·layer·dir).
실행: <ca1sim python> .../02_neurons/11_model_gallery.py
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
from common.plotstyle import set_korean_font          # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
REGISTRY = os.path.join(SHARED, "models", "models_registry.json")

# SWC type → (색, 두께, z순서, 투명도)  : 1 soma · 2 axon · 3 basal · 4 apical
# 축삭은 dense 해서 반투명·얇게(수상돌기 가독성 ↑)
TYPE_STYLE = {1: ("black", 1.8, 5, 1.0), 2: ("0.75", 0.35, 0, 0.45),
              3: ("tab:blue", 0.7, 2, 0.95), 4: ("tab:red", 0.7, 2, 0.95)}
ETYPE_FULL = {"cACpyr": "연속적응(피라미드)", "bAC": "버스트적응", "cAC": "연속적응",
              "cNAC": "연속비적응(FS)"}


def parse_swc(path):
    """morphology.swc → {type: [ [[px,py],[x,y]], ... ]}.  (cell_types.parse_swc 와 동일 규약)"""
    pts = {}
    by_type = {1: [], 2: [], 3: [], 4: []}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            c = ln.split()
            if len(c) < 7:
                continue
            i = int(c[0]); t = int(c[1]); x = float(c[2]); y = float(c[3]); par = int(c[6])
            pts[i] = (x, y, t)
            if par in pts:
                px, py, _ = pts[par]
                by_type.setdefault(t if t in by_type else 3, []).append([[px, py], [x, y]])
    return by_type


def draw(ax, by_type, title, fs=8):
    """2D 형태를 축에 그림 (axon→basal→apical→soma 순서로 z겹침)."""
    for t in (2, 3, 4, 1):
        segs = by_type.get(t)
        if not segs:
            continue
        col, lw, z, al = TYPE_STYLE[t]
        ax.add_collection(LineCollection(segs, colors=col, linewidths=lw, zorder=z, alpha=al))
    ax.set_title(title, fontsize=fs)
    ax.set_aspect("equal"); ax.axis("off"); ax.autoscale()


def load_models():
    with open(REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)
    return reg["models"], reg.get("summary", {})


def swc_path(model):
    return os.path.join(ROOT, model["dir"], "morphology", "morphology.swc")


def main():
    os.makedirs(OUT, exist_ok=True)
    models, summary = load_models()
    print(f"[갤러리] 모델 {len(models)}개 "
          f"(e-type {summary.get('by_etype', {})})", flush=True)

    # ── 모델당 개별 그림 (23장) ──
    drawn = []
    for k, m in enumerate(models, 1):
        sp = swc_path(m)
        if not os.path.isfile(sp):
            print(f"  [{k:2d}] {m['mtype']:9s} {m['etype']:7s}  [SWC 없음] {sp}", flush=True)
            continue
        by_type = parse_swc(sp)
        fig, ax = plt.subplots(figsize=(3.4, 4.2))
        title = (f"{m['mtype']} · {m['etype']}\n{m['layer']} · {m['morph']}")
        draw(ax, by_type, title, fs=9)
        out = os.path.join(OUT, f"11_morph_{k:02d}_{m['mtype']}_{m['etype']}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        drawn.append((m, by_type))
        print(f"  [{k:2d}] {m['mtype']:9s} {m['etype']:7s} {m['layer']:4s} → {os.path.basename(out)}",
              flush=True)

    # ── 전체 몽타주 (6×4) ──
    ncol = 6
    nrow = int(np.ceil(len(drawn) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 3.0))
    fig.suptitle(f"CA1 me-type 모델 갤러리 — {len(drawn)}개 (12 m-type × 4 e-type)",
                 fontsize=15, fontweight="bold")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (m, by_type) in zip(axes, drawn):
        draw(ax, by_type, f"{m['mtype']}·{m['etype']}\n{m['layer']}", fs=7.5)
    # 범례
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color="black", lw=2, label="체세포(soma)"),
           Line2D([0], [0], color="tab:red", lw=2, label="정점수상돌기(apical)"),
           Line2D([0], [0], color="tab:blue", lw=2, label="기저수상돌기(basal)"),
           Line2D([0], [0], color="0.7", lw=2, label="축삭(axon)")]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = os.path.join(OUT, "11_model_gallery.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[몽타주] {out}  ({len(drawn)}/{len(models)} 모델)", flush=True)


if __name__ == "__main__":
    main()
