# -*- coding: utf-8 -*-
"""
01_tissue/3_atlas_prep/crop_atlas.py  —  Stage 3: atlas 창 크롭 + 국소 질의 준비 (V2-prep)

Romani atlas는 이미 후처리 완료(coordinates·orientation·brain_regions·[PH]*).
→ 재전처리 불필요. 여기선 확정 창(config/window_layout.json) 주변으로 **크롭**해 경량화하고,
   임의 3D 점의 **층(SO/SP/SR/SLM)·정규화깊이(nd 0=SO→1=SLM)**를 질의할 재료를 만든다.

크롭 대상:
  - brain_regions (층 라벨 1~4)
  - [PH]y (깊이 좌표)  ·  [PH]SO[0]=base  ·  [PH]SLM[1]=top   → nd=([PH]y-base)/(top-base)

산출: data/derived/atlas_crop.npz  (lib/atlas_query.py 가 읽음) + figures/1-3_atlas_crop.png
검증: 확정 전극(E1 SO·E2 SP·E3 SR)을 질의해 config 층 라벨과 일치하는지.

실행: python 01_tissue/3_atlas_prep/crop_atlas.py
"""
import os
import glob
import json
import logging

import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
DERIVED = os.path.join(DATA, "derived")
FIG = os.path.join(HERE, "figures")
CFG = os.path.join(ROOT, "config", "window_layout.json")
os.makedirs(DERIVED, exist_ok=True); os.makedirs(FIG, exist_ok=True)

MARGIN_UM = 200.0                       # 창 밖 여유(수상돌기·기여반경)
LAYERS = {1: "SO", 2: "SP", 3: "SR", 4: "SLM"}
LC = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}


def find(p):
    # glob.escape: 파일명의 [PH] 브래킷이 문자클래스로 해석되는 것 방지
    hits = sorted(glob.glob(os.path.join(DATA, "**", glob.escape(p)), recursive=True), key=len)
    if not hits:
        raise SystemExit(f"[에러] {p} 없음 (data/ 확인)")
    return hits[0]


def window_corners_xyz(cfg):
    """확정 창의 8모서리 물리좌표(+MARGIN)."""
    fr = cfg["frame_um"]; w = cfg["window_um"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])
    c = w["center_local"]
    hu = w["long"] / 2 + MARGIN_UM; hr = w["radial"] / 2 + MARGIN_UM; ht = w["thick"] / 2 + MARGIN_UM
    pts = []
    for su in (-1, 1):
        for sr in (-1, 1):
            for sw in (-1, 1):
                u = c["u"] + su * hu; r = c["r"] + sr * hr; ww = c["w"] + sw * ht
                pts.append(seed + u * L + r * R + ww * Tk)
    return np.array(pts)


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    print(f"[창] {cfg['name']}  {cfg['window_um']['long']}×{cfg['window_um']['radial']}×{cfg['window_um']['thick']}µm (+여유 {MARGIN_UM:.0f})")

    # atlas 격자 정보(brain_regions 기준)
    br_path = find("brain_regions.nrrd")
    regions, h = nrrd.read(br_path)
    origin = np.array(h.get("space origin", [0, 0, 0]), float)
    vs = float(np.linalg.norm(np.array(h["space directions"], float)[0]))
    dims = np.array(regions.shape)

    # 창 모서리 → 복셀 bbox
    corners = window_corners_xyz(cfg)
    vox = (corners - origin) / vs
    lo = np.clip(np.floor(vox.min(0)).astype(int), 0, dims - 1)
    hi = np.clip(np.ceil(vox.max(0)).astype(int) + 1, 1, dims)
    (i0, j0, k0), (i1, j1, k1) = lo, hi
    print(f"[크롭] 복셀 bbox [{i0}:{i1}, {j0}:{j1}, {k0}:{k1}]  = {i1-i0}×{j1-j0}×{k1-k0} "
          f"(전체 {tuple(dims)}의 {100*(i1-i0)*(j1-j0)*(k1-k0)/regions.size:.2f}%)")

    def crop3d(path, comp=None):
        d, _ = nrrd.read(path)
        if comp is not None:      # [PH]* 는 (2,X,Y,Z) → comp 슬라이스
            d = d[comp]
        return np.ascontiguousarray(d[i0:i1, j0:j1, k0:k1])

    reg_c = np.ascontiguousarray(regions[i0:i1, j0:j1, k0:k1])
    phy_c = crop3d(find("[PH]y.nrrd"))
    base_c = crop3d(find("[PH]SO.nrrd"), comp=0)          # SO 하단
    top_c = crop3d(find("[PH]SLM.nrrd"), comp=1)          # SLM 상단
    new_origin = origin + np.array([i0, j0, k0]) * vs

    np.savez_compressed(os.path.join(DERIVED, "atlas_crop.npz"),
                        regions=reg_c.astype(np.int8), phy=phy_c.astype(np.float32),
                        base=base_c.astype(np.float32), top=top_c.astype(np.float32),
                        origin=new_origin, vsize=vs,
                        layer_labels=np.array([f"{k}:{v}" for k, v in LAYERS.items()]))
    print(f"[저장] data/derived/atlas_crop.npz  (regions {reg_c.shape})")

    # --- 층별 복셀수 ---
    print("[크롭 층별 복셀]")
    for lab, nm in LAYERS.items():
        n = int((reg_c == lab).sum())
        if n:
            print(f"   {nm:<4} {n:>8,}")

    # --- 검증: 확정 전극 질의 ---
    print("\n[검증] 전극 질의 (층이 config와 일치해야):")
    total = top_c - base_c
    for e in cfg["electrodes"]["list"]:
        xyz = np.array(e["xyz_um"])
        vi = np.floor((xyz - new_origin) / vs).astype(int)
        if (vi >= 0).all() and (vi < np.array(reg_c.shape)).all():
            lab = int(reg_c[vi[0], vi[1], vi[2]])
            lyr = LAYERS.get(lab, "밖")
            t = total[vi[0], vi[1], vi[2]]
            nd = float((phy_c[vi[0], vi[1], vi[2]] - base_c[vi[0], vi[1], vi[2]]) / t) if t else float("nan")
            ok = "✅" if lyr == e["layer"] else "⚠️"
            print(f"   {e['id']} config={e['layer']:<4} 질의={lyr:<4} nd={nd:.2f} {ok}")
        else:
            print(f"   {e['id']} — 크롭 밖")

    fig_crop(reg_c, cfg, new_origin, vs)
    fig_local(cfg, reg_c, new_origin, vs)
    fig_context(regions, (i0, i1, j0, j1, k0, k1))
    print(f"\n[V2-prep] 그림 저장 -> {FIG} (crop · local · context)")


def fig_crop(reg, cfg, origin, vs):
    maxlab = int(reg.max())
    colors = ["#ffffff"] * (maxlab + 1)
    for lab, nm in LAYERS.items():
        if lab <= maxlab:
            colors[lab] = LC[nm]
    cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5, maxlab + 1.5, 1), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for a in range(3):
        counts = (reg > 0).sum(axis=tuple(i for i in range(3) if i != a))
        k = int(np.argmax(counts))
        axes[a].imshow(np.take(reg, k, axis=a).T, origin="lower", cmap=cmap, norm=norm,
                       interpolation="nearest")
        axes[a].set_title(f"축{a} 단면 {k}")
    axes[0].legend(handles=[Patch(facecolor=LC[v], label=v) for v in LAYERS.values()],
                   loc="upper right", title="층", fontsize=8)
    fig.suptitle(f"V2-prep  atlas 크롭 (창 {cfg['name']} +여유{MARGIN_UM:.0f}µm) — 층 라벨 확인", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "1-3_atlas_crop.png"), dpi=130)
    plt.close(fig)


def fig_context(regions, bbox):
    """전체 atlas 단면 3장(크롭 영역 검정박스) + 크롭 단면 3장 = 크롭 위치 이해용."""
    i0, i1, j0, j1, k0, k1 = bbox
    ctr = [(i0 + i1) // 2, (j0 + j1) // 2, (k0 + k1) // 2]
    crop = regions[i0:i1, j0:j1, k0:k1]
    cc = [d // 2 for d in crop.shape]
    maxlab = int(regions.max())
    colors = ["#ffffff"] * (maxlab + 1)
    for lab, nm in LAYERS.items():
        if lab <= maxlab:
            colors[lab] = LC[nm]
    cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5, maxlab + 1.5, 1), cmap.N)
    rects = {0: (j0, j1, k0, k1), 1: (i0, i1, k0, k1), 2: (i0, i1, j0, j1)}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for a in range(3):
        axes[0, a].imshow(np.take(regions, ctr[a], axis=a).T, origin="lower",
                          cmap=cmap, norm=norm, interpolation="nearest")
        xl, xh, yl, yh = rects[a]
        axes[0, a].add_patch(Rectangle((xl, yl), xh - xl, yh - yl, fill=False, ec="black", lw=2.2))
        axes[0, a].set_title(f"전체 atlas · 축{a} 단면 {ctr[a]} (검정=크롭)")
        axes[1, a].imshow(np.take(crop, cc[a], axis=a).T, origin="lower",
                          cmap=cmap, norm=norm, interpolation="nearest")
        axes[1, a].set_title(f"크롭 · 축{a} 단면 {cc[a]}")
    axes[0, 0].legend(handles=[Patch(facecolor=LC[v], label=v) for v in LAYERS.values()],
                      title="층", loc="upper right", fontsize=8)
    fig.suptitle("1-3  atlas 크롭 위치 — 위: 전체(검정박스=크롭영역) · 아래: 크롭", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "1-3_crop_context.png"), dpi=130)
    plt.close(fig)


def fig_local(cfg, reg, origin, vs):
    """atlas 층을 창의 국소 프레임(종축 u × 층관통 r)에 투영 → 창·전극과 직접 대응."""
    fr = cfg["frame_um"]; w = cfg["window_um"]; c = w["center_local"]
    seed = np.array(fr["seed"]); L = np.array(fr["long_dir"])
    R = np.array(fr["radial_dir"]); Tk = np.array(fr["thick_dir"])
    us = np.arange(c["u"] - w["long"] / 2 - 120, c["u"] + w["long"] / 2 + 120, 6.0)
    rs = np.arange(c["r"] - w["radial"] / 2 - 120, c["r"] + w["radial"] / 2 + 120, 6.0)
    U, Rg = np.meshgrid(us, rs)
    xyz = seed[None, None, :] + U[..., None] * L + Rg[..., None] * R + c["w"] * Tk
    vi = np.floor((xyz - origin) / vs).astype(int)
    dims = np.array(reg.shape)
    inb = ((vi >= 0) & (vi < dims)).all(-1)
    vic = np.clip(vi, 0, dims - 1)
    lab = reg[vic[..., 0], vic[..., 1], vic[..., 2]].astype(float)
    lab[(~inb) | (lab == 0)] = np.nan

    cmap = ListedColormap([LC[LAYERS[i]] for i in (1, 2, 3, 4)])
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pcolormesh(U, Rg, lab, cmap=cmap, vmin=0.5, vmax=4.5, shading="auto", alpha=0.85)
    ax.add_patch(Rectangle((c["u"] - w["long"] / 2, c["r"] - w["radial"] / 2),
                           w["long"], w["radial"], fill=False, ec="black", lw=2.4))
    stim = cfg["electrodes"]["stim_id"]
    for e in cfg["electrodes"]["list"]:
        isS = e["id"] == stim; col = "#e23b3b" if isS else "#111111"
        ax.plot(e["u"], e["r"], marker="P" if isS else "o", ms=17 if isS else 12,
                mfc=col, mec="white", mew=1.6, zorder=5)
        ax.annotate(f'{e["id"]}·{e["layer"]}', (e["u"], e["r"]), textcoords="offset points",
                    xytext=(13, 0), va="center", fontsize=10, fontweight="bold", color=col)
    ax.set_aspect("equal"); ax.set_xlabel("종축 proximodistal (µm)"); ax.set_ylabel("층관통 radial (µm, SP=0)")
    ax.set_title(f'1-3  atlas 층 (국소 프레임) — 창·전극 대응 「{cfg["name"]}」')
    ax.legend(handles=[Patch(facecolor=LC[v], label=v) for v in LAYERS.values()],
              title="층(atlas)", loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "1-3_atlas_local.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
