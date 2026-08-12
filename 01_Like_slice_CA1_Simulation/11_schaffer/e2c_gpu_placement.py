# -*- coding: utf-8 -*-
"""
11_schaffer/e2c_gpu_placement.py
--------------------------------
전슬라이스 결정론 GPU 1초 런(17,647세포·SC경로·A6000)의 "구조" 3-패널 도식.

  (A) 17,647세포 배치     : 분산 큰 2축(x=종축 · nd=정규화깊이) 투영 산점도, 층(SO/SP/SR/SLM)별 색
  (B) 내부 커넥텀 구조     : pruned_connectivity 무작위 소표본을 pre->post 위치 선분(faint)으로,
                            흥분(PC->)=주황 / 억제=파랑
  (C) SC 자극 구도         : 가상 SC fiber(CA3 대용) -> SR층 시냅스(PC 60/INT 40) 도식.
                            SR 깊이 밴드 강조 + 화살표/주석으로 "자극이 SR로 들어감"을 직관화.

좌표 주의: pre/post(node_id)와 SC gid 는 SC_positions.xyz(gid 0..N-1 순서)로 인덱싱한다.
           (slice_cells 의 node_id 는 비단조이므로 위치는 SC_positions.xyz 를 사용)

출력: like_slice_CA1/11_schaffer/figures/E2cGPU_placement.png  (dpi 130)
실행: Windows ca1sim python (Malgun Gothic 한글 폰트)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # like_slice_CA1
FIG = os.path.join(HERE, "figures")
POS = os.path.join(HERE, "sc_det_gpu", "fullscale_n4", "SC_positions.npz")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
CONN = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")

LAYER_ORDER = ["SO", "SP", "SR", "SLM"]
LAYER_COLOR = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}
LAYER_KOR = {"SO": "SO 지향층", "SP": "SP 추체세포층", "SR": "SR 방사층", "SLM": "SLM 상분자층"}

rng = np.random.default_rng(7)


def load():
    p = np.load(POS, allow_pickle=True)
    xyz = p["xyz"].astype(float)          # (N,3) gid-순서
    typ = p["type"].astype(str)           # PC/PV/bAC/cAC
    c = np.load(CELLS, allow_pickle=True)
    layer = c["layer"].astype(str)        # SO/SP/SR/SLM (소마 층)
    nd = c["nd"].astype(float)            # 정규화 깊이(층 축)
    sclass = c["sclass"].astype(str)      # EXC/INH
    conn = np.load(CONN, allow_pickle=True)
    return dict(xyz=xyz, typ=typ, layer=layer, nd=nd, sclass=sclass,
                pre=conn["pre"], post=conn["post"], cls=conn["cls"],
                classes=[str(x) for x in conn["classes"]])


def main():
    os.makedirs(FIG, exist_ok=True)
    d = load()
    xyz, layer, nd, sclass = d["xyz"], d["layer"], d["nd"], d["sclass"]
    N = len(xyz)
    xax = xyz[:, 0]                        # 종축(가장 분산 큰 축)
    # 투영 2축: 가로=x(종축, µm), 세로=nd(정규화 깊이=층 축, 표층->심층)
    n_exc = int((sclass == "EXC").sum())
    n_inh = int((sclass == "INH").sum())
    print(f"[data] 세포 {N:,} (EXC {n_exc:,} / INH {n_inh:,})  "
          f"층 {dict(zip(*np.unique(layer, return_counts=True)))}")

    # SR 깊이 밴드(패널 C 강조용): SR 소마 세포 nd 범위 + 여유
    sr_lo, sr_hi = 0.35, 0.62             # SP 하단~SLM 상단 사이 = SR 수상돌기 대역
    if (layer == "SR").any():
        sr_lo = min(sr_lo, float(nd[layer == "SR"].min()) - 0.02)
        sr_hi = max(sr_hi, float(nd[layer == "SR"].max()) + 0.02)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(20, 7.4))

    # ---------------- 패널 A : 세포 배치 ----------------
    for L in LAYER_ORDER:
        m = layer == L
        if not m.any():
            continue
        sz = 3 if L == "SP" else 26
        alpha = 0.28 if L == "SP" else 0.95
        axA.scatter(xax[m], nd[m], s=sz, c=LAYER_COLOR[L], alpha=alpha,
                    edgecolors="none", label=f"{LAYER_KOR[L]} ({m.sum():,})")
    axA.invert_yaxis()                    # 표층(nd 작음) 위, 심층 아래
    axA.set_xlabel("종축 x (µm)")
    axA.set_ylabel("정규화 깊이 nd  (0=표층 SO → 심층 SLM)")
    axA.set_title(f"(A) 세포 배치 — 전슬라이스 {N:,}개\n종축 x × 정규화 깊이 투영 · 층별 색",
                  fontsize=12)
    axA.legend(fontsize=8, loc="lower right", framealpha=0.9,
               markerscale=1.6)
    axA.grid(alpha=0.15)

    # ---------------- 패널 B : 내부 커넥텀 소표본 선분 ----------------
    pre, post, cls = d["pre"], d["post"], d["cls"]
    NSAMPLE = 3500
    idx = rng.choice(len(pre), NSAMPLE, replace=False)
    pr, po, cl = pre[idx], post[idx], cls[idx]
    exc_class = {4, 5, 6}                  # PC-> (E1/E2) = 흥분
    is_exc = np.isin(cl, list(exc_class))

    def segs(mask):
        p0 = np.column_stack([xax[pr[mask]], nd[pr[mask]]])
        p1 = np.column_stack([xax[po[mask]], nd[po[mask]]])
        return np.stack([p0, p1], axis=1)

    # 억제 먼저(뒤), 흥분 위에
    lc_inh = LineCollection(segs(~is_exc), colors="#3B6FB0",
                            linewidths=0.28, alpha=0.16)
    lc_exc = LineCollection(segs(is_exc), colors="#E8873A",
                            linewidths=0.28, alpha=0.16)
    axB.add_collection(lc_inh)
    axB.add_collection(lc_exc)
    # 소마 옅은 배경
    axB.scatter(xax, nd, s=1, c="#cccccc", alpha=0.12, edgecolors="none",
                zorder=0)
    axB.set_xlim(axA.get_xlim())
    axB.set_ylim(nd.max() + 0.03, nd.min() - 0.03)   # invert
    axB.set_xlabel("종축 x (µm)")
    axB.set_ylabel("정규화 깊이 nd")
    axB.set_title(f"(B) 내부 커넥텀 구조 — 무작위 {NSAMPLE:,} 연결 표본\n"
                  f"흥분 PC-> ({int(is_exc.sum()):,}) 주황 · "
                  f"억제 ({int((~is_exc).sum()):,}) 파랑", fontsize=12)
    axB.legend(handles=[
        Line2D([0], [0], color="#E8873A", lw=2, label="흥분 시냅스 (PC->)"),
        Line2D([0], [0], color="#3B6FB0", lw=2, label="억제 시냅스 (INH->)"),
    ], fontsize=9, loc="lower right", framealpha=0.9)
    axB.grid(alpha=0.15)

    # ---------------- 패널 C : SC 자극 구도 ----------------
    # 배경 소마(층별 옅게)
    for L in LAYER_ORDER:
        m = layer == L
        if not m.any():
            continue
        sz = 3 if L == "SP" else 22
        al = 0.14 if L == "SP" else 0.55
        axC.scatter(xax[m], nd[m], s=sz, c=LAYER_COLOR[L], alpha=al,
                    edgecolors="none")
    x0, x1 = axA.get_xlim()
    # SR 깊이 밴드 강조(SC 시냅스가 들어가는 대역)
    axC.axhspan(sr_lo, sr_hi, xmin=0, xmax=1, color="#55A868", alpha=0.18,
                zorder=0)
    axC.text(x1 - (x1 - x0) * 0.02, (sr_lo + sr_hi) / 2,
             "SR 방사층\n(SC 시냅스 표적 대역)", ha="right", va="center",
             fontsize=10, color="#2f6b3f", fontweight="bold")

    # 가상 SC fiber(CA3 대용): 위쪽(표층 바깥)에서 SR 밴드로 내려오는 화살표 다발
    y_fiber = nd.min() - 0.10             # 그림 상단(SLM 바깥)에서 진입
    xf = np.linspace(x0 + (x1 - x0) * 0.10, x1 - (x1 - x0) * 0.10, 7)
    for xi in xf:
        axC.add_patch(FancyArrowPatch(
            (xi, y_fiber), (xi, (sr_lo + sr_hi) / 2),
            arrowstyle="-|>", mutation_scale=14, lw=1.8,
            color="#B5179E", alpha=0.9, zorder=5))
    axC.scatter(xf, np.full_like(xf, y_fiber), s=70, marker="v",
                c="#B5179E", zorder=6)
    axC.text((x0 + x1) / 2, y_fiber - 0.015,
             "가상 SC fiber (CA3 대용) — Schaffer collateral 입력",
             ha="center", va="bottom", fontsize=11, color="#B5179E",
             fontweight="bold")

    # 주석: 시냅스 규칙
    axC.text(x0 + (x1 - x0) * 0.02, sr_hi + 0.06,
             "SC 경로: 모든 세포의 SR 수상돌기에 시냅스\n"
             "  · PC 60 시냅스 / 세포\n  · INT 40 시냅스 / 세포",
             ha="left", va="top", fontsize=9.5, color="#222",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff7e6",
                       ec="#E8873A", alpha=0.95), zorder=7)

    axC.set_xlim(x0, x1)
    axC.set_ylim(nd.max() + 0.05, y_fiber - 0.06)    # invert, 상단 fiber 여유
    axC.set_xlabel("종축 x (µm)")
    axC.set_ylabel("정규화 깊이 nd")
    axC.set_title("(C) SC 자극 구도 — CA3 대용 가상 fiber → SR층 시냅스\n"
                  "PC 60 · INT 40 시냅스/세포 (SR 수상돌기 표적)", fontsize=12)
    axC.legend(handles=[Patch(color=LAYER_COLOR[L], label=LAYER_KOR[L])
                        for L in LAYER_ORDER],
               fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle("전슬라이스 결정론 GPU 런 구조 도식 — 배치 · 내부 커넥텀 · SC 자극  "
                 f"(17,647세포 · SC 경로 · A6000)", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG, "E2cGPU_placement.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[OK] -> {out}")


if __name__ == "__main__":
    main()
