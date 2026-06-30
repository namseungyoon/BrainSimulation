"""
1_build_connectivity.py — 축소 CA1 마이크로서킷: 거리의존 연결 생성
============================================================================
Source: Ecker(2020) §3.5(연결→9클래스 시냅스); Bezaire & Soltesz(2013) 연결 통계.

세포를 3D 층(SP/SO/SR)에 배치하고 **거리의존 확률**로 연결을 생성한 뒤, 각 연결에
**9클래스 시냅스(Table 3)** 와 시냅스 클래스를 배정한다. NEURON 불필요(순수 기하·통계).
산출 connectivity.json 을 `2_run_and_analyze.py` 가 읽어 실제 시뮬레이션을 돌린다.

축소 구성(기본): PC 100 + PV형(cNAC) 10 + cAC 6 + bAC 6 = 122 세포.
  - 우리가 보유한 4개 대표 me-model 의 e-type 을 기능적 역할로 매핑(축소 가정):
    PV형(cNAC)=주변표적 바스켓, cAC=수상돌기표적(CCK형), bAC=수상돌기표적(SOM형).
  - m-type 정밀 배정은 미보유(다운로드명 e-type 기준) → 근사 매핑임을 명시.

실행: <ca1sim python> papers/01_Ecker2020_CA1_synaptic/04_network/1_build_connectivity.py
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

THIS = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(THIS)
ROOT = os.path.dirname(os.path.dirname(PAPER))
SHARED = os.path.join(ROOT, "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(PAPER, "03_synapses"))   # params_table3
from common.plotstyle import set_korean_font   # noqa: E402
import params_table3 as P3                      # noqa: E402

set_korean_font()
OUT = os.path.join(THIS, "figures")
SEED = 20260622

# ── 축소 구성: 세포 종류·개수·배치 층(µm) ───────────────────────────────────
# type: 코드명. role: 기능 역할. y_range: 소마 깊이 범위(SP=0 기준, +위 SO / -아래 SR).
CELLTYPES = {
    "PC":  dict(n=100, color="tab:red",    y=(-15, 15),  ei="E"),   # 피라미드(SP)
    "PV":  dict(n=10,  color="tab:blue",   y=(-30, 40),  ei="I"),   # cNAC 바스켓(주변표적)
    "cAC": dict(n=6,   color="tab:green",  y=(-80, 20),  ei="I"),   # CCK형(수상돌기표적)
    "bAC": dict(n=6,   color="tab:orange", y=(0, 100),   ei="I"),   # SOM형(수상돌기표적)
}
TISSUE_XZ = 500.0      # 가로 타일 크기(µm)

# ── (pre_type, post_type) → 9클래스(Table 3) 매핑 ──────────────────────────
CLASS_MAP = {
    ("PC", "PC"):  "PC->PC (E2)",
    ("PC", "PV"):  "PC->SOM- (E2)",     # PC→바스켓(PC->PVBC 류)
    ("PC", "cAC"): "PC->SOM- (E2)",     # PC→CCK형
    ("PC", "bAC"): "PC->SOM+ (E1)",     # PC→SOM형(촉진성)
    ("PV", "PC"):  "PV+->PC (I2)",      # 주변표적 억제
    ("cAC", "PC"): "CCK+->PC (I3)",     # 수상돌기 억제(CCK)
    ("bAC", "PC"): "SOM+->PC (I2)",     # 수상돌기 억제(SOM/OLM)
}

# ── 거리의존 연결: 정점확률 p0(거리0) × exp(-d²/2σ²) ───────────────────────
P0 = {
    ("PC", "PC"): 0.04,
    ("PC", "PV"): 0.20, ("PC", "cAC"): 0.15, ("PC", "bAC"): 0.15,
    ("PV", "PC"): 0.55, ("cAC", "PC"): 0.30, ("bAC", "PC"): 0.30,
}
SIGMA = 150.0          # 연결 공간상수(µm)


def build():
    rng = np.random.RandomState(SEED)
    # 세포 배치
    cells = []   # dict(id, type, pos)
    for tname, spec in CELLTYPES.items():
        for _ in range(spec["n"]):
            x, z = rng.uniform(0, TISSUE_XZ, 2)
            y = rng.uniform(*spec["y"])
            cells.append(dict(id=len(cells), type=tname, pos=[float(x), float(y), float(z)]))
    pos = np.array([c["pos"] for c in cells])
    types = [c["type"] for c in cells]
    N = len(cells)

    # 연결 생성
    edges = []   # dict(pre, post, cls)
    for i in range(N):
        ti = types[i]
        for j in range(N):
            if i == j:
                continue
            tj = types[j]
            key = (ti, tj)
            if key not in CLASS_MAP:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            p = P0[key] * np.exp(-d * d / (2 * SIGMA * SIGMA))
            if rng.rand() < p:
                edges.append(dict(pre=i, post=j, cls=CLASS_MAP[key]))
    return cells, edges


def save(cells, edges):
    os.makedirs(THIS, exist_ok=True)
    meta = dict(celltypes={k: {kk: vv for kk, vv in v.items() if kk != "color"}
                           for k, v in CELLTYPES.items()},
                tissue_xz=TISSUE_XZ, sigma=SIGMA, seed=SEED,
                class_map={f"{a}->{b}": c for (a, b), c in CLASS_MAP.items()})
    out = os.path.join(THIS, "connectivity.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(cells=cells, edges=edges, meta=meta), f)
    return out


def plot(cells, edges):
    os.makedirs(OUT, exist_ok=True)
    pos = np.array([c["pos"] for c in cells])
    types = [c["type"] for c in cells]
    tnames = list(CELLTYPES.keys())

    fig = plt.figure(figsize=(17, 5.2))
    fig.suptitle(f"축소 CA1 마이크로서킷 — 배치·연결 ({len(cells)}세포, {len(edges)}연결, 9클래스 시냅스)",
                 fontsize=13, fontweight="bold")

    # (A) 배치(측면도 X–Y) + 연결 일부
    axA = fig.add_subplot(1, 3, 1)
    for tn in tnames:
        idx = [k for k, t in enumerate(types) if t == tn]
        axA.scatter(pos[idx, 0], pos[idx, 1], s=18, color=CELLTYPES[tn]["color"],
                    label=f"{tn}({len(idx)})", edgecolor="k", linewidth=0.3, zorder=3)
    rng = np.random.RandomState(1)
    for e in [edges[k] for k in rng.choice(len(edges), min(250, len(edges)), replace=False)]:
        a, b = pos[e["pre"]], pos[e["post"]]
        ei = "I" if "->PC" in e["cls"] and not e["cls"].startswith("PC") else "E"
        axA.plot([a[0], b[0]], [a[1], b[1]], color=("tab:gray" if ei == "E" else "tab:purple"),
                 lw=0.2, alpha=0.3, zorder=1)
    axA.set_title("(A) 배치(측면 X–Y) + 연결 표본", fontsize=10)
    axA.set_xlabel("X (µm)"); axA.set_ylabel("깊이 Y (µm, SP=0)"); axA.legend(fontsize=7, loc="upper right")

    # (B) 종류×종류 연결수 행렬
    axB = fig.add_subplot(1, 3, 2)
    M = np.zeros((len(tnames), len(tnames)), int)
    ti = {t: k for k, t in enumerate(tnames)}
    for e in edges:
        M[ti[types[e["pre"]]], ti[types[e["post"]]]] += 1
    im = axB.imshow(M, cmap="viridis")
    axB.set_xticks(range(len(tnames))); axB.set_xticklabels(tnames)
    axB.set_yticks(range(len(tnames))); axB.set_yticklabels(tnames)
    for r in range(len(tnames)):
        for c in range(len(tnames)):
            if M[r, c]:
                axB.text(c, r, M[r, c], ha="center", va="center",
                         color="w" if M[r, c] < M.max() * 0.6 else "k", fontsize=9)
    axB.set_title("(B) 연결수 행렬 (전→후)", fontsize=10)
    axB.set_xlabel("후(post)"); axB.set_ylabel("전(pre)")
    fig.colorbar(im, ax=axB, fraction=0.046)

    # (C) PC 입력 차수 분포(흥분/억제)
    axC = fig.add_subplot(1, 3, 3)
    pc_idx = [k for k, t in enumerate(types) if t == "PC"]
    ein = np.zeros(len(cells)); iin = np.zeros(len(cells))
    for e in edges:
        if types[e["post"]] == "PC":
            if types[e["pre"]] == "PC":
                ein[e["post"]] += 1
            else:
                iin[e["post"]] += 1
    axC.hist([ein[pc_idx], iin[pc_idx]], bins=12, label=["흥분 입력(PC→PC)", "억제 입력(INT→PC)"],
             color=["tab:gray", "tab:purple"])
    axC.set_title("(C) PC당 입력 연결수 분포", fontsize=10)
    axC.set_xlabel("입력 연결수"); axC.set_ylabel("PC 수"); axC.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "1_connectivity.png")
    fig.savefig(out, dpi=120)
    print(f"[그림] {out}")


def main():
    # 클래스명 유효성 검증
    for key, cname in CLASS_MAP.items():
        assert cname in P3.CLASSES, f"미등록 클래스: {cname}"
    cells, edges = build()
    out = save(cells, edges)
    plot(cells, edges)

    # 요약
    types = [c["type"] for c in cells]
    print(f"[연결] {out}")
    print(f"  세포 {len(cells)}개 " + ", ".join(f"{t}={types.count(t)}" for t in CELLTYPES))
    print(f"  연결 {len(edges)}개")
    from collections import Counter
    cc = Counter(e["cls"] for e in edges)
    for cls, n in sorted(cc.items(), key=lambda x: -x[1]):
        print(f"    {cls:18s} {n}")
    npc = types.count("PC")
    ne = sum(1 for e in edges if types[e["pre"]] == "PC" and types[e["post"]] == "PC")
    ni = sum(1 for e in edges if types[e["post"]] == "PC" and types[e["pre"]] != "PC")
    print(f"  PC당 평균 흥분입력 {ne/npc:.1f} · 억제입력 {ni/npc:.1f}")


if __name__ == "__main__":
    main()
