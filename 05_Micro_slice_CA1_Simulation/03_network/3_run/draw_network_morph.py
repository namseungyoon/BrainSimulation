# 그림 가-4: 마이크로 슬라이스 네트워크 형태 — 5,610 소마 배치(층별 색) + 층 밴드 + fEPSP 전극 E1·E2·E3.
# 층관통 뷰: x=종축 u(long), y=반경 r(radial=깊이, SO→SP→SR→SLM). 전세포 완전형태 세포의 소마 위치.
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import font_manager
for fn in ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if any(fn in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fn; break
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/mnt/d/PROJECT/Project_2025_2026_HIPPO/03_Research_Development/02_BrainSimulator/05_Micro_slice_CA1_Simulation"
wc = np.load(os.path.join(ROOT, "data/derived/window_cells.npz"), allow_pickle=True)
xyz = wc["xyz"].astype(float); layer = wc["layer"].astype(str); mt = wc["mtype"].astype(str)
cfg = json.load(open(os.path.join(ROOT, "config/window_layout.json"), encoding="utf-8"))
seed = np.array(cfg["frame_um"]["seed"], float)
long_d = np.array(cfg["frame_um"]["long_dir"], float)
rad_d = np.array(cfg["frame_um"]["radial_dir"], float)

def loc(P):  # global xyz -> (u,r) local
    d = P - seed
    return d @ long_d, d @ rad_d
u, r = loc(xyz)

# 층 색·순서(SO 위 → SLM 아래)
LORDER = ["SO", "SP", "SR", "SLM"]
LCOL = {"SO": "#d1730a", "SP": "#c0392b", "SR": "#2b7cc4", "SLM": "#7b5ea7"}
LNAME = {"SO": "SO (oriens)", "SP": "SP (pyramidale)", "SR": "SR (radiatum)", "SLM": "SLM (lac.-mol.)"}
present = [L for L in LORDER if np.any(layer == L)] + [L for L in np.unique(layer) if L not in LORDER]

fig, ax = plt.subplots(figsize=(12.5, 8.4))
# 층 밴드(각 층 소마의 r 범위)
rmin, rmax = r.min() - 20, r.max() + 20
for L in present:
    m = layer == L
    if not m.any():
        continue
    lo, hi = r[m].min(), r[m].max()
    ax.add_patch(Rectangle((u.min() - 40, lo), (u.max() - u.min()) + 80, hi - lo,
                           facecolor=LCOL.get(L, "#888"), alpha=0.07, zorder=0))
    ax.text(u.max() + 46, (lo + hi) / 2, LNAME.get(L, L), color=LCOL.get(L, "#555"),
            fontsize=11, fontweight="bold", va="center", ha="left")
# 소마 산점(층별 색). 추체(SP_PC)는 진하게, 인터뉴런은 작게.
is_pc = mt == "SP_PC"
for L in present:
    m = layer == L
    ax.scatter(u[m & is_pc], r[m & is_pc], s=9, c=LCOL.get(L, "#888"), alpha=0.55,
               edgecolors="none", zorder=2)
    ax.scatter(u[m & ~is_pc], r[m & ~is_pc], s=16, marker="^", c=LCOL.get(L, "#888"),
               alpha=0.9, edgecolors="w", linewidths=0.3, zorder=3)
# 전극 E1·E2·E3
for e in cfg["electrodes"]["list"]:
    eu, er = loc(np.array(e["xyz_um"], float))
    ax.scatter(eu, er, s=340, marker="s", facecolor="#111", edgecolors="w", linewidths=1.6, zorder=6)
    ax.annotate(f'{e["id"]}\n({e["layer"]})', (eu, er), xytext=(eu - 70, er), ha="right", va="center",
                fontsize=11.5, fontweight="bold", color="#111",
                arrowprops=dict(arrowstyle="-", color="#111", lw=1.0), zorder=7)

ax.set_xlabel("종축 u (long, µm)", fontsize=12)
ax.set_ylabel("반경 r (radial = 깊이, µm)  ·  SO→SLM", fontsize=12)
ax.invert_yaxis()  # SO(작은 r) 위로
ax.set_aspect("equal"); ax.grid(True, alpha=0.2, zorder=0)
nPC = int(is_pc.sum()); nIN = int((~is_pc).sum())
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=8, label=f"추체세포 PC (n={nPC:,})"),
       Line2D([0], [0], marker="^", color="w", markerfacecolor="#888", markersize=9, markeredgecolor="w", label=f"인터뉴런 (n={nIN:,})"),
       Line2D([0], [0], marker="s", color="w", markerfacecolor="#111", markersize=11, label="fEPSP 전극")]
ax.legend(handles=leg, loc="lower left", frameon=True, fontsize=10, framealpha=0.9)
ax.set_title(f"그림 가-4. 마이크로 슬라이스 네트워크 형태 — 전세포 완전형태 {len(mt):,}세포 소마 배치 · 층 구조 · fEPSP 전극 E1·E2·E3\n"
             f"(창 종축 500 × 반경 800 × 두께 400 µm · 층관통 투영)", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
OUTDIR = os.path.join(ROOT, "04_experiments/Ex2b_connection_matrix/figures")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "network_morphology.png")
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("저장:", OUT, "| PC", nPC, "IN", nIN, "| 층", present)
