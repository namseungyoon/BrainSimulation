# -*- coding: utf-8 -*-
"""가-1 뉴런 모델 그림 — 실제 SP_PC 추체세포 SWC 형태 위에 구획별 이온채널을 직접 주석.
표·수식은 이미지에 굽지 않음(보고서에 별도 표·수식 블록으로 삽입). matplotlib만 사용(NEURON 불필요)."""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

for fam in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = fam; break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MORPHDIR = os.path.join(ROOT, "data", "circuit", "circuit", "components", "morphologies")

# ── 대표 추체세포(SP_PC) morphology 선택 ──
wc = np.load(os.path.join(ROOT, "data", "derived", "window_cells.npz"), allow_pickle=True)
mt = wc["mtype"].astype(str); morph = wc["morphology"].astype(str)
swc = None; name = None
for gid in np.where(mt == "SP_PC")[0]:
    cand = os.path.join(MORPHDIR, morph[gid] + ".swc")
    if os.path.isfile(cand):
        swc, name = cand, morph[gid]; break
    g = glob.glob(os.path.join(MORPHDIR, morph[gid] + "*.swc"))
    if g:
        swc, name = g[0], morph[gid]; break
if swc is None:
    raise SystemExit("SP_PC SWC를 찾지 못함")
print("morphology:", name, "->", os.path.basename(swc))

# ── SWC 파싱 ──
rows = []
for ln in open(swc):
    ln = ln.strip()
    if not ln or ln[0] == "#":
        continue
    p = ln.split()
    rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[6])))
idm = {r[0]: r for r in rows}
XYZ = {r[0]: np.array([r[2], r[3], r[4]]) for r in rows}
soma_c = np.array([XYZ[r[0]] for r in rows if r[1] == 1]).mean(axis=0)

# ── 첨두 위로 세우는 2D 투영 ──
apic = np.array([XYZ[r[0]] - soma_c for r in rows if r[1] == 4])
up = apic.mean(axis=0); up = up / (np.linalg.norm(up) + 1e-9)
P = np.array([XYZ[r[0]] - soma_c for r in rows])
Pp = P - np.outer(P @ up, up)
horiz = np.linalg.svd(Pp, full_matrices=False)[2][0]
def proj(v):
    return np.array([v @ horiz, v @ up])

TYPE_COL = {3: "#6a3fb0", 4: "#9b59b6"}   # basal/apical (축삭은 stub로 대체하므로 미표시)
TYPE_LW = {3: 1.0, 4: 1.1}

fig, ax = plt.subplots(figsize=(8.4, 8.6))
segs = {3: [], 4: []}
for r in rows:
    nid, typ, _, _, _, par = r
    if par in idm and par != -1 and typ in segs:
        segs[typ].append([proj(XYZ[nid] - soma_c), proj(XYZ[par] - soma_c)])
for typ in (3, 4):
    if segs[typ]:
        ax.add_collection(LineCollection(segs[typ], colors=TYPE_COL[typ], linewidths=TYPE_LW[typ], zorder=3))

# 수상돌기·소마만으로 뷰 프레임 (전체 축삭 제외 → 실제 모델과 일치)
dpts = np.array([proj(XYZ[r[0]] - soma_c) for r in rows if r[1] in (1, 3, 4)])
xmin, xmax = dpts[:, 0].min(), dpts[:, 0].max(); ymin, ymax = dpts[:, 1].min(), dpts[:, 1].max()
xr = xmax - xmin; yr = ymax - ymin
sc = proj(np.zeros(3))
# 축삭 stub (AIS): 소마에서 기저쪽으로 짧게
ax.plot([sc[0], sc[0]], [sc[1], sc[1] - yr * 0.09], color="#2b6cb0", lw=5.5, solid_capstyle="round", zorder=2)
ax.scatter([sc[0]], [sc[1]], s=230, color="#8e44ad", edgecolor="#3f2378", lw=1.6, zorder=6)

ax.set_xlim(xmin - xr * 0.40, xmax + xr * 0.40); ax.set_ylim(ymin - yr * 0.14, ymax + yr * 0.06)
ax.set_aspect("equal"); ax.axis("off")

# ── 층 근사 밴드 (소마 v=0 = SP 중심) ──
bands = [("SLM", 0.70 * ymax, ymax + yr * 0.05, "#eaf1fb"),
         ("SR", 0.08 * yr, 0.70 * ymax, "#e8f8ef"),
         ("SP", -0.05 * yr, 0.08 * yr, "#fdf0e0"),
         ("SO", ymin - yr * 0.14, -0.05 * yr, "#f3ecfa")]
xl = ax.get_xlim()
for nm, y0, y1, col in bands:
    ax.axhspan(y0, y1, facecolor=col, zorder=0)
    ax.text(xl[0] + xr * 0.02, (y0 + y1) / 2, nm, fontsize=11, fontweight="bold", color="#8a8a8a", va="center")

# ── 세포 위 채널 주석 (콜아웃) ──
def callout(tx, ty, target, txt, fc):
    ax.annotate(txt, xy=target, xytext=(tx, ty), fontsize=9.2, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec="#666", lw=1.0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3,
                                connectionstyle="arc3,rad=0.15"), zorder=8)
apical_tgt = (dpts[dpts[:, 1].argmax()][0] * 0.5, ymax * 0.55)          # SR 중간 첨두
basal_tgt = (dpts[dpts[:, 1].argmin()][0] * 0.6, ymin * 0.6)            # SO 기저
callout(xmax + xr * 0.26, ymax * 0.55,
        apical_tgt, "수상돌기\nCa: cal·can·cat\nIh: hd (HCN)\nA형 K: kad·kap", "#e8f8ef")
callout(xmin - xr * 0.26, 0.0,
        (sc[0], sc[1] - yr * 0.05), "소마·축삭(AIS)\nNa: na3·nax\nKdr: kdr·kdrb\nKm: kmb", "#fdf0e0")
callout(xmin - xr * 0.24, ymin * 0.7,
        basal_tgt, "공통(전 구획)\n칼슘의존 K: kca·cagk\n칼슘축적: cacum\n누설: passive", "#f3ecfa")

leg = [Line2D([0], [0], color=TYPE_COL[4], lw=2.4, label="첨두 수상돌기"),
       Line2D([0], [0], color=TYPE_COL[3], lw=2.4, label="기저 수상돌기"),
       Line2D([0], [0], color="#2b6cb0", lw=3.2, label="축삭 stub (AIS)"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#8e44ad", markersize=11, label="soma")]
ax.legend(handles=leg, loc="lower right", fontsize=9, framealpha=0.95)
ax.set_title("가-1. 뉴런 모델 — 실제 SP_PC 추체세포 재구성 형태 + 구획별 이온채널",
             fontsize=12.5, fontweight="bold", pad=10)
fig.savefig(os.path.join(HERE, "neuron_model_schematic.png"), dpi=150, bbox_inches="tight")
print("saved:", os.path.join(HERE, "neuron_model_schematic.png"))
