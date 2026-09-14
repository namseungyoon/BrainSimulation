# -*- coding: utf-8 -*-
"""STDP 재현 그림 — scratch/stdp_decomp.json (Graupner&Brunel 2012 칼슘 분해 방식).
4패널: (A) DP 고전 헤비안(검증) · (B) D 억압전용 · (C) P 강화전용 · (D) (C_pre,C_post) 위상도.
matplotlib만. 출력 stdp_types.png (가-14B)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
for fam in ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    try: matplotlib.rcParams["font.family"] = fam; break
    except Exception: pass
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
d = json.load(open(os.path.join(ROOT, "scratch", "stdp_decomp.json"), encoding="utf-8"))
types = {t["name"]: t for t in d["types"]}
cst = d["const"]

fig = plt.figure(figsize=(13.6, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1.15],
                      hspace=0.42, wspace=0.34)

def curve_panel(ax, t, star=False):
    x = np.array([r["dt"] for r in t["res"]]); y = np.array([r["dchg"] for r in t["res"]])
    ax.axhspan(0, 1, facecolor="#e9f6ee", zorder=0); ax.axhspan(-1, 0, facecolor="#fdeceb", zorder=0)
    ax.axhline(0, color="#8a8a8a", lw=0.9); ax.axvline(0, color="#b0b0b0", lw=0.8, ls=":")
    col = "#6a3fb0" if star else "#2f6fd0"
    ax.plot(x, y, "-", color=col, lw=2.6 if star else 2.0, zorder=3)
    ax.fill_between(x, y, 0, where=y > 0, color="#57b06a", alpha=0.35, zorder=2)
    ax.fill_between(x, y, 0, where=y < 0, color="#d1544c", alpha=0.35, zorder=2)
    mx = max(0.09, np.abs(y).max() * 1.25); ax.set_ylim(-mx, mx); ax.set_xlim(-100, 100)
    vtag = "검증 파라미터" if t["verified"] else "예시 파라미터"
    ax.set_title(f"{t['name']}  {t['label'].split('(')[0].strip()}", fontsize=11.5,
                 fontweight=("bold" if star else "normal"), color=(col if star else "#333"))
    ax.text(0.5, 0.965, f"C_pre {t['cpre']} · C_post {t['cpost']} · D {t['D']}ms · {vtag}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555")
    ax.set_xlabel("Δt = t_post - t_pre (ms)", fontsize=9.5)
    ax.set_ylabel("시냅스 변화 Δρ", fontsize=9.5)
    if star:
        for sp in ax.spines.values(): sp.set_edgecolor("#6a3fb0"); sp.set_linewidth(1.8)
        ax.annotate("post→pre\nLTD", (-27, y.min()*0.72), ha="center", fontsize=8.5, color="#b23b33")
        ax.annotate("pre→post\nLTP", (18, y.max()*0.72), ha="center", fontsize=8.5, color="#2e7d43")

curve_panel(fig.add_subplot(gs[0, 0]), types["DP"], star=True)
curve_panel(fig.add_subplot(gs[1, 0]), types["D"])
curve_panel(fig.add_subplot(gs[1, 1]), types["P"])

# (D) 위상도 — (C_pre, C_post) 평면에서 유형 구역 (Graupner Fig.2A 대응)
axp = fig.add_subplot(gs[0:1, 1:3]) if False else fig.add_subplot(gs[0, 1:3])
ph = d["phase"]; grid = np.array(ph["grid"]); cpre_ax = np.array(ph["cpre_ax"]); cpost_ax = np.array(ph["cpost_ax"])
cmap = ListedColormap(["#f0f0f0", "#e8a39c", "#8fc79a", "#c9b6e0"])   # 무변화·LTD·LTP·biphasic
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
axp.pcolormesh(cpre_ax, cpost_ax, grid, cmap=cmap, norm=norm, shading="auto")
for nm, mk in [("DP", "*"), ("D", "o"), ("P", "s")]:
    t = types[nm]; axp.scatter([t["cpre"]], [t["cpost"]], marker=mk, s=(150 if nm=="DP" else 70),
                                c="k", edgecolors="w", linewidths=1.2, zorder=5)
    axp.annotate(nm, (t["cpre"], t["cpost"]), textcoords="offset points", xytext=(7, 5),
                 fontsize=10, fontweight="bold")
axp.set_xlabel("C_pre (전시냅스 칼슘)", fontsize=9.5); axp.set_ylabel("C_post (후시냅스 칼슘)", fontsize=9.5)
axp.set_title("STDP 유형 위상도 — 하나의 칼슘 규칙, 여러 STDP (Graupner Fig.2A 대응)", fontsize=11)
from matplotlib.patches import Patch
axp.legend(handles=[Patch(facecolor="#e8a39c", label="LTD 전용"), Patch(facecolor="#8fc79a", label="LTP 전용"),
                    Patch(facecolor="#c9b6e0", label="이상성(LTD+LTP)"), Patch(facecolor="#f0f0f0", label="무변화")],
           loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle("가-14B. STDP 재현 — Graupner & Brunel 2012 칼슘 분해 방식 (60쌍@1Hz · 노이즈 제외 결정론)",
             fontsize=13.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.005, f"상수(검증): tau_ca {cst['tau_ca']}ms · θ_d {cst['theta_d']} · θ_p {cst['theta_p']} "
                     f"· γ_d {cst['gamma_d']} · γ_p {cst['gamma_p']}",
         ha="center", fontsize=8.5, color="#555")
out = os.path.join(HERE, "stdp_types.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
