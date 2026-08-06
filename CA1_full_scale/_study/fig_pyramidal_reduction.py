"""STAGE 1 figure — pyramidal cell: original multicompartment -> 3-compartment reduction.
Schematic (Bezaire poolosyncell has no detailed 3D morphology, geometry is L/diam per section).
Saves PNG into the repo so it can be dragged into Notion 04 §1.2."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = "/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/CA1_full_scale/_study/figures/pyramidal_reduction.png"
import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 5.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
BLUE, GREEN, ORANGE, GREY = "#2c6fbb", "#3a9a56", "#d98a34", "#666"

# ---------- LEFT: original multicompartment ----------
ax.text(2.4, 5.65, "Original (Bezaire NEURON)", ha="center", fontsize=12, fontweight="bold")
ax.text(2.4, 5.25, "single morphology template  ->  cloned x 311,500", ha="center", fontsize=8.5, color=GREY)
# apical tree (up)
import numpy as np
np.random.seed(0)
def branch(x0, y0, ang, length, depth, w):
    if depth == 0: return
    x1 = x0 + length*np.cos(ang); y1 = y0 + length*np.sin(ang)
    ax.plot([x0, x1], [y0, y1], color=BLUE, lw=w)
    branch(x1, y1, ang+0.5, length*0.72, depth-1, max(w-0.4, 0.5))
    branch(x1, y1, ang-0.5, length*0.72, depth-1, max(w-0.4, 0.5))
branch(2.4, 3.05, np.pi/2, 0.85, 5, 2.4)          # apical (up)
# basal tree (down)
for a in (-np.pi/2-0.5, -np.pi/2, -np.pi/2+0.5):
    branch(2.4, 2.55, a, 0.5, 3, 1.6)
for spine in ([], ):
    pass
# recolor basal green by overplotting short lines
def branch_c(x0, y0, ang, length, depth, w, c):
    if depth == 0: return
    x1 = x0 + length*np.cos(ang); y1 = y0 + length*np.sin(ang)
    ax.plot([x0, x1], [y0, y1], color=c, lw=w)
    branch_c(x1, y1, ang+0.55, length*0.7, depth-1, max(w-0.4, 0.5), c)
    branch_c(x1, y1, ang-0.55, length*0.7, depth-1, max(w-0.4, 0.5), c)
branch_c(2.4, 2.5, -np.pi/2, 0.55, 3, 1.6, GREEN)
# soma (triangle, pyramidal)
ax.add_patch(Polygon([[2.1, 2.5], [2.7, 2.5], [2.4, 3.05]], closed=True, facecolor="#111", edgecolor="k"))
# axon (down, orange)
ax.plot([2.4, 2.4], [2.5, 1.25], color=ORANGE, lw=2.0)
ax.plot([2.4, 2.15], [1.25, 0.95], color=ORANGE, lw=1.4)
# labels
ax.text(4.15, 4.4, "apical", color=BLUE, fontsize=9)
ax.text(0.6, 2.55, "soma", color="#111", fontsize=9)
ax.text(0.5, 1.7, "basal", color=GREEN, fontsize=9)
ax.text(2.6, 1.0, "axon", color=ORANGE, fontsize=9)
ax.text(2.4, 0.35,
        "202 sections  ->  566 compartments\n(soma 13 | apical 370 | basal 168 | axon 15)",
        ha="center", fontsize=9.5, fontweight="bold")

# ---------- ARROW ----------
ax.add_patch(FancyArrowPatch((5.1, 3.0), (6.7, 3.0), arrowstyle="-|>", mutation_scale=26, lw=2.5, color="#b03030"))
ax.text(5.9, 3.35, "reduce", ha="center", fontsize=11, fontweight="bold", color="#b03030")
ax.text(5.9, 2.55, "aglif_dend\n(CMA-ES fit to\nf-I/Rin/tau/sag)", ha="center", fontsize=8, color=GREY)

# ---------- RIGHT: 3-compartment ----------
ax.text(9.2, 5.65, "Reduced (NEST-GPU, aglif_dend)", ha="center", fontsize=12, fontweight="bold")
ax.text(9.2, 5.25, "3 compartments  (all 338,740 cells)", ha="center", fontsize=8.5, color=GREY)
boxes = [("dist  (V_dist)", 4.05, BLUE), ("prox  (V_d)", 3.15, GREEN), ("soma  (V_m)", 2.25, "#111")]
for label, y, c in boxes:
    ax.add_patch(FancyBboxPatch((8.3, y), 1.8, 0.7, boxstyle="round,pad=0.03", fc="white", ec=c, lw=2.2))
    ax.text(9.2, y+0.35, label, ha="center", va="center", fontsize=9.5, color=c, fontweight="bold")
for y in (3.85, 2.95):
    ax.add_patch(FancyArrowPatch((9.2, y), (9.2, y-0.20), arrowstyle="<->", mutation_scale=12, color=GREY))
ax.text(10.35, 3.4, "g_c\ncoupling", fontsize=7.5, color=GREY)
ax.text(9.2, 1.75, "6 ion-channel types -> lumped\nadaptation (I_adap, I_dep)", ha="center", fontsize=8, color=GREY)
ax.text(9.2, 0.5, "566 -> 3 compartments/cell\n=> single GPU can hold 338,740 cells", ha="center", fontsize=9, fontweight="bold", color="#b03030")

fig.suptitle("Bezaire pyramidal cell: multicompartment (566) -> aglif 3-compartment reduction", fontsize=12.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)
