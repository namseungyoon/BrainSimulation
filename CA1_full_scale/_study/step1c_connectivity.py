"""STAGE 1 / step 1c — Bezaire CONNECTIVITY structure (paper Table 1 essence).

AUTHORITATIVE SOURCE = the raw ModelDB dataset the model actually runs:
  bezaire_modeldb/datasets/conndata_430.dat   (config full_scale_3dtopo.yaml: conndata_index=430)

Column semantics are taken verbatim from the ORIGINAL NEURON loader
(bezaire_modeldb/setupfiles/load_cell_conns.hoc, lines 78-80):
    col1 pre_type
    col2 post_type
    col3 wgt  = synapse weight  (max conductance, uS)
    col4 num  = CONVERGENCE     (# presynaptic cells of that type onto one postsynaptic cell)
    col5 syn  = synapses per connection

conndata_101.dat is a DIAGNOSTIC table (col4 = network-wide total, and Pyr->Pyr is zeroed);
we parse it too, purely to document why the earlier connectivity.json (built from 101)
showed Pyr->Pyr = 0.  The paper's recurrent collateral lives in conndata_430."""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/CA1_full_scale")
DS = ROOT / "bezaire_modeldb/datasets"

# HOC identifier -> short canonical label (order = Bezaire cell-type order)
HOC = {
    "axoaxoniccell": "Axo", "bistratifiedcell": "Bist", "cckcell": "CCK",
    "ivycell": "Ivy", "ngfcell": "NGF", "olmcell": "OLM",
    "pvbasketcell": "PV", "pyramidalcell": "Pyr", "scacell": "SCA",
}

def parse_conndata(path: Path):
    """Return list of (pre, post, wgt, num_col4, syn) using canonical labels."""
    rows = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # first line is "<numPre> <numPost>"
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 5:
            continue
        pre, post, wgt, num, syn = parts
        rows.append((HOC.get(pre, pre), HOC.get(post, post),
                     float(wgt), float(num), float(syn)))
    return rows

r430 = parse_conndata(DS / "conndata_430.dat")
r101 = parse_conndata(DS / "conndata_101.dat")

order = ["Pyr", "PV", "Axo", "Bist", "CCK", "Ivy", "OLM", "NGF", "SCA"]

def matrix(rows, col):
    """col: 'num'(convergence) matrix pre x post."""
    M = np.zeros((len(order), len(order)))
    for pre, post, wgt, num, syn in rows:
        if pre in order and post in order:
            M[order.index(pre), order.index(post)] = num
    return M

M430 = matrix(r430, "num")

print("=== conndata_430 (paper Table 1, MODEL ACTUALLY RUNS THIS) ===")
print("CONVERGENCE matrix  (rows=pre, cols=post) = # presynaptic cells onto one postsynaptic cell")
hdr = "pre\\post".ljust(7) + "".join(f"{q:>7s}" for q in order)
print(hdr)
for i, p in enumerate(order):
    print(p.ljust(7) + "".join(f"{int(M430[i,j]):>7d}" for j in range(len(order))))

nz = sum(1 for pre, post, w, n, s in r430 if n > 0 and pre in order and post in order)
print(f"\nnon-zero recurrent projections (conndata_430): {nz}")

# spotlight the recurrent collateral that the paper emphasizes
def get(rows, pre, post):
    for a, b, w, n, s in rows:
        if a == pre and b == post:
            return w, n, s
    return None
w, n, s = get(r430, "Pyr", "Pyr")
print(f"\n*** Pyr->Pyr (recurrent collateral) ***")
print(f"  conndata_430 : weight={w} uS, CONVERGENCE={int(n)} presyn PCs/cell, {int(s)} syn/conn  -> PRESENT")
w0, n0, s0 = get(r101, "Pyr", "Pyr")
print(f"  conndata_101 : weight={w0}, col4={n0}, syn={int(s0)}  -> ZERO (diagnostic table; source of earlier false alarm)")

# ---- figure: recurrent convergence heatmap (conndata_430) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.5, 6.5))
Mm = np.ma.masked_equal(M430, 0)
im = ax.imshow(Mm, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
ax.set_xlabel("postsynaptic"); ax.set_ylabel("presynaptic")
ax.set_title("Bezaire 2016 CA1 recurrent connectivity — convergence (conndata_430 = paper Table 1)")
for i in range(len(order)):
    for j in range(len(order)):
        if M430[i, j] > 0:
            ax.text(j, i, f"{int(M430[i,j])}", ha="center", va="center", color="w", fontsize=7)
# highlight Pyr->Pyr diagonal cell
pi = order.index("Pyr")
ax.add_patch(plt.Rectangle((pi-0.5, pi-0.5), 1, 1, fill=False, edgecolor="red", lw=2.2))
fig.colorbar(im, ax=ax, label="convergence (presynaptic cells / postsynaptic cell)")
fig.tight_layout()
out = ROOT / "_study/figures/connectivity_matrix.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved:", out)
