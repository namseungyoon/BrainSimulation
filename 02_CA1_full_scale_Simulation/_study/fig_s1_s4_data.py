"""STAGE 1 — §1-§4 데이터 그래프 PNG 생성 (Notion 04 삽입용).
Renders four figures from source-verified numbers:
  fig_cell_counts.png       (§2 세포 유형별 수, log)         connectivity.json / Fig 1D
  fig_ephys_table3.png      (§3 전기생리 Rin/tau/rheobase)   paper Table 3
  fig_pyr_divergence.png    (§4 Pyr 발산 수렴도)             conndata_430
  fig_afferent_synapses.png (§4 구심 시냅스 수, log)          paper Table 1
English in-figure labels (WSL font-safe); Korean captions live in Notion."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("/mnt/d/Project_2025_2026_HIPPO/Workspace/03_BrainSimulator/02_CA1_full_scale_Simulation/_study/figures")
OUT.mkdir(parents=True, exist_ok=True)
TEAL, CORAL = "#1D9E75", "#D85A30"

# --- 1. cell counts (log horizontal bar) ---
types = ["Pyramidal", "Ivy", "PV+ Basket", "CCK+ Basket", "Neurogliaform",
         "Bistratified", "O-LM", "Axo-axonic", "SCA"]
counts = [311500, 8810, 5530, 3600, 3580, 2210, 1640, 1470, 400]
fig, ax = plt.subplots(figsize=(8, 4.4))
y = np.arange(len(types))[::-1]
ax.barh(y, counts, color=TEAL)
ax.set_yticks(y); ax.set_yticklabels(types, fontsize=9)
ax.set_xscale("log"); ax.set_xlim(200, 1e6)
ax.set_xlabel("Number of cells (log scale)")
ax.set_title("CA1 cell counts by type (total 338,740)")
for yi, c in zip(y, counts):
    ax.text(c * 1.15, yi, f"{c:,}", va="center", fontsize=8)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "fig_cell_counts.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# --- 2. electrophysiology Table 3 (Rin, tau, rheobase) ---
et = ["Pyr", "PV", "CCK", "SCA", "Axo", "Bis", "OLM", "Ivy", "NGF"]
rin = [62.2, 52.0, 211.0, 272.4, 52.0, 98.7, 343.8, 100.0, 100.0]
tau = [4.8, 6.9, 22.6, 24.4, 7.0, 14.7, 22.4, 21.1, 21.1]
rheo = [250, 300, 60, 40, 200, 350, 50, 160, 170]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (vals, title, unit) in zip(
        axs, [(rin, "Input resistance", "MOhm"), (tau, "Membrane tau", "ms"), (rheo, "Rheobase", "pA")]):
    ax.bar(et, vals, color=TEAL)
    ax.set_title(f"{title} ({unit})", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Single-cell electrophysiology (paper Table 3)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / "fig_ephys_table3.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# --- 3. Pyr divergence convergence (horizontal bar) ---
dt = ["O-LM", "PV+ Basket", "Bistratified", "Pyr (recurrent)", "Axo-axonic",
      "SCA", "Ivy", "CCK+ Basket", "Neurogliaform"]
dv = [2379, 424, 366, 197, 162, 105, 9, 0, 0]
fig, ax = plt.subplots(figsize=(8, 4.4))
y = np.arange(len(dt))[::-1]
ax.barh(y, dv, color=CORAL)
ax.set_yticks(y); ax.set_yticklabels(dt, fontsize=9)
ax.set_xlabel("Convergence (presynaptic Pyr per postsynaptic cell)")
ax.set_title("Pyramidal divergence - Pyr is CA1's only excitatory source (conndata_430)")
for yi, v in zip(y, dv):
    ax.text(v + 30, yi, str(v), va="center", fontsize=8)
ax.set_xlim(0, 2650); ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "fig_pyr_divergence.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# --- 4. afferent synapse counts (log grouped horizontal bar) ---
tg = ["Pyr", "PV+ Basket", "Ivy", "CCK+ Basket", "Axo-axonic", "Bistratified", "SC-A", "NGF", "O-LM"]
ca3 = np.nan_to_num(np.array([3.73e9, 6.69e7, 3.39e7, 1.44e7, 1.23e7, 2.56e6, 1.55e6, np.nan, np.nan]))
ec3 = np.nan_to_num(np.array([8.09e8, np.nan, np.nan, 4.02e6, 1.43e6, 1.91e6, 4.58e5, 3.75e6, np.nan]))
fig, ax = plt.subplots(figsize=(8.5, 4.8))
y = np.arange(len(tg))[::-1]
h = 0.38
ax.barh(y + h / 2, ca3, height=h, color=TEAL, label="CA3 (Schaffer)")
ax.barh(y - h / 2, ec3, height=h, color=CORAL, label="ECIII (perforant)")
ax.set_yticks(y); ax.set_yticklabels(tg, fontsize=9)
ax.set_xscale("log"); ax.set_xlim(1e5, 1e10)
ax.set_xlabel("Afferent synapses (log scale)")
ax.set_title("Afferent synapse counts (paper Table 1)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "fig_afferent_synapses.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("saved:", sorted(p.name for p in OUT.glob("fig_*.png")))
