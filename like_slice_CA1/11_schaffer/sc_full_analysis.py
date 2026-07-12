# -*- coding: utf-8 -*-
"""
11_schaffer/sc_full_analysis.py  —  E2-c 결과 시각화 (그림 E2-c)

sc_full_slice.py 산출물(SC_spikes_all.csv + SC_positions.npz + 20kHz 막전위)로:
  (A) 집단 발화율 시간경과 (PC/INT, 100ms bin) — 9초 내내 지속되는지
  (B) 대표 세포 raster (PC/INT 구분)
  (C) 대표 PC 1개 20kHz 막전위 파형 (1초 창) — 실제 스파이크 확인
실행: python 11_schaffer/sc_full_analysis.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "sc_full_spikes"); FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# ── 스파이크 로드 ─────────────────────────────────────────────
gids, types, ts = [], [], []
with open(os.path.join(D, "SC_spikes_all.csv"), encoding="utf-8") as f:
    rd = csv.reader(f); next(rd, None)
    for g, ty, t in rd:
        gids.append(int(g)); types.append(ty); ts.append(float(t))
gids = np.array(gids); types = np.array(types); ts = np.array(ts)
is_pc = types == "PC"
TSTOP = 9000.0
pos = np.load(os.path.join(D, "SC_positions.npz"), allow_pickle=True)
N = len(pos["gid"]); n_pc = int((pos["type"] == "PC").sum()); n_int = N - n_pc

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.4, 1.0], hspace=0.35)

# (A) 집단 발화율 시간경과
axA = fig.add_subplot(gs[0])
bin_ms = 100.0
edges = np.arange(0, TSTOP + bin_ms, bin_ms)
for mask, n_cells, col, lab in [(is_pc, n_pc, "#2f6fb0", f"추체세포 PC (n={n_pc})"),
                                (~is_pc, n_int, "#C0392B", f"인터뉴런 INT (n={n_int})")]:
    h, _ = np.histogram(ts[mask], bins=edges)
    rate = h / n_cells / (bin_ms / 1000.0)   # Hz/cell
    axA.plot(edges[:-1] / 1000.0, rate, color=col, lw=1.6, label=lab)
axA.set_xlabel("시간 (초)"); axA.set_ylabel("집단 발화율 (Hz/세포)")
axA.set_title("(A) 집단 발화율 시간경과 — 9초 내내 지속 (온셋 후 안정)", fontsize=12, fontweight="bold")
axA.legend(fontsize=9, loc="upper right"); axA.grid(alpha=0.3); axA.set_xlim(0, 9)

# (B) raster (대표 세포 subsample)
axB = fig.add_subplot(gs[1])
rng = np.random.RandomState(0)
pc_ids = np.unique(gids[is_pc]); int_ids = np.unique(gids[~is_pc])
sel_pc = rng.choice(pc_ids, min(250, len(pc_ids)), replace=False)
sel_int = rng.choice(int_ids, min(50, len(int_ids)), replace=False) if len(int_ids) else np.array([], int)
row = {}
for i, g in enumerate(sorted(sel_int)):
    row[g] = i
for i, g in enumerate(sorted(sel_pc)):
    row[g] = len(sel_int) + i
sel = set(sel_pc.tolist()) | set(sel_int.tolist())
mask = np.array([g in sel for g in gids])
yy = np.array([row[g] for g in gids[mask]])
cc = np.where(is_pc[mask], "#2f6fb0", "#C0392B")
axB.scatter(ts[mask] / 1000.0, yy, s=1.2, c=cc, marker="|", linewidths=0.6)
axB.axhline(len(sel_int) - 0.5, color="k", lw=0.5, ls=":")
axB.set_xlabel("시간 (초)"); axB.set_ylabel("세포 (아래 INT 빨강 / 위 PC 파랑)")
axB.set_title(f"(B) 발화 raster — 대표 {len(sel_pc)} PC + {len(sel_int)} INT", fontsize=12, fontweight="bold")
axB.set_xlim(0, 9); axB.set_ylim(-1, len(sel) )

# (C) 20kHz 막전위 (대표 PC 1개, 1초 창)
axC = fig.add_subplot(gs[2])
try:
    vm = np.load(os.path.join(D, "_rank0_vm.npy"))
    vmg = np.load(os.path.join(D, "_rank0_vmgids.npy"))
    tvm = np.load(os.path.join(D, "SC_vm_time_ms.npy"))
    # 첫 PC 찾기
    pc_set = set(pos["gid"][pos["type"] == "PC"].tolist())
    idx = next((i for i, g in enumerate(vmg) if int(g) in pc_set), 0)
    w = (tvm >= 4000) & (tvm <= 5000)
    axC.plot(tvm[w] / 1000.0, vm[idx][w], color="#2E8B57", lw=0.7)
    axC.set_title(f"(C) 대표 PC(gid {int(vmg[idx])}) 막전위 20kHz — 4~5초 창 (실제 스파이크)",
                  fontsize=12, fontweight="bold")
    axC.set_xlabel("시간 (초)"); axC.set_ylabel("막전위 (mV)"); axC.grid(alpha=0.3)
except Exception as e:
    axC.text(0.5, 0.5, f"막전위 로드 실패: {e}", ha="center")

fig.suptitle("E2-c  전 슬라이스식 SC 경로 + 포아송 지속구동 (subset 2,000세포·dt 0.025·9초)\n"
             "SC→PC 10nS·SC→INT 3nS·fiber 150Hz(감소 보정) · PC 10.6Hz 지속 · INT 0.2Hz(정상상태 억제 미미)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, "E2c_full_firing.png")
fig.savefig(out, dpi=130); plt.close(fig)
print(f"[OK] {out}")
print(f"  스파이크 {len(ts):,} · PC {(is_pc).sum():,} · INT {(~is_pc).sum():,}")
