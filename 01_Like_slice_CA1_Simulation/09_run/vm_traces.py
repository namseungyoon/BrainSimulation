# -*- coding: utf-8 -*-
"""
09_run/vm_traces.py  —  20kHz 소마 막전위 관점 시각화

전세포 20kHz Vm(_rank{r}_vm.npy + _rank{r}_vmgids.npy + FULL_vm_time_ms.npy)에서
m-type별 대표 세포 파형을 뽑아 (A) 12종 대표 막전위 겹쳐그리기, (B) 단일 PC 확대(20kHz 해상도).

대표 세포 = 각 m-type에서 발화수가 그 그룹 '중앙값'에 가장 가까운 세포.
실행: python 09_run/vm_traces.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures"); SD = os.path.join(HERE, "spikes")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
NHOST = 10
EI_COLOR = {"EXC": "#DD8452", "INH": "#4C72B0"}


def main():
    c = np.load(CELLS, allow_pickle=True)
    mtype = c["mtype"].astype(str); layer = c["layer"].astype(str)
    sclass = c["sclass"].astype(str); N = len(mtype)

    # 세포별 스파이크 수
    nspk = np.zeros(N, dtype=int)
    with open(os.path.join(SD, "FULL_spikes_all.csv"), encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            nspk[int(row[0])] += 1

    t = np.load(os.path.join(SD, "FULL_vm_time_ms.npy")).astype(float)
    # 랭크별 vmgids → gid가 어느 랭크·행에 있는지
    gid2loc = {}
    for r in range(NHOST):
        gids_r = np.load(os.path.join(SD, f"_rank{r}_vmgids.npy"))
        for row_i, g in enumerate(gids_r):
            gid2loc[int(g)] = (r, row_i)
    vm_mm = {r: np.load(os.path.join(SD, f"_rank{r}_vm.npy"), mmap_mode="r")
             for r in range(NHOST)}

    def trace(gid):
        r, row_i = gid2loc[int(gid)]
        return np.array(vm_mm[r][row_i], dtype=float)

    # m-type별 대표: 발화수 중앙값에 가장 가까운 세포
    uniq = sorted(set(mtype), key=lambda m: (sclass[mtype == m][0] != "EXC", m))
    reps = []
    for m in uniq:
        ids = np.where(mtype == m)[0]
        med = np.median(nspk[ids])
        g = ids[np.argmin(np.abs(nspk[ids] - med))]
        reps.append((m, int(g)))

    # ── (A) 12 대표 파형 겹쳐그리기 (0~500ms) ─────────────────────────────
    win = (t >= 0) & (t <= 500)
    fig = plt.figure(figsize=(16, 10))
    axA = fig.add_subplot(1, 2, 1)
    off = 0; step = 95
    yticks = []; ylabels = []
    for m, g in reps:
        v = trace(g)[win]
        ei = sclass[g]
        axA.plot(t[win], v + off, color=EI_COLOR[ei], lw=0.6)
        yticks.append(off - 65); ylabels.append(f"{m}\n({nspk[g]}회, {layer[g]})")
        off += step
    axA.set_yticks(yticks); axA.set_yticklabels(ylabels, fontsize=8)
    axA.set_xlabel("시간 (ms)"); axA.set_title("(A) m-type 12종 대표 세포 소마 막전위 (0-500ms, 20kHz)\n주황=흥분, 파랑=억제")
    axA.set_xlim(0, 500)
    # 스케일바
    axA.plot([505, 505], [off-step, off-step+50], "k-", lw=2)
    axA.text(510, off-step+25, "50mV", fontsize=8, va="center")

    # ── (B) 단일 PC 확대 (0~150ms) — 20kHz 해상도 강조 ────────────────────
    axB = fig.add_subplot(1, 2, 2)
    # SP_PC 중 중간 발화 대표
    pc_ids = np.where(mtype == "SP_PC")[0]
    med = np.median(nspk[pc_ids]); gpc = int(pc_ids[np.argmin(np.abs(nspk[pc_ids]-med))])
    wz = (t >= 0) & (t <= 150)
    axB.plot(t[wz], trace(gpc)[wz], color="#C0392B", lw=0.9)
    axB.set_xlabel("시간 (ms)"); axB.set_ylabel("막전위 (mV)")
    axB.set_title(f"(B) 대표 추체세포(SP_PC gid={gpc}) 소마 막전위 확대\n0-150ms · 20kHz(0.05ms) — 스파이크+역치하 EPSP/IPSP")
    axB.set_xlim(0, 150)
    axB.axhline(-70, color="gray", ls=":", lw=0.6); axB.text(150, -70, " 휴지 -70", fontsize=7, va="center", color="gray")

    fig.suptitle("V6-6  관점: 20kHz 소마 막전위 파형 — 세포유형별 동역학 (스파이크만이 아닌 연속 신호)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "V6_6_vm_traces.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[OK] {out}", flush=True)
    print("  대표세포: " + ", ".join(f"{m}(gid{g},{nspk[g]}회)" for m, g in reps), flush=True)


if __name__ == "__main__":
    main()
