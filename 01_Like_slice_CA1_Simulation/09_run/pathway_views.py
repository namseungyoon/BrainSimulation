# -*- coding: utf-8 -*-
"""
09_run/pathway_views.py  —  경로(pathway) 추적 관점

(A) 9경로별 연결거리(전-후 소마간 거리) 분포 — 각 경로의 공간적 도달범위(Romani 거리의존 연결).
(B) 한 대표 억제세포(SP_PVBC)의 출력 발자국 — PV+->PC 표적 PC를 발화율 색으로 2D 투영.

실행: python 09_run/pathway_views.py
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
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
TSTOP = 1000.0


def main():
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); mtype = c["mtype"].astype(str); N = len(xyz)
    nspk = np.zeros(N, dtype=int)
    with open(os.path.join(SD, "FULL_spikes_all.csv"), encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            nspk[int(row[0])] += 1
    rate = nspk / (TSTOP/1000.0)

    q = np.load(PRUNED, allow_pickle=True)
    pre = q["pre"].astype(int); post = q["post"].astype(int); cls = q["cls"].astype(int)
    classes = list(q["classes"].astype(str))
    dist = np.linalg.norm(xyz[pre] - xyz[post], axis=1)   # 6.4M 연결 거리(벡터화)

    # ── (A) 경로별 연결거리 분포 ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    order = sorted(range(len(classes)), key=lambda k: np.median(dist[cls == k]))
    edges = np.linspace(0, np.percentile(dist, 99), 60)
    ctr = (edges[:-1]+edges[1:])/2
    cmap = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    for j, k in enumerate(order):
        d = dist[cls == k]
        h, _ = np.histogram(d, bins=edges, density=True)
        ax.plot(ctr, h + j*0.006, color=cmap[j], lw=1.6,
                label=f"{classes[k]}  (중앙값 {np.median(d):.0f}um, n={len(d):,})")
    ax.set_xlabel("전-후 세포 소마간 거리 (um)"); ax.set_ylabel("정규화 밀도 (경로별 오프셋)")
    ax.set_title("V6-7  관점: 9경로별 연결거리 분포 — 경로의 공간적 도달범위\n(거리의존 커넥텀; 아래일수록 국소, 위일수록 원거리 연결)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.set_xlim(0, edges[-1])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V6_7_pathway_distance.png"), dpi=125); plt.close(fig)
    med_by = {classes[k]: float(np.median(dist[cls == k])) for k in range(len(classes))}
    print("[V6-7] 경로별 중앙거리: " + ", ".join(f"{c}={v:.0f}um" for c, v in med_by.items()), flush=True)

    # ── (B) 한 대표 PV+ 바스켓세포의 PV+->PC 출력 발자국 ─────────────────
    pv_cls = classes.index("PV+->PC (I2)")
    pvbc = np.where(mtype == "SP_PVBC")[0]
    # 출력 연결이 많은(=발자국 큰) 대표 PVBC 선택
    out_counts = {int(g): int(((pre == g) & (cls == pv_cls)).sum()) for g in pvbc[:200]}
    src = max(out_counts, key=out_counts.get)
    sel = (pre == src) & (cls == pv_cls)
    tgt = post[sel]
    # 2D 투영: 분산 큰 두 축
    P = xyz - xyz.mean(0)
    u, s, vt = np.linalg.svd(P[::37], full_matrices=False)   # 부분표본으로 주축
    proj = P @ vt[:2].T
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(proj[:, 0], proj[:, 1], s=1, c="#dddddd", label="전체 세포")
    sc = ax.scatter(proj[tgt, 0], proj[tgt, 1], s=14, c=rate[tgt], cmap="inferno",
                    vmin=0, vmax=np.percentile(rate, 99), label="표적 PC")
    for tt in tgt:
        ax.plot([proj[src, 0], proj[tt, 0]], [proj[src, 1], proj[tt, 1]],
                color="#4C72B0", lw=0.3, alpha=0.25)
    ax.scatter([proj[src, 0]], [proj[src, 1]], s=260, marker="*", c="red",
               edgecolors="k", zorder=5, label=f"전세포 PV+바스켓(gid{src})")
    fig.colorbar(sc, ax=ax, label="표적 PC 발화율 (Hz)")
    ax.set_title(f"V6-8  관점: 한 PV+ 바스켓세포의 PV+->PC 출력 발자국\n"
                 f"gid{src} → 표적 PC {len(tgt)}개 (주변억제 공간범위), 색=표적 발화율",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("주축1 (um)"); ax.set_ylabel("주축2 (um)"); ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "V6_8_cell_footprint.png"), dpi=125); plt.close(fig)
    print(f"[V6-8] 대표 PV+바스켓 gid{src} → PC표적 {len(tgt)}개", flush=True)
    print(f"[완료] 경로 관점 2종 → {FIG}", flush=True)


if __name__ == "__main__":
    main()
