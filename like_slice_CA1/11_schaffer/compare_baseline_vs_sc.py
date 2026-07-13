# -*- coding: utf-8 -*-
"""
11_schaffer/compare_baseline_vs_sc.py — 전체 baseline(1초) vs E2-c SC 포아송(9초 중 1초) 비교

두 기존 데이터를 첫 1,000ms로 맞춰 비교 가능한 지표를 모두 뽑음:
  A) 유형별 평균 발화율 + 발화 세포% (막대)
  B) 집단 발화율 시간경과 (PC, 0~1초)
  C) raster (각 200세포 표본)
  D) PC 세포별 발화율 분포
⚠️ 공정 비교 아님: 세포수(17,647 vs 2,000)·구동방식(일반 Poisson vs SC 포아송)·시냅스(확률 vs 결정론) 다름. 정성 대조용.
실행: python 11_schaffer/compare_baseline_vs_sc.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
WIN = 1000.0  # 첫 1초

DATASETS = [
    ("전체 baseline\n(17,647세포·일반 Poisson)", "#8c8c8c",
     os.path.join(ROOT, "09_run", "spikes", "FULL_spikes_all.csv"),
     os.path.join(ROOT, "09_run", "spikes", "FULL_positions.npz")),
    ("E2-c SC 포아송\n(2,000세포·SC 입력, 9초 중 1초)", "#2f6fb0",
     os.path.join(HERE, "sc_full_spikes", "SC_spikes_all.csv"),
     os.path.join(HERE, "sc_full_spikes", "SC_positions.npz")),
]


def load(csv_path, npz_path):
    pos = np.load(npz_path, allow_pickle=True)
    types = pos["type"].astype(str); N = len(types)
    n_pc = int((types == "PC").sum()); n_int = N - n_pc
    g, ty, t = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            tt = float(row[2])
            if tt >= WIN:
                continue
            g.append(int(row[0])); ty.append(row[1]); t.append(tt)
    g = np.array(g); ty = np.array(ty); t = np.array(t)
    is_pc = ty == "PC"
    return dict(N=N, n_pc=n_pc, n_int=n_int, g=g, ty=ty, t=t, is_pc=is_pc)


def metrics(d):
    dur = WIN / 1000.0
    pc_sp = int(d["is_pc"].sum()); int_sp = len(d["t"]) - pc_sp
    pc_rate = pc_sp / max(1, d["n_pc"]) / dur
    int_rate = int_sp / max(1, d["n_int"]) / dur
    fired_pc = len(set(d["g"][d["is_pc"]].tolist())); fired_int = len(set(d["g"][~d["is_pc"]].tolist()))
    # per-PC rate
    pcg = d["g"][d["is_pc"]]
    cnt = {}
    for gi in pcg:
        cnt[gi] = cnt.get(gi, 0) + 1
    per_pc = np.array(list(cnt.values())) / dur if cnt else np.array([0.0])
    # ISI CV (PC, ≥3 spikes)
    tp = d["t"][d["is_pc"]]
    cvs = []
    bygid = {}
    for gi, tt in zip(pcg, tp):
        bygid.setdefault(gi, []).append(tt)
    for gi, ts in bygid.items():
        ts = np.sort(ts)
        if len(ts) >= 3:
            isi = np.diff(ts)
            if isi.mean() > 0:
                cvs.append(isi.std() / isi.mean())
    return dict(pc_rate=pc_rate, int_rate=int_rate,
                fired_pc_pct=100 * fired_pc / max(1, d["n_pc"]),
                fired_int_pct=100 * fired_int / max(1, d["n_int"]),
                total=len(d["t"]), per_pc=per_pc,
                isi_cv=float(np.median(cvs)) if cvs else float("nan"))


data = [(lab, col, load(c, n)) for lab, col, c, n in DATASETS]
mets = [metrics(d) for _, _, d in data]

# ── 콘솔 지표표 ──
print(f"{'지표':<22}", end="")
for lab, _, _ in data:
    print(f"{lab.replace(chr(10),' '):>42}", end="")
print()
rows = [("세포수(PC/INT)", lambda d, m: f"{d['n_pc']}/{d['n_int']}"),
        ("총 스파이크(1초)", lambda d, m: f"{m['total']:,}"),
        ("PC 발화율(Hz)", lambda d, m: f"{m['pc_rate']:.2f}"),
        ("INT 발화율(Hz)", lambda d, m: f"{m['int_rate']:.2f}"),
        ("PC 발화 세포%", lambda d, m: f"{m['fired_pc_pct']:.0f}%"),
        ("INT 발화 세포%", lambda d, m: f"{m['fired_int_pct']:.0f}%"),
        ("PC ISI CV(중앙)", lambda d, m: f"{m['isi_cv']:.2f}")]
for name, fn in rows:
    print(f"{name:<22}", end="")
    for (_, _, d), m in zip(data, mets):
        print(f"{fn(d, m):>42}", end="")
    print()

# ── 그림 ──
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

# (A) 발화율 + 발화% 막대
axA = fig.add_subplot(gs[0, 0])
x = np.arange(2); w = 0.35
pc_rates = [m["pc_rate"] for m in mets]; int_rates = [m["int_rate"] for m in mets]
axA.bar(x - w / 2, pc_rates, w, label="PC", color="#DD8452")
axA.bar(x + w / 2, int_rates, w, label="INT", color="#4C72B0")
for i, (pr, ir) in enumerate(zip(pc_rates, int_rates)):
    axA.text(i - w / 2, pr, f"{pr:.1f}", ha="center", va="bottom", fontsize=9)
    axA.text(i + w / 2, ir, f"{ir:.1f}", ha="center", va="bottom", fontsize=9)
axA.set_xticks(x); axA.set_xticklabels([lab for lab, _, _ in data], fontsize=9)
axA.set_ylabel("평균 발화율 (Hz)"); axA.set_title("(A) 유형별 평균 발화율 (첫 1초)", fontsize=12, fontweight="bold")
axA.legend(fontsize=9); axA.grid(alpha=0.3, axis="y")

# (B) 집단 PC 발화율 시간경과
axB = fig.add_subplot(gs[0, 1])
binm = 20.0; edges = np.arange(0, WIN + binm, binm)
for (lab, col, d), m in zip(data, mets):
    h, _ = np.histogram(d["t"][d["is_pc"]], bins=edges)
    axB.plot(edges[:-1], h / d["n_pc"] / (binm / 1000.0), color=col, lw=1.6, label=lab.replace(chr(10), " "))
axB.set_xlabel("시간 (ms)"); axB.set_ylabel("PC 집단 발화율 (Hz)")
axB.set_title("(B) PC 발화율 시간경과 (첫 1초, 20ms bin)", fontsize=12, fontweight="bold")
axB.legend(fontsize=8); axB.grid(alpha=0.3); axB.set_xlim(0, WIN)

# (C) raster (각 200세포)
axC = fig.add_subplot(gs[1, 0])
rng = np.random.RandomState(0); yoff = 0; yticks = []; ylabs = []
for (lab, col, d) in data:
    pcids = np.unique(d["g"][d["is_pc"]])
    sel = rng.choice(pcids, min(200, len(pcids)), replace=False) if len(pcids) else np.array([], int)
    row = {gid: yoff + i for i, gid in enumerate(sorted(sel))}
    mask = np.array([gi in row for gi in d["g"]])
    yy = np.array([row[gi] for gi in d["g"][mask]])
    axC.scatter(d["t"][mask], yy, s=1.0, c=col, marker="|", linewidths=0.5)
    yticks.append(yoff + len(sel) / 2); ylabs.append(lab.replace(chr(10), " ")); yoff += len(sel) + 20
axC.set_yticks(yticks); axC.set_yticklabels(ylabs, fontsize=8)
axC.set_xlabel("시간 (ms)"); axC.set_title("(C) PC raster 표본 (각 200세포)", fontsize=12, fontweight="bold")
axC.set_xlim(0, WIN)

# (D) PC 세포별 발화율 분포
axD = fig.add_subplot(gs[1, 1])
for (lab, col, d), m in zip(data, mets):
    axD.hist(m["per_pc"], bins=40, alpha=0.55, color=col, label=lab.replace(chr(10), " "), density=True)
axD.set_xlabel("PC 세포별 발화율 (Hz)"); axD.set_ylabel("확률밀도")
axD.set_title("(D) PC 발화율 분포", fontsize=12, fontweight="bold")
axD.legend(fontsize=8); axD.grid(alpha=0.3)

fig.suptitle("전체 baseline(1초) vs E2-c SC 포아송(9초 중 1초) — 비교\n"
             "주의: 공정 비교 아님 — 세포수(17,647 vs 2,000)·구동(일반 Poisson vs SC 포아송)·시냅스(확률 vs 결정론) 다름 · 정성 대조",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "E2c_compare_baseline_vs_sc.png")
fig.savefig(out, dpi=125); plt.close(fig)
print(f"\n[OK] {out}")
