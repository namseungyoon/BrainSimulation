# -*- coding: utf-8 -*-
"""baseline 전체망 발화 그림 (4패널) — mpi_baseline.npz 실측 데이터로 생성.

패널: (a) 래스터(E/I 색분리)  (b) PSTH(1ms bin 집단발화율)
      (c) 공간맵(슬라이스 평면 long×radial, 발화세포 + E3 자극전극)  (d) E/I 발화분율
사용: python viz_baseline_full.py   (scratch/mpi_baseline.npz 필요)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
NPZ = os.path.join(ROOT, "scratch", "mpi_baseline.npz")

# ── 데이터 로드 ────────────────────────────────────────────────────────────
d = np.load(NPZ, allow_pickle=True)
spk_t = d["spk_t"].astype(float); spk_id = d["spk_id"].astype(int)
fired = d["fired"].astype(bool); is_pc = d["is_pc"].astype(bool)
RADIUS = float(d["radius"]); STIM_T = float(d["stim_t"]); SETTLE = float(d["settle"]); N = int(d["n"])
rel = spk_t - STIM_T                                  # 자극기준 시간(ms)

wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
XYZ = wc["xyz"].astype(float); mt = wc["mtype"].astype(str)
cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
fr = cfg["frame_um"]; seed = np.array(fr["seed"], float)
long_d = np.array(fr["long_dir"], float); rad_d = np.array(fr["radial_dir"], float)
els = {e["id"]: e for e in cfg["electrodes"]["list"]}
E3 = np.array(els["E3"]["xyz_um"], float)

# 슬라이스 평면 투영 (µm): u=장축, r=방사축
u = (XYZ - seed) @ long_d; r = (XYZ - seed) @ rad_d
e3u = (E3 - seed) @ long_d; e3r = (E3 - seed) @ rad_d

# 통계
nPC, nI = int(is_pc.sum()), int((~is_pc).sum())
fPC, fI = int((fired & is_pc).sum()), int((fired & ~is_pc).sum())
nspk = spk_t.size
post = rel[rel >= -1]
C_E, C_I, C_BG = "#C44E52", "#4C72B0", "#d9d9d9"

# ── 그림 ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.32, wspace=0.22)

# (a) 래스터
ax = fig.add_subplot(gs[0, 0])
m_e = is_pc[spk_id]; m_i = ~m_e
ax.scatter(rel[m_i], spk_id[m_i], s=2, c=C_I, alpha=0.5, label=f"억제 I ({fI}/{nI})", rasterized=True)
ax.scatter(rel[m_e], spk_id[m_e], s=2, c=C_E, alpha=0.5, label=f"추체 E ({fPC}/{nPC})", rasterized=True)
ax.axvline(0, ls=":", color="k", lw=1.2)
ax.set_xlabel("자극기준 시간 (ms)"); ax.set_ylabel("세포 gid")
ax.set_title("(a) 래스터 — 단일 volley 후 집단발화"); ax.set_xlim(rel.min() - 1 if nspk else -5, rel.max() + 1 if nspk else 30)
ax.legend(loc="upper right", fontsize=8, markerscale=3)

# (b) PSTH (1ms bin)
ax = fig.add_subplot(gs[0, 1])
if post.size:
    bins = np.arange(np.floor(post.min()), np.ceil(post.max()) + 1, 1.0)
    ax.hist(rel[m_e], bins=bins, color=C_E, alpha=0.8, label="추체 E")
    ax.hist(rel[m_i], bins=bins, color=C_I, alpha=0.6, label="억제 I")
ax.axvline(0, ls=":", color="k", lw=1.2)
ax.set_xlabel("자극기준 시간 (ms)"); ax.set_ylabel("스파이크 수 / 1ms")
ax.set_title("(b) PSTH — 집단발화율(1ms bin)"); ax.legend(fontsize=8)

# (c) 공간맵 (슬라이스 평면)
ax = fig.add_subplot(gs[1, 0])
ax.scatter(u[~fired], r[~fired], s=4, c=C_BG, alpha=0.5, label="미발화", rasterized=True)
fe = fired & is_pc; fi = fired & ~is_pc
ax.scatter(u[fi], r[fi], s=10, c=C_I, alpha=0.8, label="발화 I", rasterized=True)
ax.scatter(u[fe], r[fe], s=10, c=C_E, alpha=0.8, label="발화 E", rasterized=True)
ax.scatter([e3u], [e3r], marker="*", s=320, c="gold", edgecolors="k", lw=1.2, zorder=5, label="E3 기록전극(SR)")
th = np.linspace(0, 2 * np.pi, 100)
ax.plot(e3u + RADIUS * np.cos(th), e3r + RADIUS * np.sin(th), "--", color="k", lw=1, alpha=0.7, label=f"SC 자극 locus r={RADIUS:.0f}µm")
ax.set_xlabel("장축 long (µm)"); ax.set_ylabel("방사축 radial (µm)")
ax.set_title("(c) 공간 발화맵 — SC 자극 locus (E3 SR위치 중심·E3=기록전극)"); ax.set_aspect("equal", "box")
ax.legend(loc="best", fontsize=7.5, markerscale=1.3)

# (d) E/I 발화분율
ax = fig.add_subplot(gs[1, 1])
cats = ["추체 E", "억제 I", "전체"]
frac = [100 * fPC / max(nPC, 1), 100 * fI / max(nI, 1), 100 * (fPC + fI) / N]
cnts = [f"{fPC}/{nPC}", f"{fI}/{nI}", f"{fPC+fI}/{N}"]
bars = ax.bar(cats, frac, color=[C_E, C_I, "#7f7f7f"], width=0.6)
for b, f, c in zip(bars, frac, cnts):
    ax.annotate(f"{f:.0f}%\n{c}", (b.get_x() + b.get_width() / 2, f), ha="center", va="bottom", fontsize=10)
ax.set_ylabel("발화 세포 비율 (%)"); ax.set_ylim(0, 100)
ax.set_title("(d) E/I 발화 분율"); ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"baseline 전체망(5,610세포·590만 시냅스) — 단일 volley(E3, R={RADIUS:.0f}µm) · "
             f"총 스파이크 {nspk:,} · 발화 {fPC+fI}/{N} ({100*(fPC+fI)/N:.0f}%)",
             fontsize=13, y=0.98)
out = os.path.join(FIG, "baseline_full.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"[그림] -> {out}")
print(f"  총 스파이크 {nspk:,} · 발화 E {fPC}/{nPC} ({100*fPC/max(nPC,1):.0f}%) · I {fI}/{nI} ({100*fI/max(nI,1):.0f}%)")
