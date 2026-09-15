# -*- coding: utf-8 -*-
"""HFS·TBS 단일 시냅스 결과 그림 — 빈도 기반 LTP가 정상 재현됨(θ위상 실패와 대비).
 (A) 칼슘 c(t) — 유도 중 누적하여 θ_p 초과 · (B) ρ(t) 궤적 상승 · (C) Δρ 막대
실행: python 03_network/3_run/make_hfs_tbs_fig.py  (기본 both)
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATD = os.path.join(ROOT, "04_experiments", "Ex_hfs_tbs", "data")
FIGD = os.path.join(ROOT, "04_experiments", "Ex_hfs_tbs", "figures"); os.makedirs(FIGD, exist_ok=True)
for fp in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if os.path.exists(fp): fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
C_HFS = "#0072B2"; C_TBS = "#009E73"; C_TP = "#c0392b"; C_TD = "#888"
TAG = sys.argv[1] if len(sys.argv) > 1 else "both"

meta = json.load(open(os.path.join(DATD, f"hfs_tbs_{TAG}.json"), encoding="utf-8"))
tr = np.load(os.path.join(DATD, f"hfs_tbs_{TAG}_traces.npz"))
theta_d, theta_p = meta["theta_d"], meta["theta_p"]
res = {r["proto"]: r for r in meta["results"]}
COL = {"hfs": C_HFS, "tbs": C_TBS}; LAB = {"hfs": "HFS 100Hz×100", "tbs": "TBS 4p×10@5Hz"}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})

# (A) 칼슘 c(t)
for p in res:
    t = tr[f"{p}_t"]; spk = tr[f"{p}_spk"]; t0 = spk.min()
    m = (t >= t0 - 30) & (t <= spk.max() + 200)
    ax[0].plot(t[m] - t0, tr[f"{p}_c"][m], color=COL[p], lw=1.3, label=f"{LAB[p]} · post AP {res[p]['npost']}")
ax[0].axhline(theta_p, color=C_TP, ls="--", lw=1.4, label=f"θ_p 강화 = {theta_p}")
ax[0].axhline(theta_d, color=C_TD, ls=":", lw=1.4, label=f"θ_d 억압 = {theta_d}")
ax[0].set_xlabel("유도 시작 기준 시간 (ms)"); ax[0].set_ylabel("시냅스 칼슘 c")
ax[0].set_title("(A) 칼슘 — 유도 중 누적하여 θ_p 초과", fontsize=10.5); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

# (B) rho(t)
for p in res:
    t = tr[f"{p}_t"]; spk = tr[f"{p}_spk"]; t0 = spk.min()
    ax[1].plot((t - t0) / 1000.0, tr[f"{p}_rho"], color=COL[p], lw=2, label=LAB[p])
ax[1].axhline(0.5, color="#999", ls="--", lw=1, label="ρ 초기 0.5")
ax[1].set_xlabel("유도 시작 기준 시간 (초)"); ax[1].set_ylabel("시냅스 효능 ρ")
ax[1].set_title("(B) ρ 궤적 — 유도로 강화(LTP)", fontsize=10.5); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

# (C) Δρ 막대
ps = list(res.keys()); dr = [res[p]["dr"] for p in ps]
ax[2].bar(range(len(ps)), dr, color=[COL[p] for p in ps], width=0.6, edgecolor="white")
for i, p in enumerate(ps):
    ax[2].text(i, dr[i] + 0.005, f"{dr[i]:+.3f}\n{res[p]['direction']}", ha="center", va="bottom", fontsize=9)
ax[2].axhline(0, color="#333", lw=1); ax[2].set_xticks(range(len(ps))); ax[2].set_xticklabels([LAB[p] for p in ps], fontsize=8.5)
ax[2].set_ylabel("Δρ (유도 전→후)"); ax[2].set_ylim(0, max(dr) * 1.35 if dr else 1)
ax[2].set_title("(C) Δρ — 빈도 기반 LTP 재현", fontsize=10.5); ax[2].grid(alpha=0.25, axis="y")

fig.suptitle(f"HFS·TBS(단일 시냅스) — 현 칼슘 모델은 빈도 기반 LTP 정상 재현  ·  gid {meta['gid']}  "
             f"(θ위상 의존 실패와 대비: 빈도는 되고 위상은 안 됨)", fontsize=12.5, fontweight="bold", y=1.02)
out = os.path.join(FIGD, f"hfs_tbs_{TAG}.png")
plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("저장:", out)
for p in ps: print(f"  {p}: Δρ {res[p]['dr']:+.3f} · {res[p]['direction']} · post AP {res[p]['npost']}")
