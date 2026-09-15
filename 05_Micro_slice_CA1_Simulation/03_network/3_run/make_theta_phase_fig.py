# -*- coding: utf-8 -*-
"""θ위상 단일 시냅스 결과 그림 — '왜 현 칼슘 모델로 재현 안 되는지'를 드러냄.
 (A) 칼슘 c(t) peak vs trough + θ_d/θ_p 임계선 → 위상 무관하게 둘 다 θ_p 초과
 (B) 위상-Δρ 스윕 곡선 → 평평(문헌의 peak↑/trough↓ 양방향 없음)
 (C) 소마 Vm peak vs trough → 게이팅 확인(peak 발화·trough 무발화)
실행: python 03_network/3_run/make_theta_phase_fig.py [tag]  (기본 sweep_ncyc1)
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATD = os.path.join(ROOT, "04_experiments", "Ex_theta_phase", "data")
FIGD = os.path.join(ROOT, "04_experiments", "Ex_theta_phase", "figures"); os.makedirs(FIGD, exist_ok=True)
for fp in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if os.path.exists(fp): fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
C_PEAK = "#0072B2"; C_TROUGH = "#D55E00"; C_TP = "#c0392b"; C_TD = "#888"

TAG = sys.argv[1] if len(sys.argv) > 1 else "sweep_ncyc1"
meta = json.load(open(os.path.join(DATD, f"theta_phase_{TAG}.json"), encoding="utf-8"))
tr = np.load(os.path.join(DATD, f"theta_phase_{TAG}_traces.npz"))
theta_d, theta_p = meta["theta_d"], meta["theta_p"]
res = {int(r["phase"]): r for r in meta["results"]}
phases = sorted(res.keys())


def win(p):  # 버스트 주변 창으로 자르기
    t = tr[f"p{p}_t"]; spk = tr[f"p{p}_spk"]
    lo, hi = spk.min() - 40, spk.max() + 180
    m = (t >= lo) & (t <= hi)
    return t[m] - spk.min(), m


fig = plt.figure(figsize=(16, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.1], wspace=0.28)

# (A) 칼슘 c(t) — peak vs trough
axA = fig.add_subplot(gs[0, 0])
for p, c, lab in [(0, C_PEAK, "peak (φ=0°)"), (180, C_TROUGH, "trough (φ=180°)")]:
    if p not in res: continue
    x, m = win(p); axA.plot(x, tr[f"p{p}_c"][m], color=c, lw=2.2, label=f"{lab} · post AP {res[p]['npost']}")
axA.axhline(theta_p, color=C_TP, ls="--", lw=1.4, label=f"θ_p 강화임계 = {theta_p}")
axA.axhline(theta_d, color=C_TD, ls=":", lw=1.4, label=f"θ_d 억압임계 = {theta_d}")
axA.axhspan(theta_d, theta_p, color="#f0d000", alpha=0.10)
axA.text(axA.get_xlim()[1], (theta_d + theta_p) / 2, " LTD 대역", va="center", fontsize=8, color="#a07a00")
axA.set_xlabel("버스트 시작 기준 시간 (ms)"); axA.set_ylabel("시냅스 칼슘 c")
axA.set_title("(A) 칼슘 트레이스 — 위상 무관 둘 다 θ_p 초과\n(C_pre 버스트 칼슘이 지배)", fontsize=10.5)
axA.legend(fontsize=8, loc="upper right"); axA.grid(alpha=0.25)

# (B) 위상-Δρ 스윕
axB = fig.add_subplot(gs[0, 1])
ph = np.array(phases); dr = np.array([res[p]["dr"] for p in phases])
axB.plot(ph, dr, "-o", color="#333", lw=2, ms=7, mfc="#333", zorder=3, label="시뮬 Δρ (현 모델)")
axB.axhline(0, color="#999", lw=1)
# 문헌 기대(모식): peak(0°)↑ LTP, trough(180°)↓ LTD
exp = 0.05 * np.cos(np.deg2rad(ph))
axB.plot(ph, exp, "--", color="#2ca02c", lw=1.6, alpha=0.8, label="문헌 기대(모식): peak↑/trough↓")
axB.axvspan(-20, 20, color=C_PEAK, alpha=0.08); axB.axvspan(160, 200, color=C_TROUGH, alpha=0.08)
axB.set_xticks(phases); axB.set_xlabel("θ 위상 φ (°)  ·  0=peak, 180=trough")
axB.set_ylabel("Δρ (버스트 전→후)")
axB.set_title("(B) 위상-Δρ — 모든 위상 평평(LTP)\n양방향 스위치 없음", fontsize=10.5)
axB.legend(fontsize=8, loc="lower left"); axB.grid(alpha=0.25)

# (C) Vm peak vs trough
axC = fig.add_subplot(gs[0, 2])
for p, c, lab in [(0, C_PEAK, "peak"), (180, C_TROUGH, "trough")]:
    if p not in res: continue
    x, m = win(p); axC.plot(x, tr[f"p{p}_v"][m], color=c, lw=1.3, label=f"{lab} · AP {res[p]['npost']}")
axC.axhline(-10, color="#aaa", ls=":", lw=1, label="AP 임계 근사")
axC.set_xlabel("버스트 시작 기준 시간 (ms)"); axC.set_ylabel("소마 전압 (mV)")
axC.set_title("(C) 소마 Vm — 게이팅 정상\npeak 발화 · trough 무발화", fontsize=10.5)
axC.legend(fontsize=8, loc="upper right"); axC.grid(alpha=0.25)

dpk = res.get(0, {}).get("dr"); dtr = res.get(180, {}).get("dr")
fig.suptitle(f"θ위상 타이밍(단일 시냅스) — 현 칼슘 모델의 재현 실패 원인  ·  peak Δρ={dpk:+.3f} / trough Δρ={dtr:+.3f} (동일)"
             f"  ·  gid {meta['gid']} · θ {meta['thetaHz']}Hz · 버스트 {meta['npulse']}p@{meta['burstHz']}Hz×{meta['ncyc']}",
             fontsize=12.5, fontweight="bold", y=1.02)
out = os.path.join(FIGD, f"theta_phase_{TAG}.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("저장:", out)
print(f"  peak Δρ={dpk:+.3f} · trough Δρ={dtr:+.3f} · 차이 {abs(dpk-dtr):.4f}")
