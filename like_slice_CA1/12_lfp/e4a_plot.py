# -*- coding: utf-8 -*-
"""12_lfp/e4a_plot.py  —  E4a fEPSP 결과 그림 (_e4a_results.npz -> E4a_*.png)

E4a_fepsp.png (2x2):
  (A) SR 단일전극 fEPSP 파형(P1 단일볼리) + slope
  (B) 깊이x시간 heatmap: sink(음)/source(양) 극성
  (C) 깊이 프로파일 극값 vs 깊이: SR 음성 <-> SP/SO 양성 극성반전
  (D) paired-pulse(P2) + PPR
E4a_placement.png: 형태(깊이축 y vs x 투영) + SC 시냅스 + 전극

실행: <ca1sim>/python.exe 12_lfp/e4a_plot.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
D = np.load(os.path.join(FIG, "_e4a_results.npz"), allow_pickle=True)

t1 = D["t1"]; V_SR1 = D["V_SR1"] * 1e3          # mV -> uV
t2 = D["t2"]; V_SR2 = D["V_SR2"] * 1e3
V_prof1 = D["V_prof1"] * 1e3                     # (28, N_t) uV
depths = D["depths"]; prof_ext = D["prof_ext"] * 1e3
stim_t = float(D["stim_t"]); ipi = float(D["ipi"]); ncd = float(D["nc_delay"])
syn_depth = float(D["syn_depth"]); n_syn = int(D["n_syn"]); n_seg = int(D["n_seg"])
fSR_slope = float(D["fSR_slope"]) * 1e3; fSR_amp = float(D["fSR_amp"]) * 1e3
ppr_slope = float(D["ppr_slope"]); ppr_amp = float(D["ppr_amp"])
e1_slope = float(D["e1_slope"]) * 1e3; e2_slope = float(D["e2_slope"]) * 1e3
cons_ratio = float(D["cons_ratio"]); vpk = float(D["vpk"]); spiked = bool(D["spiked"])
g_nS = float(D["g_nS"]); sigma = float(D["sigma"])
soma_c = D["soma_c"]; syn_pos = D["syn_pos"]; elec_SR = D["elec_SR"]
elec_profile = D["elec_profile"]; seg_mid = D["seg_mid"]; depth_axis = D["depth_axis"]

t0 = stim_t + ncd

# =====================================================================
# 그림 1: E4a_fepsp.png (2x2)
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

# (A) SR fEPSP 파형 + slope
a = ax[0, 0]
m = (t1 >= stim_t - 3) & (t1 <= stim_t + 40)
a.plot(t1[m], V_SR1[m], color="#C0392B", lw=1.8, label="SR 전극 fEPSP")
a.axhline(0, color="gray", lw=0.6)
a.axvline(t0, color="gray", ls=":", lw=0.8)
ipk = np.argmin(V_SR1[(t1 >= t0) & (t1 < t0 + 30)])
tt = t1[(t1 >= t0) & (t1 < t0 + 30)]; vv = V_SR1[(t1 >= t0) & (t1 < t0 + 30)]
a.plot(tt[ipk], vv[ipk], "o", color="#7d1a0f", ms=6)
a.annotate(f"음성피크 {fSR_amp:.2f} uV", (tt[ipk], vv[ipk]), textcoords="offset points",
           xytext=(8, -4), fontsize=9, color="#7d1a0f")
# slope 가이드선(음성 하강 접선)
tg = np.array([t0, t0 + min(4.0, abs(0.8 * fSR_amp / (fSR_slope if fSR_slope != 0 else -1))) + 0.5])
a.plot(tg, fSR_slope * (tg - t0), "--", color="k", lw=1.0)
a.text(0.03, 0.06, f"slope = {fSR_slope:.3f} uV/ms\nsink -> 음성(Colbert&Levy 1992)", transform=a.transAxes,
       fontsize=9, va="bottom", bbox=dict(fc="white", ec="0.7", alpha=0.9))
a.set_xlabel("시간 (ms)"); a.set_ylabel("세포외 전위 (uV)")
a.set_title("(A) SR 단일전극 fEPSP  (P1 단일볼리)")
a.legend(loc="upper right", fontsize=8)

# (B) 깊이 x 시간 heatmap
b = ax[0, 1]
mm = (t1 >= stim_t - 2) & (t1 <= stim_t + 25)
Z = V_prof1[:, mm]                         # (28, T)
vmax = np.abs(Z).max()
im = b.imshow(Z, aspect="auto", origin="lower", cmap="RdBu_r",
              vmin=-vmax, vmax=vmax,
              extent=[t1[mm][0], t1[mm][-1], depths[0], depths[-1]])
b.axhline(0, color="k", lw=0.8, ls="--")            # 소마(SP)
b.axhline(syn_depth, color="green", lw=0.8, ls=":")  # 시냅스 평균깊이
b.text(t1[mm][-1], 0, " SP(소마)", fontsize=8, va="center")
b.text(t1[mm][-1], syn_depth, " SR(시냅스)", fontsize=8, va="center", color="green")
cb = fig.colorbar(im, ax=b); cb.set_label("V (uV)  파랑=sink(음)/빨강=source(양)", fontsize=8)
b.set_xlabel("시간 (ms)"); b.set_ylabel("깊이축 좌표 (um, 소마=0, +=SR/SLM)")
b.set_title("(B) 깊이 x 시간  sink(파랑)/source(빨강)")

# (C) 극값 vs 깊이 (source-sink-source 삼중극)
c = ax[1, 0]
revs = D["revs"] if "revs" in D.files else np.array([])
c.plot(prof_ext, depths, "o-", color="#2c3e50", lw=1.4, ms=4)
c.axvline(0, color="gray", lw=0.8)
c.axhline(0, color="k", ls="--", lw=0.8); c.text(prof_ext.max(), 3, "SP(소마)", fontsize=8, va="bottom", ha="right")
c.axhline(syn_depth, color="green", ls=":", lw=0.8); c.text(prof_ext.min(), syn_depth + 6, "SR(시냅스)", fontsize=8, color="green")
c.fill_betweenx(depths, 0, prof_ext, where=(prof_ext < 0), color="#3498db", alpha=0.25)
c.fill_betweenx(depths, 0, prof_ext, where=(prof_ext > 0), color="#e74c3c", alpha=0.25)
for rv in np.atleast_1d(revs):
    c.axhline(float(rv), color="orange", ls="-", lw=0.8, alpha=0.7)
    c.text(0.05, float(rv), f"반전 {float(rv):.0f}", fontsize=7, color="#b8730a", va="bottom")
c.set_xlabel("자극후 극값 전위 (uV)"); c.set_ylabel("깊이축 좌표 (um)")
c.set_title("(C) 깊이 극성: source-sink-source 삼중극(tripole)\n소마측 양(+) / SR sink 음(-) / 원위 약양(+), 반전 2곳")

# (D) paired-pulse + PPR
d = ax[1, 1]
md = (t2 >= stim_t - 3) & (t2 <= stim_t + ipi + 40)
d.plot(t2[md], V_SR2[md], color="#8e44ad", lw=1.8)
d.axhline(0, color="gray", lw=0.6)
for k, tp in enumerate([t0, t0 + ipi]):
    d.axvline(tp, color="gray", ls=":", lw=0.7)
    d.text(tp + 1, V_SR2[md].min() * 0.9, f"E{k+1}", fontsize=9)
d.text(0.03, 0.06,
       f"PPR(slope) = {ppr_slope:.2f}\nPPR(amp) = {ppr_amp:.2f}\n"
       f"-> {'depression' if ppr_slope < 1 else 'facilitation'} (Ecker E2 대용 시냅스)",
       transform=d.transAxes, fontsize=9, va="bottom",
       bbox=dict(fc="white", ec="0.7", alpha=0.9))
d.set_xlabel("시간 (ms)"); d.set_ylabel("세포외 전위 (uV)")
d.set_title(f"(D) paired-pulse (IPI={ipi:.0f}ms)  PPR")

fig.suptitle(
    f"E4a. 세포외 fEPSP 순방향 계산 (상세형태 대표 PC 1개, {n_seg}세그먼트, LSA sigma={sigma} S/m 무한매질)\n"
    f"SC {n_syn}시냅스 SR 동기볼리(g={g_nS}nS, 결정론) -> 역치하 Vm {vpk:.1f}mV | 전류보존 |sumI|/max|I|={cons_ratio:.1e} | "
    f"단일세포라 uV 규모(집단 fEPSP는 mV, 향후 앙상블)",
    fontsize=10.5, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out1 = os.path.join(FIG, "E4a_fepsp.png")
fig.savefig(out1, dpi=140, bbox_inches="tight"); plt.close(fig)
print("saved:", out1)

# =====================================================================
# 그림 2: E4a_placement.png (형태 y-vs-x 투영 + 시냅스 + 전극)
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(7.5, 9))
# 형태: 세그먼트 중점 산점 (x축=원좌표 x, y축=원좌표 y ~ 깊이)
ax2.scatter(seg_mid[:, 0], seg_mid[:, 1], s=2, color="0.7", label=f"형태 {n_seg}세그먼트")
ax2.scatter(syn_pos[:, 0], syn_pos[:, 1], s=28, color="#27ae60", marker="^",
            edgecolor="k", linewidths=0.3, zorder=4, label=f"SC 시냅스 x{n_syn} (SR)")
ax2.scatter([soma_c[0]], [soma_c[1]], s=120, color="#f39c12", marker="*",
            edgecolor="k", zorder=5, label="소마(SP)")
ax2.scatter([elec_SR[0]], [elec_SR[1]], s=90, color="#C0392B", marker="s",
            edgecolor="k", zorder=5, label="SR 단일전극")
ax2.plot(elec_profile[:, 0], elec_profile[:, 1], "-", color="#2980b9", lw=1.2, marker=".",
         ms=5, zorder=3, label=f"깊이 프로파일 전극 x{len(elec_profile)}")
ax2.set_xlabel("x (um)"); ax2.set_ylabel("y (um)  ~ 깊이축 (SO 아래 <-> SR/SLM 위)")
ax2.set_title("E4a. 세포 형태 + SC 시냅스(SR) + 전극 배치\n(대표 PC, world 좌표 xy 투영; 깊이축~y)")
ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax2.set_aspect("equal", adjustable="datalim")
fig2.tight_layout()
out2 = os.path.join(FIG, "E4a_placement.png")
fig2.savefig(out2, dpi=140, bbox_inches="tight"); plt.close(fig2)
print("saved:", out2)
