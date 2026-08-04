# -*- coding: utf-8 -*-
"""13_net_fepsp/net_fepsp_plot.py  —  전체-네트워크 fEPSP 결과 + 순방향 모델 대비 비교

_net_fepsp_<tag>.npz(전세포 실제 막전류 fEPSP)를 그림으로:
 (A) 24전극 유발 fEPSP 파형(음성 SR sink)  (B) paired-pulse 촉진(SC E1s)
 (C) 전극 공간맵  (D) 순방향 상한값(E4b-9, 정렬·동기) vs 실제 네트워크(이질·억제·지터)
실행: <ca1sim>/python.exe 13_net_fepsp/net_fepsp_plot.py [tag]
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
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
LFPFIG = os.path.join(ROOT, "12_lfp", "figures")
tag = sys.argv[1] if len(sys.argv) > 1 else "full"

D = np.load(os.path.join(FIG, f"_net_fepsp_{tag}.npz"), allow_pickle=True)
t = D["t"]; Ve = D["Ve"]; E = D["E"]; over = D["over"]; amp = D["amp"]
stim = D["stim"]; N = int(D["N"]); jmax = int(D["jmax"]); sc_class = str(D["sc_class"])
NELEC = Ve.shape[0]
base_med = np.median(np.abs(amp[over]))
vm = Ve[jmax]

# paired-pulse 진폭(전극 jmax)
pk = []
for ts in stim:
    m = (t >= ts) & (t <= ts + 40); seg = vm[m]; pk.append(seg[np.argmax(np.abs(seg))])
ppr = abs(pk[1]) / max(abs(pk[0]), 1e-12) if len(stim) > 1 else float("nan")

# 순방향 상한값(E4b-9) 비교
fwd = None
fp = os.path.join(LFPFIG, "_e4b_band_3x8.npz")
if os.path.exists(fp):
    Dg = np.load(fp, allow_pickle=True); ov = Dg["over"]; fwd = float(np.median(np.abs(Dg["amp"][ov])))

fig = plt.figure(figsize=(15, 8.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.3)

# (A) 24전극 파형
axA = fig.add_subplot(gs[0, :])
for j in np.where(over)[0]:
    axA.plot(t, Ve[j], color="0.7", lw=0.7, zorder=2)
axA.plot(t, vm, color="#c0392b", lw=1.8, zorder=5, label=f"최대 전극 #{jmax}")
for s in stim:
    axA.axvline(s, color="#2980b9", lw=0.8, ls=":")
axA.axhline(0, color="0.6", lw=0.5)
axA.set_xlabel("시간 (ms)"); axA.set_ylabel("전극 fEPSP (µV)")
axA.legend(fontsize=9)
axA.set_title(f"(A) 전체-네트워크 유발 fEPSP — 24전극(회색)+최대(#{jmax}) · 실제 {N:,}세포 막전류 · SC={sc_class}\n"
              f"음성=SR 시냅스 sink(subthreshold) · 자극 {list(stim)}ms(점선)", fontsize=11)

# (B) paired-pulse
axB = fig.add_subplot(gs[1, 0])
if len(stim) > 1:
    axB.bar([0, 1], [abs(pk[0]), abs(pk[1])], color=["#7f8c8d", "#2980b9"], edgecolor="0.3")
    axB.set_xticks([0, 1]); axB.set_xticklabels([f"PP-1\n({t[np.argmin(np.abs(t-stim[0]))]:.0f}ms)", f"PP-2\n({stim[1]:.0f}ms)"])
    axB.set_ylabel("유발 |fEPSP| (µV)")
    axB.set_title(f"(B) paired-pulse (전극#{jmax})\nPPR = {ppr:.2f} " + ("(촉진·SC 현실성 ✓)" if ppr > 1 else "(억압)"), fontsize=10.5)
    axB.grid(axis="y", alpha=0.3)

# (C) 공간맵
axC = fig.add_subplot(gs[1, 1])
amax = np.abs(amp).max()
sc = axC.scatter(E[:, 0], E[:, 1], c=amp, cmap="RdBu_r", vmin=-amax, vmax=amax, s=170,
                 edgecolors=["k" if o else "0.6" for o in over], linewidths=1.2)
fig.colorbar(sc, ax=axC, label="유발 fEPSP (µV)")
axC.set_aspect("equal"); axC.set_xlabel("면 가로 (µm)"); axC.set_ylabel("세로 (µm)")
axC.set_title(f"(C) 전극 공간맵 · 중앙 |{base_med:.2f}|µV", fontsize=10.5)

# (D) 순방향 상한값 vs 실제 네트워크
axD = fig.add_subplot(gs[1, 2])
labels = ["순방향\n(정렬·동기\n상한값)", "실제 네트워크\n(이질·억제\n·지터)"]
vals = [fwd if fwd else 0, base_med]
axD.bar([0, 1], vals, color=["#b9722e", "#1f6fb2"], edgecolor="0.3")
axD.set_xticks([0, 1]); axD.set_xticklabels(labels, fontsize=8.5)
axD.set_ylabel("유발 fEPSP 중앙 |µV|")
red = (fwd / base_med) if (fwd and base_med > 0) else float("nan")
axD.set_title(f"(D) 순방향 상한값 vs 실제\n순방향 {fwd:.0f}µV → 실제 {base_med:.2f}µV" + (f" (~{red:.0f}배↓)" if red == red else ""), fontsize=10)
axD.grid(axis="y", alpha=0.3)
if fwd:
    axD.set_yscale("log")

fig.suptitle(f"E4c — 현실 전체-네트워크 fEPSP (복제 아님·전세포 실제 막전류·촉진 SC·확률 방출)  "
             f"실제 {N:,}세포 · WSL MPI\n"
             f"이상화 순방향 상한값 대비 실제 네트워크는 이질적 방향·억제·지터로 크게 감소(정직한 현실)",
             fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, f"net_fepsp_{tag}.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"[요약] 실제 네트워크 유발 fEPSP 중앙 |{base_med:.3f}|µV · 최대 |{np.abs(amp).max():.3f}|µV · PPR {ppr:.2f}", flush=True)
if fwd:
    print(f"[비교] 순방향 상한값 {fwd:.1f}µV → 실제 {base_med:.3f}µV (~{fwd/max(base_med,1e-9):.0f}배 감소)", flush=True)
