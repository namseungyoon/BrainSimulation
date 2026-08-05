# -*- coding: utf-8 -*-
"""13_net_fepsp/stdp_plot.py  —  STDP 검증 그림 (mod↔Python 참조↔문헌 현상)

_stdp_verify.npz 로드 → (A) STDP 곡선(단일 vs doublet) (B) mod↔참조 일치도
(C) Δt 부호별 요약 · 문헌(Wittenberg & Wang 2006) 대조 판정.
실행: <ca1sim>/py 13_net_fepsp/stdp_plot.py
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
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
D = np.load(os.path.join(FIG, "_stdp_verify.npz"), allow_pickle=True)
S = D["single"]; B = D["doublet"]      # 열: [Δt, ρ_mod, ρ_py, w_mod, w_py]
npair = int(D["npair"]); freq = float(D["freq"]); rho0 = float(D["rho0"])
pset = str(D["pset"])

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.28)

# (A) STDP 곡선
axA = fig.add_subplot(gs[0, :2])
axA.axhline(1.0, color="0.5", lw=1, ls="--")
axA.axvline(0.0, color="0.5", lw=1, ls=":")
axA.plot(S[:, 0], S[:, 3], "o-", color="#1f6fb2", lw=2, ms=7, label=f"단일 post 1발 (n={npair}짝 @{freq:.0f}Hz)")
axA.plot(B[:, 0], B[:, 3], "s-", color="#c0392b", lw=2, ms=7, label="post doublet 2발(10ms)")
axA.fill_between(S[:, 0], 1.0, S[:, 3], color="#1f6fb2", alpha=0.12)
axA.fill_between(B[:, 0], 1.0, B[:, 3], color="#c0392b", alpha=0.12)
axA.set_xlabel("Δt = t_post − t_pre  (ms)   ·   왼쪽=post 먼저(역인과) · 오른쪽=pre 먼저(인과)")
axA.set_ylabel("시냅스 세기비 (자극후/전)")
axA.set_title(f"(A) STDP 곡선 — 우리 LTP 엔진(GBPlasticitySyn, {pset})\n"
              "세기비 >1 = LTP · <1 = LTD · ρ0 = %.2f" % rho0, fontsize=11)
axA.legend(fontsize=9); axA.grid(alpha=0.3)

# (B) mod ↔ Python 참조 일치도
axB = fig.add_subplot(gs[0, 2])
axB.plot(S[:, 2], S[:, 1], "o", color="#1f6fb2", ms=7, label="단일")
axB.plot(B[:, 2], B[:, 1], "s", color="#c0392b", ms=7, label="doublet")
lim = [min(S[:, 1].min(), B[:, 1].min(), S[:, 2].min()) - 0.02,
       max(S[:, 1].max(), B[:, 1].max(), S[:, 2].max()) + 0.02]
axB.plot(lim, lim, "k--", lw=1, alpha=0.6)
axB.set_xlim(lim); axB.set_ylim(lim)
dmax = max(np.abs(S[:, 1] - S[:, 2]).max(), np.abs(B[:, 1] - B[:, 2]).max())
axB.set_xlabel("Python 참조 ρ"); axB.set_ylabel("NEURON mod ρ")
axB.set_title(f"(B) 엔진 정확성 검증\n최대 |Δρ| = {dmax:.1e} " +
              ("✅ 일치" if dmax < 1e-3 else "⚠ 불일치"), fontsize=10.5)
axB.legend(fontsize=8); axB.grid(alpha=0.3)

# (C) Δt 부호별 요약
axC = fig.add_subplot(gs[1, 0])
lbl = ["Δt<0\n(역인과)", "Δt>0\n(인과)"]
sn = [S[S[:, 0] < 0, 3].mean(), S[S[:, 0] > 0, 3].mean()]
bn = [B[B[:, 0] < 0, 3].mean(), B[B[:, 0] > 0, 3].mean()]
x = np.arange(2); w = 0.36
axC.bar(x - w / 2, sn, w, color="#1f6fb2", edgecolor="0.3", label="단일")
axC.bar(x + w / 2, bn, w, color="#c0392b", edgecolor="0.3", label="doublet")
axC.axhline(1.0, color="0.5", ls="--", lw=1)
axC.set_xticks(x); axC.set_xticklabels(lbl, fontsize=9)
axC.set_ylabel("평균 세기비"); axC.legend(fontsize=8)
axC.set_title("(C) Δt 부호별 평균", fontsize=10.5); axC.grid(axis="y", alpha=0.3)

# (D) ρ 변화 곡선
axD = fig.add_subplot(gs[1, 1])
axD.axhline(rho0, color="0.5", ls="--", lw=1)
axD.plot(S[:, 0], S[:, 1], "o-", color="#1f6fb2", lw=1.8, ms=5, label="단일")
axD.plot(B[:, 0], B[:, 1], "s-", color="#c0392b", lw=1.8, ms=5, label="doublet")
axD.axvline(0.0, color="0.5", lw=1, ls=":")
axD.set_xlabel("Δt (ms)"); axD.set_ylabel("최종 효능 ρ")
axD.set_title(f"(D) 효능 ρ (시작 {rho0:.2f})\nρ>0.5=UP(LTP) · <0.5=DOWN(LTD)", fontsize=10.5)
axD.legend(fontsize=8); axD.grid(alpha=0.3)

# (E) 문헌 대조 판정
axE = fig.add_subplot(gs[1, 2]); axE.axis("off")
axE.text(0, 1.0, "(E) 문헌 대조 — Wittenberg & Wang 2006", fontsize=10.5, fontweight="bold", va="top")
s_pos, s_neg = sn[1], sn[0]
b_pos, b_neg = bn[1], bn[0]
lines = [
    "해마 CA3→CA1 슬라이스 실측 현상:",
    " · 단일 pre-post 짝 → 거의 LTD 전용",
    " · post 버스트(doublet) → 양방향 STDP",
    "",
    f"우리 결과(단일)  : Δt>0 {s_pos:.2f} · Δt<0 {s_neg:.2f}",
    f"우리 결과(doublet): Δt>0 {b_pos:.2f} · Δt<0 {b_neg:.2f}",
    "",
    ("판정: 단일=LTD 우세 ✅" if s_pos < 1.02 else "판정: 단일에서 LTP 발생 ⚠"),
    ("      doublet=LTP 전환 ✅" if b_pos > s_pos + 0.02 else "      doublet 전환 미약 ⚠"),
]
for i, s in enumerate(lines):
    axE.text(0, 0.88 - i * 0.098, s, fontsize=9, va="top", family="Malgun Gothic")

fig.suptitle(f"STDP 검증 — 스파이크 타이밍 의존성 · 우리 엔진 vs Python 참조 vs 문헌 현상\n"
             f"프로토콜: pre-post 짝 {npair}회 @ {freq:.0f}Hz · 파라미터 {pset}",
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "STDP_verify.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"[STDP] 단일 Δt>0 {s_pos:.3f} / Δt<0 {s_neg:.3f} · doublet Δt>0 {b_pos:.3f} / Δt<0 {b_neg:.3f}", flush=True)
print(f"[정확성] mod↔참조 최대 |Δρ| = {dmax:.2e}", flush=True)
