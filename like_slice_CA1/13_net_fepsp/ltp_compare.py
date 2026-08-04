# -*- coding: utf-8 -*-
"""13_net_fepsp/ltp_compare.py  —  LTP 가소성 vs 대조군 비교 (핵심 판정)

_mea_ltp_plastic.npz(칼슘 가소성) vs _mea_ltp_control.npz(가소성 없음)를 나란히 비교해
TBS 후 fEPSP 증가가 **장기가소성 고유 효과**인지 판정한다.
⚠️ 정직: 두 런은 시냅스 mod가 달라(가소성=GBPlasticitySyn 단기가소성 없음 /
   대조군=DetAMPANMDA 단기가소성 있음) 완벽한 대조가 아니다 → 그림에 명시.
실행: <ca1sim>/py 13_net_fepsp/ltp_compare.py
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

def load(tag):
    D = np.load(os.path.join(FIG, f"_mea_{tag}.npz"), allow_pickle=True)
    g = lambda k, d=0.0: (D[k].item() if (k in D.files and D[k].shape == ()) else (D[k] if k in D.files else d))
    return dict(t=D["t"], Ve=D["Ve"], rec_j=int(g("rec_j")), tb=D["t_base"], tt=D["t_tbs"], tp=D["t_post"],
                sb=np.abs(D["slope_base"]), sp=np.abs(D["slope_post"]), pct=float(g("ltp_pct", np.nan)),
                rho=float(g("rho_mean")), nspk=int(g("nspk")), N=int(g("N")))

P = load("ltp_plastic"); C = load("ltp_control")
fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.28)

# (A,B) 시간경과 2종
for row, (D_, name, col) in enumerate([(P, "칼슘 가소성 (GBPlasticitySyn)", "#c0392b"),
                                       (C, "대조군: 가소성 없음 (DetAMPANMDA)", "#7f8c8d")]):
    ax = fig.add_subplot(gs[row, 0:2])
    ax.plot(D_["t"], D_["Ve"][D_["rec_j"]], color=col, lw=0.7)
    ax.axvspan(D_["tb"][0] - 60, D_["tb"][-1] + 60, color="#ecf0f1", alpha=0.85, zorder=0)
    ax.axvspan(D_["tt"][0] - 30, D_["tt"][-1] + 60, color="#fdebd0", alpha=0.9, zorder=0)
    ax.axvspan(D_["tp"][0] - 60, D_["tp"][-1] + 60, color="#d5f5e3", alpha=0.85, zorder=0)
    ax.set_ylabel("fEPSP (µV)"); ax.set_title(
        f"({'AB'[row]}) {name} — baseline {D_['sb'].mean():.3f} → 사후 {D_['sp'].mean():.3f} µV/ms "
        f"= {D_['pct']:+.1f}%   (TBS 유발 스파이크 {D_['nspk']:,})", fontsize=10.5)
    if row == 1:
        ax.set_xlabel("시간 (ms)  ·  회색=baseline · 주황=TBS · 녹색=사후")

# (C) 변화율 비교
axC = fig.add_subplot(gs[0, 2])
axC.bar([0, 1], [P["pct"], C["pct"]], color=["#c0392b", "#7f8c8d"], edgecolor="0.3", width=0.6)
axC.set_xticks([0, 1]); axC.set_xticklabels(["가소성", "대조군"], fontsize=10)
axC.set_ylabel("TBS 후 fEPSP slope 변화 (%)")
for x, v in zip([0, 1], [P["pct"], C["pct"]]):
    axC.annotate(f"{v:+.1f}%", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10, fontweight="bold")
net = P["pct"] - C["pct"]
axC.set_title(f"(C) 변화율 비교\n가소성 고유 효과 ≒ {net:+.1f}%p", fontsize=10.5)
axC.grid(axis="y", alpha=0.3); axC.axhline(0, color="0.5", lw=0.8)

# (D) 테스트펄스별 slope 궤적
axD = fig.add_subplot(gs[1, 2])
for D_, name, col, mk in [(P, "가소성", "#c0392b", "o"), (C, "대조군", "#7f8c8d", "s")]:
    y = np.concatenate([D_["sb"], D_["sp"]])
    x = np.arange(len(y))
    axD.plot(x, y, mk + "-", color=col, lw=1.8, ms=6, label=name)
axD.axvline(len(P["sb"]) - 0.5, color="#e67e22", lw=1.6, ls="--")
axD.text(len(P["sb"]) - 0.45, axD.get_ylim()[0], " TBS", color="#e67e22", fontsize=9, va="bottom")
axD.set_xticks(np.arange(len(P["sb"]) + len(P["sp"])))
axD.set_xticklabels([f"B{i+1}" for i in range(len(P["sb"]))] + [f"P{i+1}" for i in range(len(P["sp"]))], fontsize=8)
axD.set_ylabel("fEPSP |slope| (µV/ms)"); axD.legend(fontsize=8); axD.grid(alpha=0.3)
axD.set_title("(D) 테스트펄스별 궤적\n대조군은 baseline이 이미 상승(단기가소성 누적)", fontsize=10)

fig.suptitle(f"LTP 판정 — 칼슘 가소성 vs 대조군 (실제 {P['N']:,}세포 · TBS 3버스트×4@100Hz)\n"
             f"⚠️ 두 런은 시냅스 mod가 달라(가소성=단기가소성 없음 / 대조군=단기가소성 있음) 완벽한 대조 아님 → 엄격 대조는 ρ 고정판 예정",
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "MEA_ltp_compare.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"[판정] 가소성 {P['pct']:+.1f}% vs 대조군 {C['pct']:+.1f}% → 고유 효과 {net:+.1f}%p · ρ {P['rho']:.3f}", flush=True)
print(f"[주의] 대조군 baseline 상승: {C['sb'][0]:.3f}→{C['sb'][-1]:.3f} µV/ms (단기가소성 누적, 200ms 간격)", flush=True)
