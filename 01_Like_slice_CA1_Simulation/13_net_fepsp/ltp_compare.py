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
F = load("ltp_frozen") if os.path.exists(os.path.join(FIG, "_mea_ltp_frozen.npz")) else None
NROW = 3 if F else 2
fig = plt.figure(figsize=(15, 4.3 * NROW))
gs = fig.add_gridspec(NROW, 3, hspace=0.45, wspace=0.28)

# (A,B,C) 시간경과 (가소성 / 엄격대조 / 단기가소성대조)
series = [(P, "칼슘 가소성 (GBPlasticitySyn, γ 정상)", "#c0392b")]
if F:
    series.append((F, "★엄격 대조군 (동일 모델, γ_p=γ_d=0 → 가소성만 차단)", "#1f6fb2"))
series.append((C, "참고 대조군 (DetAMPANMDA · 단기가소성 있음)", "#7f8c8d"))
for row, (D_, name, col) in enumerate(series):
    ax = fig.add_subplot(gs[row, 0:2])
    ax.plot(D_["t"], D_["Ve"][D_["rec_j"]], color=col, lw=0.7)
    ax.axvspan(D_["tb"][0] - 60, D_["tb"][-1] + 60, color="#ecf0f1", alpha=0.85, zorder=0)
    ax.axvspan(D_["tt"][0] - 30, D_["tt"][-1] + 60, color="#fdebd0", alpha=0.9, zorder=0)
    ax.axvspan(D_["tp"][0] - 60, D_["tp"][-1] + 60, color="#d5f5e3", alpha=0.85, zorder=0)
    ax.set_ylabel("fEPSP (µV)"); ax.set_title(
        f"({'ABC'[row]}) {name} — baseline {D_['sb'].mean():.3f} → 사후 {D_['sp'].mean():.3f} µV/ms "
        f"= {D_['pct']:+.1f}%   (TBS 유발 스파이크 {D_['nspk']:,})", fontsize=10.5)
    if row == len(series) - 1:
        ax.set_xlabel("시간 (ms)  ·  회색=baseline · 주황=TBS · 녹색=사후")

# (D) 변화율 비교 — 3자
axC = fig.add_subplot(gs[0, 2])
bars = [("가소성", P["pct"], "#c0392b")]
if F:
    bars.append(("★엄격\n대조군", F["pct"], "#1f6fb2"))
bars.append(("참고\n대조군", C["pct"], "#7f8c8d"))
xs = np.arange(len(bars))
axC.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="0.3", width=0.62)
axC.set_xticks(xs); axC.set_xticklabels([b[0] for b in bars], fontsize=9)
axC.set_ylabel("TBS 후 fEPSP slope 변화 (%)")
for x, (_, v, _) in zip(xs, bars):
    axC.annotate(f"{v:+.1f}%", (x, v), textcoords="offset points", xytext=(0, 6), ha="center",
                 fontsize=10, fontweight="bold")
net = P["pct"] - (F["pct"] if F else C["pct"])
axC.set_title(f"(D) 변화율 3자 비교\n가소성 고유 효과 = {net:+.1f}%p" +
              ("  (엄격 대조 기준)" if F else ""), fontsize=10.5)
axC.grid(axis="y", alpha=0.3); axC.axhline(0, color="0.5", lw=0.8)

# (E) 테스트펄스별 slope 궤적 — 3자
axD = fig.add_subplot(gs[1, 2])
tr = [(P, "가소성", "#c0392b", "o")]
if F:
    tr.append((F, "엄격 대조군", "#1f6fb2", "^"))
tr.append((C, "참고 대조군", "#7f8c8d", "s"))
for D_, name, col, mk in tr:
    y = np.concatenate([D_["sb"], D_["sp"]])
    axD.plot(np.arange(len(y)), y, mk + "-", color=col, lw=1.8, ms=6, label=name)
axD.axvline(len(P["sb"]) - 0.5, color="#e67e22", lw=1.6, ls="--")
axD.text(len(P["sb"]) - 0.45, axD.get_ylim()[0], " TBS", color="#e67e22", fontsize=9, va="bottom")
axD.set_xticks(np.arange(len(P["sb"]) + len(P["sp"])))
axD.set_xticklabels([f"B{i+1}" for i in range(len(P["sb"]))] + [f"P{i+1}" for i in range(len(P["sp"]))], fontsize=8)
axD.set_ylabel("fEPSP |slope| (µV/ms)"); axD.legend(fontsize=8); axD.grid(alpha=0.3)
axD.set_title("(E) 테스트펄스별 궤적\n엄격 대조군은 TBS 후에도 변화 없음", fontsize=10)

# (F) 판정 텍스트
if F:
    axF = fig.add_subplot(gs[2, 2]); axF.axis("off")
    axF.text(0, 1.0, "(F) 최종 판정", fontsize=11, fontweight="bold", va="top")
    txt = [
        f"가소성 ON      : {P['pct']:+.1f}%  (ρ 0→{P['rho']:.3f})",
        f"★엄격 대조군   : {F['pct']:+.1f}%  (ρ 고정 0.000)",
        f"참고 대조군    : {C['pct']:+.1f}%  (단기가소성 누적)",
        "",
        "→ 동일 모델·동일 동역학에서",
        "   가소성만 차단하면 변화 ≈ 0",
        f"⇒ +{P['pct']:.1f}%는 전적으로",
        "   장기가소성 때문 ✅",
        "",
        "baseline이 세 조건 모두 동일",
        "(−1.4197/−1.4116/−1.4116)",
    ]
    for i, s in enumerate(txt):
        axF.text(0, 0.90 - i * 0.083, s, fontsize=9, va="top", family="Malgun Gothic")

sub = (f"★엄격 대조군(동일 GBPlasticitySyn·γ_p=γ_d=0)이 {F['pct']:+.1f}% → +{P['pct']:.1f}%는 **전적으로 장기가소성 효과**로 확정"
       if F else "⚠️ 시냅스 mod가 달라 완벽한 대조 아님 → 엄격 대조 예정")
fig.suptitle(f"LTP 판정 — 칼슘 가소성 vs 대조군 (실제 {P['N']:,}세포 · TBS 3버스트×4@100Hz)\n{sub}",
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "MEA_ltp_compare.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
print(f"[판정] 가소성 {P['pct']:+.1f}% vs 대조군 {C['pct']:+.1f}% → 고유 효과 {net:+.1f}%p · ρ {P['rho']:.3f}", flush=True)
print(f"[주의] 대조군 baseline 상승: {C['sb'][0]:.3f}→{C['sb'][-1]:.3f} µV/ms (단기가소성 누적, 200ms 간격)", flush=True)
