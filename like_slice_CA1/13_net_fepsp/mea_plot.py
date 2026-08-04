# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_plot.py  —  MEA 실험 결과 그림 (I-O 곡선 · PPF)

_mea_<tag>.npz(mea_experiment.py 출력)를 종류(kind)에 따라 그림:
  io  : fEPSP slope·amp vs 자극세기(활성 SC 섬유 수) — sigmoid I-O 곡선
  ppf : PPR(=slope2/slope1) vs ISI — 짝펄스 촉진(>1)
실행: <ca1sim>/py 13_net_fepsp/mea_plot.py <tag>
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
tag = sys.argv[1] if len(sys.argv) > 1 else "io"
D = np.load(os.path.join(FIG, f"_mea_{tag}.npz"), allow_pickle=True)
kind = str(D["kind"]); N = int(D["N"]); stim_elec = int(D["stim_elec"]); rec_j = int(D["rec_j"])
E = D["E"]; over = D["over"]; r_stim = float(D["r_stim"])


LAY_COL = {"SO": "#2980b9", "SP": "#c0392b", "SR": "#27ae60", "SLM": "#8e44ad"}
el_layer = D["el_layer"].astype(str) if "el_layer" in D.files else None
G = lambda k, d=None: (D[k].item() if (k in D.files and D[k].shape == ()) else (D[k] if k in D.files else d))
# 실험 메타 캡션(사용자 표준 보고 항목: 시냅스 수·자극층·전극당 기여 세포)
_p = []
if "n_syn" in D.files:
    _p.append(f"내부 시냅스 {int(G('n_syn')):,}")
if "n_sc" in D.files:
    _p.append(f"SC 시냅스 {int(G('n_sc')):,}(세포 {int(G('n_sccell', 0)):,}개)")
if "stim_layer" in D.files:
    _p.append(f"자극 {str(G('stim_layer'))}층")
if "neff" in D.files and float(G("neff", 0)) > 0:
    _p.append(f"전극당 유효세포 Neff {float(G('neff')):.0f}·신호90% {float(G('r90', 0)):.0f}µm")
cap = ("\n" + " · ".join(_p)) if _p else ""


def elec_inset(ax):
    if el_layer is not None:                               # 층별 색(전극이 어느 층 위인가)
        for Ln, col in LAY_COL.items():
            m = (el_layer == Ln) & over
            if m.any():
                ax.scatter(E[m, 0], E[m, 1], s=55, c=col, edgecolors="0.3", label=Ln, alpha=0.85)
    else:
        ax.scatter(E[over, 0], E[over, 1], s=40, c="0.8", edgecolors="0.5")
    ax.scatter(E[~over, 0], E[~over, 1], s=18, c="none", edgecolors="0.8")
    ax.scatter(E[stim_elec, 0], E[stim_elec, 1], s=190, marker="*", facecolor="none",
               edgecolors="k", linewidths=1.6, label="자극", zorder=6)
    ax.scatter(E[rec_j, 0], E[rec_j, 1], s=150, marker="s", facecolor="none",
               edgecolors="k", linewidths=1.6, label="기록", zorder=6)
    ax.set_aspect("equal"); ax.legend(fontsize=6.5, loc="upper right", ncol=2, framealpha=0.9)
    ax.set_title(f"전극 배치(층별) · 자극#{stim_elec} 기록#{rec_j}", fontsize=9)
    ax.set_xlabel("면 가로 µm", fontsize=8); ax.tick_params(labelsize=7)


if kind == "io":
    nact = D["nact"]; slope = np.abs(D["slope"]); amp = np.abs(D["amp"]); nspk = D["nspk"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    ax[0].plot(nact, slope, "o-", color="#c0392b", lw=2)
    ax[0].set_xlabel("자극세기 (활성 SC 섬유 수)"); ax[0].set_ylabel("fEPSP |slope| (µV/ms)")
    ax[0].set_title(f"(A) Input-Output 곡선\n기록전극#{rec_j} · 실제 {N:,}세포"); ax[0].grid(alpha=0.3)
    ax2 = ax[0].twinx(); ax2.plot(nact, nspk, "s--", color="#7f8c8d", ms=4, alpha=0.6)
    ax2.set_ylabel("유발 스파이크 수", color="#7f8c8d"); ax2.tick_params(axis="y", labelcolor="#7f8c8d")
    ax[1].plot(nact, amp, "o-", color="#1f6fb2", lw=2)
    ax[1].set_xlabel("자극세기 (활성 SC 섬유 수)"); ax[1].set_ylabel("fEPSP |진폭| (µV)")
    ax[1].set_title("(B) fEPSP 진폭 vs 세기"); ax[1].grid(alpha=0.3)
    elec_inset(ax[2])
    fig.suptitle(f"MEA I-O 실험 — 자극전극 국소 SC({r_stim:.0f}µm 층대) 세기 스윕 → fEPSP slope (실제 네트워크){cap}", fontsize=12, y=1.02)
    print(f"[I-O] slope 범위 {slope.min():.3f}~{slope.max():.3f} µV/ms · 스파이크 {int(nspk.min())}~{int(nspk.max())}", flush=True)

elif kind == "ppf":
    isi = D["isi"]; s1 = np.abs(D["slope1"]); s2 = np.abs(D["slope2"]); ppr = D["ppr"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    ax[0].plot(isi, ppr, "o-", color="#8e44ad", lw=2, ms=7)
    ax[0].axhline(1.0, color="0.6", ls="--", lw=1)
    ax[0].fill_between(isi, 1.0, np.maximum(ppr, 1.0), color="#8e44ad", alpha=0.15)
    ax[0].set_xlabel("ISI (ms)"); ax[0].set_ylabel("PPR (slope2/slope1)")
    ax[0].set_title(f"(A) 짝펄스 비 PPR vs ISI\n>1=촉진(SC E1s) · 기록전극#{rec_j}"); ax[0].grid(alpha=0.3)
    for x, y in zip(isi, ppr):
        ax[0].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax[1].plot(isi, s1, "o-", color="#7f8c8d", lw=2, label="1번째 펄스")
    ax[1].plot(isi, s2, "s-", color="#c0392b", lw=2, label="2번째 펄스")
    ax[1].set_xlabel("ISI (ms)"); ax[1].set_ylabel("fEPSP |slope| (µV/ms)")
    ax[1].set_title("(B) 펄스별 fEPSP slope"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    elec_inset(ax[2])
    fig.suptitle(f"MEA PPF 실험 — 국소 SC 짝펄스 → 촉진(PPR>1) (실제 {N:,}세포){cap}", fontsize=12, y=1.02)
    print(f"[PPF] PPR {ppr.min():.2f}~{ppr.max():.2f} · slope1 {s1.min():.4f}~{s1.max():.4f} µV/ms", flush=True)

elif kind == "ltp":
    t = D["t"]; Ve = D["Ve"]; tb = D["t_base"]; tt_ = D["t_tbs"]; tp = D["t_post"]
    sb = np.abs(D["slope_base"]); sp_ = np.abs(D["slope_post"])
    pct = float(G("ltp_pct", np.nan)); rho_m = float(G("rho_mean", 0)); rup = int(G("rho_up", 0))
    rn = int(G("rho_n", 0)); nspk = int(G("nspk", 0)); is_pl = bool(G("plastic", False))
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.28)
    # (A) 전체 시간경과
    axA = fig.add_subplot(gs[0, :])
    axA.plot(t, Ve[rec_j], color="#1f6fb2", lw=0.7)
    axA.axvspan(tb[0] - 60, tb[-1] + 60, color="#ecf0f1", alpha=0.8, zorder=0, label="baseline(약자극)")
    axA.axvspan(tt_[0] - 30, tt_[-1] + 60, color="#fdebd0", alpha=0.9, zorder=0, label="TBS 유도(강자극)")
    axA.axvspan(tp[0] - 60, tp[-1] + 60, color="#d5f5e3", alpha=0.8, zorder=0, label="사후(약자극)")
    for x in list(tb) + list(tp):
        axA.axvline(x, color="0.55", lw=0.6, ls=":")
    axA.set_xlabel("시간 (ms)"); axA.set_ylabel("전극 fEPSP (µV)"); axA.legend(fontsize=8, ncol=3, loc="lower left")
    axA.set_title(f"(A) LTP 프로토콜 전체 시간경과 — 기록전극#{rec_j}(SR) · "
                  f"{'칼슘 가소성 시냅스' if is_pl else '대조군(가소성 없음)'}", fontsize=11)
    # (B) 테스트펄스별 slope
    axB = fig.add_subplot(gs[1, 0])
    xb = np.arange(len(sb)); xp = np.arange(len(sp_)) + len(sb) + 0.6
    axB.bar(xb, sb, color="#7f8c8d", edgecolor="0.3", label="baseline")
    axB.bar(xp, sp_, color="#c0392b", edgecolor="0.3", label="사후(TBS 후)")
    axB.axhline(sb.mean(), color="#7f8c8d", ls="--", lw=1)
    axB.axhline(sp_.mean(), color="#c0392b", ls="--", lw=1)
    axB.set_xticks(list(xb) + list(xp))
    axB.set_xticklabels([f"B{i+1}" for i in range(len(sb))] + [f"P{i+1}" for i in range(len(sp_))], fontsize=8)
    axB.set_ylabel("fEPSP |slope| (µV/ms)"); axB.legend(fontsize=8)
    axB.set_title(f"(B) 테스트펄스별 slope\n{sb.mean():.3f} → {sp_.mean():.3f} = **{pct:+.1f}%**", fontsize=10.5)
    axB.grid(axis="y", alpha=0.3)
    # (C) 효능 ρ
    axC = fig.add_subplot(gs[1, 1]); axC.axis("off")
    axC.text(0, 1.0, "(C) 시냅스 효능 ρ (칼슘 가소성)", fontsize=11, fontweight="bold", va="top")
    lines = [f"가소성 시냅스 : {rn:,} 개", f"ρ0 (유도 전)  : 0.000 (DOWN)",
             f"ρ 평균(유도 후): {rho_m:.3f}", f"ρ>0.5 (UP)    : {rup:,} 개 ({100*rup/max(rn,1):.1f}%)",
             f"전달강도 w    : 1 → {1 + rho_m * (5.28145 - 1):.2f} 배",
             f"TBS 유발 스파이크: {nspk:,}", "",
             f"fEPSP 변화     : {pct:+.1f}%"]
    for i, s in enumerate(lines):
        axC.text(0, 0.88 - i * 0.105, s, fontsize=9.5, va="top", family="Malgun Gothic")
    elec_inset(fig.add_subplot(gs[1, 2]))
    fig.suptitle(f"MEA LTP 실험 — baseline → TBS 유도 → 사후 (실제 {N:,}세포·연속 구동){cap}", fontsize=12, y=1.0)
    print(f"[LTP] {sb.mean():.4f} → {sp_.mean():.4f} µV/ms = {pct:+.1f}% · ρ평균 {rho_m:.3f} · UP {rup}/{rn}", flush=True)

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, f"MEA_{tag}.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
