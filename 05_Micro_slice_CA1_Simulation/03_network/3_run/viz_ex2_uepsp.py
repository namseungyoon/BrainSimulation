# -*- coding: utf-8 -*-
"""Ex2 uEPSP 결과 그림 (4패널) — ex2_uepsp.npz/json 실측으로 생성.
(a) uEPSP 파형(소마+수상돌기, 시행 겹침+평균) (b) 진폭 분포 (c) 페어펄스 PPR (d) Sayer 1990 대조표.
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
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

d = np.load(os.path.join(ROOT, "scratch", "ex2_uepsp.npz"), allow_pickle=True)
J = json.load(open(os.path.join(ROOT, "scratch", "ex2_uepsp.json"), encoding="utf-8"))
t = d["t"].astype(float); sv = d["soma_v"].astype(float)
dv = d["dend_v"].astype(float) if d["dend_v"].size else None
stims = d["stims"].astype(float); SETTLE = float(d["settle"]); ISO = float(d["iso"]); ISI = float(d["isi"])
single = d["single_amps"]; pa1 = d["pair_a1"].astype(float); pa2 = d["pair_a2"].astype(float)
C_S, C_D, C_M = "#C44E52", "#4C72B0", "#2a2a2a"

WIN = (-5.0, 45.0)   # 자극기준 발췌창


def window(arr, t0):
    m = (t >= t0 + WIN[0]) & (t <= t0 + WIN[1])
    tt = t[m] - t0; vv = arr[m]
    base = vv[tt < 0].mean() if (tt < 0).any() else vv[0]
    return tt, vv - base

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.34, wspace=0.22)

# (a) uEPSP 파형 — 소마(시행 겹침 + 평균) + 수상돌기
ax = fig.add_subplot(gs[0, 0])
segs = []
for st in stims:
    tt, vv = window(sv, st)
    if vv.max() > 0.05:                       # 성공 시행만 겹침
        ax.plot(tt, vv, color=C_S, alpha=0.18, lw=0.8)
        segs.append((tt, vv))
if segs:
    L = min(len(s[1]) for s in segs)
    mean_v = np.mean([s[1][:L] for s in segs], axis=0)
    ax.plot(segs[0][0][:L], mean_v, color=C_S, lw=2.4, label="소마 평균 (성공시행)")
if dv is not None:
    kb = int(d["kbest"]); tt, vv = window(dv, stims[kb])
    ax.plot(tt, vv, color=C_D, lw=2.0, ls="--", label="수상돌기(시냅스 자리)")
ax.axvline(0, ls=":", color="k", lw=1); ax.set_xlabel("자극기준 시간 (ms)"); ax.set_ylabel("ΔVm (mV)")
ax.set_title("(a) 단발 uEPSP — 수상돌기→소마 감쇠·전파"); ax.legend(fontsize=8)

# (b) 진폭 분포
ax = fig.add_subplot(gs[0, 1])
amps = np.concatenate([np.array(a, float) for a in single])
succ = amps[amps > 0.05]
ax.hist(amps, bins=np.linspace(0, max(amps.max(), 0.5), 26), color=C_S, alpha=0.85)
ax.axvline(J["median_uEPSP_mV"], color="k", ls="--", lw=1.5, label=f"중앙 {J['median_uEPSP_mV']:.2f} mV")
ax.axvline(0.05, color="gray", ls=":", lw=1, label="실패 임계 0.05")
ax.set_xlabel("uEPSP 진폭 (mV)"); ax.set_ylabel("시행 수")
ax.set_title(f"(b) 진폭 분포 — 평균 {J['mean_uEPSP_mV']:.2f}·중앙 {J['median_uEPSP_mV']:.2f} mV · 실패 {J['fail_rate']*100:.0f}%")
ax.legend(fontsize=8)

# (c) 페어펄스 PPR
ax = fig.add_subplot(gs[1, 0])
m1, m2 = pa1.mean(), pa2.mean()
bars = ax.bar(["EPSP1", f"EPSP2 (ISI {ISI:.0f}ms)"], [m1, m2], color=[C_D, C_S], width=0.55)
for b, v in zip(bars, [m1, m2]):
    ax.annotate(f"{v:.3f} mV", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=10)
ax.set_ylabel("평균 진폭 (mV)")
ax.set_title(f"(c) 페어펄스 촉진 — PPR {J['PPR']:.2f} (mean E2/E1, >1=촉진)")
ax.set_ylim(0, max(m1, m2) * 1.25)

# (d) Sayer 1990 대조표
ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
rows = [
    ["지표", "Ex2 (우리)", "Sayer 1990 (실측)"],
    ["uEPSP 진폭", f"중앙 {J['median_uEPSP_mV']:.2f} mV", "~0.13 mV (0.03–0.66)"],
    ["지연", f"{J['rep_latency_ms']:.2f} ms", "~1–3 ms"],
    ["10-90% 상승", f"{J['rep_rise_ms']:.2f} ms", "~1–3 ms"],
    ["감쇠 τ", f"{J['rep_decay_ms']:.2f} ms", "수십 ms"],
    ["실패율", f"{J['fail_rate']*100:.0f}%", "존재(확률방출)"],
    ["PPR (50ms)", f"{J['PPR']:.2f}", "~1.3–2.0 (촉진)"],
]
tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.7)
for j in range(3):
    tbl[(0, j)].set_facecolor("#3a3f4b"); tbl[(0, j)].set_text_props(color="w", fontweight="bold")
ax.set_title("(d) Sayer 1990 실측 대조", y=0.86)

fig.suptitle(f"Ex2 — Schaffer collateral 단발 uEPSP (섬유 {int(d['fiber'])} · 추체 {J['n_targets']}타깃 · {J['ntrial']}시행)",
             fontsize=13, y=0.98)
out = os.path.join(FIG, "ex2_uepsp.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"[그림] -> {out}")
print(f"  uEPSP 중앙 {J['median_uEPSP_mV']:.3f} mV · PPR {J['PPR']:.2f} · 실패 {J['fail_rate']*100:.0f}% · 지연 {J['rep_latency_ms']:.2f}ms")
