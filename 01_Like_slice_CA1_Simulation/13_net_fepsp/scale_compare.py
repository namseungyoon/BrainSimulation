# -*- coding: utf-8 -*-
"""13_net_fepsp/scale_compare.py  —  Stage A 규모 보정: 2,000세포 서브셋 vs 전규모(17,647)

같은 LTP 프로토콜을 두 규모에서 돌린 결과를 나란히 놓고 **무엇이 규모에 의존하고
무엇이 규모 불변인지**를 가른다. 핵심 질문 하나:

    fEPSP 진폭은 당연히 규모에 비례해 커진다. 그런데 **가소성 비율**(LTP %)도
    규모에 따라 변하는가? 변하지 않는다면 이후 배터리를 서브셋으로 돌려도 된다.

판정은 가소성판 단독이 아니라 **(가소성 ON) - (엄격 대조군 γ=0)** 의 차이로 한다.
대조군이 없으면 그 규모는 "판정 불가"로 표시하고 넘어간다(추정하지 않는다).

입력(figures/):
  2k   : _mea_ltp_plastic.npz      · _mea_ltp_frozen.npz
  전규모: _mea_full_ltp_plastic.npz · _mea_full_ltp_frozen.npz
실행: <ca1sim>/py 13_net_fepsp/scale_compare.py
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
    """없으면 None. 옛 npz엔 chunk_ms/no_conn 키가 없으므로 전부 기본값 허용."""
    p = os.path.join(FIG, f"_mea_{tag}.npz")
    if not os.path.exists(p):
        return None
    D = np.load(p, allow_pickle=True)

    def g(k, d=np.nan):
        if k not in D.files:
            return d
        a = D[k]
        return a.item() if a.shape == () else a

    return dict(
        tag=tag, t=D["t"], Ve=D["Ve"], rec_j=int(g("rec_j", 0)),
        tb=np.asarray(D["t_base"], float), tt=np.asarray(D["t_tbs"], float),
        tp=np.asarray(D["t_post"], float),
        sb=np.abs(np.asarray(D["slope_base"], float)),
        sp=np.abs(np.asarray(D["slope_post"], float)),
        pct=float(g("ltp_pct")), rho=float(g("rho_mean", 0.0)), rho_n=int(g("rho_n", 0)),
        nspk=int(g("nspk", 0)), N=int(g("N", 0)), n_sc=int(g("n_sc", 0)),
        n_syn=int(g("n_syn", 0)), n_sccell=int(g("n_sccell", 0)),
        io_test=int(g("io_test", 0)), det=bool(g("det", True)),
        nhost=int(g("nhost", 0)), Hh=float(g("Hh", np.nan)),
        chunk=float(g("chunk_ms", 0.0)), rec_dt=float(D["t"][1] - D["t"][0]),
    )


# ── 두 규모 로드 ──────────────────────────────────────────────────────────
SC = [("2,000세포 서브셋", "ltp_plastic", "ltp_frozen", "#c0392b", "#e8a49c"),
      ("전규모 17,647세포", "full_ltp_plastic", "full_ltp_frozen", "#1f6fb2", "#9fc3e0")]
sets = []
for name, tp_, tf_, cP, cF in SC:
    P, F = load(tp_), load(tf_)
    if P is None:
        print(f"[없음] {name}: _mea_{tp_}.npz 아직 없음 — 건너뜀")
        continue
    if F is None:
        print(f"[주의] {name}: 엄격 대조군 _mea_{tf_}.npz 없음 → 고유 효과 판정 불가")
    sets.append(dict(name=name, P=P, F=F, cP=cP, cF=cF,
                     net=(P["pct"] - F["pct"]) if F else np.nan))
if not sets:
    sys.exit("입력 npz가 하나도 없다 — 먼저 LTP 런을 돌려야 한다.")

# ── 프로토콜 일치 점검(다르면 비교 자체가 무의미하므로 먼저 본다) ──────────
ref = sets[0]["P"]
mismatch = []
for s in sets[1:]:
    for k, lab in [("tb", "baseline 시각"), ("tt", "TBS 시각"), ("tp", "사후 시각")]:
        if len(s["P"][k]) != len(ref[k]) or not np.allclose(s["P"][k], ref[k]):
            mismatch.append(f"{lab}: {ref['tag']}={ref[k]} vs {s['P']['tag']}={s['P'][k]}")
    for k, lab in [("io_test", "테스트 섬유 수"), ("det", "방출 모드"), ("rec_dt", "기록 dt")]:
        if s["P"][k] != ref[k]:
            mismatch.append(f"{lab}: {ref['tag']}={ref[k]} vs {s['P']['tag']}={s['P'][k]}")
for m in mismatch:
    print("[불일치]", m)
if not mismatch:
    print("[점검] 두 규모의 프로토콜(자극 시각·테스트 세기·방출·기록 dt) 완전 일치")

# ── 그림 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 12.5))
gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.30)

# (A) 원 스케일 시간경과 — 진폭이 규모에 어떻게 붙는지
axA = fig.add_subplot(gs[0, 0:2])
for s in sets:
    D_ = s["P"]
    axA.plot(D_["t"], D_["Ve"][D_["rec_j"]], color=s["cP"], lw=0.7,
             label=f"{s['name']} (가소성)  baseline {D_['sb'].mean():.3f} µV/ms")
axA.axvspan(ref["tb"][0] - 60, ref["tb"][-1] + 60, color="#ecf0f1", alpha=0.85, zorder=0)
axA.axvspan(ref["tt"][0] - 30, ref["tt"][-1] + 60, color="#fdebd0", alpha=0.9, zorder=0)
axA.axvspan(ref["tp"][0] - 60, ref["tp"][-1] + 60, color="#d5f5e3", alpha=0.85, zorder=0)
axA.set_ylabel("fEPSP (µV)"); axA.legend(fontsize=8.5, loc="lower left")
axA.set_title("(A) 원 스케일 시간경과 — 진폭은 규모에 의존한다\n"
              "회색=baseline · 주황=TBS · 녹색=사후", fontsize=10.5)

# (B) baseline 정규화 — 모양(가소성 비율)이 규모 불변인지
axB = fig.add_subplot(gs[1, 0:2])
for s in sets:
    D_ = s["P"]; b = max(D_["sb"].mean(), 1e-12)
    axB.plot(D_["t"], D_["Ve"][D_["rec_j"]] / b, color=s["cP"], lw=0.7, label=f"{s['name']} 가소성")
    if s["F"]:
        G = s["F"]; bF = max(G["sb"].mean(), 1e-12)
        axB.plot(G["t"], G["Ve"][G["rec_j"]] / bF, color=s["cF"], lw=0.7, ls="--",
                 label=f"{s['name']} 엄격 대조군(γ=0)")
axB.set_xlabel("시간 (ms)"); axB.set_ylabel("fEPSP / baseline slope  (ms)")
axB.legend(fontsize=8.5, loc="lower left")
axB.set_title("(B) baseline slope로 정규화 — 겹치면 '모양은 규모 불변'", fontsize=10.5)

# (C) 규모 대비 절대 진폭
axC = fig.add_subplot(gs[0, 2])
xs = np.arange(len(sets))
axC.bar(xs, [s["P"]["sb"].mean() for s in sets], color=[s["cP"] for s in sets],
        edgecolor="0.3", width=0.55)
for x, s in zip(xs, sets):
    v = s["P"]["sb"].mean()
    axC.annotate(f"{v:.3f}\n(세포 {s['P']['N']:,})", (x, v), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=8.5)
axC.set_xticks(xs); axC.set_xticklabels([s["name"].split()[0] for s in sets], fontsize=9)
axC.set_ylabel("baseline |slope| (µV/ms)"); axC.grid(axis="y", alpha=0.3)
if len(sets) == 2:
    r_amp = sets[1]["P"]["sb"].mean() / max(sets[0]["P"]["sb"].mean(), 1e-12)
    r_cell = sets[1]["P"]["N"] / max(sets[0]["P"]["N"], 1)
    axC.set_title(f"(C) 절대 진폭\n진폭비 {r_amp:.2f}x  vs  세포비 {r_cell:.2f}x", fontsize=10.5)
else:
    axC.set_title("(C) 절대 진폭", fontsize=10.5)

# (D) LTP% — 규모 x 조건
axD = fig.add_subplot(gs[1, 2])
w = 0.35
for i, s in enumerate(sets):
    axD.bar(i - w / 2, s["P"]["pct"], width=w, color=s["cP"], edgecolor="0.3", label="가소성" if i == 0 else None)
    if s["F"]:
        axD.bar(i + w / 2, s["F"]["pct"], width=w, color=s["cF"], edgecolor="0.3",
                label="엄격 대조군(γ=0)" if i == 0 else None)
    axD.annotate(f"{s['P']['pct']:+.1f}%", (i - w / 2, s["P"]["pct"]), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")
    if s["F"]:
        axD.annotate(f"{s['F']['pct']:+.1f}%", (i + w / 2, s["F"]["pct"]), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9)
axD.set_xticks(xs); axD.set_xticklabels([s["name"].split()[0] for s in sets], fontsize=9)
axD.axhline(0, color="0.5", lw=0.8); axD.grid(axis="y", alpha=0.3)
axD.set_ylabel("TBS 후 slope 변화 (%)"); axD.legend(fontsize=8)
axD.set_title("(D) 가소성 vs 엄격 대조군", fontsize=10.5)

# (E) 테스트펄스별 궤적(정규화) — 규모 간 궤적 형태 비교
axE = fig.add_subplot(gs[2, 0])
for s in sets:
    for D_, lab, ls, col in ((s["P"], "가소성", "-", s["cP"]), (s["F"], "γ=0", "--", s["cF"])):
        if D_ is None:
            continue
        y = np.concatenate([D_["sb"], D_["sp"]]) / max(D_["sb"].mean(), 1e-12)
        axE.plot(np.arange(len(y)), y, ls, marker="o", color=col, lw=1.6, ms=4.5,
                 label=f"{s['name'].split()[0]} {lab}")
axE.axvline(len(ref["sb"]) - 0.5, color="#e67e22", lw=1.6, ls=":")
axE.set_xticks(np.arange(len(ref["sb"]) + len(ref["tp"])))
axE.set_xticklabels([f"B{i+1}" for i in range(len(ref["sb"]))] +
                    [f"P{i+1}" for i in range(len(ref["tp"]))], fontsize=8)
axE.set_ylabel("slope / baseline 평균"); axE.grid(alpha=0.3); axE.legend(fontsize=7.5)
axE.set_title("(E) 테스트펄스별 궤적(정규화)\n주황 점선 = TBS", fontsize=10)

# (F) 규모·시냅스 구성 표
axF = fig.add_subplot(gs[2, 1]); axF.axis("off")
axF.text(0, 1.0, "(F) 규모 구성 (npz 기록값)", fontsize=11, fontweight="bold", va="top")
rows = [("항목", *[s["name"].split()[0] for s in sets])]
rows += [
    ("세포 수", *[f"{s['P']['N']:,}" for s in sets]),
    ("내부 시냅스", *[f"{s['P']['n_syn']:,}" for s in sets]),
    ("SC 시냅스", *[f"{s['P']['n_sc']:,}" for s in sets]),
    ("SC받은 세포", *[f"{s['P']['n_sccell']:,}" for s in sets]),
    ("테스트 섬유", *[f"{s['P']['io_test']}" for s in sets]),
    ("방출", *["결정론" if s["P"]["det"] else "확률" for s in sets]),
    ("랭크", *[f"{s['P']['nhost']}" for s in sets]),
    ("청크(ms)", *[f"{s['P']['chunk']:.0f}" if s["P"]["chunk"] else "없음" for s in sets]),
    ("기록 dt(ms)", *[f"{s['P']['rec_dt']:.2f}" for s in sets]),
    ("TBS 스파이크", *[f"{s['P']['nspk']:,}" for s in sets]),
    ("rho 평균", *[f"{s['P']['rho']:.3f}" for s in sets]),
]
for i, r in enumerate(rows):
    for j, cell in enumerate(r):
        axF.text(0.02 + j * 0.34, 0.90 - i * 0.077, str(cell), fontsize=8.5, va="top",
                 fontweight="bold" if i == 0 else "normal")

# (G) 판정
axG = fig.add_subplot(gs[2, 2]); axG.axis("off")
axG.text(0, 1.0, "(G) 규모 스케일링 판정", fontsize=11, fontweight="bold", va="top")
lines = []
for s in sets:
    if s["F"]:
        lines.append(f"{s['name'].split()[0]}: 고유 효과 {s['net']:+.1f}%p")
        lines.append(f"   (가소성 {s['P']['pct']:+.1f}% - γ=0 {s['F']['pct']:+.1f}%)")
    else:
        lines.append(f"{s['name'].split()[0]}: 대조군 없음 -> 판정 불가")
lines.append("")
if len(sets) == 2 and all(s["F"] for s in sets):
    d = abs(sets[1]["net"] - sets[0]["net"])
    rel = 100.0 * d / max(abs(sets[0]["net"]), 1e-12)
    lines += [f"규모 간 차이 {d:.1f}%p ({rel:.0f}%)",
              "-> 가소성 비율은 규모 불변에 가깝다" if rel < 20 else
              "-> 가소성 비율이 규모에 의존한다",
              "" if rel < 20 else "   서브셋 외삽 금지",
              "   서브셋으로 배터리 진행 가능" if rel < 20 else ""]
else:
    lines += ["전규모 결과가 아직 없어", "스케일링 법칙 미확정"]
if mismatch:
    lines += ["", "[불일치] 프로토콜이 달라 비교 주의:"] + ["  " + m[:44] for m in mismatch[:3]]
for i, s in enumerate(lines):
    axG.text(0, 0.90 - i * 0.072, s, fontsize=8.8, va="top")

have_full = any(s["P"]["N"] > 10000 for s in sets)
fig.suptitle("Stage A 규모 보정 — 같은 LTP 프로토콜을 2,000세포 서브셋과 전규모에서 비교\n"
             + ("전규모 결과 포함" if have_full else "[주의] 전규모 결과 미포함 — 서브셋만 표시"),
             fontsize=12.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = os.path.join(FIG, "MEA_scale_compare.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
for s in sets:
    print(f"[{s['name']}] N={s['P']['N']:,} · baseline {s['P']['sb'].mean():.4f} µV/ms · "
          f"가소성 {s['P']['pct']:+.2f}%"
          + (f" · γ=0 {s['F']['pct']:+.2f}% · 고유 효과 {s['net']:+.2f}%p" if s["F"] else " · 대조군 없음"))
