# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_elec_diag.py  —  전극 24개가 왜 쓸 수 있고 왜 못 쓰는지 **눈으로** 보는 그림

배경: 1단계 완주 결과에서 24전극 중 기울기를 믿을 수 있는 전극이 7개뿐이었다.
"나머지는 동작도 안 하는 건가"라는 질문에 표로 답하는 대신 **파형 자체**를 보인다.

★2026-08-07 정정 — 처음엔 edge_peak 전극 6개(#0,1,2,3,8,9)를 "반응이 없는 위치"라고 적었는데
  **틀렸다.** 이들은 자극 후 3.2 ms에 +2,186~+12,197 µV 의 거대한 **양(+)** 피크를 낸다.
  싱크 전극이 최저를 찍는 바로 그 순간이다 = **전류원(source)**. measure_fepsp 가 음(-)
  최소값만 찾기 때문에 그 봉우리를 건너뛰고 24 ms의 -25~-81 µV 꼬리를 피크로 잡았을 뿐이다.
  그래서 판정에 "전류원(+)"을 따로 두고, 그림에도 양(+) 최대를 청록 ▲로 반드시 표시한다.

만드는 그림 2장 (기본은 마지막 세기 = 가장 센 자극):
  MEA_<tag>_elec_grid.png  3x8 실제 전극 배치 그대로 24전극 파형. 기록 표본을 점으로
                           찍어 20~80% 띠에 표본이 몇 개 들어왔는지가 바로 보인다.
  MEA_<tag>_elec_why.png   판정별 대표 전극 1개씩 크게 + 왜 실격인지 주석 (4칸).

판정 규칙은 mea_io_pick.py 와 **같은 근거**를 쓴다(mea_postproc 의 EDGE_FRAC / POP_REV_FRAC).
실행: <ca1sim>/python.exe 13_net_fepsp/mea_elec_diag.py <tag> [세기인덱스]
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
sys.path.insert(0, HERE)
from mea_postproc import measure_fepsp, EDGE_FRAC, POP_REV_FRAC  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "S1_io_gb"
D = np.load(os.path.join(FIG, f"_mea_{tag}.npz"), allow_pickle=True)
assert str(D["kind"]) == "io", f"io 결과가 아님: {D['kind']}"

lv = np.asarray(D["levels"], float)
na = np.asarray(D["nact"], float)
LI = int(sys.argv[2]) if len(sys.argv) > 2 else len(lv) - 1     # 기본 = 가장 센 자극
tw = np.asarray(D["twin"], float)
W = np.asarray(D["waves"], float)[LI]                            # (24, nt)
stim_t = float(D["stim_t"])
el_layer = D["el_layer"].astype(str)
s_el = np.asarray(D["s_el"], float)
over = np.asarray(D["over"]).astype(int)
E = np.asarray(D["E"], float)                                    # (24, 2) x,y µm
stim_elec = int(D["stim_elec"])
r_stim = float(D["r_stim"])
DUR = 30.0
NEL = W.shape[0]

# ── 판정 (mea_io_pick.py 와 동일 규칙 + 극성 구분) ────────────────────────────
# ★2026-08-07 추가 — edge_peak(=창 뒤쪽에서 음의 최소) 만으로 "반응 없음"이라 부르면 틀린다.
#   measure_fepsp 는 **음(-) 최소값만** 찾는다. 전류원(source) 위에 놓인 전극은
#   같은 시각에 거대한 **양(+)** 피크를 내는데 그건 검출기가 통째로 건너뛴다.
#   그래서 |양최대| > |음최소| 이면 "흐름꼬리"가 아니라 "전류원(+)"로 따로 부른다.
COL = {"정상": "#1b7a3d", "흐름꼬리": "#b06c00", "집단스파이크": "#b0182a",
       "표본부족": "#4a4a9c", "자극전극": "#666666", "전류원(+)": "#0d7d8c"}
fes, verd, pmax, tpmax = [], [], [], []
for j in range(NEL):
    fe = measure_fepsp(tw, W[j], stim_t, DUR, 5.0)
    fes.append(fe)
    yj = W[j] - fe["base"]
    ip = int(np.argmax(yj))
    pmax.append(float(yj[ip]))
    tpmax.append(float(tw[ip] - stim_t))
    if j == stim_elec:
        v = "자극전극"
    elif fe["edge_peak"] and pmax[j] > abs(fe["amp"]):
        v = "전류원(+)"
    elif fe["edge_peak"]:
        v = "흐름꼬리"
    elif fe["pop_spike"]:
        v = "집단스파이크"
    elif fe["n_band"] < 2 and over[j]:
        v = "표본부족"
    else:
        v = "정상"
    verd.append(v)
pmax = np.asarray(pmax, float)
tpmax = np.asarray(tpmax, float)

band_lo = s_el[stim_elec] - r_stim
band_hi = s_el[stim_elec] + r_stim
in_band = ((s_el >= band_lo) & (s_el <= band_hi)).astype(int)

print(f"[{tag}] 세기 {100*lv[LI]:.1f}% (섬유 {na[LI]:.0f}개) · 창 {DUR:.0f} ms · "
      f"기록간격 {tw[1]-tw[0]:.2f} ms · 창내 표본 {len(tw)}개")
print(f"{'전극':>4} {'층':>4} {'층좌표':>8} {'띠안':>4} {'음최소µV':>11} {'@ms':>6} "
      f"{'양최대µV':>11} {'@ms':>6} {'띠표본':>6} {'되돌림%':>8} {'판정':>9}")
for j in range(NEL):
    fe = fes[j]
    print(f"{j:>4} {el_layer[j]:>4} {s_el[j]:>8.1f} {in_band[j]:>4} {fe['amp']:>11.2f} "
          f"{fe['tpk']-stim_t:>6.2f} {pmax[j]:>11.2f} {tpmax[j]:>6.2f} "
          f"{fe['n_band']:>6d} {100*fe['rev_frac']:>8.1f} {verd[j]:>9}")
from collections import Counter                                    # noqa: E402
print("  집계 " + " · ".join(f"{k} {v}개" for k, v in Counter(verd).most_common()))
n_amp = sum(1 for j in range(NEL) if verd[j] not in ("흐름꼬리", "전류원(+)", "자극전극"))
n_slp = sum(1 for j in range(NEL) if verd[j] == "정상")
n_src = sum(1 for j in range(NEL) if verd[j] == "전류원(+)")
print(f"  → 음(-)기울기 사용가능 {n_slp}/{NEL} · 음(-)진폭 사용가능 {n_amp}/{NEL} · "
      f"전류원(+) {n_src}/{NEL} (극성 반대 · 반응 자체는 있음 · CSD용)")


def draw_wave(ax, j, small=True):
    """전극 j의 창내 파형 + 기록 표본 점 + 20~80% 띠. 실격 근거가 보이도록."""
    fe = fes[j]
    x = tw - stim_t
    y = W[j] - fe["base"]
    amp = fe["amp"]
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=1)
    # 흐름꼬리 문턱(창 뒤쪽)을 회색으로 깔아 "여기서 피크가 나면 실격"을 보인다
    ax.axvspan(EDGE_FRAC * DUR, DUR, color="#000000", alpha=0.055, zorder=0, lw=0)
    if amp < 0:
        lo, hi = 0.2 * amp, 0.8 * amp
        ax.axhspan(hi, lo, color="#f5c518", alpha=0.20, zorder=0, lw=0)
        ax.axhline(lo, color="#c9a100", lw=0.6, ls=":", zorder=1)
        ax.axhline(hi, color="#c9a100", lw=0.6, ls=":", zorder=1)
        ipk = int(np.argmin(y))
        seg = y[:ipk + 1]
        inb = np.zeros(len(y), bool)
        inb[:ipk + 1] = (seg <= lo) & (seg >= hi)
    else:
        inb = np.zeros(len(y), bool)
    ax.plot(x, y, "-", color="#333333", lw=1.0, zorder=3)
    ax.plot(x, y, "o", color="#333333", ms=2.0 if small else 3.4, zorder=4)
    if inb.any():                                    # 20~80% 띠에 실제로 들어온 표본
        ax.plot(x[inb], y[inb], "o", color="#e02020", ms=4.5 if small else 8.0,
                zorder=6, mec="white", mew=0.6)
    ax.plot([fe["tpk"] - stim_t], [amp], "v", color="#1f6fd0",
            ms=6 if small else 11, zorder=7, mec="white", mew=0.7)
    # ★양(+) 최대 = 전류원(source). 검출기가 못 보는 쪽이라 눈으로는 반드시 보여야 한다.
    if pmax[j] > 0.25 * abs(amp):
        ax.plot([tpmax[j]], [pmax[j]], "^", color="#0d7d8c",
                ms=6 if small else 11, zorder=7, mec="white", mew=0.7)
    ax.set_xlim(-0.5, DUR + 0.5)
    return fe


# ══ 그림 1 — 3x8 배치 그대로 24전극 ══════════════════════════════════════════
ys = np.unique(np.round(E[:, 1], 1))[::-1]          # 위쪽(큰 y)부터
xs = np.unique(np.round(E[:, 0], 1))
nr, nc = len(ys), len(xs)
fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.55 * nr))
for j in range(NEL):
    r = int(np.argmin(np.abs(ys - E[j, 1])))
    c = int(np.argmin(np.abs(xs - E[j, 0])))
    ax = axes[r, c]
    fe = draw_wave(ax, j, small=True)
    ax.set_title(f"#{j} {el_layer[j]} · {verd[j]}", color=COL[verd[j]],
                 fontsize=10.5, fontweight="bold", pad=3)
    box = (f"음 {fe['amp']:,.0f} µV @ {fe['tpk']-stim_t:.1f} ms\n"
           f"양 {pmax[j]:+,.0f} µV @ {tpmax[j]:.1f} ms\n띠표본 {fe['n_band']}개")
    ax.text(0.97, 0.06, box,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.4))
    ax.tick_params(labelsize=7.5)
    if r == nr - 1:
        ax.set_xlabel("자극 후 시간 (ms)", fontsize=8.5)
    if c == 0:
        ax.set_ylabel(f"y = {E[j,1]:+.0f} µm\nVe (µV)", fontsize=8.5)
for r in range(nr):
    for c in range(nc):
        if not axes[r, c].lines:
            axes[r, c].axis("off")
fig.suptitle(
    f"전극 24개 전부의 fEPSP 파형 — 실제 3×8 배치 그대로 · {tag} · "
    f"자극세기 {100*lv[LI]:.0f}% (섬유 {na[LI]:.0f}/200) · 자극전극 #{stim_elec}\n"
    f"검은 점 = 실제 기록 표본(0.4 ms 간격) · 빨간 점 = 20~80% 띠 안에 들어온 표본"
    f"(기울기는 이 점들로 정해진다) · 노란 띠 = 20~80% 구간 · "
    f"회색 영역 = 창 뒤쪽 {EDGE_FRAC*DUR:.0f}~{DUR:.0f} ms(여기서 음의 피크가 나면 실격)\n"
    f"파란 ▼ = 검출기가 잡은 음(-) 최소 · 청록 ▲ = 양(+) 최대(전류원). "
    f"검출기는 음(-)만 찾으므로 ▲가 ▼보다 큰 전극은 '반응 없음'이 아니라 **극성이 반대**다\n"
    f"세로 눈금은 전극마다 다르다 — 행이 바뀌면 진폭이 수백 배 달라지기 때문",
    fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.945])
p1 = os.path.join(FIG, f"MEA_{tag}_elec_grid.png")
fig.savefig(p1, dpi=125)
plt.close(fig)
print("saved:", p1)


# ══ 그림 2 — 판정 4종 대표 전극을 크게 + 왜 실격인지 ═══════════════════════════
def pick(v):
    """그 판정 중 반응이 가장 큰(= 가장 설득력 있는) 전극.
    전류원(+)은 음최소가 아니라 **양최대**로 골라야 대표가 된다."""
    cand = [j for j in range(NEL) if verd[j] == v]
    if not cand:
        return None
    key = (lambda j: pmax[j]) if v == "전류원(+)" else (lambda j: abs(fes[j]["amp"]))
    return max(cand, key=key)


order = ["정상", "표본부족", "집단스파이크", "전류원(+)", "흐름꼬리"]
sel = [(v, pick(v)) for v in order]
sel = [(v, j) for v, j in sel if j is not None]
if len(sel) > 4:                                   # 4칸뿐 — 빠진 판정은 반드시 알린다
    print("  ※ 그림2는 4칸이라 다음 판정은 빠짐: " + ", ".join(v for v, _ in sel[4:]))
    sel = sel[:4]
fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.2))
for ax, (v, j) in zip(axes.ravel(), sel):
    fe = draw_wave(ax, j, small=False)
    amp = fe["amp"]
    x = tw - stim_t
    y = W[j] - fe["base"]
    ipk = int(np.argmin(y))
    note = ""
    if v == "정상":
        t20, t80 = fe["t20"] - stim_t, fe["t80"] - stim_t
        for tc, lab in ((t20, "t20"), (t80, "t80")):
            ax.axvline(tc, color="#e02020", lw=1.2, ls="--", zorder=5)
            ax.text(tc, 0.02 * amp, f" {lab}={tc:.2f}", color="#e02020",
                    fontsize=10, rotation=90, va="top")
        ax.plot([t20, t80], [0.2 * amp, 0.8 * amp], "-", color="#e02020", lw=2.6, zorder=8)
        note = (f"쓸 수 있다 — 20%·80%를 지나는 시각이 {t80-t20:.3f} ms 떨어져 있고,\n"
                f"그 사이 기록 표본이 {fe['n_band']}개 있다. 되돌림 {100*fe['rev_frac']:.1f}%"
                f"(문턱 {100*POP_REV_FRAC:.0f}%)로 깨끗하고 피크도 {fe['tpk']-stim_t:.1f} ms로 이르다.\n"
                f"기울기 = 0.6 x 진폭 / (t80-t20) = {0.6*amp/(t80-t20):,.1f} µV/ms")
    elif v == "표본부족":
        note = (f"진폭은 멀쩡하다 ({amp:,.0f} µV) — 전극이 고장난 게 아니다.\n"
                f"문제는 20~80% 띠(노란 구간) 안에 기록 표본이 **{fe['n_band']}개**뿐이라는 것.\n"
                f"0.4 ms 간격이 이 빠른 상승을 못 쪼갠다. --rec_dt 를 줄이면 살아난다.")
    elif v == "집단스파이크":
        run = np.minimum.accumulate(y[:ipk + 1])
        ax.plot(x[:ipk + 1], run, color="#8e44ad", lw=1.3, ls="--", zorder=5)
        k = int(np.argmax((y[:ipk + 1] - run) / np.where(np.abs(run) < 1e-9, 1e-9, np.abs(run))
                          * (np.abs(run) >= 0.1 * abs(amp))))
        ax.annotate("", xy=(x[k], y[k]), xytext=(x[k], run[k]),
                    arrowprops=dict(arrowstyle="<->", color="#b0182a", lw=2.2), zorder=9)
        ax.text(x[k] + 0.4, 0.5 * (y[k] + run[k]), "되돌림", color="#b0182a",
                fontsize=11, fontweight="bold", va="center")
        note = (f"진폭은 쓸 수 있지만 기울기는 못 쓴다.\n"
                f"내려가다 위로 되올라간 양이 최대 **{100*fe['rev_frac']:.1f}%**(문턱 "
                f"{100*POP_REV_FRAC:.0f}%) — 보라 점선(그때까지의 최저값)에서 떨어진 만큼이다.\n"
                f"fEPSP 위에 집단발화가 겹쳐 두 성분이 됐다는 뜻. 20~80% 교차가\n"
                f"집단발화의 상승면에 몰려 기울기가 부풀려진다.")
    elif v == "전류원(+)":
        # ★검출기가 못 보는 쪽을 보여준다. 음최소는 y=0 근처에 붙어 거의 안 보이는데,
        #   그게 바로 "왜 -80 µV 를 피크로 잡았나"의 답이다.
        jm = int(np.argmin(y))
        ax.annotate("", xy=(tpmax[j], pmax[j]), xytext=(tpmax[j], 0),
                    arrowprops=dict(arrowstyle="->", color="#0d7d8c", lw=2.6), zorder=9)
        ax.annotate(f"진짜 반응은 이 위쪽 봉우리\n{pmax[j]:+,.0f} µV @ {tpmax[j]:.1f} ms",
                    xy=(tpmax[j], 0.80 * pmax[j]), xytext=(0.22, 0.86),
                    textcoords="axes fraction", color="#0d7d8c", fontsize=12,
                    fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="->", color="#0d7d8c", lw=1.4))
        ax.annotate(f"검출기가 잡은 '피크' {amp:,.1f} µV @ {fe['tpk']-stim_t:.1f} ms\n"
                    f"— 위 봉우리의 {100*abs(amp)/pmax[j]:.2f}% 밖에 안 된다",
                    xy=(x[jm], y[jm]), xytext=(0.97, 0.50),
                    textcoords="axes fraction", color="#1f6fd0", fontsize=10.5,
                    ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="->", color="#1f6fd0", lw=1.4))
        note = (f"★정정 — 반응이 없는 게 아니다. **극성이 반대다.**\n"
                f"자극 {tpmax[j]:.1f} ms 뒤 {pmax[j]:+,.0f} µV — 싱크 전극이\n"
                f"최저를 찍는 바로 그 순간이다. 전류가 빠져나가는\n"
                f"쪽(source)에 놓였다는 뜻이고 물리적으로 당연한 짝이다.\n"
                f"measure_fepsp 는 **음(-) 최소만** 찾으므로 이 봉우리를\n"
                f"건너뛰고, 남은 {amp:,.1f} µV 꼬리({fe['tpk']-stim_t:.1f} ms, 회색)를\n"
                f"피크로 잡아 실격시켰다. |양|/|음| = {pmax[j]/max(abs(amp),1e-9):,.0f}배.\n"
                f"싱크 기울기로는 못 쓰지만 **CSD 분석에는 이쪽이 필요하다.**")
    else:  # 흐름꼬리 (음/양 둘 다 작은 진짜 흐름 꼬리)
        ax.annotate("", xy=(2.0, 0.12 * amp), xytext=(6.0, 0.12 * amp),
                    arrowprops=dict(arrowstyle="<->", color="#1b7a3d", lw=2.0))
        ax.text(4.0, 0.05 * amp, "진짜 fEPSP는 여기서 피크", color="#1b7a3d",
                fontsize=10.5, ha="center", va="bottom")
        note = (f"쓸 수 없다 — 피크가 {fe['tpk']-stim_t:.1f} ms, 즉 30 ms 창의 끝자락(회색)에\n"
                f"붙어 있다. 양(+)쪽 최대도 {pmax[j]:+,.0f} µV 로 작아, 이건 자극에 대한 반응이\n"
                f"아니라 느린 흐름의 꼬리다. 창을 늘리면 '진폭'도 따라 커지는 가짜 값.\n"
                f"진폭 {amp:,.0f} µV 는 같은 세기 최대 전극의 {100*abs(amp)/max(abs(f['amp']) for f in fes):.2f}%.")
    ax.set_title(f"[{v}]  전극 #{j} · {el_layer[j]}층 · 층좌표 {s_el[j]:+.1f} µm · "
                 f"SC 층대 안={in_band[j]} · 위치 (x {E[j,0]:+.0f}, y {E[j,1]:+.0f}) µm",
                 color=COL[v], fontsize=12.5, fontweight="bold")
    ax.set_xlabel("자극 후 시간 (ms)", fontsize=10.5)
    ax.set_ylabel("Ve - 기준선 (µV)", fontsize=10.5)
    ax.text(0.015, 0.03, note, transform=ax.transAxes, fontsize=10.2, va="bottom",
            bbox=dict(fc="#fffbe8", ec=COL[v], lw=1.2, alpha=0.95, pad=5))
fig.suptitle(
    f"판정이 실제로 어떻게 생겼나 — {tag} · 자극세기 {100*lv[LI]:.0f}% "
    f"(섬유 {na[LI]:.0f}/200) · 각 판정에서 반응이 가장 큰 전극\n"
    f"음(-)기울기 사용가능 {n_slp}/{NEL} · 음(-)진폭 사용가능 {n_amp}/{NEL} · "
    f"전류원(+) {n_src}/{NEL} — 고장나거나 무반응인 전극은 하나도 없다",
    fontsize=14, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.935])
p2 = os.path.join(FIG, f"MEA_{tag}_elec_why.png")
fig.savefig(p2, dpi=125)
plt.close(fig)
print("saved:", p2)
