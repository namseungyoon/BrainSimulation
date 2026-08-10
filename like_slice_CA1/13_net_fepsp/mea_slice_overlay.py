# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_slice_overlay.py — fEPSP 24파형을 **실제 슬라이스 위 실제 전극 자리**에 얹는 그림

왜 만드나
  전극 번호(#0~#23)만 보면 그 전극이 어느 층 위에 있는지, 자극전극에서 얼마나 떨어졌는지가
  안 보인다. 그래서 "왜 이 전극은 싱크(-)이고 저 전극은 전류원(+)인가"가 표로는 설명이 안 된다.
  → 세포 17,647개를 층 색으로 깔고, 그 위 **저장된 전극 좌표 그대로** 파형을 얹는다.

★무엇이 정확하고 무엇이 관례인가 (캡션에도 같은 내용을 적는다)
  정확:
    - 전극 24개 좌표      = 결과 npz 의 E (µm, 전극면 좌표) 그대로
    - 세포 17,647개 좌표  = 05_placement/slice_cells.npz 원좌표를 **시뮬과 같은 SVD**로 투영
    - 층                  = 경계선을 상상해 긋는 게 아니라 **세포의 층 라벨**로 칠한다
    - 층 좌표 s           = mea_experiment.py:278-280 과 같은 식 (SP=0, SLM 쪽 +)
    - SC 자극 띠          = s ∈ [s_stim - r_stim, s_stim + r_stim]
    - 파형                = 저장된 원파형 (기저선 = 자극 전 5 ms 평균만 뺌)
  관례(물리적 대응 없음):
    - 파형의 µV→µm 표시 축척
    - 화면 회전: 층 깊이축 u_layer 를 세로로 세웠다. **강체 회전이라 왜곡이 아니다**
      (전극면 기저는 SVD가 정한 임의 기저라 어느 방향으로 세워도 같은 그림이다)
  근사:
    - 2D 투영 — 슬라이스 두께축을 무너뜨린다. 층중심의 두께축 퍼짐은 89 µm 로
      면내 퍼짐(663·656 µm) 대비 작아 손실은 작지만 0은 아니다
    - 세포는 소마 점만. 수상돌기는 안 그린다

★기하 재현 검증 (통과 못하면 그림을 안 그리고 멈춘다)
  numpy/LAPACK 판에 따라 SVD 축의 **부호**가 뒤집힐 수 있다. 그래서 재현한 축으로
  전극의 층좌표를 다시 계산해 **npz 에 저장된 s_el 24개와 대조**하고, 덤으로
  over(조직 위 450 µm) 24개도 대조한다. 둘 다 맞아야 진행한다.

만드는 그림 2장
  MEA_<tag>_slice_common.png  24전극 **공통 축척** — 크기 비교가 정직하다(작은 전극은 평평)
  MEA_<tag>_slice_norm.png    24전극 **각자 정규화** — 모양·타이밍·극성 비교용
실행: <ca1sim>/python.exe 13_net_fepsp/mea_slice_overlay.py [tag] [세기인덱스]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
sys.path.insert(0, HERE)
from mea_postproc import measure_fepsp  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "S1_io_gb"
D = np.load(os.path.join(FIG, f"_mea_{tag}.npz"), allow_pickle=True)
assert str(D["kind"]) == "io", f"io 결과가 아님: {D['kind']}"

lv = np.asarray(D["levels"], float)
na = np.asarray(D["nact"], float)
LI = int(sys.argv[2]) if len(sys.argv) > 2 else len(lv) - 1
tw = np.asarray(D["twin"], float)
W = np.asarray(D["waves"], float)[LI]                 # (24, nt) µV
stim_t = float(D["stim_t"])
E = np.asarray(D["E"], float)                         # (24,2) 전극면 좌표 µm
s_el_ref = np.asarray(D["s_el"], float)
el_layer = D["el_layer"].astype(str)
over_ref = np.asarray(D["over"]).astype(bool)
stim_elec = int(D["stim_elec"])
r_stim = float(D["r_stim"])
NEL = W.shape[0]
DUR = 25.0                                            # 파형 표시 구간 (자극 후 ms)

G = lambda k, d=None: (D[k].item() if (k in D.files and D[k].shape == ()) else (D[k] if k in D.files else d))

# ══ 1. 기하 재현 (mea_experiment.py:245-282 와 같은 식) ═══════════════════════
C = np.load(CELLS, allow_pickle=True)
xyz = C["xyz"].astype(float)
layer = C["layer"].astype(str)
etype = C["etype"].astype(str)
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype])

c0 = xyz.mean(0)
Vall = np.linalg.svd(xyz - c0, full_matrices=False)[2]
spreads = []
for i in range(3):
    pr = (xyz - c0) @ Vall[i]
    cen = [pr[layer == Ln].mean() for Ln in ("SO", "SP", "SR", "SLM") if (layer == Ln).any()]
    spreads.append(float(np.ptp(cen)))
i_thick = int(np.argmin(spreads))
i_face = [i for i in range(3) if i != i_thick]

# ── SVD 축 부호/순서 모호성 해소: 저장된 s_el 과 맞는 조합을 찾는다 ──
best = None
for order in ([0, 1], [1, 0]):
    for sg0 in (1.0, -1.0):
        for sg1 in (1.0, -1.0):
            fa = np.vstack([Vall[i_face[order[0]]] * sg0, Vall[i_face[order[1]]] * sg1])
            Ftry = (xyz - c0) @ fa.T
            lc = {Ln: Ftry[layer == Ln].mean(0) for Ln in ("SO", "SP", "SR", "SLM") if (layer == Ln).any()}
            u = lc["SLM"] - lc["SP"]
            u = u / (np.linalg.norm(u) + 1e-12)
            err = float(np.max(np.abs((E - lc["SP"]) @ u - s_el_ref)))
            if best is None or err < best[0]:
                best = (err, fa, lc, u, order, sg0, sg1)
err, face_ax, lay_cen, u_layer, order, sg0, sg1 = best
F = (xyz - c0) @ face_ax.T
tree = cKDTree(F[t4 == "PC"])
over_chk = tree.query(E)[0] < 450.0
s_lay = {Ln: float((lay_cen[Ln] - lay_cen["SP"]) @ u_layer) for Ln in lay_cen}

print(f"[검증] SVD 축 조합 order={order} 부호=({sg0:+.0f},{sg1:+.0f})"
      f" · 두께축=축{i_thick}(층중심 퍼짐 {spreads[i_thick]:.0f}µm)")
print(f"[검증] 전극 층좌표 s_el 최대 오차 {err:.3e} µm  (24개 전부 대조)")
print(f"[검증] over(조직 위 450µm) 일치 {int((over_chk == over_ref).sum())}/24")
print(f"[검증] 층 중심 s  SO {s_lay['SO']:+.0f} · SP {s_lay['SP']:+.0f}"
      f" · SR {s_lay['SR']:+.0f} · SLM {s_lay['SLM']:+.0f} µm")
if err > 1e-3 or not np.array_equal(over_chk, over_ref):
    sys.exit("★기하 재현 실패 — 저장값과 안 맞는다. 그림을 그리지 않는다.")

# ── 화면 좌표: 세로 = 층 깊이(SO 위 → SLM 아래), 가로 = 층을 따라가는 방향 ──
u_perp = np.array([-u_layer[1], u_layer[0]])
oS = lay_cen["SP"]
cell_y = -((F - oS) @ u_layer)                 # -s : SO(음수 s)가 위로
cell_x = (F - oS) @ u_perp
Ey = -s_el_ref
Ex = (E - oS) @ u_perp

# 층 경계: 인접 층 세포 분포의 (아래층 99%, 위층 1%) 중간 — 관례이므로 캡션에 적는다
LORD = ["SO", "SP", "SR", "SLM"]
cs = {Ln: -((F[layer == Ln] - oS) @ u_layer) for Ln in LORD}       # 화면 y
bnds = []
for a, b in zip(LORD[:-1], LORD[1:]):                              # a가 위(y 큼), b가 아래
    bnds.append(0.5 * (np.percentile(cs[a], 1) + np.percentile(cs[b], 99)))

# ══ 2. 전극 판정 (mea_elec_diag.py 와 동일 규칙) ═════════════════════════════
COL = {"정상": "#1b7a3d", "흐름꼬리": "#b06c00", "집단스파이크": "#b0182a",
       "표본부족": "#4a4a9c", "자극전극": "#666666", "전류원(+)": "#0d7d8c"}
fes, verd, pmax, nmin = [], [], [], []
for j in range(NEL):
    fe = measure_fepsp(tw, W[j], stim_t, 30.0, 5.0)
    fes.append(fe)
    yj = W[j] - fe["base"]
    pmax.append(float(yj.max()))
    nmin.append(float(yj.min()))
    if j == stim_elec:
        v = "자극전극"
    elif fe["edge_peak"] and pmax[j] > abs(fe["amp"]):
        v = "전류원(+)"
    elif fe["edge_peak"]:
        v = "흐름꼬리"
    elif fe["pop_spike"]:
        v = "집단스파이크"
    elif fe["n_band"] < 2 and over_ref[j]:
        v = "표본부족"
    else:
        v = "정상"
    verd.append(v)
pmax = np.asarray(pmax); nmin = np.asarray(nmin)

# ══ 3. 그림 ═════════════════════════════════════════════════════════════════
LAY_COL = {"SO": "#4C72B0", "SP": "#DD8452", "SR": "#55A868", "SLM": "#C44E52"}
C_SINK, C_SRC = "#1f4e9c", "#0d7d8c"
m = (tw >= stim_t) & (tw <= stim_t + DUR)
TT = tw[m] - stim_t
XS = 150.0 / DUR                                   # µm per ms (전극 간격 200 µm 대비)
HALF = 88.0                                        # 파형 세로 반높이 µm
VMAX = float(np.max([max(abs(nmin[j]), pmax[j]) for j in range(NEL)]))


def draw(norm):
    fig, ax = plt.subplots(figsize=(19.5, 12.6))
    # 층 띠
    ylo, yhi = cell_y.min() - 60, cell_y.max() + 60
    edges = [yhi] + list(bnds) + [ylo]
    for k, Ln in enumerate(LORD):
        ax.axhspan(edges[k + 1], edges[k], color=LAY_COL[Ln], alpha=0.07, zorder=0)
        ax.axhline(edges[k + 1], color="0.55", lw=0.7, ls=":", zorder=1)
    # 세포 소마 (배경) — PC 와 억제세포를 구분
    for Ln in LORD:
        q = (layer == Ln) & (t4 == "PC")
        ax.scatter(cell_x[q], cell_y[q], s=1.1, c=LAY_COL[Ln], alpha=0.30, lw=0, zorder=2)
    q = t4 != "PC"
    ax.scatter(cell_x[q], cell_y[q], s=2.6, c="0.25", alpha=0.45, lw=0, zorder=3)
    # SC 자극 층대
    ax.axhspan(-(s_el_ref[stim_elec] + r_stim), -(s_el_ref[stim_elec] - r_stim),
               facecolor="none", edgecolor="#b0182a", lw=1.4, ls="--", hatch="///",
               alpha=0.45, zorder=4)
    # 층 이름
    xr = cell_x.max()
    for Ln in LORD:
        ax.text(xr + 250, -s_lay[Ln], f"{Ln}\n(s={s_lay[Ln]:+.0f} µm)", color=LAY_COL[Ln],
                fontsize=15, fontweight="bold", va="center", ha="center", zorder=6)
    # 파형
    for j in range(NEL):
        y0 = W[j][m] - fes[j]["base"]
        sc = (HALF / max(abs(y0).max(), 1e-9)) if norm else (HALF / VMAX)
        X = Ex[j] + TT * XS
        Y = Ey[j] + y0 * sc
        ax.fill_between(X, Ey[j], Y, where=(y0 < 0), color=C_SINK, alpha=0.30, lw=0, zorder=7)
        ax.fill_between(X, Ey[j], Y, where=(y0 > 0), color=C_SRC, alpha=0.30, lw=0, zorder=7)
        ax.plot(X, Y, "-", color="0.12", lw=1.15, zorder=8)
        ax.plot([Ex[j], Ex[j] + DUR * XS], [Ey[j]] * 2, "-", color="0.55", lw=0.6, zorder=6)
        ax.plot([Ex[j]], [Ey[j]], "o", ms=11, mfc="white", mec=COL[verd[j]], mew=2.4, zorder=9)
        if j == stim_elec:
            ax.plot([Ex[j]], [Ey[j]], "*", ms=21, mfc="none", mec="#b0182a", mew=2.0, zorder=10)
        ax.text(Ex[j] - 14, Ey[j], f"#{j}", fontsize=11.5, fontweight="bold",
                ha="right", va="center", color=COL[verd[j]], zorder=10)
        ax.text(Ex[j] + 4, Ey[j] - HALF - 26,
                f"{el_layer[j]} · 음 {nmin[j]:,.0f} · 양 {pmax[j]:+,.0f} µV",
                fontsize=8.6, color="0.25", zorder=10)
    # 축척 막대
    bx, by = cell_x.min() - 40, ylo + 130
    ax.plot([bx, bx + 10 * XS], [by, by], "-", color="k", lw=2.2, zorder=11)
    ax.text(bx + 5 * XS, by - 34, "10 ms", fontsize=11, ha="center", zorder=11)
    if not norm:
        uV = 5000.0
        ax.plot([bx, bx], [by, by + uV * HALF / VMAX], "-", color="k", lw=2.2, zorder=11)
        ax.text(bx - 18, by + 0.5 * uV * HALF / VMAX, f"{uV:,.0f} µV", fontsize=11,
                rotation=90, va="center", ha="right", zorder=11)
    ax.set_aspect("equal")
    ax.set_xlim(cell_x.min() - 320, xr + 470)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("층을 따라가는 방향 (µm)", fontsize=12)
    ax.set_ylabel("층 깊이 -s  (위 = SO / SP / SR / SLM = 아래, µm)", fontsize=12)
    ax.tick_params(labelsize=10)

    hs = [Line2D([], [], marker="o", ls="", mfc="white", mec=COL[v], mew=2.4, ms=10,
                 label=f"{v} {sum(1 for k in range(NEL) if verd[k]==v)}개")
          for v in ["정상", "표본부족", "집단스파이크", "전류원(+)", "흐름꼬리", "자극전극"]
          if any(verd[k] == v for k in range(NEL))]
    hs += [Line2D([], [], color=C_SINK, lw=8, alpha=0.45, label="음(-) = 전류 싱크"),
           Line2D([], [], color=C_SRC, lw=8, alpha=0.45, label="양(+) = 전류원 source"),
           Line2D([], [], color="#b0182a", lw=1.6, ls="--", label=f"SC 자극 층대 ±{r_stim:.0f} µm"),
           Line2D([], [], marker="o", ls="", color="0.25", ms=5, label="억제세포 소마")]
    ax.legend(handles=hs, fontsize=10.5, loc="lower right", ncol=2, framealpha=0.94)

    n_syn = int(G("n_syn", 0)); n_sc = int(G("n_sc", 0)); n_sccell = int(G("n_sccell", 0))
    scale = ("전극마다 자기 최대로 정규화 — **모양·타이밍·극성** 비교용, 크기 비교 불가"
             if norm else
             f"24전극 **공통 축척**(반높이 {HALF:.0f} µm = {VMAX:,.0f} µV) — 크기 비교가 정직하다")
    fig.suptitle(
        f"슬라이스 위 실제 전극 자리의 fEPSP 24파형 — {tag} · 자극세기 {lv[LI]*100:.0f}%"
        f"(SC 섬유 {na[LI]:.0f}/200 발화) · 자극 후 0~{DUR:.0f} ms\n"
        f"{scale}", fontsize=15.5, y=0.982)
    fig.text(0.012, 0.012,
             f"규모 세포 {len(xyz):,} (PC {int((t4=='PC').sum()):,}) · 내부 연결 {n_syn:,} "
             f"DetAMPANMDA/DetGABAAB **결정론** · SC 자극 경로 {n_sc:,} GBPlasticitySyn **결정론**"
             f"(SC 받은 세포 {n_sccell:,}) · 전극 3×8 간격 200 µm 회전 0° · 자극전극 "
             f"#{stim_elec}({el_layer[stim_elec]}, s={s_el_ref[stim_elec]:+.0f} µm) 층대 반경 {r_stim:.0f} µm\n"
             f"정확: 전극 좌표 = npz E 그대로 · 세포 좌표 = slice_cells.npz 를 시뮬과 같은 SVD 로 투영"
             f"(저장된 s_el 24개와 대조, 최대 오차 {err:.1e} µm) · 층 색 = 세포의 층 라벨 · 파형 = 원파형"
             f"(자극 전 5 ms 평균만 뺌)\n"
             f"관례: 파형의 µV→µm 축척 · 층 깊이축을 세로로 세운 화면 회전(강체 회전이라 왜곡 아님) · "
             f"층 경계선 = 인접 층 세포분포의 (99 %, 1 %) 중간.  근사: 2D 투영(층중심의 두께축 퍼짐 "
             f"{spreads[i_thick]:.0f} µm) · 소마 점만(수상돌기 미표시)",
             fontsize=9.4, va="bottom", color="0.28")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.935, bottom=0.105)
    out = os.path.join(FIG, f"MEA_{tag}_slice_{'norm' if norm else 'common'}.png")
    fig.savefig(out, dpi=155)
    plt.close(fig)
    print("  저장:", out)


draw(False)
draw(True)
print(f"[요약] 세기 {lv[LI]*100:.0f}% · 섬유 {na[LI]:.0f}/200 · 판정 "
      + " · ".join(f"{v} {sum(1 for k in range(NEL) if verd[k]==v)}개"
                   for v in ["정상", "표본부족", "집단스파이크", "전류원(+)", "흐름꼬리", "자극전극"]
                   if any(verd[k] == v for k in range(NEL))))
