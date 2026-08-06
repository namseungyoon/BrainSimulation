# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_io_pick.py — 1단계 판정: 자극세기-반응 곡선에서 **테스트 세기**를 고른다.

계획 1단계(자극세기 정하기)의 통과 기준 4개를 코드로 판정하고, 2·3단계에 쓸
`--io_test` 값을 확정한다. NEURON이 필요 없다(결과 npz만 읽는다).

  통과 기준 (계획 5절 1단계)
    (1) 곡선이 S자          — 단조 증가 + 저세기 발끝 + 고세기 포화, Hill 적합 R^2
    (2) 선택 지점 유발 스파이크 0개 — 재는 행위가 대상을 바꾸면 안 된다
    (3) 선택 지점이 가파른 구간 안  — 적합 곡선 기울기가 최대의 50% 이상
    (4) fEPSP가 음(-) 방향  — SR층은 전류가 빨려 들어가는 곳(싱크)

  ★고르는 규칙: **실제로 측정한 레벨 중에서만** 고른다. 측정 안 한 세기는 그 지점의
    유발 스파이크 수를 모르므로 기준 (2)를 확인할 방법이 없다.

실행: <ca1sim>/py 13_net_fepsp/mea_io_pick.py [tag]      (기본 tag = S1_io_gb)
출력: figures/S1_io_levels.csv · figures/MEA_<tag>.png · 판정 로그
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)
from mea_postproc import g, write_csv, cfg_str, measure_fepsp, SLOPE_METHOD  # noqa: E402

LAY_COL = {"SO": "#2980b9", "SP": "#c0392b", "SR": "#27ae60", "SLM": "#8e44ad"}
STEEP_FRAC = 0.50        # 적합 곡선 기울기가 최대의 몇 배 이상이어야 "가파른 구간"인가
SAT_FRAC = 0.85          # 최대 응답의 몇 배를 넘으면 "천장에 붙었다"고 보는가


# ══════════════════════════════════════════════════════════════════════════════
# Hill(시그모이드) 적합 — S자인지 판정하고 가파른 구간을 찾는 데만 쓴다
# ══════════════════════════════════════════════════════════════════════════════
def hill(n, smax, n50, h):
    """S(n) = smax * n^h / (n50^h + n^h).  h > 1 이면 저세기에 '발끝'이 생긴다(S자)."""
    n = np.maximum(np.asarray(n, float), 1e-9)
    return smax * n ** h / (n50 ** h + n ** h)


def fit_hill(x, y):
    """실패해도 죽지 않는다 — 적합이 안 되면 (None, nan)을 돌려주고 판정에서 걸러진다."""
    p0 = [max(y.max(), 1e-9), float(np.median(x)), 2.0]
    lo = [1e-12, 1e-6, 0.2]
    hi = [y.max() * 20 + 1e-9, x.max() * 20, 20.0]
    try:
        p, _ = curve_fit(hill, x, y, p0=p0, bounds=(lo, hi), maxfev=200000)
    except Exception as e:
        print(f"[적합 실패] {e}")
        return None, float("nan")
    resid = y - hill(x, *p)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return p, r2


def main():
    ap = argparse.ArgumentParser(description="1단계 I-O 판정 + 테스트 세기 확정")
    ap.add_argument("tag", nargs="?", default="S1_io_gb")
    ap.add_argument("--dur", type=float, default=30.0, help="측정창 길이(ms)")
    ap.add_argument("--pre", type=float, default=5.0, help="기준선 구간(ms)")
    ap.add_argument("--steep", type=float, default=STEEP_FRAC)
    args = ap.parse_args()

    f = os.path.join(FIG, f"_mea_{args.tag}.npz")
    if not os.path.exists(f):
        print(f"★없음: {f}")
        return 2
    d = np.load(f, allow_pickle=True)
    kind = str(g(d, "kind", "?"))
    if kind != "io":
        print(f"[경고] kind='{kind}' — 이 스크립트는 io 결과를 대상으로 합니다")
        return 2

    lv = np.asarray(d["levels"], float)
    na = np.asarray(d["nact"], float)
    slope = np.asarray(d["slope"], float)         # 부호 있는 값(음이어야 정상)
    amp = np.asarray(d["amp"], float)
    nspk = np.asarray(d["nspk"], float)
    pk = np.asarray(d["pk_abs"], float)           # 창내 |Ve| 최대 지점의 **부호 있는** 값
    rec_j = int(g(d, "rec_j", 0))
    n_fiber = int(g(d, "n_fiber", 200))
    stim_t = float(g(d, "stim_t", 100.0))
    rec_dt = float(g(d, "rec_dt", 0.4))

    # ── 실행 조건 ─────────────────────────────────────────────────────────────
    #    ★방출 모드는 **두 줄**로 적는다 — 내부 커넥톰과 SC 자극 경로가 서로 다른 모델을 쓴다.
    print("=" * 84)
    print(f"[파일] {os.path.basename(f)}  ·  {cfg_str(d)}")
    print(f"[규모] 세포 {int(g(d,'N',0)):,}(PC {int(g(d,'n_pc',0)):,}) / 전체 {int(g(d,'n_tot',0)):,}")
    print(f"[방출1] 내부 연결 {int(g(d,'n_syn',0)):,}개 — DetAMPANMDA/DetGABAAB · "
          f"{'결정론(룰베이스)' if bool(g(d,'det',True)) else '확률(BBP EMS Random123)'}")
    print(f"[방출2] SC 자극 경로 {int(g(d,'n_sc',0)):,}개(세포 {int(g(d,'n_sccell',0)):,}개) — "
          f"모델 {str(g(d,'syn_model','?'))} · "
          f"{'확률(소포단위 MVR)' if bool(g(d,'syn_prob',False)) else '결정론'}")
    print(f"[전극] 자극 #{int(g(d,'stim_elec',0))}({str(g(d,'stim_layer','?'))}층) · "
          f"기록 #{rec_j} · 기록전극 {len(np.atleast_1d(g(d,'rec_idx',[])))}개 · "
          f"전극당 유효세포 Neff {float(g(d,'neff',0)):.0f} · 신호90% 반경 {float(g(d,'r90',0)):.0f}um")
    print("=" * 84)

    # ── ★기울기 재계산 (판정의 기준) ──────────────────────────────────────────
    #   npz의 slope는 **그 런이 돌던 시점의 정의**로 계산된 값이다. 정의를 바꾸면
    #   옛 파일과 새 파일이 뒤섞인다. 그래서 판정은 항상 저장된 **파형에서 다시 재서** 한다.
    #   ※ waves는 자극 시각부터 잘려 있어 자극 전 구간이 없다. 기준선은 창의 첫 표본이
    #      된다 — 배경 구동이 없는 조용한 슬라이스라 자극 전 Ve가 0 근처이므로 타당하다.
    waves = np.asarray(d["waves"], float)              # (레벨, 전극, 창내시간)
    twin = np.asarray(d["twin"], float)
    fes = [measure_fepsp(twin, waves[i, rec_j], stim_t, args.dur, args.pre)
           for i in range(len(lv))]
    slope_re = np.array([f["slope"] for f in fes])
    amp_re = np.array([f["amp"] for f in fes])
    nband = np.array([f["n_band"] for f in fes])
    s_leg = np.array([f["slope_legacy"] for f in fes])
    S = np.abs(slope_re)
    print(f"[기울기 정의] {SLOPE_METHOD}(교차시각 20~80%) 로 파형에서 재계산 · "
          f"측정창 {args.dur:.0f}ms · 기록간격 {rec_dt:g}ms")
    if np.max(np.abs(slope_re - slope)) > 1e-6 * max(np.abs(slope).max(), 1.0):
        print(f"   ※ 파일에 저장된 기울기와 다릅니다(정의가 바뀜). 저장 "
              f"{np.array2string(np.abs(slope), precision=4)} → 재계산 "
              f"{np.array2string(S, precision=4)}")
        print(f"   ※ 옛 정의(표본회귀)로 재현하면 "
              f"{np.array2string(np.abs(s_leg), precision=4)} · "
              f"20~80% 띠에 든 표본 수 {nband.tolist()} (2개 미만이면 옛 정의가 무너진다)")
    amp = amp_re
    slope = slope_re

    # ── 레벨별 표 ─────────────────────────────────────────────────────────────
    print(f"{'세기%':>6} {'활성섬유':>8} {'|slope|µV/ms':>13} {'진폭µV':>10} "
          f"{'창내최대Ve µV':>14} {'방향':>5} {'유발스파이크':>11}")
    rows = []
    for i in range(len(lv)):
        direc = "음(-)" if pk[i] < 0 else "양(+)"
        print(f"{100*lv[i]:>6.1f} {na[i]:>8.0f} {S[i]:>13.4f} {amp[i]:>10.4f} "
              f"{pk[i]:>14.4f} {direc:>5} {nspk[i]:>11.0f}")
        rows.append([f"{lv[i]:.4f}", f"{na[i]:.0f}", f"{slope[i]:.6f}", f"{S[i]:.6f}",
                     f"{amp[i]:.6f}", f"{pk[i]:.6f}", direc, f"{nspk[i]:.0f}",
                     f"{s_leg[i]:.6f}", f"{nband[i]:d}"])

    # ── Hill 적합 ─────────────────────────────────────────────────────────────
    p, r2 = fit_hill(na, S)
    if p is not None:
        smax, n50, hh = p
        print(f"\n[Hill 적합] smax {smax:.4f} µV/ms · 반응반값 세기 n50 {n50:.1f}섬유"
              f"({100*n50/n_fiber:.1f}%) · 기울기지수 h {hh:.2f} · R^2 {r2:.4f}")
        ng = np.linspace(max(na.min() * 0.2, 1e-6), na.max() * 1.05, 2000)
        dS = np.gradient(hill(ng, *p), ng)
        dmax = float(dS.max())
        steep_lo = float(ng[dS >= args.steep * dmax].min())
        steep_hi = float(ng[dS >= args.steep * dmax].max())
        print(f"[가파른 구간] 적합 기울기가 최대의 {100*args.steep:.0f}% 이상 = "
              f"{steep_lo:.0f}~{steep_hi:.0f}섬유 ({100*steep_lo/n_fiber:.1f}~{100*steep_hi/n_fiber:.1f}%)")
    else:
        smax = n50 = hh = float("nan")
        ng = dS = None
        steep_lo = steep_hi = float("nan")

    # ── 통과 기준 판정 ────────────────────────────────────────────────────────
    print("\n" + "-" * 84)
    ok = {}
    # (1) S자 — 단조 증가 + 포화 + 적합 양호
    mono = bool(np.all(np.diff(S) > -1e-12))
    d1 = np.diff(S) / np.diff(na)                      # 구간별 기울기
    satur = bool(len(d1) >= 2 and d1[-1] < 0.5 * d1.max())
    ok["S자"] = mono and satur and (r2 == r2 and r2 > 0.95)
    print(f"(1) S자 곡선          : 단조증가 {'예' if mono else '아니오'} · "
          f"고세기 포화 {'예' if satur else '아니오'}(끝 구간 기울기 = 최대의 "
          f"{100*d1[-1]/d1.max() if len(d1) and d1.max()>0 else float('nan'):.0f}%) · "
          f"Hill R^2 {r2:.4f} -> {'통과' if ok['S자'] else '미달'}")
    # (4) 음(-) 방향 — 전 레벨
    neg_all = bool(np.all(pk < 0))
    ok["음방향"] = neg_all
    print(f"(4) fEPSP 음(-) 방향  : {int((pk<0).sum())}/{len(pk)}레벨 음 -> "
          f"{'통과' if neg_all else '미달'}")

    # ── 테스트 세기 고르기 ────────────────────────────────────────────────────
    silent = nspk <= 0
    unsat = S < SAT_FRAC * (smax if smax == smax else S.max())
    inside = np.array([steep_lo <= x <= steep_hi for x in na]) if p is not None \
        else np.ones_like(na, bool)
    cand = silent & unsat & inside
    print(f"\n[후보] 유발스파이크 0 {int(silent.sum())}개 · 천장 전({100*SAT_FRAC:.0f}% 미만) "
          f"{int(unsat.sum())}개 · 가파른 구간 안 {int(inside.sum())}개 -> 교집합 "
          f"{int(cand.sum())}개 {[f'{100*lv[i]:.0f}%' for i in np.where(cand)[0]]}")

    if cand.any():
        # 가파른 한가운데(n50)에 가장 가까운 것
        k = int(np.where(cand)[0][np.argmin(np.abs(na[cand] - n50))]) if n50 == n50 \
            else int(np.where(cand)[0][-1])
        ok["스파이크0"] = True
        ok["가파른구간"] = True
    else:
        k = int(np.argmax(np.where(silent, S, -np.inf))) if silent.any() else int(np.argmin(S))
        ok["스파이크0"] = bool(silent.any())
        ok["가파른구간"] = False
    pick = float(lv[k])
    print(f"(2) 선택 지점 스파이크 0 : 세기 {100*pick:.0f}%({na[k]:.0f}섬유)에서 "
          f"{nspk[k]:.0f}개 -> {'통과' if nspk[k] == 0 else '미달'}")
    print(f"(3) 선택 지점이 가파른 구간 안 : "
          f"{'예' if ok['가파른구간'] else '아니오'} -> {'통과' if ok['가파른구간'] else '미달'}")
    allok = all(ok.values()) and nspk[k] == 0
    print("-" * 84)
    print(f"★확정 테스트 세기: **{100*pick:.0f}%** = 섬유 {na[k]:.0f}/{n_fiber}개 "
          f"· |slope| {S[k]:.4f} µV/ms · 진폭 {amp[k]:.3f} µV")
    print(f"   2·3단계 실행:  IO_TEST={pick:g} bash _wsl_stage.sh 2 <모델>")
    print(f"★1단계 통과 기준 4개: {'전부 통과' if allok else '미달 있음 — 다음 단계로 가지 않는다'}")
    print("-" * 84)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv = os.path.join(FIG, f"{args.tag}_levels.csv")
    write_csv(csv, ["level", "n_fiber_active", "slope_uV_per_ms", "abs_slope",
                    "amp_uV", "peak_Ve_uV", "direction", "n_spike",
                    "slope_legacy_uV_per_ms", "n_band_samples"], rows)

    # ── 그림 ─────────────────────────────────────────────────────────────────
    E = d["E"]; over = d["over"]
    el_layer = d["el_layer"].astype(str) if "el_layer" in d.files else None
    stim_elec = int(g(d, "stim_elec", 0))

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    # (A) I-O 곡선 + Hill 적합 + 선택 지점
    axA = fig.add_subplot(gs[0, 0])
    if p is not None:
        axA.plot(ng, hill(ng, *p), "-", color="#c0392b", lw=1.6, alpha=0.75,
                 label=f"Hill 적합 R$^2$={r2:.3f}\nn50={n50:.0f}섬유 h={hh:.2f}")
        axA.axvspan(steep_lo, steep_hi, color="#f39c12", alpha=0.13,
                    label=f"가파른 구간(기울기 {100*args.steep:.0f}%+)")
    axA.plot(na, S, "o", color="#c0392b", ms=8, zorder=5, label="측정")
    axA.plot([na[k]], [S[k]], "*", color="k", ms=20, zorder=6,
             label=f"확정 테스트 세기 {100*pick:.0f}%")
    axA.set_xlabel("자극세기 — 활성 SC 섬유 수 (총 200)")
    axA.set_ylabel("fEPSP |기울기| (µV/ms)")
    axA.set_title(f"(A) 자극세기-반응 곡선 · 기록전극 #{rec_j}", fontsize=10)
    axA.grid(alpha=0.3); axA.legend(fontsize=7.5, loc="upper left")

    # (B) 유발 스파이크 — '재는 자가 대상을 바꾸지 않는가'
    axB = fig.add_subplot(gs[0, 1])
    axB.bar(na, nspk, width=max(na.max() * 0.045, 1.0), color="#7f8c8d")
    axB.bar([na[k]], [nspk[k]], width=max(na.max() * 0.045, 1.0), color="#27ae60")
    for x, y in zip(na, nspk):
        axB.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=8)
    axB.set_xlabel("활성 SC 섬유 수"); axB.set_ylabel("유발 스파이크 수")
    axB.set_title("(B) 세기별 유발 스파이크 — 테스트 세기는 0이어야 한다", fontsize=10)
    axB.grid(alpha=0.3, axis="y")

    # (C) 전극 배치
    axC = fig.add_subplot(gs[0, 2])
    if el_layer is not None:
        for Ln, col in LAY_COL.items():
            m = (el_layer == Ln) & over
            if m.any():
                axC.scatter(E[m, 0], E[m, 1], s=55, c=col, edgecolors="0.3", label=Ln, alpha=0.85)
    axC.scatter(E[stim_elec, 0], E[stim_elec, 1], s=200, marker="*", facecolor="none",
                edgecolors="k", linewidths=1.6, label="자극", zorder=6)
    axC.scatter(E[rec_j, 0], E[rec_j, 1], s=150, marker="s", facecolor="none",
                edgecolors="k", linewidths=1.6, label="기록", zorder=6)
    axC.set_aspect("equal"); axC.legend(fontsize=7, loc="upper right", ncol=2)
    axC.set_title(f"(C) 전극 3x8 배치(층별) · 자극#{stim_elec} 기록#{rec_j}", fontsize=10)
    axC.set_xlabel("면 가로 µm", fontsize=9); axC.tick_params(labelsize=8)

    # (D) 세기별 기록전극 파형 — 음(-) 방향 확인
    axD = fig.add_subplot(gs[1, 0])
    cols = plt.cm.viridis(np.linspace(0.1, 0.92, len(lv)))
    for i in range(len(lv)):
        axD.plot(twin - stim_t, waves[i, rec_j], color=cols[i], lw=1.4,
                 label=f"{100*lv[i]:.0f}% ({na[i]:.0f}섬유)")
    axD.axhline(0, color="0.6", lw=0.8)
    axD.set_xlabel("자극 후 시간 (ms)"); axD.set_ylabel("Ve (µV)")
    axD.set_title(f"(D) 세기별 fEPSP 파형 · 기록전극 #{rec_j}\n아래로 내려가야(음) 정상", fontsize=10)
    axD.legend(fontsize=7.5); axD.grid(alpha=0.3)

    # (E) 최대 세기에서 전극 24개 전부
    axE = fig.add_subplot(gs[1, 1])
    for j in range(waves.shape[1]):
        c = LAY_COL.get(el_layer[j], "0.6") if el_layer is not None else "0.6"
        axE.plot(twin - stim_t, waves[-1, j], color=c, lw=0.9, alpha=0.75)
    axE.axhline(0, color="0.6", lw=0.8)
    axE.set_xlabel("자극 후 시간 (ms)"); axE.set_ylabel("Ve (µV)")
    axE.set_title(f"(E) 최대 세기({100*lv[-1]:.0f}%)에서 전극 {waves.shape[1]}개 전부 · 층별 색", fontsize=10)
    axE.grid(alpha=0.3)

    # (F) 전극별 창내 최대 |Ve| — 자극전극 거리 대비
    axF = fig.add_subplot(gs[1, 2])
    dist = np.linalg.norm(E - E[stim_elec], axis=1)
    pkall = np.array([waves[-1, j][np.argmax(np.abs(waves[-1, j]))] for j in range(waves.shape[1])])
    for Ln, col in LAY_COL.items():
        m = (el_layer == Ln) if el_layer is not None else np.zeros(len(dist), bool)
        if m.any():
            axF.scatter(dist[m], pkall[m], s=48, c=col, edgecolors="0.3", label=Ln, alpha=0.9)
    axF.axhline(0, color="0.6", lw=0.8)
    axF.set_xlabel("자극전극으로부터 거리 (µm)"); axF.set_ylabel("창내 최대 Ve (µV, 부호 포함)")
    axF.set_title(f"(F) 최대 세기 · 전극별 응답 vs 거리", fontsize=10)
    axF.legend(fontsize=7.5); axF.grid(alpha=0.3)

    n_sc = int(g(d, "n_sc", 0)); n_syn = int(g(d, "n_syn", 0)); n_cell = int(g(d, "N", 0))
    fig.suptitle(
        f"1단계 자극세기-반응 곡선 — 세포 {n_cell:,} · 내부시냅스 {n_syn:,}(결정론) · "
        f"SC {n_sc:,}(모델 {str(g(d,'syn_model','?'))}) · 확정 테스트 세기 {100*pick:.0f}%",
        fontsize=12.5, y=0.985)
    png = os.path.join(FIG, f"MEA_{args.tag}.png")
    fig.savefig(png, dpi=145, bbox_inches="tight")
    print(f"saved: {png}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
