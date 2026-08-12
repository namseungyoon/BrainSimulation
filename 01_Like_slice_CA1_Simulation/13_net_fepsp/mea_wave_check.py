# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_wave_check.py — 대표 전극 파형에 **집단스파이크가 섞였는가**를 가린다.

왜 필요한가 (2026-08-08)
  1단계 통과 기준 ②를 "유발 스파이크 0개" -> "파형에 집단스파이크 없음"으로 바꾸면서,
  판정이 `mea_postproc.measure_fepsp` 의 `rev_frac`(되돌림 비율) 하나에 얹혔다.
  그런데 `rev_frac` 은 **피크 이전 구간만** 본다(mea_postproc.py 의 `_seg = vv[:ipk+1]`).

      되돌림 = (그 시점 값 - 그때까지의 최저값) / |그때까지의 최저값| 의 최대

  이 정의는 "fEPSP 위에 집단스파이크가 **얹혀서** 파형이 두 성분으로 꺾이는" 경우를
  잡는다. 하지만 집단스파이크가 fEPSP보다 **더 깊게 꽂혀 그 자체가 최저점이 되면**
  하강이 단조로워 되돌림이 0이 나온다. **사각지대다.**

  전규모 전극 #18 은 최대 세기(섬유 160 · 유발 스파이크 4,017)에서도 되돌림 0.0% 다.
  진폭은 -17.7 mV. 이게 정말 순수 fEPSP인지, 사각지대에 빠진 것인지 가려야 한다.

어떻게 가리나 — **모양 비교**가 결정적이다
  섬유 2개는 유발 스파이크가 **0개**다. 즉 그 파형은 집단스파이크가 섞일 수 없는
  **순수 fEPSP 기준자**다. 세기를 올렸을 때 파형이 단순히 **커지기만** 하면(모양 동일)
  새 성분이 안 생긴 것이고, **모양이 달라지면** 무언가 얹힌 것이다.

      각 세기 파형을 기준자에 최소제곱으로 스케일 맞춤 -> 잔차 RMS / 진폭 = 모양 차이

  모양 차이가 세기와 함께 커지면 = 새 성분(집단스파이크)이 자라는 것.
  거의 일정하면 = 같은 파형이 배율만 커진 것 = 순수 fEPSP.

같이 보는 것
  - 피크 **이후** 되돌림(사각지대 정의) — 피크 이전만 보는 rev_frac 과 짝으로 낸다
  - 상승시간(20~80%)과 반치폭 — 집단스파이크가 섞이면 상승이 급해지고 폭이 좁아진다

실행: <ca1sim>/python.exe 13_net_fepsp/mea_wave_check.py [tag] [--merge <태그>] [--elec 18]
출력: figures/MEA_<out>_wave_check.png  ·  figures/<out>_waveshape.csv
"""
import os
import sys
import argparse
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
from mea_postproc import g, write_csv, measure_fepsp, POP_REV_FRAC   # noqa: E402
from mea_io_pick import load_merged                                   # noqa: E402


def shape_metrics(tt, vv, ref):
    """기준자 파형 `ref` 에 대한 **모양** 차이와, 파형 자체의 모양 지표들.

    스케일은 최소제곱으로 맞춘다 — 크기 차이가 아니라 **모양** 차이만 보기 위해서다.
        a* = argmin_a ||vv - a*ref||^2 = <vv,ref>/<ref,ref>
    """
    out = {}
    amp = float(vv.min())
    out["amp"] = amp
    ipk = int(np.argmin(vv))
    out["tpk"] = float(tt[ipk])

    # ── 모양 차이 (기준자 대비) ───────────────────────────────────────────────
    denom = float(ref @ ref)
    a = float(vv @ ref) / denom if denom > 0 else 0.0
    resid = vv - a * ref
    out["scale"] = a
    out["shape_rms"] = float(np.sqrt((resid ** 2).mean()) / abs(amp)) if amp else float("nan")

    # ── 피크 **이후** 되돌림 — rev_frac 의 사각지대를 메우는 짝 지표 ──────────
    #   피크를 지나 위로 올라갔다가 **다시 내려가는** 성분이 있으면 두 번째 성분이다.
    post = vv[ipk:]
    if post.size > 2:
        run = np.maximum.accumulate(post)          # 피크 후 그때까지의 최고(=0에 가까운 쪽)
        redip = (run - post) / abs(amp)
        out["post_redip"] = float(redip.max())
    else:
        out["post_redip"] = 0.0

    # ── 상승 20~80% 시간 · 반치폭 ─────────────────────────────────────────────
    def _cross(level):
        seg = vv[:ipk + 1]
        idx = np.where(seg <= level)[0]
        if idx.size == 0 or idx[0] == 0:
            return float(tt[0])
        kk = int(idx[0])
        v0, v1 = float(seg[kk - 1]), float(seg[kk])
        fr = (level - v0) / (v1 - v0) if v1 != v0 else 0.0
        return float(tt[kk - 1] + fr * (tt[kk] - tt[kk - 1]))

    out["t_rise_20_80"] = _cross(0.8 * amp) - _cross(0.2 * amp)
    half = vv <= 0.5 * amp
    out["fwhm"] = float(tt[half].max() - tt[half].min()) if half.any() else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description="대표 전극 파형의 집단스파이크 오염 검사")
    ap.add_argument("tag", nargs="?", default="S1_io_gb")
    ap.add_argument("--merge", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--elec", type=int, default=18)
    ap.add_argument("--dur", type=float, default=30.0)
    args = ap.parse_args()

    tags = [args.tag] + [t.strip() for t in args.merge.split(",") if t.strip()]
    out_tag = args.out or (args.tag if len(tags) == 1 else f"{args.tag}_all")
    d, note = load_merged(tags)
    if d is None:
        return 2
    if note:
        print(note)

    lv = np.asarray(d["levels"], float)
    na = np.asarray(d["nact"], float)
    nspk = np.asarray(d["nspk"], float)
    waves = np.asarray(d["waves"], float)
    twin = np.asarray(d["twin"], float)
    stim_t = float(g(d, "stim_t", 100.0))
    n_pc = int(g(d, "n_pc", 0))
    j = args.elec

    m = (twin >= stim_t) & (twin < stim_t + args.dur)
    tt = twin[m] - stim_t
    V = np.array([waves[i, j][m] - waves[i, j][m][0] for i in range(len(lv))])

    # 기준자 = **유발 스파이크가 0개인 가장 약한 세기**. 집단스파이크가 섞일 수 없다.
    silent = np.where(nspk <= 0)[0]
    if silent.size == 0:
        print("★기준자 없음 — 유발 스파이크 0인 세기가 하나도 없습니다. "
              "이 검사는 '스파이크 0 파형'을 기준으로 삼으므로 성립하지 않습니다.")
        return 2
    iref = int(silent[0])
    ref = V[iref]
    print(f"\n[기준자] 세기 {100*lv[iref]:.1f}%({na[iref]:.0f}섬유) · 유발 스파이크 "
          f"{nspk[iref]:.0f}개 · 진폭 {ref.min():.1f} µV — 집단스파이크가 섞일 수 없는 순수 fEPSP")
    print(f"[전극] #{j} · 측정창 {args.dur:.0f} ms · 표본 {len(tt)}개")

    fes = [measure_fepsp(twin, waves[i, j], stim_t, args.dur, 5.0) for i in range(len(lv))]
    rows, M = [], []
    print(f"\n{'세기%':>6} {'섬유':>5} {'유발스파이크':>11} {'발화율%PC':>10} "
          f"{'진폭µV':>11} {'피크ms':>7} {'상승20-80ms':>12} {'반치폭ms':>9} "
          f"{'되돌림%(피크전)':>15} {'되돌림%(피크후)':>15} {'모양차이%':>10}")
    for i in range(len(lv)):
        s = shape_metrics(tt, V[i], ref)
        M.append(s)
        rev_pre = 100 * fes[i]["rev_frac"]
        frac_pc = 100.0 * nspk[i] / n_pc if n_pc else float("nan")
        print(f"{100*lv[i]:>6.1f} {na[i]:>5.0f} {nspk[i]:>11.0f} {frac_pc:>10.3f} "
              f"{s['amp']:>11.1f} {s['tpk']:>7.2f} {s['t_rise_20_80']:>12.3f} "
              f"{s['fwhm']:>9.2f} {rev_pre:>15.1f} {100*s['post_redip']:>15.1f} "
              f"{100*s['shape_rms']:>10.2f}")
        rows.append([f"{lv[i]:.4f}", f"{na[i]:.0f}", f"{nspk[i]:.0f}", f"{frac_pc:.4f}",
                     f"{s['amp']:.4f}", f"{s['tpk']:.3f}", f"{s['t_rise_20_80']:.4f}",
                     f"{s['fwhm']:.3f}", f"{fes[i]['rev_frac']:.4f}",
                     f"{s['post_redip']:.4f}", f"{s['shape_rms']:.4f}",
                     f"{s['scale']:.4f}", j])

    sh = np.array([s["shape_rms"] for s in M])
    pr = np.array([s["post_redip"] for s in M])
    print(f"\n[판정] 모양차이는 기준자({100*lv[iref]:.1f}%) 대비 잔차 RMS / 진폭.")
    print(f"  · 모양차이 {100*sh.min():.2f}% ~ {100*sh.max():.2f}%  "
          f"(세기 8배 이상 늘려도 이 정도면 '배율만 커진 같은 파형')")
    print(f"  · 피크 후 되돌림 최대 {100*pr.max():.1f}% (문턱 {100*POP_REV_FRAC:.0f}%)")
    grow = bool(len(sh) >= 3 and sh[-1] > 3.0 * max(sh[1], 1e-9))
    print(f"  · 세기와 함께 모양이 커지는가: {'예 ★오염 의심' if grow else '아니오'}")

    # ── 그림 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.0, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    cols = plt.cm.viridis(np.linspace(0.08, 0.92, len(lv)))

    # (A) 원래 크기 — 약한 세기는 안 보인다. 그래서 (B)가 필요하다는 것을 보여준다
    axA = fig.add_subplot(gs[0, 0])
    for i in range(len(lv)):
        axA.plot(tt, V[i], color=cols[i], lw=1.4,
                 label=f"{100*lv[i]:.1f}% ({na[i]:.0f}섬유)")
    axA.axhline(0, color="0.6", lw=0.8)
    axA.set_xlabel("자극 후 시간 (ms)"); axA.set_ylabel("Ve - 기준선 (µV)")
    axA.set_title(f"(A) 전극 #{j} 세기별 파형 — 실제 크기", fontsize=10)
    axA.legend(fontsize=6.8, ncol=2); axA.grid(alpha=0.3)

    # (B) ★각자 자기 최저값으로 나눔 — 모양만 남는다. 겹치면 새 성분이 없다는 뜻
    axB = fig.add_subplot(gs[0, 1])
    for i in range(len(lv)):
        axB.plot(tt, V[i] / abs(V[i].min()), color=cols[i], lw=1.5,
                 label=f"{100*lv[i]:.1f}%")
    axB.axhline(0, color="0.6", lw=0.8); axB.axhline(-1, color="0.85", lw=0.8, ls=":")
    axB.set_xlabel("자극 후 시간 (ms)"); axB.set_ylabel("자기 최저값으로 나눈 값")
    axB.set_title("(B) ★모양만 — 겹치면 '배율만 커진 같은 파형'\n"
                  "갈라지면 새 성분(집단스파이크)이 자란 것", fontsize=10)
    axB.legend(fontsize=6.8, ncol=2); axB.grid(alpha=0.3); axB.set_xlim(-0.5, 12)

    # (C) 최대 세기 파형 + 누적최저 — rev_frac 이 무엇을 재는지, 왜 0인지
    axC = fig.add_subplot(gs[0, 2])
    vlast = V[-1]
    ipk = int(np.argmin(vlast))
    axC.plot(tt, vlast, color="k", lw=1.8, label=f"{100*lv[-1]:.0f}% ({na[-1]:.0f}섬유)")
    axC.plot(tt[:ipk + 1], np.minimum.accumulate(vlast[:ipk + 1]), color="#8e44ad",
             lw=1.2, ls="--", label="그때까지의 최저 (되돌림 기준선)")
    axC.plot(tt[ipk:], np.maximum.accumulate(vlast[ipk:]), color="#e67e22",
             lw=1.2, ls="--", label="피크 후 최고 (사각지대 기준선)")
    axC.axvline(tt[ipk], color="0.6", lw=0.8, ls=":")
    axC.axhline(0, color="0.6", lw=0.8)
    axC.set_xlabel("자극 후 시간 (ms)"); axC.set_ylabel("Ve - 기준선 (µV)")
    axC.set_title(f"(C) 되돌림은 무엇을 재나 — 최대 세기\n"
                  f"피크전 {100*fes[-1]['rev_frac']:.1f}% · "
                  f"피크후 {100*M[-1]['post_redip']:.1f}%", fontsize=10)
    axC.legend(fontsize=7); axC.grid(alpha=0.3); axC.set_xlim(-0.5, 15)

    # (D) 약한 3점만 크게 — "안 보인다"가 아니었음을 보인다
    axD = fig.add_subplot(gs[1, 0])
    nw = min(3, len(lv))
    for i in range(nw):
        axD.plot(tt, V[i], color=cols[i], lw=1.8,
                 label=f"{100*lv[i]:.1f}% ({na[i]:.0f}섬유) · "
                       f"{V[i].min():.1f} µV · 스파이크 {nspk[i]:.0f}")
    axD.axhline(0, color="0.6", lw=0.8)
    axD.set_xlabel("자극 후 시간 (ms)"); axD.set_ylabel("Ve - 기준선 (µV)")
    axD.set_title("(D) 가장 약한 세기들 — 확대", fontsize=10)
    axD.legend(fontsize=7.5); axD.grid(alpha=0.3); axD.set_xlim(-0.5, 15)

    # (E) 모양 차이 vs 세기 — 자라면 오염
    axE = fig.add_subplot(gs[1, 1])
    axE.plot(na, 100 * sh, "o-", color="#c0392b", lw=1.6, ms=7, label="모양 차이 (잔차/진폭)")
    axE.plot(na, 100 * pr, "s--", color="#e67e22", lw=1.4, ms=6, label="피크 후 되돌림")
    axE.axhline(100 * POP_REV_FRAC, color="#8e44ad", lw=1.2, ls=":",
                label=f"집단스파이크 문턱 {100*POP_REV_FRAC:.0f}%")
    axE.set_xlabel("활성 SC 섬유 수"); axE.set_ylabel("%")
    axE.set_title("(E) ★세기를 올려도 모양이 그대로인가", fontsize=10)
    axE.legend(fontsize=7.5); axE.grid(alpha=0.3)

    # (F) 유발 스파이크 — 파형이 깨끗해도 이건 별개 문제다
    axF = fig.add_subplot(gs[1, 2])
    axF.bar(np.arange(len(lv)), nspk, color="#7f8c8d")
    for i, y in enumerate(nspk):
        axF.annotate(f"{y:.0f}", (i, y), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8)
    axF.set_xticks(np.arange(len(lv)))
    axF.set_xticklabels([f"{100*x:.1f}%\n{n:.0f}섬유" for x, n in zip(lv, na)], fontsize=7)
    axF.set_ylabel("유발 스파이크 수")
    axF.set_title(f"(F) 유발 스파이크 — PC {n_pc:,}개 대비\n"
                  f"파형이 깨끗한 것과 '자극이 시냅스를 바꾸는가'는 다른 문제", fontsize=10)
    axF.grid(alpha=0.3, axis="y")

    fig.suptitle(f"전극 #{j} 파형에 집단스파이크가 섞였는가 — 기준자 {100*lv[iref]:.1f}%"
                 f"(스파이크 0) 대비 모양 비교 · {out_tag}", fontsize=12.5, y=0.985)
    png = os.path.join(FIG, f"MEA_{out_tag}_wave_check.png")
    fig.savefig(png, dpi=145, bbox_inches="tight")
    print(f"saved: {png}")
    write_csv(os.path.join(FIG, f"{out_tag}_waveshape.csv"),
              ["level", "n_fiber_active", "n_spike", "spike_per_PC_pct",
               "amp_uV", "peak_time_ms", "rise_20_80_ms", "fwhm_ms",
               "rev_frac_pre_peak", "rev_frac_post_peak", "shape_rms_frac",
               "best_scale_vs_ref", "elec"], rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
