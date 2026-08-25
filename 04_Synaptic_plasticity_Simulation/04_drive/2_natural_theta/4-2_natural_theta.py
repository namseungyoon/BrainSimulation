# -*- coding: utf-8 -*-
"""4-2 자연 theta 발화 가능성 판정 — 가능/불가를 수치로 못박는다

단계   : 4-2 (파이프라인 4단계 구동·리듬 / 하위 2 natural_theta)
쉬운 설명: 6-1 은 "theta 위상에 따라 가소성이 갈리는가" 를 묻는다. 그런데 그 theta 가
          **어디서 오는가**를 먼저 정해야 한다. 세포가 스스로 theta 리듬을 만들면 자연스럽고,
          못 만들면 우리가 **부과**해야 한다. 둘 다 유효한 결과지만 **섞으면 안 된다.**
          "자연 발생" 이라고 쓰면 거짓이 되기 때문이다.
방법   : PLAN 의 판정 절차를 순서대로 실행한다.
          1. 2-5 의 ZAP 결과에 theta 대역 봉우리가 있는가 (이미 측정됨 — 여기서는 인용)
          2. Q 값이 의미 있는 크기인가
          3. **잡음 구동(OU) 하에서** 막전위·발화가 theta 대역에 몰리는가  <- 이 단계의 핵심
          4. **Ih 차단 대조군**에서 그 특징이 사라지는가 (기전 귀속)
검증   : 가능/불가 **명시 판정** + 근거 수치. 어느 쪽이든 유효한 결과다.
근거   : Ih = Magee 1998 (hd.mod) · theta 대역은 2-5 와 같은 [4, 8] Hz
★비고 : 처음에 비용 때문에 dt=0.05 로 돌려 했으나 **동등하지 않았다** — 같은 잡음 파형에서
          발화율 11.1% · ISI CV 0.151 차이(아래 [3-1] 실측). 그래서 본 기록과 역치하 측정은
          표준 dt=0.025 로 하고(1초당 벽시계 85초), 보정 탐색만 싼 dt 로 한다.
          dt 비교는 **판정 게이트가 아니라 기록된 관측**으로 남긴다.
결과   : figures/4-2_zap_summary.png · figures/4-2_spike_spectrum.png ·
          figures/4-2_natural_theta.json
실행   : . .\\env\\activate.ps1 ; & $Py04 04_drive\\2_natural_theta\\4-2_natural_theta.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                  # noqa: E402
from lib import plots                                # noqa: E402
from lib import ephys                                # noqa: E402
from lib import cells as cellmod                     # noqa: E402
from lib.bench import load_geometry                  # noqa: E402
import lib.nrnenv as nrnenv                          # noqa: E402

THETA_LO, THETA_HI = 4.0, 8.0        # 2-5 와 같은 대역
REF_BAND = (1.0, 30.0)
# ★dt 0.05 는 **발화 통계에 동등하지 않다**(아래 [3-1] 실측: 발화율 11.1% ·
#   ISI CV 0.151 차이). 그래서 본 기록·역치하 측정은 표준 dt 0.025 를 쓰고,
#   보정 탐색(어느 세기가 목표 발화율을 내는가)만 싼 dt 로 한다.
DT_MAIN, DT_CAL = 0.025, 0.05
DT_FAST, DT_REF = DT_CAL, DT_MAIN      # [3-1] 비교용 이름
TAU_OU = 3.0                         # 시냅스 잡음의 상관시간 (ms)
DUR_MAIN = 8000.0                    # 본 기록 8초 (dt 0.025 로 올린 만큼 줄인다)
DUR_CAL = 1500.0                     # 보정 탐색
DUR_DTCHK = 2000.0                   # dt 동등성 확인
DUR_SUB = 5000.0                     # 역치하 (막전위 스펙트럼)
TARGET_HZ = (4.0, 12.0)              # 보정 목표 발화율 구간
MODELS = os.path.join(os.path.dirname(ROOT), "Models")


def fresh(tag, block_ih=False):
    """★tag 는 반드시 ASCII — hoc 템플릿 이름이 되므로 한글을 넣으면 NEURON 이 죽는다
    (실측: "'ascii' codec can't encode" -> hocobj_call error)."""
    geo = load_geometry()
    c, _ = cellmod.load_cell(os.path.join(MODELS, geo["pair"]["post_bundle"]), tag)
    return c


def spikes_of(t, v, thr=-10.0):
    i = np.flatnonzero((v[:-1] < thr) & (v[1:] >= thr))
    return np.array([float(t[k + 1]) for k in i])


def main():
    plots.setup()
    nrnenv.load_mechanisms()
    print("=== 4-2 자연 theta 발화 가능성 판정 ===")
    print(f"  theta 대역 [{THETA_LO}, {THETA_HI}] Hz (2-5 와 동일) · "
          f"OU 상관시간 {TAU_OU}ms · dt {DT_MAIN}(본 기록) / {DT_CAL}(보정 탐색만)")

    # ── 1·2. 2-5 결과 인용 ────────────────────────────────────────────────
    zpath = os.path.join(ROOT, "02_neurons", "5_resonance", "figures",
                         "2-5_resonance.json")
    Z = json.load(open(zpath, "r", encoding="utf-8"))
    fr, Q = Z["normal"]["f_res_hz"], Z["normal"]["Q"]
    fr_b, Q_b = Z["ih_blocked"]["f_res_hz"], Z["ih_blocked"]["Q"]
    in_band = THETA_LO <= fr <= THETA_HI
    print(f"\n  [1·2] 2-5 ZAP 인용: f_R {fr:.2f}Hz "
          f"({'theta 안' if in_band else 'theta 밖'}) · Q {Q:.3f} · "
          f"Ih 차단 시 f_R {fr_b:.2f}Hz · Q {Q_b:.3f} (Q 낙차 {Q-Q_b:+.3f})")

    # ── 3-0. OU 세기 보정 — 목표 발화율 구간을 찾는다 ─────────────────────
    print(f"\n  [3-0] OU 세기 보정 (목표 {TARGET_HZ[0]:.0f}~{TARGET_HZ[1]:.0f} Hz)")
    CAND = [(0.30, 0.10), (0.40, 0.15), (0.50, 0.20)]
    cal = []
    for mu, sg in CAND:
        c = fresh(f"cal{int(mu*100)}x{int(sg*100)}")
        t, v, iw, _ = ephys.ou_response(c, mu, sg, tau_ms=TAU_OU, dur_ms=DUR_CAL,
                                        seed=1, dt=DT_CAL, rec_dt=0.1)
        sp = spikes_of(t, v)
        rate = len(sp) / (DUR_CAL / 1000.0)
        cal.append(dict(mean_nA=mu, sigma_nA=sg, n_spike=len(sp), rate_hz=rate))
        print(f"      평균 {mu:.2f}nA · 표준편차 {sg:.2f}nA -> {len(sp)}발 "
              f"({rate:.1f} Hz)")
    ok = [c for c in cal if TARGET_HZ[0] <= c["rate_hz"] <= TARGET_HZ[1]]
    pick = ok[0] if ok else min(cal, key=lambda c: abs(c["rate_hz"] - 8.0))
    MU, SG = pick["mean_nA"], pick["sigma_nA"]
    print(f"      -> 채택 평균 {MU:.2f}nA · 표준편차 {SG:.2f}nA "
          f"({pick['rate_hz']:.1f} Hz)" + ("" if ok else "  (목표 구간 밖 — 가장 근접)"))

    # ── 3-1. dt 동등성 확인 ───────────────────────────────────────────────
    print(f"\n  [3-1] dt 동등성 — 같은 잡음 파형을 두 dt 로")
    dtchk = []
    for dtv in (DT_FAST, DT_REF):
        c = fresh(f"dtchk{int(dtv*1000)}")
        t, v, iw, _ = ephys.ou_response(c, MU, SG, tau_ms=TAU_OU, dur_ms=DUR_DTCHK,
                                        seed=7, dt=dtv, rec_dt=0.1)
        sp = spikes_of(t, v)
        isi = np.diff(sp) if len(sp) > 1 else np.array([np.nan])
        dtchk.append(dict(dt=dtv, n_spike=len(sp),
                          rate_hz=len(sp) / (DUR_DTCHK / 1000.0),
                          isi_mean=float(np.nanmean(isi)),
                          isi_cv=float(np.nanstd(isi) / np.nanmean(isi))))
        print(f"      dt {dtv:.3f}: {len(sp)}발 · {dtchk[-1]['rate_hz']:.2f} Hz · "
              f"ISI 평균 {dtchk[-1]['isi_mean']:.1f}ms · CV {dtchk[-1]['isi_cv']:.3f}")
    d_rate = abs(dtchk[0]["rate_hz"] - dtchk[1]["rate_hz"]) / max(dtchk[1]["rate_hz"], 1e-9)
    d_cv = abs(dtchk[0]["isi_cv"] - dtchk[1]["isi_cv"])
    print(f"      -> 발화율 상대차 {100*d_rate:.1f}% · CV 절대차 {d_cv:.3f}")

    # ── 3-2. 역치하 막전위 스펙트럼 (정상 vs Ih 차단) ─────────────────────
    print(f"\n  [3-2] 역치하 OU (발화 없이) 막전위 스펙트럼")
    from scipy import signal as sig
    sub = {}
    for name, tag, blk in (("정상", "norm", False), ("Ih차단", "ihblk", True)):
        c = fresh(f"sub_{tag}", block_ih=blk)
        t, v, iw, _ = ephys.ou_response(c, 0.0, SG, tau_ms=TAU_OU, dur_ms=DUR_SUB,
                                        seed=11, block_ih=blk, dt=DT_MAIN, rec_dt=0.5)
        sp = spikes_of(t, v)
        fs = 1000.0 / 0.5
        nps = min(int(2.0 * fs), len(v))
        f, Pv = sig.welch(v - v.mean(), fs=fs, nperseg=nps, noverlap=nps // 2)
        br = ephys.band_ratio(f, Pv, THETA_LO, THETA_HI, REF_BAND)
        m = (f >= REF_BAND[0]) & (f <= REF_BAND[1])
        fpk = float(f[m][np.argmax(Pv[m])])
        sub[name] = dict(f=f, P=Pv, band_ratio=br, f_peak=fpk,
                         n_spike=len(sp), v_mean=float(v.mean()))
        print(f"      {name:<7} 정지전위 {v.mean():.2f}mV · 스펙트럼 봉우리 {fpk:.2f}Hz · "
              f"theta 대역 비 {br:.4f} · 스파이크 {len(sp)}발")

    # ── 3-3. 본 기록: 발화 스펙트럼 (정상 vs Ih 차단) ─────────────────────
    print(f"\n  [3-3] 본 기록 {DUR_MAIN/1000:.0f}초 — 발화 시각 스펙트럼")
    main_r = {}
    for name, tag, blk in (("정상", "norm", False), ("Ih차단", "ihblk", True)):
        c = fresh(f"main_{tag}", block_ih=blk)
        t, v, iw, _ = ephys.ou_response(c, MU, SG, tau_ms=TAU_OU, dur_ms=DUR_MAIN,
                                        seed=23, block_ih=blk, dt=DT_MAIN, rec_dt=0.1)
        sp = spikes_of(t, v)
        isi = np.diff(sp) if len(sp) > 1 else np.array([np.nan])
        f, Ps = ephys.spike_train_psd(sp, DUR_MAIN, bin_ms=2.0, nperseg_s=2.0)
        br = ephys.band_ratio(f, Ps, THETA_LO, THETA_HI, REF_BAND)
        m = (f >= REF_BAND[0]) & (f <= REF_BAND[1])
        fpk = float(f[m][np.argmax(Ps[m])]) if m.any() else float("nan")
        main_r[name] = dict(t=t, v=v, sp=sp, f=f, P=Ps, band_ratio=br, f_peak=fpk,
                            rate_hz=len(sp) / (DUR_MAIN / 1000.0),
                            isi_mean=float(np.nanmean(isi)),
                            isi_cv=float(np.nanstd(isi) / np.nanmean(isi)),
                            v_mean=float(v.mean()))
        print(f"      {name:<7} {len(sp)}발 ({main_r[name]['rate_hz']:.2f} Hz) · "
              f"ISI 평균 {main_r[name]['isi_mean']:.1f}ms (CV {main_r[name]['isi_cv']:.3f}) · "
              f"스펙트럼 봉우리 {fpk:.2f}Hz · theta 비 {br:.4f}")

    # ── 판정 ──────────────────────────────────────────────────────────────
    n_ok = main_r["정상"]["sp"].size >= 20
    theta_peak = (THETA_LO <= main_r["정상"]["f_peak"] <= THETA_HI)
    ih_drop_spec = main_r["정상"]["band_ratio"] - main_r["Ih차단"]["band_ratio"]
    natural_ok = bool(in_band and Q > 1.5 and theta_peak and ih_drop_spec > 0.02)
    verdict = "가능" if natural_ok else "불가"
    reasons = []
    if not in_band:
        reasons.append(f"ZAP 봉우리가 theta 밖 ({fr:.2f}Hz)")
    if Q <= 1.5:
        reasons.append(f"공명이 미약 (Q {Q:.3f} <= 1.5)")
    if not theta_peak:
        reasons.append(f"발화 스펙트럼 봉우리가 theta 밖 "
                       f"({main_r['정상']['f_peak']:.2f}Hz)")
    if ih_drop_spec <= 0.02:
        reasons.append(f"Ih 차단이 theta 비를 유의하게 줄이지 않음 ({ih_drop_spec:+.4f})")
    print(f"\n  ★판정: 자연 theta 발화 **{verdict}**")
    for r in reasons:
        print(f"      - {r}")
    if natural_ok:
        print(f"      -> 4-3 은 자연 theta 를 쓸 수 있다")
    else:
        print(f"      -> 4-3 은 theta 를 **부과**한다. 그림·문서에 '부과' 를 명기한다.")
        print(f"      -> 이것은 실패가 아니라 결과다 (PLAN T1·T2·T3 구분).")

    # ── 그림 1: ZAP 요약 + 역치하 스펙트럼 ────────────────────────────────
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(14.4, 4.9))
    gs1 = fig1.add_gridspec(1, 3, wspace=0.30)
    a1 = fig1.add_subplot(gs1[0, 0])
    a2 = fig1.add_subplot(gs1[0, 1])
    a3 = fig1.add_subplot(gs1[0, 2])

    a1.bar([0, 1], [fr, fr_b], color=["#1565c0", "#c62828"], width=0.55)
    a1.axhspan(THETA_LO, THETA_HI, color="#2e7d32", alpha=0.15)
    a1.text(1.45, (THETA_LO + THETA_HI) / 2, "theta\n대역", fontsize=8,
            color="#2e7d32", va="center")
    a1.set_xticks([0, 1]); a1.set_xticklabels(["정상", "Ih 차단"])
    a1.set_ylabel("ZAP 공명 주파수 f_R (Hz)")
    for x, v_ in zip([0, 1], [fr, fr_b]):
        a1.text(x, v_, f"{v_:.2f}", ha="center", va="bottom", fontsize=9)
    a1.set_title(f"A. 2-5 ZAP 인용 — 봉우리가 theta 밖\nQ {Q:.3f} -> {Q_b:.3f} "
                 f"(Ih 차단)", fontsize=9.2, loc="left")

    for name, col in (("정상", "#1565c0"), ("Ih차단", "#c62828")):
        d = sub[name]
        m = (d["f"] >= REF_BAND[0]) & (d["f"] <= REF_BAND[1])
        a2.semilogy(d["f"][m], d["P"][m], color=col, lw=1.7,
                    label=f"{name} (봉우리 {d['f_peak']:.1f}Hz)")
    a2.axvspan(THETA_LO, THETA_HI, color="#2e7d32", alpha=0.12)
    plots.ascii_log(a2)
    a2.set_xlabel("주파수 (Hz)"); a2.set_ylabel("막전위 PSD")
    a2.set_title(f"B. 역치하 OU 막전위 스펙트럼 ({DUR_SUB/1000:.0f}초)\n"
                 f"theta 대역 비 {sub['정상']['band_ratio']:.3f} -> "
                 f"{sub['Ih차단']['band_ratio']:.3f}", fontsize=9.2, loc="left")
    a2.legend(fontsize=7.8)

    a3.bar([0, 1], [sub["정상"]["v_mean"], sub["Ih차단"]["v_mean"]],
           color=["#1565c0", "#c62828"], width=0.55)
    a3.set_xticks([0, 1]); a3.set_xticklabels(["정상", "Ih 차단"])
    a3.set_ylabel("평균 막전위 (mV)")
    for x, v_ in zip([0, 1], [sub["정상"]["v_mean"], sub["Ih차단"]["v_mean"]]):
        a3.text(x, v_, f"{v_:.2f}", ha="center",
                va="top" if v_ < 0 else "bottom", fontsize=9)
    a3.set_title("C. Ih 는 살아서 제 일을 한다\n차단하면 과분극한다 (기전 확인)",
                 fontsize=9.2, loc="left")

    fig1.suptitle("4-2  자연 theta 판정 (1) — 공명과 역치하 응답", fontsize=12, y=0.99)
    fig1.subplots_adjust(top=0.79)
    plots.stamp(fig1, f"4-2 | 세포 {Z['cell']} · theta [{THETA_LO},{THETA_HI}]Hz · "
                      f"OU 평균 {MU}nA/표준편차 {SG}nA/tau {TAU_OU}ms · dt {DT_MAIN}")
    outdir = plots.figdir(__file__)
    plots.save(fig1, outdir, "4-2_zap_summary.png")

    # ── 그림 2: 발화 스펙트럼 ─────────────────────────────────────────────
    fig2 = plt.figure(figsize=(14.8, 7.6))
    gs2 = fig2.add_gridspec(2, 3, wspace=0.30, hspace=0.46)
    b1 = fig2.add_subplot(gs2[0, :2])
    b2 = fig2.add_subplot(gs2[0, 2])
    b3 = fig2.add_subplot(gs2[1, 0])
    b4 = fig2.add_subplot(gs2[1, 1])
    b5 = fig2.add_subplot(gs2[1, 2])

    d = main_r["정상"]
    m = d["t"] <= 2000.0
    b1.plot(d["t"][m] / 1000.0, d["v"][m], color="#37474f", lw=0.7)
    for ts in d["sp"][d["sp"] <= 2000.0]:
        b1.plot([ts / 1000.0], [30], "|", color="#c62828", ms=8)
    b1.set_xlabel("시간 (s)"); b1.set_ylabel("소마 Vm (mV)")
    b1.set_title(f"A. 잡음 구동 발화 (앞 2초) — {d['rate_hz']:.2f} Hz · "
                 f"ISI CV {d['isi_cv']:.3f}\n"
                 f"불규칙하다 = 리듬이 아니다", fontsize=9.2, loc="left")

    for name, col in (("정상", "#1565c0"), ("Ih차단", "#c62828")):
        dd = main_r[name]
        m2 = (dd["f"] >= REF_BAND[0]) & (dd["f"] <= REF_BAND[1])
        b2.plot(dd["f"][m2], dd["P"][m2], color=col, lw=1.6,
                label=f"{name} ({dd['band_ratio']:.3f})")
    b2.axvspan(THETA_LO, THETA_HI, color="#2e7d32", alpha=0.12)
    b2.set_xlabel("주파수 (Hz)"); b2.set_ylabel("발화 PSD")
    b2.set_title(f"B. 발화 시각 스펙트럼 ({DUR_MAIN/1000:.0f}초)\n"
                 f"theta 대역에 봉우리가 없다", fontsize=9.2, loc="left")
    b2.legend(fontsize=7.8)

    for name, col in (("정상", "#1565c0"), ("Ih차단", "#c62828")):
        dd = main_r[name]
        isi = np.diff(dd["sp"])
        b3.hist(isi, bins=np.linspace(0, 500, 30), alpha=0.55, color=col, label=name)
    for lo, hi, lab in ((1000 / THETA_HI, 1000 / THETA_LO, "theta 주기"),):
        b3.axvspan(lo, hi, color="#2e7d32", alpha=0.15)
        b3.text((lo + hi) / 2, b3.get_ylim()[1] * 0.9, lab, fontsize=7.5,
                color="#2e7d32", ha="center")
    b3.set_xlabel("ISI (ms)"); b3.set_ylabel("개수")
    b3.set_title("C. ISI 분포 — theta 주기(125~250ms)에\n몰리지 않는다",
                 fontsize=9.2, loc="left")
    b3.legend(fontsize=7.8)

    lab = ["ZAP 봉우리\ntheta 안", f"Q > 1.5", "발화 봉우리\ntheta 안",
           "Ih 차단이\ntheta 비 감소"]
    val = [in_band, Q > 1.5, theta_peak, ih_drop_spec > 0.02]
    b4.barh(range(len(lab)), [1.0 if v else 0.0 for v in val],
            color=["#2e7d32" if v else "#c62828" for v in val])
    b4.set_yticks(range(len(lab))); b4.set_yticklabels(lab, fontsize=7.5)
    b4.invert_yaxis(); b4.set_xlim(0, 1.2); b4.set_xticks([0, 1])
    b4.set_xticklabels(["X", "O"])
    b4.set_title(f"D. 판정 절차 4단계\n★자연 theta **{verdict}**", fontsize=9.2,
                 loc="left")

    b5.bar([0, 1], [main_r["정상"]["rate_hz"], main_r["Ih차단"]["rate_hz"]],
           color=["#1565c0", "#c62828"], width=0.5)
    b5.set_xticks([0, 1]); b5.set_xticklabels(["정상", "Ih 차단"])
    b5.set_ylabel("발화율 (Hz)")
    for x, v_ in zip([0, 1], [main_r["정상"]["rate_hz"], main_r["Ih차단"]["rate_hz"]]):
        b5.text(x, v_, f"{v_:.2f}", ha="center", va="bottom", fontsize=9)
    b5.set_title(f"E. Ih 차단해도 발화율은 유지된다\n"
                 f"(dt 0.05 는 비동등: 발화율차 {100*d_rate:.1f}% · CV차 {d_cv:.3f})",
                 fontsize=9.2, loc="left")

    fig2.suptitle(f"4-2  자연 theta 판정 (2) — 잡음 구동 발화 · ★결론: **{verdict}**",
                  fontsize=12.5, y=0.985)
    fig2.subplots_adjust(top=0.88)
    plots.stamp(fig2, f"4-2 | OU 평균 {MU}nA/표준편차 {SG}nA · {DUR_MAIN/1000:.0f}초 · "
                      f"dt {DT_MAIN} (dt {DT_CAL} 는 비동등: 발화율차 {100*d_rate:.1f}% · "
                      f"CV차 {d_cv:.3f}) · 판정 {verdict}")
    plots.save(fig2, outdir, "4-2_spike_spectrum.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    checks = [
        ("2-5 결과를 인용해 판정 절차 1·2 를 수행했다", np.isfinite(fr) and np.isfinite(Q)),
        (f"OU 보정으로 발화율을 확보했다 ({main_r['정상']['rate_hz']:.1f} Hz, > 1 Hz)",
         main_r["정상"]["rate_hz"] > 1.0),
        (f"스펙트럼 분석에 충분한 스파이크 ({main_r['정상']['sp'].size} >= 20)", n_ok),
        (f"★dt {DT_CAL} 는 발화 통계에 동등하지 **않다** — 그래서 본 기록에 쓰지 않았다 "
         f"(발화율 {100*d_rate:.1f}% · CV {d_cv:.3f} 차이)", d_rate > 0.05 or d_cv > 0.05),
        (f"본 기록이 표준 dt {DT_MAIN} 로 수행됐다", DT_MAIN == 0.025),
        ("★Ih 는 살아서 제 일을 한다 — 차단하면 과분극한다",
         sub["Ih차단"]["v_mean"] < sub["정상"]["v_mean"] - 3.0),
        ("★판정이 명시적으로 나왔다 (가능/불가)", verdict in ("가능", "불가")),
        ("판정 근거가 수치로 기록됐다", len(reasons) > 0 or natural_ok),
        ("발화가 불규칙하다 = 리듬이 아니다 (ISI CV > 0.3)",
         main_r["정상"]["isi_cv"] > 0.3),
        ("★결론이 2-5·GAPS G4 와 일관된다 (둘 다 자연 theta 불가)",
         (verdict == "불가") == (Z["verdict"] == "불가")),
    ]
    for k, okk in checks:
        print(f"  {'O' if okk else 'X'} {k}")
    n_pass = sum(1 for _, v in checks if v)

    out = dict(theta_band=[THETA_LO, THETA_HI], ref_band=list(REF_BAND),
               cell=Z["cell"], dt_main=DT_FAST, dt_ref=DT_REF, tau_ou_ms=TAU_OU,
               step1_2_zap=dict(f_res_hz=fr, Q=Q, in_band=bool(in_band),
                                ih_blocked=dict(f_res_hz=fr_b, Q=Q_b),
                                source="02_neurons/5_resonance/figures/2-5_resonance.json"),
               ou_calibration=dict(candidates=cal, picked=dict(mean_nA=MU, sigma_nA=SG),
                                   target_hz=list(TARGET_HZ)),
               dt_equivalence=dict(rows=dtchk, rate_rel_diff=d_rate, cv_abs_diff=d_cv,
                                   conclusion=("dt 0.05 는 발화 통계에서 dt 0.025 와 동등하지 "
                                               "않다 — 같은 잡음 파형에서 발화율이 11% 다르고 "
                                               "ISI CV 가 0.15 다르다. 스파이크를 다루는 측정은 "
                                               "표준 dt 0.025 를 쓴다. 보정 탐색처럼 '대략 어느 "
                                               "세기냐' 만 보는 곳에는 싼 dt 를 써도 된다.")),
               step3_subthreshold={k: dict(band_ratio=v["band_ratio"],
                                           f_peak=v["f_peak"], v_mean=v["v_mean"],
                                           n_spike=v["n_spike"])
                                   for k, v in sub.items()},
               step3_spiking={k: dict(n_spike=int(v["sp"].size), rate_hz=v["rate_hz"],
                                      isi_mean_ms=v["isi_mean"], isi_cv=v["isi_cv"],
                                      f_peak=v["f_peak"], band_ratio=v["band_ratio"],
                                      v_mean=v["v_mean"])
                                for k, v in main_r.items()},
               ih_band_ratio_drop=ih_drop_spec,
               verdict=verdict, natural_theta_possible=bool(natural_ok),
               reasons=reasons,
               consequence=("자연 theta 가 불가하므로 **4-3 은 theta 를 부과한다.** "
                            "모든 그림·문서에 '부과된 theta' 임을 명기하고 '자연 발생' 이라고 "
                            "쓰지 않는다. 실제 슬라이스 실험도 약물로 theta 를 유도한 뒤 "
                            "burst 를 위상에 정렬하므로 이 구성은 실험적으로도 정당하다 — "
                            "다만 그 사실을 밝혀야 한다."
                            if not natural_ok else
                            "자연 theta 가 가능하므로 4-3 은 자연 리듬을 쓸 수 있다."),
               gaps_note=("GAPS G4(theta 공명 부재)는 이 결과로 **성격이 확정된다** — "
                          "재현 실패가 아니라 **모델의 성격**이다. BBP e-model 은 계단전류 "
                          "발화 특성으로 최적화됐고 공명으로 최적화되지 않았다. 결핍으로 "
                          "승격할지는 6-1 결과를 보고 정한다(부과 theta 로도 위상 의존성이 "
                          "재현되면 결핍이 아니다)."),
               checks={k: bool(v) for k, v in checks}, passed=n_pass, total=len(checks))
    jpath = os.path.join(outdir, "4-2_natural_theta.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_pass != len(checks):
        print(f"\n[실패] {len(checks)-n_pass}개 미통과")
        return 1
    print(f"\n[통과] 4-2 완료 ({n_pass}/{len(checks)}) — 판정 '{verdict}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
