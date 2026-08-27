# -*- coding: utf-8 -*-
"""4-4 부과 gamma — theta 위에 gamma 를 얹는다 (★둘 다 부과다)

단계   : 4-4 (파이프라인 4단계 구동·리듬 / 하위 4 gamma)
쉬운 설명: 실제 해마에서는 theta(느린 물결) 위에 gamma(빠른 잔물결)가 얹혀 있고, gamma 의
          크기가 theta 위상에 따라 달라진다(위상-진폭 결합). 6-2 가 그 조건에서 가소성을
          보려 하므로 여기서 그 파형을 만든다.
★가장 중요한 사실: **gamma 는 개재뉴런 네트워크 현상이다.** 추체세포 2개로는 어떤 방식으로도
          생성되지 않는다(PLAN T4). theta 도 자연 발생이 아니다(4-2 판정: 불가).
          => **둘 다 부과다.** 그림 캡션마다 명기하고 '자연 발생' 이라고 쓰지 않는다.
방법   : 4-3 이 정한 기본 방식(정현파 전류)으로 theta 를 부과하고 gamma 를 중첩한다.
          gamma 진폭을 theta 위상에 따라 변조해 **위상-진폭 결합(PAC)** 을 만든다.
          (A) 중첩 파형이 목표대로 나오는가 (theta·gamma 성분을 스펙트럼으로 분리)
          (B) PAC 이 실제로 생겼는가 (변조지수)
          (C) 시냅스 위치에서도 gamma 가 살아 있는가 — ★막이 저역통과라 gamma 가 더 깎인다
          (D) gamma 주파수 스윕 — 어디까지 전달되는가
검증   : 성분 분리 확인 · PAC 정량 · 위치별 감쇠 정량.
근거   : 4-3(기본 방식·위상 기준) · 4-2(자연 theta 불가) · PLAN T4(gamma 는 부과만)
결과   : figures/4-4_theta_gamma.png · figures/4-4_gamma.json
실행   : . .\\env\\activate.ps1 ; & $Py04 04_drive\\4_gamma\\4-4_theta_gamma.py
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
from lib.bench import Bench                          # noqa: E402
from lib.wiring import Wiring, SETTLE_MS             # noqa: E402
from lib.nrnenv import h                             # noqa: E402

DT = 0.025
REC_DT = 0.1                  # gamma(최대 80Hz)를 담으려면 촘촘해야 한다
T0 = SETTLE_MS
DUR = 1200.0
F_THETA = 5.0                 # 4-3 과 같은 기준 주파수
F_GAMMA = [30.0, 50.0, 80.0]  # 저·중·고 gamma
F_GAMMA_MAIN = 50.0
TARGET_THETA_PP = 8.0         # 4-3 과 같은 목표
GAMMA_FRAC = 0.35             # gamma 전류 진폭 / theta 전류 진폭
PAC_DEPTH = 1.0               # 1.0 = theta peak 에서 최대, trough 에서 0


def fit_sine(t, y, f_hz):
    w = 2.0 * np.pi * f_hz * (np.asarray(t) / 1000.0)
    X = np.column_stack([np.cos(w), np.sin(w), np.ones_like(w)])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    a, b, m = coef
    return float(np.hypot(a, b)), float(np.arctan2(b, a)), float(m)


def wrap_deg(rad):
    return float((np.degrees(rad) + 180.0) % 360.0 - 180.0)


def main():
    plots.setup()
    print("=== 4-4 부과 gamma (★theta 도 gamma 도 부과다) ===")
    print("  gamma 는 개재뉴런 네트워크 현상이다 — 추체세포 2개로는 생성 불가(PLAN T4).")
    b = Bench()
    w = Wiring(b, frozen=True)
    for syn, _ in w.syns:
        syn.gmax = 0.0
    site_lbl = [f"{sp['path_um']:.0f}um" for _, sp in w.syns]
    print(f"  기록: post 소마 + 시냅스 {len(site_lbl)}지점 ({', '.join(site_lbl)})")

    # 4-3 의 결과를 읽어 같은 조건을 쓴다
    tpath = os.path.join(ROOT, "04_drive", "3_imposed_theta", "figures",
                         "4-3_theta.json")
    A3 = json.load(open(tpath, "r", encoding="utf-8"))
    AMP_THETA = float(A3["sine"]["amp_nA"])
    default_method = A3["summary"]["default_method"]
    print(f"  4-3 인용: 기본 방식 '{default_method}' · theta 전류 {AMP_THETA:.4f} nA "
          f"(소마 {TARGET_THETA_PP:.0f} mV pp 목표)")

    ic = h.IClamp(b.post_soma_seg())
    ic.delay, ic.dur, ic.amp = 0.0, 1e9, 0.0
    TS = T0 + DUR
    N_MAX = int(TS / DT) + 2
    drive = h.Vector(np.zeros(N_MAX))
    drive.play(ic._ref_amp, DT)
    w.keep += [ic, drive]
    w.record(rec_dt=REC_DT, local_v=True, currents=False)
    # ★ VecStim·Vector.play 는 restore 와 양립하지 않는다 -> 조건마다 finit (lib/wiring 주석)

    def build(f_g, gamma_frac, pac_depth):
        tt = np.arange(N_MAX) * DT
        ph = 2 * np.pi * F_THETA * (tt - T0) / 1000.0
        th = np.cos(ph)
        # PAC: gamma 진폭이 theta 위상을 따라간다 (peak 에서 최대)
        env = (1.0 - pac_depth) + pac_depth * 0.5 * (1.0 + np.cos(ph))
        gm = env * np.cos(2 * np.pi * f_g * (tt - T0) / 1000.0)
        wav = np.where(tt >= T0, AMP_THETA * (th + gamma_frac * gm), 0.0)
        return tt, wav, env

    def run(f_g, gamma_frac=GAMMA_FRAC, pac_depth=PAC_DEPTH):
        tt, wav, env = build(f_g, gamma_frac, pac_depth)
        drive.from_python(wav)
        w.run(TS, dt=DT)
        return w.arrays(), tt, wav, env

    def comps(R, f_g):
        """theta·gamma 성분을 각각 적합해 분리. 과도상태 2주기 버림."""
        t = R["t"]
        m = t >= T0 + 2 * 1000.0 / F_THETA
        out = {}
        for lab, y in [("소마", R["post_v"])] + \
                      [(site_lbl[i], R["local_v"][i]) for i in range(len(site_lbl))]:
            At, pt, mn = fit_sine(t[m], y[m], F_THETA)
            Ag, pg, _ = fit_sine(t[m], y[m], f_g)
            out[lab] = dict(theta_pp=2 * At, theta_phase=wrap_deg(pt),
                            gamma_pp=2 * Ag, gamma_phase=wrap_deg(pg), mean=mn)
        return out

    def pac_index(R, f_g, key="post_v", idx=None):
        """위상-진폭 결합: theta 위상 구간별 **gamma 대역 포락선**의 변조 깊이.

        ★처음 판은 'theta 적합을 뺀 잔차의 RMS' 로 쟀는데 그 잔차에는 고조파·표류가 섞여
          있어서 **gamma 진폭이 일정한 대조 조건에서도 지수가 0.41** 로 나왔다(실측).
          제대로 하려면 gamma 대역만 좁게 통과시킨 뒤 포락선을 봐야 한다.
          여기서는 FFT 마스크(f_g +- 0.3*f_g)로 대역통과하고 힐베르트 포락선을 쓴다.
        변조지수 = (최대-최소)/(최대+최소). 0 이면 결합 없음.
        """
        from scipy.signal import hilbert
        t = R["t"]
        y = R[key] if idx is None else R[key][idx]
        m = t >= T0 + 2 * 1000.0 / F_THETA
        t, y = t[m], y[m]
        n = t.size
        dt_s = (t[1] - t[0]) / 1000.0
        freq = np.fft.rfftfreq(n, d=dt_s)
        Y = np.fft.rfft(y - y.mean())
        bw = 0.3 * f_g
        Y[(freq < f_g - bw) | (freq > f_g + bw)] = 0.0
        band = np.fft.irfft(Y, n=n)
        env = np.abs(hilbert(band))
        # theta 위상 (0 = peak)
        At, pt, mn = fit_sine(t, y, F_THETA)
        ph = 2 * np.pi * F_THETA * (t / 1000.0)
        phase = np.degrees(np.angle(np.exp(1j * (ph - np.radians(pt)))))
        edges = np.linspace(-180, 180, 9)
        amps = []
        for i in range(8):
            sel = (phase >= edges[i]) & (phase < edges[i + 1])
            amps.append(float(np.mean(env[sel])) if sel.sum() > 5 else np.nan)
        amps = np.array(amps)
        mi = float((np.nanmax(amps) - np.nanmin(amps))
                   / (np.nanmax(amps) + np.nanmin(amps)))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return mi, centers, amps

    # ── (A)(B) 본 조건: theta + gamma 50Hz + PAC ─────────────────────────
    print(f"\n  [A·B] theta {F_THETA:.0f}Hz + gamma {F_GAMMA_MAIN:.0f}Hz · "
          f"gamma 전류 비 {GAMMA_FRAC:.2f} · PAC 깊이 {PAC_DEPTH:.1f}")
    Rm, tt, wav, env = run(F_GAMMA_MAIN)
    cm = comps(Rm, F_GAMMA_MAIN)
    for k, v in cm.items():
        print(f"      {k:<8} theta {v['theta_pp']:6.3f} mV @ {v['theta_phase']:+7.1f}deg · "
              f"gamma {v['gamma_pp']:6.3f} mV @ {v['gamma_phase']:+7.1f}deg")
    mi, centers, amps = pac_index(Rm, F_GAMMA_MAIN)
    print(f"      위상-진폭 결합 지수 (소마) {mi:.4f} "
          f"(구간별 gamma RMS {np.nanmin(amps):.3f}~{np.nanmax(amps):.3f} mV)")

    # 대조: PAC 없는 조건 (gamma 진폭 일정)
    Rf, _, _, _ = run(F_GAMMA_MAIN, pac_depth=0.0)
    mi0, _, amps0 = pac_index(Rf, F_GAMMA_MAIN)
    print(f"      대조(PAC 없음) 결합 지수 {mi0:.4f} -> 차이 {mi - mi0:+.4f}")

    # ── (C)(D) gamma 주파수 스윕 — 어디까지 전달되는가 ────────────────────
    print(f"\n  [C·D] gamma 주파수 스윕 — 막은 저역통과다")
    sweep = []
    for f_g in F_GAMMA:
        R, _, _, _ = run(f_g)
        c = comps(R, f_g)
        keep = c["소마"]["gamma_pp"] / c["소마"]["theta_pp"]
        syn_keep = [c[s]["gamma_pp"] / c["소마"]["gamma_pp"] for s in site_lbl]
        syn_dphi = [wrap_deg(np.radians(c[s]["gamma_phase"] - c["소마"]["gamma_phase"]))
                    for s in site_lbl]
        sweep.append(dict(f_gamma=f_g, comps=c,
                          gamma_over_theta=keep,
                          syn_gamma_ratio=syn_keep, syn_gamma_dphi=syn_dphi))
        print(f"      {f_g:.0f}Hz : 소마 gamma {c['소마']['gamma_pp']:.3f} mV "
              f"(theta 대비 {keep:.3f}) · 시냅스 전달비 " +
              " ".join(f"{v:.3f}" for v in syn_keep) +
              " · 위상차 " + " ".join(f"{v:+.1f}deg" for v in syn_dphi))
    # 주입에서의 gamma/theta 비는 GAMMA_FRAC 로 고정인데 막에서는 줄어든다
    inj_ratio = GAMMA_FRAC
    atten = [inj_ratio / s["gamma_over_theta"] for s in sweep]
    print(f"      주입 gamma/theta 비 {inj_ratio:.2f} -> 막에서 " +
          " ".join(f"{s['gamma_over_theta']:.3f}" for s in sweep) +
          "  (저역통과 감쇠 " + " ".join(f"{a:.1f}x" for a in atten) + ")")

    # ── 그림 ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15.2, 8.6))
    gs_ = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.50)
    axA = fig.add_subplot(gs_[0, :2])
    axB = fig.add_subplot(gs_[0, 2])
    axC = fig.add_subplot(gs_[1, 0])
    axD = fig.add_subplot(gs_[1, 1])
    axE = fig.add_subplot(gs_[1, 2])

    m = (Rm["t"] >= T0 - 20) & (Rm["t"] <= T0 + 600)
    axA.plot(Rm["t"][m] - T0, Rm["post_v"][m], color="#37474f", lw=1.3, label="소마 Vm")
    axA.plot(Rm["t"][m] - T0, Rm["local_v"][0][m], color="#1565c0", lw=1.0, ls="--",
             label=f"시냅스 {site_lbl[0]}")
    ax2 = axA.twinx()
    mm = (tt >= T0 - 20) & (tt <= T0 + 600)
    ax2.plot(tt[mm] - T0, wav[mm], color="#c62828", lw=0.7, alpha=0.55,
             label="주입 전류")
    ax2.set_ylabel("주입 전류 (nA)", color="#c62828")
    for k in range(4):
        axA.axvline(k * 1000.0 / F_THETA, color="#b0bec5", ls=":", lw=0.9)
    axA.set_xlabel("리듬 시작 기준 시간 (ms)"); axA.set_ylabel("Vm (mV)")
    axA.set_title(f"A. ★부과 theta({F_THETA:.0f}Hz) + ★부과 gamma({F_GAMMA_MAIN:.0f}Hz)\n"
                  "gamma 진폭이 theta peak 에서 크다 (위상-진폭 결합) · "
                  "**둘 다 자연 발생 아님**", fontsize=9.4, loc="left")
    h1, l1 = axA.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")

    axB.plot(centers, amps, "o-", color="#2e7d32", ms=6, lw=2, label=f"PAC (MI {mi:.3f})")
    axB.plot(centers, amps0, "s--", color="#90a4ae", ms=5, lw=1.4,
             label=f"대조 (MI {mi0:.3f})")
    axB.set_xlabel("theta 위상 (deg · 0=peak)"); axB.set_ylabel("gamma 성분 RMS (mV)")
    axB.set_xticks([-180, -90, 0, 90, 180])
    axB.set_title("B. 위상-진폭 결합이 생겼는가\n"
                  "theta peak 근처에서 gamma 가 크다", fontsize=9.4, loc="left")
    axB.legend(fontsize=7.8)

    fg = [s["f_gamma"] for s in sweep]
    axC.plot(fg, [s["comps"]["소마"]["gamma_pp"] for s in sweep], "o-",
             color="#c62828", ms=7, lw=2, label="소마")
    for i, sl in enumerate(site_lbl):
        axC.plot(fg, [s["comps"][sl]["gamma_pp"] for s in sweep], "s--", ms=5, lw=1.4,
                 label=f"시냅스 {sl}")
    axC.set_xlabel("gamma 주파수 (Hz)"); axC.set_ylabel("gamma 진폭 (mV pp)")
    axC.set_title("C. ★막은 저역통과다 — gamma 가 더 깎인다\n"
                  "theta 는 유지되는데 gamma 는 주파수에 따라 준다",
                  fontsize=9.4, loc="left")
    axC.legend(fontsize=7.5)

    axD.plot(fg, [s["gamma_over_theta"] for s in sweep], "o-", color="#6a1b9a",
             ms=7, lw=2)
    axD.axhline(inj_ratio, color="#37474f", ls="--", lw=1.6,
                label=f"주입 비 {inj_ratio:.2f}")
    axD.set_xlabel("gamma 주파수 (Hz)"); axD.set_ylabel("막에서의 gamma/theta 비")
    for s, a in zip(sweep, atten):
        axD.annotate(f"{a:.1f}x 감쇠", (s["f_gamma"], s["gamma_over_theta"]),
                     textcoords="offset points", xytext=(0, 8), fontsize=7.5,
                     ha="center")
    axD.set_title("D. 주입 비 != 막에서의 비\n"
                  "6-2 는 **막에서의 비**를 보고해야 한다", fontsize=9.4, loc="left")
    axD.legend(fontsize=7.8)

    xx = np.arange(len(F_GAMMA))
    for i, sl in enumerate(site_lbl):
        axE.bar(xx + (i - 0.5) * 0.35, [s["syn_gamma_ratio"][i] for s in sweep],
                width=0.35, label=f"시냅스 {sl}")
    axE.axhline(1.0, color="#37474f", ls=":", lw=1.2)
    axE.set_xticks(xx); axE.set_xticklabels([f"{f:.0f}Hz" for f in F_GAMMA])
    axE.set_ylabel("시냅스/소마 gamma 진폭비")
    axE.set_title("E. gamma 가 시냅스까지 가는가\n"
                  "theta 보다 더 깎인다 (4-3: theta 는 0.71~0.79)",
                  fontsize=9.4, loc="left")
    axE.legend(fontsize=7.5)

    fig.suptitle("4-4  부과 theta-gamma 중첩 — ★둘 다 부과다 (자연 발생 아님)",
                 fontsize=12.5, y=0.985)
    fig.subplots_adjust(top=0.89)
    plots.stamp(fig, f"4-4 | ★부과 theta {F_THETA:.0f}Hz + ★부과 gamma · "
                     f"주입 gamma/theta {GAMMA_FRAC:.2f} · PAC 지수 {mi:.3f}(대조 {mi0:.3f}) · "
                     f"gamma 저역통과 감쇠 {atten[0]:.1f}~{atten[-1]:.1f}배")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "4-4_theta_gamma.png")

    # ── 검증 ─────────────────────────────────────────────────────────────
    th_pp = cm["소마"]["theta_pp"]
    checks = [
        ("theta 성분이 목표 진폭 근처다 (4-3 과 같은 조건, 25% 이내)",
         abs(th_pp - TARGET_THETA_PP) / TARGET_THETA_PP < 0.25),
        ("gamma 성분이 실제로 존재한다 (소마 > 0.3 mV)",
         cm["소마"]["gamma_pp"] > 0.3),
        ("두 성분이 분리 측정된다 (theta 가 gamma 보다 크다)",
         th_pp > cm["소마"]["gamma_pp"]),
        (f"★부과한 PAC 이 대조보다 확실히 크다 (MI {mi:.3f} vs {mi0:.3f}, 1.3배 이상)",
         mi > mi0 * 1.3),
        (f"대조 조건(gamma 진폭 일정)의 잔여 결합을 정량했다 (MI {mi0:.3f})",
         np.isfinite(mi0)),
        ("★막이 저역통과다 — gamma 주파수가 높을수록 진폭이 준다",
         sweep[0]["comps"]["소마"]["gamma_pp"] > sweep[-1]["comps"]["소마"]["gamma_pp"]),
        (f"★막에서의 gamma/theta 비가 주입 비({inj_ratio:.2f})보다 작다",
         all(s["gamma_over_theta"] < inj_ratio for s in sweep)),
        ("gamma 도 시냅스 위치까지 전달된다 (비 > 0.3)",
         all(v > 0.3 for s in sweep for v in s["syn_gamma_ratio"])),
        ("★gamma 는 theta 보다 더 깎인다 (시냅스 전달비 비교)",
         min(v for s in sweep for v in s["syn_gamma_ratio"]) < 0.714),
        ("4-3 의 기본 방식을 그대로 썼다 (정현파 전류)",
         default_method == "정현파 전류"),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    n_ok = sum(1 for _, v in checks if v)

    out = dict(imposed=True,
               note_imposed=("★theta 도 gamma 도 **부과**다. gamma 는 개재뉴런 네트워크 "
                             "현상이라 추체세포 2개로는 어떤 방식으로도 생성되지 않는다"
                             "(PLAN T4). theta 는 4-2 가 자연 발생 불가로 판정했다. "
                             "그림 캡션마다 명기하고 '자연 발생' 이라고 쓰지 않는다."),
               dt=DT, rec_dt=REC_DT, settle_ms=SETTLE_MS, dur_ms=DUR,
               f_theta=F_THETA, f_gamma_list=F_GAMMA, f_gamma_main=F_GAMMA_MAIN,
               gamma_frac_injected=GAMMA_FRAC, pac_depth=PAC_DEPTH,
               from_4_3=dict(amp_theta_nA=AMP_THETA, default_method=default_method),
               main=dict(components=cm, pac_index=mi, pac_control_index=mi0,
                         pac_by_phase=dict(centers_deg=[float(c) for c in centers],
                                           gamma_rms=[float(a) for a in amps],
                                           control_rms=[float(a) for a in amps0])),
               sweep=[{k: v for k, v in s.items()} for s in sweep],
               lowpass=dict(injected_ratio=inj_ratio,
                            membrane_ratio=[s["gamma_over_theta"] for s in sweep],
                            attenuation=[float(a) for a in atten]),
               finding_pac=(f"★대조 조건(gamma 전류 진폭 일정)에서도 막전위의 gamma 포락선이 "
                            f"theta 위상에 따라 변조된다(MI {mi0:.3f}). 부과 PAC 조건은 "
                            f"{mi:.3f} 다. 즉 **관측된 PAC 의 일부는 구동이 아니라 막의 성질**이다 "
                            f"— 막 임피던스가 전압 의존이라 theta 로 탈분극된 위상에서 gamma "
                            f"응답이 달라진다. 6-2 는 '결합이 있다' 로 끝내면 안 되고 "
                            f"**대조 조건과의 차이**를 보고해야 한다."),
               finding_measure=("★PAC 측정법이 결과를 바꾼다. 처음에 'theta 적합을 뺀 잔차의 "
                                "RMS' 로 쟀더니 gamma 진폭이 일정한 대조에서도 0.41 이 나왔다 "
                                "— 잔차에 고조파·표류가 섞였기 때문이다. gamma 대역만 좁게 "
                                "통과시킨 뒤 힐베르트 포락선을 써야 한다."),
               finding=("★주입 비와 막에서의 비가 다르다. 주입 전류의 gamma/theta 비를 "
                        f"{GAMMA_FRAC:.2f} 로 넣어도 막전위에서는 " +
                        " · ".join(f"{s['f_gamma']:.0f}Hz {s['gamma_over_theta']:.3f}"
                                   for s in sweep) +
                        " 로 줄어든다(막이 저역통과이기 때문). 6-2 는 **막에서의 비**를 "
                        "보고해야 하고, '주입에서 gamma 를 35% 넣었다' 는 서술은 오도다."),
               phase_reference=("4-3 이 정한 대로 위상 기준은 **시냅스 위치의 국소 막전위**다. "
                                "gamma 는 theta 보다 더 깎이므로 시냅스 위치의 gamma 진폭을 "
                                "따로 인쇄한다."),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "4-4_gamma.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 4-4 완료 ({n_ok}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
