# -*- coding: utf-8 -*-
"""2-5 공명 — 세포가 스스로 theta 대역에 크게 반응하는가 (ZAP + Ih 차단 대조)

단계   : 2-5 (파이프라인 2단계 뉴런 / 하위 5 resonance)
쉬운 설명: 그네를 고유 리듬에 맞춰 밀면 크게 흔들리듯, 뉴런도 특정 주파수 입력에 막전위가
          크게 출렁인다. 그 주파수를 '공명 주파수'라 한다. 우리 목표(theta 가소성) 때문에,
          이 세포가 theta 대역(약 4~8Hz)에 스스로 잘 반응하는지 본다.
방법   : ZAP 전류(주파수가 서서히 오르는 정현파)를 넣고 주파수별 임피던스 |Z(f)| 를 구해
          공명 주파수 f_R 와 강도 Q 를 잰다. Ih(과분극활성 전류)를 끈 대조군과 비교해
          공명이 정말 Ih 때문인지 확인한다.
근거   : Ih = Magee 1998 (hd.mod) — CA1 추체세포 theta 공명의 기전.
★결과 두 갈래 모두 유효: theta 공명이 있으면 theta 반응 자연스러움 / 없으면 theta 는 부과.
결과   : figures/2-5_zap_impedance.png · figures/2-5_zap_trace.png · figures/2-5_resonance.json

실행:
  . .\\env\\activate.ps1
  & $Py04 02_neurons\\5_resonance\\2-5_zap_resonance.py
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

import numpy as np                          # noqa: E402
from lib import plots                        # noqa: E402
from lib import ephys                         # noqa: E402
from lib.bench import Bench                   # noqa: E402
from lib.nrnenv import h                     # noqa: E402

THETA_LO, THETA_HI = 4.0, 8.0                # theta 대역 (판정용)
F0, F1 = 0.5, 20.0                           # ZAP 주파수 범위
DUR = 10000.0                                # 10초 (theta 분해능 충분) · dt=0.1 로 실행(~4배 빠름)
AMP = 0.03                                   # ZAP 진폭 nA (subthreshold 유지)


def run_one(cell, block_ih):
    t, v, iw = ephys.zap_response(cell, f0=F0, f1=F1, amp_nA=AMP, dur_ms=DUR,
                                  block_ih=block_ih, v_init=-70.0)
    fc, Z, fr, Q = ephys.impedance_profile(t, v, iw)
    return dict(t=t, v=v, iw=iw, fc=fc, Z=Z, f_res=fr, Q=Q)


def verdict(fr, Q):
    """theta 대역 공명 판정 — 쉬운 한 줄 결론."""
    in_theta = (not np.isnan(fr)) and (THETA_LO <= fr <= THETA_HI)
    strong = (not np.isnan(Q)) and (Q >= 1.05)
    if in_theta and strong:
        return "가능", f"공명 봉우리가 theta 대역({fr:.1f}Hz)에 있고 뚜렷(Q={Q:.2f})"
    if in_theta and not strong:
        return "약함", f"봉우리가 theta({fr:.1f}Hz)에 있으나 약함(Q={Q:.2f})"
    if not np.isnan(fr):
        return "불가", f"봉우리가 theta 밖({fr:.1f}Hz) 또는 없음(Q={Q:.2f})"
    return "불가", "공명 봉우리 측정 실패"


def main():
    plots.setup()
    print("=== 2-5 공명 (ZAP) ===")
    b = Bench()
    # 대표로 post 세포(기록 세포)로 판정. Ih 차단은 세포를 새로 로드해야 하므로
    # post 를 두 번(정상/차단) 돌린다. 재로드로 원본 보존.
    cells = {"정상": b.post}
    R = {}
    print("  [정상 Ih] post 세포 ZAP 20s ...")
    R["정상"] = run_one(b.post, block_ih=False)

    # Ih 차단용으로 post 를 새로 로드 (block 이 원본을 바꾸므로)
    from lib import cells as cellmod
    post2, _ = cellmod.load_cell(
        os.path.join(os.path.dirname(ROOT), "Models", b.geo["pair"]["post_bundle"]), "post_ihblk")
    print("  [Ih 차단] 대조군 ZAP 20s ...")
    R["Ih차단"] = run_one(post2, block_ih=True)

    v0, vb = verdict(R["정상"]["f_res"], R["정상"]["Q"]), None
    print(f"  정상 : f_R={R['정상']['f_res']:.2f}Hz · Q={R['정상']['Q']:.3f} -> {v0[0]} ({v0[1]})")
    print(f"  Ih차단: f_R={R['Ih차단']['f_res']:.2f}Hz · Q={R['Ih차단']['Q']:.3f}")
    ih_drop = R["정상"]["Q"] - R["Ih차단"]["Q"]
    print(f"  Ih 차단 시 Q 변화: {ih_drop:+.3f} (양수면 공명이 Ih 때문)")

    import matplotlib.pyplot as plt

    # 그림 1: 임피던스 프로파일 (핵심)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axvspan(THETA_LO, THETA_HI, color="#ffb300", alpha=0.16, zorder=0, label="theta 대역")
    ax.plot(R["정상"]["fc"], R["정상"]["Z"], "-o", color="#1565c0", ms=3, label="정상 (Ih 있음)")
    ax.plot(R["Ih차단"]["fc"], R["Ih차단"]["Z"], "-o", color="#c62828", ms=3,
            label="Ih 차단 (대조)")
    if not np.isnan(R["정상"]["f_res"]):
        zpk = np.nanmax(R["정상"]["Z"])
        ax.plot([R["정상"]["f_res"]], [zpk], "*", color="#1565c0", ms=18, zorder=5)
        ax.annotate(f"공명 {R['정상']['f_res']:.1f}Hz\nQ={R['정상']['Q']:.2f}",
                    xy=(R["정상"]["f_res"], zpk), xytext=(R["정상"]["f_res"]+2, zpk),
                    fontsize=10, color="#0d47a1", fontweight="bold")
    ax.set_xlabel("입력 주파수 (Hz)"); ax.set_ylabel("임피던스 |Z| (MOhm)")
    ax.set_title(f"2-5  막전위 공명 — post 세포  [판정: {v0[0]}]", fontsize=12, loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.02, 0.03, v0[1], transform=ax.transAxes, fontsize=9, color="#555",
            bbox=dict(fc="white", ec="#ccc", alpha=0.9))
    plots.stamp(fig, f"2-5 | ZAP {F0}~{F1}Hz·{DUR/1000:.0f}s·{AMP}nA | theta {THETA_LO}~{THETA_HI}Hz")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "2-5_zap_impedance.png")

    # 그림 2: ZAP 입력 + 막전위 응답 (직관)
    fig2, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 2]})
    tt, wave, _, _, _ = R["정상"]["iw"]
    a1.plot(tt/1000.0, wave, color="#7b1fa2", lw=0.5)
    a1.set_ylabel("주입 전류 (nA)")
    a1.set_title("ZAP 입력 (주파수 0.5→20Hz 서서히 상승) 과 막전위 응답", fontsize=10, loc="left")
    a2.plot(R["정상"]["t"]/1000.0, R["정상"]["v"], color="#1565c0", lw=0.5, label="정상")
    a2.plot(R["Ih차단"]["t"]/1000.0, R["Ih차단"]["v"], color="#c62828", lw=0.4, alpha=0.6,
            label="Ih 차단")
    a2.set_xlabel("시간 (s)"); a2.set_ylabel("막전위 (mV)")
    a2.legend(fontsize=8, loc="upper right")
    # theta 대역이 지나가는 시각 표시 (주파수→시각 역산)
    for f in (THETA_LO, THETA_HI):
        ts = (f - F0) / (F1 - F0) * (DUR/1000.0)
        a2.axvline(ts, color="#ffb300", ls="--", lw=1, alpha=0.7)
    a2.text((((THETA_LO+THETA_HI)/2 - F0)/(F1-F0))*(DUR/1000.0), a2.get_ylim()[1],
            " theta 통과 구간", fontsize=8, color="#b26a00", va="top")
    plots.stamp(fig2, "2-5 | 공명 주파수 부근에서 막전위 출렁임이 최대")
    plots.save(fig2, outdir, "2-5_zap_trace.png")

    out = dict(cell=b.geo["pair"]["post_tag"], theta_band=[THETA_LO, THETA_HI],
               zap=dict(f0=F0, f1=F1, dur_s=DUR/1000.0, amp_nA=AMP),
               normal=dict(f_res_hz=round(float(R["정상"]["f_res"]), 2),
                           Q=round(float(R["정상"]["Q"]), 3)),
               ih_blocked=dict(f_res_hz=round(float(R["Ih차단"]["f_res"]), 2),
                               Q=round(float(R["Ih차단"]["Q"]), 3)),
               ih_Q_drop=round(float(ih_drop), 3),
               verdict=v0[0], verdict_reason=v0[1])
    jpath = os.path.join(outdir, "2-5_resonance.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    print(f"\n[통과] 2-5 완료 — 판정: {v0[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
