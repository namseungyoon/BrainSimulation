# -*- coding: utf-8 -*-
"""3-1 시냅스 파라미터 확정 — 값·출처 명시 + 촉진/억압 STP 시각화

단계   : 3-1 (파이프라인 3단계 시냅스 / 하위 1 params)
방법   : config/synapse.yaml 의 두 클래스(SC->PC 촉진형 / PC->PC 억압형) 파라미터를 표로
         확정하고(각 값에 paper/tuned/ours/mod 출처 태그), 순수 numpy TM(lib.refs.tm)으로
         8펄스 트레인의 정규화 방출량을 그려 "왜 SC->PC 가 촉진이고 PC->PC 가 억압인지" 를
         눈으로 보인다. NEURON 불필요(빠름).
근거   : Ecker2020 Table3 · Andrasfalvy&Magee 2001(NMDA) · Moradi&Ascoli 2020(역전위) ·
         Dobrunz&Stevens 1997(SC 촉진). 상세 출처는 config/synapse.yaml.
결과   : figures/3-1_param_table.png · figures/3-1_stp_classes.png · figures/3-1_params.json
실행   : .venv\\Scripts\\python.exe 03_synapse\\1_params\\3-1_param_table.py  (NEURON 불필요)
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
import yaml                                 # noqa: E402
from lib import plots                        # noqa: E402
from lib.refs import tm                       # noqa: E402

SRC_COLOR = {"paper": "#2e7d32", "tuned": "#ef6c00", "ours": "#7b1fa2", "mod": "#1565c0"}
SRC_KO = {"paper": "측정/논문", "tuned": "튜닝값", "ours": "우리선택", "mod": "mod기본"}


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "synapse.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sc = cfg["classes"]["SC->PC"]
    pc = cfg["classes"]["PC->PC"]

    print("=== 3-1 시냅스 파라미터 확정 ===")
    print(f"  기본 클래스: {cfg['default_class']} (촉진형, Schaffer collateral 대리)")

    # 표에 넣을 항목 순서
    keys = [("g_nS", "g (nS)"), ("e_rev_mV", "역전위 (mV)"),
            ("tau_r_AMPA", "AMPA τr (ms)"), ("tau_d_AMPA", "AMPA τd (ms)"),
            ("tau_r_NMDA", "NMDA τr (ms)"), ("tau_d_NMDA", "NMDA τd (ms)"),
            ("NMDA_ratio", "NMDA:AMPA"), ("Use", "Use"),
            ("Dep_ms", "Dep (ms)"), ("Fac_ms", "Fac (ms)"), ("Nrrp", "Nrrp")]

    import matplotlib.pyplot as plt

    # ---- 그림 1: 파라미터 표 (출처 태그 색으로) ----
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.axis("off")
    ax.set_title("3-1  시냅스 전달 파라미터 확정 (색=출처)", fontsize=12.5, loc="left", pad=12)
    # 헤더
    ax.text(0.30, len(keys)+0.5, "SC→PC (촉진, 기본)", fontsize=10, fontweight="bold",
            ha="center", color="#2e7d32")
    ax.text(0.62, len(keys)+0.5, "PC→PC (억압, 대조)", fontsize=10, fontweight="bold",
            ha="center", color="#d84315")
    ax.text(0.02, len(keys)+0.5, "파라미터", fontsize=10, fontweight="bold")
    y = len(keys) - 1
    for key, label in keys:
        ax.text(0.02, y, label, fontsize=9.5, va="center")
        for x, cls in [(0.30, sc), (0.62, pc)]:
            e = cls[key]
            c = SRC_COLOR.get(e["src"], "#000")
            ax.text(x, y, f"{e['v']}", fontsize=9.5, va="center", ha="center", color=c)
        y -= 1
    ax.set_xlim(0, 1); ax.set_ylim(-2.4, len(keys)+1.2)
    # 범례 (표 아래로 충분히 내림)
    for i, (s, ko) in enumerate(SRC_KO.items()):
        ax.text(0.02 + i*0.20, -1.2, f"■ {ko}", fontsize=8.5, color=SRC_COLOR[s])
    ax.text(0.02, -1.9,
            "핵심: SC→PC 의 Use·Dep·Fac·Nrrp 는 튜닝값(Ecker Table3에 SC 없음) · 역전위 -8.5 는 mod 기본 0 을 덮어씀",
            fontsize=8, color="#555")
    plots.stamp(fig, "3-1 | config/synapse.yaml 단일 출처 | NMDA τd 61ms(Andrasfalvy) 채택, Ecker 148.5 미채택")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-1_param_table.png")

    # ---- 그림 2: 촉진 vs 억압 STP (순수 numpy TM) ----
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    freqs = [10, 20, 50]
    for ax_, cls, name, col in [(a1, sc, "SC→PC (촉진형)", "#2e7d32"),
                                (a2, pc, "PC→PC (억압형)", "#d84315")]:
        U, D, F = cls["Use"]["v"], cls["Dep_ms"]["v"], cls["Fac_ms"]["v"]
        for fq in freqs:
            st, amp = tm.train(8, fq, U, D, F)
            ax_.plot(range(1, 9), amp, "-o", ms=4, label=f"{fq} Hz")
        ax_.axhline(1.0, ls=":", color="#999", lw=0.9)
        ax_.set_xlabel("펄스 번호"); ax_.set_ylabel("정규화 방출량 (첫 펄스=1)")
        ax_.set_title(f"{name}\nUse={U} Dep={D:.0f} Fac={F:.0f}ms", fontsize=10, loc="left")
        ax_.legend(fontsize=8.5, title="트레인 주파수")
    a1.set_ylim(0.4, None)
    fig2.suptitle("3-1  단기가소성 클래스 — SC→PC 는 촉진(Fac>Dep), PC→PC 는 억압(Fac<Dep) [순수 numpy TM]",
                  fontsize=12, y=0.99)
    fig2.subplots_adjust(top=0.83, wspace=0.25)
    plots.stamp(fig2, "3-1 | 8펄스 트레인 · lib.refs.tm (Fuhrmann2002/Ecker Eq5-6) · 5-9 에서 NEURON mod 와 대조")
    plots.save(fig2, outdir, "3-1_stp_classes.png")

    # ---- 검증 수치: 촉진/억압 방향 확인 ----
    _, sc_amp = tm.train(8, 20, sc["Use"]["v"], sc["Dep_ms"]["v"], sc["Fac_ms"]["v"])
    _, pc_amp = tm.train(8, 20, pc["Use"]["v"], pc["Dep_ms"]["v"], pc["Fac_ms"]["v"])
    sc_facil = sc_amp[-1] > sc_amp[0]     # SC 는 촉진(마지막>첫)
    pc_depr = pc_amp[-1] < pc_amp[0]      # PC 는 억압
    print(f"  SC->PC 20Hz 8펄스 방출량비 첫→끝: {sc_amp[0]:.2f}→{sc_amp[-1]:.2f} "
          f"({'촉진 OK' if sc_facil else '★촉진 아님'})")
    print(f"  PC->PC 20Hz 8펄스 방출량비 첫→끝: {pc_amp[0]:.2f}→{pc_amp[-1]:.2f} "
          f"({'억압 OK' if pc_depr else '★억압 아님'})")

    out = dict(default_class=cfg["default_class"],
               SC_PC={k: sc[k] for k, _ in keys},
               PC_PC={k: pc[k] for k, _ in keys},
               sc_facilitation=bool(sc_facil), pc_depression=bool(pc_depr),
               sc_train_20hz=[round(float(x), 3) for x in sc_amp],
               pc_train_20hz=[round(float(x), 3) for x in pc_amp])
    jpath = os.path.join(outdir, "3-1_params.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if not (sc_facil and pc_depr):
        print("\n[실패] STP 방향이 클래스 정의와 다름")
        return 1
    print("\n[통과] 3-1 완료 — SC→PC 촉진·PC→PC 억압 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
