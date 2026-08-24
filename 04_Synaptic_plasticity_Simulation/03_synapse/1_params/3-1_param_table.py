# -*- coding: utf-8 -*-
"""3-1 시냅스 파라미터 확정 — PC->PC 단일 클래스 (값·출처 명시 + 억압 STP)

단계   : 3-1 (파이프라인 3단계 시냅스 / 하위 1 params)
방법   : config/synapse.yaml 의 PC->PC 파라미터를 표로 확정하고(각 값에 paper/ours/mod 출처 태그),
         순수 numpy TM(lib.refs.tm)으로 8펄스 트레인 정규화 방출량을 그려 "왜 억압인지"를 보인다.
         촉진 대조는 없는 클래스를 만들지 않고 Ecker Table3 의 실측 촉진 클래스
         PC->SOM+ (E1) 를 참고로 나란히 그린다(우리 연결 아님을 명기). NEURON 불필요(빠름).
근거   : 이 벤치는 CA1 추체세포 2개를 붙인 페어 벤치 -> 연결은 PC->PC 하나다 (D9).
         Ecker2020 Table3 PC->PC(E2) 실측 · §2.3 NMDA(D11) · Moradi&Ascoli 2020(역전위).
         ⚠️ SC->PC 는 Ecker Table3 에 없는 튜닝 클래스여서 삭제했다(D9). 이 스크립트는
            config 에 SC->PC 가 되살아나면 실패한다(회귀 방지 단언).
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
from lib import refdata                       # noqa: E402
from lib.refs import tm                       # noqa: E402

# 출처 성격 — Ecker2020 Table3 각주를 그대로 반영 (D15). '논문에 실린 값' 과
# '해마에서 측정된 값' 은 다르다. 가소성 연구에서는 이 구분이 중요하다.
SRC_COLOR = {"measured": "#2e7d32", "insilico": "#ef6c00", "xregion": "#c62828",
             "xpathway": "#6a1b9a", "mod": "#1565c0", "tuned": "#ff6f00", "ours": "#7b1fa2"}
SRC_KO = {"measured": "해마 실측", "insilico": "논문 in silico 보정",
          "xregion": "체감각피질 일반화", "xpathway": "다른 경로 외삽",
          "mod": "mod 기본값"}

# Ecker2020 Table3 PC->PC(E2) 기대값 — config 가 조용히 바뀌면 잡는다
EXPECT = {"g_nS": 0.6, "tau_d_AMPA": 3.0, "NMDA_ratio": 1.22,
          "Use": 0.50, "Dep_ms": 671.0, "Fac_ms": 17.0, "Nrrp": 2,
          "tau_r_NMDA": 3.9, "tau_d_NMDA": 148.5}


def main():
    plots.setup()
    with open(os.path.join(ROOT, "config", "synapse.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]
    pc = classes["PC->PC"]
    E1 = refdata.ECKER_E1_CONTRAST

    print("=== 3-1 시냅스 파라미터 확정 (PC->PC 단일) ===")
    print(f"  기본 클래스: {cfg['default_class']} — CA1 추체세포 -> CA1 추체세포 (이 벤치의 유일한 연결)")
    print(f"  등록 클래스: {list(classes)}")

    keys = [("g_nS", "g (nS)"), ("e_rev_mV", "역전위 (mV)"),
            ("tau_r_AMPA", "AMPA tau_r (ms)"), ("tau_d_AMPA", "AMPA tau_d (ms)"),
            ("tau_r_NMDA", "NMDA tau_r (ms)"), ("tau_d_NMDA", "NMDA tau_d (ms)"),
            ("NMDA_ratio", "NMDA:AMPA"), ("Use", "Use"),
            ("Dep_ms", "Dep (ms)"), ("Fac_ms", "Fac (ms)"), ("Nrrp", "Nrrp")]

    import matplotlib.pyplot as plt

    # ---- 그림 1: 파라미터 표 (출처 태그 색) ----
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ax.axis("off")
    ax.set_title("3-1  시냅스 전달 파라미터 확정 — PC→PC (CA1 추체세포 쌍, 이 벤치의 유일한 연결)",
                 fontsize=12, loc="left", pad=12)
    n = len(keys)
    ax.text(0.02, n + 0.5, "파라미터", fontsize=10, fontweight="bold")
    ax.text(0.42, n + 0.5, "값", fontsize=10, fontweight="bold", ha="center", color="#d84315")
    ax.text(0.58, n + 0.5, "출처", fontsize=10, fontweight="bold")
    y = n - 1
    for key, label in keys:
        e = pc[key]
        ax.text(0.02, y, label, fontsize=9.5, va="center")
        c = SRC_COLOR.get(e["src"], "#000")
        ax.text(0.42, y, f"{e['v']}", fontsize=10, va="center", ha="center",
                color=c, fontweight="bold")
        ref = e.get("ref", "")
        ax.text(0.58, y, (ref[:66] + "…") if len(ref) > 67 else ref,
                fontsize=7.6, va="center", color="#555")
        y -= 1
    cnt = {}
    for key, _ in keys:
        cnt[pc[key]["src"]] = cnt.get(pc[key]["src"], 0) + 1
    ax.set_xlim(0, 1.24); ax.set_ylim(-3.4, n + 1.2)
    for i, (sname, ko) in enumerate(SRC_KO.items()):
        ax.text(0.02 + i * 0.20, -1.1, f"■ {ko} ({cnt.get(sname, 0)})",
                fontsize=8.2, color=SRC_COLOR[sname])
    ax.text(0.02, -1.85,
            "★ 우리가 임의로 정한 튜닝값은 0개다. 다만 '논문에 실린 값' 과 '해마에서 측정된 값' 은 다르다 — 아래 두 줄이 그 구분이다.",
            fontsize=8.3, color="#333", fontweight="bold")
    ax.text(0.02, -2.35,
            "※ Use·Dep·Fac 는 Ecker Table3 상첨자 a = 체감각 피질(Markram 2015) 일반화값이다. CA1 PC→PC 논문에 "
            "전시냅스 2발 이상 원시 트레이스가 없어서다(Ecker p12).",
            fontsize=7.6, color="#c62828")
    ax.text(0.02, -2.8,
            "   억압 '방향' 은 Deuchars&Thomson 1996 이 CA1 PC→PC 4쌍 전부에서 짝펄스 억압을 관찰해 뒷받침한다 — 크기는 피질 값이다.",
            fontsize=7.6, color="#c62828")
    ax.text(0.02, -3.25,
            "※ [Ca2+]o: 논문은 전도도를 2.5mM(슬라이스) PSP 진폭에 맞췄다. PC→PC 는 in vivo 수준(1.1~1.3mM)에서 "
            "E3(준선형)이 된다(Ecker p8) — 우리 값은 슬라이스 조건이다.",
            fontsize=7.6, color="#c62828")
    plots.stamp(fig, "3-1 | config/synapse.yaml 단일 출처 | Ecker2020 원문 대조 완료(DOI 10.1002/hipo.23220, D15)")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "3-1_param_table.png")

    # ---- 그림 2: 억압(우리) vs 실측 촉진(참고) ----
    U, D, F = pc["Use"]["v"], pc["Dep_ms"]["v"], pc["Fac_ms"]["v"]
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    freqs = [10, 20, 50]
    panels = [(a1, (U, D, F), f"PC→PC (우리 연결) — 억압형\nUse={U} Dep={D:.0f} Fac={F:.0f}ms",
               "#d84315"),
              (a2, (E1["Use"], E1["Dep_ms"], E1["Fac_ms"]),
               f"{E1['name']} — 실측 촉진형 (참고)\nUse={E1['Use']} Dep={E1['Dep_ms']:.0f} Fac={E1['Fac_ms']:.0f}ms",
               "#2e7d32")]
    for ax_, (u, d, f_), title, col in panels:
        for fq in freqs:
            _, amp = tm.train(8, fq, u, d, f_)
            ax_.plot(range(1, 9), amp, "-o", ms=4, label=f"{fq} Hz")
        ax_.axhline(1.0, ls=":", color="#999", lw=0.9)
        ax_.set_xlabel("펄스 번호"); ax_.set_ylabel("정규화 방출량 (첫 펄스=1)")
        ax_.set_title(title, fontsize=10, loc="left", color=col)
        ax_.legend(fontsize=8.5, title="트레인 주파수")
    a2.text(0.98, 0.03, "※ 이 벤치의 연결 아님\n(표적이 개재뉴런)", transform=a2.transAxes,
            fontsize=8, color="#c62828", ha="right", va="bottom")
    fig2.suptitle("3-1  단기가소성 — 우리 연결 PC→PC 는 억압(Fac<Dep). 촉진 대조는 Ecker 실측 E1 클래스 [순수 numpy TM]",
                  fontsize=11.5, y=0.99)
    fig2.subplots_adjust(top=0.82, wspace=0.25)
    plots.stamp(fig2, "3-1 | 8펄스 트레인 · lib.refs.tm (Fuhrmann2002/Ecker Eq5-6) · 5-9 에서 NEURON mod 와 대조")
    plots.save(fig2, outdir, "3-1_stp_classes.png")

    # ---- 검증 ----
    _, pc_amp = tm.train(8, 20, U, D, F)
    _, e1_amp = tm.train(8, 20, E1["Use"], E1["Dep_ms"], E1["Fac_ms"])
    print(f"  PC->PC 20Hz 8펄스 방출량비 첫→끝: {pc_amp[0]:.2f}→{pc_amp[-1]:.2f}")
    print(f"  (참고) {E1['name']} 20Hz: {e1_amp[0]:.2f}→{e1_amp[-1]:.2f}")

    mism = {k: (pc[k]["v"], v) for k, v in EXPECT.items() if abs(float(pc[k]["v"]) - v) > 1e-9}
    tuned = [k for k, _ in keys if pc[k]["src"] in ("tuned", "ours")]
    xreg = [k for k, _ in keys if pc[k]["src"] == "xregion"]
    checks = [
        ("기본 클래스 = PC->PC", cfg["default_class"] == "PC->PC"),
        ("SC->PC 클래스 없음 (D9 회귀 방지)", "SC->PC" not in classes),
        ("등록 클래스 1개뿐", len(classes) == 1),
        ("Ecker Table3/§2.3 값과 일치", not mism),
        ("우리 임의 튜닝값 0개 (논문값만 사용)", not tuned),
        (f"출처 성격 태그로 구분됨 (체감각피질 일반화 {len(xreg)}개 명시)", len(xreg) > 0),
        ("PC->PC 는 억압 (끝 < 첫)", pc_amp[-1] < pc_amp[0]),
        ("참고 E1 은 촉진 (끝 > 첫)", e1_amp[-1] > e1_amp[0]),
    ]
    for k, ok in checks:
        print(f"  {'O' if ok else 'X'} {k}")
    if mism:
        print(f"    불일치: {mism}")
    if tuned:
        print(f"    튜닝 항목: {tuned}")
    print(f"  출처 내역: " + " · ".join(f"{SRC_KO.get(k,k)} {v}" for k, v in cnt.items()))
    n_ok = sum(1 for _, ok in checks if ok)

    out = dict(default_class=cfg["default_class"], classes=list(classes),
               PC_PC={k: pc[k] for k, _ in keys},
               contrast_reference=E1,
               pc_train_20hz=[round(float(x), 3) for x in pc_amp],
               e1_train_20hz=[round(float(x), 3) for x in e1_amp],
               pc_depression=bool(pc_amp[-1] < pc_amp[0]),
               tuned_params=tuned, xregion_params=xreg, src_breakdown=cnt,
               ecker_verified=dict(doi=refdata.ECKER2020["doi"],
                                   cite=refdata.ECKER2020["cite"],
                                   table3_has_no_schaffer=refdata.ECKER2020["table3_has_no_schaffer"],
                                   ca_o_mM=refdata.ECKER2020["ca_o_mM_calibration"],
                                   note=refdata.ECKER2020["note"]),
               checks={k: bool(v) for k, v in checks}, passed=n_ok, total=len(checks))
    jpath = os.path.join(outdir, "3-1_params.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    if n_ok != len(checks):
        print(f"\n[실패] {len(checks)-n_ok}개 미통과")
        return 1
    print(f"\n[통과] 3-1 완료 ({n_ok}/{len(checks)}) — PC→PC 단일 클래스·튜닝값 0개·억압 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
