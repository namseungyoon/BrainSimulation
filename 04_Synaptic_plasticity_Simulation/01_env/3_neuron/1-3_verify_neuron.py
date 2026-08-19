# -*- coding: utf-8 -*-
"""1-3 NEURON 설치 검증 — 임포트가 아니라 '실제로 적분하는가'를 본다

단계   : 1-3 (파이프라인 1단계 환경 / 하위 3 neuron)
방법   : (a) 설치 구성요소 존재 확인  (b) 수동 구획 하나에 전류를 넣어 실제로 풀어 본다.
         `import neuron` 성공은 설치의 절반일 뿐이다. 솔버가 돌고, 고정 dt 가 걸리고,
         해석해와 맞는지까지 봐야 1-4(mod 빌드)로 넘어갈 근거가 된다.
근거   : docs/DECISIONS.md D4 (NEURONHOME 환경변수로 공유 파일 수정 회피)
         docs/DECISIONS.md D6 (모든 하위 단계는 직관적 산출물을 낸다)
재료   : C:\\Users\\USER\\nrn (1-3 설치) · lib/nrnenv.py · lib/plots.py
결과   : figures/1-3_neuron_verify.png · figures/1-3_neuron_verify.json

실행:
  . .\\env\\activate.ps1
  .venv\\Scripts\\python.exe 01_env\\3_neuron\\1-3_verify_neuron.py
"""
import os
import sys
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from lib import plots                      # noqa: E402
from lib import nrnenv                     # noqa: E402
from lib.nrnenv import h                   # noqa: E402
import numpy as np                         # noqa: E402


def passive_step():
    """수동 구획 + 전류 계단. tau = Rm*Cm 이므로 해석해와 직접 비교할 수 있다.

    단일 구획(공간적으로 균일)이므로 케이블 방정식이 1차 RC 로 축약된다:
        V(t) = V_rest + I*R_in * (1 - exp(-t/tau)),   tau = cm / g_pas   [ms]
    """
    soma = h.Section(name="soma")
    soma.L = soma.diam = 20.0               # um
    soma.insert("pas")
    g_pas = 3e-4                            # S/cm2
    cm = 1.0                                # uF/cm2
    e_pas = -70.0
    for seg in soma:
        seg.pas.g = g_pas
        seg.pas.e = e_pas
    soma.cm = cm

    # 면적 (구 근사 대신 NEURON 의 원통 면적을 그대로 쓴다)
    area_um2 = sum(seg.area() for seg in soma)          # um^2
    area_cm2 = area_um2 * 1e-8
    R_in = 1.0 / (g_pas * area_cm2)                     # ohm
    # tau = Rm*Cm.  cm/g_pas 의 단위는 (uF/cm2)/(S/cm2) = uF/S = 1e-6 s = us.
    # 따라서 ms 로 바꾸려면 1e-3 을 곱해야 한다.
    # (최초 구현은 이 환산을 빼먹어 이론값을 1000배로 계산했다 -> 3333 ms vs 실측 3.35 ms.
    #  실측이 맞고 식이 틀렸던 경우다.)
    tau_theory = (cm / g_pas) * 1e-3                    # ms

    amp_nA = 0.05
    stim = h.IClamp(soma(0.5))
    stim.delay, stim.dur, stim.amp = 20.0, 200.0, amp_nA

    t = h.Vector().record(h._ref_t)
    v = h.Vector().record(soma(0.5)._ref_v)

    nrnenv.finit(v_init=e_pas)
    h.continuerun(300.0)

    t = np.array(t)
    v = np.array(v)

    dv_theory_mV = amp_nA * 1e-9 * R_in * 1e3           # nA*ohm -> V -> mV
    # 정상상태: 자극 종료 직전 5 ms 평균
    m = (t > stim.delay + stim.dur - 5.0) & (t <= stim.delay + stim.dur)
    dv_sim_mV = float(v[m].mean() - e_pas)

    # tau 추정: 자극 개시 후 상승분이 (1-1/e) 에 도달하는 시각
    seg_m = (t >= stim.delay) & (t <= stim.delay + stim.dur)
    tt, vv = t[seg_m], v[seg_m]
    target = e_pas + dv_sim_mV * (1.0 - 1.0 / math.e)
    idx = int(np.argmax(vv >= target))
    tau_sim = float(tt[idx] - stim.delay) if idx > 0 else float("nan")

    return dict(t=t, v=v, stim=(stim.delay, stim.dur, amp_nA), e_pas=e_pas,
                area_um2=area_um2, R_in_MOhm=R_in / 1e6,
                dv_theory_mV=dv_theory_mV, dv_sim_mV=dv_sim_mV,
                tau_theory_ms=tau_theory, tau_sim_ms=tau_sim,
                n_points=int(len(t)), dt=float(h.dt),
                cvode=int(h.cvode_active()))


def main():
    plots.setup()
    print("=== 1-3 NEURON 검증 ===")
    inf = nrnenv.info()
    for k, val in inf.items():
        print(f"  {k:18s}: {val}")

    r = passive_step()
    dv_err = abs(r["dv_sim_mV"] - r["dv_theory_mV"]) / r["dv_theory_mV"]
    tau_err = abs(r["tau_sim_ms"] - r["tau_theory_ms"]) / r["tau_theory_ms"]
    print(f"\n  R_in            : {r['R_in_MOhm']:.1f} MOhm")
    print(f"  dV 이론/실측     : {r['dv_theory_mV']:.4f} / {r['dv_sim_mV']:.4f} mV  "
          f"(상대오차 {dv_err:.3%})")
    print(f"  tau 이론/실측    : {r['tau_theory_ms']:.2f} / {r['tau_sim_ms']:.2f} ms  "
          f"(상대오차 {tau_err:.3%})")
    print(f"  dt / cvode      : {r['dt']} / {r['cvode']}")

    # h.nrnversion() 은 "NEURON -- VERSION 8.2.7+ HEAD (34cf696+) 2025-05-21" 처럼
    # 길다. 표에 그대로 넣으면 판정 열을 침범하므로 버전 토큰만 뽑는다.
    _tok = inf["nrnversion"].split()
    ver_short = _tok[_tok.index("VERSION") + 1] if "VERSION" in _tok else inf["nrnversion"]

    checks = [
        ("NEURON 임포트",       ver_short,                            True),
        ("NEURONHOME",         "설정됨",                             bool(inf["neuronhome"])),
        ("nrnivmodl (mod 빌더)", "있음" if inf["nrnivmodl_exists"] else "없음",
         inf["nrnivmodl_exists"]),
        ("mingw 툴체인",        "있음" if inf["mingw_exists"] else "없음", inf["mingw_exists"]),
        ("고정 dt",             f"{r['dt']} ms", abs(r["dt"] - nrnenv.DT) < 1e-12),
        ("cvode 꺼짐",          "예" if r["cvode"] == 0 else "아니오", r["cvode"] == 0),
        ("정상상태 dV",         f"{r['dv_sim_mV']:.3f} mV (이론 {r['dv_theory_mV']:.3f})",
         dv_err < 0.02),
        ("막 시상수 tau",       f"{r['tau_sim_ms']:.2f} ms (이론 {r['tau_theory_ms']:.2f})",
         tau_err < 0.05),
    ]

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 4.9),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # --- 왼쪽: 판정표 -----------------------------------------------------
    axL.axis("off")
    axL.set_title("1-3  NEURON 설치 검증", loc="left", pad=12)
    y = len(checks)
    for label, value, ok in checks:
        c = plots.OK if ok else plots.NG
        axL.text(0.02, y, label, fontsize=10, va="center")
        axL.text(0.44, y, str(value), fontsize=9.5, va="center", color=c)
        axL.text(0.96, y, "O" if ok else "X", fontsize=11, va="center", ha="center",
                 color=c, fontweight="bold")
        y -= 1
    axL.set_xlim(0, 1)
    axL.set_ylim(-0.3, len(checks) + 1.1)
    n_ok = sum(1 for _, _, ok in checks if ok)
    axL.text(0.02, len(checks) + 0.7,
             f"통과 {n_ok}/{len(checks)}   ·   NEURON {ver_short}   ·   {inf['neuronhome']}",
             fontsize=9.5, color=plots.OK if n_ok == len(checks) else plots.NG)

    # --- 오른쪽: 실제로 적분한 결과 ---------------------------------------
    d, dur, amp = r["stim"]
    axR.plot(r["t"], r["v"], lw=1.6, color="#1565c0", label="NEURON 실측")
    ss = r["e_pas"] + r["dv_theory_mV"]
    axR.axhline(ss, ls="--", lw=1.1, color=plots.NG,
                label=f"이론 정상상태 {ss:.2f} mV")
    axR.axvspan(d, d + dur, color="#000000", alpha=0.05)
    axR.annotate(f"IClamp {amp} nA", xy=(d + dur / 2, r["e_pas"] - 0.15),
                 ha="center", fontsize=9, color="#555555")
    axR.set_xlabel("시간 (ms)")
    axR.set_ylabel("막전위 (mV)")
    axR.set_title("수동 구획 전류 계단 응답 — 솔버가 실제로 도는가", loc="left", pad=12)
    axR.legend(loc="center right")
    # 파형이 상단을 채우므로 정보 박스는 자극 시작 전 빈 좌하단에 둔다.
    axR.text(0.02, 0.30,
             f"R_in {r['R_in_MOhm']:.1f} MOhm\n"
             f"tau  {r['tau_sim_ms']:.2f} ms (이론 {r['tau_theory_ms']:.2f})\n"
             f"dV   {r['dv_sim_mV']:.3f} mV, 오차 {dv_err:.2%}",
             transform=axR.transAxes, va="top", fontsize=9,
             bbox=dict(fc="white", ec="#cccccc", alpha=0.9))
    axR.margins(y=0.16)

    plots.stamp(fig, f"1-3 | NEURONHOME={inf['neuronhome']} | dt={r['dt']} ms, cvode off")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "1-3_neuron_verify.png")

    out = {k: v for k, v in r.items() if k not in ("t", "v")}
    out["stim"] = list(out["stim"])
    out.update({"info": inf, "dv_rel_err": dv_err, "tau_rel_err": tau_err,
                "checks_passed": n_ok, "checks_total": len(checks)})
    jpath = os.path.join(outdir, "1-3_neuron_verify.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")

    if n_ok != len(checks):
        print(f"\n[실패] {len(checks) - n_ok}개 항목 미통과")
        return 1
    print("\n[통과] 1-3 검증 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
