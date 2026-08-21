# -*- coding: utf-8 -*-
"""2-4 전기생리 — 두 세포의 f-I·입력저항·sag·활동전위 특성 + 실제 파형

단계   : 2-4 (파이프라인 2단계 뉴런 / 하위 4 ephys)
방법   : lib.bench 로 pre·post 를 로드하고 소마에 계단전류를 걸어 발화율·입력저항·sag·
         AP 진폭/반치폭/역치·발화적응을 잰다. 실제 막전위 파형을 겹쳐 그린다.
검증   : cACpyr 문헌 대략 범위 안(lib.refdata). 범위 밖이면 표시하되 실패로 보지 않고 기록.
재료   : lib/bench · lib/ephys · lib/refdata
결과   : figures/2-4_ephys_battery.png · figures/2-4_ephys.json

실행:
  . .\\env\\activate.ps1
  & $Py04 02_neurons\\4_ephys\\2-4_ephys_battery.py
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
from lib import refdata                       # noqa: E402
from lib.bench import Bench                   # noqa: E402

AMPS = [-0.05, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # nA
FI_AMPS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def measure(cell):
    ir = ephys.input_resistance(cell)
    amps, rates, traces = ephys.fi_curve(cell, FI_AMPS)
    fired = amps[rates > 0]
    rheobase = float(fired.min()) if len(fired) else float("nan")
    # AP 특성·적응은 f-I 에서 이미 구한 파형을 재사용(중복 시뮬 제거).
    # 발화가 넉넉한 계단을 고른다: 3발 이상이면 우선, 없으면 발화하는 가장 강한 계단.
    strong = None
    for a in amps:
        _, _, sp = traces[a]
        during = sp[(sp >= 100.0) & (sp <= 600.0)]
        if len(during) >= 3:
            strong = a; break
    if strong is None:
        firing = [a for a in amps if len(traces[a][2]) > 0]
        strong = firing[-1] if firing else None
    apf, adapt = None, None
    if strong is not None:
        t, v, sp = traces[strong]
        apf = ephys.ap_features_from_trace(t, v, sp)
        adapt = ephys.adaptation_from_spikes(sp)
    return dict(ir=ir, amps=amps, rates=rates, traces=traces,
                rheobase=rheobase, apf=apf, adapt=adapt, strong=strong)


def in_range(val, rng):
    return rng[0] <= val <= rng[1]


def main():
    plots.setup()
    print("=== 2-4 전기생리 ===")
    b = Bench()
    cells = {"pre": b.pre, "post": b.post}
    tags = {"pre": b.geo["pair"]["pre_tag"], "post": b.geo["pair"]["post_tag"]}
    colors = {"pre": "#2e7d32", "post": "#d84315"}

    R = {}
    for role, cell in cells.items():
        m = measure(cell)
        R[role] = m
        ir = m["ir"]; apf = m["apf"]
        print(f"  [{role}] {tags[role]}")
        print(f"    Rin {ir['Rin_MOhm']:.0f} MOhm · Vrest {ir['vrest_mV']:.1f} mV · "
              f"sag {ir['sag_ratio']:.3f} · rheobase {m['rheobase']:.2f} nA")
        if apf:
            adapt_s = f"{m['adapt']:.2f}" if m['adapt'] is not None else "—"
            print(f"    AP 진폭 {apf['ap_amplitude_mV']:.0f} mV · 반치폭 {apf['ap_halfwidth_ms']:.2f} ms · "
                  f"역치 {apf['ap_threshold_mV']:.1f} mV · 적응 {adapt_s} (@{m['strong']}nA)")
        else:
            print("    AP: 발화 파형 없음 (계단 범위 내 미발화?)")

    # ---- 그림: 2세포 x (f-I 곡선 · 파형 · 판정표) ----
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14.5, 8.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.3, 1.0], hspace=0.42, wspace=0.30)

    checks_all = {}
    for r, role in enumerate(("pre", "post")):
        m = R[role]; c = colors[role]
        # f-I 곡선
        axf = fig.add_subplot(gs[r, 0])
        axf.plot(m["amps"], m["rates"], "-o", color=c, ms=4)
        axf.set_xlabel("주입 전류 (nA)"); axf.set_ylabel("발화율 (Hz)")
        axf.set_title(f"{role} ({tags[role]})\nf-I 곡선", fontsize=10, loc="left")
        axf.axvline(m["rheobase"], ls=":", color="#999", lw=1)
        axf.text(m["rheobase"], axf.get_ylim()[1]*0.1, f" rheobase\n {m['rheobase']:.2f}nA",
                 fontsize=8, color="#666")

        # 막전위 파형 (여러 계단 겹침) — 음전류는 ir trace, 나머지는 f-I trace 재사용
        axv = fig.add_subplot(gs[r, 1])
        axv.plot(m["ir"]["trace"][0], m["ir"]["trace"][1], lw=0.9, label="-0.05 nA")
        show = [a for a in [0.1, 0.2, m["strong"], 0.5] if a is not None and a in m["traces"]]
        for a in sorted(set(show)):
            t, v, _ = m["traces"][a]
            axv.plot(t, v, lw=0.9, label=f"{a} nA")
        axv.set_xlabel("시간 (ms)"); axv.set_ylabel("Vm (mV)")
        axv.set_title("막전위 파형 (계단별)", fontsize=10, loc="left")
        axv.set_xlim(50, 650); axv.legend(fontsize=7, ncol=2, loc="upper right")

        # 판정표
        axt = fig.add_subplot(gs[r, 2]); axt.axis("off")
        ir = m["ir"]; apf = m["apf"]
        nan = float("nan")
        rows = [
            ("Rin (MOhm)", ir["Rin_MOhm"], refdata.CACPYR["Rin_MOhm"]),
            ("Vrest (mV)", ir["vrest_mV"], refdata.CACPYR["Vrest_mV"]),
            ("sag ratio", ir["sag_ratio"], refdata.CACPYR["sag_ratio"]),
            ("rheobase (nA)", m["rheobase"], refdata.CACPYR["rheobase_nA"]),
            ("AP 진폭 (mV)", apf["ap_amplitude_mV"] if apf else nan,
             refdata.CACPYR["AP_amplitude_mV"]),
            ("AP 반치폭 (ms)", apf["ap_halfwidth_ms"] if apf else nan,
             refdata.CACPYR["AP_halfwidth_ms"]),
            ("AP 역치 (mV)", apf["ap_threshold_mV"] if apf else nan,
             refdata.CACPYR["AP_threshold_mV"]),
            ("발화적응", m["adapt"] if m["adapt"] is not None else nan,
             refdata.CACPYR["adaptation_index"]),
        ]
        checks = {}
        y = len(rows)
        axt.text(0.0, y + 0.8, f"{role} — cACpyr 문헌 범위 대조", fontsize=9.5,
                 fontweight="bold", color=c)
        for name, val, rng in rows:
            ok = (not np.isnan(val)) and in_range(val, rng)
            checks[name] = bool(ok)
            col = plots.OK if ok else plots.WARN
            axt.text(0.0, y, name, fontsize=8.5, va="center")
            axt.text(0.52, y, f"{val:.2f}" if not np.isnan(val) else "—",
                     fontsize=8.5, va="center", color=col)
            axt.text(0.74, y, f"[{rng[0]}~{rng[1]}]", fontsize=7.5, va="center", color="#999")
            axt.text(0.99, y, "O" if ok else "~", fontsize=10, va="center", ha="right",
                     color=col, fontweight="bold")
            y -= 1
        axt.set_xlim(0, 1); axt.set_ylim(-0.3, len(rows) + 1.3)
        n_ok = sum(1 for v in checks.values() if v)
        axt.text(0.0, -0.1, f"범위 안 {n_ok}/{len(rows)}", fontsize=8.5,
                 color=plots.OK if n_ok >= len(rows)-1 else plots.WARN)
        checks_all[role] = checks

    fig.suptitle("2-4  두 세포 전기생리 — f-I 곡선 · 막전위 파형 · cACpyr 문헌 범위 대조",
                 fontsize=12.5, y=0.98)
    plots.stamp(fig, "2-4 | 계단전류 500ms · Rin 음전류 -0.05nA · AP/적응 0.3nA · 범위=문헌 근사")
    outdir = plots.figdir(__file__)
    plots.save(fig, outdir, "2-4_ephys_battery.png")

    out = {}
    for role in ("pre", "post"):
        m = R[role]; ir = m["ir"]; apf = m["apf"]
        out[role] = dict(tag=tags[role], Rin_MOhm=round(ir["Rin_MOhm"], 1),
                         vrest_mV=round(ir["vrest_mV"], 2), sag_ratio=round(ir["sag_ratio"], 3),
                         rheobase_nA=round(m["rheobase"], 3),
                         ap_amplitude_mV=round(apf["ap_amplitude_mV"], 1) if apf else None,
                         ap_halfwidth_ms=round(apf["ap_halfwidth_ms"], 2) if apf else None,
                         ap_threshold_mV=round(apf["ap_threshold_mV"], 1) if apf else None,
                         adaptation_index=round(m["adapt"], 3) if m["adapt"] is not None else None,
                         fi_amps=[round(float(a), 3) for a in m["amps"]],
                         fi_rates_hz=[round(float(x), 1) for x in m["rates"]],
                         in_range=checks_all[role])
    jpath = os.path.join(outdir, "2-4_ephys.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved: {jpath}")
    print("\n[통과] 2-4 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
