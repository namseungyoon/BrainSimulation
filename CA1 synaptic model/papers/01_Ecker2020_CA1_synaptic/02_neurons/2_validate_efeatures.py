"""
2_validate_efeatures.py — 뉴런 검증: eFEL 로 f-I·AP파형·Rin 추출 (세포 20개)
============================================================================
Source: Ecker(2020) §2.2; Migliore(2018) 단일세포 e-model; eFEL(BBP).

각 세포(피라미드1+인터뉴런19)를 격리 프로세스로 로드 → 전류계단 스윕 →
eFEL 로 발화 특징을 뽑아 "진짜 CA1 세포처럼 발화하나" 검증.
  - f-I 곡선(전류↔스파이크수)  - AP 진폭/반치폭/역치  - 입력저항 Rin  - rheobase

이중 모드:
  python 2_validate_efeatures.py --cell <model_dir>   # 워커: 1세포 → JSON 출력
  python 2_validate_efeatures.py                       # 부모: 20세포 수집 → 그림
"""
import os
import sys
import json
import subprocess

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS)))   # 02_neurons→paper→papers→root
SHARED = os.path.join(ROOT, "shared")
MODELS = os.path.join(SHARED, "models")
OUT = os.path.join(THIS, "figures")
sys.path.insert(0, SHARED)
from common.model_naming import etype_of   # noqa: E402

# 전류계단(nA): 음수=Rin/sag, 양수=발화
AMPS = [-0.05, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
DELAY, DUR, TSTOP = 200.0, 500.0, 800.0


# ----------------------------- 워커 -----------------------------
def run_worker(model_dir):
    import numpy as np
    import efel
    from common.nrn_env import h
    from common.cell_loader import load_cell
    try:
        efel.set_setting("Threshold", -20.0)
    except Exception:
        pass
    h.load_file("stdrun.hoc")
    cell, tname = load_cell(model_dir)
    soma = cell.soma[0]
    ic = h.IClamp(soma(0.5)); ic.delay = DELAY; ic.dur = DUR
    vv = h.Vector().record(soma(0.5)._ref_v)
    tv = h.Vector().record(h._ref_t)
    h.celsius = 34.0
    h.cvode_active(1)

    def simulate(amp):
        ic.amp = amp
        h.finitialize(-70.0)
        h.continuerun(TSTOP)
        return np.array(tv), np.array(vv)

    def efeat(t, v, names):
        tr = {"T": t, "V": v, "stim_start": [DELAY], "stim_end": [DELAY + DUR]}
        try:
            return efel.get_feature_values([tr], names, raise_warnings=False)[0] or {}
        except Exception:
            return {}

    fI, ap = [], None
    rin = None
    for amp in AMPS:
        t, v = simulate(amp)
        sc_raw = efeat(t, v, ["Spikecount"]).get("Spikecount")
        sc = int(sc_raw[0]) if sc_raw is not None and len(sc_raw) else 0
        fI.append((amp, sc))
        # AP 파형: 스파이크 있는 첫 양수 step 에서
        if ap is None and amp > 0 and sc >= 1:
            f = efeat(t, v, ["AP_amplitude", "AP_width", "AP_begin_voltage", "AHP_depth_abs"])
            def m(key):
                a = f.get(key)
                return float(np.mean(a)) if a is not None and len(a) else None
            ap = dict(amp=amp, AP_amplitude=m("AP_amplitude"), AP_width=m("AP_width"),
                      AP_threshold=m("AP_begin_voltage"), AHP=m("AHP_depth_abs"))
        # Rin: 음수 step 에서 ΔV/|I|
        if amp < 0 and rin is None:
            i0 = np.searchsorted(t, DELAY - 20); i1 = np.searchsorted(t, DELAY - 1)
            j0 = np.searchsorted(t, DELAY + DUR - 80); j1 = np.searchsorted(t, DELAY + DUR - 1)
            base = float(np.mean(v[i0:i1])); steady = float(np.mean(v[j0:j1]))
            rin = abs(base - steady) / abs(amp)   # MΩ (mV/nA)

    rheobase = next((a for a, s in fI if a > 0 and s >= 1), None)
    print("EFEAT_JSON " + json.dumps(dict(
        template=tname, fI=fI, rheobase=rheobase, Rin=rin, **(ap or {}))))


# ----------------------------- 부모 -----------------------------
def list_cells():
    cells = []
    pyr = os.path.join(MODELS, "pyramidal")
    for d in sorted(os.listdir(pyr)):
        cells.append(("PC", "cACpyr", os.path.join(pyr, d)))
    intd = os.path.join(MODELS, "interneurons")
    for d in sorted(os.listdir(intd)):
        et = etype_of(d) or "?"
        cells.append(("INT", et, os.path.join(intd, d)))
    return cells


def run_parent():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(SHARED))
    from common.plotstyle import set_korean_font
    set_korean_font()
    os.makedirs(OUT, exist_ok=True)

    cells = list_cells()
    results = []
    print(f"[검증] 세포 {len(cells)}개 격리 프로세스로 eFEL 추출 …")
    for role, et, d in cells:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--cell", d],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("EFEAT_JSON ")), None)
        if not line:
            print(f"  [실패] {os.path.basename(d)}: {r.stderr.strip().splitlines()[-1:]}")
            continue
        data = json.loads(line[len("EFEAT_JSON "):])
        data.update(role=role, etype=et, name=os.path.basename(d))
        results.append(data)
        print(f"  {role:3s} {et:5s} rheo={data['rheobase']} Rin={data['Rin'] and round(data['Rin'])}MΩ "
              f"APamp={data.get('AP_amplitude') and round(data['AP_amplitude'],1)}")

    # ---- 그림: (A) f-I, (B) Rin, (C) AP 반치폭 vs 진폭 ----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.9))
    fig.suptitle(f"뉴런 검증 — eFEL 전기생리 특징 ({len(results)}개 세포)",
                 fontsize=13, fontweight="bold")
    col = {"PC": "tab:red", "bAC": "tab:green", "cAC": "tab:blue", "cNAC": "tab:orange"}
    for d in results:
        amps = [a for a, s in d["fI"]]; scs = [s for a, s in d["fI"]]
        axA.plot(amps, scs, "-", color=col.get(d["etype"] if d["role"] == "INT" else "PC", "0.5"),
                 lw=1, alpha=0.6)
    axA.set_title("(A) f–I 곡선 (전류→발화수)", fontsize=10)
    axA.set_xlabel("주입 전류 (nA)"); axA.set_ylabel("스파이크 수 (500ms)")
    from matplotlib.lines import Line2D
    axA.legend(handles=[Line2D([0], [0], color=c, lw=2) for c in
                        ["tab:red", "tab:green", "tab:blue", "tab:orange"]],
               labels=["PC", "INT-bAC", "INT-cAC", "INT-cNAC"], fontsize=8)

    rins = [(d["name"], d["Rin"], d["role"]) for d in results if d["Rin"]]
    axB.bar(range(len(rins)), [r[1] for r in rins],
            color=["tab:red" if r[2] == "PC" else "tab:blue" for r in rins])
    axB.set_title("(B) 입력저항 Rin (세포별)", fontsize=10)
    axB.set_xlabel("세포"); axB.set_ylabel("Rin (MΩ)"); axB.set_xticks([])

    for d in results:
        if d.get("AP_width") and d.get("AP_amplitude"):
            axC.scatter(d["AP_width"], d["AP_amplitude"],
                        color="tab:red" if d["role"] == "PC" else "tab:blue", s=40, alpha=0.7)
    axC.set_title("(C) AP 반치폭 vs 진폭", fontsize=10)
    axC.set_xlabel("AP 반치폭 (ms)"); axC.set_ylabel("AP 진폭 (mV)")
    axC.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9)
                        for c in ["tab:red", "tab:blue"]], labels=["PC", "인터뉴런"], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "2_validate_efeatures.png")
    plt.savefig(out, dpi=120)
    print(f"[그림] {out}")
    ok = sum(1 for d in results if d["rheobase"] and d["Rin"] and d["Rin"] > 0)
    print(f"[검증] {ok}/{len(results)} 세포: f-I 발화·Rin>0 정상 (rheobase·입력저항 추출됨)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--cell":
        run_worker(sys.argv[2])
    else:
        run_parent()
