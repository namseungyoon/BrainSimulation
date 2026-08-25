# -*- coding: utf-8 -*-
"""Ex3 I-O 지표표 생성 — E3(SR) fEPSP의 slope·tpeak·peak를 normal/block 나란히.
입력: scratch/ex3_io_traces*.npz  (조건별 스파이크+fEPSP 파형)
출력: 04_experiments/Ex3_io_inhibition/<out>.md  (Notion용 파이프 표 + 요약)
실행: python ex3_metrics_table.py [--traces scratch/ex3_io_traces_saturated.npz] [--out ex3_metrics_saturated.md] [--elec E3]
지표 정의:
  - Vpeak : 관측창(0~25ms)에서 가장 음성인 값(µV)  ← sink 최고점
  - tpeak : 그 peak까지의 자극 후 잠복기(ms)
  - slope : 0.2ms~tpeak 상승부의 최대 하강기울기 dV/dt 최소값(µV/ms)  ← fEPSP slope 대용
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


TR = arg("--traces", os.path.join(ROOT, "scratch", "ex3_io_traces_saturated.npz"))
OUT = arg("--out", "ex3_metrics_saturated.md")
ELEC = arg("--elec", "E3")


def metrics(f, i):
    t = np.asarray(f["t"], float) - STIM
    V = np.asarray(f["V"], float)[i]
    base = V[t < 0].mean() if (t < 0).any() else 0.0          # 자극 전 baseline 차감(라이브 로그와 일치)
    V = V - base
    m = (t >= 0) & (t <= 25); tw = t[m]; Vw = V[m]
    k = int(np.argmin(Vw)); Vpk = float(Vw[k]); tpk = float(tw[k])
    sel = (tw >= 0.2) & (tw <= tpk)
    slope = float(np.gradient(Vw[sel], tw[sel]).min()) if sel.sum() >= 2 else float("nan")
    return slope, tpk, Vpk


def main():
    global STIM
    tr = np.load(TR, allow_pickle=True)
    fep = tr["fep"]; spk = tr["spk"]; STIM = float(tr["stim_t"])
    en = [str(x) for x in tr["enames"]]
    ie = next((j for j, e in enumerate(en) if e.startswith(ELEC)), 1)   # "E3(SR)" 접두어 매칭

    byfrac = {}
    for i in range(len(fep)):
        c = str(spk[i]["cond"]); fr = int(round(float(spk[i]["frac"]) * 100))
        nf = len(set(np.asarray(spk[i]["sid"], int)))
        byfrac.setdefault(fr, {})[c] = (*metrics(fep[i], ie), nf)

    L = []
    L.append(f"### Ex3 I-O 지표표 — {ELEC}(SR) fEPSP · 억제 전(normal)/후(block)\n")
    L.append("| 세기(volley%) | slope N (µV/ms) | slope B | tpeak N (ms) | tpeak B | peak N (µV) | peak B | 발화 N | 발화 B |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for fr in sorted(byfrac):
        n = byfrac[fr].get("normal"); b = byfrac[fr].get("block")
        if n is None or b is None:
            continue
        L.append(f"| {fr} | {n[0]:.0f} | {b[0]:.0f} | {n[1]:.2f} | {b[1]:.2f} | "
                 f"{n[2]:.0f} | {b[2]:.0f} | {n[3]} ({n[3]*100//5610}%) | {b[3]} ({b[3]*100//5610}%) |")
    L.append("")
    L.append("- **slope**: 0.2ms~tpeak 상승부 최대 하강기울기(음성이 강할수록 큰 응답). "
             "**tpeak**: peak 잠복기(집단스파이크 등장 시 짧아짐). **peak**: E3 최음성값(sink 크기).")
    txt = "\n".join(L)
    outd = os.path.join(ROOT, "04_experiments", "Ex3_io_inhibition")
    os.makedirs(outd, exist_ok=True)
    outp = os.path.join(outd, OUT)
    open(outp, "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    print(f"\n[ex3-metrics] saved -> {outp}", flush=True)


if __name__ == "__main__":
    main()
