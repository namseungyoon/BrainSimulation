# -*- coding: utf-8 -*-
"""θ위상 UI 템플릿에 시뮬 데이터 주입 → 로컬 HTML(자극패턴+결과). git 미push.
실행: python 03_network/3_run/fill_theta_ui.py [sweep_tag]   (기본 sweep_ncyc1)
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATD = os.path.join(ROOT, "04_experiments", "Ex_theta_phase", "data")
UID = os.path.join(ROOT, "04_experiments", "Ex_theta_phase", "ui")
TAG = sys.argv[1] if len(sys.argv) > 1 else "sweep_ncyc1"


def win_ds(t, arrs, spk, n=420):
    lo, hi = spk.min() - 40, spk.max() + 180
    m = (t >= lo) & (t <= hi)
    tt = t[m] - spk.min()
    idx = np.linspace(0, len(tt) - 1, min(n, len(tt))).astype(int)
    return tt[idx], {k: v[m][idx] for k, v in arrs.items()}


def main():
    meta = json.load(open(os.path.join(DATD, f"theta_phase_{TAG}.json"), encoding="utf-8"))
    tr = np.load(os.path.join(DATD, f"theta_phase_{TAG}_traces.npz"))
    res = {int(r["phase"]): r for r in meta["results"]}
    traces = {}
    for p in (0, 180):
        if p not in res:
            continue
        t = tr[f"p{p}_t"]; spk = tr[f"p{p}_spk"]
        tt, a = win_ds(t, {"c": tr[f"p{p}_c"], "v": tr[f"p{p}_v"]}, spk)
        traces[str(p)] = {"t": [round(float(x), 2) for x in tt],
                          "c": [round(float(x), 4) for x in a["c"]],
                          "v": [round(float(x), 2) for x in a["v"]]}
    phases = sorted(res.keys())
    sweep = {"phase": phases, "dr": [round(res[p]["dr"], 4) for p in phases],
             "npost": [res[p]["npost"] for p in phases], "cpeak": [round(res[p]["c_peak"], 3) for p in phases]}
    metao = {k: meta[k] for k in ("gid", "thetaHz", "A_theta", "Bamp", "npulse", "burstHz", "ncyc",
                                  "theta_d", "theta_p", "C_pre", "C_post", "tau_ca", "sc")}
    DATA = {"meta": metao, "traces": traces, "sweep": sweep}

    tpl = open(os.path.join(UID, "theta_phase_ui_tpl.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/ null", "/*__DATA__*/ " + json.dumps(DATA, ensure_ascii=False))
    out = os.path.join(UID, f"theta_phase_ui_{TAG}.html")
    open(out, "w", encoding="utf-8").write(html)
    print("저장:", out, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
