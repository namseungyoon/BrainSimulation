# -*- coding: utf-8 -*-
"""Ex2b 결과 UI 빌더 — traces/pair_*.npz(2세포 벤치 측정) → 하나의 인터랙티브 UI.
매트릭스(색=PPR 억압/촉진, 진하기=|uPSP|) + 셀 클릭 시 파형·지표. 미측정 쌍은 빈칸.
실행: python build_ex2b_results_ui.py   (배치 도는 중에 반복 실행하면 누적 갱신)
"""
import os, io, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TRD = os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "traces")
NPTS = 360


def ds(t, *arrs, lo, hi):
    m = (t >= lo) & (t <= hi)
    tt = t[m]
    if len(tt) <= NPTS:
        idx = np.arange(len(tt))
    else:
        idx = np.linspace(0, len(tt) - 1, NPTS).astype(int)
    return [np.round(tt[idx], 2).tolist()] + [np.round(a[m][idx], 3).tolist() for a in arrs]


def main():
    nodes = json.load(io.open(os.path.join(ROOT, "scratch", "connectome_graph.json"), encoding="utf-8"))["nodes"]
    nodes = [n for n in nodes if n["id"] != "SC"]
    pairs = []
    for f in sorted(glob.glob(os.path.join(TRD, "pair_*.npz"))):
        d = np.load(f, allow_pickle=True)
        stim = float(d["stim"]); isi = float(d["isi"])
        t, v, pv = ds(np.asarray(d["t"], float), np.asarray(d["v"], float), np.asarray(d["preV"], float),
                      lo=stim - 8, hi=stim + isi + 70)
        pairs.append({
            "pre": str(d["pre"]), "post": str(d["post"]), "cls": str(d["cls"]), "mech": str(d["mech"]),
            "ns": int(d["ns"]), "gsyn": round(float(d["gsyn"]), 3), "base": float(d["base"]),
            "a1": float(d["a1"]), "a2": float(d["a2"]), "ppr": float(d["ppr"]), "presp": float(d["presp"]),
            "stim": stim, "isi": isi, "U": float(d["U"]), "D": float(d["D"]), "F": float(d["F"]),
            "t": t, "v": v, "preV": pv,
        })
    out = {"nodes": nodes, "pairs": pairs}
    data = json.dumps(out, separators=(",", ":"))
    tpl = io.open(os.path.join(HERE, "ex2b_results_tpl.html"), encoding="utf-8").read()
    outd = os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "ui"); os.makedirs(outd, exist_ok=True)
    outp = os.path.join(outd, "ex2b_results.html")
    io.open(outp, "w", encoding="utf-8").write(tpl.replace("__DATA__", data))
    print(f"[ex2b-results] {outp} ({len(data)//1024}KB · {len(pairs)}/132 측정 완료)", flush=True)


if __name__ == "__main__":
    main()
