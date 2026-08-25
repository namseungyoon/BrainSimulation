# -*- coding: utf-8 -*-
"""Ex2b 결과 UI 빌더 (실측) — traces/pair_*.npz(다중ISI STP+train 벤치) → 하나의 UI.
예시와 동일 포맷: STP곡선(실측 vs TM점선)·Train응답·kinetics·대표조합 강조. 미측정=빈칸.
실행: python build_ex2b_results_ui.py   (배치 도는 중 반복 실행 → 누적 갱신)
"""
import os, io, json, glob
import numpy as np
import gen_ex2b_example as EX          # tm_ppr, REP 재사용

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TRD = os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "traces")


def main():
    nodes = [n for n in json.load(io.open(os.path.join(ROOT, "scratch", "connectome_graph.json"), encoding="utf-8"))["nodes"] if n["id"] != "SC"]
    pairs = []
    for f in sorted(glob.glob(os.path.join(TRD, "pair_*.npz"))):
        d = np.load(f, allow_pickle=True)
        isis = [float(x) for x in d["isis"]]
        U, D, F = float(d["U"]), float(d["D"]), float(d["F"])
        pprs = [round(float(x), 3) for x in d["pprs"]]
        pprs_tm = [round(EX.tm_ppr(U, D, F, dt), 3) for dt in isis]     # 이론 오버레이
        key = str(d["pre"]) + "->" + str(d["post"])
        pairs.append({
            "pre": str(d["pre"]), "post": str(d["post"]), "cls": str(d["cls"]), "mech": str(d["mech"]),
            "ns": int(d["ns"]), "gsyn": round(float(d["gsyn"]), 3), "base": round(float(d["base"]), 2), "presp": round(float(d["presp"]), 1),
            "U": U, "D": D, "F": F, "lat": round(float(d["lat"]), 2), "rise": round(float(d["rise"]), 2), "tau": round(float(d["tau"]), 1),
            "stim": float(d["stim"]), "rep_isi": float(d["rep_isi"]), "isis": isis,
            "a1s": [round(float(x), 3) for x in d["a1s"]], "a2s": [round(float(x), 3) for x in d["a2s"]],
            "pprs": pprs, "pprs_meas": pprs, "pprs_tm": pprs_tm,
            "train20": [round(float(x), 3) for x in d["train20"]] if "train20" in d.files else None,
            "freqs": [float(x) for x in d["freqs"]] if "freqs" in d.files else None,
            "freqR": [round(float(x), 3) for x in d["freqR"]] if "freqR" in d.files else None,
            "rep": key in EX.REP, "ref": EX.REP.get(key, ""),
            "t": [round(float(x), 2) for x in d["t"]], "v": [round(float(x), 3) for x in d["v"]], "preV": [round(float(x), 2) for x in d["preV"]],
        })
    out = {"nodes": nodes, "pairs": pairs, "example": False}
    data = json.dumps(out, separators=(",", ":"))
    tpl = io.open(os.path.join(HERE, "ex2b_results_tpl.html"), encoding="utf-8").read()
    outd = os.path.join(ROOT, "04_experiments", "Ex2b_connection_matrix", "ui"); os.makedirs(outd, exist_ok=True)
    io.open(os.path.join(outd, "ex2b_matrix.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", data))
    print(f"[ex2b-results] ex2b_matrix.html · {len(pairs)}/132 측정 완료", flush=True)


if __name__ == "__main__":
    main()
