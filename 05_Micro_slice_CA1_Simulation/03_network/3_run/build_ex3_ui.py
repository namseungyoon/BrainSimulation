# -*- coding: utf-8 -*-
"""Ex3 실측 3D UI 빌더 — 세기(volley) 선택 + 정상/억제차단 토글 + 발화 구름 + fEPSP 흐름.
입력: scratch/ex3_io_traces*.npz (조건별 스파이크+fEPSP) + window_cells + 프레임.
출력: 04_experiments/Ex3_io_inhibition/ui/ex3_io_3d.html
실행: python build_ex3_ui.py [--traces scratch/ex3_io_traces_saturated.npz]
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
NF = 80


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d

TR = arg("--traces", os.path.join(ROOT, "scratch", "ex3_io_traces_saturated.npz"))


def main():
    tr = np.load(TR, allow_pickle=True)
    spk = tr["spk"]; fep = tr["fep"]; elec = tr["elec"].astype(float)
    enames = [str(x) for x in tr["enames"]]; stim = float(tr["stim_t"])
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    xyz = wc["xyz"].astype(float); mt = wc["mtype"].astype(str); is_pc = (mt == "SP_PC")
    N = len(xyz)

    # 국소 프레임 (장축u·층관통r·두께w) + 박스중심
    cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
    fr = cfg["frame_um"]; seed = np.array(fr["seed"])
    M = np.column_stack([fr["long_dir"], fr["radial_dir"], fr["thick_dir"]])
    cl = cfg["window_um"]["center_local"]; off = np.array([cl["u"], cl["r"], 0.0])
    to_local = lambda g: (np.asarray(g, float) - seed) @ M - off
    P = to_local(xyz)
    clay = wc["layer"].astype(str); cr = P[:, 1]
    layers = [{"name": L, "r0": round(float(np.percentile(cr[clay == L], 2)), 0),
               "r1": round(float(np.percentile(cr[clay == L], 98)), 0)}
              for L in ["SO", "SP", "SR", "SLM"] if (clay == L).sum()]
    box = {"u": cfg["window_um"]["long"], "r": cfg["window_um"]["radial"], "w": cfg["window_um"]["thick"]}
    elocal = to_local(elec)
    sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    stim_local = to_local(sc["e3_xyz"].astype(float))

    tf = np.linspace(-2.0, 25.0, NF)                       # 자극기준 프레임(ms)
    conds = []
    for i in range(len(spk)):
        s = spk[i]; f = fep[i]
        st = np.asarray(s["st"], float) - stim; sid = np.asarray(s["sid"], int)
        firefr = {}
        for t, c in zip(st, sid):
            k = int(np.searchsorted(tf, t))
            if 0 <= k < NF and (c not in firefr or k < firefr[c]):
                firefr[c] = k
        fired = [[int(c), int(k)] for c, k in firefr.items()]
        ft = np.asarray(f["t"], float) - stim; Vc = np.asarray(f["V"], float)   # (3, n) µV
        Vi = np.array([np.interp(tf, ft, Vc[j]) for j in range(3)])
        conds.append({"cond": str(s["cond"]), "volley": int(round(float(s["frac"]) * 100)),
                      "fired": fired, "nfired": len(fired),
                      "V": [[round(float(Vi[j][k]), 1) for k in range(NF)] for j in range(3)]})

    out = {
        "ncell": N,
        "pos": [[round(float(P[i, 0]), 1), round(float(P[i, 1]), 1), round(float(P[i, 2]), 1)] for i in range(N)],
        "ispc": [int(x) for x in is_pc],
        "box": box, "layers": layers, "elec_diam": float(cfg["electrodes"]["diameter_um"]),
        "elec": [[round(float(x), 1) for x in elocal[j]] for j in range(len(elec))], "enames": enames,
        "stim_locus": [round(float(x), 1) for x in stim_local],
        "nf": NF, "t": [round(float(x), 2) for x in tf],
        "conds": conds,
    }
    data = json.dumps(out, separators=(",", ":"))
    tpl = os.path.join(HERE, "ex3_io_3d_tpl.html")
    html = open(tpl, encoding="utf-8").read().replace("__INJECT__", data)
    outd = os.path.join(ROOT, "04_experiments", "Ex3_io_inhibition", "ui"); os.makedirs(outd, exist_ok=True)
    outp = os.path.join(outd, "ex3_io_3d.html"); open(outp, "w", encoding="utf-8").write(html)
    print(f"[ex3-ui] {outp} ({len(html)//1024}KB · {N}세포 · {len(conds)}조건 · {NF}프레임)", flush=True)


if __name__ == "__main__":
    main()
