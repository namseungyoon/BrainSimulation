# -*- coding: utf-8 -*-
"""전체망 fEPSP 3D UI 빌더 — scratch/mpi_fepsp.npz → ex_fepsp_3d_tpl.html 재사용.
입력: mpi_fepsp.npz(segpos·segsoma·segI·V·t·elec·enames) + mpi_baseline.npz(발화수) + sc_synapses(자극 시냅스).
출력: 04_experiments/Ex1_baseline/ui/fepsp_3d_full.html
실행: python 03_network/3_run/build_fepsp3d_full.py
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DERIVED = os.path.join(ROOT, "data", "derived")
NFRAME = 90


def main():
    fe = np.load(os.path.join(ROOT, "scratch", "mpi_fepsp.npz"), allow_pickle=True)
    segpos = fe["segpos"].astype(float); segsoma = fe["segsoma"].astype(int); segI = fe["segI"].astype(float)
    V = fe["V"].astype(float); t = fe["t"].astype(float); elec = fe["elec"].astype(float)
    enames = [str(x) for x in fe["enames"]]; stim = float(fe["stim_t"])

    nt = len(t); fstep = max(1, nt // NFRAME); fi = np.arange(0, nt, fstep)
    tf = t[fi] - stim
    # ── 국소 해부 프레임(장축 u · 층관통 r · 두께 w)으로 변환 + 박스중심 정렬 ──
    cfgw = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
    frw = cfgw["frame_um"]; seed = np.array(frw["seed"])
    Mfr = np.column_stack([frw["long_dir"], frw["radial_dir"], frw["thick_dir"]])   # global→local(u,r,w)
    cl = cfgw["window_um"]["center_local"]; off = np.array([cl["u"], cl["r"], 0.0])
    to_local = lambda g: (np.asarray(g, float) - seed) @ Mfr - off                  # 박스중심 정렬
    P = to_local(segpos)                                    # (nseg,3) = (u,r,w)
    Isub = segI[:, fi]
    cmax = float(np.percentile(np.abs(Isub), 99.5)) or 1.0
    cur = np.clip(np.round(Isub / cmax * 100), -100, 100).astype(int)
    Vf = V[:, fi]
    sm = segsoma.astype(bool)
    Isoma = segI[np.ix_(np.where(sm)[0], fi)].sum(0) if sm.any() else np.zeros(len(fi))
    Idend = segI[np.ix_(np.where(~sm)[0], fi)].sum(0) if (~sm).any() else np.zeros(len(fi))

    # 자극 시냅스(구동된 것) 위치 → 국소
    sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    scxyz = sc["xyz"].astype(float); dist = sc["dist_e3"].astype(float)
    drv = np.where(dist < 150.0)[0]
    synP = to_local(scxyz[drv]); synP = synP[::max(1, len(synP) // 2500)]
    # 층 경계(r), 슬라이스 박스, 전극·자극 국소좌표
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    clay = wc["layer"].astype(str); cr = to_local(wc["xyz"].astype(float))[:, 1]
    layers = [{"name": L, "r0": round(float(np.percentile(cr[clay == L], 2)), 0),
               "r1": round(float(np.percentile(cr[clay == L], 98)), 0)}
              for L in ["SO", "SP", "SR", "SLM"] if (clay == L).sum()]
    box = {"u": cfgw["window_um"]["long"], "r": cfgw["window_um"]["radial"], "w": cfgw["window_um"]["thick"]}
    elocal = to_local(elec); stim_local = to_local(sc["e3_xyz"].astype(float))
    edia = float(cfgw["electrodes"]["diameter_um"])

    try:
        b = np.load(os.path.join(ROOT, "scratch", "mpi_baseline.npz"), allow_pickle=True)
        nfired = int(b["fired"].sum()); ncell = int(b["n"])
    except Exception:
        nfired = 0; ncell = len(P)

    out = {
        "n": int(len(P)), "nf": int(len(fi)),
        "pos": [[round(float(P[i, 0]), 1), round(float(P[i, 1]), 1), round(float(P[i, 2]), 1)] for i in range(len(P))],
        "soma": [int(x) for x in segsoma],
        "cur": [[int(cur[i, f]) for i in range(cur.shape[0])] for f in range(len(fi))],
        "t": [round(float(x), 2) for x in tf],
        "elec": [[round(float(x), 1) for x in elocal[j]] for j in range(len(elec))],
        "enames": enames,
        "V": [[round(float(Vf[i, f] * 1000), 3) for f in range(len(fi))] for i in range(len(elec))],
        "Isoma": [round(float(x), 3) for x in Isoma], "Idend": [round(float(x), 3) for x in Idend],
        "syn": [[round(float(v), 1) for v in synP[k]] for k in range(len(synP))],
        "ncell": ncell, "nfired": nfired, "cmax_nA": round(cmax, 4), "stim": 0.0,
        "box": box, "layers": layers, "elec_diam": edia,
        "stim_locus": [round(float(x), 1) for x in stim_local],
        "axes": {"u": "장축 500µm", "r": "층관통 800µm", "w": "두께 400µm"},
    }
    data = json.dumps(out, separators=(",", ":"))
    tpl = os.path.join(HERE, "ex_fepsp_3d_tpl.html")
    html = open(tpl, encoding="utf-8").read().replace("__INJECT__", data)
    outd = os.path.join(ROOT, "04_experiments", "Ex4_fepsp", "ui"); os.makedirs(outd, exist_ok=True)
    outp = os.path.join(outd, "fepsp_3d_full.html"); open(outp, "w", encoding="utf-8").write(html)
    print(f"[fepsp3d-full] {outp} ({len(html)//1024}KB · 세그 {out['n']} · 프레임 {out['nf']} · 시냅스 {len(synP)} · cmax {cmax:.3f}nA)", flush=True)


if __name__ == "__main__":
    main()
