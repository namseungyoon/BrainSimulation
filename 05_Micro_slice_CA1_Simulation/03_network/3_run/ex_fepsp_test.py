# -*- coding: utf-8 -*-
"""fEPSP 인프라 소규모 검증 — locus 근처 PC 소수만 빌드 → SC volley → 막전류 fEPSP.
큰 런 전 mea_forward+fepsp_record가 실제 NEURON 전류에서 **E3(SR) 음성 sink**를
내는지·단위가 맞는지 확인. 단일 프로세스(빠름). 결과: figures/fepsp_test.png + 콘솔.
실행: python 03_network/3_run/ex_fepsp_test.py [--r 120] [--cap 60]
"""
import os, sys, json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
DERIVED = os.path.join(ROOT, "data", "derived")

def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
R = arg("--r", 120.0); CAP = arg("--cap", 60); GSCALE = arg("--gscale", 1.0)
STIM = 5.0; OBS = 40.0; TSTOP = STIM + OBS


def seg_kdtree(cell):
    P, ref = [], []
    for sec in cell.all:
        n = int(sec.n3d())
        if n < 2:
            continue
        Lt = sec.arc3d(n - 1) or 1.0
        for i in range(n):
            P.append((sec.x3d(i), sec.y3d(i), sec.z3d(i)))
            ref.append((sec, min(max(sec.arc3d(i) / Lt, 0.0), 1.0)))
    return cKDTree(np.array(P)), ref


def main():
    import net_build as nb
    import mea_forward as mf
    import fepsp_record as fr
    B = nb.NetBuilder(); h.load_file("stdrun.hoc"); h.cvode_active(0)
    wc = np.load(os.path.join(DERIVED, "window_cells.npz"), allow_pickle=True)
    XYZ = wc["xyz"].astype(float); Q = wc["orientation_wxyz"]; mt = B.mt
    cfg = json.load(open(os.path.join(ROOT, "config", "window_layout.json"), encoding="utf-8"))
    elec_list = cfg["electrodes"]["list"]
    elec = np.array([e["xyz_um"] for e in elec_list], float)         # E1,E2,E3 global
    enames = [e["id"] + "(" + e["layer"] + ")" for e in elec_list]
    sc = np.load(os.path.join(DERIVED, "sc_synapses.npz"), allow_pickle=True)
    fib = np.load(os.path.join(DERIVED, "sc_fibers.npz"), allow_pickle=True)
    scpost = sc["post_gid"].astype(int); scxyz = sc["xyz"].astype(float)
    fiber_id = fib["fiber_id"].astype(int); locus = sc["e3_xyz"].astype(float)

    is_pc = (mt == "SP_PC")
    dist_e3 = sc["dist_e3"].astype(float)
    near = np.where(dist_e3 < R)[0]                      # locus 근처 SC 시냅스
    npost = scpost[near]; npost = npost[is_pc[npost]]    # 그 중 PC 타깃
    uid, cnt = np.unique(npost, return_counts=True)      # PC별 근처 시냅스 수
    cand = uid[np.argsort(cnt)[::-1][:CAP]]              # 근처 입력 많은 PC 우선
    print(f"[fepsp-test] locus(dist_e3<{R}) 근처 입력 받는 PC {len(cand)}개 (cap={CAP})", flush=True)

    rec = fr.FEPSPRecorder(elec)
    cells = {}; keep = []; t0 = time.time()
    for g in cand:
        cell = B.build_cell(g); cells[g] = cell
        rec.add_cell(cell, XYZ[g], Rot.from_quat(Q[g][[1, 2, 3, 0]]))
    print(f"[fepsp-test] 빌드 {len(cand)}세포 · 세그먼트 {rec.n_seg()} · {time.time()-t0:.0f}s", flush=True)

    # subset 세포에 닿는 SC 시냅스·섬유
    scsel = np.where(np.isin(scpost, cand))[0]
    fibers = np.unique(fiber_id[scsel])
    vstim = {}
    for f in fibers:
        vs = h.VecStim(); tv = h.Vector([STIM]); vs.play(tv); vstim[int(f)] = vs; keep.append((vs, tv))
    nsyn = 0; syn_pos = []
    for g in cand:
        cell = cells[g]; tree, ref = seg_kdtree(cell); rot = Rot.from_quat(Q[g][[1, 2, 3, 0]]); soma = cell.soma[0]
        for si in scsel[scpost[scsel] == g]:
            mp = rot.inv().apply(scxyz[si] - XYZ[g]); _, k = tree.query(mp, k=1); sec, x = ref[k]
            syn_pos.append(scxyz[si])
            syn = h.GBPlasticityStpProbSyn(sec(x))
            syn.Use = 0.14; syn.Dep = 186.0; syn.Fac = 129.0; syn.Nrrp = 12; syn.gmax = 0.8 / 1000.0 * GSCALE
            syn.gamma_p = 0.0; syn.gamma_d = 0.0; syn.setRNG(int(g) + 1, int(si) + 1, 3)
            nc = h.NetCon(vstim[int(fiber_id[si])], syn); nc.weight[0] = 1.0; nc.delay = 1.0
            ncs = h.NetCon(soma(0.5)._ref_v, syn, sec=soma); ncs.weight[0] = -1.0
            keep.append((syn, nc, ncs)); nsyn += 1
    print(f"[fepsp-test] SC 시냅스 {nsyn} · 섬유 {len(fibers)} (volley t={STIM}ms)", flush=True)

    # 소마 스파이크 카운트
    spk = h.Vector(); ncsp = []
    fired = h.Vector()
    apc = []
    for g in cand:
        a = h.APCount(cells[g].soma[0](0.5)); a.thresh = -10; apc.append(a)

    rec.finalize(rec_dt=0.1)
    h.celsius = 34; h.dt = 0.025
    h.finitialize(-70); h.continuerun(TSTOP)

    V = rec.potential_local()          # (3, nt) mV  (단일프로세스)
    t = rec.times()
    nfired = sum(1 for a in apc if a.n > 0)
    # 각 전극 자극후 피크(절댓값 최대) 시각·값
    m = t >= STIM
    print(f"[fepsp-test] 발화 {nfired}/{len(cand)} PC", flush=True)
    for i, nm in enumerate(enames):
        seg = V[i][m]; tt = t[m]
        k = np.argmax(np.abs(seg - seg[0]))
        print(f"   {nm}: 피크 {seg[k]-seg[0]:+.4f} mV @ {tt[k]-STIM:.1f}ms (baseline {seg[0]:+.4f})", flush=True)
    # 판정: E3(SR)가 자극후 음(–)으로 편향?
    e3 = V[2][m] - V[2][m][0]
    print(f"[fepsp-test] E3(SR) 자극후 최소 {e3.min():+.4f} mV → {'sink(음성) OK' if e3.min()<0 else '주의: 음성 아님'}", flush=True)

    # 그림
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ["#33d6c4", "#5b93ff", "#ff6b6b"]
    for i, nm in enumerate(enames):
        ax.plot(t - STIM, V[i] - V[i][t <= STIM].mean(), color=cols[i], lw=1.8, label=nm)
    ax.axvline(0, color="crimson", ls="--", lw=1)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("time from stim [ms]"); ax.set_ylabel("field potential [mV]")
    ax.set_title(f"fEPSP infra test — {len(cand)} PC near locus, {nfired} fired\n(E3/SR should dip negative = synaptic sink)")
    ax.legend(); ax.set_xlim(-2, OBS)
    FIG = os.path.join(ROOT, "04_experiments", "Ex4_fepsp", "figures"); os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "fepsp_test.png"); fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"[fepsp-test] 그림 {p} · 총 {time.time()-t0:.0f}s", flush=True)

    # ── 3D UI 데이터 저장 + 자립형 HTML 생성 (--save) ──
    if "--save" in sys.argv:
        nseg = rec.n_seg()
        pos = np.array(rec._pos)                                    # (nseg,3) global
        Im = np.array([np.array(v) for v in rec.vecs])             # (nseg, nt) nA
        soma = np.array(rec._soma); ttv = rec.times()
        fstep = max(1, len(ttv) // 80); fidx = np.arange(0, len(ttv), fstep)
        tf = ttv[fidx] - STIM
        keep = soma.copy(); others = np.where(~soma)[0]
        sstep = max(1, len(others) // 9000); keep[others[::sstep]] = True
        sidx = np.where(keep)[0]
        P = pos[sidx]; c = P.mean(0); P = P - c
        Isub = Im[np.ix_(sidx, fidx)]                              # (nsub, nframe)
        cmax = float(np.percentile(np.abs(Im[:, fidx]), 99.5)) or 1.0
        cur = np.clip(np.round(Isub / cmax * 100), -100, 100).astype(int)
        Vf = V[:, fidx]                                            # (3, nframe) mV
        somai = np.where(soma)[0]; dendi = np.where(~soma)[0]
        Isoma_t = Im[np.ix_(somai, fidx)].sum(axis=0)              # 소마 총 막전류 (nA)
        Idend_t = Im[np.ix_(dendi, fidx)].sum(axis=0)              # 수상돌기 총 막전류 (nA)
        out = {
            "n": int(len(sidx)), "nf": int(len(fidx)),
            "pos": [[round(float(P[i, 0]), 1), round(float(P[i, 1]), 1), round(float(P[i, 2]), 1)] for i in range(len(P))],
            "soma": [int(x) for x in soma[sidx]],
            "cur": [[int(cur[i, f]) for i in range(cur.shape[0])] for f in range(cur.shape[1])],  # [frame][seg]
            "t": [round(float(x), 2) for x in tf],
            "elec": [[round(float(x), 1) for x in (elec[i] - c)] for i in range(len(elec))],
            "enames": enames,
            "V": [[round(float(Vf[i, f] * 1000), 3) for f in range(Vf.shape[1])] for i in range(3)],  # µV
            "Isoma": [round(float(x), 3) for x in Isoma_t],        # 소마 총 막전류 (nA)
            "Idend": [round(float(x), 3) for x in Idend_t],        # 수상돌기 총 막전류 (nA)
            "syn": [[round(float(v), 1) for v in (np.array(syn_pos)[k] - c)]
                    for k in range(0, len(syn_pos), max(1, len(syn_pos) // 2500))],  # 자극 시냅스 위치(서브샘플)
            "cmax_nA": round(cmax, 4), "nfired": int(nfired), "ncell": int(len(cand)), "stim": 0.0,
        }
        data = json.dumps(out, separators=(",", ":"))
        open(os.path.join(ROOT, "scratch", "fepsp_3d.json"), "w").write(data)
        tpl = os.path.join(HERE, "ex_fepsp_3d_tpl.html")
        if os.path.exists(tpl):
            html = open(tpl, encoding="utf-8").read().replace("__INJECT__", data)
            uidir = os.path.join(ROOT, "04_experiments", "Ex4_fepsp", "ui"); os.makedirs(uidir, exist_ok=True)
            outp = os.path.join(uidir, "fepsp_3d_50cell.html"); open(outp, "w", encoding="utf-8").write(html)
            print(f"[fepsp-test] 3D UI -> {outp} ({len(html)//1024}KB · 세그 {out['n']} · 프레임 {out['nf']})", flush=True)


if __name__ == "__main__":
    main()
