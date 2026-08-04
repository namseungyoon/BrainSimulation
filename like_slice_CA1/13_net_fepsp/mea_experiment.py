# -*- coding: utf-8 -*-
"""13_net_fepsp/mea_experiment.py  —  in silico MEA 실험 (실제 in vitro 프로토콜 재현)

실제 Schaffer-collateral fEPSP 실험을 그대로: MEA 전극 1개=자극(국소 SC 활성),
나머지=기록(fEPSP slope). 전세포 실제 동역학(net_fepsp 엔진) + MoI fEPSP.
프로토콜(--protocol):
  io    : Input-Output 곡선 — 자극세기(활성 SC 섬유 수) 스윕 → fEPSP slope
  ppf   : Paired-Pulse — ISI 스윕 → PPR=slope2/slope1 (SC->PC E1s 촉진)
  (ltp는 별도 확장: GBPlasticitySyn + TBS)
실행(서브셋): <ca1sim>/py mea_experiment.py --counts 300,80,60,60 --protocol io --tstop 80
실행(전규모): bash _wsl_net_fepsp.sh 20 mea_experiment.py --counts full --protocol io  (드라이버 재사용)
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRAIN = os.path.dirname(ROOT)
SHARED = os.path.join(BRAIN, "shared")
PAPER = os.path.join(BRAIN, "papers", "01_Ecker2020_CA1_synaptic")
for p in (SHARED, os.path.join(PAPER, "03_synapses"), os.path.join(PAPER, "04_network"), HERE,
          os.path.join(ROOT, "12_lfp")):
    sys.path.insert(0, p)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from common.nrn_env import h
from common.cell_loader import load_cell
import network_lib as net
import params_table3 as P3
from synapse_pair import build_synapse
import lfp_calc as L
from scipy.spatial import cKDTree

pc = h.ParallelContext()
RANK = int(pc.id()); NHOST = int(pc.nhost())
MODELS = os.environ.get("MODELS_DIR") or os.path.join(SHARED, "models")
CELLS = os.path.join(ROOT, "05_placement", "slice_cells.npz")
PRUNED = os.path.join(ROOT, "07_connectivity", "pruned_connectivity.npz")
FIG = os.path.join(HERE, "figures")
if RANK == 0:
    os.makedirs(FIG, exist_ok=True)
ETYPE_TO_T4 = {"cACpyr": "PC", "cNAC": "PV", "cAC": "cAC", "bAC": "bAC"}
SYN_DELAY = 1.0
SIG_T, SIG_S, SIG_G, N_IMG = 0.3, 1.5, 0.0, 20
PITCH, R_ON, NCOL, NROW = 200.0, 100.0, 8, 3
Z_GLASS_MARGIN = 20.0


def log(m):
    if RANK == 0:
        print(m, flush=True)


def argval(flag, d):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


def quat_to_R(q):
    w, x, y, z = q; n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def measure_fepsp(t, v, t0, dur=30.0):
    """자극 t0 후 dur창: 음성 fEPSP 진폭 + 초기 slope(mV/ms). 20~80% 하강 선형회귀.
    (12_lfp/e4a_fepsp.py measure_fepsp 이식)."""
    m = (t >= t0) & (t < t0 + dur)
    tt = t[m]; vv = v[m] - (v[m][0] if m.sum() else 0.0)
    if len(tt) < 5:
        return dict(amp=0.0, slope=0.0, tpk=t0)
    ipk = int(np.argmin(vv)); amp = vv[ipk]; tpk = tt[ipk]
    if ipk < 2 or amp >= 0:
        return dict(amp=float(amp), slope=0.0, tpk=float(tpk))
    lo, hi = 0.2 * amp, 0.8 * amp
    seg = (vv[:ipk + 1] <= lo) & (vv[:ipk + 1] >= hi)
    idx = np.where(seg)[0]
    if len(idx) >= 2:
        a, b = idx[0], idx[-1]; slope = np.polyfit(tt[a:b + 1], vv[a:b + 1], 1)[0]
    else:
        slope = (vv[ipk] - vv[0]) / (tpk - tt[0] + 1e-9)
    return dict(amp=float(amp), slope=float(slope), tpk=float(tpk))


def sr_or_dend(cell, is_pc, rng):
    segs = [s for s in cell.all if ".apic" in s.name()] if is_pc else []
    if not segs:
        segs = [s for s in cell.all if (".dend" in s.name() or ".apic" in s.name())]
    return (segs[rng.randint(len(segs))] if segs else cell.soma[0])(0.5)


def main():
    t_all = time.time()
    counts_s = argval("--counts", "300,80,60,60")
    protocol = argval("--protocol", "io")
    tstop = float(argval("--tstop", "80"))
    dt = float(argval("--dt", "0.025")); rec_dt = float(argval("--rec_dt", "0.1"))
    det = "--det" not in sys.argv and "--prob" in sys.argv    # 기본 결정론(룰베이스)
    det = not ("--prob" in sys.argv)                          # 기본 det=True, --prob면 확률
    sc_class = argval("--sc_class", "SC->PC (E1s)")
    sc_pc = int(argval("--sc_pc", "40")); sc_int = int(argval("--sc_int", "20"))
    sc_g_pc = float(argval("--sc_g_pc", "1.5")); sc_g_int = float(argval("--sc_g_int", "1.0"))
    n_fiber = int(argval("--n_fiber", "200"))
    r_stim = float(argval("--r_stim", "200"))                 # 자극전극 국소 반경(µm)
    stim_t = float(argval("--stim_t", "20"))
    seed = int(argval("--seed", "1"))
    tag = argval("--tag", protocol)
    io_levels = [float(x) for x in argval("--io_levels", "0.05,0.1,0.2,0.35,0.5,0.75,1.0").split(",")]
    ppf_isi = [float(x) for x in argval("--ppf_isi", "10,20,50,100,200").split(",")]
    no_inh = "--no_inh" in sys.argv

    # ---- 세포 ----
    c = np.load(CELLS, allow_pickle=True)
    xyz = c["xyz"].astype(float); etype = c["etype"].astype(str); quat = c["quat_wxyz"].astype(float)
    t4 = np.array([ETYPE_TO_T4.get(e, "cAC") for e in etype]); Ntot = len(xyz)
    if counts_s == "full":
        keep = np.arange(Ntot)
    else:
        counts = dict(zip(["PC", "PV", "cAC", "bAC"], map(int, counts_s.split(","))))
        ctr = xyz[t4 == "PC"].mean(0); dist = np.linalg.norm(xyz - ctr, axis=1)
        ks = []
        for tn, k in counts.items():
            ids = np.where(t4 == tn)[0]; ks.extend(ids[np.argsort(dist[ids])[:k]].tolist())
        keep = np.array(sorted(ks))
    N = len(keep); orig2gid = {int(o): g for g, o in enumerate(keep)}
    gtype = [t4[o] for o in keep]

    # ---- 기하 좌표계(PC PCA: 밴드면·깊이) + 전극 ----
    Ppc = xyz[t4 == "PC"]; c0 = Ppc.mean(0); Vt = np.linalg.svd(Ppc - c0, full_matrices=False)[2]
    face_ax = Vt[:2]; depth_ax = Vt[2]; nd = c["nd"].astype(float)
    if np.corrcoef((xyz - c0) @ depth_ax, nd)[0, 1] < 0:
        depth_ax = -depth_ax
    facepc = (Ppc - c0) @ face_ax.T
    gx = (np.arange(NCOL) - (NCOL - 1) / 2) * PITCH; gy = (np.arange(NROW) - (NROW - 1) / 2) * PITCH
    Gx, Gy = np.meshgrid(gx, gy); G0 = np.column_stack([Gx.ravel(), Gy.ravel()]); NELEC = G0.shape[0]
    tree = cKDTree(facepc); fc = facepc.mean(0); best = (-1, None, 0.0)
    for th in np.deg2rad(np.arange(0, 180, 10)):
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); Grot = G0 @ Rm.T
        for dxx in np.linspace(-400, 400, 9):
            for dyy in np.linspace(-200, 200, 9):
                E2 = Grot + fc + [dxx, dyy]; on = int((tree.query(E2)[0] < R_ON).sum())
                if on > best[0]:
                    best = (on, E2.copy(), th)
    n_on, E2d, th = best; over = tree.query(E2d)[0] < R_ON
    # 자극전극: 조직 위 전극 중 밴드 중앙 근처 1개. 기록전극: 나머지 조직 위.
    on_idx = np.where(over)[0]
    stim_elec = int(argval("--stim_elec", str(on_idx[np.argmin(np.linalg.norm(E2d[on_idx] - fc, axis=1))])))
    rec_idx = [j for j in on_idx if j != stim_elec]
    log(f"[MEA] 3x8 회전{np.rad2deg(th):.0f}° 조직위 {n_on}/24 · 자극전극 #{stim_elec} · 기록전극 {len(rec_idx)}개 · 국소반경 {r_stim}µm")

    # ---- 네트워크 구축 ----
    type_dir = net.load_representatives(MODELS)
    my = [g for g in range(N) if g % NHOST == RANK]; cells = {}; keeph = []
    t0 = time.time()
    for g in my:
        cell, _ = load_cell(type_dir[gtype[g]], gid=g)
        for sec in cell.all:
            sec.nseg = 1
        cells[g] = cell
        s = cell.soma[0]; nc = h.NetCon(s(0.5)._ref_v, None, sec=s); nc.threshold = -20.0
        pc.set_gid2node(g, RANK); pc.cell(g, nc); keeph.append(nc)
    h.define_shape(); pc.barrier()
    spt = h.Vector(); spg = h.Vector(); pc.spike_record(-1, spt, spg)
    log(f"[1/4 구축] rank0 {len(my)}세포 · {time.time()-t0:.0f}s")

    # ---- 내부 커넥텀 ----
    prc = np.load(PRUNED, allow_pickle=True)
    pre = prc["pre"]; post = prc["post"]; cid = prc["cls"]; classes = list(prc["classes"].astype(str))
    inh_cls = set(i for i, cl in enumerate(classes) if not cl.startswith("PC->"))
    rng = np.random.RandomState(1000 + RANK + seed * 97); n_syn = 0
    for i in range(len(pre)):
        a = int(pre[i]); b = int(post[i])
        if (a not in orig2gid) or (b not in orig2gid):
            continue
        gb = orig2gid[b]
        if gb % NHOST != RANK:
            continue
        if no_inh and int(cid[i]) in inh_cls:
            continue
        ga = orig2gid[a]; cls = classes[int(cid[i])]
        try:
            prm = P3.CLASSES[cls]; seg = net._placement(cells[gb], cls, rng)
            syn = build_synapse(seg, prm, seeds=(i + 1 + seed * 100000, 1, 1), deterministic=det)
            nc = pc.gid_connect(ga, syn); nc.threshold = -20.0
            nc.weight[0] = prm["g_nS"]; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_syn += 1
        except Exception:
            pass
    n_syn_all = int(pc.allreduce(n_syn, 1)); pc.barrier()
    log(f"[2/4 내부연결] {n_syn_all:,} 시냅스" + (" (억제off)" if no_inh else ""))

    # ---- 국소 SC: 자극전극 반경 R 내 세포에만 SC 시냅스 + 섬유(NetStim, 토글가능) ----
    face_all = (xyz - c0) @ face_ax.T
    fibers = []
    for k in range(n_fiber):
        ns = h.NetStim(); ns.number = 0; ns.start = stim_t; ns.noise = 0; ns.interval = 1e9
        fibers.append(ns); keeph.append(ns)
    prm = P3.CLASSES[sc_class]; scrng = np.random.RandomState(7000 + RANK + seed * 131); n_sc = 0
    for g in my:
        d2 = np.sum((face_all[keep[g]] - E2d[stim_elec]) ** 2)
        if d2 > r_stim * r_stim:                       # 자극전극 국소 반경 밖이면 SC 없음
            continue
        is_pc = gtype[g] == "PC"; k_syn = sc_pc if is_pc else sc_int; gnS = sc_g_pc if is_pc else sc_g_int
        for _ in range(k_syn):
            seg = sr_or_dend(cells[g], is_pc, scrng)
            syn = build_synapse(seg, prm, seeds=(90000 + n_sc + RANK * 100000 + seed * 7, 1, 1), deterministic=det)
            nc = h.NetCon(fibers[scrng.randint(n_fiber)], syn); nc.weight[0] = gnS; nc.delay = SYN_DELAY
            keeph += [syn, nc]; n_sc += 1
    n_sc_all = int(pc.allreduce(n_sc, 1)); pc.barrier()
    log(f"[3/4 국소SC] {n_sc_all:,} SC시냅스(자극전극 {r_stim}µm 내)")

    # ---- 전세포 기하 + 막전류 기록 + 전달행렬 ----
    t0 = time.time(); mids = []; rads = []; vecs = []
    cv = h.CVode(); cv.use_fast_imem(1)
    for g in my:
        geom = L.collect_segments(list(cells[g].all)); Rc = quat_to_R(quat[keep[g]])
        mids.append(geom["mid"] @ Rc.T + xyz[keep[g]]); rads.append(geom["radius"])
        for seg in geom["segs"]:
            v = h.Vector(); v.record(seg._ref_i_membrane_, rec_dt); vecs.append(v)
    mids = np.vstack(mids) if mids else np.zeros((0, 3)); rads = np.concatenate(rads) if rads else np.zeros(0)
    uv = (mids - c0) @ face_ax.T; dep = (mids - c0) @ depth_ax
    dmax = pc.allreduce(float(dep.max()) if len(dep) else -1e18, 2)
    dmin = pc.allreduce(float(dep.min()) if len(dep) else 1e18, 3)
    zloc = (dmax - dep) + Z_GLASS_MARGIN; Hh = (dmax - dmin) + 2 * Z_GLASS_MARGIN
    geom_r = dict(mid=np.column_stack([uv[:, 0], uv[:, 1], zloc]), radius=rads)
    E3 = np.column_stack([E2d[:, 0], E2d[:, 1], np.zeros(NELEC)])
    M_rank = L.moi_point_matrix(geom_r, E3, SIG_T, SIG_S, SIG_G, Hh, N_IMG) if len(rads) else np.zeros((NELEC, 0))
    nt = int(round(tstop / rec_dt)) + 1
    log(f"[4/4 전달행렬] rank세그 {len(rads)} · Hh={Hh:.0f}µm · {time.time()-t0:.0f}s")

    h.celsius = 34.0; h.cvode_active(0); h.dt = dt; pc.set_maxstep(10)

    def run_once(n_active, times):
        """활성 섬유 n_active개를 times(ms)에 발화 → rank fEPSP(NELEC,nt) 합."""
        for k, ns in enumerate(fibers):
            if k < n_active:
                ns.number = len(times); ns.start = times[0]
                ns.interval = (times[1] - times[0]) if len(times) > 1 else 1e9
            else:
                ns.number = 0
        spt.resize(0); spg.resize(0)
        h.finitialize(-70.0); pc.psolve(tstop)
        I = np.array([np.asarray(v) for v in vecs]) if vecs else None
        Ve_local = (M_rank @ I) * 1e3 if (I is not None and I.size) else np.zeros((NELEC, nt))
        if Ve_local.shape[1] != nt:
            Ve_local = np.zeros((NELEC, nt))
        if NHOST > 1:
            parts = pc.py_allgather(Ve_local.tolist()); Ve = np.sum([np.array(p) for p in parts], axis=0)
        else:
            Ve = Ve_local
        nspk = int(pc.allreduce(len(spt), 1))
        return Ve, nspk

    tarr = np.arange(nt) * rec_dt
    out = os.path.join(FIG, f"_mea_{tag}.npz")
    rec_j = rec_idx[int(np.argmin([np.linalg.norm(E2d[j] - E2d[stim_elec]) for j in rec_idx]))] if rec_idx else 0

    if protocol == "io":
        rows = []; waves = []
        log(f"{'세기(섬유)':>10} {'slope(µV/ms)':>13} {'amp(µV)':>9} {'창내최대|Ve|':>12} {'스파이크':>7}")
        for lv in io_levels:
            na = max(1, int(round(lv * n_fiber)))
            Ve, nspk = run_once(na, [stim_t])
            if RANK == 0:
                fe = measure_fepsp(tarr, Ve[rec_j], stim_t, 30.0)
                w = (tarr >= stim_t) & (tarr <= stim_t + 30.0)
                pk_abs = float(Ve[rec_j][w][np.argmax(np.abs(Ve[rec_j][w]))]) if w.sum() else 0.0
                rows.append((lv, na, fe["slope"], fe["amp"], nspk, pk_abs))
                waves.append(Ve[:, w])                     # 진단: 전극별 창내 파형
                log(f"{na:>10} {fe['slope']:>13.4f} {fe['amp']:>9.4f} {pk_abs:>12.4f} {nspk:>7}")
        if RANK == 0:
            R = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], float)
            np.savez(out, kind="io", levels=R[:, 0], nact=R[:, 1], slope=R[:, 2], amp=R[:, 3],
                     nspk=R[:, 4], pk_abs=R[:, 5], waves=np.array(waves), twin=tarr[(tarr >= stim_t) & (tarr <= stim_t + 30.0)],
                     stim_elec=stim_elec, rec_j=rec_j, E=E2d, over=over, r_stim=r_stim, N=N, n_fiber=n_fiber)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    elif protocol == "ppf":
        rows = []
        na = max(1, int(round(float(argval('--io_test', '0.4')) * n_fiber)))   # 테스트 세기
        log(f"{'ISI(ms)':>8} {'slope1':>9} {'slope2':>9} {'PPR':>6}")
        for isi in ppf_isi:
            Ve, nspk = run_once(na, [stim_t, stim_t + isi])
            if RANK == 0:
                f1 = measure_fepsp(tarr, Ve[rec_j], stim_t, min(isi, 30.0))
                f2 = measure_fepsp(tarr, Ve[rec_j], stim_t + isi, 30.0)
                ppr = abs(f2["slope"]) / max(abs(f1["slope"]), 1e-9)
                rows.append((isi, f1["slope"], f2["slope"], ppr))
                log(f"{isi:>8.0f} {f1['slope']:>9.3f} {f2['slope']:>9.3f} {ppr:>6.2f}")
        if RANK == 0:
            R = np.array(rows, float)
            np.savez(out, kind="ppf", isi=R[:, 0], slope1=R[:, 1], slope2=R[:, 2], ppr=R[:, 3],
                     stim_elec=stim_elec, rec_j=rec_j, E=E2d, over=over, r_stim=r_stim, N=N)
            print("saved:", out, f"· 총 {time.time()-t_all:.0f}s", flush=True)

    pc.barrier(); pc.done()


if __name__ == "__main__":
    main()
